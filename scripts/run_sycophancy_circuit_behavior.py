#!/usr/bin/env python3
"""Generate held-out sycophancy responses for one prompt-vector circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "EAP-IG")]

from refusal_circuit.circuit import (
    _construction_hooks,
    _receiver_cache_hooks,
    _source_cache_hooks,
    _sparse_circuit_only_construction_hooks,
)
from refusal_circuit.eapig import (
    prepare_paired_coordinate_source_corruption,
)
from refusal_circuit.graph import (
    load_circuit_edge_names,
    make_projection_graph,
    set_selected_edges,
)
from refusal_circuit.interventions import (
    direction_tensor_coordinate_clamp_hook,
)
from refusal_circuit.model import format_system_instruction, tokenize_prompts
from refusal_circuit.persona import make_system_prompt
from refusal_circuit.workflow import _load_vector, _model, load_config
from transformer_lens.past_key_value_caching import (
    HookedTransformerKeyValueCache,
)


CONDITIONS = (
    "baseline_negative",
    "baseline_positive",
    "full_projection",
    "target_projection",
    "circuit_only",
    "circuit_ablated",
)


def enable_expanded_cache_validation(model) -> None:
    """Accept EAP's expanded KV heads without changing model computation."""

    def get_pos_offset(self, past_kv_cache, batch_size):
        if past_kv_cache is None:
            return 0
        cached_batch, context, cached_heads, d_head = (
            past_kv_cache[0].past_keys.shape
        )
        expected_heads = (
            self.cfg.n_heads
            if self.cfg.ungroup_grouped_query_attention
            else self.cfg.n_key_value_heads or self.cfg.n_heads
        )
        assert cached_batch == batch_size
        assert cached_heads == expected_heads
        assert d_head == self.cfg.d_head
        return context

    model.get_pos_offset = types.MethodType(get_pos_offset, model)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def formatted_prompt(tokenizer, row: dict, polarity: str) -> str:
    instruction = row[f"{polarity}_system_instruction"]
    return format_system_instruction(
        tokenizer,
        make_system_prompt("sycophantic", instruction),
        row["prompt"],
    )


def normal_prefill(model, tokens, attention_mask, hooks=()):
    cache = generation_cache(model)
    with torch.inference_mode(), model.hooks(fwd_hooks=list(hooks)):
        logits = model(
            tokens,
            attention_mask=attention_mask,
            past_kv_cache=cache,
        )
    return logits, cache


def circuit_prefill(
    model,
    graph,
    tokens,
    attention_mask,
    *,
    corrupt_cache_transform,
    corrupt_fwd_hooks,
    patch_selected: bool,
):
    corrupt_sources: dict[str, torch.Tensor] = {}
    corrupt_receivers: dict[tuple[str, str | None], torch.Tensor] = {}
    corrupt_hooks = _source_cache_hooks(
        model,
        graph,
        corrupt_sources,
        cache_transform=corrupt_cache_transform,
        selected_only=True,
    )
    corrupt_hooks += list(corrupt_fwd_hooks)
    if not patch_selected:
        corrupt_hooks += _receiver_cache_hooks(graph, corrupt_receivers)
    with torch.inference_mode(), model.hooks(fwd_hooks=corrupt_hooks):
        model(tokens, attention_mask=attention_mask)

    clean_sources: dict[str, torch.Tensor] = {}
    mixed_hooks = _source_cache_hooks(
        model,
        graph,
        clean_sources,
        selected_only=True,
    )
    mixed_hooks += (
        _construction_hooks(
            graph,
            corrupt_sources,
            clean_sources,
            patch_selected=True,
        )
        if patch_selected
        else _sparse_circuit_only_construction_hooks(
            graph,
            corrupt_sources,
            clean_sources,
            corrupt_receivers,
        )
    )
    cache = generation_cache(model)
    with torch.inference_mode(), model.hooks(fwd_hooks=mixed_hooks):
        logits = model(
            tokens,
            attention_mask=attention_mask,
            past_kv_cache=cache,
        )
    return logits, cache


def paired_prompt_corruption(
    model,
    graph,
    direction,
    clean_tokens,
    clean_mask,
    reference_tokens,
    reference_mask,
    target_hook: str,
):
    if tuple(clean_tokens.shape) != tuple(reference_tokens.shape):
        raise ValueError("Positive and negative prompts must be token-aligned.")
    if not torch.equal(clean_mask, reference_mask):
        raise ValueError("Positive and negative attention masks must match.")

    source_transform, source_hooks, _ = (
        prepare_paired_coordinate_source_corruption(
            model,
            graph,
            reference_tokens,
            reference_mask,
            [(direction, clean_mask.bool())],
        )
    )
    target: dict[str, torch.Tensor] = {}

    def capture_target(activations: torch.Tensor, hook):
        del hook
        unit = direction.to(
            device=activations.device,
            dtype=activations.dtype,
        )
        unit = unit / unit.norm().clamp_min(1e-8)
        target["coordinates"] = torch.einsum(
            "bpd,d->bp", activations, unit
        ).detach().float()
        return activations

    with torch.inference_mode(), model.hooks(
        fwd_hooks=[(target_hook, capture_target)]
    ):
        model(reference_tokens, attention_mask=reference_mask)
    target_hook_pair = (
        target_hook,
        direction_tensor_coordinate_clamp_hook(
            direction,
            target_coordinates=target["coordinates"],
            mask=clean_mask.bool(),
        ),
    )
    return source_transform, source_hooks, [*source_hooks, target_hook_pair]


def generation_cache(model):
    """Initialize a cache matching EAP's expanded grouped-query heads."""
    original = model.cfg.n_key_value_heads
    try:
        model.cfg.n_key_value_heads = model.cfg.n_heads
        return HookedTransformerKeyValueCache.init_cache(
            model.cfg, model.cfg.device, batch_size=1
        )
    finally:
        model.cfg.n_key_value_heads = original


def decode(model, logits, cache, *, max_new_tokens: int, hooks=()):
    eos = model.tokenizer.eos_token_id
    generated: list[int] = []
    for step in range(max_new_tokens):
        token = logits[:, -1].argmax(dim=-1)
        token_id = int(token.item())
        generated.append(token_id)
        if eos is not None and token_id == eos:
            break
        if step + 1 < max_new_tokens:
            with torch.inference_mode(), model.hooks(fwd_hooks=list(hooks)):
                logits = model(token[:, None], past_kv_cache=cache)
    return generated


def generate_condition(
    model,
    graph,
    direction,
    prompt: str,
    reference_prompt: str,
    condition: str,
    max_new_tokens: int,
    target_hook: str,
):
    tokens, attention_mask = tokenize_prompts(model, [prompt])
    reference_tokens, reference_mask = tokenize_prompts(
        model, [reference_prompt]
    )
    if condition in {"baseline_negative", "baseline_positive"}:
        logits, cache = normal_prefill(model, tokens, attention_mask)
        decode_hooks = ()
    elif condition == "full_projection":
        _, _, full_hooks = paired_prompt_corruption(
            model,
            graph,
            direction,
            tokens,
            attention_mask,
            reference_tokens,
            reference_mask,
            target_hook,
        )
        logits, cache = normal_prefill(
            model, tokens, attention_mask, hooks=full_hooks
        )
        decode_hooks = ()
    elif condition == "target_projection":
        _, _, full_hooks = paired_prompt_corruption(
            model,
            graph,
            direction,
            tokens,
            attention_mask,
            reference_tokens,
            reference_mask,
            target_hook,
        )
        logits, cache = normal_prefill(
            model,
            tokens,
            attention_mask,
            hooks=[full_hooks[-1]],
        )
        decode_hooks = ()
    else:
        source_transform, source_hooks, _ = paired_prompt_corruption(
            model,
            graph,
            direction,
            tokens,
            attention_mask,
            reference_tokens,
            reference_mask,
            target_hook,
        )
        logits, cache = circuit_prefill(
            model,
            graph,
            tokens,
            attention_mask,
            corrupt_cache_transform=source_transform,
            corrupt_fwd_hooks=source_hooks,
            patch_selected=condition == "circuit_ablated",
        )
        decode_hooks = ()
    generated = decode(
        model,
        logits,
        cache,
        max_new_tokens=max_new_tokens,
        hooks=decode_hooks,
    )
    return {
        "answer": model.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip(),
        "generated_token_count": len(generated),
        "hit_max_new_tokens": len(generated) >= max_new_tokens,
    }


def generate(args) -> None:
    config = load_config(args.config)
    circuit_metadata = json.loads(args.circuit.read_text(encoding="utf-8"))
    circuit_k = int(circuit_metadata.get("retained_edges", -1))
    if circuit_k <= 0:
        raise ValueError("Circuit metadata has no retained edges.")
    model = _model(config, hook_mode="eap")
    enable_expanded_cache_validation(model)
    direction, _raw, vector = _load_vector(config, model)
    graph = make_projection_graph(model, int(vector["layer"]))
    set_selected_edges(graph, load_circuit_edge_names(args.circuit))
    target_hook = str(circuit_metadata["target"]["target_hook"])

    rows = read_jsonl(args.dataset)[: args.num_prompts]
    selected = [
        row
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"responses_shard_{args.shard_index:02d}.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists.")

    with temporary.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(selected, start=1):
            prompts = {
                "negative": formatted_prompt(model.tokenizer, row, "negative"),
                "positive": formatted_prompt(model.tokenizer, row, "positive"),
            }
            for condition in args.conditions:
                prompt = (
                    prompts["negative"]
                    if condition == "baseline_negative"
                    else prompts["positive"]
                )
                reference_prompt = prompts["negative"]
                result = generate_condition(
                    model,
                    graph,
                    direction,
                    prompt,
                    reference_prompt,
                    condition,
                    args.max_new_tokens,
                    target_hook,
                )
                record = {
                    "id": row["id"],
                    "source_index": row["source_index"],
                    "instruction_pair_index": row["instruction_pair_index"],
                    "question": row["prompt"],
                    "condition": condition,
                    "circuit_k": circuit_k,
                    **result,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "completed_prompts": row_index,
                        "total_prompts": len(selected),
                    }
                ),
                flush=True,
            )
    temporary.replace(output)


def aggregate(args) -> None:
    rows = read_jsonl(args.dataset)[: args.num_prompts]
    circuit_metadata = json.loads(args.circuit.read_text(encoding="utf-8"))
    circuit_k = int(circuit_metadata["retained_edges"])
    shard_paths = [
        args.output_dir / f"responses_shard_{index:02d}.jsonl"
        for index in range(args.num_shards)
    ]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard outputs: {missing}")
    generated = [row for path in shard_paths for row in read_jsonl(path)]
    expected = len(rows) * len(args.conditions)
    keys = {(row["id"], row["condition"]) for row in generated}
    if len(generated) != expected or len(keys) != expected:
        raise RuntimeError(
            f"Expected {expected} unique responses, found "
            f"{len(generated)} rows and {len(keys)} keys."
        )

    order = {row["id"]: index for index, row in enumerate(rows)}
    condition_order = {
        name: index for index, name in enumerate(args.conditions)
    }
    generated.sort(
        key=lambda row: (order[row["id"]], condition_order[row["condition"]])
    )
    combined = args.output_dir / "responses.jsonl"
    with combined.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "status": "generation_complete_pending_persona_judge",
        "model": load_config(args.config)["model"],
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": sha256(args.dataset),
            "prompts": len(rows),
        },
        "circuit": {
            "path": str(args.circuit.resolve()),
            "sha256": sha256(args.circuit),
            "k": circuit_k,
        },
        "conditions": {
            "baseline_negative": "anti-sycophantic instruction, no intervention",
            "baseline_positive": "sycophantic instruction, no intervention",
            "full_projection": (
                "sycophantic instruction; every prompt-source coordinate and "
                "the layer-15 target coordinate replaced by its token-aligned "
                "anti-sycophantic reference value"
            ),
            "target_projection": (
                "sycophantic instruction, vector projection only at the saved "
                "layer/position during prompt prefill"
            ),
            "circuit_only": (
                f"matched anti-coordinate prompt baseline plus the retained "
                f"k={circuit_k} prompt-circuit edges"
            ),
            "circuit_ablated": (
                f"sycophantic prompt with the retained k={circuit_k} "
                "prompt-circuit edges replaced by matched anti-coordinate values"
            ),
        }
        | {
            "selected": args.conditions,
        },
        "generation": {
            "deterministic": True,
            "max_new_tokens": args.max_new_tokens,
            "responses": len(generated),
            "prompt_intervention": (
                "matched-coordinate corruption during prompt prefill only; "
                "continuation uses the resulting KV cache"
            ),
        },
        "judge": {
            "status": "pending",
            "protocol": "Persona Vectors sycophancy and coherence judges",
        },
        "artifact": {
            "path": str(combined.resolve()),
            "sha256": sha256(combined),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/sycophancy_prompt.json")
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/processed/qwen2.5-7b-instruct/trait_circuits/"
            "sycophancy/final_behavior.jsonl"
        ),
    )
    parser.add_argument(
        "--circuit",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/sycophancy/prompt/eapig/"
            "circuits/eapig_top_100.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/sycophancy/prompt/eapig/"
            "behavioral_final_k100"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard selection.")
    if args.max_new_tokens <= 0 or args.num_prompts <= 0:
        raise ValueError("Token and prompt counts must be positive.")
    aggregate(args) if args.aggregate else generate(args)


if __name__ == "__main__":
    main()

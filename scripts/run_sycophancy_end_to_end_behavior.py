#!/usr/bin/env python3
"""Generate responses under a natural-contrast end-to-end circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "EAP-IG"), str(ROOT / "scripts")]

from refusal_circuit.circuit import (
    mixed_logits_from_prompt_pair,
    mixed_logits_with_intervention,
)
from refusal_circuit.eapig import (
    prepare_paired_coordinate_source_corruption,
)
from refusal_circuit.graph import (
    circuit_connectivity,
    load_circuit_edge_names,
    make_logit_graph,
    set_selected_edges,
    validate_intervention_connectivity,
)
from refusal_circuit.model import (
    install_memory_efficient_split_qkv_projection,
    load_model,
)
from refusal_circuit.persona import read_jsonl
from refusal_circuit.workflow import _load_vector, load_config
from run_sycophancy_end_to_end_eapig import (
    formatted_prompt,
    load_instruction_pairs,
    token_ids,
)


CONDITIONS = (
    "baseline_negative",
    "baseline_positive",
    "full_corrupt",
    "circuit_only",
    "circuit_ablated",
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_tensor(ids: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    return tokens, torch.ones_like(tokens)


def baseline_logits(model, prompt_ids: list[int], generated: list[int]) -> torch.Tensor:
    tokens, mask = token_tensor(prompt_ids + generated, next(model.parameters()).device)
    with torch.inference_mode():
        return model(tokens, attention_mask=mask)


def circuit_logits(
    model,
    graph,
    positive_ids: list[int],
    negative_ids: list[int],
    generated: list[int],
    *,
    patch_selected: bool,
) -> torch.Tensor:
    device = next(model.parameters()).device
    clean_tokens, clean_mask = token_tensor(positive_ids + generated, device)
    corrupt_tokens, corrupt_mask = token_tensor(negative_ids + generated, device)
    return mixed_logits_from_prompt_pair(
        model,
        graph,
        clean_tokens,
        clean_mask,
        corrupt_tokens,
        corrupt_mask,
        patch_selected=patch_selected,
        cache_dtype=torch.bfloat16,
    )


def paired_coordinate_logits(
    model,
    graph,
    positive_ids: list[int],
    negative_ids: list[int],
    generated: list[int],
    *,
    prompt_direction: torch.Tensor,
    generation_direction: torch.Tensor,
    patch_selected: bool | None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    clean_tokens, clean_mask = token_tensor(positive_ids + generated, device)
    reference_tokens, reference_mask = token_tensor(
        negative_ids + generated, device
    )
    prompt_mask = torch.zeros_like(clean_mask, dtype=torch.bool)
    prompt_mask[:, : len(positive_ids)] = True
    generation_mask = torch.zeros_like(clean_mask, dtype=torch.bool)
    generation_mask[:, len(positive_ids) :] = True
    source_transform, corrupt_hooks, _ = (
        prepare_paired_coordinate_source_corruption(
            model,
            graph,
            reference_tokens,
            reference_mask,
            [
                (prompt_direction, prompt_mask),
                (generation_direction, generation_mask),
            ],
        )
    )
    if patch_selected is None:
        with torch.inference_mode(), model.hooks(fwd_hooks=corrupt_hooks):
            return model(clean_tokens, attention_mask=clean_mask).detach()
    return mixed_logits_with_intervention(
        model,
        graph,
        clean_tokens,
        clean_mask,
        corrupt_fwd_hooks=corrupt_hooks,
        corrupt_cache_transform=source_transform,
        patch_selected=patch_selected,
        cache_dtype=torch.bfloat16,
    )


def generate(
    model,
    graph,
    positive_ids: list[int],
    negative_ids: list[int],
    condition: str,
    max_new_tokens: int,
    *,
    corruption: str,
    prompt_direction: torch.Tensor | None,
    generation_direction: torch.Tensor | None,
) -> list[int]:
    generated: list[int] = []
    eos = model.tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        if condition == "baseline_positive":
            logits = baseline_logits(model, positive_ids, generated)
        elif condition == "baseline_negative":
            logits = baseline_logits(model, negative_ids, generated)
        elif condition == "full_corrupt" and corruption == "natural_prompt":
            logits = baseline_logits(model, negative_ids, generated)
        elif corruption == "paired_coordinate":
            if prompt_direction is None or generation_direction is None:
                raise ValueError(
                    "Paired-coordinate generation requires both vectors."
                )
            logits = paired_coordinate_logits(
                model,
                graph,
                positive_ids,
                negative_ids,
                generated,
                prompt_direction=prompt_direction,
                generation_direction=generation_direction,
                patch_selected=(
                    None
                    if condition == "full_corrupt"
                    else condition == "circuit_ablated"
                ),
            )
        else:
            logits = circuit_logits(
                model,
                graph,
                positive_ids,
                negative_ids,
                generated,
                patch_selected=condition == "circuit_ablated",
            )
        token_id = int(logits[0, -1].argmax().item())
        generated.append(token_id)
        if eos is not None and token_id == eos:
            break
    return generated


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
        "--instruction-pairs",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/sycophancy/end_to_end/"
            "instruction_contrast_n16/instruction_pairs.json"
        ),
    )
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--corruption",
        choices=("natural_prompt", "paired_coordinate"),
        default="natural_prompt",
    )
    parser.add_argument(
        "--prompt-vector-config",
        type=Path,
        default=Path(
            "configs/sycophancy_prompt_paired_coordinate_eapig_a100.json"
        ),
    )
    parser.add_argument(
        "--generation-vector-config",
        type=Path,
        default=Path(
            "configs/sycophancy_generation_paired_coordinate_eapig.json"
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")

    config_path = resolve(args.config)
    dataset_path = resolve(args.dataset)
    instruction_path = resolve(args.instruction_pairs)
    circuit_path = resolve(args.circuit)
    output_dir = resolve(args.output_dir)
    suffix = "" if args.num_shards == 1 else f"_shard_{args.shard_index:02d}"
    responses_path = output_dir / f"responses{suffix}.jsonl"
    metadata_path = output_dir / f"metadata{suffix}.json"
    if (responses_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError("Behavioral output exists; pass --overwrite.")

    all_rows = read_jsonl(dataset_path)[: args.num_prompts]
    if len(all_rows) != args.num_prompts:
        raise ValueError(
            f"Expected {args.num_prompts} prompts, found {len(all_rows)}."
        )
    rows = [
        row
        for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_index
    ]
    instruction_pairs = load_instruction_pairs(instruction_path)
    config = load_config(config_path)

    install_memory_efficient_split_qkv_projection()
    model_config = config["model"]
    model = load_model(
        model_config["name"],
        revision=model_config.get("revision"),
        dtype=model_config.get("dtype", "bfloat16"),
        device=model_config.get("device"),
        hook_mode="eap",
    )
    prompt_direction = None
    generation_direction = None
    vector_metadata = None
    if args.corruption == "paired_coordinate":
        prompt_config = load_config(resolve(args.prompt_vector_config))
        generation_config = load_config(resolve(args.generation_vector_config))
        prompt_direction, _, prompt_metadata = _load_vector(
            prompt_config, model
        )
        generation_direction, _, generation_metadata = _load_vector(
            generation_config, model
        )
        vector_metadata = {
            "prompt": {
                "config": str(resolve(args.prompt_vector_config)),
                "layer": prompt_metadata["layer"],
                "position": prompt_metadata["position"],
            },
            "generation": {
                "config": str(resolve(args.generation_vector_config)),
                "layer": generation_metadata["layer"],
            },
        }
    graph = make_logit_graph(model)
    edge_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, edge_names)
    connectivity = circuit_connectivity(graph, edge_names)
    validate_intervention_connectivity(
        {"circuit": connectivity}, require_source_to_sink=True
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = responses_path.with_suffix(".jsonl.tmp")
    started = time.monotonic()
    completed: dict[tuple[str, str], dict] = {}
    if temporary.exists() and not args.overwrite:
        for record in read_jsonl(temporary):
            completed[(record["id"], record["condition"])] = record
    response_count = len(completed)
    mode = "w" if args.overwrite or not temporary.exists() else "a"
    with temporary.open(mode, encoding="utf-8") as handle:
        for prompt_index, row in enumerate(rows, start=1):
            positive_instruction, negative_instruction = instruction_pairs[
                int(row["instruction_pair_index"])
            ]
            positive_prompt = formatted_prompt(
                model.tokenizer, row, positive_instruction
            )
            negative_prompt = formatted_prompt(
                model.tokenizer, row, negative_instruction
            )
            positive_ids = token_ids(model.tokenizer, positive_prompt)
            negative_ids = token_ids(model.tokenizer, negative_prompt)
            if len(positive_ids) != len(negative_ids):
                raise ValueError(f"Prompt alignment failed for {row['id']}.")

            generated_by_condition: dict[str, list[int]] = {}
            for condition in args.conditions:
                key = (row["id"], condition)
                if key in completed:
                    continue
                if (
                    condition == "full_corrupt"
                    and args.corruption == "natural_prompt"
                    and "baseline_negative" in generated_by_condition
                ):
                    generated = generated_by_condition["baseline_negative"]
                else:
                    generated = generate(
                        model,
                        graph,
                        positive_ids,
                        negative_ids,
                        condition,
                        args.max_new_tokens,
                        corruption=args.corruption,
                        prompt_direction=prompt_direction,
                        generation_direction=generation_direction,
                    )
                generated_by_condition[condition] = generated
                record = {
                    "id": row["id"],
                    "source_index": row["source_index"],
                    "instruction_pair_index": row["instruction_pair_index"],
                    "question": row["prompt"],
                    "condition": condition,
                    "answer": model.tokenizer.decode(
                        generated, skip_special_tokens=True
                    ).strip(),
                    "generated_token_count": len(generated),
                    "hit_max_new_tokens": len(generated) >= args.max_new_tokens,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                response_count += 1
            print(
                json.dumps(
                    {
                        "completed_prompts": prompt_index,
                        "total_prompts": len(rows),
                    }
                ),
                flush=True,
            )
    temporary.replace(responses_path)

    circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
    metadata = {
        "status": "generation_complete_pending_persona_judge",
        "model": {**model_config, "hook_mode": "eap"},
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256(dataset_path),
            "num_prompts": len(rows),
            "total_prompts": len(all_rows),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "instruction_pairs": {
            "path": str(instruction_path),
            "sha256": sha256(instruction_path),
        },
        "circuit": {
            "path": str(circuit_path),
            "sha256": sha256(circuit_path),
            "requested_k": circuit["requested_k"],
            "retained_edges": circuit["retained_edges"],
            "connectivity": connectivity,
        },
        "conditions": list(args.conditions),
        "corruption": args.corruption,
        "vectors": vector_metadata,
        "generation": {
            "deterministic": True,
            "intervention_scope": "every autoregressive generation step",
            "max_new_tokens": args.max_new_tokens,
            "responses": response_count,
            "elapsed_seconds": time.monotonic() - started,
        },
        "artifact": {
            "path": str(responses_path),
            "sha256": sha256(responses_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

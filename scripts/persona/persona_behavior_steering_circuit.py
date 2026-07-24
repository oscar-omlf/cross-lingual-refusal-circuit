#!/usr/bin/env python3
"""Generate behavior under a steering-path circuit intervention.

Circuit ablation patches selected downstream edges with values from the
behaviorally successful negative-steering counterfactual at every decoding
step.  This is the decisive causal test for a circuit discovered with
``persona.run-steering-eapig``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--question-indices", type=int, nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("normal", "negative_steered", "circuit_only", "circuit_ablated"),
        default=("normal", "negative_steered", "circuit_ablated"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("question indices must be unique")
    if len(args.conditions) != len(set(args.conditions)):
        raise ValueError("conditions must be unique")

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import mixed_logits_with_intervention
    from refusal_circuit.graph import (
        load_circuit_edge_names,
        make_steering_to_logits_graph,
        set_selected_edges,
    )
    from refusal_circuit.interventions import activation_addition_hook
    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        make_system_prompt,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    trait_slug = config["trait"]["name"]
    result_root = ROOT / config["paths"]["results_root"]
    discovery = result_root / "circuit_discovery_steering_path" / args.run_name
    discovery_meta_path = discovery / "eapig_run_metadata.json"
    discovery_meta = json.loads(discovery_meta_path.read_text(encoding="utf-8"))
    corrupt_coefficient = float(
        discovery_meta["steering"]["corrupt_coefficient"]
    )
    circuit_path = (
        discovery
        / "circuits_nested"
        / f"steering_path_eapig_top_{args.k}.json"
    )
    output = (
        ROOT
        / config["paths"]["intermediate_root"]
        / "steering_circuit_behavior"
        / args.run_name
        / f"k{args.k}_q{'-'.join(map(str, args.question_indices))}"
    )
    raw_path = output / "responses.jsonl"
    metadata_path = output / "metadata.json"
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output} already contains behavior output")
        raw_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    official = ROOT / config["paths"]["official_root"] / "eval.json"
    eval_data = json.loads(official.read_text(encoding="utf-8"))
    questions = eval_data["questions"]
    instructions = eval_data["instruction"]
    designed = []
    for question_index in args.question_indices:
        if not 0 <= question_index < len(questions):
            raise IndexError(f"Question index {question_index} is out of range.")
        instruction_index = question_index % len(instructions)
        instruction = instructions[instruction_index]["pos"]
        system_prompt = make_system_prompt(
            config["trait"]["positive_assistant_name"], instruction
        )
        designed.append(
            {
                "question_id": f"{trait_slug}_{question_index}",
                "question_index": question_index,
                "instruction_index": instruction_index,
                "question": questions[question_index],
                "instruction": instruction,
                "system_prompt": system_prompt,
            }
        )

    model_cfg = config["model"]
    model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=args.dtype,
        revision=model_cfg.get("revision"),
    )
    for row in designed:
        row["prompt"] = model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    vector_path = result_root / f"vector/{trait_slug}_response_avg_diff.pt"
    vector_stack = torch.load(
        vector_path, map_location=next(model.parameters()).device, weights_only=True
    )
    layer = int(config["vector"]["paper_layer"])
    raw_direction = vector_stack[layer] if vector_stack.ndim == 2 else vector_stack
    target_hook = config["vector"]["transformer_lens_target_hook"]
    graph = make_steering_to_logits_graph(model, layer)
    set_selected_edges(graph, load_circuit_edge_names(circuit_path))

    def append(source: dict, condition: str, token_ids: list[int]) -> None:
        row = {
            **source,
            "schema_version": 1,
            "record_id": (
                f"steering-circuit-{args.run_name}-k{args.k}-{condition}-"
                f"q{source['question_index']:02d}"
            ),
            "behavior_protocol_id": config["protocol_id"],
            "trait": trait_slug,
            "model": model_cfg["name"],
            "model_revision": model_cfg.get("revision"),
            "condition": condition,
            "requested_k": args.k,
            "answer": model.tokenizer.decode(token_ids, skip_special_tokens=True),
            "generated_token_count": len(token_ids),
            "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
            "decoding": "greedy",
            "corrupt_coefficient": corrupt_coefficient,
        }
        write_jsonl(raw_path, [row], mode="a")

    for condition in args.conditions:
        for source in tqdm(designed, desc=condition):
            prompt_tokens, _ = tokenize_prompts(
                model, [source["prompt"]], max_length=args.max_length
            )
            if condition in {"normal", "negative_steered"}:
                model.set_ungroup_grouped_query_attention(False)
                try:
                    hooks = []
                    if condition == "negative_steered":
                        hooks = [
                            (
                                target_hook,
                                activation_addition_hook(
                                    raw_direction,
                                    coefficient=corrupt_coefficient,
                                    position=-1,
                                ),
                            )
                        ]
                    with model.hooks(fwd_hooks=hooks):
                        generated = model.generate(
                            prompt_tokens,
                            max_new_tokens=args.max_new_tokens,
                            stop_at_eos=True,
                            do_sample=False,
                            use_past_kv_cache=True,
                            return_type="tokens",
                            verbose=False,
                        )
                finally:
                    model.set_ungroup_grouped_query_attention(True)
                append(
                    source,
                    condition,
                    generated[0, prompt_tokens.shape[1] :].tolist(),
                )
                continue

            # EAP's graph has 28 explicit query-head nodes. Keep TransformerLens
            # grouped-query expansion enabled for circuit cache and patch hooks.
            model.set_ungroup_grouped_query_attention(True)
            tokens = prompt_tokens
            token_ids: list[int] = []
            eos_id = model.tokenizer.eos_token_id
            for _ in tqdm(
                range(args.max_new_tokens),
                desc=f"{condition}-q{source['question_index']:02d}",
                leave=False,
            ):
                corrupt_hook = (
                    target_hook,
                    activation_addition_hook(
                        raw_direction,
                        coefficient=corrupt_coefficient,
                        position=-1,
                    ),
                )

                def cache_transform(name: str, activations: torch.Tensor):
                    if name != target_hook:
                        return activations
                    changed = activations.clone()
                    changed[:, -1, :] += (
                        corrupt_coefficient
                        * raw_direction.to(
                            device=changed.device, dtype=changed.dtype
                        )
                    )
                    return changed

                logits = mixed_logits_with_intervention(
                    model,
                    graph,
                    tokens,
                    None,
                    corrupt_fwd_hooks=[corrupt_hook],
                    corrupt_cache_transform=cache_transform,
                    patch_selected=condition == "circuit_ablated",
                )
                next_token = int(logits[0, -1].argmax().item())
                token_ids.append(next_token)
                tokens = torch.cat(
                    [
                        tokens,
                        torch.tensor(
                            [[next_token]],
                            device=tokens.device,
                            dtype=tokens.dtype,
                        ),
                    ],
                    dim=1,
                )
                if eos_id is not None and next_token == eos_id:
                    break
            append(source, condition, token_ids)

    rows = read_jsonl(raw_path)
    expected = len(designed) * len(args.conditions)
    if len(rows) != expected or len({row["record_id"] for row in rows}) != expected:
        raise RuntimeError(f"Incomplete behavioral output: {len(rows)}/{expected}")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "canonical_protocol": False,
        "behavior_protocol_id": config["protocol_id"],
        "claim_scope": "free-generation causal circuit validation",
        "model": model_cfg,
        "trait": config["trait"],
        "eval_artifact": str(official.relative_to(ROOT)),
        "eval_artifact_sha256": sha256_file(official),
        "discovery_metadata": str(discovery_meta_path.relative_to(ROOT)),
        "discovery_metadata_sha256": sha256_file(discovery_meta_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "requested_k": args.k,
        "question_indices": args.question_indices,
        "conditions": list(args.conditions),
        "corrupt_coefficient": corrupt_coefficient,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "outputs": [
            {
                "path": str(raw_path.relative_to(ROOT)),
                "rows": len(rows),
                "sha256": sha256_file(raw_path),
            }
        ],
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

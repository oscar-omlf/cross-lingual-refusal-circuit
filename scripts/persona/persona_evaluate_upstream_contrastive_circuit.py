#!/usr/bin/env python3
"""Evaluate an upstream contrastive persona circuit on held-out prompt pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import (
    _align_negative_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=int, default=3000)
    parser.add_argument("--circuit-path", type=Path, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--num-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import (
        _normal_forward,
        mixed_forward_from_prompt_pair,
        normalized_faithfulness,
        normalized_necessity,
    )
    from refusal_circuit.graph import (
        load_circuit_edge_names,
        make_projection_graph,
        set_selected_edges,
    )
    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import make_system_prompt, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    trait_slug = config["trait"]["name"]
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    official_root = ROOT / config["paths"]["official_root"]
    discovery = (
        result_root / "circuit_discovery_upstream_contrastive" / args.run_name
    )
    metadata_path = discovery / "eapig_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or not metadata.get(
        "no_projection_ablation"
    ):
        raise ValueError("Discovery is not a completed non-projection run")

    import random

    trait_data_path = official_root / "eval.json"
    trait_data = json.loads(trait_data_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=args.dtype,
        revision=model_cfg.get("revision"),
    )
    by_question = {}
    rng = random.Random(args.seed)
    for question_index, question in enumerate(trait_data["questions"]):
        instruction_indices = list(range(len(trait_data["instruction"])))
        rng.shuffle(instruction_indices)
        by_question[question_index] = []
        for instruction_index in instruction_indices:
            instruction = trait_data["instruction"][instruction_index]
            positive_system = make_system_prompt(
                config["trait"]["positive_assistant_name"], instruction["pos"]
            )
            positive_prompt = model.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": positive_system},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            positive_ids = model.tokenizer.encode(
                positive_prompt, add_special_tokens=False
            )
            negative_system = make_system_prompt(
                config["trait"]["negative_assistant_name"], instruction["neg"]
            )
            negative_prompt = model.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": negative_system},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            negative_ids = model.tokenizer.encode(
                negative_prompt, add_special_tokens=False
            )
            # Skip pairs that cannot be token-aligned; this is the same
            # alignment requirement used by the upstream EAP-IG discovery path.
            if len(negative_ids) > len(positive_ids):
                continue
            aligned, alignment = _align_negative_prompt(
                model.tokenizer,
                question=question,
                negative_assistant_name=config["trait"]["negative_assistant_name"],
                negative_instruction=instruction["neg"],
                positive_prompt=positive_prompt,
            )
            by_question[question_index].append(
                {
                    "question_index": question_index,
                    "instruction_index": instruction_index,
                    "positive_prompt": positive_prompt,
                    "aligned_negative_prompt": aligned,
                    "alignment": alignment,
                }
            )
    selected = []
    for depth in range(max(map(len, by_question.values()))):
        for question_index in sorted(by_question):
            if depth < len(by_question[question_index]):
                selected.append(by_question[question_index][depth])
    selected = selected[: args.num_pairs]
    if len(selected) != args.num_pairs:
        raise ValueError(
            f"Requested {args.num_pairs} held-out pairs, found {len(selected)}"
        )

    # Use the trait-specific vector artifact so held-out evaluation stays aligned.
    direction_path = result_root / f"vector/{trait_slug}_direction.pt"
    direction = torch.load(
        direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    target_layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    circuit_path = (
        args.circuit_path
        if args.circuit_path is not None
        else discovery / "circuits_nested" / f"eapig_top_{args.k}.json"
    )
    circuit_path = circuit_path if circuit_path.is_absolute() else ROOT / circuit_path
    circuit_payload = json.loads(circuit_path.read_text(encoding="utf-8"))
    if int(circuit_payload["requested_k"]) != args.k:
        raise ValueError("Circuit payload k differs from requested k.")
    if circuit_payload.get("score_source") != metadata.get("score_file"):
        raise ValueError("Circuit was not derived from the frozen discovery scores.")
    graph = make_projection_graph(model, target_layer)
    edge_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, edge_names)

    values = {
        name: []
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    instances = []
    for row in selected:
        clean_tokens, clean_mask = tokenize_prompts(
            model, [row["positive_prompt"]], max_length=args.max_length
        )
        corrupt_tokens, corrupt_mask = tokenize_prompts(
            model, [row["aligned_negative_prompt"]], max_length=args.max_length
        )
        results = {
            "clean": _normal_forward(
                model,
                clean_tokens,
                clean_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
            )[0],
            "corrupt": _normal_forward(
                model,
                corrupt_tokens,
                corrupt_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
            )[0],
            "circuit_only": mixed_forward_from_prompt_pair(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_tokens,
                corrupt_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
                patch_selected=False,
            )[0],
            "circuit_ablated": mixed_forward_from_prompt_pair(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_tokens,
                corrupt_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
                patch_selected=True,
            )[0],
        }
        record = {
            "question_index": row["question_index"],
            "instruction_index": row["instruction_index"],
        }
        for name, tensor in results.items():
            value = float(tensor.item())
            values[name].append(value)
            record[name] = value
        instances.append(record)

    tensors = {name: torch.tensor(items) for name, items in values.items()}
    output_name = (
        args.output_tag
        or f"k{args.k}_n{args.num_pairs}_seed{args.seed}"
    )
    output = discovery / "evaluation" / output_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(instances).to_csv(output / "instance_metrics.csv", index=False)
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "method": "contrastive-state-path EAP-IG",
        "no_projection_ablation": True,
        "claim_scope": "held-out upstream positive-versus-negative persona mediation",
        "run_name": args.run_name,
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "n_pairs": len(selected),
        "seed": args.seed,
        "prompt_pairs": [
            {
                "question_index": row["question_index"],
                "instruction_index": row["instruction_index"],
            }
            for row in selected
        ],
        "source": str(trait_data_path.relative_to(ROOT)),
        "source_sha256": sha256_file(trait_data_path),
        "discovery_metadata": str(metadata_path.relative_to(ROOT)),
        "discovery_metadata_sha256": sha256_file(metadata_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "metric": (
            "final-prompt signed projection onto unit "
            f"{trait_slug} direction"
        ),
        "corrupt_baseline": "matched token-aligned negative-persona prompt state",
        "means": {
            name: float(tensor.mean().item()) for name, tensor in tensors.items()
        },
        "faithfulness": normalized_faithfulness(
            tensors["clean"], tensors["corrupt"], tensors["circuit_only"]
        ),
        "necessity": normalized_necessity(
            tensors["clean"], tensors["corrupt"], tensors["circuit_ablated"]
        ),
        "max_length": args.max_length,
        "behavioral_validation_required": True,
    }
    write_json(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

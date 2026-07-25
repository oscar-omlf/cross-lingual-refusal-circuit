#!/usr/bin/env python3
"""Evaluate a refusal-equivalent upstream persona circuit on held-out prompts."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _bootstrap import ROOT


def _heldout_prompt_groups(
    rows: list[dict],
    *,
    excluded: set[tuple[int, int]],
    n: int,
    seed: int,
) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (int(row["question_index"]), int(row["instruction_index"]))
        if key not in excluded:
            groups[key].append(row)
    representatives = []
    for (question_index, instruction_index), values in groups.items():
        prompts = {row["prompt"] for row in values}
        if len(prompts) != 1:
            raise ValueError(f"Non-unique held-out prompt group {(question_index, instruction_index)}")
        representatives.append(
            {
                "question_index": question_index,
                "instruction_index": instruction_index,
                "prompt": next(iter(prompts)),
                "mean_trait_score": sum(float(row["trait_score"]) for row in values)
                / len(values),
            }
        )
    by_question: dict[int, list[dict]] = defaultdict(list)
    for row in representatives:
        by_question[row["question_index"]].append(row)
    rng = random.Random(seed)
    for values in by_question.values():
        rng.shuffle(values)
        values.sort(key=lambda row: row["mean_trait_score"], reverse=True)
    balanced = []
    depth = 0
    while True:
        added = False
        for question_index in sorted(by_question):
            if depth < len(by_question[question_index]):
                balanced.append(by_question[question_index][depth])
                added = True
        if not added:
            break
        depth += 1
    if len(balanced) < n:
        raise ValueError(f"Requested {n} held-out prompt groups, found {len(balanced)}")
    return balanced[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=int, default=3000)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import (
        _mixed_forward,
        _normal_forward,
        normalized_faithfulness,
        normalized_necessity,
    )
    from refusal_circuit.graph import (
        load_circuit_edge_names,
        make_projection_graph,
        set_selected_edges,
    )
    from refusal_circuit.interventions import all_layer_ablation_hooks
    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    discovery = result_root / "circuit_discovery_prompt_last" / args.run_name
    metadata_path = discovery / "eapig_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("Discovery run is not complete")
    if not metadata.get("refusal_equivalent_protocol"):
        raise ValueError("Discovery run is not refusal-equivalent")
    if int(metadata["requested_k"]) != args.k:
        raise ValueError("Requested k differs from the discovery circuit")

    excluded = {
        (int(row["question_index"]), int(row["instruction_index"]))
        for row in metadata["selection"]["prompt_groups"]
    }
    positive_path = intermediate / "judged/pos_scored.jsonl"
    selected = _heldout_prompt_groups(
        read_jsonl(positive_path),
        excluded=excluded,
        n=args.num_prompts,
        seed=args.seed,
    )
    model_cfg = config["model"]
    model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=args.dtype,
        revision=model_cfg.get("revision"),
    )
    direction_path = result_root / "vector/sycophantic_direction.pt"
    direction = torch.load(
        direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    target_layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    circuit_path = discovery / "circuits_nested" / f"eapig_top_{args.k}.json"
    graph = make_projection_graph(model, target_layer)
    edge_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, edge_names)

    values = {
        name: []
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    instances = []
    for row in selected:
        tokens, attention_mask = tokenize_prompts(
            model, [row["prompt"]], max_length=args.max_length
        )
        results = {
            "clean": _normal_forward(
                model,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
            )[0],
            "corrupt": _normal_forward(
                model,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
                fwd_hooks=all_layer_ablation_hooks(model, direction),
            )[0],
            "circuit_only": _mixed_forward(
                model,
                graph,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
                patch_selected=False,
            )[0],
            "circuit_ablated": _mixed_forward(
                model,
                graph,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
                patch_selected=True,
            )[0],
        }
        record = {
            "question_index": row["question_index"],
            "instruction_index": row["instruction_index"],
            "mean_trait_score": row["mean_trait_score"],
        }
        for name, tensor in results.items():
            value = float(tensor.item())
            values[name].append(value)
            record[name] = value
        instances.append(record)

    tensors = {name: torch.tensor(items) for name, items in values.items()}
    output = discovery / "evaluation" / f"k{args.k}_n{args.num_prompts}_seed{args.seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(instances).to_csv(output / "instance_metrics.csv", index=False)
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "refusal_equivalent_protocol": True,
        "claim_scope": "held-out prompt-time upstream projection mediation",
        "run_name": args.run_name,
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "n_prompts": len(selected),
        "seed": args.seed,
        "prompt_groups": [
            {
                "question_index": row["question_index"],
                "instruction_index": row["instruction_index"],
            }
            for row in selected
        ],
        "source": str(positive_path.relative_to(ROOT)),
        "source_sha256": sha256_file(positive_path),
        "discovery_metadata": str(metadata_path.relative_to(ROOT)),
        "discovery_metadata_sha256": sha256_file(metadata_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "metric": "final-prompt-token residual projection onto unit sycophancy direction",
        "corrupt_baseline": (
            "same prompt with unit sycophancy direction removed at every residual layer"
        ),
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

#!/usr/bin/env python3
"""Discover the refusal-equivalent upstream persona-vector circuit.

The graph begins at the layer-0 residual input and terminates at the
layer-20 persona-vector projection on the final prompt token. Clean and
corrupt use the same prompt; corrupt removes the unit persona direction from
every residual stream, exactly matching the intervention path used by the
refusal circuit.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from _bootstrap import ROOT


def _select_prompt_groups(rows: list[dict], *, n: int, seed: int) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["question_index"]), int(row["instruction_index"]))].append(row)
    representatives = []
    for (question_index, instruction_index), values in groups.items():
        prompts = {row["prompt"] for row in values}
        if len(prompts) != 1:
            raise ValueError(
                f"Prompt group {(question_index, instruction_index)} is not unique"
            )
        representatives.append(
            {
                "question_index": question_index,
                "instruction_index": instruction_index,
                "prompt": next(iter(prompts)),
                "mean_trait_score": sum(float(row["trait_score"]) for row in values)
                / len(values),
                "mean_coherence_score": sum(
                    float(row["coherence_score"]) for row in values
                )
                / len(values),
                "rollouts": len(values),
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
    for depth in range(max(map(len, by_question.values()))):
        for question_index in sorted(by_question):
            if depth < len(by_question[question_index]):
                balanced.append(by_question[question_index][depth])
    if len(balanced) < n:
        raise ValueError(f"Requested {n} prompt groups, found {len(balanced)}")
    return balanced[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--k", type=int, default=3000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("--run-name must be a filesystem-safe identifier")

    import torch

    from refusal_circuit.eapig import AttributionConfig, attribute
    from refusal_circuit.graph import (
        edge_record,
        make_projection_graph,
        paper_greedy_edges,
        write_edge_scores_parquet,
        write_top_edges_summary,
    )
    from refusal_circuit.model import load_model
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    output = result_root / "circuit_discovery_prompt_last" / args.run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results")

    positive_path = intermediate / "judged/pos_scored.jsonl"
    selected = _select_prompt_groups(
        read_jsonl(positive_path), n=args.num_prompts, seed=args.seed
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
    graph = make_projection_graph(model, target_layer)
    attribute(
        model,
        graph,
        [row["prompt"] for row in selected],
        direction=direction,
        target_hook=target_hook,
        target_position=-1,
        config=AttributionConfig(
            batch_size=args.batch_size,
            max_length=args.max_length,
            ig_steps=args.ig_steps,
        ),
        integrated_gradients=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "eapig_edge_scores.parquet"
    write_edge_scores_parquet(graph, score_path)
    write_top_edges_summary(
        graph, output / "top_edges_summary_eapig.md", method="prompt_last_eapig"
    )
    selected_edges = paper_greedy_edges(
        graph, min(args.k, len(graph.edges)), prune_parentless=False
    )
    circuit_path = output / "circuits_nested" / f"eapig_top_{args.k}.json"
    write_json(
        circuit_path,
        {
            "schema_version": 1,
            "method": "intervention-path EAP-IG",
            "selection": "paper_greedy_absolute_score_with_childless_pruning_only",
            "claim_scope": "upstream prompt-time persona-vector formation circuit",
            "requested_k": args.k,
            "retained_edges": len(selected_edges),
            "score_source": str(score_path.relative_to(ROOT)),
            "target": {
                "hook": target_hook,
                "layer": target_layer,
                "position": -1,
                "timing": "final prompt token before response generation",
                "metric": "residual_projection_onto_unit_sycophancy_direction",
            },
            "edges": [edge_record(edge) for edge in selected_edges],
        },
    )
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "refusal_equivalent_protocol": True,
        "method": "intervention-path EAP-IG",
        "graph": "blocks.0.hook_resid_pre through blocks.20.hook_resid_pre",
        "metric": "final-prompt-token residual projection onto unit sycophancy direction",
        "target_hook": target_hook,
        "target_layer": target_layer,
        "target_position": -1,
        "corrupt_baseline": (
            "same prompt with unit sycophancy direction removed at every residual layer"
        ),
        "interpolation": "global direction-ablation strength 1 (corrupt) to 0 (clean)",
        "model": model_cfg,
        "trait": config["trait"]["name"],
        "source": str(positive_path.relative_to(ROOT)),
        "source_sha256": sha256_file(positive_path),
        "selection": {
            "method": "question-balanced positive prompt groups ranked within question by mean judged trait",
            "seed": args.seed,
            "n_prompts": len(selected),
            "prompt_groups": [
                {
                    key: row[key]
                    for key in (
                        "question_index",
                        "instruction_index",
                        "mean_trait_score",
                        "mean_coherence_score",
                        "rollouts",
                    )
                }
                for row in selected
            ],
        },
        "direction": str(direction_path.relative_to(ROOT)),
        "direction_sha256": sha256_file(direction_path),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "ig_steps": args.ig_steps,
        "eligible_edges": len(graph.edges),
        "score_file": str(score_path.relative_to(ROOT)),
        "score_sha256": sha256_file(score_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "requested_k": args.k,
        "retained_edges": len(selected_edges),
    }
    write_json(output / "eapig_run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

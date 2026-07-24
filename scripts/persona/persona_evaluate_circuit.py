#!/usr/bin/env python3
"""Evaluate post-response projection faithfulness on held-out transcripts.

This is a representation diagnostic, not a held-out behavioral-causality test:
the target is evaluated after the response tokens are supplied to the model.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _bootstrap import ROOT


def _balanced(rows: list[dict], *, seed: int) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["question_index"])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda row: row["pair_id"])
        rng.shuffle(values)
    output = []
    for depth in range(max(map(len, groups.values()))):
        for question in sorted(groups):
            if depth < len(groups[question]):
                output.append(groups[question][depth])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--k", type=int, default=3000)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--reuse-baseline-from",
        type=Path,
        default=None,
        help=(
            "Completed circuit-evaluation directory with identical selected records. "
            "Reuses only its k-independent clean/corrupt values."
        ),
    )
    args = parser.parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import (
        _mixed_forward,
        _normal_forward,
        normalized_faithfulness,
        normalized_necessity,
    )
    from refusal_circuit.eapig import _response_mask
    from refusal_circuit.graph import load_circuit_edge_names, make_projection_graph, set_selected_edges
    from refusal_circuit.interventions import all_layer_ablation_hooks
    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    discovery_meta = json.loads((result_root / "circuit_discovery/eapig_run_metadata.json").read_text())
    excluded = set(discovery_meta["record_ids"])
    effective = set(pd.read_csv(result_root / "vector/effective_pairs.csv")["pair_id"].astype(str))
    positive_path = intermediate / "judged/pos_scored.jsonl"
    candidates = [
        row for row in read_jsonl(positive_path)
        if row["pair_id"] in effective and row["record_id"] not in excluded
    ]
    selected = _balanced(candidates, seed=args.seed)[: args.num_prompts]
    if len(selected) != args.num_prompts:
        raise ValueError(f"Only {len(selected)} held-out transcripts available")
    selected_ids = [row["record_id"] for row in selected]

    reused_baseline = None
    baseline_by_id: dict[str, dict] = {}
    if args.reuse_baseline_from is not None:
        baseline_root = (
            args.reuse_baseline_from
            if args.reuse_baseline_from.is_absolute()
            else ROOT / args.reuse_baseline_from
        )
        baseline_aggregate_path = baseline_root / "aggregate.json"
        baseline_instances_path = baseline_root / "instance_metrics.csv"
        baseline_aggregate = json.loads(baseline_aggregate_path.read_text(encoding="utf-8"))
        baseline_instances = pd.read_csv(baseline_instances_path)
        if baseline_aggregate.get("status") != "complete":
            raise ValueError("Reusable baseline evaluation is not complete")
        if baseline_aggregate.get("record_ids") != selected_ids:
            raise ValueError("Reusable baseline record IDs differ from this evaluation")
        if baseline_aggregate.get("source_sha256") != sha256_file(positive_path):
            raise ValueError("Reusable baseline source hash differs from this evaluation")
        if set(baseline_instances["record_id"]) != set(selected_ids):
            raise ValueError("Reusable baseline instance rows differ from this evaluation")
        baseline_by_id = {
            str(row["record_id"]): {
                "clean": float(row["clean"]),
                "corrupt": float(row["corrupt"]),
            }
            for row in baseline_instances.to_dict(orient="records")
        }
        reused_baseline = {
            "directory": str(baseline_root.relative_to(ROOT)),
            "aggregate_sha256": sha256_file(baseline_aggregate_path),
            "instance_metrics_sha256": sha256_file(baseline_instances_path),
            "fields": ["clean", "corrupt"],
        }

    model_cfg = config["model"]
    model = load_model(
        model_cfg["name"], device=args.device, dtype=args.dtype, revision=model_cfg.get("revision")
    )
    direction_path = result_root / "vector/sycophantic_direction.pt"
    direction = torch.load(direction_path, map_location=next(model.parameters()).device, weights_only=True)
    if reused_baseline is not None and baseline_aggregate.get("direction_sha256") != sha256_file(
        direction_path
    ):
        raise ValueError("Reusable baseline direction hash differs from this evaluation")
    target_layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    circuit_path = result_root / f"circuit_discovery/circuits_nested/eapig_top_{args.k}.json"
    edge_names = load_circuit_edge_names(circuit_path)
    graph = make_projection_graph(model, target_layer)
    set_selected_edges(graph, edge_names)

    values = {name: [] for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")}
    instance_rows = []
    for row in selected:
        text = row["prompt"] + row["answer"]
        start = len(model.tokenizer.encode(row["prompt"], add_special_tokens=False))
        tokens, attention_mask = tokenize_prompts(model, [text], max_length=args.max_length)
        mask = _response_mask(tokens, attention_mask, [start])
        if int(mask.sum()) == 0:
            raise ValueError(f"No retained response tokens for {row['record_id']}")
        results = {
            "circuit_only": _mixed_forward(
                model, graph, tokens, attention_mask, target_hook=target_hook, target_position=-1,
                direction=direction, target_mask=mask, patch_selected=False,
            )[0],
            "circuit_ablated": _mixed_forward(
                model, graph, tokens, attention_mask, target_hook=target_hook, target_position=-1,
                direction=direction, target_mask=mask, patch_selected=True,
            )[0],
        }
        if baseline_by_id:
            results = {**baseline_by_id[row["record_id"]], **results}
        else:
            results = {
                "clean": _normal_forward(
                    model, tokens, attention_mask, target_hook=target_hook, target_position=-1,
                    direction=direction, target_mask=mask,
                )[0],
                "corrupt": _normal_forward(
                    model, tokens, attention_mask, target_hook=target_hook, target_position=-1,
                    direction=direction, target_mask=mask,
                    fwd_hooks=all_layer_ablation_hooks(model, direction),
                )[0],
                **results,
            }
        record = {"record_id": row["record_id"], "pair_id": row["pair_id"]}
        for name, tensor in results.items():
            value = float(tensor.item()) if hasattr(tensor, "item") else float(tensor)
            values[name].append(value)
            record[name] = value
        instance_rows.append(record)

    tensors = {name: torch.tensor(items) for name, items in values.items()}
    faithfulness = normalized_faithfulness(
        tensors["clean"], tensors["corrupt"], tensors["circuit_only"]
    )
    necessity = normalized_necessity(
        tensors["clean"], tensors["corrupt"], tensors["circuit_ablated"]
    )
    output = result_root / f"circuit_evaluation/k{args.k}_heldout_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(instance_rows).to_csv(output / "instance_metrics.csv", index=False)
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "n_prompts": len(selected),
        "seed": args.seed,
        "selection": "question-balanced effective positive transcripts excluding discovery records",
        "record_ids": selected_ids,
        "source": str(positive_path.relative_to(ROOT)),
        "source_sha256": sha256_file(positive_path),
        "direction_sha256": sha256_file(direction_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "means": {name: float(tensor.mean()) for name, tensor in tensors.items()},
        "faithfulness": faithfulness,
        "necessity": necessity,
        "metric": "mean response-token residual projection onto unit sycophancy direction",
        "claim_scope": "post-generation representation diagnostic only",
        "behavior_leakage_warning": "Held-out responses are teacher-forced before the metric is measured.",
        "corrupt_baseline": "same transcript with global sycophancy-direction ablation",
        "max_length": args.max_length,
        "reused_baseline": reused_baseline,
    }
    write_json(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

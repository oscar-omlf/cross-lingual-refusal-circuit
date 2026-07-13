#!/usr/bin/env python3
"""Evaluate a frozen source-language circuit on one target PolyRefuse language.

For each source circuit this reports two non-interchangeable evaluations:

* ``target_local_anchor_metric``: does the source edge set preserve the
  target language's own anchor refusal representation?
* ``source_english_anchor_metric``: does it preserve the exact English
  anchor representation on target-language prompts?

Both use a clean/corrupt baseline defined by the *same metric direction* in
that row, so normalized faithfulness and necessity remain well-defined.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--method", choices=("eapig", "eap"), default="eapig")
    parser.add_argument("--k-values", type=int, nargs="+", default=(50, 100, 200, 500, 1000, 2000))
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _summarise(conditions, refusal_ids: list[int]):
    from refusal_circuit.circuit import normalized_faithfulness, normalized_necessity
    from refusal_circuit.metrics import refusal_log_odds

    row = {
        f"m_{name}": result.projection.mean().item() for name, result in conditions.items()
    }
    row["faithfulness"] = normalized_faithfulness(
        conditions["clean"].projection,
        conditions["corrupt"].projection,
        conditions["circuit_only"].projection,
    )
    row["necessity"] = normalized_necessity(
        conditions["clean"].projection,
        conditions["corrupt"].projection,
        conditions["circuit_ablated"].projection,
    )
    for name, result in conditions.items():
        row[f"first_token_refusal_log_odds_{name}"] = refusal_log_odds(
            result.logits.unsqueeze(1), refusal_ids
        ).mean().item()
    return row


def main():
    args = parse_args()
    import pandas as pd
    import torch

    from refusal_circuit.circuit import evaluate_conditions
    from refusal_circuit.data import load_processed_split
    from refusal_circuit.graph import load_circuit_edge_names, make_projection_graph, set_selected_edges
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import experiment_data_dir, experiment_results_dir, read_json, write_json

    source = args.source_language
    target = args.target_language
    default_results_root = experiment_results_dir(args.model, "polyrefuse", "en").parent
    results_root = args.results_root or default_results_root
    data_root = args.data_root or experiment_data_dir(args.model, "polyrefuse", "en").parent
    output_dir = args.output_dir or results_root / "cross_language" / f"{source}_to_{target}"
    source_direction_dir = results_root / source / "direction"
    target_direction_dir = results_root / target / "direction"
    source_meta = read_json(source_direction_dir / "anchor_direction_metadata.json")
    target_meta = read_json(target_direction_dir / "anchor_direction_metadata.json")
    for metadata, label in ((source_meta, "source"), (target_meta, "target")):
        if metadata.get("model") != args.model:
            raise ValueError(f"{label} anchor direction is for {metadata.get('model')}, not {args.model}.")
    if (source_meta["layer"], source_meta["position"]) != (target_meta["layer"], target_meta["position"]):
        raise ValueError("Cross-language circuit transfer requires common anchor metadata.")
    target_rows = load_processed_split("test", "harmful", data_dir=data_root / target)[: args.num_prompts]
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    prompts = format_instructions(model.tokenizer, [row["instruction"] for row in target_rows])
    source_direction = torch.load(
        source_direction_dir / "anchor_refusal_direction.pt",
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    target_direction = torch.load(
        target_direction_dir / "anchor_refusal_direction.pt",
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    target_refusal_ids = [int(value) for value in target_meta.get("refusal_token_ids", [])]
    if not target_refusal_ids:
        raise ValueError("Target anchor metadata has no language-local refusal token IDs.")

    rows = []
    for requested_k in args.k_values:
        circuit_path = (
            results_root / source / "circuit_discovery" / "circuits" / f"{args.method}_top_{requested_k}.json"
        )
        if not circuit_path.exists():
            raise FileNotFoundError(f"Missing frozen source circuit {circuit_path}.")
        edge_names = load_circuit_edge_names(circuit_path)
        for metric_name, direction, metadata in (
            ("target_local_anchor_metric", target_direction, target_meta),
            ("source_anchor_metric", source_direction, source_meta),
        ):
            graph = make_projection_graph(model, int(metadata["layer"]))
            set_selected_edges(graph, edge_names)
            conditions = evaluate_conditions(
                model,
                graph,
                prompts,
                direction=direction,
                target_hook=metadata["target_hook"],
                target_position=int(metadata["position"]),
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            row = {
                "model": args.model,
                "dataset": "PolyRefuse",
                "source_language": source,
                "target_language": target,
                "metric": metric_name,
                "method": args.method,
                "requested_k": requested_k,
                "retained_edges": len(edge_names),
                "n_prompts": len(prompts),
                "metric_layer": int(metadata["layer"]),
                "metric_position": int(metadata["position"]),
            }
            row.update(_summarise(conditions, target_refusal_ids))
            rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "cross_language_faithfulness_necessity.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    write_json(
        output_dir / "cross_language_evaluation_metadata.json",
        {
            "model": args.model,
            "dataset": "PolyRefuse",
            "source_language": source,
            "target_language": target,
            "source_circuit": "frozen source-language EAP-IG greedy circuit",
            "target_prompts": "target-language held-out harmful split; prompt text is not copied to results",
            "metric_spaces": ["target_local_anchor_metric", "source_anchor_metric"],
            "baseline_rule": "Each metric uses clean and global-direction-ablated corrupt runs with that same metric direction.",
            "generated_text_saved": False,
        },
    )
    print(f"Saved source={source} target={target} transfer table to {table_path}")


if __name__ == "__main__":
    main()

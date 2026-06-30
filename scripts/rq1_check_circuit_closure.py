#!/usr/bin/env python3
"""Gate RQ1 attribution on clean/corrupt circuit-construction closure.

For a same-input, intervention-generated corrupt baseline, a faithful circuit
evaluator must satisfy four identities before edge attribution is meaningful:
the empty circuit is corrupt, the full graph is clean, ablating no edges is
clean, and ablating all graph edges is corrupt.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--direction-path", type=Path, default=Path("results/gemma-2-2b-it/en/rq1/refusal_direction.pt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("results/gemma-2-2b-it/en/rq1/refusal_direction_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gemma-2-2b-it/en/rq1"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/gemma-2-2b-it/en/rq1"),
        help="Processed-data namespace used for this closure check.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    import torch

    from refusal_circuit.circuit import (
        evaluate_conditions,
        normalized_faithfulness,
        normalized_necessity,
    )
    from refusal_circuit.data import load_processed_split
    from refusal_circuit.graph import make_projection_graph, set_selected_edges
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import read_json

    metadata = read_json(args.metadata_path)
    if metadata["model"] != args.model:
        raise ValueError(f"Direction is for {metadata['model']}, but --model is {args.model}.")
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(args.direction_path, map_location=next(model.parameters()).device, weights_only=True)
    prompts = format_instructions(
        model.tokenizer,
        [
            row["instruction"]
            for row in load_processed_split(args.split, "harmful", data_dir=args.data_dir)[: args.num_prompts]
        ],
    )
    if not prompts:
        raise ValueError("No harmful prompts were available for the closure check.")

    rows = []
    expected = {
        "empty": {"faithfulness": 0.0, "necessity": 0.0},
        "full": {"faithfulness": 1.0, "necessity": 1.0},
    }
    for label, selected_all in (("empty", False), ("full", True)):
        graph = make_projection_graph(model, int(metadata["layer"]))
        selected = set(graph.edges) if selected_all else set()
        set_selected_edges(graph, selected)
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
        clean = conditions["clean"].projection
        corrupt = conditions["corrupt"].projection
        circuit_only = conditions["circuit_only"].projection
        circuit_ablated = conditions["circuit_ablated"].projection
        rows.append(
            {
                "circuit": label,
                "selected_edges": len(selected),
                "m_clean": clean.mean().item(),
                "m_corrupt": corrupt.mean().item(),
                "m_circuit_only": circuit_only.mean().item(),
                "m_circuit_ablated": circuit_ablated.mean().item(),
                "faithfulness": normalized_faithfulness(clean, corrupt, circuit_only),
                "necessity": normalized_necessity(clean, corrupt, circuit_ablated),
                "expected_faithfulness": expected[label]["faithfulness"],
                "expected_necessity": expected[label]["necessity"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "baseline_closure.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    failures = []
    for row in rows:
        for metric in ("faithfulness", "necessity"):
            value = row[metric]
            expected_value = row[f"expected_{metric}"]
            if not math.isfinite(value) or abs(value - expected_value) > args.tolerance:
                failures.append(
                    f"{row['circuit']} {metric}={value:.5f}; expected {expected_value:.1f} +/- {args.tolerance}"
                )
    print(frame.to_string(index=False))
    print(f"Saved closure diagnostics to {output_path}")
    if failures:
        raise RuntimeError("Circuit closure gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()

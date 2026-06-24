#!/usr/bin/env python3
"""Run an aggregate-only behavioral check for one saved RQ1 circuit.

The script deliberately writes only rates and logit margins; generated harmful
completions are kept in memory solely for prefix matching and are never saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--circuit-path",
        type=Path,
        default=Path("results/rq1/circuits/eapig_top_1000.json"),
    )
    parser.add_argument("--direction-path", type=Path, default=Path("results/rq1/refusal_direction.pt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("results/rq1/refusal_direction_metadata.json"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/rq1/behavioural_sanity_top1000.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    import torch

    from refusal_circuit.circuit import behavioural_sanity_check
    from refusal_circuit.data import load_processed_split
    from refusal_circuit.graph import load_circuit_edge_names, make_projection_graph, set_selected_edges
    from refusal_circuit.metrics import DEFAULT_ANSWER_STARTS, REFUSAL_PREFIXES
    from refusal_circuit.model import first_token_ids, format_instructions, load_model
    from refusal_circuit.paths import read_json

    metadata = read_json(args.metadata_path)
    if metadata["model"] != args.model:
        raise ValueError(f"Direction is for {metadata['model']}, but --model is {args.model}.")
    if not args.circuit_path.exists():
        raise FileNotFoundError(f"Missing circuit JSON: {args.circuit_path}")
    circuit = json.loads(args.circuit_path.read_text(encoding="utf-8"))
    if circuit.get("method") != "eapig":
        raise ValueError("Behavioral check is intended for an EAP-IG circuit.")

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(args.direction_path, map_location=next(model.parameters()).device, weights_only=True)
    prompts = format_instructions(
        model.tokenizer,
        [row["instruction"] for row in load_processed_split(args.split, "harmful")[: args.num_prompts]],
    )
    graph = make_projection_graph(model, int(metadata["layer"]))
    set_selected_edges(graph, load_circuit_edge_names(args.circuit_path))
    rows = behavioural_sanity_check(
        model,
        graph,
        prompts,
        direction=direction,
        target_hook=metadata["target_hook"],
        target_position=int(metadata["position"]),
        refusal_ids=first_token_ids(model.tokenizer, REFUSAL_PREFIXES),
        answer_ids=first_token_ids(model.tokenizer, DEFAULT_ANSWER_STARTS),
        max_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
    )
    frame = pd.DataFrame(rows)
    frame["circuit_requested_k"] = int(circuit["requested_k"])
    frame["circuit_retained_edges"] = int(circuit["retained_edges"])
    frame["circuit_path"] = str(args.circuit_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_path, index=False)
    print(frame.to_string(index=False))
    print(f"Saved aggregate-only behavioral summary to {args.output_path}")


if __name__ == "__main__":
    main()

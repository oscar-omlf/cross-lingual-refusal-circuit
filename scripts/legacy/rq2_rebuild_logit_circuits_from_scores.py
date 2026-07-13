#!/usr/bin/env python3
"""Rebuild logit-target circuit JSONs from saved EAP/EAP-IG edge scores."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-path",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/polyrefuse/en/"
            "circuit_discovery_logit_english_vector_pilot/eapig_edge_scores.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/polyrefuse/en/"
            "circuit_discovery_logit_english_vector_pilot/circuits_greedy_no_parent_prune"
        ),
    )
    parser.add_argument("--method", choices=("eapig", "eap"), default="eapig")
    parser.add_argument("--k-values", type=int, nargs="+", default=(5000, 10000, 20000, 50000))
    parser.add_argument("--refusal-token-ids", type=int, nargs="+", default=(40, 2121))
    return parser.parse_args()


def _infer_n_heads(edge_names) -> int:
    heads: list[int] = []
    for name in edge_names:
        heads.extend(int(match) for match in re.findall(r"\.h(\d+)", str(name)))
    if not heads:
        raise ValueError("Could not infer number of heads from edge names.")
    return max(heads) + 1


def _infer_n_layers(edge_names) -> int:
    layers: list[int] = []
    for name in edge_names:
        layers.extend(int(match) for match in re.findall(r"(?:^|[>.])(?:a|m)(\d+)", str(name)))
    if not layers:
        raise ValueError("Could not infer number of layers from edge names.")
    return max(layers) + 1


def main() -> None:
    args = parse_args()

    import pandas as pd
    from eap.graph import Graph

    from refusal_circuit.graph import write_logit_circuit_jsons

    frame = pd.read_parquet(args.score_path)
    n_heads = _infer_n_heads(frame["name"])
    n_layers = _infer_n_layers(frame["name"])
    graph = Graph.from_model(
        {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "parallel_attn_mlp": False,
        }
    )
    graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"

    missing = set(frame["name"]) - set(graph.edges)
    extra = set(graph.edges) - set(frame["name"])
    if missing or extra:
        raise ValueError(
            f"Score table and logit graph do not match: {len(missing)} missing, {len(extra)} extra."
        )
    scores = dict(zip(frame["name"], frame["score"]))
    for name, edge in graph.edges.items():
        edge.score = float(scores[name])

    paths = write_logit_circuit_jsons(
        graph,
        method=args.method,
        refusal_token_ids=list(args.refusal_token_ids),
        output_dir=args.output_dir,
        k_values=list(args.k_values),
        prune_parentless=False,
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build connected sycophancy input-to-logits circuits from saved scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "EAP-IG")]

from refusal_circuit.graph import (
    circuit_connectivity,
    edge_record,
    paper_greedy_edges,
)
from refusal_circuit.persona import sha256_file
from refusal_circuit.paths import write_json


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def make_graph():
    from eap.graph import Graph

    graph = Graph.from_model(
        {
            "n_layers": 28,
            "n_heads": 28,
            "parallel_attn_mlp": False,
        }
    )
    graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/sycophancy/end_to_end/"
            "eapig_n64_ig5/edge_scores.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/sycophancy/end_to_end/"
            "eapig_n64_ig5/circuits"
        ),
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[100, 500, 1000, 3000]
    )
    parser.add_argument(
        "--corruption",
        choices=("natural_prompt", "paired_coordinate"),
        default="natural_prompt",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scores_path = resolve(args.scores)
    output_dir = resolve(args.output_dir)
    graph = make_graph()
    scores = pd.read_parquet(scores_path, columns=["name", "score"])
    if len(scores) != len(graph.edges) or scores["name"].nunique() != len(graph.edges):
        raise ValueError("Saved scores do not contain exactly one row per graph edge.")
    score_by_name = dict(zip(scores["name"], scores["score"], strict=True))
    unknown = set(score_by_name) - set(graph.edges)
    missing = set(graph.edges) - set(score_by_name)
    if unknown or missing:
        raise ValueError(
            f"Score/graph mismatch: {len(unknown)} unknown, {len(missing)} missing."
        )
    for name, edge in graph.edges.items():
        edge.score = float(score_by_name[name])

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for k in args.k_values:
        path = output_dir / f"eapig_top_{k}.json"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite.")
        selected = paper_greedy_edges(graph, k, prune_parentless=True)
        if not selected:
            outputs.append(
                {
                    "requested_k": k,
                    "status": "no_complete_source_to_logits_path",
                    "selection_diagnostics": graph.last_selection_diagnostics,
                }
            )
            continue
        connectivity = circuit_connectivity(
            graph, {edge.name for edge in selected}
        )
        if not connectivity["all_edges_on_source_to_sink_paths"]:
            raise RuntimeError(f"k={k} is not a connected source-to-logits circuit.")
        payload = {
            "method": (
                f"{args.corruption}_contrastive_response_logprob_eapig"
            ),
            "selection": "paper_greedy_absolute_score_with_source_sink_pruning",
            "requested_k": k,
            "retained_edges": len(selected),
            "score_file": {
                "path": str(scores_path),
                "sha256": sha256_file(scores_path),
            },
            "graph": {
                "source": "blocks.0.hook_resid_pre",
                "target": "response logits",
            },
            "target": (
                "[mean logP(sycophantic answer)-mean logP(normal answer)] "
                "under the positive instruction; clean minus the selected "
                "corruption"
            ),
            "selection_diagnostics": graph.last_selection_diagnostics,
            "connectivity": connectivity,
            "edges": [edge_record(edge) for edge in selected],
        }
        write_json(path, payload)
        outputs.append(
            {
                "path": str(path),
                "status": "complete",
                "requested_k": k,
                "retained_edges": len(selected),
                "connectivity": connectivity,
            }
        )
    write_json(
        output_dir / "metadata.json",
        {
            "score_file": str(scores_path),
            "score_sha256": sha256_file(scores_path),
            "circuits": outputs,
        },
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

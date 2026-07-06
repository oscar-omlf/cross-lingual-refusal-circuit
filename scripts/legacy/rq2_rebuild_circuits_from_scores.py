#!/usr/bin/env python3
"""Rebuild RQ2 circuit JSONs from saved EAP/EAP-IG edge scores.

This is intentionally a post-processing script: it does not rerun attribution
or load the model.  It is useful when the same score table should be converted
into a different retained-edge convention.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/qwen2.5-7b-instruct/polyrefuse"),
    )
    parser.add_argument("--languages", nargs="+", default=("en", "de", "ru"))
    parser.add_argument("--method", choices=("eapig", "eap"), default="eapig")
    parser.add_argument("--k-values", type=int, nargs="+", default=(50, 100, 200, 500, 1000, 2000))
    parser.add_argument(
        "--selection",
        choices=("greedy_no_parent_prune", "raw_topk"),
        default="greedy_no_parent_prune",
        help=(
            "greedy_no_parent_prune uses the paper greedy frontier but does not "
            "discard parentless selected nodes; raw_topk takes the global "
            "absolute-score top-k edge set."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under --discovery-subdir/. Defaults to circuits_<selection>.",
    )
    parser.add_argument(
        "--discovery-subdir",
        default="circuit_discovery",
        help="Per-language result subdirectory containing edge-score parquet files.",
    )
    parser.add_argument(
        "--metadata-subpath",
        default=None,
        help=(
            "Optional per-language metadata path relative to the language result root. "
            "Defaults to direction/{anchor,refusal}_direction_metadata.json."
        ),
    )
    return parser.parse_args()


def _metadata_path(language_root: Path, metadata_subpath: str | None = None) -> Path:
    if metadata_subpath:
        path = language_root / metadata_subpath
        if path.exists():
            return path
        raise FileNotFoundError(f"Missing metadata file: {path}")
    for name in ("anchor_direction_metadata.json", "refusal_direction_metadata.json"):
        path = language_root / "direction" / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No direction metadata found under {language_root / 'direction'}")


def _infer_n_heads(edge_names) -> int:
    heads: list[int] = []
    for name in edge_names:
        heads.extend(int(match) for match in re.findall(r"\.h(\d+)", str(name)))
    if not heads:
        raise ValueError("Could not infer number of heads from edge names.")
    return max(heads) + 1


def _build_graph_from_scores(score_path: Path, *, target_layer: int):
    import pandas as pd
    from eap.graph import Graph

    frame = pd.read_parquet(score_path)
    n_heads = _infer_n_heads(frame["name"])
    graph = Graph.from_model(
        {
            "n_layers": target_layer,
            "n_heads": n_heads,
            # Qwen2.5 and Gemma-style TransformerLens models used here do not
            # use parallel attention/MLP edges in the EAP graph.  This is
            # verified by exact edge-name equality against the saved scores.
            "parallel_attn_mlp": False,
        }
    )
    graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
    graph.nodes["logits"].layer = target_layer
    graph.nodes["logits"].in_hook = f"blocks.{target_layer}.hook_resid_pre"

    missing = set(frame["name"]) - set(graph.edges)
    extra = set(graph.edges) - set(frame["name"])
    if missing or extra:
        raise ValueError(
            f"Score table and graph do not match for {score_path}: "
            f"{len(missing)} missing, {len(extra)} extra."
        )
    scores = dict(zip(frame["name"], frame["score"]))
    for name, edge in graph.edges.items():
        edge.score = float(scores[name])
    return graph


def main():
    args = parse_args()
    from refusal_circuit.graph import edge_record, paper_greedy_edges, sorted_edges
    from refusal_circuit.paths import write_json

    output_subdir = args.output_subdir or f"circuits_{args.selection}"
    for language in args.languages:
        language_root = args.results_root / language
        metadata = json.loads(_metadata_path(language_root, args.metadata_subpath).read_text(encoding="utf-8"))
        target_layer = int(metadata["layer"])
        target_position = int(metadata["position"])
        score_path = language_root / args.discovery_subdir / f"{args.method}_edge_scores.parquet"
        output_dir = language_root / args.discovery_subdir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        graph = _build_graph_from_scores(score_path, target_layer=target_layer)
        score_source = str(score_path)
        for requested_k in args.k_values:
            if args.selection == "raw_topk":
                selected = sorted_edges(graph)[: min(requested_k, len(graph.edges))]
                selection_label = "raw_global_absolute_score_topk_no_connectivity_pruning"
            else:
                selected = paper_greedy_edges(
                    graph,
                    min(requested_k, len(graph.edges)),
                    prune_parentless=False,
                )
                selection_label = "paper_greedy_absolute_score_with_childless_pruning_only"

            payload = {
                "method": args.method,
                "selection": selection_label,
                "requested_k": requested_k,
                "retained_edges": len(selected),
                "score_source": score_source,
                "target": {
                    "hook": f"blocks.{target_layer}.hook_resid_pre",
                    "layer": target_layer,
                    "position": target_position,
                    "metric": "residual_projection_onto_refusal_direction",
                },
                "edges": [edge_record(edge) for edge in selected],
            }
            write_json(output_dir / f"{args.method}_top_{requested_k}.json", payload)
            print(
                f"{language} {args.method} k={requested_k}: "
                f"saved {len(selected)} edges to {output_dir}"
            )


if __name__ == "__main__":
    main()

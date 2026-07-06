#!/usr/bin/env python3
"""Build multilingual consensus circuits from averaged edge scores.

This complements structural intersections.  For each language we take the
full EAP/EAP-IG score table, normalize its absolute scale, average signed edge
scores across languages, and then run the same connected greedy frontier used
for ordinary circuit extraction.  Functional evaluation is handled separately
by ``rq1_evaluate_faithfulness.py`` on each target language.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--method", choices=("eapig", "eap"), default="eapig")
    parser.add_argument("--k-values", type=int, nargs="+", default=(10, 25, 50, 100, 200, 500, 1000, 2000))
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--discovery-subdir", default="circuit_discovery_english_vector")
    parser.add_argument("--metadata-subpath", default="shared_metrics/english_vector/metadata.json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-subdir", default="circuits_greedy_no_parent_prune_small_to_large")
    parser.add_argument(
        "--normalization",
        choices=("l1_abs", "l2"),
        default="l1_abs",
        help="Per-language score normalization before averaging.",
    )
    return parser.parse_args()


def _infer_n_heads(edge_names) -> int:
    heads: list[int] = []
    for name in edge_names:
        heads.extend(int(match) for match in re.findall(r"\.h(\d+)", str(name)))
    if not heads:
        raise ValueError("Could not infer number of heads from edge names.")
    return max(heads) + 1


def _build_graph(edge_names, scores, *, target_layer: int):
    from eap.graph import Graph

    graph = Graph.from_model(
        {
            "n_layers": target_layer,
            "n_heads": _infer_n_heads(edge_names),
            "parallel_attn_mlp": False,
        }
    )
    graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
    graph.nodes["logits"].layer = target_layer
    graph.nodes["logits"].in_hook = f"blocks.{target_layer}.hook_resid_pre"
    missing = set(edge_names) - set(graph.edges)
    extra = set(graph.edges) - set(edge_names)
    if missing or extra:
        raise ValueError(f"Consensus graph mismatch: {len(missing)} missing, {len(extra)} extra.")
    for name, edge in graph.edges.items():
        edge.score = float(scores[name])
    return graph


def main():
    args = parse_args()
    import pandas as pd

    from refusal_circuit.graph import edge_record, paper_greedy_edges
    from refusal_circuit.paths import experiment_results_dir, write_json

    languages = list(dict.fromkeys(args.languages))
    results_root = args.results_root or experiment_results_dir(args.model, "polyrefuse", "en").parent
    output_root = args.output_root or results_root / "shared_circuits" / args.discovery_subdir / "consensus_mean_scores"
    score_frames = []
    metadata = None
    for language in languages:
        score_path = results_root / language / args.discovery_subdir / f"{args.method}_edge_scores.parquet"
        metadata_path = results_root / language / args.metadata_subpath
        if not score_path.exists():
            raise FileNotFoundError(f"Missing score table: {score_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metric metadata: {metadata_path}")
        current_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata is None:
            metadata = current_meta
        elif (int(metadata["layer"]), int(metadata["position"])) != (
            int(current_meta["layer"]),
            int(current_meta["position"]),
        ):
            raise ValueError("Consensus circuits require all languages to share one metric site.")
        frame = pd.read_parquet(score_path).set_index("name")["score"].astype(float).sort_index()
        if args.normalization == "l1_abs":
            scale = frame.abs().sum()
        else:
            scale = (frame.pow(2).sum()) ** 0.5
        if not scale:
            raise ValueError(f"All-zero score table for {language}: {score_path}")
        score_frames.append(frame / scale)

    joined = pd.concat(score_frames, axis=1, join="inner")
    joined.columns = languages
    consensus = joined.mean(axis=1)
    target_layer = int(metadata["layer"])
    target_position = int(metadata["position"])
    graph = _build_graph(consensus.index, consensus.to_dict(), target_layer=target_layer)

    circuits_dir = output_root / args.output_subdir
    circuits_dir.mkdir(parents=True, exist_ok=True)
    for requested_k in args.k_values:
        selected = paper_greedy_edges(graph, min(requested_k, len(graph.edges)), prune_parentless=False)
        payload = {
            "method": args.method,
            "selection": f"consensus_mean_signed_scores_{args.normalization}_paper_greedy_childless_pruning_only",
            "requested_k": requested_k,
            "retained_edges": len(selected),
            "source_languages": languages,
            "target": {
                "hook": f"blocks.{target_layer}.hook_resid_pre",
                "layer": target_layer,
                "position": target_position,
                "metric": "residual_projection_onto_refusal_direction",
            },
            "edges": [edge_record(edge) for edge in selected],
        }
        write_json(circuits_dir / f"{args.method}_top_{requested_k}.json", payload)
        print(f"Wrote consensus {args.method} k={requested_k}: {len(selected)} edges")

    consensus_frame = consensus.rename("consensus_score").to_frame()
    for language in languages:
        consensus_frame[f"{language}_normalized_score"] = joined[language]
    output_root.mkdir(parents=True, exist_ok=True)
    consensus_frame.reset_index(names="name").to_parquet(output_root / f"{args.method}_consensus_edge_scores.parquet", index=False)
    write_json(
        output_root / f"{args.method}_consensus_metadata.json",
        {
            "model": args.model,
            "dataset": "PolyRefuse",
            "languages": languages,
            "method": args.method,
            "discovery_subdir": args.discovery_subdir,
            "metadata_subpath": args.metadata_subpath,
            "normalization": args.normalization,
            "target_layer": target_layer,
            "target_position": target_position,
            "selection": "average normalized signed edge scores, then paper greedy frontier",
        },
    )


if __name__ == "__main__":
    main()

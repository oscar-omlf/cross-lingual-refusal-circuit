#!/usr/bin/env python3
"""Build exploratory post-response persona circuits from frozen EAP-IG scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--k-values", type=int, nargs="+", default=(3000,))
    args = parser.parse_args()

    import pandas as pd
    from eap.graph import Graph

    from refusal_circuit.graph import edge_record, paper_greedy_edges
    from refusal_circuit.paths import write_json

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    discovery = result_root / "circuit_discovery"
    scores_path = discovery / "eapig_edge_scores.parquet"
    scores = pd.read_parquet(scores_path)
    target_layer = int(config["vector"]["hf_hidden_state_index"])
    graph = Graph.from_model({"n_layers": target_layer, "n_heads": 28, "parallel_attn_mlp": False})
    graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
    graph.nodes["logits"].layer = target_layer
    graph.nodes["logits"].in_hook = config["vector"]["transformer_lens_target_hook"]
    if set(scores["name"]) != set(graph.edges):
        raise ValueError("Score table does not exactly match the reconstructed Qwen graph.")
    score_by_name = dict(zip(scores["name"], scores["score"]))
    for name, edge in graph.edges.items():
        edge.score = float(score_by_name[name])

    output = discovery / "circuits_nested"
    output.mkdir(parents=True, exist_ok=True)
    for requested_k in args.k_values:
        selected = paper_greedy_edges(graph, requested_k, prune_parentless=False)
        payload = {
            "schema_version": 1,
            "method": "eapig",
            "selection": "paper_greedy_absolute_score_with_childless_pruning_only",
            "claim_scope": "post-generation representation diagnostic only",
            "requested_k": requested_k,
            "retained_edges": len(selected),
            "score_source": str(scores_path.relative_to(ROOT)),
            "target": {
                "hook": config["vector"]["transformer_lens_target_hook"],
                "layer": target_layer,
                "positions": "mean_response_tokens",
                "metric": "residual_projection_onto_sycophancy_direction",
                "causal_timing": "after behavior tokens are teacher-forced",
            },
            "edges": [edge_record(edge) for edge in selected],
        }
        path = output / f"eapig_top_{requested_k}.json"
        write_json(path, payload)
        print(f"Saved {len(selected)} retained edges to {path}")


if __name__ == "__main__":
    main()

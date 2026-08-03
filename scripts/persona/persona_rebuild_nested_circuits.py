#!/usr/bin/env python3
"""Derive additional nested circuits from a frozen persona edge-score table.

This is deliberately CPU-only. It reconstructs the exact graph used by a
completed upstream or downstream discovery run, loads the saved edge scores,
and applies the same backward greedy selection without rerunning EAP-IG.
Existing circuit files are accepted only when their edge names and scores
exactly match the deterministic reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--scope", choices=("upstream", "downstream"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k-values", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if any(k <= 0 for k in args.k_values):
        raise ValueError("All k values must be positive.")
    if len(args.k_values) != len(set(args.k_values)):
        raise ValueError("k values must be unique.")

    import pandas as pd
    from eap.graph import Graph

    from refusal_circuit.graph import (
        edge_record,
        make_steering_to_logits_graph,
        paper_greedy_edges,
    )
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    n_heads = 28
    n_layers = 28

    if args.scope == "upstream":
        discovery = (
            result_root / "circuit_discovery_upstream_contrastive" / args.run_name
        )
        filename = "eapig_top_{k}.json"
        graph = Graph.from_model(
            {
                "n_layers": layer,
                "n_heads": n_heads,
                "parallel_attn_mlp": False,
            }
        )
        graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
        graph.nodes["logits"].layer = layer
        graph.nodes["logits"].in_hook = target_hook
        method = "contrastive_state_path_eapig"
        claim_scope = "upstream natural persona-state formation candidate"
        target = {
            "hook": target_hook,
            "layer": layer,
            "position": -1,
            "timing": "final prompt token before response generation",
            "metric": (
                "signed_projection_onto_unit_"
                f"{config['trait']['name']}_direction"
            ),
        }
        extra = {}
    else:
        discovery = result_root / "circuit_discovery_steering_path" / args.run_name
        filename = "steering_path_eapig_top_{k}.json"
        model_stub = SimpleNamespace(
            cfg=SimpleNamespace(
                n_layers=n_layers,
                n_heads=n_heads,
                parallel_attn_mlp=False,
            )
        )
        graph = make_steering_to_logits_graph(model_stub, layer)
        method = "steering_path_eapig"
        claim_scope = (
            "downstream mediator candidate; free-generation validation required"
        )
        target = {
            "hook": graph.nodes["logits"].in_hook,
            "positions": "response_next_token_logits",
            "metric": None,
            "causal_timing": "each logit precedes its scored response token",
        }
        extra = {}

    metadata_path = discovery / "eapig_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("Discovery metadata is not complete.")
    score_path = ROOT / metadata["score_file"]
    if sha256_file(score_path) != metadata["score_sha256"]:
        raise ValueError("Frozen score table hash differs from discovery metadata.")
    scores = pd.read_parquet(score_path)
    if set(scores["name"]) != set(graph.edges):
        missing = set(graph.edges) - set(scores["name"])
        extra_scores = set(scores["name"]) - set(graph.edges)
        raise ValueError(
            "Score table does not match reconstructed graph: "
            f"{len(missing)} missing, {len(extra_scores)} unexpected."
        )
    score_by_name = dict(zip(scores["name"], scores["score"], strict=True))
    for name, edge in graph.edges.items():
        edge.score = float(score_by_name[name])

    if args.scope == "downstream":
        target["metric"] = metadata["metric"]["name"]
        extra = {
            "corrupt_coefficient": metadata["steering"]["corrupt_coefficient"],
            "source_intervention": {
                "hook": metadata["steering"]["hook"],
                "layer": metadata["steering"]["layer"],
                "coefficient": metadata["steering"]["corrupt_coefficient"],
                "vector": metadata["steering"]["vector"],
            },
        }

    circuit_root = discovery / "circuits_nested"
    circuit_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for requested_k in sorted(args.k_values):
        selected = paper_greedy_edges(
            graph,
            min(requested_k, len(graph.edges)),
            prune_parentless=False,
        )
        payload = {
            "schema_version": 1,
            "method": method,
            "selection": (
                "paper_greedy_absolute_score_with_childless_pruning_only"
            ),
            "claim_scope": claim_scope,
            "requested_k": requested_k,
            "retained_edges": len(selected),
            "score_source": str(score_path.relative_to(ROOT)),
            "target": target,
            **extra,
            "edges": [edge_record(edge) for edge in selected],
        }
        path = circuit_root / filename.format(k=requested_k)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_edges = [
                (edge["name"], float(edge["score"])) for edge in existing["edges"]
            ]
            rebuilt_edges = [
                (edge["name"], float(edge["score"])) for edge in payload["edges"]
            ]
            if existing_edges != rebuilt_edges:
                raise ValueError(f"Existing circuit differs from reconstruction: {path}")
            status = "verified_existing"
        else:
            write_json(path, payload)
            status = "created"
        manifest_rows.append(
            {
                "k": requested_k,
                "retained_edges": len(selected),
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "status": status,
            }
        )
        print(f"{status}: {path} ({len(selected)} edges)")

    manifest_path = circuit_root / f"derived_{args.scope}_circuits_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "scope": args.scope,
            "discovery_metadata": str(metadata_path.relative_to(ROOT)),
            "discovery_metadata_sha256": sha256_file(metadata_path),
            "score_file": str(score_path.relative_to(ROOT)),
            "score_sha256": sha256_file(score_path),
            "circuits": manifest_rows,
        },
    )
    print(manifest_path)


if __name__ == "__main__":
    main()

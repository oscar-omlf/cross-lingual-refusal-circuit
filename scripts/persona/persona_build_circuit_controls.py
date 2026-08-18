#!/usr/bin/env python3
"""Build topology-stratified random persona circuit controls from frozen scores."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from _bootstrap import ROOT


def _depth_bucket(
    node_type: str,
    layer: int | float | None,
    *,
    n_layers: int,
) -> str:
    if node_type == "input":
        return "input"
    if node_type in {"logit_metric", "residual_metric"}:
        return "metric"
    if layer is None or (
        isinstance(layer, float) and math.isnan(layer)
    ):
        return "unknown"
    value = int(layer)
    bucket = min(3, max(0, (4 * value) // max(1, n_layers)))
    return f"depth_{bucket}"


def _span_bucket(
    source_layer: int | float | None,
    target_layer: int | float | None,
) -> str:
    if (
        source_layer is None
        or target_layer is None
        or (
            isinstance(source_layer, float)
            and math.isnan(source_layer)
        )
        or (
            isinstance(target_layer, float)
            and math.isnan(target_layer)
        )
    ):
        return "boundary"
    span = int(target_layer) - int(source_layer)
    if span <= 0:
        return "same"
    if span == 1:
        return "next"
    if span <= 3:
        return "span_2_3"
    if span <= 7:
        return "span_4_7"
    return "span_8_plus"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument(
        "--scope",
        choices=(
            "upstream",
            "downstream",
            "end_to_end",
            "edge_pruning_upstream",
            "edge_pruning_downstream",
        ),
        required=True,
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--control-name", default="topology_stratified_v2"
    )
    parser.add_argument("--k-values", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument(
        "--family",
        choices=("source_to_sink", "causal_cut"),
        default="source_to_sink",
    )
    parser.add_argument(
        "--match-retained-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Search the random frontier budget for the retained size closest "
            "to the corresponding real circuit after connectivity pruning."
        ),
    )
    args = parser.parse_args()
    if any(k <= 0 for k in args.k_values):
        raise ValueError("All k values must be positive.")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Control seeds must be unique.")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.control_name
    ):
        raise ValueError("--control-name must be a filesystem-safe identifier.")

    import numpy as np
    import pandas as pd
    from eap.graph import Graph

    from refusal_circuit.graph import (
        circuit_connectivity,
        edge_record,
        make_projection_graph,
        make_steering_to_logits_graph,
        paper_greedy_edges,
    )
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    layer = int(config["vector"]["hf_hidden_state_index"])
    architecture = config.get("model_architecture")
    if architecture is None:
        raise ValueError(
            "Config must pin model_architecture for CPU graph reconstruction."
        )
    n_layers = int(architecture["n_layers"])
    n_heads = int(architecture["n_heads"])
    parallel_attn_mlp = bool(architecture["parallel_attn_mlp"])
    edge_pruning_stage = {
        "edge_pruning_upstream": "upstream",
        "edge_pruning_downstream": "downstream",
    }.get(args.scope)
    if edge_pruning_stage is not None:
        if args.family != "source_to_sink":
            raise ValueError(
                "Edge-pruning controls are defined only for source-to-sink "
                "circuits."
            )
        discovery = (
            result_root
            / "circuit_discovery_edge_pruning"
            / edge_pruning_stage
            / args.run_name
        )
        model_stub = SimpleNamespace(
            cfg=SimpleNamespace(
                n_layers=n_layers,
                n_heads=n_heads,
                parallel_attn_mlp=parallel_attn_mlp,
            )
        )
        graph = (
            make_projection_graph(model_stub, layer)
            if edge_pruning_stage == "upstream"
            else make_steering_to_logits_graph(model_stub, layer)
        )
        filename = "edge_pruned_top_{k}.json"
    elif args.scope == "upstream":
        discovery = (
            result_root
            / "circuit_discovery_upstream_contrastive"
            / args.run_name
        )
        graph = Graph.from_model(
            {
                "n_layers": layer,
                "n_heads": n_heads,
                "parallel_attn_mlp": parallel_attn_mlp,
            }
        )
        graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
        graph.nodes["logits"].layer = layer
        graph.nodes["logits"].in_hook = config["vector"][
            "transformer_lens_target_hook"
        ]
        filename = "eapig_top_{k}.json"
    elif args.scope == "downstream":
        discovery = (
            result_root / "circuit_discovery_steering_path" / args.run_name
        )
        model_stub = SimpleNamespace(
            cfg=SimpleNamespace(
                n_layers=n_layers,
                n_heads=n_heads,
                parallel_attn_mlp=parallel_attn_mlp,
            )
        )
        graph = make_steering_to_logits_graph(model_stub, layer)
        filename = "steering_path_eapig_top_{k}.json"
    else:
        discovery = (
            result_root / "circuit_discovery_end_to_end" / args.run_name
        )
        graph = Graph.from_model(
            {
                "n_layers": n_layers,
                "n_heads": n_heads,
                "parallel_attn_mlp": parallel_attn_mlp,
            }
        )
        graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
        filename = "end_to_end_eapig_top_{k}.json"

    metadata_path = discovery / (
        "metadata.json"
        if edge_pruning_stage is not None
        else "eapig_run_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("Discovery metadata is not complete.")
    extra = {}
    if edge_pruning_stage is not None:
        if metadata.get("stage") != edge_pruning_stage:
            raise ValueError("Edge-pruning stage and metadata disagree.")
        extra = {
            "edge_pruning_stage": edge_pruning_stage,
            "interface": metadata["objective"]["interface"],
            "position_scope": metadata["objective"]["position_scope"],
        }
    elif args.scope == "downstream":
        extra = {
            "corrupt_coefficient": metadata["steering"][
                "corrupt_coefficient"
            ],
            "source_intervention": {
                "hook": metadata["steering"]["hook"],
                "layer": metadata["steering"]["layer"],
                "coefficient": metadata["steering"][
                    "corrupt_coefficient"
                ],
                "vector": metadata["steering"]["vector"],
            },
        }
    score_path = ROOT / (
        metadata["outputs"]["scores"]
        if edge_pruning_stage is not None
        else metadata["score_file"]
    )
    expected_score_hash = (
        metadata["outputs"]["scores_sha256"]
        if edge_pruning_stage is not None
        else metadata["score_sha256"]
    )
    if sha256_file(score_path) != expected_score_hash:
        raise ValueError("Frozen score table hash changed.")
    if edge_pruning_stage is None:
        scores = pd.read_parquet(score_path).reset_index(drop=True)
    else:
        gate_scores = {}
        with score_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gate_scores[row["edge"]] = float(row["gate_score"])
        rows = []
        for edge in graph.edges.values():
            record = edge_record(edge)
            rows.append(
                {
                    "name": record["name"],
                    "source_type": record["source"]["type"],
                    "source_layer": record["source"].get("layer"),
                    "target_type": record["target"]["type"],
                    "target_layer": record["target"].get("layer"),
                    "qkv": record["qkv"],
                    "score": gate_scores.get(record["name"]),
                }
            )
        scores = pd.DataFrame(rows)
        if scores["score"].isna().any() or set(gate_scores) != set(graph.edges):
            raise ValueError(
                "Learned gate score table does not match the reconstructed graph."
            )
    if set(scores["name"]) != set(graph.edges):
        raise ValueError("Score table does not match the reconstructed graph.")

    target_root = (
        discovery / "circuits_source_to_sink"
        if args.family == "source_to_sink"
        else discovery / "circuits_cut_sets"
    )
    targets = {}
    for requested_k in sorted(set(args.k_values)):
        target_path = target_root / filename.format(k=requested_k)
        target_payload = json.loads(target_path.read_text(encoding="utf-8"))
        if (
            int(target_payload["requested_k"]) != requested_k
            or target_payload.get("score_source")
            != str(score_path.relative_to(ROOT))
        ):
            raise ValueError(f"Target circuit provenance mismatch: {target_path}")
        target_retained = int(target_payload["retained_edges"])
        if target_retained <= 0:
            raise ValueError(
                f"Cannot size-match an empty target circuit: {target_path}"
            )
        targets[requested_k] = {
            "path": target_path,
            "retained_edges": target_retained,
            "sha256": sha256_file(target_path),
        }

    scores["source_depth_bucket"] = [
        _depth_bucket(node_type, layer, n_layers=n_layers)
        for node_type, layer in zip(
            scores["source_type"], scores["source_layer"], strict=True
        )
    ]
    scores["target_depth_bucket"] = [
        _depth_bucket(node_type, layer, n_layers=n_layers)
        for node_type, layer in zip(
            scores["target_type"], scores["target_layer"], strict=True
        )
    ]
    scores["span_bucket"] = [
        _span_bucket(source, target)
        for source, target in zip(
            scores["source_layer"], scores["target_layer"], strict=True
        )
    ]
    strata = [
        "source_type",
        "target_type",
        "qkv",
        "source_depth_bucket",
        "target_depth_bucket",
        "span_bucket",
    ]
    grouped_indices = [
        np.asarray(indices, dtype=int)
        for indices in scores.groupby(
            strata, dropna=False, sort=False
        ).indices.values()
    ]
    fixed_edges = sum(len(indices) for indices in grouped_indices if len(indices) == 1)
    output_root = (
        discovery / "controls" / args.control_name / args.family
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"{output_root} already contains controls; use a new run artifact."
        )
    manifest_rows = []

    def select_control(
        *,
        requested_k: int,
        target_retained: int,
        retained_cache: dict[int, int],
    ):
        if not args.match_retained_size:
            selected = paper_greedy_edges(
                graph,
                min(requested_k, len(graph.edges)),
                prune_parentless=args.family == "source_to_sink",
            )
            return selected, min(requested_k, len(graph.edges))

        def retained_at(frontier_k: int) -> int:
            frontier_k = max(1, min(frontier_k, len(graph.edges)))
            if frontier_k not in retained_cache:
                retained_cache[frontier_k] = len(
                    paper_greedy_edges(
                        graph,
                        frontier_k,
                        prune_parentless=args.family == "source_to_sink",
                    )
                )
            return retained_cache[frontier_k]

        starting_frontier = min(requested_k, len(graph.edges))
        starting_retained = retained_at(starting_frontier)
        if starting_retained >= target_retained:
            low = 1
            high = starting_frontier
        else:
            low = starting_frontier + 1
            high = starting_frontier
            while (
                high < len(graph.edges)
                and retained_at(high) < target_retained
            ):
                low = high + 1
                high = min(len(graph.edges), high * 2)
            if retained_at(high) < target_retained:
                raise RuntimeError(
                    "Randomized graph cannot reach the target retained size."
                )
        while low < high:
            middle = (low + high) // 2
            if retained_at(middle) >= target_retained:
                high = middle
            else:
                low = middle + 1
        candidate_frontiers = {low, max(1, low - 1)}
        candidate_sizes = {
            frontier: retained_at(frontier)
            for frontier in candidate_frontiers
        }
        control_frontier = min(
            candidate_frontiers,
            key=lambda frontier: (
                abs(candidate_sizes[frontier] - target_retained),
                candidate_sizes[frontier] < target_retained,
                frontier,
            ),
        )
        selected = paper_greedy_edges(
            graph,
            control_frontier,
            prune_parentless=args.family == "source_to_sink",
        )
        return selected, control_frontier

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        permuted = scores["score"].to_numpy(dtype=float).copy()
        for indices in grouped_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        score_by_name = dict(
            zip(scores["name"], permuted, strict=True)
        )
        for name, edge in graph.edges.items():
            edge.score = float(score_by_name[name])

        retained_cache: dict[int, int] = {}
        for requested_k in sorted(set(args.k_values)):
            target = targets[requested_k]
            selected, control_frontier = select_control(
                requested_k=requested_k,
                target_retained=target["retained_edges"],
                retained_cache=retained_cache,
            )
            connectivity = circuit_connectivity(
                graph, {edge.name for edge in selected}
            )
            path = (
                output_root
                / f"seed_{seed}"
                / filename.format(k=requested_k)
            )
            write_json(
                path,
                {
                    "schema_version": 2,
                    "method": "topology_stratified_score_permutation_control",
                    "control_scope": args.scope,
                    "discovery_method": metadata.get("method"),
                    "circuit_family": args.family,
                    "selection": (
                        "paper_greedy_absolute_score_with_recursive_source_sink_pruning"
                        if args.family == "source_to_sink"
                        else "paper_greedy_absolute_score_with_sink_pruning_only"
                    ),
                    "requested_k": requested_k,
                    "control_frontier_k": control_frontier,
                    "target_retained_edges": target["retained_edges"],
                    "retained_edges": len(selected),
                    "retained_size_difference": (
                        len(selected) - target["retained_edges"]
                    ),
                    "retained_size_exact_match": (
                        len(selected) == target["retained_edges"]
                    ),
                    "seed": seed,
                    "strata": strata,
                    "strata_definition": (
                        "source/target component type, qkv role, quartile depth "
                        "buckets, and coarse edge-span bucket"
                    ),
                    "fixed_singleton_edges": fixed_edges,
                    "fixed_singleton_fraction": fixed_edges / len(scores),
                    "selection_diagnostics": graph.last_selection_diagnostics,
                    "connectivity": connectivity,
                    "score_source": str(score_path.relative_to(ROOT)),
                    "score_source_sha256": sha256_file(score_path),
                    "target_circuit": str(
                        target["path"].relative_to(ROOT)
                    ),
                    "target_circuit_sha256": target["sha256"],
                    **extra,
                    "edges": [edge_record(edge) for edge in selected],
                },
            )
            manifest_rows.append(
                {
                    "seed": seed,
                    "requested_k": requested_k,
                    "control_frontier_k": control_frontier,
                    "target_retained_edges": target["retained_edges"],
                    "retained_edges": len(selected),
                    "retained_size_difference": (
                        len(selected) - target["retained_edges"]
                    ),
                    "retained_size_exact_match": (
                        len(selected) == target["retained_edges"]
                    ),
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "connectivity": connectivity,
                }
            )

    manifest_path = output_root / "manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 2,
            "status": "complete",
            "scope": args.scope,
            "discovery_method": metadata.get("method"),
            "model_architecture": architecture,
            "control_name": args.control_name,
            "family": args.family,
            "control": "topology-stratified signed-score permutation",
            "match_retained_size": args.match_retained_size,
            "strata": strata,
            "strata_definition": (
                "source/target component type, qkv role, quartile depth "
                "buckets, and coarse edge-span bucket"
            ),
            "fixed_singleton_edges": fixed_edges,
            "fixed_singleton_fraction": fixed_edges / len(scores),
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

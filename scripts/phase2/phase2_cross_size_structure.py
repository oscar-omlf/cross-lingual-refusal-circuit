#!/usr/bin/env python3
"""Measure multilingual circuit overlap across both language and circuit size.

This is CPU-only and reads serialized masks.  It explicitly tests the
"different small subsets of one larger shared circuit" hypothesis with
cross-size coverage, rather than relying only on same-k IoU.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path

from _bootstrap import ROOT


DEFAULT_CONFIG = ROOT / "configs" / "phase2_qwen_refusal.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _edge_map(payload: dict) -> dict[str, dict]:
    return {str(edge["name"]): edge for edge in payload["edges"]}


def _nodes(edges: dict[str, dict], *, physical_gqa: bool, group_size: int) -> set[str]:
    output: set[str] = set()
    for edge in edges.values():
        for side in ("source", "target"):
            node = edge[side]
            if node["type"] in {"residual_metric", "logit_metric"}:
                continue
            name = str(node["name"])
            if physical_gqa and side == "target" and edge.get("qkv") in {"k", "v"}:
                match = re.fullmatch(r"a(\d+)\.h(\d+)", name)
                if match:
                    layer, head = map(int, match.groups())
                    name = f"a{layer}.kv{head // group_size}"
            output.add(name)
    return output


def _overlap(a: set[str], b: set[str], universe: int) -> dict:
    intersection = len(a & b)
    union = len(a | b)
    expected = len(a) * len(b) / universe
    expected_iou = expected / max(len(a) + len(b) - expected, 1e-12)
    maximum = min(len(a), len(b))
    adjusted = (
        (intersection - expected) / (maximum - expected)
        if maximum > expected
        else float("nan")
    )
    variance = (
        len(a)
        * len(b)
        * (universe - len(a))
        * (universe - len(b))
        / (universe * universe * max(universe - 1, 1))
    )
    return {
        "left_size": len(a),
        "right_size": len(b),
        "intersection": intersection,
        "union": union,
        "iou": intersection / max(union, 1),
        "left_covered_by_right": intersection / max(len(a), 1),
        "right_covered_by_left": intersection / max(len(b), 1),
        "overlap_coefficient": intersection / max(min(len(a), len(b)), 1),
        "random_expected_intersection": expected,
        "random_expected_iou_approx": expected_iou,
        "chance_adjusted_overlap": adjusted,
        "hypergeometric_z": (intersection - expected) / math.sqrt(variance) if variance > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = _read(args.config)
    results_root = _resolve(cfg["artifacts"]["results_root"])
    discovery = cfg["artifacts"]["local_discovery_subdir"]
    circuits_subdir = cfg["artifacts"]["circuits_subdir"]
    languages = cfg["dataset"]["languages"]
    k_values = cfg["circuit_sweep"]["requested_k"]
    edge_universe = int(cfg["metric"]["eligible_edges"])
    group_size = cfg["model"]["n_attention_heads"] // cfg["model"]["n_key_value_heads"]
    output_dir = args.output_dir or _resolve(cfg["artifacts"]["phase2_results_root"]) / "structure"

    payloads: dict[str, dict[int, dict]] = {}
    for language in languages:
        payloads[language] = {}
        for k in k_values:
            path = results_root / language / discovery / circuits_subdir / f"eapig_top_{k}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = _read(path)
            if int(payload["requested_k"]) != int(k):
                raise ValueError(f"Wrong requested_k in {path}")
            payloads[language][k] = payload

    # Greedy runs at successive budgets must form prefixes. If this fails, a
    # size curve is not a nested ablation experiment and must not be analyzed
    # as one.
    nesting_rows = []
    for language in languages:
        previous: set[str] = set()
        for k in k_values:
            current = set(_edge_map(payloads[language][k]))
            nesting_rows.append(
                {
                    "language": language,
                    "requested_k": k,
                    "retained_edges": len(current),
                    "edge_fraction": len(current) / edge_universe,
                    "contains_previous": previous <= current,
                    "new_edges_since_previous": len(current - previous),
                }
            )
            if not previous <= current:
                raise ValueError(f"{language} masks are not nested at requested_k={k}")
            previous = current

    cross_rows: list[dict] = []
    same_size_rows: list[dict] = []
    for left, right in itertools.combinations(languages, 2):
        for left_k in k_values:
            left_edges = set(_edge_map(payloads[left][left_k]))
            for right_k in k_values:
                right_edges = set(_edge_map(payloads[right][right_k]))
                row = {
                    "left_language": left,
                    "right_language": right,
                    "left_requested_k": left_k,
                    "right_requested_k": right_k,
                    "left_edge_fraction": len(left_edges) / edge_universe,
                    "right_edge_fraction": len(right_edges) / edge_universe,
                }
                row.update(_overlap(left_edges, right_edges, edge_universe))
                cross_rows.append(row)
                if left_k == right_k:
                    same_size_rows.append(dict(row))

    component_rows: list[dict] = []
    pair_component_rows: list[dict] = []
    for language in languages:
        for k in k_values:
            edges = _edge_map(payloads[language][k])
            terminal = {
                name
                for name, edge in edges.items()
                if edge["target"]["type"] in {"residual_metric", "logit_metric"}
            }
            upstream = set(edges) - terminal
            logical_nodes = _nodes(edges, physical_gqa=False, group_size=group_size)
            physical_nodes = _nodes(edges, physical_gqa=True, group_size=group_size)
            component_rows.append(
                {
                    "language": language,
                    "requested_k": k,
                    "retained_edges": len(edges),
                    "edge_fraction": len(edges) / edge_universe,
                    "terminal_edges": len(terminal),
                    "upstream_edges": len(upstream),
                    "logical_nodes": len(logical_nodes),
                    "physical_gqa_nodes": len(physical_nodes),
                }
            )
    for left, right in itertools.combinations(languages, 2):
        for k in k_values:
            left_map = _edge_map(payloads[left][k])
            right_map = _edge_map(payloads[right][k])
            for subset in ("all", "upstream", "terminal", "logical_nodes", "physical_gqa_nodes"):
                if subset == "all":
                    a, b, universe = set(left_map), set(right_map), edge_universe
                elif subset in {"upstream", "terminal"}:
                    want_terminal = subset == "terminal"
                    a = {
                        name
                        for name, edge in left_map.items()
                        if (edge["target"]["type"] in {"residual_metric", "logit_metric"}) == want_terminal
                    }
                    b = {
                        name
                        for name, edge in right_map.items()
                        if (edge["target"]["type"] in {"residual_metric", "logit_metric"}) == want_terminal
                    }
                    universe = int(
                        cfg["metric"][
                            "eligible_terminal_edges"
                            if subset == "terminal"
                            else "eligible_upstream_edges"
                        ]
                    )
                else:
                    physical = subset == "physical_gqa_nodes"
                    a = _nodes(left_map, physical_gqa=physical, group_size=group_size)
                    b = _nodes(right_map, physical_gqa=physical, group_size=group_size)
                    universe = max(len(a | b), 1)
                row = {
                    "left_language": left,
                    "right_language": right,
                    "requested_k": k,
                    "subset": subset,
                }
                row.update(_overlap(a, b, universe))
                if subset in {"logical_nodes", "physical_gqa_nodes"}:
                    row["random_expected_intersection"] = ""
                    row["random_expected_iou_approx"] = ""
                    row["chance_adjusted_overlap"] = ""
                    row["hypergeometric_z"] = ""
                pair_component_rows.append(row)

    _write_csv(output_dir / "mask_nesting_and_sizes.csv", nesting_rows)
    _write_csv(output_dir / "edge_overlap_cross_size.csv", cross_rows)
    _write_csv(output_dir / "edge_overlap_same_size.csv", same_size_rows)
    _write_csv(output_dir / "circuit_composition.csv", component_rows)
    _write_csv(output_dir / "pairwise_component_overlap_same_size.csv", pair_component_rows)
    metadata = {
        "model": cfg["model"]["name"],
        "metric": cfg["metric"]["definition"],
        "languages": languages,
        "requested_k": k_values,
        "eligible_edge_universe": edge_universe,
        "gqa_group_size": group_size,
        "definitions": {
            "left_covered_by_right": "|C_left intersect C_right| / |C_left|",
            "chance_adjusted_overlap": "(observed intersection - ab/N) / (min(a,b) - ab/N)",
            "cross_size_test": "coverage for every k_left by every k_right; directly tests subset-of-larger-core hypothesis",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote Phase 2 cross-size structural tables under {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build uniform and topology-stratified random masks matched to a reference circuit."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _bootstrap import ROOT


def _stratum(row) -> tuple:
    return (
        str(row["source_type"]),
        str(row["target_type"]),
        int(row["target_layer"]) if row["target_layer"] == row["target_layer"] else -1,
        str(row["qkv"]) if row["qkv"] == row["qkv"] else "",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-circuit", type=Path, required=True)
    parser.add_argument("--score-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("uniform", "stratified", "stratified_min_overlap"),
        default=("stratified",),
    )
    args = parser.parse_args()

    import pandas as pd

    reference = json.loads(args.reference_circuit.read_text(encoding="utf-8"))
    selected_names = {str(edge["name"]) for edge in reference["edges"]}
    universe = pd.read_parquet(args.score_path)
    if set(selected_names) - set(universe["name"]):
        raise ValueError("Reference circuit contains edges absent from the score universe")
    universe_records = universe.set_index("name").to_dict("index")
    all_names = sorted(universe_records)
    selected_by_stratum: dict[tuple, list[str]] = {}
    universe_by_stratum: dict[tuple, list[str]] = {}
    for name, row in universe_records.items():
        universe_by_stratum.setdefault(_stratum(row), []).append(name)
        if name in selected_names:
            selected_by_stratum.setdefault(_stratum(row), []).append(name)

    for strategy in args.strategies:
        for seed in args.seeds:
            rng = random.Random(seed)
            if strategy == "uniform":
                names = rng.sample(all_names, len(selected_names))
            else:
                names = []
                for stratum in sorted(selected_by_stratum, key=str):
                    count = len(selected_by_stratum[stratum])
                    pool = sorted(universe_by_stratum[stratum])
                    if count > len(pool):
                        raise ValueError(f"Stratum {stratum} needs {count}, has {len(pool)}")
                    if strategy == "stratified":
                        names.extend(rng.sample(pool, count))
                    else:
                        reference_names = set(selected_by_stratum[stratum])
                        non_reference = sorted(set(pool) - reference_names)
                        take_non_reference = min(count, len(non_reference))
                        names.extend(rng.sample(non_reference, take_non_reference))
                        remaining = count - take_non_reference
                        if remaining:
                            names.extend(rng.sample(sorted(reference_names), remaining))
            output_dir = args.output_root / strategy / f"seed_{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            requested_k = int(reference["requested_k"])
            payload = {
                "method": "random_control",
                "selection": f"{strategy}_matched_to_reference_retained_edges",
                "requested_k": requested_k,
                "retained_edges": len(names),
                "random_seed": seed,
                "overlap_with_reference_edges": len(set(names) & selected_names),
                "overlap_fraction_of_reference": len(set(names) & selected_names)
                / max(len(selected_names), 1),
                "reference_circuit": str(args.reference_circuit),
                "score_universe": str(args.score_path),
                "stratification": (
                    ["source_type", "target_type", "target_layer", "qkv"]
                    if strategy in {"stratified", "stratified_min_overlap"}
                    else []
                ),
                "overlap_policy": (
                    "minimize_reference_overlap_within_each_stratum"
                    if strategy == "stratified_min_overlap"
                    else "unconstrained"
                ),
                "target": reference["target"],
                "edges": [{"name": name} for name in sorted(names)],
            }
            path = output_dir / f"eapig_top_{requested_k}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Wrote {strategy} seed={seed} control to {path}")


if __name__ == "__main__":
    main()

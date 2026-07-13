#!/usr/bin/env python3
"""Structural comparison of independently discovered multilingual circuits.

IoU is reported at requested greedy budgets and accompanied by the overlap
coefficient (which is less distorted when recursive pruning retains different
numbers of edges) and full-graph EAP-IG score-rank correlation. Structural
overlap alone is never treated as evidence of functional transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--method", choices=("eapig", "eap"), default="eapig")
    parser.add_argument("--k-values", type=int, nargs="+", default=(50, 100, 200, 500, 1000, 2000))
    parser.add_argument("--circuits-subdir", default="circuits")
    parser.add_argument("--discovery-subdir", default="circuit_discovery")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    return parser.parse_args()


def _load_circuit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method") not in {"eapig", "eap"}:
        raise ValueError(f"Not an EAP circuit JSON: {path}")
    return payload


def _stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def main():
    args = parse_args()
    import pandas as pd

    from refusal_circuit.paths import experiment_results_dir, write_json

    languages = list(dict.fromkeys(args.languages))
    results_root = args.results_root or experiment_results_dir(args.model, "polyrefuse", "en").parent
    output_dir = args.output_dir or results_root / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_rows: list[dict] = []
    score_frames = {}
    for language in languages:
        discovery_dir = results_root / language / args.discovery_subdir
        score_path = discovery_dir / f"{args.method}_edge_scores.parquet"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing edge-score file: {score_path}")
        score_frames[language] = pd.read_parquet(score_path).set_index("name")["score"].sort_index()
    for requested_k in args.k_values:
        circuits = {}
        terminal_layers = set()
        for language in languages:
            path = (
                results_root
                / language
                / args.discovery_subdir
                / args.circuits_subdir
                / f"{args.method}_top_{requested_k}.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing circuit: {path}")
            payload = _load_circuit(path)
            circuits[language] = {edge["name"] for edge in payload["edges"]}
            terminal_layers.add(int(payload["target"]["layer"]))
        if len(terminal_layers) != 1:
            raise ValueError("Circuits have different terminal layers; compare only common-site circuits.")
        universe = set.intersection(*(set(frame.index) for frame in score_frames.values()))
        for left, right in itertools.combinations(languages, 2):
            a, b = circuits[left], circuits[right]
            intersection = len(a & b)
            union = len(a | b)
            joined_scores = pd.concat(
                {
                    left: score_frames[left].abs(),
                    right: score_frames[right].abs(),
                },
                axis=1,
                join="inner",
            ).fillna(0.0)
            numerator = joined_scores.min(axis=1).sum()
            denominator = joined_scores.max(axis=1).sum()
            overlap_rows.append(
                {
                    "method": args.method,
                    "requested_k": requested_k,
                    "left_language": left,
                    "right_language": right,
                    "left_retained_edges": len(a),
                    "right_retained_edges": len(b),
                    "intersection_edges": intersection,
                    "union_edges": union,
                    "iou": intersection / max(union, 1),
                    "overlap_coefficient": intersection / max(min(len(a), len(b)), 1),
                    "weighted_jaccard_abs_scores": numerator / denominator if denominator else float("nan"),
                    "baseline": "observed",
                }
            )
            for seed in args.random_seeds:
                rng = random.Random(_stable_seed(args.method, requested_k, left, right, seed))
                universe_list = sorted(universe)
                random_a = set(rng.sample(universe_list, k=min(len(a), len(universe_list))))
                random_b = set(rng.sample(universe_list, k=min(len(b), len(universe_list))))
                random_intersection = len(random_a & random_b)
                random_union = len(random_a | random_b)
                overlap_rows.append(
                    {
                        "method": args.method,
                        "requested_k": requested_k,
                        "left_language": left,
                        "right_language": right,
                        "left_retained_edges": len(random_a),
                        "right_retained_edges": len(random_b),
                        "intersection_edges": random_intersection,
                        "union_edges": random_union,
                        "iou": random_intersection / max(random_union, 1),
                        "overlap_coefficient": random_intersection / max(min(len(random_a), len(random_b)), 1),
                        "weighted_jaccard_abs_scores": float("nan"),
                        "baseline": f"random_seed_{seed}",
                    }
                )
    score_rows = []
    for left, right in itertools.combinations(languages, 2):
        joined = pd.concat({left: score_frames[left], right: score_frames[right]}, axis=1, join="inner")
        # Avoid requiring SciPy on the cluster just for Spearman correlation:
        # Spearman is Pearson correlation of ranks.
        rank_joined = joined.rank(method="average")
        score_rows.append(
            {
                "method": args.method,
                "left_language": left,
                "right_language": right,
                "n_common_edges": len(joined),
                "spearman_score_correlation": rank_joined[left].corr(rank_joined[right], method="pearson"),
                "pearson_score_correlation": joined[left].corr(joined[right], method="pearson"),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(output_dir / f"{args.method}_circuit_structural_overlap.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output_dir / f"{args.method}_edge_score_correlations.csv", index=False)
    write_json(
        output_dir / f"{args.method}_circuit_comparison_metadata.json",
        {
            "model": args.model,
            "dataset": "PolyRefuse",
            "method": args.method,
            "discovery_subdir": args.discovery_subdir,
            "circuits_subdir": args.circuits_subdir,
            "languages": languages,
            "definition": {
                "iou": "|C_i intersect C_j| / |C_i union C_j|",
                "overlap_coefficient": "|C_i intersect C_j| / min(|C_i|, |C_j|)",
                "weighted_jaccard_abs_scores": "sum_e min(|s_i(e)|, |s_j(e)|) / sum_e max(|s_i(e)|, |s_j(e)|)",
                "random_baseline": "uniform random edge sets matched to each pair's retained edge counts",
                "score_correlation": "correlation across all common graph edges before greedy pruning",
            },
            "caveat": "Use cross-language faithfulness/necessity to establish functional transfer.",
        },
    )
    print(f"Compared {len(languages)} {args.method} circuits; wrote structural tables under {output_dir}")


if __name__ == "__main__":
    main()

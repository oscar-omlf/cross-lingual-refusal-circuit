#!/usr/bin/env python3
"""Build shared-core circuit JSONs from multilingual circuit intersections.

For each requested k, this script takes independently discovered circuits and
saves strict intersections and optional majority-vote cores in the same JSON
format consumed by ``rq1_evaluate_faithfulness.py``.  It is intentionally
structural: functional sufficiency/necessity must be evaluated separately on
each language.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--method", choices=("eapig", "eap"), default="eapig")
    parser.add_argument("--k-values", type=int, nargs="+", default=(25, 50, 100, 200, 500, 1000, 2000))
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--discovery-subdir", default="circuit_discovery")
    parser.add_argument("--circuits-subdir", default="circuits_greedy_no_parent_prune_small_to_large")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--include-pairs",
        action="store_true",
        help="Also write every pairwise intersection. The all-language intersection is always written.",
    )
    parser.add_argument(
        "--majority-threshold",
        type=int,
        default=None,
        help="Also write a majority core containing edges present in at least this many languages.",
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "edges" not in payload:
        raise ValueError(f"Circuit JSON has no edges: {path}")
    return payload


def _write_intersection(
    *,
    output_dir: Path,
    method: str,
    requested_k: int,
    languages: tuple[str, ...],
    payloads: dict[str, dict],
    source_paths: dict[str, Path],
) -> None:
    reference_language = languages[0]
    edge_maps = {
        language: {edge["name"]: edge for edge in payloads[language]["edges"]}
        for language in languages
    }
    shared_names = set(edge_maps[reference_language])
    for language in languages[1:]:
        shared_names &= set(edge_maps[language])
    # Preserve the source circuit's score ordering where possible.
    reference_order = [edge["name"] for edge in payloads[reference_language]["edges"]]
    shared_edges = [edge_maps[reference_language][name] for name in reference_order if name in shared_names]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "selection": f"structural_intersection_of_{'_'.join(languages)}",
        "requested_k": requested_k,
        "retained_edges": len(shared_edges),
        "source_languages": list(languages),
        "source_circuit_files": {language: str(source_paths[language]) for language in languages},
        "target": payloads[reference_language]["target"],
        "edges": shared_edges,
    }
    path = output_dir / f"{method}_top_{requested_k}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path} with {len(shared_edges)} shared edges")


def _write_majority(
    *,
    output_dir: Path,
    method: str,
    requested_k: int,
    languages: tuple[str, ...],
    payloads: dict[str, dict],
    source_paths: dict[str, Path],
    threshold: int,
) -> set[str]:
    edge_maps = {
        language: {edge["name"]: edge for edge in payloads[language]["edges"]}
        for language in languages
    }
    counts: dict[str, int] = {}
    for edge_map in edge_maps.values():
        for name in edge_map:
            counts[name] = counts.get(name, 0) + 1
    selected_names = {name for name, count in counts.items() if count >= threshold}
    reference_language = languages[0]
    reference_order = [edge["name"] for edge in payloads[reference_language]["edges"]]
    ordered_names = [name for name in reference_order if name in selected_names]
    ordered_names.extend(sorted(selected_names - set(ordered_names)))
    shared_edges = []
    for name in ordered_names:
        for language in languages:
            if name in edge_maps[language]:
                record = dict(edge_maps[language][name])
                record["language_support"] = counts[name]
                shared_edges.append(record)
                break
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "selection": f"structural_majority_core_at_least_{threshold}_of_{len(languages)}",
        "requested_k": requested_k,
        "retained_edges": len(shared_edges),
        "source_languages": list(languages),
        "majority_threshold": threshold,
        "source_circuit_files": {language: str(source_paths[language]) for language in languages},
        "target": payloads[reference_language]["target"],
        "edges": shared_edges,
    }
    path = output_dir / f"{method}_top_{requested_k}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path} with {len(shared_edges)} majority-core edges")
    return selected_names


def main():
    args = parse_args()
    import pandas as pd

    from refusal_circuit.paths import experiment_results_dir, write_json

    languages = tuple(dict.fromkeys(args.languages))
    majority_threshold = args.majority_threshold
    if majority_threshold is not None and not 1 <= majority_threshold <= len(languages):
        raise ValueError("--majority-threshold must be between 1 and the number of languages.")
    results_root = args.results_root or experiment_results_dir(args.model, "polyrefuse", "en").parent
    output_root = args.output_root or results_root / "shared_circuits" / args.discovery_subdir
    rows = []
    combinations: list[tuple[str, ...]] = [languages]
    if args.include_pairs:
        combinations.extend(tuple(pair) for pair in itertools.combinations(languages, 2))

    for requested_k in args.k_values:
        payloads: dict[str, dict] = {}
        source_paths: dict[str, Path] = {}
        for language in languages:
            path = (
                results_root
                / language
                / args.discovery_subdir
                / args.circuits_subdir
                / f"{args.method}_top_{requested_k}.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing circuit JSON: {path}")
            payloads[language] = _load(path)
            source_paths[language] = path
        for combo in combinations:
            label = "_".join(combo)
            output_dir = output_root / f"intersection_{label}" / args.circuits_subdir
            _write_intersection(
                output_dir=output_dir,
                method=args.method,
                requested_k=requested_k,
                languages=combo,
                payloads=payloads,
                source_paths=source_paths,
            )
            edge_sets = {
                language: {edge["name"] for edge in payloads[language]["edges"]}
                for language in combo
            }
            shared = set.intersection(*edge_sets.values()) if edge_sets else set()
            union = set.union(*edge_sets.values()) if edge_sets else set()
            rows.append(
                {
                    "method": args.method,
                    "requested_k": requested_k,
                    "languages": label,
                    "n_languages": len(combo),
                    "intersection_edges": len(shared),
                    "union_edges": len(union),
                    "intersection_over_min": len(shared) / max(min(len(values) for values in edge_sets.values()), 1),
                    "intersection_over_union": len(shared) / max(len(union), 1),
                    "core_type": "strict_intersection",
                }
            )
        if majority_threshold is not None:
            output_dir = output_root / f"majority_at_least_{majority_threshold}_of_{len(languages)}" / args.circuits_subdir
            selected = _write_majority(
                output_dir=output_dir,
                method=args.method,
                requested_k=requested_k,
                languages=languages,
                payloads=payloads,
                source_paths=source_paths,
                threshold=majority_threshold,
            )
            union = set.union(
                *({edge["name"] for edge in payloads[language]["edges"]} for language in languages)
            )
            rows.append(
                {
                    "method": args.method,
                    "requested_k": requested_k,
                    "languages": "_".join(languages),
                    "n_languages": len(languages),
                    "intersection_edges": len(selected),
                    "union_edges": len(union),
                    "intersection_over_min": len(selected)
                    / max(
                        min(len(payloads[language]["edges"]) for language in languages),
                        1,
                    ),
                    "intersection_over_union": len(selected) / max(len(union), 1),
                    "core_type": f"majority_at_least_{majority_threshold}",
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_root / f"{args.method}_intersection_summary.csv", index=False)
    write_json(
        output_root / f"{args.method}_intersection_metadata.json",
        {
            "model": args.model,
            "dataset": "PolyRefuse",
            "languages": list(languages),
            "discovery_subdir": args.discovery_subdir,
            "circuits_subdir": args.circuits_subdir,
            "definition": "edge-name intersection of independently discovered circuit JSONs",
            "majority_threshold": majority_threshold,
            "functional_evaluation_required": True,
        },
    )


if __name__ == "__main__":
    main()

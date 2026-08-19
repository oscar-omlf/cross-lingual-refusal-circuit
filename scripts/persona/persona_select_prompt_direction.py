#!/usr/bin/env python3
"""Select a prompt-derived persona direction from judged frozen-grid outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values:
        raise ValueError(f"No values for {key}")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--scored", type=Path, default=None)
    args = parser.parse_args()

    import torch

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        read_jsonl,
        select_prompt_direction_candidate,
        sha256_file,
    )

    search_root = _resolve(args.search_root)
    responses_path = search_root / "responses.jsonl"
    scored_path = _resolve(args.scored) if args.scored else search_root / "scored.jsonl"
    diagnostics_path = search_root / "candidate_diagnostics.jsonl"
    vectors_path = search_root / "candidate_vectors.pt"
    metadata_path = search_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["artifacts"]["responses"]["sha256"] != sha256_file(responses_path):
        raise ValueError("Response hash does not match search metadata.")
    scored = read_jsonl(scored_path)
    if len(scored) != metadata["artifacts"]["responses"]["rows"]:
        raise ValueError("Judged row count does not match generated responses.")
    if any(row.get("trait_score") is None or row.get("coherence_score") is None for row in scored):
        raise ValueError("Judged rows contain missing scores.")
    diagnostics = {
        row["candidate_id"]: row for row in read_jsonl(diagnostics_path)
    }
    if len(diagnostics) != 10:
        raise ValueError("Expected exactly ten frozen candidate diagnostics.")

    positive_rows = [
        row for row in scored if row["condition"] == "baseline_positive"
    ]
    negative_rows = [
        row for row in scored if row["condition"] == "baseline_negative"
    ]
    positive_mean = _mean(positive_rows, "trait_score")
    negative_mean = _mean(negative_rows, "trait_score")
    gap = positive_mean - negative_mean
    if gap <= 20.0:
        raise RuntimeError(
            f"Natural q0-q4 persona gap is too small for selection: {gap:.3f}"
        )

    summaries = []
    for candidate_id, diagnostic in sorted(diagnostics.items()):
        induction = [
            row
            for row in scored
            if row["condition"] == "negative_plus_raw_direction"
            and row["candidate_id"] == candidate_id
        ]
        ablation = [
            row
            for row in scored
            if row["condition"] == "positive_global_direction_ablation"
            and row["candidate_id"] == candidate_id
        ]
        if len(induction) != len(positive_rows) or len(ablation) != len(positive_rows):
            raise ValueError(f"Incomplete behavior rows for {candidate_id}")
        induction_mean = _mean(induction, "trait_score")
        ablation_mean = _mean(ablation, "trait_score")
        induction_effect = (induction_mean - negative_mean) / gap
        ablation_effect = (positive_mean - ablation_mean) / gap
        summaries.append(
            {
                **diagnostic,
                "positive_baseline_mean": positive_mean,
                "negative_baseline_mean": negative_mean,
                "natural_gap": gap,
                "induction_trait_mean": induction_mean,
                "ablation_trait_mean": ablation_mean,
                "induction_normalized_effect": induction_effect,
                "ablation_normalized_effect": ablation_effect,
                "balanced_normalized_effect": min(
                    induction_effect,
                    ablation_effect,
                ),
                "induction_coherence_mean": _mean(
                    induction,
                    "coherence_score",
                ),
                "ablation_coherence_mean": _mean(
                    ablation,
                    "coherence_score",
                ),
                "cap_hits": sum(
                    bool(row["hit_max_new_tokens"])
                    for row in induction + ablation
                ),
            }
        )

    selected = select_prompt_direction_candidate(summaries)
    payload = torch.load(vectors_path, map_location="cpu", weights_only=True)
    selected_vector = payload["vectors"][selected["candidate_id"]]
    selected_path = search_root / "selected_direction.pt"
    torch.save(selected_vector, selected_path)

    table_path = search_root / "candidate_summary.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        "schema_version": 1,
        "status": "complete",
        "selection_is_exploratory": True,
        "selection_rule": metadata["selection_rule"],
        "selected": selected,
        "inputs": {
            "responses_sha256": sha256_file(responses_path),
            "scored": str(scored_path.relative_to(ROOT)),
            "scored_sha256": sha256_file(scored_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "vectors_sha256": sha256_file(vectors_path),
        },
        "outputs": {
            "selected_direction": str(selected_path.relative_to(ROOT)),
            "selected_direction_sha256": sha256_file(selected_path),
            "candidate_summary": str(table_path.relative_to(ROOT)),
            "candidate_summary_sha256": sha256_file(table_path),
        },
    }
    write_json(search_root / "selection.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

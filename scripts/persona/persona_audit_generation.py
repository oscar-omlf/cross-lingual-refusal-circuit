#!/usr/bin/env python3
"""Validate the completed S1 generation and freeze a compact results manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    args = parser.parse_args()

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = _resolve(config["paths"]["intermediate_root"]) / "generation"
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = (
        int(config["extraction"]["questions"])
        * int(config["extraction"]["instructions_per_polarity"])
        * int(config["extraction"]["rollouts_per_question_instruction"])
    )
    pair_sets = {}
    summaries = []
    for polarity in ("pos", "neg"):
        path = root / f"{polarity}_responses.jsonl"
        rows = read_jsonl(path)
        if len(rows) != expected:
            raise RuntimeError(f"{polarity} has {len(rows)}/{expected} rows")
        if len({row["record_id"] for row in rows}) != expected:
            raise RuntimeError(f"{polarity} record IDs are not unique")
        pair_sets[polarity] = {row["pair_id"] for row in rows}
        if len(pair_sets[polarity]) != expected:
            raise RuntimeError(f"{polarity} pair IDs are not unique")
        if any(row["polarity"] != polarity for row in rows):
            raise RuntimeError(f"wrong polarity marker in {polarity} output")
        if any(row.get("generation_protocol_id") != config["protocol_id"] for row in rows):
            raise RuntimeError(
                f"{polarity} output predates the frozen exact protocol; regenerate before judging"
            )
        if any(not row["answer"].strip() for row in rows):
            raise RuntimeError(f"empty answer in {polarity} output")
        expected_hash = next(
            item["sha256"] for item in metadata["outputs"] if item["polarity"] == polarity
        )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"{polarity} output hash changed after generation")
        token_counts = [int(row["generated_token_count"]) for row in rows]
        summaries.append(
            {
                "polarity": polarity,
                "rows": len(rows),
                "sha256": actual_hash,
                "mean_generated_tokens": sum(token_counts) / len(token_counts),
                "max_generated_tokens": max(token_counts),
                "empty_answers": 0,
                "hit_max_new_tokens": sum(bool(row["hit_max_new_tokens"]) for row in rows),
            }
        )
    if pair_sets["pos"] != pair_sets["neg"]:
        raise RuntimeError("positive and negative pair-ID sets differ")

    report = {
        "schema_version": 1,
        "status": "pass",
        "stage": "S1_generation",
        "canonical_protocol": True,
        "generation_protocol_id": config["protocol_id"],
        "trait": config["trait"]["name"],
        "model": config["model"],
        "sampling": metadata["generation"],
        "official_artifact_sha256": metadata["official_artifact_sha256"],
        "generation_metadata_sha256": sha256_file(metadata_path),
        "paired_rows": expected,
        "outputs": summaries,
        "gpu_wall_time_seconds": metadata["wall_time_seconds"],
        "next_required_gate": "canonical GPT-4.1-mini trait/coherence judging",
    }
    output = _resolve(config["paths"]["results_root"]) / "s1_generation_manifest.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

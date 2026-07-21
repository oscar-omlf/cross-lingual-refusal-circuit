#!/usr/bin/env python3
"""Freeze a reviewed development k and all inputs before confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ROOT


DEFAULT_CONFIG = ROOT / "configs" / "phase2_qwen_refusal.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--review-record", type=Path, required=True)
    parser.add_argument("--circuit-label", default="consensus")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = _read(args.config)
    selection = _read(args.selection)
    review = _read(args.review_record)
    if selection.get("split") != "development":
        raise ValueError("Only development selection can be frozen for confirmation")
    selected_k = args.k or selection.get("common_k_if_all_languages_pass", {}).get(args.circuit_label)
    if not selected_k:
        raise ValueError(f"No common selected k for {args.circuit_label}; pass --k only after review")
    if review.get("selected_circuit_label") != args.circuit_label:
        raise ValueError("Review record circuit label does not match")
    if int(review.get("selected_k") or -1) != int(selected_k):
        raise ValueError("Review record selected_k does not match")
    if not review.get("benign_controls_pass") or not review.get("random_controls_pass"):
        raise ValueError("Benign and random control reviews must both pass")
    audits = review.get("judge_refusal_audit_pass_by_language", {})
    failed = [language for language in cfg["dataset"]["languages"] if not audits.get(language)]
    if failed:
        raise ValueError(f"Judge refusal audit did not pass for: {failed}")
    if not str(review.get("reviewer", "")).strip() or not str(review.get("review_date", "")).strip():
        raise ValueError("Review record requires reviewer and review_date")

    results_root = _resolve(cfg["artifacts"]["results_root"])
    discovery = cfg["artifacts"]["local_discovery_subdir"]
    circuits_subdir = cfg["artifacts"]["circuits_subdir"]
    circuit_hashes = {}
    for language in cfg["dataset"]["languages"]:
        path = results_root / language / discovery / circuits_subdir / f"eapig_top_{selected_k}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        circuit_hashes[f"source_{language}"] = {"path": str(path), "sha256": _sha256(path)}
    consensus_path = (
        results_root
        / "shared_circuits"
        / discovery
        / "consensus_mean_scores"
        / circuits_subdir
        / f"eapig_top_{selected_k}.json"
    )
    if not consensus_path.exists():
        raise FileNotFoundError(consensus_path)
    circuit_hashes["consensus"] = {"path": str(consensus_path), "sha256": _sha256(consensus_path)}
    data_manifest = _resolve(cfg["dataset"]["phase2_data_root"]) / "manifest.json"
    if not data_manifest.exists():
        raise FileNotFoundError(data_manifest)
    payload = {
        "schema_version": 1,
        "status": "frozen_for_confirmation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_circuit_label": args.circuit_label,
        "selected_k": int(selected_k),
        "model": cfg["model"]["name"],
        "metric": cfg["metric"],
        "judge_model": cfg["behavior"]["judge_model"],
        "generation_protocol": cfg["behavior"],
        "config": {"path": str(args.config), "sha256": _sha256(args.config)},
        "development_selection": {"path": str(args.selection), "sha256": _sha256(args.selection)},
        "control_review": {"path": str(args.review_record), "sha256": _sha256(args.review_record)},
        "phase2_data_manifest": {"path": str(data_manifest), "sha256": _sha256(data_manifest)},
        "circuit_masks": circuit_hashes,
        "confirmation_policy": "evaluate these hashes once; do not reselect k on confirmation",
    }
    output = args.output or _resolve(cfg["artifacts"]["phase2_results_root"]) / "frozen_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Frozen selected k={selected_k} and confirmation inputs in {output}")


if __name__ == "__main__":
    main()

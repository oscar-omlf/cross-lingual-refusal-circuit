#!/usr/bin/env python3
"""Audit and prepare the frozen Qwen refusal Phase 2 experiment.

The ``audit`` command is read-only. ``prepare-data`` makes new semantic-ID
splits without loading a model. ``build-masks`` derives every local, shared,
and consensus mask from already-saved EAP-IG score tables; it never reruns
attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT


DEFAULT_CONFIG = ROOT / "configs" / "phase2_qwen_refusal.json"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config(path: Path) -> dict:
    payload = _read_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported Phase 2 config schema in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def audit(config_path: Path, *, require_phase2_data: bool = False) -> None:
    cfg = _config(config_path)
    results_root = _resolve(cfg["artifacts"]["results_root"])
    historical_data = _resolve(cfg["dataset"]["historical_data_root"])
    languages = cfg["dataset"]["languages"]
    expected_site = (
        cfg["metric"]["target_layer"],
        cfg["metric"]["target_position"],
        cfg["metric"]["target_hook"],
    )
    source_records = []
    for language in languages:
        language_root = results_root / language
        metadata_path = language_root / cfg["artifacts"]["metadata_subpath"]
        direction_path = language_root / cfg["artifacts"]["direction_subpath"]
        score_path = (
            language_root
            / cfg["artifacts"]["local_discovery_subdir"]
            / cfg["artifacts"]["score_filename"]
        )
        discovery_meta_path = score_path.with_name("eapig_run_metadata.json")
        data_meta_path = historical_data / language / "preparation_metadata.json"
        historical_test_path = historical_data / language / "test.jsonl"
        for path in (
            metadata_path,
            direction_path,
            score_path,
            discovery_meta_path,
            data_meta_path,
            historical_test_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        metadata = _read_json(metadata_path)
        site = (int(metadata["layer"]), int(metadata["position"]), metadata["target_hook"])
        if site != expected_site:
            raise ValueError(f"{language} metric site {site} != frozen site {expected_site}")
        if metadata.get("model") != cfg["model"]["name"]:
            raise ValueError(f"{language} metric is for {metadata.get('model')}")
        discovery = _read_json(discovery_meta_path)
        if discovery.get("n_prompts") != 64 or discovery.get("split") != "train":
            raise ValueError(f"Unexpected {language} discovery provenance: {discovery}")
        data_meta = _read_json(data_meta_path)
        if data_meta.get("source_commit") != cfg["dataset"]["source_commit"]:
            raise ValueError(f"{language} PolyRefuse source commit is not frozen")
        record = {
                "language": language,
                "direction": str(direction_path.relative_to(ROOT)),
                "direction_sha256": _sha256(direction_path),
                "scores": str(score_path.relative_to(ROOT)),
                "scores_sha256": _sha256(score_path),
                "metric_metadata_sha256": _sha256(metadata_path),
                "discovery_metadata_sha256": _sha256(discovery_meta_path),
                "historical_test_sha256": _sha256(historical_test_path),
                "preparation_metadata_sha256": _sha256(data_meta_path),
            }
        expected_hashes = cfg.get("frozen_sha256", {}).get(language)
        if not expected_hashes:
            raise ValueError(f"No frozen SHA-256 manifest for {language}")
        actual_hashes = {
            "direction": record["direction_sha256"],
            "scores": record["scores_sha256"],
            "metric_metadata": record["metric_metadata_sha256"],
            "discovery_metadata": record["discovery_metadata_sha256"],
            "historical_test": record["historical_test_sha256"],
            "preparation_metadata": record["preparation_metadata_sha256"],
        }
        if actual_hashes != expected_hashes:
            raise ValueError(
                f"Frozen artifact hash mismatch for {language}: "
                f"actual={actual_hashes}, expected={expected_hashes}"
            )
        source_records.append(record)
    if require_phase2_data:
        phase2_data = _resolve(cfg["dataset"]["phase2_data_root"])
        for language in languages:
            for split in ("development", "confirmation", "reserve"):
                path = phase2_data / language / f"{split}.jsonl"
                if not path.exists():
                    raise FileNotFoundError(path)
    print(json.dumps({"status": "PASS", "config": str(config_path), "sources": source_records}, indent=2))


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _stable_order(ids: list[str], seed: int) -> list[str]:
    return sorted(
        ids,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )


def prepare_data(config_path: Path, external_root: Path, *, dry_run: bool = False) -> None:
    cfg = _config(config_path)
    external_root = external_root.resolve()
    expected_commit = cfg["dataset"]["source_commit"]
    actual_commit = _git_commit(external_root)
    if actual_commit != expected_commit:
        raise ValueError(
            f"PolyRefuse checkout is {actual_commit}; expected frozen commit {expected_commit}."
        )
    languages = cfg["dataset"]["languages"]
    historical_root = _resolve(cfg["dataset"]["historical_data_root"])
    output_root = _resolve(cfg["dataset"]["phase2_data_root"])

    # Exclude an ID if any language used it before. This is stricter than
    # language-wise exclusion and preserves a genuinely parallel new pool.
    used_ids: set[str] = set()
    for language in languages:
        for row in _jsonl_rows(historical_root / language / "test.jsonl"):
            if row.get("label") == "harmful":
                used_ids.add(str(row["parallel_id"]))

    raw_by_language: dict[str, list[dict]] = {}
    lengths = set()
    source_paths = {}
    for language in languages:
        path = external_root / "PolyRefuse" / f"harmful_test_translated_{language}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list: {path}")
        raw_by_language[language] = rows
        source_paths[language] = str(path)
        lengths.add(len(rows))
    if len(lengths) != 1:
        raise ValueError(f"Translated harmful test files are not parallel: lengths={lengths}")
    raw_count = lengths.pop()
    candidate_ids = [
        f"polyrefuse-harmful-test-{index}"
        for index in range(raw_count)
        if f"polyrefuse-harmful-test-{index}" not in used_ids
    ]
    ordered = _stable_order(candidate_ids, int(cfg["dataset"]["selection_seed"]))
    split_sizes = {
        "development": int(cfg["dataset"]["development_harmful_prompts"]),
        "confirmation": int(cfg["dataset"]["confirmation_harmful_prompts"]),
        "reserve": int(cfg["dataset"]["reserve_harmful_prompts"]),
    }
    if sum(split_sizes.values()) > len(ordered):
        raise ValueError(
            f"Need {sum(split_sizes.values())} unseen parallel IDs, found {len(ordered)}."
        )
    split_ids: dict[str, list[str]] = {}
    start = 0
    for split, count in split_sizes.items():
        split_ids[split] = ordered[start : start + count]
        start += count
    if set(split_ids["development"]) & set(split_ids["confirmation"]):
        raise AssertionError("Development and confirmation IDs overlap")

    if dry_run:
        for language in languages:
            for ids in split_ids.values():
                for parallel_id in ids:
                    index = int(parallel_id.rsplit("-", 1)[1])
                    raw = raw_by_language[language][index]
                    field = (
                        "instruction_translated"
                        if language != "en" and isinstance(raw.get("instruction_translated"), str)
                        else "instruction"
                    )
                    if not isinstance(raw.get(field), str) or not raw[field].strip():
                        raise ValueError(f"Missing {field} in {language} raw row {index}")
        print(
            json.dumps(
                {
                    "status": "PASS_DRY_RUN",
                    "raw_parallel_harmful_count": raw_count,
                    "prior_union_excluded_ids": len(used_ids),
                    "eligible_unseen_parallel_ids": len(ordered),
                    "split_counts": split_sizes,
                    "would_write_under": str(output_root),
                },
                indent=2,
            )
        )
        return

    for language in languages:
        for split, ids in split_ids.items():
            output_path = output_root / language / f"{split}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                for parallel_id in ids:
                    index = int(parallel_id.rsplit("-", 1)[1])
                    raw = raw_by_language[language][index]
                    field = (
                        "instruction_translated"
                        if language != "en" and isinstance(raw.get("instruction_translated"), str)
                        else "instruction"
                    )
                    instruction = raw.get(field)
                    if not isinstance(instruction, str) or not instruction.strip():
                        raise ValueError(f"Missing {field} in {language} raw row {index}")
                    row = {
                        "id": f"phase2-{language}-{parallel_id}",
                        "parallel_id": parallel_id,
                        "instruction": instruction.strip(),
                        "label": "harmful",
                        "category": str(raw.get("category", "unspecified")),
                        "language": language,
                        "split": split,
                        "source": "PolyRefuse",
                        "source_commit": expected_commit,
                        "source_row_index": index,
                        "source_text_field": field,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "protocol": "parallel pre-output split; union-exclude all prior four-language test IDs",
        "model": cfg["model"]["name"],
        "dataset": "PolyRefuse",
        "source_repository": cfg["dataset"]["source_repository"],
        "source_commit": expected_commit,
        "languages": languages,
        "raw_parallel_harmful_count": raw_count,
        "prior_union_excluded_ids": len(used_ids),
        "eligible_unseen_parallel_ids": len(ordered),
        "selection_seed": cfg["dataset"]["selection_seed"],
        "split_counts": split_sizes,
        "split_id_sha256": {
            split: hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
            for split, ids in split_ids.items()
        },
        "source_paths_at_preparation": source_paths,
        "model_outputs_used_for_selection": False,
        "generated_text_saved": False,
    }
    _write_json(output_root / "manifest.json", manifest)
    print(f"Wrote new parallel Phase 2 splits and manifest under {output_root}")


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def build_masks(config_path: Path) -> None:
    cfg = _config(config_path)
    languages = cfg["dataset"]["languages"]
    k_values = [str(value) for value in cfg["circuit_sweep"]["requested_k"]]
    results_root = cfg["artifacts"]["results_root"]
    discovery = cfg["artifacts"]["local_discovery_subdir"]
    circuits = cfg["artifacts"]["circuits_subdir"]
    metadata = cfg["artifacts"]["metadata_subpath"]
    python = sys.executable
    _run(
        [
            python,
            "scripts/rq.py",
            "rq2.rebuild-circuits",
            "--results-root",
            results_root,
            "--languages",
            *languages,
            "--method",
            "eapig",
            "--selection",
            "greedy_no_parent_prune",
            "--discovery-subdir",
            discovery,
            "--metadata-subpath",
            metadata,
            "--output-subdir",
            circuits,
            "--k-values",
            *k_values,
        ]
    )
    _run(
        [
            python,
            "scripts/legacy/rq2_make_consensus_circuits.py",
            "--model",
            cfg["model"]["name"],
            "--languages",
            *languages,
            "--method",
            "eapig",
            "--results-root",
            results_root,
            "--discovery-subdir",
            discovery,
            "--metadata-subpath",
            metadata,
            "--output-subdir",
            circuits,
            "--k-values",
            *k_values,
        ]
    )
    _run(
        [
            python,
            "scripts/legacy/rq2_make_intersection_circuits.py",
            "--model",
            cfg["model"]["name"],
            "--languages",
            *languages,
            "--method",
            "eapig",
            "--results-root",
            results_root,
            "--discovery-subdir",
            discovery,
            "--circuits-subdir",
            circuits,
            "--summary-output-root",
            str(Path(cfg["artifacts"]["phase2_results_root"]) / "structure" / "intersection_build"),
            "--include-pairs",
            "--majority-threshold",
            "3",
            "--k-values",
            *k_values,
        ]
    )
    print("Built masks only from frozen score tables; no model or attribution run was launched.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="Read-only frozen-artifact preflight.")
    audit_parser.add_argument("--require-phase2-data", action="store_true")
    data_parser = subparsers.add_parser("prepare-data", help="Create disjoint development/confirmation IDs.")
    data_parser.add_argument(
        "--external-root", type=Path, default=ROOT / "external" / "multilingual_refusal"
    )
    data_parser.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("build-masks", help="CPU-only circuit-mask post-processing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        audit(args.config, require_phase2_data=args.require_phase2_data)
    elif args.command == "prepare-data":
        prepare_data(args.config, args.external_root, dry_run=args.dry_run)
    elif args.command == "build-masks":
        audit(args.config)
        build_masks(args.config)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

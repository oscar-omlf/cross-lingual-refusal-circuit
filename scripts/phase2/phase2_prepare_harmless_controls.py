#!/usr/bin/env python3
"""Prepare model-output-independent, language-native harmless control splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT


LANGUAGES = ("en", "de", "zh", "th")


def _read_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 200:
        raise ValueError(f"Expected 200 harmless rows in {path}")
    return payload


def _hash_order(index: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:harmless-val:{index}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=ROOT / "data/processed/qwen2.5-7b-instruct/polyrefuse",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=ROOT / "external/multilingual_refusal",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/processed/qwen2.5-7b-instruct/polyrefuse_phase2_harmless",
    )
    parser.add_argument("--development-count", type=int, default=32)
    parser.add_argument("--reserve-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=240720)
    args = parser.parse_args()

    source_paths = {
        "en": args.external_root / "dataset/splits/harmless_val_200_sampled.json",
        "de": args.external_root / "PolyRefuse/harmless_val_translated_de.json",
        "zh": args.external_root / "PolyRefuse/harmless_val_translated_zh.json",
        "th": args.external_root / "PolyRefuse/harmless_val_translated_th.json",
    }
    sources = {language: _read_rows(path) for language, path in source_paths.items()}

    historical_union: set[int] = set()
    for language in LANGUAGES:
        path = args.historical_root / language / "val.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("label") != "harmless":
                continue
            prefix = "polyrefuse-harmless-val-"
            parallel_id = str(row["parallel_id"])
            if not parallel_id.startswith(prefix):
                raise ValueError(f"Unexpected harmless ID {parallel_id}")
            historical_union.add(int(parallel_id.removeprefix(prefix)))
    eligible = sorted(set(range(200)) - historical_union, key=lambda index: _hash_order(index, args.seed))
    total = args.development_count + args.reserve_count
    if len(eligible) < total:
        raise ValueError(f"Need {total} unused indices, found {len(eligible)}")
    split_indices = {
        "development": eligible[: args.development_count],
        "reserve": eligible[args.development_count : total],
    }

    for language in LANGUAGES:
        language_root = args.output_root / language
        language_root.mkdir(parents=True, exist_ok=True)
        for split, indices in split_indices.items():
            lines = []
            for index in indices:
                source = sources[language][index]
                row = {
                    "id": f"polyrefuse-harmless-val-{index}",
                    "parallel_id": f"polyrefuse-harmless-val-{index}",
                    "instruction": source["instruction"],
                    "label": "harmless",
                    "category": source.get("category"),
                    "source": "PolyRefuse",
                    "source_file": str(source_paths[language].relative_to(ROOT)),
                    "source_index": index,
                    "language": language,
                    "selection_used_model_outputs": False,
                }
                lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
            (language_root / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "dataset": "PolyRefuse",
        "purpose": "Phase 2 harmless specificity controls",
        "languages": list(LANGUAGES),
        "source_split": "harmless_val",
        "historical_union_excluded_indices": len(historical_union),
        "eligible_unused_indices": len(eligible),
        "selection_seed": args.seed,
        "selection_used_model_outputs": False,
        "split_counts": {name: len(indices) for name, indices in split_indices.items()},
        "split_id_sha256": {
            name: hashlib.sha256("\n".join(map(str, indices)).encode()).hexdigest()
            for name, indices in split_indices.items()
        },
        "source_paths": {language: str(path.relative_to(ROOT)) for language, path in source_paths.items()},
        "note": "Chinese harmless-val is language-native but is not index-parallel to the other translations; analyses are language-wise.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", **manifest}, indent=2))


if __name__ == "__main__":
    main()

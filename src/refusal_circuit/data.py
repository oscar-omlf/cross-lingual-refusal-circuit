"""Input-data loading and JSONL persistence for the balanced RQ1 dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .paths import DATA_DIR, ensure_parent


SPLIT_SIZES = {
    "train": {"harmful": 128, "harmless": 128},
    "val": {"harmful": 32, "harmless": 32},
    "test": {"harmful": 64, "harmless": 64},
}


def _normalise_row(row: dict, label: str, source: str, index: int) -> dict:
    instruction = row.get("instruction") or row.get("prompt") or row.get("goal")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Could not find a non-empty instruction in source row {index}.")
    return {
        "id": f"{source}-{label}-{index}",
        "instruction": instruction.strip(),
        "label": label,
        "category": str(row.get("category", "unspecified")),
        "source": source,
    }


def _read_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return parsed


def load_candidate_pool(
    label: str,
    split: str,
    *,
    external_dir: Path | None = None,
    local_raw_dir: Path | None = None,
) -> list[dict]:
    """Load candidates without making the external checkout a runtime dependency.

    Priority is a user-provided local JSON/JSONL pool, then the one-time
    ``external/refusal_direction`` checkout requested for this project, and
    finally public Hugging Face datasets.  After preparation all RQ1 scripts
    use only ``data/processed/rq1/*.jsonl``.
    """
    if label not in {"harmful", "harmless"}:
        raise ValueError("label must be 'harmful' or 'harmless'.")
    if split not in SPLIT_SIZES:
        raise ValueError(f"Unknown split {split!r}.")

    local_raw_dir = local_raw_dir or Path("data/raw/rq1")
    for suffix in (".jsonl", ".json"):
        local_path = local_raw_dir / f"{label}_{split}{suffix}"
        if local_path.exists():
            return [
                _normalise_row(row, label, f"local:{local_path.name}", index)
                for index, row in enumerate(_read_json_or_jsonl(local_path))
            ]

    external_dir = external_dir or Path("external/refusal_direction")
    external_path = external_dir / "dataset" / "splits" / f"{label}_{split}.json"
    if external_path.exists():
        return [
            _normalise_row(row, label, "refusal_direction", index)
            for index, row in enumerate(_read_json_or_jsonl(external_path))
        ]

    # This fallback is intentionally explicit and only used when source files
    # are absent. Dataset schemas have changed over time, hence normalisation.
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FileNotFoundError(
            "No local source pool was found and `datasets` is unavailable. "
            "Provide data/raw/rq1/{harmful,harmless}_{split}.jsonl or clone "
            "external/refusal_direction as documented."
        ) from exc

    if label == "harmful":
        dataset = load_dataset("walledai/AdvBench", split="train")
        source = "hf:walledai/AdvBench"
    else:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        source = "hf:tatsu-lab/alpaca"
    return [_normalise_row(dict(row), label, source, index) for index, row in enumerate(dataset)]


def processed_path(split: str) -> Path:
    return DATA_DIR / f"{split}.jsonl"


def write_processed_split(split: str, rows: Iterable[dict]) -> Path:
    path = ensure_parent(processed_path(split))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_processed_split(split: str, label: str | None = None) -> list[dict]:
    path = processed_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/rq1_prepare_dataset.py` first."
        )
    rows = _read_json_or_jsonl(path)
    if label is not None:
        rows = [row for row in rows if row["label"] == label]
    return rows

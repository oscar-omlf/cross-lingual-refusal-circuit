"""Repository-relative paths and small serialisation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "rq1"
RESULTS_DIR = REPO_ROOT / "results" / "rq1"


def ensure_parent(path: Path) -> Path:
    """Create ``path.parent`` and return ``path`` for concise call sites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: Any) -> None:
    with ensure_parent(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

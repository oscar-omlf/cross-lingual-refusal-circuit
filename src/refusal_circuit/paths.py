"""Repository-relative paths and small serialisation helpers.

The first RQ1 run pre-dated model/language namespaces.  Results now live under
``{model}/{dataset}/{language}`` so an intervention artefact can never be
silently reused with an incompatible model or translated prompt set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GEMMA_RQ1_DATA_DIR = REPO_ROOT / "data" / "processed" / "gemma-2-2b-it" / "en" / "rq1"
GEMMA_RQ1_RESULTS_DIR = REPO_ROOT / "results" / "gemma-2-2b-it" / "en" / "rq1"

# Backwards-compatible names for the RQ1 helpers.  New code should accept an
# explicit path (or call ``experiment_*_dir``) rather than relying on either.
DATA_DIR = GEMMA_RQ1_DATA_DIR
RESULTS_DIR = GEMMA_RQ1_RESULTS_DIR


def model_slug(model_name: str) -> str:
    """Stable filesystem name for a Hugging Face model identifier."""
    return model_name.rsplit("/", 1)[-1].lower().replace("_", "-")


def experiment_data_dir(model_name: str, dataset: str, language: str) -> Path:
    """Return the processed-data namespace for one model/dataset/language."""
    return REPO_ROOT / "data" / "processed" / model_slug(model_name) / dataset / language


def experiment_results_dir(model_name: str, dataset: str, language: str) -> Path:
    """Return the result namespace for one model/dataset/language."""
    return REPO_ROOT / "results" / model_slug(model_name) / dataset / language


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

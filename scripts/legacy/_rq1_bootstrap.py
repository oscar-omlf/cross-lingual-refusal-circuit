"""Allow RQ1 scripts to run from a source checkout before editable install."""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Find the checkout root from either scripts/ or scripts/legacy/."""

    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "EAP-IG").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


ROOT = _find_repo_root(Path(__file__).resolve().parent)
for path in (ROOT / "src", ROOT / "EAP-IG"):
    rendered = str(path)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)

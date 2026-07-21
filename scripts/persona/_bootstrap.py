"""Make project packages importable for persona entry points."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "EAP-IG", ROOT / "scripts" / "legacy"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

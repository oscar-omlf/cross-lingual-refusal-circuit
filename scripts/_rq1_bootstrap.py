"""Allow RQ1 scripts to run from a source checkout before editable install."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "EAP-IG"):
    rendered = str(path)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)

#!/usr/bin/env python3
"""Extract the configured trait vector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "EAP-IG")]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

from refusal_circuit.workflow import extract_vector, load_config

print(extract_vector(load_config(args.config), overwrite=args.overwrite))

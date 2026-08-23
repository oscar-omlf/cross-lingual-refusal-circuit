#!/usr/bin/env python3
"""Run the configured vector and circuit tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "EAP-IG")]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--circuit-path", type=Path)
parser.add_argument("--output-path", type=Path)
args = parser.parse_args()

from refusal_circuit.workflow import load_config, run_tests

config = load_config(args.config)
if args.circuit_path is not None:
    config["tests"]["circuit_path"] = str(args.circuit_path)
if args.output_path is not None:
    config["tests"]["output_path"] = str(args.output_path)
print(run_tests(config))

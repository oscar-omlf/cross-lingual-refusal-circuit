#!/usr/bin/env python3
"""Judge every frozen Qwen persona layer/coefficient steering output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT


def _label(coefficient: float) -> str:
    return f"{coefficient:+g}".replace("+", "pos").replace("-", "neg").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    intermediate = ROOT / config["paths"]["intermediate_root"]
    layers = config["layer_selection"]["candidate_layers_one_indexed"]
    coefficients = config["steering_validation"]["coefficients"]
    judge_script = ROOT / "scripts/persona/persona_judge_responses.py"
    for layer in layers:
        layer_root = intermediate / "steering" / f"layer_{int(layer):02d}"
        for coefficient in coefficients:
            name = f"coef_{_label(float(coefficient))}.jsonl"
            raw = layer_root / name
            scored = layer_root / "scored" / name
            subprocess.run(
                [
                    sys.executable,
                    str(judge_script),
                    "--config",
                    str(args.config),
                    "--artifact",
                    "eval",
                    "--input",
                    str(raw),
                    "--output",
                    str(scored),
                    "--resume",
                ],
                check=True,
                cwd=ROOT,
            )


if __name__ == "__main__":
    main()

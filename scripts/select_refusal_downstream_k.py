#!/usr/bin/env python3
"""Print the smallest downstream circuit passing frozen representation thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--k-values", default="100,500,1000,3000")
    parser.add_argument("--min-faithfulness", type=float, default=0.85)
    parser.add_argument("--min-necessity", type=float, default=0.95)
    args = parser.parse_args()

    for k in [int(value) for value in args.k_values.split(",") if value]:
        path = args.metrics_dir / f"k{k}.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if (
            float(metrics["faithfulness"]) >= args.min_faithfulness
            and float(metrics["necessity"]) >= args.min_necessity
        ):
            print(k)
            return
    raise SystemExit("No downstream circuit passed faithfulness/necessity thresholds.")


if __name__ == "__main__":
    main()

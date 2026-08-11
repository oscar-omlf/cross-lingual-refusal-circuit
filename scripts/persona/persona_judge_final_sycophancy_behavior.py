#!/usr/bin/env python3
"""Canonically judge only missing rows in the final Qwen sycophancy suite."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from _bootstrap import ROOT
from persona_run_final_sycophancy_behavior import (
    CONFIG,
    CONNECTED_BEHAVIOR,
    DOWNSTREAM_BEHAVIOR,
    UPSTREAM_BEHAVIOR,
)

QSLUG = "q0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19"
BASELINE_SCORED = UPSTREAM_BEHAVIOR / f"k1000_{QSLUG}_max640_final/scored.jsonl"


def judged_is_current(raw: Path, scored: Path) -> bool:
    metadata = scored.with_suffix(".metadata.json")
    if not raw.exists() or not scored.exists() or not metadata.exists():
        return False
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    from refusal_circuit.persona import sha256_file
    return (
        payload.get("status") == "complete"
        and payload.get("input_sha256") == sha256_file(raw)
        and payload.get("rows") == sum(1 for line in raw.read_text().splitlines() if line)
    )


def judge(directory: Path) -> None:
    raw = directory / "responses.jsonl"
    scored = directory / "scored.jsonl"
    if judged_is_current(raw, scored):
        print(f"SKIP current judgment {scored.relative_to(ROOT)}", flush=True)
        return
    argv = [
        sys.executable,
        str(ROOT / "scripts/rq.py"),
        "persona.judge",
        "--config", str(CONFIG),
        "--input", str(raw),
        "--output", str(scored),
        "--artifact", "eval",
        "--protocol-field", "behavior_protocol_id",
        "--concurrency", "8",
        "--reuse-output", str(BASELINE_SCORED),
    ]
    if scored.exists():
        argv.append("--resume")
    print("RUN", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def main() -> None:
    for k in (2000, 3000):
        judge(UPSTREAM_BEHAVIOR / f"k{k}_{QSLUG}_max640_final")
    judge(DOWNSTREAM_BEHAVIOR / f"k100_{QSLUG}_max640_a100_necessity_final")
    for k in (250, 500, 1000, 2000, 3000):
        judge(DOWNSTREAM_BEHAVIOR / f"k{k}_{QSLUG}_max1000_a100_necessity_lowmem_v1")
    union = CONNECTED_BEHAVIOR / "ku5000_kd3000"
    judge(union / f"union_both_circuit_only_ku5000_kd3000_{QSLUG}_max1000_a100_reference_v3")
    judge(union / f"union_both_ablated_ku5000_kd3000_{QSLUG}_max1000_a100_reference_v3")


if __name__ == "__main__":
    main()

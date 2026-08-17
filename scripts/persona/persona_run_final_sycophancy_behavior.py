#!/usr/bin/env python3
"""Run missing cells in the historical cut-set sycophancy behavior suite.

This orchestrator predates the source-to-sink methodology correction. It is
retained only for explicit legacy diagnostics and must not be used to produce
the primary circuit claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from _bootstrap import ROOT


CONFIG = ROOT / "configs/qwen_sycophancy.json"
UPSTREAM_RUN = "sharedprompt_n64_ig5_k3000_20260726"
DOWNSTREAM_RUN = "sharedprompt_n64_ig5_full768_h100_cneg2_20260802"
QUESTIONS = [str(index) for index in range(20)]
INTERMEDIATE = ROOT / "data/intermediate/persona/qwen2.5-7b-instruct/sycophantic"
UPSTREAM_BEHAVIOR = INTERMEDIATE / "natural_upstream_circuit_behavior" / UPSTREAM_RUN
DOWNSTREAM_BEHAVIOR = INTERMEDIATE / "steering_circuit_behavior" / DOWNSTREAM_RUN
CONNECTED_BEHAVIOR = (
    INTERMEDIATE
    / "connected_circuit_behavior"
    / f"{UPSTREAM_RUN}__{DOWNSTREAM_RUN}"
)
BASELINE = (
    UPSTREAM_BEHAVIOR
    / "k1000_q0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19_max640_final"
    / "responses.jsonl"
)


def run(command: str, *arguments: str) -> None:
    argv = [sys.executable, str(ROOT / "scripts/rq.py"), command, *arguments]
    print("RUN", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def output_is_complete(path: Path, conditions: set[str], expected: int) -> bool:
    metadata_path = path / "metadata.json"
    responses_path = path / "responses.jsonl"
    if not metadata_path.exists() or not responses_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        return False
    rows = [json.loads(line) for line in responses_path.read_text().splitlines() if line]
    return (
        len(rows) == expected
        and {row["condition"] for row in rows} == conditions
        and not any(row.get("hit_max_new_tokens", False) for row in rows)
    )


def upstream_extensions() -> None:
    suffix = "q0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19_max640_final"
    for k in (2000, 3000):
        output = UPSTREAM_BEHAVIOR / f"k{k}_{suffix}"
        if output_is_complete(output, {"normal", "circuit_ablated"}, 40):
            print(f"SKIP complete upstream k={k}", flush=True)
            continue
        run(
            "persona.behavior-upstream-contrastive-circuit",
            "--config", str(CONFIG),
            "--run-name", UPSTREAM_RUN,
            "--k", str(k),
            "--output-tag", f"k{k}_{suffix}",
            "--question-indices", *QUESTIONS,
            "--max-new-tokens", "1000",
            "--max-length", "1024",
            "--conditions", "normal", "circuit_ablated",
            "--resume",
            "--extend-capped",
        )


def downstream_curve() -> None:
    qslug = "q0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19"
    for k in (250, 500, 1000, 2000, 3000):
        tag = f"k{k}_{qslug}_max1000_a100_necessity_lowmem_v1"
        output = DOWNSTREAM_BEHAVIOR / tag
        if output_is_complete(output, {"normal", "circuit_ablated"}, 40):
            print(f"SKIP complete downstream k={k}", flush=True)
            continue
        arguments = [
            "--config", str(CONFIG),
            "--run-name", DOWNSTREAM_RUN,
            "--k", str(k),
            "--output-tag", tag,
            "--question-indices", *QUESTIONS,
            "--max-new-tokens", "1000",
            "--max-length", "1024",
            "--baseline-responses", str(BASELINE),
            "--conditions", "normal", "circuit_ablated",
        ]
        if output.exists() and any(output.iterdir()):
            arguments.append("--resume")
        run("persona.behavior-steering-circuit", *arguments)


def selected_union() -> None:
    qslug = "q0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19"
    base = CONNECTED_BEHAVIOR / "ku5000_kd3000"
    jobs = (
        (
            "both_circuit_only",
            f"union_both_circuit_only_ku5000_kd3000_{qslug}_max1000_a100_reference_v3",
        ),
        (
            "both_ablated",
            f"union_both_ablated_ku5000_kd3000_{qslug}_max1000_a100_reference_v3",
        ),
    )
    for condition, tag in jobs:
        output = base / tag
        if output_is_complete(output, {condition}, 20):
            print(f"SKIP complete union {condition}", flush=True)
            continue
        arguments = [
            "--config", str(CONFIG),
            "--upstream-run", UPSTREAM_RUN,
            "--downstream-run", DOWNSTREAM_RUN,
            "--upstream-k", "5000",
            "--downstream-k", "3000",
            "--condition", condition,
            "--question-indices", *QUESTIONS,
            "--output-tag", tag,
            "--max-new-tokens", "1000",
            "--max-length", "1024",
        ]
        if output.exists() and any(output.iterdir()):
            arguments.append("--resume")
        run("persona.behavior-connected-circuits", *arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        required=True,
        choices=("upstream-extensions", "downstream-curve", "selected-union"),
    )
    parser.add_argument(
        "--allow-legacy-cut-set-diagnostics",
        action="store_true",
        help="Acknowledge that these jobs use historical cut-set artifacts.",
    )
    args = parser.parse_args()
    if not args.allow_legacy_cut_set_diagnostics:
        raise RuntimeError(
            "Historical orchestrator disabled by default. Follow "
            "PERSONA_CIRCUIT_METHOD_REPAIR.md, or pass the explicit legacy "
            "acknowledgement only for a cut-set diagnostic."
        )
    {
        "upstream-extensions": upstream_extensions,
        "downstream-curve": downstream_curve,
        "selected-union": selected_union,
    }[args.family]()


if __name__ == "__main__":
    main()

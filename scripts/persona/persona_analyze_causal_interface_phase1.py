#!/usr/bin/env python3
"""Analyze judged Phase 1 causal-interface oracle generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_sycophancy.json",
    )
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--coherence-minimum", type=float, default=80.0)
    parser.add_argument("--maximum-cap-hit-rate", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    if not 0 <= args.maximum_cap_hit_rate <= 1:
        raise ValueError("--maximum-cap-hit-rate must be in [0, 1].")

    import numpy as np

    from refusal_circuit.causal_interface import (
        PHASE1_RANKS,
        PHASE1_TIMINGS,
        finite_number,
        select_rank_and_timing,
    )
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    scored_path = _resolve(args.scored)
    responses_path = scored_path.with_name("responses.jsonl")
    generation_path = scored_path.with_name("metadata.json")
    judge_path = scored_path.with_suffix(".metadata.json")
    for path in (scored_path, responses_path, generation_path, judge_path):
        if not path.exists():
            raise FileNotFoundError(path)
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    recorded_outputs = {
        row["path"]: row["sha256"] for row in generation["outputs"]
    }
    if (
        generation.get("status") != "complete"
        or generation.get("phase") != 1
        or not generation.get("identity", {}).get("passed")
        or generation.get("behavior_protocol_id") != config["protocol_id"]
        or recorded_outputs.get(str(responses_path.relative_to(ROOT)))
        != sha256_file(responses_path)
        or judge.get("status") != "complete"
        or judge.get("input_sha256") != sha256_file(responses_path)
        or judge.get("output_sha256") != sha256_file(scored_path)
    ):
        raise ValueError("Phase 1 generation/judge provenance validation failed.")
    rows = read_jsonl(scored_path)
    by_condition: dict[str, dict[int, dict]] = {}
    for row in rows:
        condition = row["condition"]
        question = int(row["question_index"])
        if question in by_condition.setdefault(condition, {}):
            raise ValueError(f"Duplicate {condition} q{question}.")
        for field in ("trait_score", "coherence_score"):
            finite_number(row[field])
        by_condition[condition][question] = row
    questions = sorted(by_condition["normal_positive"])
    expected_conditions = set(generation["generation"]["conditions"])
    if set(by_condition) != expected_conditions:
        missing = expected_conditions - set(by_condition)
        extra = set(by_condition) - expected_conditions
        raise ValueError(f"Condition grid mismatch: missing={missing}, extra={extra}")
    for condition, condition_rows in by_condition.items():
        if sorted(condition_rows) != questions:
            raise ValueError(f"Question grid mismatch for {condition}.")

    def values(condition: str, field: str) -> np.ndarray:
        return np.asarray(
            [finite_number(by_condition[condition][q][field]) for q in questions],
            dtype=float,
        )

    trait = {
        condition: values(condition, "trait_score")
        for condition in sorted(by_condition)
    }
    coherence = {
        condition: values(condition, "coherence_score")
        for condition in sorted(by_condition)
    }
    cap_hits = {
        condition: np.asarray(
            [
                bool(by_condition[condition][q]["hit_max_new_tokens"])
                for q in questions
            ],
            dtype=float,
        )
        for condition in sorted(by_condition)
    }
    natural_gap = float(
        trait["normal_positive"].mean() - trait["negative_baseline"].mean()
    )
    if not natural_gap > 0:
        raise ValueError("Natural positive trait score does not exceed negative.")

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0,
        len(questions),
        size=(args.bootstrap_samples, len(questions)),
    )

    def recovery(condition: str) -> dict:
        numerator = float(
            trait[condition].mean() - trait["negative_baseline"].mean()
        )
        point = numerator / natural_gap
        sampled_condition = trait[condition][bootstrap_indices].mean(axis=1)
        sampled_negative = trait["negative_baseline"][bootstrap_indices].mean(axis=1)
        sampled_positive = trait["normal_positive"][bootstrap_indices].mean(axis=1)
        denominator = sampled_positive - sampled_negative
        valid = denominator != 0
        samples = (sampled_condition[valid] - sampled_negative[valid]) / denominator[valid]
        return {
            "recovery": point,
            "ci95": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
            "valid_bootstrap_samples": int(valid.sum()),
            "unclipped": True,
        }

    condition_metrics = {}
    for condition in sorted(by_condition):
        condition_metrics[condition] = {
            "trait_mean": float(trait[condition].mean()),
            "coherence_mean": float(coherence[condition].mean()),
            "cap_hit_rate": float(cap_hits[condition].mean()),
            "normal_termination_rate": float(1.0 - cap_hits[condition].mean()),
            "recovery": recovery(condition),
        }

    recoveries: dict[str, dict[str, dict[str, float]]] = {
        timing: {} for timing in PHASE1_TIMINGS
    }
    full_rank = {}
    for timing in PHASE1_TIMINGS:
        full_condition = f"full_rank_{timing}"
        full_rank[timing] = condition_metrics[full_condition]["recovery"]
        full_healthy = (
            condition_metrics[full_condition]["coherence_mean"]
            >= args.coherence_minimum
            and condition_metrics[full_condition]["cap_hit_rate"]
            <= args.maximum_cap_hit_rate
            and condition_metrics[full_condition]["recovery"]["recovery"] > 0
        )
        full_rank[timing]["healthy_reference_ceiling"] = full_healthy
        for rank in PHASE1_RANKS:
            persona_condition = f"persona_rank{rank}_{timing}"
            complement_condition = f"complement_rank{rank}_{timing}"
            persona = condition_metrics[persona_condition]
            complement = condition_metrics[complement_condition]
            healthy = (
                full_healthy
                and persona["coherence_mean"] >= args.coherence_minimum
                and complement["coherence_mean"] >= args.coherence_minimum
                and persona["cap_hit_rate"] <= args.maximum_cap_hit_rate
                and complement["cap_hit_rate"] <= args.maximum_cap_hit_rate
            )
            recoveries[timing][str(rank)] = {
                "persona": persona["recovery"]["recovery"],
                "complement": complement["recovery"]["recovery"],
                "healthy": healthy,
                "persona_ci95": persona["recovery"]["ci95"],
                "complement_ci95": complement["recovery"]["ci95"],
            }

    selection = select_rank_and_timing(recoveries)
    provenance_passed = (
        generation["split_manifest_sha256"]
        and generation["basis_sha256"]
        and generation["prompt_alignment"]["method"]
        == "two_sided_neutral_whitespace_token_parity"
    )
    phase1_gates = {
        "exact_full_state_identity": bool(generation["identity"]["passed"]),
        "persona_recovery": bool(selection["passed"]),
        "matched_complement_blocking": bool(selection["passed"]),
        "full_rank_reference_ceiling": any(
            row["healthy_reference_ceiling"] for row in full_rank.values()
        ),
        "coherence_and_termination": bool(selection["passed"]),
        "provenance_and_split_disjointness": bool(provenance_passed),
    }
    phase1_passed = all(phase1_gates.values())
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "phase": 1,
        "method": "question_bootstrap_causal_interface_oracle_analysis",
        "phase1_passed": phase1_passed,
        "phase1_gates": phase1_gates,
        "selection": selection,
        "thresholds": {
            "persona_recovery_minimum": 0.80,
            "complement_recovery_maximum": 0.20,
            "prompt_only_within_recurrent": 0.05,
            "coherence_minimum": args.coherence_minimum,
            "maximum_cap_hit_rate": args.maximum_cap_hit_rate,
        },
        "questions": questions,
        "natural_gap": {
            "positive_trait_mean": float(trait["normal_positive"].mean()),
            "negative_trait_mean": float(trait["negative_baseline"].mean()),
            "points": natural_gap,
        },
        "full_rank_recovery": full_rank,
        "rank_timing_recovery": recoveries,
        "condition_metrics": condition_metrics,
        "exact_identity_token_agreement": generation["identity"][
            "exact_token_agreement"
        ],
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "unit": "behavioral_question",
        },
        "internal_activation_projection_diagnostics": {
            condition: [
                by_condition[condition][question][
                    "internal_projection_diagnostics"
                ]
                for question in questions
                if by_condition[condition][question].get(
                    "internal_projection_diagnostics"
                )
            ]
            for condition in sorted(by_condition)
            if condition.startswith(("persona_", "complement_"))
        },
        "artifacts": {
            "generation_metadata": {
                "path": str(generation_path.relative_to(ROOT)),
                "sha256": sha256_file(generation_path),
            },
            "scored": {
                "path": str(scored_path.relative_to(ROOT)),
                "sha256": sha256_file(scored_path),
            },
            "judge_metadata": {
                "path": str(judge_path.relative_to(ROOT)),
                "sha256": sha256_file(judge_path),
            },
        },
    }
    output = _resolve(args.output)
    write_json(output, aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze strictly vector-mediated sufficient-circuit generations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--normal-scored", type=Path, required=True)
    parser.add_argument("--negative-scored", type=Path, required=True)
    parser.add_argument("--oracle-vector-scored", type=Path, required=True)
    parser.add_argument("--composed-scored", type=Path, required=True)
    parser.add_argument("--blocked-scored", type=Path, required=True)
    parser.add_argument(
        "--orthogonal-scored", type=Path, nargs="+", required=True
    )
    parser.add_argument("--upstream-only-scored", type=Path, default=None)
    parser.add_argument("--downstream-only-scored", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    if len(args.orthogonal_scored) < 5:
        raise ValueError("Use at least five orthogonal-vector seeds.")

    import numpy as np

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))

    def load_scored(path_arg: Path, expected_condition: str):
        path = _resolve(path_arg)
        responses = path.with_name("responses.jsonl")
        generation_path = path.with_name("metadata.json")
        judge_path = path.with_suffix(".metadata.json")
        for required in (path, responses, generation_path, judge_path):
            if not required.exists():
                raise FileNotFoundError(required)
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        outputs = {
            item["path"]: item["sha256"]
            for item in generation.get("outputs", [])
        }
        if (
            generation.get("status") != "complete"
            or generation.get("behavior_protocol_id") != config["protocol_id"]
            or generation.get("condition") != expected_condition
            or outputs.get(str(responses.relative_to(ROOT)))
            != sha256_file(responses)
            or judge.get("status") != "complete"
            or judge.get("input_sha256") != sha256_file(responses)
            or judge.get("output_sha256") != sha256_file(path)
        ):
            raise ValueError(f"Behavior provenance mismatch: {path}")
        rows = read_jsonl(path)
        by_question = {int(row["question_index"]): row for row in rows}
        if not rows or len(by_question) != len(rows):
            raise ValueError(f"Missing or duplicate question rows: {path}")
        for row in rows:
            if row.get("condition") != expected_condition:
                raise ValueError(f"Condition mismatch in {path}")
            for key in ("trait_score", "coherence_score"):
                value = row.get(key)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite {key} in {path}")
        return path, generation, by_question

    loaded = {
        "normal": load_scored(args.normal_scored, "normal_positive"),
        "negative": load_scored(args.negative_scored, "negative_baseline"),
        "oracle": load_scored(
            args.oracle_vector_scored, "oracle_vector_interface"
        ),
        "composed": load_scored(args.composed_scored, "composed_circuit"),
        "blocked": load_scored(
            args.blocked_scored, "composed_v_blocked"
        ),
    }
    optional = {}
    if args.upstream_only_scored is not None:
        optional["upstream_only"] = load_scored(
            args.upstream_only_scored,
            "upstream_circuit_full_downstream",
        )
    if args.downstream_only_scored is not None:
        optional["downstream_only"] = load_scored(
            args.downstream_only_scored,
            "oracle_vector_downstream_circuit",
        )
    orthogonal = [
        load_scored(path, "composed_orthogonal")
        for path in args.orthogonal_scored
    ]

    reference = loaded["composed"][1]
    contract_keys = (
        "behavior_protocol_id",
        "model",
        "model_architecture",
        "question_indices",
        "decoding",
        "max_length",
    )
    for _, metadata, _ in [
        *loaded.values(),
        *optional.values(),
        *orthogonal,
    ]:
        mismatched = [
            key for key in contract_keys if metadata.get(key) != reference.get(key)
        ]
        if mismatched:
            raise ValueError(f"Generation contract mismatch: {mismatched}")
    circuit_keys = (
        "upstream_run",
        "downstream_run",
        "upstream_k",
        "downstream_k",
        "upstream_circuit_sha256",
        "downstream_circuit_sha256",
        "upstream_evaluation_sha256",
        "downstream_evaluation_sha256",
    )
    for name in ("composed", "blocked"):
        mismatched = [
            key
            for key in circuit_keys
            if loaded[name][1].get(key) != reference.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"Composed circuit provenance mismatch: {mismatched}"
            )
    for _, metadata, _ in orthogonal:
        mismatched = [
            key
            for key in circuit_keys
            if metadata.get(key) != reference.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"Orthogonal control circuit mismatch: {mismatched}"
            )
    orthogonal_seeds = [int(metadata["orthogonal_seed"]) for _, metadata, _ in orthogonal]
    if len(set(orthogonal_seeds)) != len(orthogonal_seeds):
        raise ValueError("Orthogonal-vector seeds must be unique.")

    questions = list(map(int, reference["question_indices"]))
    question_set = set(questions)
    for _, _, rows in [
        *loaded.values(),
        *optional.values(),
        *orthogonal,
    ]:
        if set(rows) != question_set:
            raise ValueError("A condition has a different question set.")
        for question in questions:
            normal = loaded["normal"][2][question]
            if any(
                rows[question].get(key) != normal.get(key)
                for key in ("question", "prompt")
            ):
                raise ValueError(f"Prompt mismatch at q{question}.")

    def values(rows, key: str):
        return np.asarray(
            [float(rows[q][key]) for q in questions], dtype=float
        )

    trait = {
        name: values(rows, "trait_score")
        for name, (_, _, rows) in {**loaded, **optional}.items()
    }
    coherence = {
        name: values(rows, "coherence_score")
        for name, (_, _, rows) in {**loaded, **optional}.items()
    }
    orthogonal_trait_by_seed = np.stack(
        [values(rows, "trait_score") for _, _, rows in orthogonal]
    )
    orthogonal_trait = orthogonal_trait_by_seed.mean(axis=0)
    orthogonal_coherence = np.stack(
        [values(rows, "coherence_score") for _, _, rows in orthogonal]
    ).mean(axis=0)

    natural_gap = trait["normal"] - trait["negative"]
    oracle_rescue = trait["oracle"] - trait["negative"]
    composed_rescue = trait["composed"] - trait["negative"]
    blocking = trait["composed"] - trait["blocked"]
    specificity = trait["composed"] - orthogonal_trait

    denominator = float(natural_gap.mean())
    oracle_denominator = float(oracle_rescue.mean())
    composed_denominator = float(composed_rescue.mean())
    if denominator <= 0:
        raise ValueError("Normal-positive behavior does not exceed the baseline.")
    if oracle_denominator <= 0:
        raise ValueError("The vector-only oracle does not rescue the trait.")
    if composed_denominator <= 0:
        raise ValueError("The composed circuit does not rescue the trait.")

    effects = {
        "natural_positive_minus_negative": natural_gap,
        "oracle_vector_minus_negative": oracle_rescue,
        "composed_minus_negative": composed_rescue,
        "composed_minus_blocked": blocking,
        "composed_minus_orthogonal": specificity,
    }
    if "upstream_only" in trait:
        effects["upstream_only_minus_negative"] = (
            trait["upstream_only"] - trait["negative"]
        )
    if "downstream_only" in trait:
        effects["downstream_only_minus_negative"] = (
            trait["downstream_only"] - trait["negative"]
        )

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(
        0, len(questions), size=(args.bootstrap_samples, len(questions))
    )

    def interval(effect):
        samples = effect[indices].mean(axis=1)
        return {
            "mean": float(effect.mean()),
            "ci95": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
            "probability_gt_zero": float((samples > 0).mean()),
        }

    condition_means = {}
    for name in trait:
        condition_means[name] = {
            "trait": float(trait[name].mean()),
            "coherence": float(coherence[name].mean()),
        }
    condition_means["orthogonal_mean"] = {
        "trait": float(orthogonal_trait.mean()),
        "coherence": float(orthogonal_coherence.mean()),
    }

    cap_hits = {}
    for name, (_, _, rows) in {**loaded, **optional}.items():
        cap_hits[name] = sum(
            bool(rows[q].get("hit_max_new_tokens", False)) for q in questions
        )
    cap_hits["orthogonal_total"] = sum(
        bool(rows[q].get("hit_max_new_tokens", False))
        for _, _, rows in orthogonal
        for q in questions
    )

    output = {
        "schema_version": 1,
        "status": "complete",
        "method": "paired strict vector-mediated sufficiency analysis",
        "n_questions": len(questions),
        "question_indices": questions,
        "condition_means": condition_means,
        "paired_effects": {
            name: interval(effect) for name, effect in effects.items()
        },
        "normalized_recovery": {
            "oracle_fraction_of_natural_gap": (
                oracle_denominator / denominator
            ),
            "composed_fraction_of_natural_gap": (
                composed_denominator / denominator
            ),
            "composed_fraction_of_oracle_rescue": (
                composed_denominator / oracle_denominator
            ),
            "fraction_of_composed_rescue_removed_by_v_block": (
                float(blocking.mean()) / composed_denominator
            ),
            "composed_minus_orthogonal_fraction_of_composed_rescue": (
                float(specificity.mean()) / composed_denominator
            ),
        },
        "orthogonal_seeds": orthogonal_seeds,
        "cap_hits": cap_hits,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "unit": "question",
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, (path, _, _) in {**loaded, **optional}.items()
        },
        "orthogonal_artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for path, _, _ in orthogonal
        ],
    }
    write_json(_resolve(args.output), output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

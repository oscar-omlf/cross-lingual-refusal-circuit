#!/usr/bin/env python3
"""Analyze causal persona-vector bottleneck rescue and downstream blocking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _generation_value(metadata: dict, key: str):
    """Read original generation semantics through an audited pooled artifact."""

    compatibility = metadata.get("compatibility", {})
    if (
        metadata.get("method") == "deterministic split-cap pooling audit"
        and key in compatibility
    ):
        return compatibility[key]
    if key in metadata:
        return metadata[key]
    return compatibility.get(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--normal-scored", type=Path, required=True)
    parser.add_argument("--upstream-ablated-scored", type=Path, required=True)
    parser.add_argument("--vector-rescue-scored", type=Path, required=True)
    parser.add_argument("--full-state-rescue-scored", type=Path, required=True)
    parser.add_argument(
        "--full-prefix-state-rescue-scored", type=Path, required=True
    )
    parser.add_argument(
        "--orthogonal-rescue-scored", type=Path, nargs="+", required=True
    )
    parser.add_argument("--unrelated-rescue-scored", type=Path, required=True)
    parser.add_argument("--downstream-blocked-scored", type=Path, required=True)
    parser.add_argument("--downstream-ablated-scored", type=Path, default=None)
    parser.add_argument("--both-ablated-scored", type=Path, default=None)
    parser.add_argument("--minimum-token-cap", type=int, default=350)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_token_cap <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("Token cap and bootstrap sample count must be positive.")
    if (args.downstream_ablated_scored is None) != (
        args.both_ablated_scored is None
    ):
        raise ValueError(
            "Provide both downstream-only and both-ablated artifacts for the "
            "optional 2x2 factorial analysis."
        )

    import numpy as np
    import pandas as pd

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))

    def load_scored(path_arg: Path, *, expected_condition: str | None):
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
            or outputs.get(str(responses.relative_to(ROOT)))
            != sha256_file(responses)
            or judge.get("status") != "complete"
            or judge.get("input_sha256") != sha256_file(responses)
            or judge.get("output_sha256") != sha256_file(path)
        ):
            raise ValueError(f"Behavior provenance mismatch: {path}")
        decoding = generation.get("decoding", {})
        if (
            decoding.get("do_sample") is not False
            or int(decoding.get("max_new_tokens", 0))
            < args.minimum_token_cap
        ):
            raise ValueError(f"Incompatible generation protocol: {path}")
        all_rows = read_jsonl(path)
        rows = all_rows
        if expected_condition is not None:
            rows = [
                row
                for row in all_rows
                if row.get("condition") == expected_condition
            ]
            if not rows:
                raise ValueError(
                    f"No {expected_condition} rows found: {path}"
                )
            declared_condition = generation.get("condition")
            declared_conditions = generation.get("conditions")
            if declared_condition not in (None, expected_condition):
                raise ValueError(f"Generation condition mismatch: {path}")
            if (
                declared_conditions is not None
                and expected_condition not in declared_conditions
            ):
                raise ValueError(f"Generation condition mismatch: {path}")
        by_question = {int(row["question_index"]): row for row in rows}
        if len(by_question) != len(rows):
            raise ValueError(f"Duplicate question rows: {path}")
        for row in rows:
            values = (
                row.get("trait_score"),
                row.get("coherence_score"),
            )
            if any(value is None or not np.isfinite(float(value)) for value in values):
                raise ValueError(f"Non-finite judged score: {path}")
            source_cap = int(decoding["max_new_tokens"])
            count = int(row["generated_token_count"])
            capped = bool(row.get("hit_max_new_tokens", False))
            if source_cap > args.minimum_token_cap and (
                count >= args.minimum_token_cap or capped
            ):
                raise ValueError(
                    f"A larger-cap row did not terminate before the common cap: {path}"
                )
        return path, generation, by_question

    normal_path, normal_metadata, normal = load_scored(
        args.normal_scored, expected_condition="normal"
    )
    condition_args = {
        "upstream_ablated": args.upstream_ablated_scored,
        "upstream_vector_rescue": args.vector_rescue_scored,
        "upstream_full_state_rescue": args.full_state_rescue_scored,
        "upstream_full_prefix_state_rescue": (
            args.full_prefix_state_rescue_scored
        ),
        "upstream_unrelated_vector_rescue": args.unrelated_rescue_scored,
        "vector_rescue_downstream_ablated": args.downstream_blocked_scored,
    }
    loaded = {
        condition: load_scored(path, expected_condition=condition)
        for condition, path in condition_args.items()
    }
    orthogonal_loaded = [
        load_scored(path, expected_condition="upstream_orthogonal_rescue")
        for path in args.orthogonal_rescue_scored
    ]
    if len(orthogonal_loaded) < 5:
        raise ValueError("Use at least five orthogonal rescue seeds.")
    factorial_loaded = {}
    if args.downstream_ablated_scored is not None:
        factorial_loaded = {
            "downstream_ablated": load_scored(
                args.downstream_ablated_scored,
                expected_condition="downstream_ablated",
            ),
            "both_ablated": load_scored(
                args.both_ablated_scored,
                expected_condition="both_ablated",
            ),
        }

    contract_keys = (
        "model",
        "model_architecture",
        "upstream_run",
        "downstream_run",
        "upstream_k",
        "downstream_k",
        "upstream_circuit_sha256",
        "downstream_circuit_sha256",
        "direction_sha256",
        "response_vector_sha256",
        "upstream_position_scope",
        "interface",
        "question_indices",
        "decoding",
        "generation_backend",
        "memory_implementation",
        "max_length",
    )
    reference_metadata = loaded["upstream_ablated"][1]
    reference_contract = {
        key: _generation_value(reference_metadata, key)
        for key in contract_keys
    }
    for _, metadata, _ in [
        *loaded.values(),
        *orthogonal_loaded,
        *factorial_loaded.values(),
    ]:
        mismatched = [
            key
            for key in contract_keys
            if _generation_value(metadata, key) != reference_contract[key]
        ]
        if mismatched:
            raise ValueError(
                f"Connected mediation artifacts disagree: {mismatched}"
            )
    if (
        _generation_value(reference_metadata, "upstream_position_scope")
        != "prompt_only"
        or (_generation_value(reference_metadata, "interface") or {}).get("timing")
        != "prompt_end"
    ):
        raise ValueError(
            "Primary bottleneck analysis requires prompt_only/prompt_end runs."
        )
    if _generation_value(normal_metadata, "method") != _generation_value(
        reference_metadata, "method"
    ):
        raise ValueError(
            "Use the normal condition from the same connected-circuit runner."
        )
    normal_mismatched = [
        key
        for key in contract_keys
        if _generation_value(normal_metadata, key) != reference_contract[key]
    ]
    if (
        normal_mismatched
        or _generation_value(normal_metadata, "decoding")
        != _generation_value(reference_metadata, "decoding")
    ):
        raise ValueError(
            "Normal baseline and mediation runs disagree: "
            f"{normal_mismatched}"
        )
    orthogonal_seeds = [
        int(_generation_value(metadata, "orthogonal_seed"))
        for _, metadata, _ in orthogonal_loaded
    ]
    if len(set(orthogonal_seeds)) != len(orthogonal_seeds):
        raise ValueError("Orthogonal rescue seeds must be unique.")

    questions = list(
        map(int, _generation_value(reference_metadata, "question_indices"))
    )
    question_set = set(questions)
    if set(normal) != question_set:
        raise ValueError("Normal baseline question set differs from mediation runs.")
    for _, _, rows in [
        *loaded.values(),
        *orthogonal_loaded,
        *factorial_loaded.values(),
    ]:
        if set(rows) != question_set:
            raise ValueError("A mediation condition has an incomplete question set.")
        for question in questions:
            if any(
                rows[question].get(key) != normal[question].get(key)
                for key in ("question", "prompt", "system_prompt")
            ):
                raise ValueError(
                    f"Prompt mismatch in mediation row q{question}."
                )

    def trait(rows: dict[int, dict]) -> np.ndarray:
        return np.asarray(
            [float(rows[q]["trait_score"]) for q in questions],
            dtype=float,
        )

    def coherence(rows: dict[int, dict]) -> np.ndarray:
        return np.asarray(
            [float(rows[q]["coherence_score"]) for q in questions],
            dtype=float,
        )

    normal_trait = trait(normal)
    cut_trait = trait(loaded["upstream_ablated"][2])
    vector_trait = trait(loaded["upstream_vector_rescue"][2])
    full_trait = trait(loaded["upstream_full_state_rescue"][2])
    full_prefix_trait = trait(
        loaded["upstream_full_prefix_state_rescue"][2]
    )
    unrelated_trait = trait(
        loaded["upstream_unrelated_vector_rescue"][2]
    )
    blocked_trait = trait(
        loaded["vector_rescue_downstream_ablated"][2]
    )
    downstream_only_trait = (
        trait(factorial_loaded["downstream_ablated"][2])
        if factorial_loaded
        else None
    )
    both_ablated_trait = (
        trait(factorial_loaded["both_ablated"][2])
        if factorial_loaded
        else None
    )
    orthogonal_traits = np.stack(
        [trait(rows) for _, _, rows in orthogonal_loaded]
    )
    orthogonal_trait = orthogonal_traits.mean(axis=0)

    denominator = float((normal_trait - cut_trait).mean())
    if denominator <= 0:
        raise ValueError("Upstream cut did not reduce mean trait behavior.")

    arrays = {
        "normal": normal_trait,
        "upstream_ablated": cut_trait,
        "vector_rescue": vector_trait,
        "full_state_rescue": full_trait,
        "full_prefix_state_rescue": full_prefix_trait,
        "orthogonal_rescue_mean": orthogonal_trait,
        "unrelated_rescue": unrelated_trait,
        "vector_rescue_downstream_ablated": blocked_trait,
    }
    if factorial_loaded:
        arrays.update(
            {
                "downstream_ablated": downstream_only_trait,
                "both_ablated": both_ablated_trait,
            }
        )
    condition_rows = []
    coherence_sources = {
        "normal": normal,
        "upstream_ablated": loaded["upstream_ablated"][2],
        "vector_rescue": loaded["upstream_vector_rescue"][2],
        "full_state_rescue": loaded["upstream_full_state_rescue"][2],
        "full_prefix_state_rescue": loaded[
            "upstream_full_prefix_state_rescue"
        ][2],
        "unrelated_rescue": loaded["upstream_unrelated_vector_rescue"][2],
        "vector_rescue_downstream_ablated": loaded[
            "vector_rescue_downstream_ablated"
        ][2],
    }
    if factorial_loaded:
        coherence_sources.update(
            {
                "downstream_ablated": factorial_loaded[
                    "downstream_ablated"
                ][2],
                "both_ablated": factorial_loaded["both_ablated"][2],
            }
        )
    for condition, values in arrays.items():
        if condition == "orthogonal_rescue_mean":
            coherence_values = np.stack(
                [coherence(rows) for _, _, rows in orthogonal_loaded]
            ).mean(axis=0)
            cap_hits = sum(
                bool(rows[q].get("hit_max_new_tokens", False))
                for _, _, rows in orthogonal_loaded
                for q in questions
            )
            generations = len(questions) * len(orthogonal_loaded)
            exact_matches = sum(
                rows[q].get("answer") == normal[q].get("answer")
                for _, _, rows in orthogonal_loaded
                for q in questions
            )
        else:
            source_rows = coherence_sources[condition]
            coherence_values = coherence(source_rows)
            cap_hits = sum(
                bool(source_rows[q].get("hit_max_new_tokens", False))
                for q in questions
            )
            generations = len(questions)
            exact_matches = sum(
                source_rows[q].get("answer") == normal[q].get("answer")
                for q in questions
            )
        condition_rows.append(
            {
                "condition": condition,
                "n_questions": len(questions),
                "n_generations": generations,
                "trait_mean": float(values.mean()),
                "trait_std": float(values.std(ddof=1)),
                "coherence_mean": float(coherence_values.mean()),
                "cap_hits": int(cap_hits),
                "normal_answer_exact_match_rate": (
                    exact_matches / generations
                ),
            }
        )

    def statistics(
        index: np.ndarray,
        *,
        orthogonal_sample: np.ndarray | None = None,
    ) -> dict[str, float]:
        normal_sample = normal_trait[index]
        cut_sample = cut_trait[index]
        vector_sample = vector_trait[index]
        if orthogonal_sample is None:
            orthogonal_sample = orthogonal_trait[index]
        cut_effect = float((normal_sample - cut_sample).mean())
        rescue_effect = float((vector_sample - cut_sample).mean())
        full_rescue = float((full_trait[index] - cut_sample).mean())
        full_prefix_rescue = float(
            (full_prefix_trait[index] - cut_sample).mean()
        )
        orthogonal_rescue = float(
            (orthogonal_sample - cut_sample).mean()
        )
        unrelated_rescue = float(
            (unrelated_trait[index] - cut_sample).mean()
        )
        downstream_blocking = float(
            (vector_sample - blocked_trait[index]).mean()
        )
        effects = {
            "upstream_cut_effect_points": cut_effect,
            "vector_rescue_points": rescue_effect,
            "full_state_rescue_points": full_rescue,
            "full_prefix_state_rescue_points": full_prefix_rescue,
            "orthogonal_rescue_points": orthogonal_rescue,
            "unrelated_rescue_points": unrelated_rescue,
            "vector_minus_orthogonal_points": float(
                (vector_sample - orthogonal_sample).mean()
            ),
            "vector_minus_unrelated_points": float(
                (vector_sample - unrelated_trait[index]).mean()
            ),
            "normal_minus_full_prefix_points": float(
                (normal_sample - full_prefix_trait[index]).mean()
            ),
            "full_prefix_minus_full_state_points": float(
                (full_prefix_trait[index] - full_trait[index]).mean()
            ),
            "full_state_minus_vector_points": float(
                (full_trait[index] - vector_sample).mean()
            ),
            "downstream_blocking_points": downstream_blocking,
        }
        if downstream_only_trait is not None and both_ablated_trait is not None:
            downstream_sample = downstream_only_trait[index]
            both_sample = both_ablated_trait[index]
            effects.update(
                {
                    "downstream_only_effect_points": float(
                        (normal_sample - downstream_sample).mean()
                    ),
                    "joint_ablation_effect_points": float(
                        (normal_sample - both_sample).mean()
                    ),
                    "factorial_interaction_trait_points": float(
                        (
                            both_sample
                            - cut_sample
                            - downstream_sample
                            + normal_sample
                        ).mean()
                    ),
                }
            )
        for name, numerator in (
            ("vector_rescue_fraction", rescue_effect),
            ("full_state_rescue_fraction", full_rescue),
            ("full_prefix_state_rescue_fraction", full_prefix_rescue),
            ("orthogonal_rescue_fraction", orthogonal_rescue),
            ("unrelated_rescue_fraction", unrelated_rescue),
        ):
            effects[name] = (
                numerator / cut_effect
                if cut_effect > 1e-8
                else float("nan")
            )
        effects["downstream_blocking_fraction"] = (
            downstream_blocking / rescue_effect
            if rescue_effect > 1e-8
            else float("nan")
        )
        return effects

    point = statistics(np.arange(len(questions)))
    rng = np.random.default_rng(0)
    bootstrap = {key: [] for key in point}
    for _ in range(args.bootstrap_samples):
        index = rng.integers(0, len(questions), size=len(questions))
        seed_index = rng.integers(
            0,
            len(orthogonal_loaded),
            size=len(orthogonal_loaded),
        )
        orthogonal_sample = orthogonal_traits[seed_index][
            :, index
        ].mean(axis=0)
        sample = statistics(
            index, orthogonal_sample=orthogonal_sample
        )
        for key, value in sample.items():
            if np.isfinite(value):
                bootstrap[key].append(value)
    effect_rows = []
    for name, value in point.items():
        samples = np.asarray(bootstrap[name], dtype=float)
        valid_fraction = len(samples) / args.bootstrap_samples
        stable = valid_fraction >= 0.95
        effect_rows.append(
            {
                "estimand": name,
                "estimate": value,
                "ci_low": (
                    float(np.quantile(samples, 0.025))
                    if stable
                    else None
                ),
                "ci_high": (
                    float(np.quantile(samples, 0.975))
                    if stable
                    else None
                ),
                "valid_bootstrap_fraction": valid_fraction,
                "stable_denominator": stable,
            }
        )

    vector_rows = loaded["upstream_vector_rescue"][2]
    clean_coordinate = np.asarray(
        [
            float(
                vector_rows[q]["initial_interface_diagnostic"][
                    "clean_target_coordinate"
                ]
            )
            for q in questions
        ]
    )
    cut_coordinate = np.asarray(
        [
            float(
                vector_rows[q]["initial_interface_diagnostic"][
                    "upstream_intervened_target_coordinate"
                ]
            )
            for q in questions
        ]
    )
    rescued_coordinate = np.asarray(
        [
            float(
                vector_rows[q]["initial_interface_diagnostic"][
                    "rescued_target_coordinate"
                ]
            )
            for q in questions
        ]
    )
    coordinate_gap = float((clean_coordinate - cut_coordinate).mean())
    if abs(coordinate_gap) <= 1e-8:
        raise ValueError("Interface coordinate denominator is degenerate.")
    coordinate_error = np.abs(rescued_coordinate - clean_coordinate)
    if not np.allclose(
        rescued_coordinate,
        clean_coordinate,
        rtol=1e-4,
        atol=1e-2,
    ):
        raise ValueError(
            "Vector rescue did not restore the intended interface coordinate."
        )
    full_prefix_rows = loaded["upstream_full_prefix_state_rescue"][2]
    full_prefix_exact_matches = sum(
        full_prefix_rows[q].get("answer") == normal[q].get("answer")
        and int(full_prefix_rows[q]["generated_token_count"])
        == int(normal[q]["generated_token_count"])
        and full_prefix_rows[q].get("generated_token_ids_sha256")
        == normal[q].get("generated_token_ids_sha256")
        and full_prefix_rows[q].get("generated_token_ids_sha256")
        is not None
        and bool(full_prefix_rows[q].get("hit_max_new_tokens", False))
        == bool(normal[q].get("hit_max_new_tokens", False))
        for q in questions
    )
    if full_prefix_exact_matches != len(questions):
        raise ValueError(
            "Full-prefix interface rescue did not exactly reproduce the "
            "same-backend normal generations."
        )
    interface_summary = {
        "clean_coordinate_mean": float(clean_coordinate.mean()),
        "cut_coordinate_mean": float(cut_coordinate.mean()),
        "rescued_coordinate_mean": float(rescued_coordinate.mean()),
        "coordinate_rescue_fraction": float(
            (rescued_coordinate - cut_coordinate).mean() / coordinate_gap
        ),
        "coordinate_rescue_max_abs_error": float(coordinate_error.max()),
        "full_prefix_normal_exact_matches": full_prefix_exact_matches,
        "full_prefix_normal_exact_match_rate": (
            full_prefix_exact_matches / len(questions)
        ),
    }

    output = _resolve(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results.")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(condition_rows).to_csv(
        output / "condition_summary.csv", index=False
    )
    pd.DataFrame(effect_rows).to_csv(
        output / "mediation_effects.csv", index=False
    )
    artifacts = {
        "normal": normal_path,
        **{
            condition: path
            for condition, (path, _, _) in loaded.items()
        },
        **{
            (
                "orthogonal_seed_"
                f"{int(_generation_value(metadata, 'orthogonal_seed'))}"
            ): path
            for path, metadata, _ in orthogonal_loaded
        },
        **{
            condition: path
            for condition, (path, _, _) in factorial_loaded.items()
        },
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "claim_scope": (
            "prompt-formed low-dimensional causal bottleneck mediation"
        ),
        "behavior_protocol_id": config["protocol_id"],
        "question_indices": questions,
        "minimum_token_cap": args.minimum_token_cap,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": (
            "paired question identities; orthogonal controls additionally "
            "resample direction seeds"
        ),
        "contract": reference_contract,
        "normal_metadata": {
            "model": _generation_value(normal_metadata, "model"),
            "decoding": _generation_value(normal_metadata, "decoding"),
            "method": _generation_value(normal_metadata, "method"),
        },
        "orthogonal_seeds": sorted(orthogonal_seeds),
        "interface_summary": interface_summary,
        "effects": effect_rows,
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
        "tables": ["condition_summary.csv", "mediation_effects.csv"],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

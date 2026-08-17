#!/usr/bin/env python3
"""Build provenance-gated end-to-end persona circuit curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def _bootstrap_internal_metrics(frame, *, samples: int, seed: int) -> dict:
    import numpy as np

    values = {
        name: frame[name].to_numpy(dtype=float)
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    denominator = float(
        values["clean"].mean() - values["corrupt"].mean()
    )
    if denominator <= 1e-8:
        raise ValueError("End-to-end internal denominator is not positive.")
    point = {
        "faithfulness": float(
            (values["circuit_only"].mean() - values["corrupt"].mean())
            / denominator
        ),
        "necessity": float(
            (values["clean"].mean() - values["circuit_ablated"].mean())
            / denominator
        ),
    }
    rng = np.random.default_rng(seed)
    estimates = {name: [] for name in point}
    for _ in range(samples):
        index = rng.integers(0, len(frame), size=len(frame))
        sampled_denominator = float(
            values["clean"][index].mean()
            - values["corrupt"][index].mean()
        )
        if sampled_denominator <= 1e-8:
            continue
        estimates["faithfulness"].append(
            float(
                (
                    values["circuit_only"][index].mean()
                    - values["corrupt"][index].mean()
                )
                / sampled_denominator
            )
        )
        estimates["necessity"].append(
            float(
                (
                    values["clean"][index].mean()
                    - values["circuit_ablated"][index].mean()
                )
                / sampled_denominator
            )
        )
    output = dict(point)
    for name, samples_for_metric in estimates.items():
        samples_for_metric = np.asarray(samples_for_metric, dtype=float)
        valid_fraction = len(samples_for_metric) / samples
        output[f"{name}_bootstrap_valid_fraction"] = valid_fraction
        output[f"{name}_ci_low"] = (
            float(np.quantile(samples_for_metric, 0.025))
            if valid_fraction >= 0.95
            else float("nan")
        )
        output[f"{name}_ci_high"] = (
            float(np.quantile(samples_for_metric, 0.975))
            if valid_fraction >= 0.95
            else float("nan")
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=(500, 1000, 3000, 5000, 10000, 20000),
    )
    parser.add_argument(
        "--behavior-k-values",
        type=int,
        nargs="*",
        default=(),
        help="Selected k values with complete judged free-generation artifacts.",
    )
    parser.add_argument(
        "--control-k-values",
        type=int,
        nargs="*",
        default=(),
        help="Selected k values with five topology-control evaluations.",
    )
    parser.add_argument(
        "--control-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4)
    )
    parser.add_argument(
        "--control-name", default="topology_stratified_v2"
    )
    parser.add_argument(
        "--control-evaluation-template",
        default="control_topology_v2_seed{seed}_k{k}_n32",
    )
    parser.add_argument(
        "--circuit-directory", default="circuits_source_to_sink"
    )
    parser.add_argument(
        "--evaluation-template",
        default="k{k}_n32_prompt_disjoint",
    )
    parser.add_argument(
        "--question-indices", type=int, nargs="+", default=tuple(range(20))
    )
    parser.add_argument("--minimum-token-cap", type=int, default=350)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    if len(args.k_values) != len(set(args.k_values)):
        raise ValueError("k values must be unique.")
    if not set(args.behavior_k_values) <= set(args.k_values):
        raise ValueError("Behavior k values must be included in --k-values.")
    if not set(args.control_k_values) <= set(args.k_values):
        raise ValueError("Control k values must be included in --k-values.")
    if len(args.control_seeds) != len(set(args.control_seeds)):
        raise ValueError("Control seeds must be unique.")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        behavioral_circuit_summary,
        read_jsonl,
        sha256_file,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    discovery = result_root / "circuit_discovery_end_to_end" / args.run_name
    behavior_root = intermediate / "end_to_end_circuit_behavior" / args.run_name
    output = result_root / (
        args.output_name or f"end_to_end_curves_{args.run_name}"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results.")
    output.mkdir(parents=True, exist_ok=True)
    wanted_questions = set(args.question_indices)

    def metadata_value(metadata: dict, key: str):
        if key in metadata:
            return metadata[key]
        return metadata.get("compatibility", {}).get(key)

    discovery_metadata_path = discovery / "eapig_run_metadata.json"
    discovery_metadata = json.loads(
        discovery_metadata_path.read_text(encoding="utf-8")
    )
    if discovery_metadata.get("status") != "complete":
        raise ValueError("End-to-end discovery is not complete.")
    eligible_edges = int(discovery_metadata["eligible_edges"])

    def circuit_path(k: int) -> Path:
        return (
            discovery
            / args.circuit_directory
            / f"end_to_end_eapig_top_{k}.json"
        )

    def validate_scored(path: Path) -> dict:
        responses_path = path.with_name("responses.jsonl")
        generation_metadata_path = path.with_name("metadata.json")
        judge_metadata_path = path.with_suffix(".metadata.json")
        for required in (
            responses_path,
            generation_metadata_path,
            judge_metadata_path,
        ):
            if not required.exists():
                raise FileNotFoundError(f"Missing behavior provenance: {required}")
        generation = json.loads(
            generation_metadata_path.read_text(encoding="utf-8")
        )
        judge = json.loads(judge_metadata_path.read_text(encoding="utf-8"))
        recorded_outputs = {
            item["path"]: item["sha256"]
            for item in generation.get("outputs", [])
        }
        if (
            generation.get("status") != "complete"
            or recorded_outputs.get(str(responses_path.relative_to(ROOT)))
            != sha256_file(responses_path)
            or judge.get("status") != "complete"
            or judge.get("input_sha256") != sha256_file(responses_path)
            or judge.get("output_sha256") != sha256_file(path)
        ):
            raise ValueError(f"Behavior provenance mismatch: {path}")
        return generation

    internal_rows = []
    for k in args.k_values:
        circuit = circuit_path(k)
        aggregate_path = (
            discovery
            / "evaluation"
            / args.evaluation_template.format(k=k)
            / "aggregate.json"
        )
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if (
            aggregate.get("status") != "complete"
            or int(aggregate.get("requested_k", -1)) != k
            or aggregate.get("circuit") != str(circuit.relative_to(ROOT))
            or aggregate.get("circuit_sha256") != sha256_file(circuit)
            or aggregate.get("truncation") is not False
        ):
            raise ValueError(f"Internal evaluation mismatch at k={k}.")
        payload = json.loads(circuit.read_text(encoding="utf-8"))
        retained = int(payload["retained_edges"])
        instances_path = aggregate_path.with_name("instance_metrics.csv")
        uncertainty = _bootstrap_internal_metrics(
            pd.read_csv(instances_path),
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + k,
        )
        if (
            not np.isclose(
                uncertainty["faithfulness"],
                float(aggregate["faithfulness"]),
                atol=1e-5,
            )
            or not np.isclose(
                uncertainty["necessity"],
                float(aggregate["necessity"]),
                atol=1e-5,
            )
        ):
            raise ValueError(
                f"End-to-end aggregate metrics disagree at k={k}."
            )
        internal_rows.append(
            {
                "k": k,
                "circuit_family": payload.get("circuit_family"),
                "retained_edges": retained,
                "eligible_edges": eligible_edges,
                "edge_fraction": retained / eligible_edges,
                "n": int(aggregate["n_prompts"]),
                "clean": float(aggregate["means"]["clean"]),
                "corrupt": float(aggregate["means"]["corrupt"]),
                "circuit_only": float(aggregate["means"]["circuit_only"]),
                "circuit_ablated": float(
                    aggregate["means"]["circuit_ablated"]
                ),
                "max_length": int(aggregate["max_length"]),
                "max_observed_tokens": int(
                    aggregate["max_observed_tokens"]
                ),
                **uncertainty,
                "artifact": str(aggregate_path.relative_to(ROOT)),
                "artifact_sha256": sha256_file(aggregate_path),
            }
        )
    internal = pd.DataFrame(internal_rows).sort_values("retained_edges")
    internal_numeric = (
        "edge_fraction",
        "clean",
        "corrupt",
        "circuit_only",
        "circuit_ablated",
        "faithfulness",
        "necessity",
    )
    if not np.isfinite(
        internal[list(internal_numeric)].to_numpy(dtype=float)
    ).all():
        raise ValueError("Internal end-to-end curve contains non-finite metrics.")

    control_rows = []
    for k in args.control_k_values:
        for seed in args.control_seeds:
            control_circuit = (
                discovery
                / "controls"
                / args.control_name
                / "source_to_sink"
                / f"seed_{seed}"
                / f"end_to_end_eapig_top_{k}.json"
            )
            aggregate_path = (
                discovery
                / "evaluation"
                / args.control_evaluation_template.format(seed=seed, k=k)
                / "aggregate.json"
            )
            instances_path = aggregate_path.with_name("instance_metrics.csv")
            aggregate = json.loads(
                aggregate_path.read_text(encoding="utf-8")
            )
            payload = json.loads(
                control_circuit.read_text(encoding="utf-8")
            )
            if (
                payload.get("method")
                != "topology_stratified_score_permutation_control"
                or aggregate.get("status") != "complete"
                or aggregate.get("circuit")
                != str(control_circuit.relative_to(ROOT))
                or aggregate.get("circuit_sha256")
                != sha256_file(control_circuit)
            ):
                raise ValueError(
                    f"End-to-end control mismatch at k={k}, seed={seed}."
                )
            uncertainty = _bootstrap_internal_metrics(
                pd.read_csv(instances_path),
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + k + seed + 100000,
            )
            control_rows.append(
                {
                    "k": k,
                    "seed": seed,
                    "control_frontier_k": int(
                        payload["control_frontier_k"]
                    ),
                    "target_retained_edges": int(
                        payload["target_retained_edges"]
                    ),
                    "retained_edges": int(payload["retained_edges"]),
                    "retained_size_difference": int(
                        payload["retained_size_difference"]
                    ),
                    **uncertainty,
                    "circuit": str(control_circuit.relative_to(ROOT)),
                    "evaluation": str(aggregate_path.relative_to(ROOT)),
                }
            )
    controls = pd.DataFrame(control_rows)
    behavior_rows = []
    if args.behavior_k_values:
        def validate_common_cap(
            row: dict, metadata: dict, *, artifact: Path
        ) -> None:
            decoding = metadata.get("decoding", {})
            if decoding.get("do_sample") is not False:
                raise ValueError(
                    "Only deterministic greedy behavior may be pooled: "
                    f"{artifact}"
                )
            source_cap = int(decoding.get("max_new_tokens", 0))
            if source_cap < args.minimum_token_cap:
                raise ValueError(
                    f"Behavior cap {source_cap} is below the common cap "
                    f"{args.minimum_token_cap}: {artifact}"
                )
            generated = int(row["generated_token_count"])
            capped = bool(row.get("hit_max_new_tokens", False))
            if source_cap > args.minimum_token_cap and (
                generated >= args.minimum_token_cap or capped
            ):
                raise ValueError(
                    f"q{row['question_index']} from a larger-cap run did not "
                    f"terminate before the common cap: {artifact}"
                )

        rng = np.random.default_rng(0)
        for k in args.behavior_k_values:
            circuit = circuit_path(k)
            rows_by_condition = {
                "normal": {},
                "negative_prompt": {},
                "circuit_ablated": {},
                "circuit_only": {},
            }
            artifacts: dict[str, list[str]] = {
                condition: [] for condition in rows_by_condition
            }
            compatibility = None
            for path in sorted(behavior_root.glob(f"k{k}_*/scored.jsonl")):
                metadata = validate_scored(path)
                if (
                    int(metadata.get("decoding", {}).get("max_new_tokens", 0))
                    < args.minimum_token_cap
                    or set(map(int, metadata.get("question_indices", [])))
                    != wanted_questions
                    or metadata_value(metadata, "circuit")
                    != str(circuit.relative_to(ROOT))
                    or metadata_value(metadata, "circuit_sha256")
                    != sha256_file(circuit)
                ):
                    continue
                decoding = dict(metadata.get("decoding", {}))
                decoding.pop("max_new_tokens", None)
                current_compatibility = {
                    "behavior_protocol_id": metadata_value(
                        metadata, "behavior_protocol_id"
                    ),
                    "model": metadata_value(metadata, "model"),
                    "model_architecture": metadata_value(
                        metadata, "model_architecture"
                    ),
                    "eval_artifact_sha256": metadata_value(
                        metadata, "eval_artifact_sha256"
                    ),
                    "discovery_metadata_sha256": metadata_value(
                        metadata, "discovery_metadata_sha256"
                    ),
                    "circuit_sha256": metadata_value(
                        metadata, "circuit_sha256"
                    ),
                    "requested_k": metadata_value(metadata, "requested_k"),
                    "max_length": metadata_value(metadata, "max_length"),
                    "decoding_without_cap": decoding,
                }
                if compatibility is None:
                    compatibility = current_compatibility
                elif current_compatibility != compatibility:
                    raise ValueError(
                        "End-to-end behavior artifacts use incompatible "
                        f"generation contracts at k={k}: {path}"
                    )
                for row in read_jsonl(path):
                    condition = row.get("condition")
                    question = int(row["question_index"])
                    if (
                        condition not in rows_by_condition
                        or question not in wanted_questions
                    ):
                        continue
                    validate_common_cap(row, metadata, artifact=path)
                    old = rows_by_condition[condition].get(question)
                    if old is not None and any(
                        old.get(key) != row.get(key)
                        for key in (
                            "answer",
                            "trait_score",
                            "coherence_score",
                            "generated_token_ids_sha256",
                        )
                    ):
                        raise ValueError(
                            f"Conflicting {condition} behavior at k={k}, q={question}."
                        )
                    rows_by_condition[condition][question] = row
                    artifact = str(path.relative_to(ROOT))
                    if artifact not in artifacts[condition]:
                        artifacts[condition].append(artifact)
            for condition, rows in rows_by_condition.items():
                if set(rows) != wanted_questions:
                    raise RuntimeError(
                        f"Incomplete same-run {condition} behavior at k={k}."
                    )
            for question in args.question_indices:
                normal_row = rows_by_condition["normal"][question]
                for condition, rows in rows_by_condition.items():
                    if any(
                        rows[question].get(key) != normal_row.get(key)
                        for key in ("question", "prompt", "system_prompt")
                    ):
                        raise ValueError(
                            f"Prompt mismatch for {condition} at k={k}, "
                            f"q={question}."
                        )

            normal = np.asarray(
                [
                    float(rows_by_condition["normal"][q]["trait_score"])
                    for q in args.question_indices
                ],
                dtype=float,
            )
            negative = np.asarray(
                [
                    float(
                        rows_by_condition["negative_prompt"][q]["trait_score"]
                    )
                    for q in args.question_indices
                ],
                dtype=float,
            )
            for regime, condition in (
                ("necessity", "circuit_ablated"),
                ("sufficiency", "circuit_only"),
            ):
                rows = rows_by_condition[condition]
                intervened = np.array(
                    [float(rows[q]["trait_score"]) for q in args.question_indices]
                )
                coherence = np.array(
                    [float(rows[q]["coherence_score"]) for q in args.question_indices]
                )
                tokens = np.array(
                    [int(rows[q]["generated_token_count"]) for q in args.question_indices]
                )
                cap_hits = np.array(
                    [
                        bool(rows[q].get("hit_max_new_tokens", False))
                        for q in args.question_indices
                    ]
                )
                summary = behavioral_circuit_summary(
                    normal.tolist(),
                    negative.tolist(),
                    intervened.tolist(),
                    regime=regime,
                )
                effect = (
                    normal - intervened
                    if regime == "necessity"
                    else intervened - negative
                )
                additive_bootstrap = []
                normalized_bootstrap = []
                for _ in range(10000):
                    index = rng.integers(0, len(effect), size=len(effect))
                    additive_bootstrap.append(float(effect[index].mean()))
                    try:
                        sampled = behavioral_circuit_summary(
                            normal[index].tolist(),
                            negative[index].tolist(),
                            intervened[index].tolist(),
                            regime=regime,
                        )
                    except ValueError:
                        continue
                    normalized_bootstrap.append(sampled["normalized_effect"])
                additive_bootstrap = np.asarray(additive_bootstrap)
                normalized_bootstrap = np.asarray(normalized_bootstrap)
                normalized_valid_fraction = (
                    len(normalized_bootstrap) / 10000
                )
                normalized_ci = (
                    (
                        float(np.quantile(normalized_bootstrap, 0.025)),
                        float(np.quantile(normalized_bootstrap, 0.975)),
                    )
                    if normalized_valid_fraction >= 0.95
                    else (float("nan"), float("nan"))
                )
                high = normal >= 50
                negative_low = negative < 50
                behavior_rows.append(
                    {
                        "k": k,
                        "retained_edges": int(
                            json.loads(circuit.read_text())["retained_edges"]
                        ),
                        "regime": regime,
                        "n": len(effect),
                        **summary,
                        "effect_ci_low": np.quantile(
                            additive_bootstrap, 0.025
                        ),
                        "effect_ci_high": np.quantile(
                            additive_bootstrap, 0.975
                        ),
                        "normalized_ci_low": normalized_ci[0],
                        "normalized_ci_high": normalized_ci[1],
                        "normalized_bootstrap_valid_fraction": (
                            normalized_valid_fraction
                        ),
                        "normal_high_count": int(high.sum()),
                        "high_to_low_count": int(
                            ((intervened < 50) & high).sum()
                        ),
                        "high_retained_count": int(
                            ((intervened >= 50) & high).sum()
                        ),
                        "negative_low_count": int(negative_low.sum()),
                        "negative_low_to_high_count": int(
                            ((intervened >= 50) & negative_low).sum()
                        ),
                        "coherence_mean": coherence.mean(),
                        "generated_tokens_mean": tokens.mean(),
                        "generated_tokens_max": tokens.max(),
                        "hit_max_new_tokens_count": int(cap_hits.sum()),
                        "artifacts": json.dumps(
                            artifacts, sort_keys=True
                        ),
                    }
                )
    behavior = pd.DataFrame(behavior_rows)
    if not behavior.empty:
        behavior_numeric = (
            "normal_trait_mean",
            "negative_trait_mean",
            "intervened_trait_mean",
            "normal_minus_negative_points",
            "additive_effect_points",
            "normalized_effect",
            "effect_ci_low",
            "effect_ci_high",
            "normalized_bootstrap_valid_fraction",
            "coherence_mean",
            "generated_tokens_mean",
            "generated_tokens_max",
        )
        if not np.isfinite(
            behavior[list(behavior_numeric)].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "End-to-end behavior curve contains non-finite metrics."
            )
    internal.to_csv(output / "internal_end_to_end.csv", index=False)
    controls.to_csv(output / "topology_controls.csv", index=False)
    behavior.to_csv(output / "behavior_end_to_end.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    axes[0].plot(internal["retained_edges"], internal["faithfulness"], "o-")
    axes[0].set(
        title="End-to-end faithfulness",
        xlabel="Retained edges",
        ylabel="Normalized faithfulness",
    )
    axes[1].plot(internal["retained_edges"], internal["necessity"], "o-")
    axes[1].set(
        title="End-to-end necessity",
        xlabel="Retained edges",
        ylabel="Normalized necessity",
    )
    if not behavior.empty:
        for regime, frame in behavior.groupby("regime"):
            axes[2].plot(
                frame["retained_edges"],
                frame["normalized_effect"],
                "o-",
                label=regime,
            )
        axes[2].legend(fontsize=8)
    axes[2].set(
        title="Held-out generation",
        xlabel="Retained edges",
        ylabel="Normalized behavioral effect",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "end_to_end_curves.pdf", bbox_inches="tight")
    fig.savefig(output / "end_to_end_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_name": args.run_name,
        "discovery_metadata": str(discovery_metadata_path.relative_to(ROOT)),
        "discovery_metadata_sha256": sha256_file(discovery_metadata_path),
        "circuit_directory": args.circuit_directory,
        "evaluation_template": args.evaluation_template,
        "k_values": args.k_values,
        "behavior_k_values": args.behavior_k_values,
        "control_k_values": args.control_k_values,
        "control_name": args.control_name,
        "control_seeds": args.control_seeds,
        "question_indices": args.question_indices,
        "minimum_token_cap": args.minimum_token_cap,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "tables": [
            "internal_end_to_end.csv",
            "topology_controls.csv",
            "behavior_end_to_end.csv",
        ],
        "figures": ["end_to_end_curves.pdf", "end_to_end_curves.png"],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

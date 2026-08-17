#!/usr/bin/env python3
"""Build final necessity and behavioral-sufficiency circuit-size curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--upstream-run", required=True)
    parser.add_argument("--downstream-run", required=True)
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=(50, 100, 250, 500, 1000, 2000, 3000)
    )
    parser.add_argument(
        "--upstream-extra-k-values", type=int, nargs="*", default=(5000, 10000, 20000)
    )
    parser.add_argument(
        "--sufficiency-k-values", type=int, nargs="+", default=(50, 250, 1000, 3000)
    )
    parser.add_argument(
        "--question-indices", type=int, nargs="+", default=tuple(range(20))
    )
    parser.add_argument(
        "--upstream-circuit-directory", default="circuits_source_to_sink"
    )
    parser.add_argument(
        "--downstream-circuit-directory", default="circuits_source_to_sink"
    )
    parser.add_argument(
        "--upstream-evaluation-template",
        default="connected_k{k}_n32_prompt_disjoint",
    )
    parser.add_argument(
        "--downstream-evaluation-template",
        default="connected_k{k}_n32_seed1",
    )
    parser.add_argument("--minimum-token-cap", type=int, default=640)
    parser.add_argument(
        "--baseline-scored",
        type=Path,
        default=None,
        help=(
            "Judged normal-condition artifact. Defaults to the historical "
            "behavioral-pilot baseline when it exists."
        ),
    )
    parser.add_argument("--output-name", default="paper_suite_final_causal_curves")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    upstream_root = (
        result_root / "circuit_discovery_upstream_contrastive" / args.upstream_run
    )
    downstream_root = (
        result_root / "circuit_discovery_steering_path" / args.downstream_run
    )
    connected_root = (
        intermediate
        / "connected_circuit_behavior"
        / f"{args.upstream_run}__{args.downstream_run}"
    )
    output = result_root / args.output_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} already contains results; use a new --output-name."
        )
    output.mkdir(parents=True, exist_ok=True)
    wanted_questions = set(args.question_indices)
    all_upstream_k = list(dict.fromkeys(args.k_values + args.upstream_extra_k_values))

    def metadata_value(metadata: dict, key: str):
        if key in metadata:
            return metadata[key]
        return metadata.get("compatibility", {}).get(key)

    def circuit_path(scope: str, k: int) -> Path:
        if scope == "upstream":
            return (
                upstream_root
                / args.upstream_circuit_directory
                / f"eapig_top_{k}.json"
            )
        return (
            downstream_root
            / args.downstream_circuit_directory
            / f"steering_path_eapig_top_{k}.json"
        )

    def retained(scope: str, k: int) -> int:
        return int(json.loads(circuit_path(scope, k).read_text())["retained_edges"])

    def validate_scored_artifact(path: Path) -> tuple[dict, dict]:
        judge_metadata_path = path.with_suffix(".metadata.json")
        generation_metadata_path = path.with_name("metadata.json")
        responses_path = path.with_name("responses.jsonl")
        for required in (
            judge_metadata_path,
            generation_metadata_path,
            responses_path,
        ):
            if not required.exists():
                raise FileNotFoundError(f"Missing scored-artifact provenance: {required}")
        judge_metadata = json.loads(
            judge_metadata_path.read_text(encoding="utf-8")
        )
        generation_metadata = json.loads(
            generation_metadata_path.read_text(encoding="utf-8")
        )
        if (
            judge_metadata.get("status") != "complete"
            or judge_metadata.get("output_sha256") != sha256_file(path)
            or judge_metadata.get("input_sha256") != sha256_file(responses_path)
        ):
            raise ValueError(f"Judged artifact provenance mismatch: {path}")
        recorded_outputs = {
            item["path"]: item["sha256"]
            for item in generation_metadata.get("outputs", [])
        }
        if (
            generation_metadata.get("status") != "complete"
            or recorded_outputs.get(str(responses_path.relative_to(ROOT)))
            != sha256_file(responses_path)
        ):
            raise ValueError(f"Generation artifact provenance mismatch: {responses_path}")
        return generation_metadata, judge_metadata

    internal_rows = []
    for scope, root, k_values, eligible_key, count_key in (
        ("upstream", upstream_root, all_upstream_k, "eligible_edges", "n_pairs"),
        (
            "downstream",
            downstream_root,
            args.k_values,
            "eligible_downstream_edges",
            "n_prompts",
        ),
    ):
        discovery = json.loads((root / "eapig_run_metadata.json").read_text())
        if discovery.get("status") != "complete":
            raise ValueError(f"Incomplete discovery metadata: {root}")
        eligible = int(discovery[eligible_key])
        for k in k_values:
            evaluation_template = (
                args.upstream_evaluation_template
                if scope == "upstream"
                else args.downstream_evaluation_template
            )
            path = (
                root
                / "evaluation"
                / evaluation_template.format(k=k)
                / "aggregate.json"
            )
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            expected_circuit = circuit_path(scope, k)
            if (
                row.get("status") != "complete"
                or int(row.get("requested_k", -1)) != k
                or row.get("circuit")
                != str(expected_circuit.relative_to(ROOT))
                or row.get("circuit_sha256") != sha256_file(expected_circuit)
                or row.get("truncation") is not False
            ):
                raise ValueError(
                    f"Internal evaluation does not match {scope} k={k} "
                    f"circuit family: {path}"
                )
            circuit_payload = json.loads(expected_circuit.read_text())
            internal_rows.append(
                {
                    "scope": scope,
                    "k": k,
                    "circuit_family": circuit_payload.get("circuit_family"),
                    "circuit_edges": retained(scope, k),
                    "eligible_edges": eligible,
                    "edge_fraction": retained(scope, k) / eligible,
                    "faithfulness": float(row["faithfulness"]),
                    "necessity": float(row["necessity"]),
                    "clean": float(row["means"]["clean"]),
                    "corrupt": float(row["means"]["corrupt"]),
                    "circuit_only": float(row["means"]["circuit_only"]),
                    "circuit_ablated": float(row["means"]["circuit_ablated"]),
                    "max_length": int(row["max_length"]),
                    "max_observed_tokens": int(row["max_observed_tokens"]),
                    "n": int(row[count_key]),
                    "artifact": str(path.relative_to(ROOT)),
                    "artifact_sha256": sha256_file(path),
                }
            )
    internal = pd.DataFrame(internal_rows).sort_values(["scope", "k"])
    internal_numeric = (
        "edge_fraction",
        "faithfulness",
        "necessity",
        "clean",
        "corrupt",
        "circuit_only",
        "circuit_ablated",
    )
    if not np.isfinite(
        internal[list(internal_numeric)].to_numpy(dtype=float)
    ).all():
        raise ValueError("Internal causal curve contains non-finite metrics.")
    expected_internal = {
        *(("upstream", k) for k in all_upstream_k),
        *(("downstream", k) for k in args.k_values),
    }
    observed_internal = set(zip(internal["scope"], internal["k"]))
    missing_internal = sorted(expected_internal - observed_internal)
    if missing_internal:
        raise RuntimeError(f"incomplete internal curve; missing {missing_internal}")

    baseline_path = args.baseline_scored or (
        intermediate / "behavioral_pilot/k3000_n20/scored.jsonl"
    )
    baseline_path = baseline_path if baseline_path.is_absolute() else ROOT / baseline_path
    if not baseline_path.exists():
        raise FileNotFoundError(
            "No judged normal baseline; pass --baseline-scored from any complete "
            "upstream/downstream behavioral artifact."
        )
    def validate_common_cap(
        row: dict, metadata: dict, *, artifact: Path
    ) -> None:
        decoding = metadata.get("decoding", {})
        if decoding.get("do_sample") is not False:
            raise ValueError(
                f"Only deterministic greedy behavior may be pooled: {artifact}"
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

    baseline_metadata, _ = validate_scored_artifact(baseline_path)
    baseline = {}
    for row in read_jsonl(baseline_path):
        question = int(row["question_index"])
        if row["condition"] != "normal" or question not in wanted_questions:
            continue
        validate_common_cap(
            row, baseline_metadata, artifact=baseline_path
        )
        baseline[question] = row
    if set(baseline) != wanted_questions:
        raise ValueError("Frozen normal baseline is incomplete.")

    def complete_scored_paths(scope: str, upstream_k: int, downstream_k: int | None):
        if scope == "upstream":
            paths = (intermediate / "natural_upstream_circuit_behavior" / args.upstream_run).glob(
                f"k{upstream_k}_*/scored.jsonl"
            )
        elif scope == "downstream":
            paths = (intermediate / "steering_circuit_behavior" / args.downstream_run).glob(
                f"k{upstream_k}_*/scored.jsonl"
            )
        else:
            paths = (connected_root / f"ku{upstream_k}_kd{downstream_k}").glob(
                "*/scored.jsonl"
            )
        accepted = []
        for path in sorted(paths):
            metadata_path = path.with_name("metadata.json")
            if not metadata_path.exists():
                continue
            metadata, _ = validate_scored_artifact(path)
            expected_circuits = {}
            if scope == "upstream":
                expected_circuits = {
                    "circuit": circuit_path("upstream", upstream_k)
                }
            elif scope == "downstream":
                expected_circuits = {
                    "circuit": circuit_path("downstream", upstream_k)
                }
            else:
                expected_circuits = {
                    "upstream_circuit": circuit_path("upstream", upstream_k),
                    "downstream_circuit": circuit_path(
                        "downstream", int(downstream_k)
                    ),
                }
            circuit_match = all(
                metadata_value(metadata, key)
                == str(expected_path.relative_to(ROOT))
                and metadata_value(metadata, f"{key}_sha256")
                == sha256_file(expected_path)
                for key, expected_path in expected_circuits.items()
            )
            if (
                metadata.get("status") == "complete"
                and int(metadata.get("decoding", {}).get("max_new_tokens", 0))
                >= args.minimum_token_cap
                and set(map(int, metadata.get("question_indices", [])))
                == wanted_questions
                and circuit_match
            ):
                accepted.append(path)
        return accepted

    def condition_rows(
        scope: str,
        upstream_k: int,
        downstream_k: int | None,
        condition: str,
    ):
        found = {}
        artifacts = []
        for path in complete_scored_paths(scope, upstream_k, downstream_k):
            metadata, _ = validate_scored_artifact(path)
            rows = [row for row in read_jsonl(path) if row["condition"] == condition]
            if not rows:
                continue
            artifacts.append(str(path.relative_to(ROOT)))
            for row in rows:
                question = int(row["question_index"])
                if question not in wanted_questions:
                    continue
                validate_common_cap(row, metadata, artifact=path)
                if question in found:
                    old = found[question]
                    comparable = ("answer", "trait_score", "coherence_score")
                    if any(old.get(key) != row.get(key) for key in comparable):
                        raise ValueError(
                            f"Conflicting {scope} {condition} q{question} artifacts."
                        )
                found[question] = row
        return found, artifacts

    rng = np.random.default_rng(0)

    def summarize(
        *,
        scope: str,
        upstream_k: int,
        downstream_k: int | None,
        regime: str,
        condition: str,
    ):
        rows, artifacts = condition_rows(
            scope, upstream_k, downstream_k, condition
        )
        if set(rows) != wanted_questions:
            return None
        normal = np.array([float(baseline[q]["trait_score"]) for q in args.question_indices])
        intervened = np.array([float(rows[q]["trait_score"]) for q in args.question_indices])
        coherence = np.array([float(rows[q]["coherence_score"]) for q in args.question_indices])
        tokens = np.array([int(rows[q]["generated_token_count"]) for q in args.question_indices])
        caps = np.array([bool(rows[q].get("hit_max_new_tokens", False)) for q in args.question_indices])
        effect = normal - intervened
        boot = np.array(
            [rng.choice(effect, size=len(effect), replace=True).mean() for _ in range(10000)]
        )
        high = normal >= 50
        crossing = ((intervened < 50) if regime == "necessity" else (intervened >= 50)) & high
        effective_downstream_k = (
            upstream_k if scope == "downstream" else downstream_k
        )
        downstream_edges = (
            retained("downstream", effective_downstream_k)
            if effective_downstream_k is not None
            else 0
        )
        return {
            "scope": scope,
            "regime": regime,
            "upstream_k": upstream_k if scope != "downstream" else 0,
            "downstream_k": effective_downstream_k or 0,
            "circuit_edges": (
                retained("upstream", upstream_k) if scope != "downstream" else 0
            ) + downstream_edges,
            "n": len(effect),
            "normal_trait_mean": normal.mean(),
            "intervened_trait_mean": intervened.mean(),
            "normal_minus_intervened_points": effect.mean(),
            "effect_ci_low": np.quantile(boot, 0.025),
            "effect_ci_high": np.quantile(boot, 0.975),
            (
                "high_to_low_count" if regime == "necessity" else "high_retained_count"
            ): int(crossing.sum()),
            "normal_high_count": int(high.sum()),
            "coherence_mean": coherence.mean(),
            "generated_tokens_mean": tokens.mean(),
            "generated_tokens_max": tokens.max(),
            "hit_max_new_tokens_count": int(caps.sum()),
            "artifacts": json.dumps(artifacts),
        }

    behavior_rows = []
    for scope, k_values in (
        ("upstream", all_upstream_k),
        ("downstream", args.k_values),
    ):
        for k in k_values:
            for regime, condition in (
                ("necessity", "circuit_ablated"),
                ("sufficiency", "circuit_only"),
            ):
                row = summarize(
                    scope=scope,
                    upstream_k=k,
                    downstream_k=None,
                    regime=regime,
                    condition=condition,
                )
                if row is not None:
                    behavior_rows.append(row)

    union_pairs = [(k, k) for k in args.k_values]
    union_pairs += [(k, max(args.sufficiency_k_values)) for k in args.upstream_extra_k_values]
    for upstream_k, downstream_k in union_pairs:
        for regime, condition in (
            ("necessity", "both_ablated"),
            ("sufficiency", "both_circuit_only"),
        ):
            row = summarize(
                scope="union",
                upstream_k=upstream_k,
                downstream_k=downstream_k,
                regime=regime,
                condition=condition,
            )
            if row is not None:
                behavior_rows.append(row)
    behavior = pd.DataFrame(behavior_rows).sort_values(
        ["scope", "regime", "circuit_edges"]
    )
    behavior_numeric = (
        "normal_trait_mean",
        "intervened_trait_mean",
        "normal_minus_intervened_points",
        "effect_ci_low",
        "effect_ci_high",
        "coherence_mean",
        "generated_tokens_mean",
        "generated_tokens_max",
    )
    if not np.isfinite(
        behavior[list(behavior_numeric)].to_numpy(dtype=float)
    ).all():
        raise ValueError("Behavioral causal curve contains non-finite metrics.")
    expected_behavior = {
        *(("upstream", "necessity", k, 0) for k in args.k_values),
        *(("upstream", "sufficiency", k, 0) for k in args.sufficiency_k_values),
        *(("upstream", "sufficiency", k, 0) for k in args.upstream_extra_k_values),
        *(("downstream", "necessity", 0, k) for k in args.k_values),
        *(("downstream", "sufficiency", 0, k) for k in args.sufficiency_k_values),
        *(("union", "necessity", k, k) for k in args.k_values),
        *(("union", "sufficiency", k, k) for k in args.sufficiency_k_values),
        *((
            "union",
            "sufficiency",
            k,
            max(args.sufficiency_k_values),
        ) for k in args.upstream_extra_k_values),
    }
    observed_behavior = set(
        zip(
            behavior["scope"],
            behavior["regime"],
            behavior["upstream_k"],
            behavior["downstream_k"],
        )
    )
    missing_behavior = sorted(expected_behavior - observed_behavior)
    if missing_behavior:
        raise RuntimeError(f"incomplete behavioral curve; missing {missing_behavior}")
    internal.to_csv(
        output / "internal_faithfulness_necessity.csv", index=False
    )
    behavior.to_csv(output / "behavioral_necessity_sufficiency.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    for scope, frame in internal.groupby("scope"):
        axes[0, 0].plot(frame["circuit_edges"], frame["faithfulness"], "o-", label=scope)
        axes[0, 1].plot(frame["circuit_edges"], frame["necessity"], "o-", label=scope)
    for scope, frame in behavior[behavior["regime"] == "necessity"].groupby("scope"):
        axes[1, 0].plot(frame["circuit_edges"], frame["normal_minus_intervened_points"], "o-", label=scope)
    for scope, frame in behavior[behavior["regime"] == "sufficiency"].groupby("scope"):
        axes[1, 1].plot(frame["circuit_edges"], frame["intervened_trait_mean"], "o-", label=scope)
    axes[0, 0].set(title="Internal faithfulness", xlabel="Retained edges", ylabel="Normalized faithfulness")
    axes[0, 1].set(title="Internal necessity", xlabel="Retained edges", ylabel="Normalized necessity")
    axes[1, 0].set(title="Behavioral necessity", xlabel="Retained edges", ylabel="Trait-score reduction")
    axes[1, 1].axhline(
        np.mean([float(baseline[q]["trait_score"]) for q in args.question_indices]),
        color="black", linestyle="--", linewidth=0.8, label="normal",
    )
    axes[1, 1].set(title="Behavioral circuit-only sufficiency", xlabel="Retained edges", ylabel="Circuit-only trait score")
    for axis in axes.flat:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "causal_circuit_size_curves.pdf", bbox_inches="tight")
    fig.savefig(output / "causal_circuit_size_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "upstream_run": args.upstream_run,
        "downstream_run": args.downstream_run,
        "question_indices": args.question_indices,
        "minimum_token_cap": args.minimum_token_cap,
        "upstream_circuit_directory": args.upstream_circuit_directory,
        "downstream_circuit_directory": args.downstream_circuit_directory,
        "upstream_evaluation_template": args.upstream_evaluation_template,
        "downstream_evaluation_template": args.downstream_evaluation_template,
        "k_values": args.k_values,
        "upstream_extra_k_values": args.upstream_extra_k_values,
        "sufficiency_k_values": args.sufficiency_k_values,
        "tables": [
            "internal_faithfulness_necessity.csv",
            "behavioral_necessity_sufficiency.csv",
        ],
        "figures": ["causal_circuit_size_curves.pdf", "causal_circuit_size_curves.png"],
        "interpretation": {
            "necessity": "normal trait score minus selected-edge ablation trait score",
            "sufficiency": "trait score when all non-circuit edges use the matched counterfactual",
            "minimality": "smallest tested nested circuit passing both criteria; not a global optimum guarantee",
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

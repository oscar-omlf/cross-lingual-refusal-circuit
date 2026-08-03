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
    output.mkdir(parents=True, exist_ok=True)
    wanted_questions = set(args.question_indices)
    all_upstream_k = list(dict.fromkeys(args.k_values + args.upstream_extra_k_values))

    def circuit_path(scope: str, k: int) -> Path:
        if scope == "upstream":
            return upstream_root / "circuits_nested" / f"eapig_top_{k}.json"
        return (
            downstream_root
            / "circuits_nested"
            / f"steering_path_eapig_top_{k}.json"
        )

    def retained(scope: str, k: int) -> int:
        return int(json.loads(circuit_path(scope, k).read_text())["retained_edges"])

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
        eligible = int(discovery[eligible_key])
        for k in k_values:
            path = root / "evaluation" / f"k{k}_n32_seed1/aggregate.json"
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            internal_rows.append(
                {
                    "scope": scope,
                    "k": k,
                    "circuit_edges": retained(scope, k),
                    "eligible_edges": eligible,
                    "edge_fraction": retained(scope, k) / eligible,
                    "faithfulness": float(row["faithfulness"]),
                    "necessity": float(row["necessity"]),
                    "clean": float(row["means"]["clean"]),
                    "corrupt": float(row["means"]["corrupt"]),
                    "circuit_only": float(row["means"]["circuit_only"]),
                    "circuit_ablated": float(row["means"]["circuit_ablated"]),
                    "n": int(row[count_key]),
                    "artifact": str(path.relative_to(ROOT)),
                    "artifact_sha256": sha256_file(path),
                }
            )
    internal = pd.DataFrame(internal_rows).sort_values(["scope", "k"])
    expected_internal = {
        *(("upstream", k) for k in all_upstream_k),
        *(("downstream", k) for k in args.k_values),
    }
    observed_internal = set(zip(internal["scope"], internal["k"]))
    missing_internal = sorted(expected_internal - observed_internal)
    if missing_internal:
        raise RuntimeError(f"incomplete internal curve; missing {missing_internal}")
    internal.to_csv(output / "internal_faithfulness_necessity.csv", index=False)

    baseline_path = args.baseline_scored or (
        intermediate / "behavioral_pilot/k3000_n20/scored.jsonl"
    )
    baseline_path = baseline_path if baseline_path.is_absolute() else ROOT / baseline_path
    if not baseline_path.exists():
        raise FileNotFoundError(
            "No judged normal baseline; pass --baseline-scored from any complete "
            "upstream/downstream behavioral artifact."
        )
    baseline = {
        int(row["question_index"]): row
        for row in read_jsonl(baseline_path)
        if row["condition"] == "normal"
        and int(row["question_index"]) in wanted_questions
    }
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
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata.get("status") == "complete"
                and int(metadata.get("decoding", {}).get("max_new_tokens", 0))
                >= args.minimum_token_cap
                and set(map(int, metadata.get("question_indices", [])))
                == wanted_questions
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
            rows = [row for row in read_jsonl(path) if row["condition"] == condition]
            if not rows:
                continue
            artifacts.append(str(path.relative_to(ROOT)))
            for row in rows:
                question = int(row["question_index"])
                if question not in wanted_questions:
                    continue
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

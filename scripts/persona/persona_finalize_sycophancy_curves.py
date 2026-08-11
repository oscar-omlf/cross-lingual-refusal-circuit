#!/usr/bin/env python3
"""Build paper-ready Qwen sycophancy circuit tables and figures.

All behavioral inputs are explicit, complete, canonically judged, uncapped
artifacts. Discovery and internal-evaluation artifacts are hashed into the
manifest. No model or external API is used by this aggregation step.
"""

from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import ROOT
from persona_run_final_sycophancy_behavior import (
    CONFIG,
    CONNECTED_BEHAVIOR,
    DOWNSTREAM_BEHAVIOR,
    DOWNSTREAM_RUN,
    UPSTREAM_BEHAVIOR,
    UPSTREAM_RUN,
)

QSLUG = "q0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19"
UPSTREAM_K = (50, 100, 250, 500, 1000, 2000, 3000, 5000, 10000, 20000)
DOWNSTREAM_K = (50, 100, 250, 500, 1000, 2000, 3000)
BEHAVIOR_K = (50, 100, 250, 500, 1000, 2000, 3000)
OUTPUT = ROOT / "results/qwen2.5-7b-instruct/persona/sycophantic/paper_suite_final_reference_v3"


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    upstream_root = result_root / "circuit_discovery_upstream_contrastive" / UPSTREAM_RUN
    downstream_root = result_root / "circuit_discovery_steering_path" / DOWNSTREAM_RUN
    OUTPUT.mkdir(parents=True, exist_ok=True)

    roots = {"upstream": upstream_root, "downstream": downstream_root}
    k_values = {"upstream": UPSTREAM_K, "downstream": DOWNSTREAM_K}
    circuit_template = {
        "upstream": "circuits_nested/eapig_top_{k}.json",
        "downstream": "circuits_nested/steering_path_eapig_top_{k}.json",
    }
    eligible_key = {"upstream": "eligible_edges", "downstream": "eligible_downstream_edges"}
    count_key = {"upstream": "n_pairs", "downstream": "n_prompts"}
    internal_rows = []
    provenance: dict[str, str] = {}
    for scope in ("upstream", "downstream"):
        discovery_path = roots[scope] / "eapig_run_metadata.json"
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
        if discovery.get("status") != "complete":
            raise RuntimeError(f"incomplete {scope} discovery")
        provenance[str(discovery_path.relative_to(ROOT))] = sha256_file(discovery_path)
        eligible = int(discovery[eligible_key[scope]])
        for k in k_values[scope]:
            circuit_path = roots[scope] / circuit_template[scope].format(k=k)
            evaluation_path = roots[scope] / "evaluation" / f"k{k}_n32_seed1/aggregate.json"
            if not circuit_path.exists() or not evaluation_path.exists():
                raise FileNotFoundError(f"missing internal cell {scope} k={k}")
            circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            retained = int(circuit["retained_edges"])
            internal_rows.append({
                "scope": scope,
                "k": k,
                "retained_edges": retained,
                "eligible_edges": eligible,
                "edge_fraction_percent": 100.0 * retained / eligible,
                "faithfulness": float(evaluation["faithfulness"]),
                "necessity": float(evaluation["necessity"]),
                "clean_metric": float(evaluation["means"]["clean"]),
                "corrupt_metric": float(evaluation["means"]["corrupt"]),
                "circuit_only_metric": float(evaluation["means"]["circuit_only"]),
                "circuit_ablated_metric": float(evaluation["means"]["circuit_ablated"]),
                "n": int(evaluation[count_key[scope]]),
                "evaluation_artifact": str(evaluation_path.relative_to(ROOT)),
            })
            provenance[str(circuit_path.relative_to(ROOT))] = sha256_file(circuit_path)
            provenance[str(evaluation_path.relative_to(ROOT))] = sha256_file(evaluation_path)
    internal = pd.DataFrame(internal_rows).sort_values(["scope", "k"])
    internal.to_csv(OUTPUT / "table_internal_faithfulness_necessity.csv", index=False)

    baseline_path = UPSTREAM_BEHAVIOR / f"k1000_{QSLUG}_max640_final/scored.jsonl"
    baseline_rows = {
        int(row["question_index"]): row
        for row in read_jsonl(baseline_path)
        if row["condition"] == "normal"
    }
    if set(baseline_rows) != set(range(20)):
        raise RuntimeError("frozen normal baseline is incomplete")
    provenance[str(baseline_path.relative_to(ROOT))] = sha256_file(baseline_path)

    def upstream_scored(k: int) -> Path:
        return UPSTREAM_BEHAVIOR / f"k{k}_{QSLUG}_max640_final/scored.jsonl"

    def downstream_scored(k: int) -> Path:
        suffix = (
            "max640_a100_necessity_final"
            if k in (50, 100)
            else "max1000_a100_necessity_lowmem_v1"
        )
        return DOWNSTREAM_BEHAVIOR / f"k{k}_{QSLUG}_{suffix}/scored.jsonl"

    rng = np.random.default_rng(20260807)

    def summarize_behavior(scope: str, k: int, path: Path, condition: str) -> dict:
        metadata_path = path.with_name("metadata.json")
        judge_metadata_path = path.with_suffix(".metadata.json")
        if not path.exists() or not metadata_path.exists() or not judge_metadata_path.exists():
            raise FileNotFoundError(f"missing judged behavior cell {scope} k={k}: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        judge_metadata = json.loads(judge_metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete" or judge_metadata.get("status") != "complete":
            raise RuntimeError(f"incomplete judged behavior cell {scope} k={k}")
        rows = {
            int(row["question_index"]): row
            for row in read_jsonl(path)
            if row["condition"] == condition
        }
        if set(rows) != set(range(20)):
            raise RuntimeError(f"wrong row set {scope} k={k} {condition}")
        if any(row.get("hit_max_new_tokens", False) for row in rows.values()):
            raise RuntimeError(f"capped behavior row {scope} k={k} {condition}")
        if any(row.get("trait_score") is None or row.get("coherence_score") is None for row in rows.values()):
            raise RuntimeError(f"judge failure {scope} k={k} {condition}")
        normal = np.array([float(baseline_rows[q]["trait_score"]) for q in range(20)])
        intervened = np.array([float(rows[q]["trait_score"]) for q in range(20)])
        coherence = np.array([float(rows[q]["coherence_score"]) for q in range(20)])
        tokens = np.array([int(rows[q]["generated_token_count"]) for q in range(20)])
        effect = normal - intervened
        samples = rng.integers(0, len(effect), size=(10000, len(effect)))
        boot = effect[samples].mean(axis=1)
        circuit_path = roots[scope] / circuit_template[scope].format(k=k)
        circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
        eligible = int(json.loads((roots[scope] / "eapig_run_metadata.json").read_text())[eligible_key[scope]])
        provenance[str(path.relative_to(ROOT))] = sha256_file(path)
        provenance[str(metadata_path.relative_to(ROOT))] = sha256_file(metadata_path)
        provenance[str(judge_metadata_path.relative_to(ROOT))] = sha256_file(judge_metadata_path)
        return {
            "scope": scope,
            "k": k,
            "retained_edges": int(circuit["retained_edges"]),
            "eligible_edges": eligible,
            "edge_fraction_percent": 100.0 * int(circuit["retained_edges"]) / eligible,
            "n": 20,
            "normal_trait_mean": normal.mean(),
            "circuit_ablated_trait_mean": intervened.mean(),
            "behavioral_reversal_points": effect.mean(),
            "reversal_ci_low": np.quantile(boot, 0.025),
            "reversal_ci_high": np.quantile(boot, 0.975),
            "baseline_high_count": int((normal >= 50).sum()),
            "high_to_low_count": int(((normal >= 50) & (intervened < 50)).sum()),
            "circuit_ablated_coherence_mean": coherence.mean(),
            "generated_tokens_mean": tokens.mean(),
            "generated_tokens_max": tokens.max(),
            "artifact": str(path.relative_to(ROOT)),
        }

    behavior_rows = []
    for k in BEHAVIOR_K:
        behavior_rows.append(summarize_behavior("upstream", k, upstream_scored(k), "circuit_ablated"))
        behavior_rows.append(summarize_behavior("downstream", k, downstream_scored(k), "circuit_ablated"))
    behavior = pd.DataFrame(behavior_rows).sort_values(["scope", "k"])
    behavior.to_csv(OUTPUT / "table_behavioral_ablation_by_size.csv", index=False)

    union_root = CONNECTED_BEHAVIOR / "ku5000_kd3000"
    union_paths = {
        "both_circuit_only": union_root / f"union_both_circuit_only_ku5000_kd3000_{QSLUG}_max1000_a100_reference_v3/scored.jsonl",
        "both_ablated": union_root / f"union_both_ablated_ku5000_kd3000_{QSLUG}_max1000_a100_reference_v3/scored.jsonl",
    }
    union_rows = []
    for condition, path in union_paths.items():
        metadata_path = path.with_name("metadata.json")
        judge_metadata_path = path.with_suffix(".metadata.json")
        if not path.exists() or not metadata_path.exists() or not judge_metadata_path.exists():
            raise FileNotFoundError(f"missing selected union artifact {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        judge_metadata = json.loads(judge_metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete" or judge_metadata.get("status") != "complete":
            raise RuntimeError(f"incomplete selected union {condition}")
        rows = {
            int(row["question_index"]): row
            for row in read_jsonl(path)
            if row["condition"] == condition
        }
        if set(rows) != set(range(20)) or any(row.get("hit_max_new_tokens", False) for row in rows.values()):
            raise RuntimeError(f"invalid selected union row set {condition}")
        scores = np.array([float(rows[q]["trait_score"]) for q in range(20)])
        coherence = np.array([float(rows[q]["coherence_score"]) for q in range(20)])
        normal = np.array([float(baseline_rows[q]["trait_score"]) for q in range(20)])
        effect = normal - scores
        samples = rng.integers(0, 20, size=(10000, 20))
        boot = effect[samples].mean(axis=1)
        union_rows.append({
            "condition": condition,
            "upstream_k": 5000,
            "downstream_k": 3000,
            "total_retained_edges": 8000,
            "n": 20,
            "normal_trait_mean": normal.mean(),
            "intervened_trait_mean": scores.mean(),
            "normal_minus_intervened_points": effect.mean(),
            "effect_ci_low": np.quantile(boot, 0.025),
            "effect_ci_high": np.quantile(boot, 0.975),
            "coherence_mean": coherence.mean(),
            "high_to_low_count": int(((normal >= 50) & (scores < 50)).sum()),
            "artifact": str(path.relative_to(ROOT)),
        })
        for artifact in (path, metadata_path, judge_metadata_path):
            provenance[str(artifact.relative_to(ROOT))] = sha256_file(artifact)
    union = pd.DataFrame(union_rows)
    union.to_csv(OUTPUT / "table_selected_union_diagnostics.csv", index=False)

    with (OUTPUT / "table_internal_faithfulness_necessity.tex").open("w", encoding="utf-8") as handle:
        handle.write(internal[["scope", "k", "retained_edges", "edge_fraction_percent", "faithfulness", "necessity", "n"]].to_latex(index=False, float_format="%.3f"))
    with (OUTPUT / "table_behavioral_ablation_by_size.tex").open("w", encoding="utf-8") as handle:
        handle.write(behavior[["scope", "k", "retained_edges", "circuit_ablated_trait_mean", "behavioral_reversal_points", "reversal_ci_low", "reversal_ci_high", "circuit_ablated_coherence_mean"]].to_latex(index=False, float_format="%.2f"))
    with (OUTPUT / "table_selected_union_diagnostics.tex").open("w", encoding="utf-8") as handle:
        handle.write(union.drop(columns=["artifact"]).to_latex(index=False, float_format="%.2f"))

    colors = {"upstream": "#3366CC", "downstream": "#D95F02"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for scope, frame in internal.groupby("scope"):
        axes[0].plot(frame["retained_edges"], frame["faithfulness"], "o-", color=colors[scope], label=scope)
        axes[1].plot(frame["retained_edges"], frame["necessity"], "o-", color=colors[scope], label=scope)
    for axis, title, ylabel in zip(axes, ("Faithfulness", "Necessity"), ("Normalized faithfulness", "Normalized necessity")):
        axis.set_xscale("log")
        axis.set(title=title, xlabel="Retained edges", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_internal_faithfulness_necessity.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_internal_faithfulness_necessity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for axis, scope in zip(axes, ("upstream", "downstream")):
        frame = behavior[behavior["scope"] == scope]
        x = frame["retained_edges"].to_numpy()
        reduction = frame["behavioral_reversal_points"].to_numpy()
        low = reduction - frame["reversal_ci_low"].to_numpy()
        high = frame["reversal_ci_high"].to_numpy() - reduction
        axis.errorbar(x, reduction, yerr=np.vstack([low, high]), fmt="o-", capsize=3, color=colors[scope])
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1, label="no effect")
        axis.set_xscale("log")
        axis.set(title=scope.capitalize(), xlabel="Retained edges", ylabel="Sycophancy-score reduction")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_behavioral_effect_by_circuit_size.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_behavioral_effect_by_circuit_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "model": config["model"],
        "upstream_run": UPSTREAM_RUN,
        "downstream_run": DOWNSTREAM_RUN,
        "question_indices": list(range(20)),
        "internal_k": {"upstream": list(UPSTREAM_K), "downstream": list(DOWNSTREAM_K)},
        "behavior_k": list(BEHAVIOR_K),
        "selected_union": {"upstream_k": 5000, "downstream_k": 3000},
        "bootstrap": {"method": "paired nonparametric", "resamples": 10000, "seed": 20260807},
        "tables": [
            "table_internal_faithfulness_necessity.csv",
            "table_internal_faithfulness_necessity.tex",
            "table_behavioral_ablation_by_size.csv",
            "table_behavioral_ablation_by_size.tex",
            "table_selected_union_diagnostics.csv",
            "table_selected_union_diagnostics.tex",
        ],
        "figures": [
            "figure_internal_faithfulness_necessity.pdf",
            "figure_internal_faithfulness_necessity.png",
            "figure_behavioral_effect_by_circuit_size.pdf",
            "figure_behavioral_effect_by_circuit_size.png",
        ],
        "provenance_sha256": dict(sorted(provenance.items())),
        "interpretation": {
            "faithfulness": "native scalar target recovered by the retained circuit under the matched counterfactual",
            "necessity": "native scalar target removed by ablating retained edges with the matched counterfactual",
            "behavioral_effect": "paired normal-minus-circuit-ablated canonical sycophancy score",
            "union": "prespecified high-coverage formation-expression composition; not a one-dimensional size curve",
        },
    }
    write_json(OUTPUT / "manifest.json", manifest)
    for name in manifest["tables"] + manifest["figures"]:
        if not (OUTPUT / name).exists():
            raise RuntimeError(f"missing final output {name}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

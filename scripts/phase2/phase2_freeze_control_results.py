#!/usr/bin/env python3
"""Freeze Qwen refusal control results as paper-ready tables and figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ROOT
from phase2_refusal_analysis import _paired_effects


DEFAULT_CONFIG = ROOT / "configs/phase2_qwen_refusal.json"
DEFAULT_JOBS = ROOT / "configs/phase2_qwen_refusal_control_job_accounting.json"
LANGUAGES = ("en", "de", "zh", "th")
LANGUAGE_NAMES = {"en": "English", "de": "German", "zh": "Chinese", "th": "Thai"}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_no_text(frame, path: Path, *, require_judge: bool = False) -> None:
    forbidden = {"prompt", "instruction", "response", "completion", "raw_output"}
    if forbidden & set(frame.columns):
        raise ValueError(f"Saved text field in {path}")
    if frame.parallel_id.nunique() != 32:
        raise ValueError(f"Expected 32 unique IDs in {path}")
    if require_judge and not frame.parse_ok.all():
        raise ValueError(f"WildGuard parse failure in {path}")


def _wilson(count: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    p = count / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - radius, center + radius


def _paired_delta(values, *, seed: int, samples: int = 10000) -> tuple[float, float, float]:
    import numpy as np

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _load_behavior(phase2_root: Path, cfg: dict):
    import pandas as pd

    input_paths: list[Path] = []
    prior_frames = []
    for path in sorted((phase2_root / "behavior/development").glob("source_*/*/instance_labels_no_text.csv")):
        frame = pd.read_csv(path)
        _validate_no_text(frame, path, require_judge=True)
        prior_frames.append(frame)
        input_paths.append(path)
    prior = pd.concat(prior_frames, ignore_index=True)
    controls = prior[prior.condition.isin(("normal", "direction_ablated"))]
    controls = controls.drop_duplicates(["language", "parallel_id", "condition"])
    if controls.groupby(["language", "parallel_id", "condition"]).refusal.nunique().max() > 1:
        raise ValueError("Conflicting deterministic behavior controls")
    controls = controls.pivot(
        index=["language", "parallel_id"], columns="condition", values="refusal"
    )

    rows = []
    validation_rows = []
    for index, language in enumerate(LANGUAGES):
        ablation_path = phase2_root / f"behavior/development/source_en/{language}/instance_labels_no_text.csv"
        sufficiency_path = phase2_root / f"behavior_sufficiency/development/source_en/{language}/instance_labels_no_text.csv"
        ablation = pd.read_csv(ablation_path)
        sufficiency = pd.read_csv(sufficiency_path)
        _validate_no_text(ablation, ablation_path, require_judge=True)
        _validate_no_text(sufficiency, sufficiency_path, require_judge=True)
        input_paths.extend((ablation_path, sufficiency_path))
        ablated = ablation[
            ablation.condition.eq("circuit_ablated") & ablation.requested_k.eq(3000)
        ].set_index("parallel_id")
        circuit_only = sufficiency.set_index("parallel_id")
        paired = controls.loc[language].join(
            circuit_only.refusal.rename("circuit_only")
        ).join(ablated.refusal.rename("circuit_ablated"))
        effects = _paired_effects(
            paired,
            samples=int(cfg["statistics"]["bootstrap_samples"]),
            seed=int(cfg["statistics"]["bootstrap_seed"]) + index,
            confidence=float(cfg["statistics"]["confidence_level"]),
        )
        rows.append({"language": language, "language_name": LANGUAGE_NAMES[language], **effects})
        for condition, frame in (("circuit_ablated", ablated), ("circuit_only", circuit_only)):
            validation_rows.append(
                {
                    "language": language,
                    "condition": condition,
                    "n": len(frame),
                    "parse_failure_rate": 1 - frame.parse_ok.mean(),
                    "hit_max_new_tokens_rate": frame.hit_max_new_tokens.mean(),
                    "mean_generated_tokens": frame.generated_token_count.mean(),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(validation_rows), input_paths


def _load_component_metrics(phase2_root: Path):
    import pandas as pd

    input_paths: list[Path] = []
    rows = []
    families = ["terminal_only", "upstream_only"] + [f"random_min_seed{seed}" for seed in range(3)]
    for family in families:
        for language in LANGUAGES:
            aggregate_path = phase2_root / f"controls/k3000_metrics/{family}/{language}/aggregate.csv"
            instance_path = phase2_root / f"controls/k3000_metrics/{family}/{language}/instance_metrics_no_text.csv"
            aggregate = pd.read_csv(aggregate_path).iloc[0]
            instances = pd.read_csv(instance_path)
            _validate_no_text(instances, instance_path)
            input_paths.extend((aggregate_path, instance_path))
            condition_means = instances.groupby("condition").agg(
                first_token_refusal_rate=("first_token_refusal_indicator", "mean"),
                first_token_refusal_log_odds=("first_token_refusal_log_odds", "mean"),
                top1_change_rate=("top1_changed_from_normal", "mean"),
            )
            rows.append(
                {
                    "mask": family,
                    "mask_family": "matched_random" if family.startswith("random") else family,
                    "language": language,
                    "retained_edges": int(aggregate.retained_edges),
                    "faithfulness": float(aggregate.faithfulness),
                    "necessity": float(aggregate.necessity),
                    "first_token_circuit_only_refusal_rate": float(condition_means.loc["circuit_only", "first_token_refusal_rate"]),
                    "first_token_circuit_ablated_refusal_rate": float(condition_means.loc["circuit_ablated", "first_token_refusal_rate"]),
                    "circuit_only_top1_change_rate": float(condition_means.loc["circuit_only", "top1_change_rate"]),
                    "circuit_ablated_top1_change_rate": float(condition_means.loc["circuit_ablated", "top1_change_rate"]),
                }
            )

    for language in LANGUAGES:
        # The full mask must be evaluated on the same Phase-2 development
        # prompts as the component and random controls. Historical offset-0
        # rows are not a valid paired comparator here.
        aggregate_path = phase2_root / f"controls/k3000_metrics/discovered_full/{language}/aggregate.csv"
        instance_path = phase2_root / f"controls/k3000_metrics/discovered_full/{language}/instance_metrics_no_text.csv"
        aggregate = pd.read_csv(aggregate_path).query("requested_k == 3000").iloc[0]
        instances = pd.read_csv(instance_path).query("requested_k == 3000")
        _validate_no_text(instances, instance_path)
        input_paths.extend((aggregate_path, instance_path))
        condition_means = instances.groupby("condition").agg(
            first_token_refusal_rate=("first_token_refusal_indicator", "mean"),
            top1_change_rate=("top1_changed_from_normal", "mean"),
        )
        rows.append(
            {
                "mask": "discovered_full",
                "mask_family": "discovered_full",
                "language": language,
                "retained_edges": int(aggregate.retained_edges),
                "faithfulness": float(aggregate.faithfulness),
                "necessity": float(aggregate.necessity),
                "first_token_circuit_only_refusal_rate": float(condition_means.loc["circuit_only", "first_token_refusal_rate"]),
                "first_token_circuit_ablated_refusal_rate": float(condition_means.loc["circuit_ablated", "first_token_refusal_rate"]),
                "circuit_only_top1_change_rate": float(condition_means.loc["circuit_only", "top1_change_rate"]),
                "circuit_ablated_top1_change_rate": float(condition_means.loc["circuit_ablated", "top1_change_rate"]),
            }
        )
    return pd.DataFrame(rows), input_paths


def _add_mask_composition(phase2_root: Path, frame):
    import pandas as pd

    reference_path = ROOT / "results/qwen2.5-7b-instruct/polyrefuse/en/circuit_discovery_english_vector/circuits_phase2_nested/eapig_top_3000.json"
    reference = _read(reference_path)
    reference_names = {edge["name"] for edge in reference["edges"]}
    mask_paths = {
        "discovered_full": reference_path,
        "terminal_only": phase2_root / "controls/k3000_masks/terminal_only/eapig_top_3000.json",
        "upstream_only": phase2_root / "controls/k3000_masks/upstream_only/eapig_top_3000.json",
        **{
            f"random_min_seed{seed}": phase2_root / f"controls/k3000_masks/random/stratified_min_overlap/seed_{seed}/eapig_top_3000.json"
            for seed in range(3)
        },
    }
    composition = []
    for name, path in mask_paths.items():
        names = {edge["name"] for edge in _read(path)["edges"]}
        terminal = {edge for edge in names if edge.endswith("->logits")}
        overlap = names & reference_names
        composition.append(
            {
                "mask": name,
                "terminal_edges": len(terminal),
                "upstream_edges": len(names - terminal),
                "overlap_with_discovered": len(overlap),
                "terminal_overlap_with_discovered": len({x for x in overlap if x.endswith("->logits")}),
            }
        )
    return frame.merge(pd.DataFrame(composition), on="mask", validate="many_to_one"), list(mask_paths.values())


def _load_harmless(phase2_root: Path, cfg: dict):
    import pandas as pd

    rows = []
    input_paths: list[Path] = []
    for index, language in enumerate(LANGUAGES):
        behavior_path = phase2_root / f"harmless/behavior/development/source_en/{language}/instance_labels_no_text.csv"
        metric_path = phase2_root / f"harmless/metrics/source_en/{language}/instance_metrics_no_text.csv"
        behavior = pd.read_csv(behavior_path)
        metrics = pd.read_csv(metric_path)
        _validate_no_text(behavior, behavior_path, require_judge=True)
        _validate_no_text(metrics, metric_path)
        input_paths.extend((behavior_path, metric_path))
        paired = behavior.pivot(index="parallel_id", columns="condition", values="refusal").astype(float)
        difference, low, high = _paired_delta(
            paired.circuit_ablated - paired.normal,
            seed=int(cfg["statistics"]["bootstrap_seed"]) + 100 + index,
        )
        circuit_metrics = metrics[metrics.condition.eq("circuit_ablated")]
        normal_behavior = behavior[behavior.condition.eq("normal")]
        ablated_behavior = behavior[behavior.condition.eq("circuit_ablated")]
        rows.append(
            {
                "language": language,
                "language_name": LANGUAGE_NAMES[language],
                "n_prompts": len(paired),
                "normal_refusal_rate": paired.normal.mean(),
                "circuit_ablated_refusal_rate": paired.circuit_ablated.mean(),
                "paired_refusal_difference": difference,
                "paired_refusal_difference_ci_low": low,
                "paired_refusal_difference_ci_high": high,
                "normal_response_harmful_rate": normal_behavior.response_harmful.mean(),
                "circuit_ablated_response_harmful_rate": ablated_behavior.response_harmful.mean(),
                "first_token_top1_change_rate": circuit_metrics.top1_changed_from_normal.mean(),
                "kl_normal_to_circuit_ablated_mean": circuit_metrics.kl_normal_to_condition.mean(),
                "kl_normal_to_circuit_ablated_median": circuit_metrics.kl_normal_to_condition.median(),
                "parse_failure_rate": 1 - behavior.parse_ok.mean(),
            }
        )
    return pd.DataFrame(rows), input_paths


def _write_tables(output_dir: Path, behavior, components, harmless, jobs):
    import pandas as pd

    behavior_path = output_dir / "table_k3000_causal_behavior.csv"
    components_path = output_dir / "table_terminal_confound_metrics.csv"
    summary_path = output_dir / "table_terminal_confound_summary.csv"
    harmless_path = output_dir / "table_harmless_specificity.csv"
    costs_path = output_dir / "table_job_costs.csv"
    behavior.to_csv(behavior_path, index=False)
    components.to_csv(components_path, index=False)
    summary = components.groupby("mask_family", as_index=False).agg(
        masks=("mask", "nunique"),
        evaluations=("language", "size"),
        retained_edges_mean=("retained_edges", "mean"),
        terminal_edges_mean=("terminal_edges", "mean"),
        faithfulness_mean=("faithfulness", "mean"),
        faithfulness_min=("faithfulness", "min"),
        faithfulness_max=("faithfulness", "max"),
        necessity_mean=("necessity", "mean"),
        necessity_min=("necessity", "min"),
        necessity_max=("necessity", "max"),
        first_token_circuit_only_refusal_rate_mean=("first_token_circuit_only_refusal_rate", "mean"),
        first_token_circuit_ablated_refusal_rate_mean=("first_token_circuit_ablated_refusal_rate", "mean"),
    )
    order = pd.CategoricalDtype(
        ["discovered_full", "terminal_only", "upstream_only", "matched_random"], ordered=True
    )
    summary["mask_family"] = summary.mask_family.astype(order)
    summary = summary.sort_values("mask_family")
    summary.to_csv(summary_path, index=False)
    harmless.to_csv(harmless_path, index=False)
    job_frame = pd.DataFrame(jobs["jobs"])
    job_frame["gpu_hours"] = job_frame.elapsed_seconds / 3600
    job_frame["estimated_sbu"] = job_frame.gpu_hours * int(jobs["billing_sbu_per_a100_hour"])
    job_frame.to_csv(costs_path, index=False)

    behavior_columns = [
        "language_name", "normal_refusal_rate", "direction_ablated_refusal_rate",
        "circuit_ablated_refusal_rate", "normalized_refusal_reversal",
        "circuit_only_refusal_rate", "circuit_only_refusal_retention",
    ]
    component_columns = [
        "mask_family", "retained_edges_mean", "terminal_edges_mean",
        "faithfulness_mean", "necessity_mean",
        "first_token_circuit_only_refusal_rate_mean",
        "first_token_circuit_ablated_refusal_rate_mean",
    ]
    harmless_columns = [
        "language_name", "normal_refusal_rate", "circuit_ablated_refusal_rate",
        "paired_refusal_difference", "paired_refusal_difference_ci_low",
        "paired_refusal_difference_ci_high", "first_token_top1_change_rate",
    ]
    latex_paths = [
        output_dir / "table_k3000_causal_behavior.tex",
        output_dir / "table_terminal_confound_summary.tex",
        output_dir / "table_harmless_specificity.tex",
    ]
    latex_paths[0].write_text(behavior[behavior_columns].to_latex(index=False, float_format="%.3f", escape=True), encoding="utf-8")
    latex_paths[1].write_text(summary[component_columns].to_latex(index=False, float_format="%.3f", escape=True), encoding="utf-8")
    latex_paths[2].write_text(harmless[harmless_columns].to_latex(index=False, float_format="%.3f", escape=True), encoding="utf-8")
    return [behavior_path, components_path, summary_path, harmless_path, costs_path, *latex_paths], summary


def _save_figures(output_dir: Path, behavior, components, harmless):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"en": "#0072B2", "de": "#D55E00", "zh": "#009E73", "th": "#CC79A7"}
    outputs = []

    x = np.arange(len(LANGUAGES))
    fig, axis = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    conditions = [
        ("normal_refusal_rate", "Normal", "#222222", "o"),
        ("direction_ablated_refusal_rate", "Direction ablated", "#999999", "s"),
        ("circuit_ablated_refusal_rate", "Circuit ablated", "#D55E00", "v"),
        ("circuit_only_refusal_rate", "Circuit only", "#0072B2", "^"),
    ]
    for column, label, color, marker in conditions:
        values = behavior.set_index("language").loc[list(LANGUAGES), column].to_numpy()
        lows, highs = zip(*[_wilson(int(round(value * 32)), 32) for value in values])
        axis.errorbar(
            x, values,
            yerr=np.vstack((values - np.asarray(lows), np.asarray(highs) - values)),
            label=label, color=color, marker=marker, linewidth=1.4, capsize=2,
        )
    axis.set(
        xticks=x,
        xticklabels=[LANGUAGE_NAMES[lang] for lang in LANGUAGES],
        ylabel="WildGuard refusal rate",
        ylim=(-0.04, 1.08),
        title="Causal behavior of the English $k=3000$ circuit",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    for suffix in ("pdf", "png"):
        path = output_dir / f"figure_k3000_causal_behavior.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)

    category_order = ["discovered_full", "terminal_only", "upstream_only", "matched_random"]
    category_labels = ["Discovered\nfull", "Terminal\nonly", "Upstream\nonly", "Matched\nrandom"]
    category_colors = ["#0072B2", "#E69F00", "#009E73", "#999999"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for axis, metric, title in zip(axes, ("faithfulness", "necessity"), ("Circuit only", "Circuit ablated")):
        for position, (family, color) in enumerate(zip(category_order, category_colors)):
            values = components.loc[components.mask_family.eq(family), metric].to_numpy()
            axis.bar(position, values.mean(), width=0.65, color=color, alpha=0.7)
            jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else np.array([0.0])
            axis.scatter(position + jitter, values, s=18, color="black", alpha=0.65, zorder=3)
        axis.axhline(0.8, color="0.35", linestyle="--", linewidth=1)
        axis.set(xticks=range(4), xticklabels=category_labels, ylabel=metric.capitalize(), title=title, ylim=(-0.05, 1.12))
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Terminal-edge diagnostic at $k=3000$", fontsize=10)
    for suffix in ("pdf", "png"):
        path = output_dir / f"figure_terminal_edge_diagnostic.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True)
    harmless_ordered = harmless.set_index("language").loc[list(LANGUAGES)]
    axes[0].plot(x, harmless_ordered.normal_refusal_rate, marker="o", color="#222222", label="Normal")
    axes[0].plot(x, harmless_ordered.circuit_ablated_refusal_rate, marker="v", color="#D55E00", label="Circuit ablated")
    axes[0].set(ylabel="Refusal rate", title="Harmless behavior", ylim=(-0.02, 0.18))
    axes[0].legend(frameon=False)
    delta = harmless_ordered.paired_refusal_difference.to_numpy()
    low = harmless_ordered.paired_refusal_difference_ci_low.to_numpy()
    high = harmless_ordered.paired_refusal_difference_ci_high.to_numpy()
    axes[1].errorbar(x, delta, yerr=np.vstack((delta - low, high - delta)), marker="o", linestyle="none", color="#0072B2", capsize=3)
    axes[1].axhline(0, color="0.25", linewidth=1)
    axes[1].axhline(0.05, color="0.55", linestyle="--", linewidth=1)
    axes[1].set(ylabel="Paired refusal increase", title="Behavioral collateral", ylim=(-0.12, 0.2))
    axes[2].bar(x, harmless_ordered.first_token_top1_change_rate, color=[colors[lang] for lang in LANGUAGES], alpha=0.75)
    axes[2].axhline(0.1, color="0.45", linestyle="--", linewidth=1)
    axes[2].set(ylabel="Top-1 change rate", title="First-token collateral", ylim=(0, 0.23))
    for axis in axes:
        axis.set_xticks(x, [LANGUAGE_NAMES[lang] for lang in LANGUAGES], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    for suffix in ("pdf", "png"):
        path = output_dir / f"figure_harmless_specificity.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = _read(args.config)
    jobs = _read(args.jobs)
    phase2_root = ROOT / cfg["artifacts"]["phase2_results_root"]
    output_dir = args.output_dir or phase2_root / "frozen_controls_2026-07-20"
    output_dir.mkdir(parents=True, exist_ok=True)

    behavior, validation, behavior_inputs = _load_behavior(phase2_root, cfg)
    components, component_inputs = _load_component_metrics(phase2_root)
    components, mask_inputs = _add_mask_composition(phase2_root, components)
    harmless, harmless_inputs = _load_harmless(phase2_root, cfg)
    generated, component_summary = _write_tables(output_dir, behavior, components, harmless, jobs)
    validation_path = output_dir / "table_generation_validation.csv"
    validation.to_csv(validation_path, index=False)
    generated.append(validation_path)
    generated.extend(_save_figures(output_dir, behavior, components, harmless))

    old_manifest_path = phase2_root / "frozen_development_2026-07-19/manifest.json"
    old_cost = _read(old_manifest_path)["gpu_cost"]
    new_cost = jobs["totals"]
    assessment = {
        "schema_version": 1,
        "status": "development_controls_complete",
        "selected_k_status": "k3000 remains provisional",
        "terminal_cut_diagnostic": (
            "Terminal-only ablation is an expected cut-set intervention and is not an upstream-specificity test. "
            "Terminal-only circuit-only execution is only partially projection-faithful; use the later "
            "fixed-terminal/random-upstream controls to test upstream specificity."
        ),
        "behavioral_sufficiency": {
            row.language: {
                "point": row.circuit_only_refusal_retention,
                "ci_low": row.circuit_only_refusal_retention_ci_low,
                "ci_high": row.circuit_only_refusal_retention_ci_high,
                "passes_point_threshold_0.9": bool(row.circuit_only_refusal_retention >= 0.9),
            }
            for row in behavior.itertuples()
        },
        "harmless_refusal_difference": {
            row.language: {
                "point": row.paired_refusal_difference,
                "ci_low": row.paired_refusal_difference_ci_low,
                "ci_high": row.paired_refusal_difference_ci_high,
                "passes_point_threshold_0.05": bool(row.paired_refusal_difference <= 0.05),
            }
            for row in harmless.itertuples()
        },
        "remaining_publication_gates": [
            "adaptive Thai circuit-only test at larger k",
            "harmless reserve and semantic-utility check",
            "64-token multilingual judge/truncation robustness audit",
            "fresh held-out confirmation at the finally selected k",
        ],
    }
    assessment_path = output_dir / "assessment.json"
    assessment_path.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated.append(assessment_path)

    input_paths = [args.config, args.jobs, old_manifest_path, *behavior_inputs, *component_inputs, *mask_inputs, *harmless_inputs]
    input_paths = list(dict.fromkeys(input_paths))
    manifest = {
        "schema_version": 1,
        "status": "frozen_development_controls_not_final",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": cfg["model"]["name"],
        "languages": list(LANGUAGES),
        "requested_k": 3000,
        "inputs": {str(path.relative_to(ROOT)): _sha256(path) for path in input_paths},
        "outputs": {str(path.relative_to(output_dir)): _sha256(path) for path in generated},
        "gpu_cost": {
            "this_wave_a100_hours": new_cost["elapsed_a100_hours"],
            "this_wave_sbu": new_cost["estimated_sbu"],
            "cumulative_a100_hours": old_cost["elapsed_a100_hours"] + new_cost["elapsed_a100_hours"],
            "cumulative_sbu": old_cost["estimated_sbu"] + new_cost["estimated_sbu"],
        },
        "component_summary": component_summary.to_dict(orient="records"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "outputs": len(generated), "gpu_cost": manifest["gpu_cost"]}, indent=2))


if __name__ == "__main__":
    main()

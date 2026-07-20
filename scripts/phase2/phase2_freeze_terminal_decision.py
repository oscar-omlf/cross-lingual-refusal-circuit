#!/usr/bin/env python3
"""Freeze the decisive English terminal-shell controls as paper-ready artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ROOT


PHASE2 = ROOT / "results/qwen2.5-7b-instruct/polyrefuse/phase2_refusal"
OUTPUT = PHASE2 / "frozen_terminal_decision_2026-07-20"
ACCOUNTING = ROOT / "configs/phase2_qwen_refusal_decisive_english_job_accounting.json"
REFERENCE = (
    ROOT
    / "results/qwen2.5-7b-instruct/polyrefuse/en/circuit_discovery_english_vector"
    / "circuits_phase2_nested/eapig_top_3000.json"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_instances(frame, path: Path) -> None:
    forbidden = {"prompt", "instruction", "response", "completion", "raw_output"}
    if forbidden & set(frame.columns):
        raise ValueError(f"Text-bearing column in {path}")
    if frame.parallel_id.nunique() != 32:
        raise ValueError(f"Expected 32 matched prompts in {path}")
    if "parse_ok" in frame and not frame.parse_ok.all():
        raise ValueError(f"Judge parse failure in {path}")


def mask_counts(path: Path, reference_names: set[str]) -> dict[str, int]:
    names = {edge["name"] for edge in read_json(path)["edges"]}
    terminal = {name for name in names if name.endswith("->logits")}
    return {
        "retained_edges": len(names),
        "terminal_edges": len(terminal),
        "upstream_edges": len(names - terminal),
        "overlap_with_discovered": len(names & reference_names),
        "terminal_overlap_with_discovered": len(terminal & reference_names),
    }


def bootstrap_delta(values, seed: int = 20260720, samples: int = 10000):
    import numpy as np

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def load_teacher_forcing():
    import pandas as pd

    reference_names = {edge["name"] for edge in read_json(REFERENCE)["edges"]}
    families = {
        "Discovered full": [
            (
                PHASE2 / "controls/k3000_metrics/discovered_full/en/aggregate.csv",
                REFERENCE,
            )
        ],
        "Terminal only": [
            (
                PHASE2 / "controls/k3000_metrics/terminal_only/en/aggregate.csv",
                PHASE2 / "controls/k3000_masks/terminal_only/eapig_top_3000.json",
            )
        ],
        "Upstream only": [
            (
                PHASE2 / "controls/k3000_metrics/upstream_only/en/aggregate.csv",
                PHASE2 / "controls/k3000_masks/upstream_only/eapig_top_3000.json",
            )
        ],
        "Matched random": [
            (
                PHASE2 / f"controls/k3000_metrics/random_min_seed{seed}/en/aggregate.csv",
                PHASE2
                / f"controls/k3000_masks/random/stratified_min_overlap/seed_{seed}/eapig_top_3000.json",
            )
            for seed in range(3)
        ],
        "Fixed terminal + random upstream": [
            (
                PHASE2
                / f"controls/k3000_metrics/fixed_terminal_random_upstream_seed{seed}/en/aggregate.csv",
                PHASE2
                / f"controls/k3000_masks/fixed_terminal_random_upstream/seed_{seed}/eapig_top_3000.json",
            )
            for seed in range(3)
        ],
    }
    detail = []
    inputs = [REFERENCE]
    for family, members in families.items():
        for seed, (aggregate_path, mask_path) in enumerate(members):
            aggregate = pd.read_csv(aggregate_path).iloc[0]
            counts = mask_counts(mask_path, reference_names)
            detail.append(
                {
                    "mask_family": family,
                    "seed": seed if len(members) > 1 else None,
                    **counts,
                    "edge_fraction": counts["retained_edges"] / 260956,
                    "faithfulness": float(aggregate.faithfulness),
                    "necessity": float(aggregate.necessity),
                }
            )
            inputs.extend((aggregate_path, mask_path))
    detail = pd.DataFrame(detail)
    summary = detail.groupby("mask_family", sort=False, as_index=False).agg(
        masks=("seed", "size"),
        retained_edges=("retained_edges", "mean"),
        terminal_edges=("terminal_edges", "mean"),
        upstream_edges=("upstream_edges", "mean"),
        overlap_with_discovered=("overlap_with_discovered", "mean"),
        faithfulness_mean=("faithfulness", "mean"),
        faithfulness_min=("faithfulness", "min"),
        faithfulness_max=("faithfulness", "max"),
        necessity_mean=("necessity", "mean"),
        necessity_min=("necessity", "min"),
        necessity_max=("necessity", "max"),
    )
    return detail, summary, inputs


def load_behavior():
    import pandas as pd

    normal_path = PHASE2 / "behavior/development/source_en/en/instance_labels_no_text.csv"
    sufficiency_path = PHASE2 / "behavior_sufficiency/development/source_en/en/instance_labels_no_text.csv"
    terminal_path = PHASE2 / "controls/k3000_behavior/terminal_only/en/instance_labels_no_text.csv"
    random_paths = [
        PHASE2 / f"controls/k3000_behavior/random_min_seed{seed}/en/instance_labels_no_text.csv"
        for seed in range(3)
    ]
    inputs = [normal_path, sufficiency_path, terminal_path, *random_paths]
    frames = {path: pd.read_csv(path) for path in inputs}
    for path, frame in frames.items():
        validate_instances(frame, path)

    baseline = frames[normal_path]
    normal = baseline[baseline.condition.eq("normal")].set_index("parallel_id")
    direction = baseline[baseline.condition.eq("direction_ablated")].set_index("parallel_id")
    discovered_ablated = baseline[
        baseline.condition.eq("circuit_ablated") & baseline.requested_k.eq(3000)
    ].set_index("parallel_id")
    discovered_only = frames[sufficiency_path].set_index("parallel_id")
    terminal = frames[terminal_path]
    terminal_only = terminal[terminal.condition.eq("circuit_only")].set_index("parallel_id")
    terminal_ablated = terminal[terminal.condition.eq("circuit_ablated")].set_index("parallel_id")

    groups = [
        ("Normal model", "normal", normal),
        ("Global direction ablated", "direction_ablated", direction),
        ("Discovered full", "circuit_only", discovered_only),
        ("Terminal only", "circuit_only", terminal_only),
        ("Discovered full", "circuit_ablated", discovered_ablated),
        ("Terminal only", "circuit_ablated", terminal_ablated),
    ]
    for seed, path in enumerate(random_paths):
        groups.append((f"Matched random seed {seed}", "circuit_ablated", frames[path].set_index("parallel_id")))

    denominator = normal.refusal.mean() - direction.refusal.mean()
    rows = []
    for family, condition, frame in groups:
        rate = float(frame.refusal.mean())
        reversal = (
            float((normal.refusal.mean() - rate) / denominator)
            if condition == "circuit_ablated"
            else float("nan")
        )
        rows.append(
            {
                "mask_family": family,
                "condition": condition,
                "n_prompts": len(frame),
                "refusal_rate": rate,
                "response_harmful_rate": float(frame.response_harmful.mean()),
                "refusal_substring_rate": float(frame.refusal_substring.mean()),
                "normalized_refusal_reversal": reversal,
                "hit_max_new_tokens_rate": float(frame.hit_max_new_tokens.mean()),
            }
        )
    detail = pd.DataFrame(rows)

    comparisons = []
    for label, condition, comparator, reference in [
        ("Terminal minus discovered (circuit only)", "circuit_only", terminal_only, discovered_only),
        ("Terminal minus discovered (circuit ablated)", "circuit_ablated", terminal_ablated, discovered_ablated),
        *[
            (
                f"Random seed {seed} minus discovered (circuit ablated)",
                "circuit_ablated",
                frames[path].set_index("parallel_id"),
                discovered_ablated,
            )
            for seed, path in enumerate(random_paths)
        ],
    ]:
        joined = comparator.refusal.rename("comparator").to_frame().join(
            reference.refusal.rename("discovered"), how="inner"
        )
        point, low, high = bootstrap_delta(
            joined.comparator.astype(float) - joined.discovered.astype(float),
            seed=20260720 + len(comparisons),
        )
        comparisons.append(
            {
                "comparison": label,
                "condition": condition,
                "n_prompts": len(joined),
                "paired_refusal_rate_difference": point,
                "ci_low": low,
                "ci_high": high,
                "discordant_prompts": int((joined.comparator != joined.discovered).sum()),
            }
        )
    return detail, pd.DataFrame(comparisons), inputs


def save_tables(teacher_detail, teacher_summary, behavior, paired, accounting):
    import pandas as pd

    outputs = []
    tables = {
        "table_terminal_shell_teacher_forcing_detail.csv": teacher_detail,
        "table_terminal_shell_teacher_forcing.csv": teacher_summary,
        "table_terminal_shell_behavior.csv": behavior,
        "table_terminal_shell_behavior_paired.csv": paired,
    }
    jobs = pd.DataFrame(accounting["jobs"])
    jobs["gpu_hours"] = jobs.elapsed_seconds / 3600
    jobs["sbu"] = jobs.gpu_hours * accounting["billing_sbu_per_a100_hour"]
    tables["table_job_costs.csv"] = jobs
    for name, frame in tables.items():
        path = OUTPUT / name
        frame.to_csv(path, index=False)
        outputs.append(path)

    latex = {
        "table_terminal_shell_teacher_forcing.tex": teacher_summary[
            [
                "mask_family", "retained_edges", "terminal_edges", "upstream_edges",
                "faithfulness_mean", "faithfulness_min", "faithfulness_max", "necessity_mean",
            ]
        ],
        "table_terminal_shell_behavior.tex": behavior[
            ["mask_family", "condition", "refusal_rate", "response_harmful_rate", "normalized_refusal_reversal"]
        ],
    }
    for name, frame in latex.items():
        path = OUTPUT / name
        path.write_text(frame.to_latex(index=False, float_format="%.3f", escape=True), encoding="utf-8")
        outputs.append(path)
    return outputs


def save_figure(teacher_detail, behavior):
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
    colors = {
        "Discovered full": "#0072B2",
        "Terminal only": "#E69F00",
        "Upstream only": "#009E73",
        "Matched random": "#999999",
        "Fixed terminal + random upstream": "#CC79A7",
    }
    order = list(colors)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)

    for position, family in enumerate(order):
        values = teacher_detail.loc[teacher_detail.mask_family.eq(family), "faithfulness"].to_numpy()
        axes[0].bar(position, values.mean(), width=0.68, color=colors[family], alpha=0.75)
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        axes[0].scatter(position + jitter, values, s=20, color="black", zorder=3)
    axes[0].set(
        xticks=range(len(order)),
        xticklabels=[
            "Discovered", "Terminal\nonly", "Upstream\nonly",
            "Topology\nrandom", "Fixed terminal\n+ random upstream",
        ],
        ylabel="Normalized faithfulness",
        ylim=(0, 1.08),
        title="Vector-score reconstruction",
    )
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].tick_params(axis="x", labelrotation=15)
    for label in axes[0].get_xticklabels():
        label.set_horizontalalignment("right")

    shown = behavior[
        ((behavior.mask_family == "Discovered full") & behavior.condition.isin(("circuit_only", "circuit_ablated")))
        | ((behavior.mask_family == "Terminal only") & behavior.condition.isin(("circuit_only", "circuit_ablated")))
        | behavior.mask_family.str.startswith("Matched random")
    ].copy()
    discovered = shown[shown.mask_family.eq("Discovered full")].set_index("condition")
    terminal = shown[shown.mask_family.eq("Terminal only")].set_index("condition")
    random = shown[shown.mask_family.str.startswith("Matched random")]
    x = np.arange(2)
    axes[1].bar(x - 0.18, [discovered.loc[c, "refusal_rate"] for c in ("circuit_only", "circuit_ablated")], width=0.36, color=colors["Discovered full"], label="Discovered")
    axes[1].bar(x + 0.18, [terminal.loc[c, "refusal_rate"] for c in ("circuit_only", "circuit_ablated")], width=0.36, color=colors["Terminal only"], label="Terminal only")
    axes[1].scatter(np.full(len(random), x[1]), random.refusal_rate, marker="x", s=34, color="black", label="Random ablations", zorder=4)
    normal = behavior.loc[behavior.mask_family.eq("Normal model"), "refusal_rate"].iloc[0]
    direction = behavior.loc[behavior.mask_family.eq("Global direction ablated"), "refusal_rate"].iloc[0]
    axes[1].axhline(normal, color="0.25", linestyle="--", linewidth=1, label="Normal")
    axes[1].axhline(direction, color="0.6", linestyle=":", linewidth=1, label="Direction ablated")
    axes[1].set(
        xticks=x,
        xticklabels=["Circuit only", "Circuit ablated"],
        ylabel="WildGuard refusal rate",
        ylim=(0, 1.08),
        title="Generated refusal behavior",
    )
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        handlelength=1.6,
        columnspacing=1.2,
    )
    outputs = []
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"figure_terminal_shell_decision.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    accounting = read_json(ACCOUNTING)
    if any(job["state"] != "COMPLETED" or job["exit_code"] != "0:0" for job in accounting["jobs"]):
        raise ValueError("Incomplete decisive-wave job")
    teacher_detail, teacher_summary, teacher_inputs = load_teacher_forcing()
    behavior, paired, behavior_inputs = load_behavior()
    outputs = save_tables(teacher_detail, teacher_summary, behavior, paired, accounting)
    outputs.extend(save_figure(teacher_detail, behavior))

    fixed = teacher_summary.set_index("mask_family").loc["Fixed terminal + random upstream"]
    discovered = teacher_summary.set_index("mask_family").loc["Discovered full"]
    terminal = behavior[
        behavior.mask_family.eq("Terminal only") & behavior.condition.eq("circuit_only")
    ].iloc[0]
    discovered_only = behavior[
        behavior.mask_family.eq("Discovered full") & behavior.condition.eq("circuit_only")
    ].iloc[0]
    conditional_discovered = (
        discovered.faithfulness_mean
        - teacher_summary.set_index("mask_family").loc["Terminal only"].faithfulness_mean
    ) / (
        1.0
        - teacher_summary.set_index("mask_family").loc["Terminal only"].faithfulness_mean
    )
    assessment = {
        "schema_version": 1,
        "status": "corrected_qwen_refusal_results_freeze",
        "terminal_cut_interpretation": (
            "Terminal-only ablation cuts every selected circuit path before the metric, so its "
            "similarity to full-circuit ablation is expected and is not an upstream-specificity test."
        ),
        "positive_result": (
            "The discovered upstream edge set specifically reconstructs the refusal-vector projection: "
            f"faithfulness {discovered.faithfulness_mean:.3f} versus "
            f"{fixed.faithfulness_mean:.3f} mean for fixed-terminal/random-upstream controls."
        ),
        "conditional_upstream_recovery": float(conditional_discovered),
        "behavioral_result": (
            f"Discovered circuit-only refusal is {discovered_only.refusal_rate:.3f} and circuit ablation "
            "strongly reverses refusal. Terminal-only behavior is saturated despite partial vector "
            f"faithfulness ({terminal.refusal_rate:.3f} refusal), so binary refusal is not an "
            "upstream-specificity metric on these prompts."
        ),
        "claim_supported": (
            "Qwen2.5-7B-Instruct contains an edge-sparse, causally validated, multilingual subgraph "
            "that reconstructs a shared refusal-vector representation; its discovered upstream "
            "organization outperforms topology-matched random alternatives, and circuit "
            "interventions strongly affect generated refusal."
        ),
        "claim_boundaries": [
            "edge-sparse, not node-sparse",
            "minimal only within the tested nested ranked family",
            "set-level necessity does not establish individual-edge necessity",
            "the layer-15 scalar is behaviorally relevant but not the sole behavioral bottleneck",
            "development results require judge/horizon audit and locked confirmation",
        ],
        "canonical_document": "docs/QWEN_REFUSAL_RESULTS_FREEZE.md",
        "gpu_cost": accounting["totals"],
    }
    assessment_path = OUTPUT / "assessment.json"
    assessment_path.write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    outputs.append(assessment_path)

    inputs = list(dict.fromkeys([ACCOUNTING, *teacher_inputs, *behavior_inputs]))
    manifest = {
        "schema_version": 1,
        "status": "frozen_corrected_terminal_cut_audit",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "language": "en",
        "split": "development",
        "requested_k": 3000,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(OUTPUT)): sha256(path) for path in outputs},
        "gpu_cost": accounting["totals"],
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(OUTPUT), "outputs": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()

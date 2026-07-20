#!/usr/bin/env python3
"""Freeze canonical Phase 2 development tables, figures, and input hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ROOT
from phase2_refusal_analysis import _paired_effects


DEFAULT_CONFIG = ROOT / "configs" / "phase2_qwen_refusal.json"
DEFAULT_JOBS = ROOT / "configs" / "phase2_qwen_refusal_jobs.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_metric_paths(results_root: Path) -> dict[str, Path]:
    return {
        "en->en": results_root / "en/circuit_evaluation_english_vector_small_to_large",
        "de->de": results_root / "de/circuit_evaluation_english_vector_small_to_large",
        "zh->zh": results_root / "zh/circuit_evaluation_english_vector_small_to_large",
        "th->th": results_root / "th/circuit_evaluation_english_vector_small_to_large",
        "en->de": results_root / "cross_language_english_vector/en_to_de",
        "en->zh": results_root / "cross_language_english_vector/en_to_zh",
        "en->th": results_root / "cross_language_english_vector/en_to_th",
    }


def _load_metrics(results_root: Path, phase2_root: Path):
    import pandas as pd

    rows = []
    input_paths: list[Path] = []
    for relationship, directory in _legacy_metric_paths(results_root).items():
        faith_path = directory / "faithfulness.csv"
        necessity_path = directory / "necessity.csv"
        input_paths.extend((faith_path, necessity_path))
        faith = pd.read_csv(faith_path).query(
            "method == 'eapig' and requested_k in [1000, 2000]"
        )
        necessity = pd.read_csv(necessity_path).query(
            "method == 'eapig' and requested_k in [1000, 2000]"
        )
        joined = faith.merge(
            necessity[["requested_k", "necessity"]], on="requested_k", validate="one_to_one"
        )
        for _, row in joined.iterrows():
            rows.append(
                {
                    "relationship": relationship,
                    "requested_k": int(row.requested_k),
                    "retained_edges": int(row.retained_edges),
                    "faithfulness": float(row.faithfulness),
                    "necessity": float(row.necessity),
                    "metric_source": "historical_test_offset0_legacy",
                }
            )
    metric_root = phase2_root / "metrics/historical_test_offset0"
    for path in sorted(metric_root.glob("source_*/*/aggregate.csv")):
        input_paths.append(path)
        relationship = f"{path.parts[-3].removeprefix('source_')}->{path.parts[-2]}"
        for _, row in pd.read_csv(path).iterrows():
            rows.append(
                {
                    "relationship": relationship,
                    "requested_k": int(row.requested_k),
                    "retained_edges": int(row.retained_edges),
                    "faithfulness": float(row.faithfulness),
                    "necessity": float(row.necessity),
                    "metric_source": "phase2_historical_test_offset0",
                }
            )
    frame = pd.DataFrame(rows)
    expected = {(relationship, k) for relationship in _legacy_metric_paths(results_root) for k in (1000, 2000, 3000, 4000, 5000)}
    actual = set(zip(frame.relationship, frame.requested_k))
    if actual != expected or frame.duplicated(["relationship", "requested_k"]).any():
        raise ValueError("Metric relationship/k grid is incomplete or duplicated")
    return frame, input_paths


def _load_behavior(phase2_root: Path, cfg: dict):
    import pandas as pd

    behavior_root = phase2_root / "behavior/development"
    paths = sorted(behavior_root.glob("source_*/*/instance_labels_no_text.csv"))
    if len(paths) != 7:
        raise ValueError(f"Expected seven behavior relationships, found {len(paths)}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if len(frame) not in (160, 224) or frame.parallel_id.nunique() != 32:
            raise ValueError(f"Unexpected behavior shape in {path}")
        if not frame.parse_ok.all():
            raise ValueError(f"WildGuard parse failure in {path}")
        forbidden = {"prompt", "instruction", "response", "completion", "raw_output"}
        if forbidden & set(frame.columns):
            raise ValueError(f"Saved text field in {path}")
        frame["source_language"] = path.parts[-3].removeprefix("source_")
        frames.append(frame)
    behavior = pd.concat(frames, ignore_index=True)
    controls = behavior[behavior.condition.isin(["normal", "direction_ablated"])].copy()
    if controls.groupby(["language", "parallel_id", "condition"]).refusal.nunique().max() > 1:
        raise ValueError("Conflicting deterministic control labels")
    controls = controls.drop_duplicates(["language", "parallel_id", "condition"])
    controls = controls.pivot(
        index=["language", "parallel_id"], columns="condition", values="refusal"
    ).reset_index()
    ablated = behavior[behavior.condition.eq("circuit_ablated")].merge(
        controls, on=["language", "parallel_id"], validate="many_to_one"
    )
    rows = []
    stats = cfg["statistics"]
    for (source, target, k), group in ablated.groupby(
        ["source_language", "language", "requested_k"]
    ):
        analysis_frame = group[["normal", "direction_ablated", "refusal"]].rename(
            columns={"refusal": "circuit_ablated"}
        )
        seed = int(stats["bootstrap_seed"]) + int(k) + sum(map(ord, source + target))
        effects = _paired_effects(
            analysis_frame,
            samples=int(stats["bootstrap_samples"]),
            seed=seed,
            confidence=float(stats["confidence_level"]),
        )
        rows.append(
            {
                "relationship": f"{source}->{target}",
                "source_language": source,
                "target_language": target,
                "requested_k": int(k),
                "retained_edges": int(group.retained_edges.iloc[0]),
                "edge_fraction": float(group.edge_fraction.iloc[0]),
                "circuit_ablated_prefix_refusal_rate": float(group.refusal_substring.mean()),
                **effects,
            }
        )
    curves = pd.DataFrame(rows)
    if len(curves) != 35:
        raise ValueError(f"Expected 35 behavior curve rows, found {len(curves)}")
    return behavior, curves, paths


def _family_summary(curves):
    import pandas as pd

    families = {
        "local": {"en->en", "de->de", "zh->zh", "th->th"},
        "one_english_circuit": {"en->en", "en->de", "en->zh", "en->th"},
    }
    rows = []
    for family, relationships in families.items():
        subset = curves[curves.relationship.isin(relationships)]
        for k, group in subset.groupby("requested_k"):
            rows.append(
                {
                    "family": family,
                    "requested_k": int(k),
                    "edge_fraction": float(group.edge_fraction.iloc[0]),
                    "faithfulness_mean": group.faithfulness.mean(),
                    "faithfulness_min": group.faithfulness.min(),
                    "necessity_mean": group.necessity.mean(),
                    "necessity_min": group.necessity.min(),
                    "normalized_refusal_reversal_mean": group.normalized_refusal_reversal.mean(),
                    "normalized_refusal_reversal_min": group.normalized_refusal_reversal.min(),
                }
            )
    return pd.DataFrame(rows)


def _write_latex_tables(output_dir: Path, curves, summary) -> list[Path]:
    headline = curves[
        curves.relationship.isin(["en->en", "en->de", "en->zh", "en->th"])
        & curves.requested_k.isin([3000, 5000])
    ][
        [
            "target_language",
            "requested_k",
            "faithfulness",
            "necessity",
            "normal_refusal_rate",
            "direction_ablated_refusal_rate",
            "circuit_ablated_refusal_rate",
            "normalized_refusal_reversal",
        ]
    ].copy()
    macro = summary[summary.family.eq("one_english_circuit")].copy()
    paths = [output_dir / "table_headline_english_circuit.tex", output_dir / "table_macro_size_curve.tex"]
    paths[0].write_text(
        headline.to_latex(index=False, float_format="%.3f", escape=True), encoding="utf-8"
    )
    paths[1].write_text(
        macro.to_latex(index=False, float_format="%.3f", escape=True), encoding="utf-8"
    )
    return paths


def _save_figures(output_dir: Path, curves, summary, structure_root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd

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
    labels = {"en": "English", "de": "German", "zh": "Chinese", "th": "Thai"}
    english = curves[curves.source_language.eq("en")].sort_values("requested_k")
    outputs: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    for target, group in english.groupby("target_language"):
        x = 100 * group.edge_fraction
        axes[0].plot(x, group.faithfulness, marker="o", color=colors[target], label=labels[target])
        axes[1].plot(x, group.necessity, marker="o", color=colors[target])
        axes[2].plot(x, group.normalized_refusal_reversal, marker="o", color=colors[target])
    for axis, title, ylabel in zip(
        axes,
        ("Representation sufficiency", "Representation necessity", "Generated behavior"),
        ("Faithfulness", "Necessity", "Normalized refusal reversal"),
    ):
        axis.axhline(0.8, color="0.55", linestyle="--", linewidth=0.9)
        axis.set(title=title, xlabel="Retained eligible edges (%)", ylabel=ylabel, ylim=(0.7, 1.05))
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="lower right")
    for suffix in ("pdf", "png"):
        path = output_dir / f"figure_english_circuit_size_curves.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.7), sharey=True, constrained_layout=True)
    for axis, target in zip(axes, ("en", "de", "zh", "th")):
        group = english[english.target_language.eq(target)]
        x = 100 * group.edge_fraction
        axis.plot(x, group.circuit_ablated_refusal_rate, marker="o", color=colors[target], label="Circuit ablated")
        axis.axhline(group.normal_refusal_rate.iloc[0], color="0.2", linestyle=":", label="Normal")
        axis.axhline(group.direction_ablated_refusal_rate.iloc[0], color="0.5", linestyle="--", label="Direction ablated")
        axis.set(title=labels[target], xlabel="Retained edges (%)", ylim=(-0.03, 1.03))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("WildGuard refusal rate")
    axes[-1].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"):
        path = output_dir / f"figure_raw_refusal_rates.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)

    overlap = pd.read_csv(structure_root / "edge_overlap_same_size.csv")
    overlap = overlap[
        overlap.left_language.eq("en")
        & overlap.right_language.isin(["de", "zh", "th"])
        & overlap.left_requested_k.isin([1000, 2000, 3000, 4000, 5000])
    ]
    composition = pd.read_csv(structure_root / "circuit_composition.csv")
    composition = composition[
        composition.language.eq("en")
        & composition.requested_k.isin([1000, 2000, 3000, 4000, 5000])
    ].sort_values("requested_k")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for target, group in overlap.groupby("right_language"):
        axes[0].plot(100 * group.left_edge_fraction, group.overlap_coefficient, marker="o", color=colors[target], label=labels[target])
    axes[0].set(xlabel="Retained eligible edges (%)", ylabel="Same-size edge overlap", ylim=(0.7, 0.9), title="English/non-English overlap")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    x = 100 * composition.edge_fraction
    axes[1].plot(x, 100 * composition.edge_fraction, marker="o", label="Eligible edges")
    axes[1].plot(x, 100 * composition.terminal_edges / 436, marker="o", label="Terminal edges")
    axes[1].plot(x, 100 * composition.logical_nodes / 436, marker="o", label="Logical nodes")
    axes[1].set(xlabel="Retained eligible edges (%)", ylabel="Fraction retained (%)", title="Edge versus component sparsity")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    for suffix in ("pdf", "png"):
        path = output_dir / f"figure_multilingual_structure_and_sparsity.{suffix}"
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

    import pandas as pd

    cfg = _read(args.config)
    results_root = ROOT / cfg["artifacts"]["results_root"]
    phase2_root = ROOT / cfg["artifacts"]["phase2_results_root"]
    output_dir = args.output_dir or phase2_root / "frozen_development_2026-07-19"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, metric_inputs = _load_metrics(results_root, phase2_root)
    behavior, behavior_curves, behavior_inputs = _load_behavior(phase2_root, cfg)
    curves = behavior_curves.merge(
        metrics,
        on=["relationship", "requested_k", "retained_edges"],
        validate="one_to_one",
    ).sort_values(["relationship", "requested_k"])
    summary = _family_summary(curves)

    curves_path = output_dir / "table_relationship_size_curves.csv"
    summary_path = output_dir / "table_multilingual_summary_by_size.csv"
    headline_path = output_dir / "table_headline_english_circuit.csv"
    transfer_path = output_dir / "table_english_vs_local_behavior.csv"
    composition_path = output_dir / "table_english_circuit_composition.csv"
    cost_path = output_dir / "table_job_costs.csv"
    curves.to_csv(curves_path, index=False)
    summary.to_csv(summary_path, index=False)
    curves[
        curves.relationship.isin(["en->en", "en->de", "en->zh", "en->th"])
        & curves.requested_k.isin([3000, 5000])
    ].to_csv(headline_path, index=False)

    transfer_rows = []
    for target in ("de", "zh", "th"):
        for k in (1000, 2000, 3000, 4000, 5000):
            local = curves[(curves.relationship == f"{target}->{target}") & (curves.requested_k == k)].iloc[0]
            source = curves[(curves.relationship == f"en->{target}") & (curves.requested_k == k)].iloc[0]
            local_labels = behavior[
                behavior.language.eq(target)
                & behavior.source_language.eq(target)
                & behavior.requested_k.eq(k)
                & behavior.condition.eq("circuit_ablated")
            ].set_index("parallel_id")
            source_labels = behavior[
                behavior.language.eq(target)
                & behavior.source_language.eq("en")
                & behavior.requested_k.eq(k)
                & behavior.condition.eq("circuit_ablated")
            ].set_index("parallel_id")
            ids = sorted(set(local_labels.index) & set(source_labels.index))
            transfer_rows.append(
                {
                    "target_language": target,
                    "requested_k": k,
                    "english_circuit_reversal": source.normalized_refusal_reversal,
                    "local_circuit_reversal": local.normalized_refusal_reversal,
                    "english_minus_local_reversal": source.normalized_refusal_reversal - local.normalized_refusal_reversal,
                    "intervention_label_agreement": (
                        local_labels.loc[ids, "refusal"].to_numpy()
                        == source_labels.loc[ids, "refusal"].to_numpy()
                    ).mean(),
                }
            )
    pd.DataFrame(transfer_rows).to_csv(transfer_path, index=False)

    structure_root = phase2_root / "structure"
    composition = pd.read_csv(structure_root / "circuit_composition.csv")
    composition[
        composition.language.eq("en")
        & composition.requested_k.isin([1000, 2000, 3000, 4000, 5000])
    ].to_csv(composition_path, index=False)
    jobs = _read(args.jobs)
    job_frame = pd.DataFrame(jobs["jobs"])
    job_frame["gpu_hours"] = job_frame.elapsed_seconds / 3600
    job_frame["estimated_sbu"] = job_frame.gpu_hours * int(jobs["billing_sbu_per_a100_hour"])
    job_frame.to_csv(cost_path, index=False)

    generated = [curves_path, summary_path, headline_path, transfer_path, composition_path, cost_path]
    generated.extend(_write_latex_tables(output_dir, curves, summary))
    generated.extend(_save_figures(output_dir, curves, summary, structure_root))

    input_paths = [args.config, args.jobs, *metric_inputs, *behavior_inputs]
    manifest = {
        "schema_version": 1,
        "status": "frozen_development",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": cfg["model"]["name"],
        "languages": cfg["dataset"]["languages"],
        "development_parallel_prompts_per_language": 32,
        "provisional_minimal_passing_k": 3000,
        "high_effect_sensitivity_k": 5000,
        "minimality_scope": "smallest passing tested member of the frozen nested score-ranked family",
        "inputs": {str(path.relative_to(ROOT)): _sha256(path) for path in input_paths},
        "outputs": {str(path.relative_to(output_dir)): _sha256(path) for path in generated},
        "gpu_cost": {
            "elapsed_a100_hours": float(job_frame.gpu_hours.sum()),
            "estimated_sbu": float(job_frame.estimated_sbu.sum()),
        },
        "known_gates_before_confirmation": [
            "multilingual judge/truncation audit",
            "terminal-only/upstream-only/topology-matched random controls",
            "circuit-only behavioral sufficiency",
            "harmless-prompt specificity",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "outputs": len(generated), "gpu_cost": manifest["gpu_cost"]}, indent=2))


if __name__ == "__main__":
    main()

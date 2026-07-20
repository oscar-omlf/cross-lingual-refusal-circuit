#!/usr/bin/env python3
"""Aggregate Phase 2 refusal curves, bootstrap intervals, and candidate k values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


DEFAULT_CONFIG = ROOT / "configs" / "phase2_qwen_refusal.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _interval(values, confidence: float):
    import numpy as np

    alpha = 1.0 - confidence
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(np.quantile(finite, [alpha / 2, 1 - alpha / 2]).tolist())


def _paired_effects(frame, *, samples: int, seed: int, confidence: float) -> dict:
    import numpy as np

    columns = ["normal", "direction_ablated", "circuit_only", "circuit_ablated"]
    frame = frame.copy()
    if "circuit_only" not in frame:
        frame["circuit_only"] = np.nan
    has_circuit_only = bool(frame["circuit_only"].notna().all())
    values = frame[columns].astype(float).to_numpy()
    normal, global_ablated, circuit_only, circuit_ablated = values.T
    normal_refusers = normal == 1

    def metrics(array):
        n = array[..., 0]
        g = array[..., 1]
        only = array[..., 2]
        ablated = array[..., 3]
        global_effect = np.mean(n - g, axis=-1)
        circuit_effect = np.mean(n - ablated, axis=-1)
        normalized = np.divide(
            circuit_effect,
            global_effect,
            out=np.full_like(circuit_effect, np.nan, dtype=float),
            where=np.abs(global_effect) > 1e-8,
        )
        denominator = np.sum(n == 1, axis=-1)
        reversal = np.divide(
            np.sum((n == 1) & (ablated == 0), axis=-1),
            denominator,
            out=np.full_like(global_effect, np.nan, dtype=float),
            where=denominator > 0,
        )
        retention = np.divide(
            np.sum((n == 1) & (only == 1), axis=-1),
            denominator,
            out=np.full_like(global_effect, np.nan, dtype=float),
            where=denominator > 0,
        )
        return global_effect, circuit_effect, normalized, reversal, retention

    point_values = metrics(values[None, :, :])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap_values = metrics(values[indices])
    names = (
        "global_direction_refusal_effect",
        "circuit_ablation_refusal_effect",
        "normalized_refusal_reversal",
        "conditional_refusal_reversal",
        "circuit_only_refusal_retention",
    )
    output = {
        "n_complete": len(values),
        "n_normal_refusers": int(normal_refusers.sum()),
        "normal_refusal_rate": float(normal.mean()),
        "direction_ablated_refusal_rate": float(global_ablated.mean()),
        "circuit_only_refusal_rate": (
            float(circuit_only.mean()) if has_circuit_only else float("nan")
        ),
        "circuit_ablated_refusal_rate": float(circuit_ablated.mean()),
    }
    for name, point, boot in zip(names, point_values, bootstrap_values):
        if name == "circuit_only_refusal_retention" and not has_circuit_only:
            output[name] = float("nan")
            output[f"{name}_ci_low"] = float("nan")
            output[f"{name}_ci_high"] = float("nan")
            continue
        low, high = _interval(boot, confidence)
        output[name] = float(point[0])
        output[f"{name}_ci_low"] = low
        output[f"{name}_ci_high"] = high
    return output


def _find_smallest_stable(group, cfg: dict):
    rule = cfg["selection_rule"]
    ordered = group.sort_values("requested_k").reset_index(drop=True)
    passes = (
        (ordered["faithfulness"] >= rule["minimum_projection_faithfulness"])
        & (ordered["necessity"] >= rule["minimum_projection_necessity"])
        & (
            ordered["normalized_refusal_reversal"]
            >= rule["minimum_normalized_refusal_reversal_point"]
        )
        & (
            ordered["circuit_only_refusal_retention"]
            >= rule["minimum_circuit_only_refusal_retention_point"]
        )
    )
    for index, passed in enumerate(passes.tolist()):
        if not passed:
            continue
        if rule["require_next_larger_k_to_pass"] and (
            index + 1 >= len(passes) or not bool(passes.iloc[index + 1])
        ):
            continue
        row = ordered.iloc[index]
        return {
            "requested_k": int(row["requested_k"]),
            "retained_edges": int(row["retained_edges"]),
            "edge_fraction": float(row["edge_fraction"]),
            "next_larger_k_also_passed": bool(rule["require_next_larger_k_to_pass"]),
        }
    return None


def _simultaneous_bands(group, *, samples: int, seed: int, confidence: float):
    """Joint non-studentized paired-bootstrap bands across every k in a curve."""
    import numpy as np

    k_values = sorted(group["requested_k"].unique())
    common_ids = None
    indexed = {}
    for k in k_values:
        frame = group[group["requested_k"] == k].set_index("parallel_id")
        indexed[k] = frame
        ids = set(frame.index)
        common_ids = ids if common_ids is None else common_ids & ids
    ordered_ids = sorted(common_ids or ())
    if not ordered_ids:
        return {}
    arrays = {
        k: indexed[k].loc[ordered_ids][
            ["normal", "direction_ablated", "circuit_only", "circuit_ablated"]
        ].astype(float).to_numpy()
        for k in k_values
    }
    has_circuit_only = all(np.isfinite(array[:, 2]).all() for array in arrays.values())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ordered_ids), size=(samples, len(ordered_ids)))
    boot_normalized = []
    boot_retention = []
    points_normalized = []
    points_retention = []
    for k in k_values:
        values = arrays[k]
        n, g, only, ablated = values.T
        global_effect = np.mean(n - g)
        points_normalized.append(
            np.mean(n - ablated) / global_effect if abs(global_effect) > 1e-8 else np.nan
        )
        normal_count = np.sum(n == 1)
        points_retention.append(
            np.sum((n == 1) & (only == 1)) / normal_count
            if normal_count and has_circuit_only
            else np.nan
        )
        sampled = values[indices]
        bn, bg, bonly, bablated = sampled.transpose(2, 0, 1)
        global_boot = np.mean(bn - bg, axis=1)
        circuit_boot = np.mean(bn - bablated, axis=1)
        boot_normalized.append(
            np.divide(
                circuit_boot,
                global_boot,
                out=np.full(samples, np.nan),
                where=np.abs(global_boot) > 1e-8,
            )
        )
        denominators = np.sum(bn == 1, axis=1)
        boot_retention.append(
            np.divide(
                np.sum((bn == 1) & (bonly == 1), axis=1),
                denominators,
                out=np.full(samples, np.nan),
                where=(denominators > 0) & has_circuit_only,
            )
        )
    output = {}
    for name, points, bootstrap in (
        ("normalized_refusal_reversal", np.asarray(points_normalized), np.stack(boot_normalized, axis=1)),
        ("circuit_only_refusal_retention", np.asarray(points_retention), np.stack(boot_retention, axis=1)),
    ):
        if not np.isfinite(points).any():
            output[name] = {k: (float("nan"), float("nan")) for k in k_values}
            continue
        deviations = np.abs(bootstrap - points[None, :])
        maxima = np.nanmax(deviations, axis=1)
        critical = float(np.nanquantile(maxima, confidence))
        output[name] = {
            k: (float(point - critical), float(point + critical))
            for k, point in zip(k_values, points)
        }
    return output


def _transfer_effects(source, local, *, samples: int, seed: int, confidence: float):
    """Paired source-circuit effect relative to target language's local circuit."""
    import numpy as np

    source = source.set_index("parallel_id")
    local = local.set_index("parallel_id")
    ids = sorted(set(source.index) & set(local.index))
    if not ids:
        return None
    normal = source.loc[ids, "normal"].astype(float).to_numpy()
    source_ablated = source.loc[ids, "circuit_ablated"].astype(float).to_numpy()
    local_ablated = local.loc[ids, "circuit_ablated"].astype(float).to_numpy()
    source_only = source.loc[ids, "circuit_only"].astype(float).to_numpy()
    local_only = local.loc[ids, "circuit_only"].astype(float).to_numpy()
    has_circuit_only = bool(
        np.isfinite(source_only).all() and np.isfinite(local_only).all()
    )

    def calculate(indices):
        n = normal[indices]
        source_effect = np.mean(n - source_ablated[indices], axis=-1)
        local_effect = np.mean(n - local_ablated[indices], axis=-1)
        effect_ratio = np.divide(
            source_effect,
            local_effect,
            out=np.full_like(source_effect, np.nan, dtype=float),
            where=np.abs(local_effect) > 1e-8,
        )
        normal_count = np.sum(n == 1, axis=-1)
        source_retention = np.divide(
            np.sum((n == 1) & (source_only[indices] == 1), axis=-1),
            normal_count,
            out=np.full_like(source_effect, np.nan, dtype=float),
            where=(normal_count > 0) & has_circuit_only,
        )
        local_retention = np.divide(
            np.sum((n == 1) & (local_only[indices] == 1), axis=-1),
            normal_count,
            out=np.full_like(source_effect, np.nan, dtype=float),
            where=(normal_count > 0) & has_circuit_only,
        )
        retention_ratio = np.divide(
            source_retention,
            local_retention,
            out=np.full_like(source_retention, np.nan, dtype=float),
            where=np.abs(local_retention) > 1e-8,
        )
        return source_effect, local_effect, effect_ratio, source_retention, local_retention, retention_ratio

    point = calculate(np.arange(len(ids))[None, :])
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, len(ids), size=(samples, len(ids)))
    bootstrap = calculate(bootstrap_indices)
    names = (
        "source_ablation_effect",
        "local_ablation_effect",
        "ablation_transfer_relative_to_local",
        "source_circuit_only_retention",
        "local_circuit_only_retention",
        "circuit_only_retention_relative_to_local",
    )
    output = {"n_paired": len(ids)}
    for name, point_values, boot_values in zip(names, point, bootstrap):
        if "retention" in name and not has_circuit_only:
            output[name] = float("nan")
            output[f"{name}_ci_low"] = float("nan")
            output[f"{name}_ci_high"] = float("nan")
            continue
        low, high = _interval(boot_values, confidence)
        output[name] = float(point_values[0])
        output[f"{name}_ci_low"] = low
        output[f"{name}_ci_high"] = high
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument(
        "--metric-split",
        default=None,
        help="Metric result directory to join; defaults to the behavioral split.",
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    cfg = _read(args.config)
    phase2_root = args.results_root or _resolve(cfg["artifacts"]["phase2_results_root"])
    behavior_files = sorted((phase2_root / "behavior" / args.split).glob("*/*/instance_labels_no_text.csv"))
    metric_split = args.metric_split or args.split
    metric_files = sorted((phase2_root / "metrics" / metric_split).glob("*/*/aggregate.csv"))
    if not behavior_files:
        raise FileNotFoundError(f"No behavior instance tables under {phase2_root}")
    if not metric_files:
        raise FileNotFoundError(f"No metric aggregate tables under {phase2_root}")
    behavior = pd.concat([pd.read_csv(path) for path in behavior_files], ignore_index=True)
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    behavior = behavior[behavior["parse_ok"] == True].copy()  # noqa: E712

    controls = behavior[behavior["condition"].isin(["normal", "direction_ablated"])].copy()
    # Deterministic generation may have repeated controls across circuit jobs.
    conflict = (
        controls.groupby(["language", "split", "parallel_id", "condition"])["refusal"]
        .nunique()
        .max()
    )
    if conflict and conflict > 1:
        raise ValueError("Repeated deterministic control generations received inconsistent refusal labels")
    controls = controls.drop_duplicates(["language", "split", "parallel_id", "condition"])
    control_wide = controls.pivot(
        index=["language", "split", "parallel_id"], columns="condition", values="refusal"
    ).reset_index()

    intervention = behavior[behavior["condition"].isin(["circuit_only", "circuit_ablated"])]
    wide = intervention.pivot_table(
        index=[
            "language",
            "split",
            "circuit_label",
            "requested_k",
            "retained_edges",
            "edge_fraction",
            "parallel_id",
        ],
        columns="condition",
        values="refusal",
        aggfunc="first",
    ).reset_index()
    joined = wide.merge(control_wide, on=["language", "split", "parallel_id"], how="inner")
    if "circuit_only" not in joined:
        joined["circuit_only"] = np.nan
    joined = joined.dropna(subset=["normal", "direction_ablated", "circuit_ablated"])
    curve_rows = []
    stats_cfg = cfg["statistics"]
    for key, group in joined.groupby(
        ["language", "split", "circuit_label", "requested_k", "retained_edges", "edge_fraction"]
    ):
        row = dict(
            zip(
                ["language", "split", "circuit_label", "requested_k", "retained_edges", "edge_fraction"],
                key,
            )
        )
        group_seed = int(stats_cfg["bootstrap_seed"]) + int(row["requested_k"]) + sum(
            ord(char) for char in f"{row['language']}:{row['circuit_label']}"
        )
        row.update(
            _paired_effects(
                group,
                samples=int(stats_cfg["bootstrap_samples"]),
                seed=group_seed,
                confidence=float(stats_cfg["confidence_level"]),
            )
        )
        curve_rows.append(row)
    curves = pd.DataFrame(curve_rows)
    metric_columns = [
        "language",
        "circuit_label",
        "requested_k",
        "faithfulness",
        "necessity",
    ]
    merged = curves.merge(
        metrics[metric_columns],
        on=["language", "circuit_label", "requested_k"],
    )

    transfer_rows = []
    for (language, label, k), source_group in joined.groupby(
        ["language", "circuit_label", "requested_k"]
    ):
        local_label = f"source_{language}"
        local_group = joined[
            (joined["language"] == language)
            & (joined["circuit_label"] == local_label)
            & (joined["requested_k"] == k)
        ]
        if local_group.empty:
            continue
        transfer = _transfer_effects(
            source_group,
            local_group,
            samples=int(stats_cfg["bootstrap_samples"]),
            seed=int(stats_cfg["bootstrap_seed"])
            + int(k)
            + sum(ord(char) for char in f"transfer:{language}:{label}"),
            confidence=float(stats_cfg["confidence_level"]),
        )
        if transfer is None:
            continue
        source_metric = metrics[
            (metrics["language"] == language)
            & (metrics["circuit_label"] == label)
            & (metrics["requested_k"] == k)
        ]
        local_metric = metrics[
            (metrics["language"] == language)
            & (metrics["circuit_label"] == local_label)
            & (metrics["requested_k"] == k)
        ]
        if len(source_metric) == 1 and len(local_metric) == 1:
            local_f = float(local_metric["faithfulness"].iloc[0])
            local_n = float(local_metric["necessity"].iloc[0])
            transfer["projection_faithfulness_relative_to_local"] = (
                float(source_metric["faithfulness"].iloc[0]) / local_f
                if abs(local_f) > 1e-8
                else float("nan")
            )
            transfer["projection_necessity_relative_to_local"] = (
                float(source_metric["necessity"].iloc[0]) / local_n
                if abs(local_n) > 1e-8
                else float("nan")
            )
        transfer_rows.append(
            {
                "target_language": language,
                "source_circuit_label": label,
                "local_circuit_label": local_label,
                "requested_k": int(k),
                **transfer,
            }
        )

    # Joint paired resampling supplies simultaneous, non-studentized bands over
    # all k in a language/family curve.
    for column in ("normalized_refusal_reversal", "circuit_only_refusal_retention"):
        merged[f"{column}_sim_ci_low"] = np.nan
        merged[f"{column}_sim_ci_high"] = np.nan
    for (language, label), group in joined.groupby(["language", "circuit_label"]):
        band_seed = int(stats_cfg["bootstrap_seed"]) + sum(
            ord(char) for char in f"joint:{language}:{label}"
        )
        bands = _simultaneous_bands(
            group,
            samples=int(stats_cfg["bootstrap_samples"]),
            seed=band_seed,
            confidence=float(stats_cfg["confidence_level"]),
        )
        for column, by_k in bands.items():
            for k, (low, high) in by_k.items():
                mask = (
                    (merged["language"] == language)
                    & (merged["circuit_label"] == label)
                    & (merged["requested_k"] == k)
                )
                merged.loc[mask, f"{column}_sim_ci_low"] = low
                merged.loc[mask, f"{column}_sim_ci_high"] = high

    candidates = []
    for (language, label), group in merged.groupby(["language", "circuit_label"]):
        selected = _find_smallest_stable(group, cfg)
        candidates.append(
            {
                "language": language,
                "circuit_label": label,
                "candidate": selected,
            }
        )
    common = {}
    for label in sorted(merged["circuit_label"].unique()):
        label_rows = [row for row in candidates if row["circuit_label"] == label]
        if {row["language"] for row in label_rows} == set(cfg["dataset"]["languages"]) and all(
            row["candidate"] for row in label_rows
        ):
            common[label] = max(row["candidate"]["requested_k"] for row in label_rows)

    output_dir = args.output_dir or phase2_root / "analysis" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["circuit_label", "language", "requested_k"]).to_csv(
        output_dir / "behavior_projection_size_curves.csv", index=False
    )
    pd.DataFrame(transfer_rows).sort_values(
        ["source_circuit_label", "target_language", "requested_k"]
    ).to_csv(output_dir / "cross_language_functional_transfer.csv", index=False)
    selection = {
        "status": (
            "provisional_pending_benign_and_random_controls"
            if args.split == "development"
            else "confirmation_report_only_no_reselection"
        ),
        "config": str(args.config),
        "split": args.split,
        "per_language_candidates": candidates,
        "common_k_if_all_languages_pass": common,
        "important": (
            "Development candidates are not globally minimal circuits. They are the smallest stable "
            "passing points within the frozen nested EAP-IG family. Confirmation must not reselect k."
        ),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for label, subset in merged.groupby("circuit_label"):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
        for language, rows in subset.groupby("language"):
            rows = rows.sort_values("edge_fraction")
            x = rows["edge_fraction"].to_numpy(dtype=float) * 100
            axes[0].plot(x, rows["normalized_refusal_reversal"], marker="o", label=language)
            axes[0].fill_between(
                x,
                rows["normalized_refusal_reversal_sim_ci_low"].to_numpy(dtype=float),
                rows["normalized_refusal_reversal_sim_ci_high"].to_numpy(dtype=float),
                alpha=0.12,
            )
            axes[1].plot(x, rows["circuit_only_refusal_retention"], marker="o", label=language)
            axes[1].fill_between(
                x,
                rows["circuit_only_refusal_retention_sim_ci_low"].to_numpy(dtype=float),
                rows["circuit_only_refusal_retention_sim_ci_high"].to_numpy(dtype=float),
                alpha=0.12,
            )
        axes[0].set_ylabel("refusal reversal / global-direction effect")
        axes[1].set_ylabel("circuit-only refusal retention")
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlabel("retained eligible edges (%)")
            axis.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            axis.legend()
        figure.suptitle(label)
        figure.tight_layout()
        safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
        figure.savefig(output_dir / f"behavior_vs_size_{safe_label}.png", dpi=180)
        plt.close(figure)
    print(f"Saved Phase 2 curve analysis under {output_dir}")


if __name__ == "__main__":
    main()

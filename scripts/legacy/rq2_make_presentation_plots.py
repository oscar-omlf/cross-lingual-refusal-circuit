#!/usr/bin/env python3
"""Make presentation plots for the RQ2 multilingual refusal-circuit results.

The script consumes only aggregate CSV/JSON results. It does not read prompts or
generated completions, so it is safe to run/share without exposing harmful text.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANGS = ["en", "de", "zh", "th"]
LANG_LABELS = {"en": "English", "de": "German", "zh": "Chinese", "th": "Thai"}
SHARED_KINDS = {
    "intersection_en_de_zh_th": "Strict shared core",
    "majority_at_least_3_of_4": "Majority core",
    "consensus_mean_scores": "Consensus circuit",
}
COLORS = {
    "Local discovered": "#1f77b4",
    "English transfer": "#ff7f0e",
    "Strict shared core": "#2ca02c",
    "Majority core": "#9467bd",
    "Consensus circuit": "#d62728",
    "Matched random": "#7f7f7f",
}
MARKERS = {
    "Local discovered": "o",
    "English transfer": "s",
    "Strict shared core": "^",
    "Majority core": "D",
    "Consensus circuit": "P",
    "Matched random": "x",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/qwen2.5-7b-instruct/polyrefuse"),
        help="Base result directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to BASE_DIR/presentation_plots.",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png")
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(svg_path)
    # Matplotlib emits SVG path data with trailing spaces; normalize it so
    # `git diff --check` stays useful.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def load_eval_rows(base_dir: Path, metric: str) -> pd.DataFrame:
    """Load faithfulness/necessity rows for local, transfer, and shared circuits."""

    rows: list[dict] = []
    value_col = metric

    # Local independently discovered circuits.
    for lang in LANGS:
        path = (
            base_dir
            / lang
            / "circuit_evaluation_english_vector_small_to_large"
            / f"{metric}.csv"
        )
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            family = "Local discovered" if row["method"] == "eapig" else "Matched random"
            rows.append(
                {
                    "family": family,
                    "language": lang,
                    "requested_k": int(row["requested_k"]),
                    "retained_edges": int(row["retained_edges"]),
                    "value": float(row[value_col]),
                    "source": "local",
                }
            )

    # English circuit transferred to non-English languages.
    for lang in ["de", "zh", "th"]:
        path = base_dir / "cross_language_english_vector" / f"en_to_{lang}" / f"{metric}.csv"
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            family = "English transfer" if row["method"] == "eapig" else "Matched random"
            rows.append(
                {
                    "family": family,
                    "language": lang,
                    "requested_k": int(row["requested_k"]),
                    "retained_edges": int(row["retained_edges"]),
                    "value": float(row[value_col]),
                    "source": "transfer",
                }
            )

    # Shared/intersection/consensus circuit evaluations.
    for shared_dir, label in SHARED_KINDS.items():
        for lang in LANGS:
            path = (
                base_dir
                / "shared_circuit_evaluation_english_vector"
                / shared_dir
                / lang
                / f"{metric}.csv"
            )
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                family = label if row["method"] == "eapig" else "Matched random"
                rows.append(
                    {
                        "family": family,
                        "language": lang,
                        "requested_k": int(row["requested_k"]),
                        "retained_edges": int(row["retained_edges"]),
                        "value": float(row[value_col]),
                        "source": shared_dir,
                    }
                )

    return pd.DataFrame(rows)


def summarize_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["family", "requested_k"], as_index=False)
        .agg(
            mean=("value", "mean"),
            min=("value", "min"),
            max=("value", "max"),
            mean_edges=("retained_edges", "mean"),
        )
        .sort_values(["family", "requested_k"])
    )


def plot_metric_curves(base_dir: Path, out_dir: Path, metric: str) -> None:
    df = load_eval_rows(base_dir, metric)
    summary = summarize_for_plot(df)

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    order = [
        "Local discovered",
        "English transfer",
        "Strict shared core",
        "Majority core",
        "Consensus circuit",
        "Matched random",
    ]

    for family in order:
        sub = summary[summary["family"] == family]
        if sub.empty:
            continue
        linestyle = "--" if family == "Matched random" else "-"
        linewidth = 1.6 if family == "Matched random" else 2.2
        alpha = 0.8 if family == "Matched random" else 1.0
        ax.plot(
            sub["requested_k"],
            sub["mean"],
            label=family,
            color=COLORS[family],
            marker=MARKERS[family],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
        if family != "Matched random":
            ax.fill_between(
                sub["requested_k"].to_numpy(dtype=float),
                sub["min"].to_numpy(dtype=float),
                sub["max"].to_numpy(dtype=float),
                color=COLORS[family],
                alpha=0.11,
                linewidth=0,
            )

    ax.axhline(1.0, color="black", linewidth=1, alpha=0.35)
    ax.set_xscale("log")
    ticks = [10, 25, 50, 100, 200, 500, 1000, 2000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("Top-k edges")
    ax.set_ylabel(metric.capitalize())
    title = "Faithfulness of multilingual refusal circuits" if metric == "faithfulness" else "Necessity of multilingual refusal circuits"
    ax.set_title(title)
    if metric == "faithfulness":
        ax.set_ylim(-0.03, 1.02)
    else:
        ax.set_ylim(-0.05, 1.10)
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, out_dir, f"{metric}_vs_k")


def plot_k1000_summary(base_dir: Path, out_dir: Path, metric: str) -> None:
    df = load_eval_rows(base_dir, metric)
    df = df[(df["requested_k"] == 1000) & (df["family"] != "Matched random")]
    families = ["Local discovered", "English transfer", "Strict shared core", "Majority core", "Consensus circuit"]
    langs = LANGS
    width = 0.16
    x = np.arange(len(langs))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for idx, family in enumerate(families):
        vals = []
        for lang in langs:
            sub = df[(df["family"] == family) & (df["language"] == lang)]
            vals.append(np.nan if sub.empty else float(sub["value"].mean()))
        offset = (idx - (len(families) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=family, color=COLORS[family], alpha=0.88)

    ax.axhline(1.0, color="black", linewidth=1, alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS[l] for l in langs])
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} at k=1000 by language")
    if metric == "faithfulness":
        ax.set_ylim(0, 1.02)
    else:
        ax.set_ylim(0, 1.10)
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, out_dir, f"{metric}_k1000_by_language")


def plot_structural_overlap(base_dir: Path, out_dir: Path) -> None:
    path = base_dir / "comparisons_english_vector" / "eapig_circuit_structural_overlap.csv"
    df = pd.read_csv(path)
    observed = df[df["baseline"] == "observed"]
    random = df[df["baseline"].astype(str).str.startswith("random_seed")]

    obs_summary = (
        observed.groupby("requested_k", as_index=False)
        .agg(
            mean_iou=("iou", "mean"),
            min_iou=("iou", "min"),
            max_iou=("iou", "max"),
            mean_overlap=("overlap_coefficient", "mean"),
            mean_weighted_jaccard=("weighted_jaccard_abs_scores", "mean"),
        )
        .sort_values("requested_k")
    )
    rand_summary = random.groupby("requested_k", as_index=False).agg(mean_iou=("iou", "mean"))

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.plot(obs_summary["requested_k"], obs_summary["mean_iou"], marker="o", linewidth=2.3, label="Observed IoU")
    ax.fill_between(
        obs_summary["requested_k"].to_numpy(dtype=float),
        obs_summary["min_iou"].to_numpy(dtype=float),
        obs_summary["max_iou"].to_numpy(dtype=float),
        alpha=0.13,
        linewidth=0,
    )
    ax.plot(obs_summary["requested_k"], obs_summary["mean_overlap"], marker="s", linewidth=2.1, label="Overlap coefficient")
    ax.plot(
        obs_summary["requested_k"],
        obs_summary["mean_weighted_jaccard"],
        marker="^",
        linewidth=2.1,
        label="Weighted Jaccard",
    )
    ax.plot(rand_summary["requested_k"], rand_summary["mean_iou"], "--", color="#7f7f7f", marker="x", label="Random IoU")
    ax.set_xscale("log")
    ticks = [10, 25, 50, 100, 200, 500, 1000, 2000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Top-k edges")
    ax.set_ylabel("Overlap")
    ax.set_title("Independent language circuits overlap far above random")
    ax.legend(frameon=False)
    save_figure(fig, out_dir, "structural_overlap_vs_k")


def heatmap(
    matrix: pd.DataFrame,
    title: str,
    out_dir: Path,
    stem: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    arr = matrix.loc[LANGS, LANGS].to_numpy(dtype=float)
    im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(LANGS)))
    ax.set_yticks(np.arange(len(LANGS)))
    ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_yticklabels([l.upper() for l in LANGS])
    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            val = arr[i, j]
            text_color = "white" if val < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_dir, stem)


def plot_iou_heatmap(base_dir: Path, out_dir: Path, k: int = 1000) -> None:
    path = base_dir / "comparisons_english_vector" / "eapig_circuit_structural_overlap.csv"
    df = pd.read_csv(path)
    df = df[(df["baseline"] == "observed") & (df["requested_k"] == k)]
    mat = pd.DataFrame(np.eye(len(LANGS)), index=LANGS, columns=LANGS)
    for _, row in df.iterrows():
        l, r = row["left_language"], row["right_language"]
        mat.loc[l, r] = mat.loc[r, l] = float(row["iou"])
    heatmap(mat, f"Pairwise circuit IoU at k={k}", out_dir, f"pairwise_iou_heatmap_k{k}", vmin=0.55, vmax=1.0)


def plot_score_corr_heatmaps(base_dir: Path, out_dir: Path) -> None:
    path = base_dir / "comparisons_english_vector" / "eapig_edge_score_correlations.csv"
    df = pd.read_csv(path)
    for col, label, stem in [
        ("spearman_score_correlation", "Spearman edge-score correlation", "edge_score_spearman_heatmap"),
        ("pearson_score_correlation", "Pearson edge-score correlation", "edge_score_pearson_heatmap"),
    ]:
        mat = pd.DataFrame(np.eye(len(LANGS)), index=LANGS, columns=LANGS)
        for _, row in df.iterrows():
            l, r = row["left_language"], row["right_language"]
            mat.loc[l, r] = mat.loc[r, l] = float(row[col])
        heatmap(mat, label, out_dir, stem, vmin=0.75 if "Spearman" in label else 0.95, vmax=1.0, cmap="magma")


def plot_shared_core_sizes(base_dir: Path, out_dir: Path) -> None:
    path = base_dir / "shared_circuits" / "circuit_discovery_english_vector" / "eapig_intersection_summary.csv"
    df = pd.read_csv(path)
    strict = df[(df["languages"] == "en_de_zh_th") & (df["core_type"] == "strict_intersection")]
    majority = df[(df["languages"] == "en_de_zh_th") & (df["core_type"] == "majority_at_least_3")]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(strict["requested_k"], strict["intersection_edges"], marker="o", linewidth=2.2, label="Strict shared core")
    ax.plot(majority["requested_k"], majority["intersection_edges"], marker="s", linewidth=2.2, label="Majority ≥3/4 core")
    ax.plot(strict["requested_k"], strict["union_edges"], marker="^", linewidth=2.0, label="Union of local circuits")
    ax.plot(strict["requested_k"], strict["requested_k"], "--", color="#7f7f7f", marker="x", label="Consensus requested k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [10, 25, 50, 100, 200, 500, 1000, 2000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("Local top-k circuit size")
    ax.set_ylabel("Edges")
    ax.set_title("Shared-core circuit sizes")
    ax.legend(frameon=False)
    save_figure(fig, out_dir, "shared_core_sizes")


def load_circuit_edges(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data["edges"]
    return data


def plot_target_layer_distribution(base_dir: Path, out_dir: Path, k: int = 1000) -> None:
    circuit_base = base_dir / "shared_circuits" / "circuit_discovery_english_vector"
    paths = {
        "Strict shared core": circuit_base
        / "intersection_en_de_zh_th"
        / "circuits_greedy_no_parent_prune_small_to_large"
        / f"eapig_top_{k}.json",
        "Majority core": circuit_base
        / "majority_at_least_3_of_4"
        / "circuits_greedy_no_parent_prune_small_to_large"
        / f"eapig_top_{k}.json",
        "Consensus circuit": circuit_base
        / "consensus_mean_scores"
        / "circuits_greedy_no_parent_prune_small_to_large"
        / f"eapig_top_{k}.json",
    }

    counts_by_family: dict[str, Counter] = {}
    all_layers: set[int] = set()
    for label, path in paths.items():
        edges = load_circuit_edges(path)
        counts = Counter(int(edge["target"]["layer"]) for edge in edges)
        counts_by_family[label] = counts
        all_layers.update(counts)

    layers = np.array(sorted(all_layers))
    width = 0.26
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for idx, label in enumerate(paths):
        vals = np.array([counts_by_family[label].get(int(layer), 0) for layer in layers])
        offset = (idx - 1) * width
        ax.bar(x + offset, vals, width=width, label=label, color=COLORS[label], alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Target component layer")
    ax.set_ylabel("Selected edges")
    ax.set_title(f"Shared-circuit target-layer distribution at k={k}")
    ax.legend(frameon=False)
    save_figure(fig, out_dir, f"target_layer_distribution_k{k}")


def write_index(out_dir: Path) -> None:
    plots = sorted(p.name for p in out_dir.glob("*.png"))
    lines = [
        "# RQ2 presentation plots",
        "",
        "Generated from aggregate CSV/JSON results. PNG files are convenient for slides; SVG files are editable/vector versions.",
        "",
    ]
    for name in plots:
        lines.append(f"- `{name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    out_dir = args.out_dir or base_dir / "presentation_plots"
    setup_style()

    for metric in ["faithfulness", "necessity"]:
        plot_metric_curves(base_dir, out_dir, metric)
        plot_k1000_summary(base_dir, out_dir, metric)

    plot_structural_overlap(base_dir, out_dir)
    plot_iou_heatmap(base_dir, out_dir, k=1000)
    plot_score_corr_heatmaps(base_dir, out_dir)
    plot_shared_core_sizes(base_dir, out_dir)
    plot_target_layer_distribution(base_dir, out_dir, k=1000)
    write_index(out_dir)
    print(f"Wrote presentation plots to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create paper- and poster-ready figures for the completed RQ1 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/gemma-2-2b-it/en/rq1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gemma-2-2b-it/en/rq1/figures"))
    parser.add_argument(
        "--graph-results-dir",
        type=Path,
        default=None,
        help="Directory containing score/circuit files when plotting evaluation results elsewhere.",
    )
    parser.add_argument("--skip-graph", action="store_true", help="Generate statistical figures only.")
    parser.add_argument("--circuit-k", type=int, default=1000)
    parser.add_argument("--graph-top-edges", type=int, default=80)
    return parser.parse_args()


def _save(figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")


def _style(*, poster: bool) -> None:
    import matplotlib.pyplot as plt

    scale = 1.55 if poster else 1.0
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10 * scale,
            "axes.titlesize": 12 * scale,
            "axes.labelsize": 11 * scale,
            "legend.fontsize": 9 * scale,
            "xtick.labelsize": 9 * scale,
            "ytick.labelsize": 9 * scale,
            "figure.dpi": 150,
            "savefig.facecolor": "white",
        }
    )


def _curve_figure(df, *, metric: str, poster: bool):
    import matplotlib.pyplot as plt

    _style(poster=poster)
    figure, axis = plt.subplots(figsize=(10.5 if poster else 7.1, 6.1 if poster else 4.4))
    colors = {"eapig": "#E76F00", "eap": "#0072B2", "random_eapig": "#D55E00", "random_eap": "#009E73"}
    labels = {"eapig": "Intervention-path EAP-IG", "eap": "EAP", "random_eapig": "Random (EAP-IG matched)", "random_eap": "Random (EAP matched)"}
    for method, subset in df.groupby("method", sort=False):
        grouped = subset.groupby(["requested_k", "retained_edges"], as_index=False)[metric].agg(["mean", "sem"]).reset_index()
        grouped = grouped.sort_values("retained_edges")
        # A requested circuit can prune to zero edges. It is a valid numerical
        # baseline but cannot appear on a logarithmic edge-count axis.
        grouped = grouped[grouped["retained_edges"] > 0]
        if grouped.empty:
            continue
        color = colors.get(method, "#444444")
        linestyle = "--" if method.startswith("random_") else "-"
        axis.plot(
            grouped["retained_edges"],
            grouped["mean"],
            marker="o",
            linewidth=2.5 if poster else 2.0,
            linestyle=linestyle,
            color=color,
            label=labels.get(method, method),
        )
        if method.startswith("random_"):
            sem = grouped["sem"].fillna(0.0)
            axis.fill_between(grouped["retained_edges"], grouped["mean"] - sem, grouped["mean"] + sem, color=color, alpha=0.14)
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    if metric == "faithfulness":
        axis.axhline(0.85, color="#777777", linestyle=":", linewidth=1.0, label="0.85 target")
    axis.set_xscale("log")
    axis.set_xlabel("Retained circuit edges after greedy pruning")
    axis.set_ylabel("Faithfulness F(C)" if metric == "faithfulness" else "Necessity N(C)")
    axis.set_ylim(-0.06, 1.06)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()
    return figure


def _pareto_figure(faithfulness, *, poster: bool):
    import matplotlib.pyplot as plt

    _style(poster=poster)
    figure, axis = plt.subplots(figsize=(10.5 if poster else 7.1, 6.1 if poster else 4.4))
    for method, color, label in (("eapig", "#E76F00", "Intervention-path EAP-IG"), ("eap", "#0072B2", "EAP")):
        subset = faithfulness[faithfulness.method == method].sort_values("retained_edges")
        subset = subset[subset.retained_edges > 0]
        axis.plot(subset.retained_edges, subset.faithfulness, marker="o", linewidth=2.6 if poster else 2.0, color=color, label=label)
        for row in subset.itertuples():
            axis.annotate(f"{int(row.retained_edges)}", (row.retained_edges, row.faithfulness), xytext=(4, 5), textcoords="offset points", color=color)
    axis.axhline(0.85, color="#555555", linestyle=":", linewidth=1.1, label="0.85 faithfulness")
    axis.set_xscale("log")
    axis.set_ylim(-0.05, 1.02)
    axis.set_xlabel("Retained circuit edges")
    axis.set_ylabel("Faithfulness F(C)")
    axis.set_title("EAP-IG reaches high faithfulness with fewer retained edges")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure


def _behavior_figure(results_dir: Path, *, poster: bool):
    import matplotlib.pyplot as plt
    import pandas as pd

    path = results_dir / "behavioural_sanity_top1000.csv"
    if not path.exists():
        path = results_dir / "behavioural_sanity.csv"
    if not path.exists():
        return None
    _style(poster=poster)
    frame = pd.read_csv(path)
    labels = frame["condition"].str.replace("_", " ").str.title()
    figure, axes = plt.subplots(1, 2, figsize=(13.0 if poster else 8.4, 5.6 if poster else 3.9))
    colors = ["#0072B2", "#999999", "#E76F00", "#CC79A7"]
    axes[0].bar(labels, frame["refusal_substring_rate"], color=colors[: len(frame)])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Refusal substring rate")
    axes[1].bar(labels, frame["refusal_token_logit_margin"], color=colors[: len(frame)])
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_ylabel("Refusal-token logit margin")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Aggregate behavioral sanity check (no generated text stored)")
    figure.tight_layout()
    return figure


def _graphviz_views(results_dir: Path, output_dir: Path, *, circuit_k: int, graph_top_edges: int) -> None:
    import pandas as pd

    from eap.graph import Graph
    from refusal_circuit.graph import load_circuit_edge_names, set_selected_edges
    from refusal_circuit.paths import read_json

    score_path = results_dir / "eapig_edge_scores.parquet"
    circuit_path = results_dir / "circuits" / f"eapig_top_{circuit_k}.json"
    metadata = read_json(results_dir / "refusal_direction_metadata.json")
    scores = pd.read_parquet(score_path)
    n_heads = int(scores["source_head"].dropna().max()) + 1
    target_layer = int(metadata["layer"])
    score_by_name = dict(zip(scores["name"], scores["score"]))
    graph = None
    for parallel in (False, True):
        candidate = Graph.from_model({"n_layers": target_layer, "n_heads": n_heads, "parallel_attn_mlp": parallel})
        if set(candidate.edges) == set(score_by_name):
            graph = candidate
            break
    if graph is None:
        raise RuntimeError("Could not reconstruct the scored EAP graph from its saved edge names.")
    for edge in graph.edges.values():
        edge.score = float(score_by_name[edge.name])
    selected_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, selected_names)
    full = graph.to_graphviz(layout="dot", seed=7)
    full.draw(output_dir / f"eapig_top{circuit_k}_full.svg", prog="dot", format="svg")

    ranked = sorted((graph.edges[name] for name in selected_names), key=lambda edge: abs(float(edge.score)), reverse=True)
    display_names = {edge.name for edge in ranked[:graph_top_edges]}
    set_selected_edges(graph, display_names)
    display = graph.to_graphviz(layout="dot", seed=7)
    display.draw(output_dir / f"eapig_top{circuit_k}_top{graph_top_edges}.png", prog="dot", format="png")
    display.draw(output_dir / f"eapig_top{circuit_k}_top{graph_top_edges}.svg", prog="dot", format="svg")
    (output_dir / f"eapig_top{circuit_k}_graph_legend.md").write_text(
        "# Graph legend\n\n"
        "This is a view of the EAP-IG greedy circuit, not a complete architectural diagram. "
        "Boxes are attention heads (`aL.hH`), MLPs (`mL`), the input residual stream, or `logits` "
        "(the terminal refusal-projection metric). Black edges are MLP/residual edges; magenta, green, "
        "and blue indicate Q, K, and V inputs respectively. Edge width is proportional to absolute score.\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    faithfulness = pd.read_csv(args.results_dir / "faithfulness.csv")
    necessity = pd.read_csv(args.results_dir / "necessity.csv")
    for metric, frame in (("faithfulness", faithfulness), ("necessity", necessity)):
        for poster in (False, True):
            figure = _curve_figure(frame, metric=metric, poster=poster)
            _save(figure, args.output_dir, f"{metric}_{'poster' if poster else 'paper'}")
            plt.close(figure)
    for poster in (False, True):
        figure = _pareto_figure(faithfulness, poster=poster)
        _save(figure, args.output_dir, f"faithfulness_pareto_{'poster' if poster else 'paper'}")
        plt.close(figure)
        behavior = _behavior_figure(args.results_dir, poster=poster)
        if behavior is not None:
            _save(behavior, args.output_dir, f"behavioral_sanity_{'poster' if poster else 'paper'}")
            plt.close(behavior)
    if not args.skip_graph:
        _graphviz_views(
            args.graph_results_dir or args.results_dir,
            args.output_dir,
            circuit_k=args.circuit_k,
            graph_top_edges=args.graph_top_edges,
        )
    print(f"Wrote paper/poster figures to {args.output_dir}")


if __name__ == "__main__":
    main()

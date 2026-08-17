#!/usr/bin/env python3
"""Compare persona edge-score tables for IG-step or resample stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from _bootstrap import ROOT


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _correlation(left, right, *, label: str) -> float:
    import numpy as np

    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        raise ValueError(f"Cannot compute {label}: one score vector is constant.")
    value = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {label}.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument(
        "--scope",
        choices=("upstream", "downstream", "end_to_end"),
        required=True,
        help="Graph topology used for complete-path circuit overlap.",
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=(50, 100, 500, 1000, 3000)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from eap.graph import Graph

    from refusal_circuit.graph import (
        make_steering_to_logits_graph,
        paper_greedy_edges,
    )
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    architecture = config["model_architecture"]
    n_layers = int(architecture["n_layers"])
    n_heads = int(architecture["n_heads"])
    parallel_attn_mlp = bool(architecture["parallel_attn_mlp"])
    interface_layer = int(config["vector"]["hf_hidden_state_index"])
    if args.scope == "upstream":
        graph = Graph.from_model(
            {
                "n_layers": interface_layer,
                "n_heads": n_heads,
                "parallel_attn_mlp": parallel_attn_mlp,
            }
        )
        graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
        graph.nodes["logits"].layer = interface_layer
        graph.nodes["logits"].in_hook = config["vector"][
            "transformer_lens_target_hook"
        ]
    elif args.scope == "downstream":
        graph = make_steering_to_logits_graph(
            SimpleNamespace(
                cfg=SimpleNamespace(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    parallel_attn_mlp=parallel_attn_mlp,
                )
            ),
            interface_layer,
        )
    else:
        graph = Graph.from_model(
            {
                "n_layers": n_layers,
                "n_heads": n_heads,
                "parallel_attn_mlp": parallel_attn_mlp,
            }
        )
        graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"

    left_path = _resolve(args.left)
    right_path = _resolve(args.right)
    left = pd.read_parquet(left_path)[["name", "score"]].rename(
        columns={"score": "left"}
    )
    right = pd.read_parquet(right_path)[["name", "score"]].rename(
        columns={"score": "right"}
    )
    merged = left.merge(right, on="name", how="outer", validate="one_to_one")
    if merged[["left", "right"]].isna().any().any():
        raise ValueError("Score tables do not contain the same edge names.")
    if set(merged["name"]) != set(graph.edges):
        raise ValueError("Score tables do not match the configured graph topology.")
    left_values = merged["left"].to_numpy(dtype=float)
    right_values = merged["right"].to_numpy(dtype=float)
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("Score tables contain non-finite values.")

    pearson = _correlation(
        left_values, right_values, label="signed-score Pearson correlation"
    )
    left_rank = merged["left"].abs().rank(method="average").to_numpy()
    right_rank = merged["right"].abs().rank(method="average").to_numpy()
    spearman_abs = _correlation(
        left_rank,
        right_rank,
        label="absolute-score Spearman correlation",
    )
    nonzero = (left_values != 0) | (right_values != 0)
    if not nonzero.any():
        raise ValueError("Both score tables are identically zero.")
    sign_agreement = float(np.mean(
        np.sign(left_values[nonzero]) == np.sign(right_values[nonzero])
    ))

    def exact_top_names(column: str, k: int, *, absolute: bool) -> set[str]:
        frame = merged[["name", column]].copy()
        if absolute:
            frame[column] = frame[column].abs()
        frame = frame.sort_values(
            [column, "name"], ascending=[False, True], kind="mergesort"
        )
        return set(frame.head(k)["name"])

    def source_to_sink_names(column: str, k: int) -> set[str]:
        score_by_name = dict(
            zip(merged["name"], merged[column], strict=True)
        )
        for name, edge in graph.edges.items():
            edge.score = float(score_by_name[name])
        return {
            edge.name
            for edge in paper_greedy_edges(
                graph,
                min(k, len(graph.edges)),
                prune_parentless=True,
            )
        }

    top_k = []
    for k in sorted(set(args.k_values)):
        effective_k = min(k, len(merged))
        left_names = exact_top_names("left", effective_k, absolute=False)
        right_names = exact_top_names("right", effective_k, absolute=False)
        left_abs_names = exact_top_names("left", effective_k, absolute=True)
        right_abs_names = exact_top_names("right", effective_k, absolute=True)
        left_connected = source_to_sink_names("left", effective_k)
        right_connected = source_to_sink_names("right", effective_k)

        def jaccard(left_set: set[str], right_set: set[str]) -> float:
            union = left_set | right_set
            return len(left_set & right_set) / len(union) if union else 1.0

        top_k.append(
            {
                "k": k,
                "effective_k": effective_k,
                "signed_top_jaccard": jaccard(left_names, right_names),
                "absolute_top_jaccard": jaccard(
                    left_abs_names, right_abs_names
                ),
                "left_source_to_sink_edges": len(left_connected),
                "right_source_to_sink_edges": len(right_connected),
                "source_to_sink_jaccard": (
                    jaccard(left_connected, right_connected)
                    if left_connected or right_connected
                    else None
                ),
            }
        )

    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"{output} already exists.")
    write_json(
        output,
        {
            "schema_version": 1,
            "status": "complete",
            "scope": args.scope,
            "model_architecture": architecture,
            "left": str(left_path.relative_to(ROOT)),
            "left_sha256": sha256_file(left_path),
            "right": str(right_path.relative_to(ROOT)),
            "right_sha256": sha256_file(right_path),
            "edges": len(merged),
            "pearson_signed_scores": pearson,
            "spearman_absolute_scores": spearman_abs,
            "nonzero_sign_agreement": sign_agreement,
            "top_k": top_k,
        },
    )
    print(json.dumps(json.loads(output.read_text()), indent=2))


if __name__ == "__main__":
    main()

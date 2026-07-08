#!/usr/bin/env python3
"""Measure RQ1 circuit faithfulness, necessity, random baselines, and behaviour."""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--k-values", type=int, nargs="+", default=(50, 100, 200, 500, 1000, 2000))
    parser.add_argument("--methods", nargs="+", choices=("eapig", "eap"), default=("eapig", "eap"))
    parser.add_argument("--random-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--direction-path", type=Path, default=Path("results/rq1/refusal_direction.pt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("results/rq1/refusal_direction_metadata.json"))
    parser.add_argument("--circuits-dir", type=Path, default=Path("results/rq1/circuits"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rq1"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Prompt-split directory. Defaults to the RQ1 refusal splits; "
        "pass data/processed/persona/<trait> for a persona trait.",
    )
    parser.add_argument("--skip-behaviour", action="store_true")
    parser.add_argument("--behaviour-k", type=int, default=500)
    parser.add_argument("--behaviour-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def _metric_row(method, requested_k, retained_edges, seed, conditions, refusal_ids, answer_ids):
    from refusal_circuit.circuit import (
        first_token_logit_margin,
        normalized_faithfulness,
        normalized_necessity,
    )

    clean = conditions["clean"].projection
    corrupt = conditions["corrupt"].projection
    circuit_only = conditions["circuit_only"].projection
    circuit_ablated = conditions["circuit_ablated"].projection
    common = {
        "method": method,
        "requested_k": requested_k,
        "retained_edges": retained_edges,
        "random_seed": seed,
        "m_clean": clean.mean().item(),
        "m_corrupt": corrupt.mean().item(),
        "m_circuit_only": circuit_only.mean().item(),
        "m_circuit_ablated": circuit_ablated.mean().item(),
        "clean_logit_margin": first_token_logit_margin(conditions["clean"], refusal_ids, answer_ids),
        "corrupt_logit_margin": first_token_logit_margin(conditions["corrupt"], refusal_ids, answer_ids),
        "circuit_only_logit_margin": first_token_logit_margin(conditions["circuit_only"], refusal_ids, answer_ids),
        "circuit_ablated_logit_margin": first_token_logit_margin(conditions["circuit_ablated"], refusal_ids, answer_ids),
    }
    faithfulness = dict(common)
    faithfulness["faithfulness"] = normalized_faithfulness(clean, corrupt, circuit_only)
    necessity = dict(common)
    necessity["necessity"] = normalized_necessity(clean, corrupt, circuit_ablated)
    return faithfulness, necessity


def _plot(frame, *, column: str, output_path: Path, ylabel: str):
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for method, subset in frame.groupby("method"):
        grouped = subset.groupby("requested_k")[column]
        mean = grouped.mean()
        sem = grouped.sem().fillna(0.0)
        axis.plot(mean.index, mean.values, marker="o", label=method)
        axis.fill_between(mean.index, mean.values - sem.values, mean.values + sem.values, alpha=0.2)
    axis.axhline(0.0, color="grey", linewidth=0.8)
    axis.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_xlabel("retained / ablated edge count")
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    import random

    import pandas as pd
    import torch

    from refusal_circuit.circuit import behavioural_sanity_check, evaluate_conditions
    from refusal_circuit.data import load_processed_split
    from refusal_circuit.graph import load_circuit_edge_names, make_projection_graph, set_selected_edges
    from refusal_circuit.metrics import DEFAULT_ANSWER_STARTS, REFUSAL_PREFIXES
    from refusal_circuit.model import first_token_ids, format_instructions, load_model
    from refusal_circuit.paths import read_json

    metadata = read_json(args.metadata_path)
    if metadata["model"] != args.model:
        raise ValueError(f"Direction is for {metadata['model']}, but --model is {args.model}.")
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(args.direction_path, map_location=next(model.parameters()).device, weights_only=True)
    target_layer = int(metadata["layer"])
    target_hook = metadata["target_hook"]
    target_position = int(metadata["position"])
    prompts = format_instructions(
        model.tokenizer,
        [
            row["instruction"]
            for row in load_processed_split(args.split, "harmful", base_dir=args.processed_dir)[: args.num_prompts]
        ],
    )
    refusal_ids = first_token_ids(model.tokenizer, REFUSAL_PREFIXES)
    answer_ids = first_token_ids(model.tokenizer, DEFAULT_ANSWER_STARTS)

    faithfulness_rows = []
    necessity_rows = []
    behaviour_graph = None
    for method in args.methods:
        for requested_k in args.k_values:
            circuit_path = args.circuits_dir / f"{method}_top_{requested_k}.json"
            if not circuit_path.exists():
                print(f"Skipping missing circuit {circuit_path}")
                continue
            edge_names = load_circuit_edge_names(circuit_path)
            graph = make_projection_graph(model, target_layer)
            set_selected_edges(graph, edge_names)
            conditions = evaluate_conditions(
                model,
                graph,
                prompts,
                direction=direction,
                target_hook=target_hook,
                target_position=target_position,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            faith, necessity = _metric_row(
                method, requested_k, len(edge_names), None, conditions, refusal_ids, answer_ids
            )
            faithfulness_rows.append(faith)
            necessity_rows.append(necessity)
            if method == "eapig" and requested_k == args.behaviour_k:
                behaviour_graph = graph

            # The RQ1 protocol asks for raw-edge random controls matched by
            # retained-edge count. These are deliberately unconstrained; the
            # scored circuits themselves use the paper's connected greedy
            # construction.
            all_edge_names = list(graph.edges)
            for seed in args.random_seeds:
                rng = random.Random(seed + 100_003 * requested_k)
                random_edges = set(rng.sample(all_edge_names, k=min(len(edge_names), len(all_edge_names))))
                random_graph = make_projection_graph(model, target_layer)
                set_selected_edges(random_graph, random_edges)
                random_conditions = evaluate_conditions(
                    model,
                    random_graph,
                    prompts,
                    direction=direction,
                    target_hook=target_hook,
                    target_position=target_position,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
                # Keep controls separate for EAP and EAP-IG: greedy pruning can
                # leave different retained-edge counts at the same requested k,
                # so a single pooled ``random`` curve would not be cardinality
                # matched to either method.
                faith, necessity = _metric_row(
                    f"random_{method}",
                    requested_k,
                    len(random_edges),
                    seed,
                    random_conditions,
                    refusal_ids,
                    answer_ids,
                )
                faithfulness_rows.append(faith)
                necessity_rows.append(necessity)

    if not faithfulness_rows:
        raise FileNotFoundError(
            "No circuit JSONs were found. Run `scripts/rq1_run_eapig.py` before evaluation."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    faithfulness = pd.DataFrame(faithfulness_rows)
    necessity = pd.DataFrame(necessity_rows)
    faithfulness.to_csv(args.output_dir / "faithfulness.csv", index=False)
    necessity.to_csv(args.output_dir / "necessity.csv", index=False)
    _plot(
        faithfulness,
        column="faithfulness",
        output_path=args.output_dir / "plots" / "faithfulness_curve.png",
        ylabel="faithfulness F(C)",
    )
    _plot(
        necessity,
        column="necessity",
        output_path=args.output_dir / "plots" / "necessity_curve.png",
        ylabel="necessity N(C)",
    )

    if not args.skip_behaviour and behaviour_graph is None:
        print(f"Skipping behavioural check: no EAP-IG top-{args.behaviour_k} circuit was evaluated.")
    elif not args.skip_behaviour:
        behaviour = behavioural_sanity_check(
            model,
            behaviour_graph,
            prompts,
            direction=direction,
            target_hook=target_hook,
            target_position=target_position,
            refusal_ids=refusal_ids,
            answer_ids=answer_ids,
            max_prompts=args.behaviour_prompts,
            max_new_tokens=args.max_new_tokens,
        )
        behaviour_frame = pd.DataFrame(behaviour)
        behaviour_frame["circuit_requested_k"] = args.behaviour_k
        behaviour_frame.to_csv(args.output_dir / "behavioural_sanity.csv", index=False)
    print(f"Saved faithfulness and necessity tables under {args.output_dir}")


if __name__ == "__main__":
    main()

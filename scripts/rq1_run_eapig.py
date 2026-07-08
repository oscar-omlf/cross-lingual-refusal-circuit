#!/usr/bin/env python3
"""Run intervention-aware EAP-IG (and optional EAP) on RQ1 harmful prompts."""

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
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--methods", nargs="+", choices=("eapig", "eap"), default=("eapig", "eap"))
    parser.add_argument("--k-values", type=int, nargs="+", default=(50, 100, 200, 500, 1000, 2000))
    parser.add_argument("--direction-path", type=Path, default=Path("results/rq1/refusal_direction.pt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("results/rq1/refusal_direction_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rq1"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Prompt-split directory. Defaults to the RQ1 refusal splits; "
        "pass data/processed/persona/<trait> for a persona trait.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.eapig import AttributionConfig, attribute
    from refusal_circuit.graph import (
        make_projection_graph,
        write_circuit_jsons,
        write_edge_scores_parquet,
        write_top_edges_summary,
    )
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import read_json, write_json

    metadata = read_json(args.metadata_path)
    if metadata["model"] != args.model:
        raise ValueError(f"Direction is for {metadata['model']}, but --model is {args.model}.")
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(args.direction_path, map_location=next(model.parameters()).device, weights_only=True)
    prompts = format_instructions(
        model.tokenizer,
        [
            row["instruction"]
            for row in load_processed_split(args.split, "harmful", base_dir=args.processed_dir)[: args.num_prompts]
        ],
    )
    target_layer = int(metadata["layer"])
    target_hook = metadata["target_hook"]
    config = AttributionConfig(batch_size=args.batch_size, max_length=args.max_length, ig_steps=args.ig_steps)

    for method in args.methods:
        graph = make_projection_graph(model, target_layer)
        attribute(
            model,
            graph,
            prompts,
            direction=direction,
            target_hook=target_hook,
            target_position=int(metadata["position"]),
            config=config,
            integrated_gradients=method == "eapig",
        )
        scores_path = args.output_dir / f"{method}_edge_scores.parquet"
        write_edge_scores_parquet(graph, scores_path)
        write_top_edges_summary(graph, args.output_dir / f"top_edges_summary_{method}.md", method=method)
        circuit_paths = write_circuit_jsons(
            graph,
            method=method,
            target_layer=target_layer,
            target_position=int(metadata["position"]),
            output_dir=args.output_dir / "circuits",
            k_values=list(args.k_values),
        )
        write_json(
            args.output_dir / f"{method}_run_metadata.json",
            {
                "model": args.model,
                "method": method,
                "n_prompts": len(prompts),
                "split": args.split,
                "ig_steps": args.ig_steps if method == "eapig" else None,
                "target_hook": target_hook,
                "graph_granularity": "attention heads and MLPs; Q/K/V input edges retained",
                "corrupt_baseline": "same harmful prompt with global refusal-direction ablation",
                "input_source": "blocks.0.hook_resid_pre (post-ablation in corrupt run)",
                "metric": "h[layer, position]^T unit_refusal_direction",
                "interpolation_path": (
                    "global refusal-direction ablation strength alpha=0 (corrupt) "
                    "to alpha=1 (clean); intervention-aware EAP-IG extension"
                ),
                "circuit_selection": "greedy backwards from terminal by absolute score, then recursive pruning",
                "circuit_files": [str(path) for path in circuit_paths],
            },
        )
        print(f"Saved {method} scores to {scores_path}")

    # Preserve the output name promised by the RQ1 protocol when EAP-IG runs.
    eapig_summary = args.output_dir / "top_edges_summary_eapig.md"
    if eapig_summary.exists():
        (args.output_dir / "top_edges_summary.md").write_text(eapig_summary.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()

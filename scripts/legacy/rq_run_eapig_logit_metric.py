#!/usr/bin/env python3
"""Run intervention-aware EAP/EAP-IG for a next-token refusal-logit metric.

This is the logit-level fallback for cases where a residual refusal-direction
projection is useful descriptively but not a local causal bottleneck for
behavior.  Clean and corrupt inputs are still the same harmful prompts; the
corrupt baseline is produced by global refusal-direction ablation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--methods", nargs="+", choices=("eapig", "eap"), default=("eapig",))
    parser.add_argument("--k-values", type=int, nargs="+", default=(10, 25, 50, 100, 200, 500, 1000, 2000))
    parser.add_argument(
        "--direction-path",
        type=Path,
        default=Path("results/qwen2.5-7b-instruct/polyrefuse/en/shared_metrics/english_vector/refusal_direction.pt"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("results/qwen2.5-7b-instruct/polyrefuse/en/shared_metrics/english_vector/metadata.json"),
        help="Metadata containing model name and language-local refusal_token_ids.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/qwen2.5-7b-instruct/polyrefuse/en"),
        help="Processed PolyRefuse language directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen2.5-7b-instruct/polyrefuse/en/circuit_discovery_logit_refusal_metric"),
    )
    parser.add_argument(
        "--skip-circuits",
        action="store_true",
        help="Only write edge scores and summaries; useful for first debugging pilots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.eapig import AttributionConfig, attribute_logit_metric
    from refusal_circuit.graph import (
        make_logit_graph,
        write_edge_scores_parquet,
        write_logit_circuit_jsons,
        write_top_edges_summary,
    )
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import read_json, write_json

    metadata = read_json(args.metadata_path)
    if metadata["model"] != args.model:
        raise ValueError(f"Direction metadata is for {metadata['model']}, but --model is {args.model}.")
    refusal_token_ids = [int(token_id) for token_id in metadata.get("refusal_token_ids", [])]
    if not refusal_token_ids:
        raise ValueError(f"No refusal_token_ids found in {args.metadata_path}.")

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(args.direction_path, map_location=next(model.parameters()).device, weights_only=True)
    prompts = format_instructions(
        model.tokenizer,
        [
            row["instruction"]
            for row in load_processed_split(args.split, "harmful", data_dir=args.data_dir)[: args.num_prompts]
        ],
    )
    config = AttributionConfig(batch_size=args.batch_size, max_length=args.max_length, ig_steps=args.ig_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        graph = make_logit_graph(model)
        attribute_logit_metric(
            model,
            graph,
            prompts,
            direction=direction,
            refusal_token_ids=refusal_token_ids,
            config=config,
            integrated_gradients=method == "eapig",
        )
        scores_path = args.output_dir / f"{method}_edge_scores.parquet"
        write_edge_scores_parquet(graph, scores_path)
        write_top_edges_summary(graph, args.output_dir / f"top_edges_summary_{method}.md", method=method)
        if args.skip_circuits:
            circuit_paths = []
        else:
            circuit_paths = write_logit_circuit_jsons(
                graph,
                method=method,
                refusal_token_ids=refusal_token_ids,
                output_dir=args.output_dir / "circuits_greedy_no_parent_prune",
                k_values=list(args.k_values),
                prune_parentless=False,
            )
        write_json(
            args.output_dir / f"{method}_run_metadata.json",
            {
                "model": args.model,
                "method": method,
                "n_prompts": len(prompts),
                "split": args.split,
                "data_dir": str(args.data_dir),
                "ig_steps": args.ig_steps if method == "eapig" else None,
                "direction_path": str(args.direction_path),
                "metadata_path": str(args.metadata_path),
                "refusal_token_ids": refusal_token_ids,
                "graph_granularity": "attention heads and MLPs; Q/K/V input edges retained",
                "target": "ordinary EAP logit node at final prompt position",
                "metric": "next-token log(P(refusal-start tokens)/(1-P(refusal-start tokens)))",
                "corrupt_baseline": "same harmful prompt with global refusal-direction ablation",
                "input_source": "blocks.0.hook_resid_pre (post-ablation in corrupt run)",
                "interpolation_path": (
                    "global refusal-direction ablation strength alpha=0 (corrupt) "
                    "to alpha=1 (clean); intervention-aware EAP-IG extension"
                ),
                "circuit_selection": "greedy backwards from logits by absolute score; no parentless recursive pruning",
                "circuit_files": [str(path) for path in circuit_paths],
            },
        )
        print(f"Saved {method} logit-metric scores to {scores_path}")

    eapig_summary = args.output_dir / "top_edges_summary_eapig.md"
    if eapig_summary.exists():
        (args.output_dir / "top_edges_summary.md").write_text(eapig_summary.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()

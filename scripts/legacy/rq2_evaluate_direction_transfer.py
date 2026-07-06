#!/usr/bin/env python3
"""Interventional transfer matrix for common-site PolyRefuse directions.

Rows answer a directional question before any circuit claim: after ablating a
source language's vector everywhere, how much of the *target language's own*
anchor projection and first-token refusal score remains?  Conversely, does
adding that source vector at the common site increase the target metric on
harmless prompts?  No generated completion text is retained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--source-languages", nargs="+", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--addition-coefficient", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--direction-subdir", default="direction")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.direction import _run_projection_and_logits
    from refusal_circuit.interventions import addition_hooks, all_layer_ablation_hooks
    from refusal_circuit.metrics import mean_last_token_kl, refusal_log_odds
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import experiment_data_dir, experiment_results_dir, read_json, write_json

    target = args.target_language
    sources = list(dict.fromkeys(args.source_languages))
    default_results_root = experiment_results_dir(args.model, "polyrefuse", "en").parent
    results_root = args.results_root or default_results_root
    data_root = args.data_root or experiment_data_dir(args.model, "polyrefuse", "en").parent
    output_dir = args.output_dir or results_root / "comparisons" / "direction_transfer"
    target_dir = results_root / target / args.direction_subdir
    target_metadata = read_json(target_dir / "anchor_direction_metadata.json")
    if target_metadata.get("model") != args.model:
        raise ValueError("Target direction/model mismatch.")
    target_refusal_ids = [int(value) for value in target_metadata.get("refusal_token_ids", [])]
    if not target_refusal_ids:
        raise ValueError("Target anchor metadata has no refusal token IDs.")
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    device = next(model.parameters()).device
    target_vector = torch.load(
        target_dir / "anchor_refusal_direction.pt", map_location=device, weights_only=True
    )
    harmful = format_instructions(
        model.tokenizer,
        [
            row["instruction"]
            for row in load_processed_split(args.split, "harmful", data_dir=data_root / target)[: args.num_prompts]
        ],
    )
    harmless = format_instructions(
        model.tokenizer,
        [
            row["instruction"]
            for row in load_processed_split(args.split, "harmless", data_dir=data_root / target)[: args.num_prompts]
        ],
    )
    common = {
        "target_hook": target_metadata["target_hook"],
        "target_position": int(target_metadata["position"]),
        "direction": target_vector,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    clean_harm_projection, clean_harm_logits = _run_projection_and_logits(
        model, harmful, fwd_hooks=None, **common
    )
    clean_safe_projection, clean_safe_logits = _run_projection_and_logits(
        model, harmless, fwd_hooks=None, **common
    )
    rows = []
    for source in sources:
        source_dir = results_root / source / args.direction_subdir
        source_metadata = read_json(source_dir / "anchor_direction_metadata.json")
        if (
            int(source_metadata["layer"]), int(source_metadata["position"])
        ) != (int(target_metadata["layer"]), int(target_metadata["position"])):
            raise ValueError("Direction-transfer matrix requires common anchor locations.")
        source_vector = torch.load(
            source_dir / "anchor_refusal_direction.pt", map_location=device, weights_only=True
        )
        ablated_projection, ablated_logits = _run_projection_and_logits(
            model,
            harmful,
            fwd_hooks=all_layer_ablation_hooks(model, source_vector),
            **common,
        )
        raw_source_vector = source_vector * float(source_metadata["raw_direction_norm"])
        added_projection, added_logits = _run_projection_and_logits(
            model,
            harmless,
            fwd_hooks=addition_hooks(
                int(source_metadata["layer"]),
                raw_source_vector,
                coefficient=args.addition_coefficient,
                position=None,
            ),
            **common,
        )
        rows.append(
            {
                "model": args.model,
                "dataset": "PolyRefuse",
                "source_language": source,
                "target_language": target,
                "split": args.split,
                "n_harmful": len(harmful),
                "n_harmless": len(harmless),
                "anchor_layer": int(target_metadata["layer"]),
                "anchor_position": int(target_metadata["position"]),
                "clean_harmful_target_projection": clean_harm_projection.mean().item(),
                "source_ablated_harmful_target_projection": ablated_projection.mean().item(),
                "target_projection_ablation_drop": (
                    clean_harm_projection.mean() - ablated_projection.mean()
                ).item(),
                "clean_harmless_target_projection": clean_safe_projection.mean().item(),
                "source_added_harmless_target_projection": added_projection.mean().item(),
                "target_projection_addition_gain": (
                    added_projection.mean() - clean_safe_projection.mean()
                ).item(),
                "clean_harmful_refusal_log_odds": refusal_log_odds(
                    clean_harm_logits.unsqueeze(1), target_refusal_ids
                ).mean().item(),
                "source_ablated_harmful_refusal_log_odds": refusal_log_odds(
                    ablated_logits.unsqueeze(1), target_refusal_ids
                ).mean().item(),
                "clean_harmless_refusal_log_odds": refusal_log_odds(
                    clean_safe_logits.unsqueeze(1), target_refusal_ids
                ).mean().item(),
                "source_added_harmless_refusal_log_odds": refusal_log_odds(
                    added_logits.unsqueeze(1), target_refusal_ids
                ).mean().item(),
                "harmless_addition_kl": mean_last_token_kl(
                    clean_safe_logits.unsqueeze(1), added_logits.unsqueeze(1)
                ).item(),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"direction_transfer_to_{target}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    write_json(
        output_dir / f"direction_transfer_to_{target}_metadata.json",
        {
            "model": args.model,
            "dataset": "PolyRefuse",
            "target_language": target,
            "source_languages": sources,
            "direction_subdir": args.direction_subdir,
            "metric": "target language's local DIM vector at the common English anchor site",
            "ablation": "source anchor vector removed globally at every residual/component write",
            "addition": "source raw DIM vector added at the common anchor layer and all positions",
            "generated_text_saved": False,
        },
    )
    print(f"Saved direction-transfer rows for target={target} to {output_path}")


if __name__ == "__main__":
    main()

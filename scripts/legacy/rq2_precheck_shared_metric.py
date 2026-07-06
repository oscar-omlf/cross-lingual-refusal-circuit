#!/usr/bin/env python3
"""Precheck standardized RQ2 metrics before circuit discovery.

For each language this evaluates the exact clean/corrupt baseline used by
EAP-IG and faithfulness: harmful prompts under the normal model versus the
same prompts with the metric direction globally ablated.  It also reports the
same intervention on harmless prompts as a sanity check.  No completions or
prompt text are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", required=True)
    parser.add_argument("--metric-name", default="english_vector")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-harmful-drop", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.direction import _run_projection_and_logits
    from refusal_circuit.interventions import all_layer_ablation_hooks
    from refusal_circuit.metrics import mean_last_token_kl, refusal_log_odds
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import experiment_data_dir, experiment_results_dir, read_json, write_json

    language = args.language
    results_root = args.results_root or experiment_results_dir(args.model, "polyrefuse", "en").parent
    data_root = args.data_root or experiment_data_dir(args.model, "polyrefuse", "en").parent
    metric_dir = results_root / language / "shared_metrics" / args.metric_name
    metadata = read_json(metric_dir / "metadata.json")
    if metadata.get("model") != args.model:
        raise ValueError(f"Metric metadata is for {metadata.get('model')}, not {args.model}.")
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    device = next(model.parameters()).device
    direction = torch.load(metric_dir / "refusal_direction.pt", map_location=device, weights_only=True)
    refusal_ids = [int(value) for value in metadata.get("refusal_token_ids", [])]
    if not refusal_ids:
        raise ValueError("Shared metric metadata has no language-local refusal token IDs.")

    rows = []
    for label in ("harmful", "harmless"):
        prompts = format_instructions(
            model.tokenizer,
            [
                row["instruction"]
                for row in load_processed_split(args.split, label, data_dir=data_root / language)[: args.num_prompts]
            ],
        )
        common = {
            "target_hook": metadata["target_hook"],
            "target_position": int(metadata["position"]),
            "direction": direction,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        }
        clean_projection, clean_logits = _run_projection_and_logits(model, prompts, fwd_hooks=None, **common)
        corrupt_projection, corrupt_logits = _run_projection_and_logits(
            model,
            prompts,
            fwd_hooks=all_layer_ablation_hooks(model, direction),
            **common,
        )
        rows.append(
            {
                "model": args.model,
                "dataset": "PolyRefuse",
                "language": language,
                "metric_name": args.metric_name,
                "split": args.split,
                "prompt_class": label,
                "n_prompts": len(prompts),
                "layer": int(metadata["layer"]),
                "position": int(metadata["position"]),
                "m_clean": clean_projection.mean().item(),
                "m_corrupt": corrupt_projection.mean().item(),
                "projection_drop_clean_minus_corrupt": (clean_projection.mean() - corrupt_projection.mean()).item(),
                "first_token_refusal_log_odds_clean": refusal_log_odds(clean_logits.unsqueeze(1), refusal_ids).mean().item(),
                "first_token_refusal_log_odds_corrupt": refusal_log_odds(corrupt_logits.unsqueeze(1), refusal_ids).mean().item(),
                "last_token_kl_clean_to_corrupt": mean_last_token_kl(
                    clean_logits.unsqueeze(1), corrupt_logits.unsqueeze(1)
                ).item(),
            }
        )

    output_dir = args.output_dir or results_root / language / "metric_precheck" / args.metric_name
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "shared_metric_precheck.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    harmful_drop = next(
        row["projection_drop_clean_minus_corrupt"]
        for row in rows
        if row["prompt_class"] == "harmful"
    )
    if not torch.isfinite(torch.tensor(harmful_drop)) or harmful_drop <= args.min_harmful_drop:
        raise RuntimeError(
            f"Shared metric failed precheck for {language}/{args.metric_name}: "
            f"harmful clean-corrupt drop={harmful_drop:.6g}, "
            f"minimum required={args.min_harmful_drop:.6g}."
        )
    write_json(
        output_dir / "shared_metric_precheck_metadata.json",
        {
            "model": args.model,
            "dataset": "PolyRefuse",
            "language": language,
            "metric_name": args.metric_name,
            "generated_text_saved": False,
            "purpose": "Verify standardized clean/corrupt projection baseline before EAP-IG.",
        },
    )
    print(f"Saved shared metric precheck to {table_path}")


if __name__ == "__main__":
    main()

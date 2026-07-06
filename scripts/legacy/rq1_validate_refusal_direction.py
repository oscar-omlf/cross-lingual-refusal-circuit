#!/usr/bin/env python3
"""Aggregate-only behavioural validation of the selected refusal direction."""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-prompts-per-condition", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--direction-path", type=Path, default=Path("results/gemma-2-2b-it/en/rq1/refusal_direction.pt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("results/gemma-2-2b-it/en/rq1/refusal_direction_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gemma-2-2b-it/en/rq1"))
    return parser.parse_args()


def _generate_aggregate(model, prompts, *, fwd_hooks, refusal_token_ids, max_new_tokens, max_length):
    """Greedily decode without retaining or printing any completion text."""
    import torch

    from refusal_circuit.metrics import is_refusal, refusal_log_odds
    from refusal_circuit.model import tokenize_prompts

    refusal_count = 0
    refusal_metrics = []
    for prompt in prompts:
        tokens, attention_mask = tokenize_prompts(model, [prompt], max_length=max_length)
        generated_ids = []
        eos_id = model.tokenizer.eos_token_id
        with torch.inference_mode(), model.hooks(fwd_hooks=fwd_hooks):
            for step in range(max_new_tokens):
                logits = model(tokens, attention_mask=attention_mask)
                if step == 0:
                    refusal_metrics.append(
                        refusal_log_odds(logits[:, -1:].float(), refusal_token_ids).item()
                    )
                next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_ids)
                tokens = torch.cat([tokens, next_ids], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_ids, dtype=attention_mask.dtype)], dim=1
                )
                if eos_id is not None and int(next_ids[0, 0]) == eos_id:
                    break
        completion = model.tokenizer.decode(torch.cat(generated_ids, dim=1)[0], skip_special_tokens=True)
        refusal_count += int(is_refusal(completion))
        # ``completion`` deliberately goes out of scope immediately and is
        # never logged or persisted; only the aggregate count is retained.
    return {
        "n_prompts": len(prompts),
        "refusal_substring_rate": refusal_count / max(len(prompts), 1),
        "first_token_refusal_metric": sum(refusal_metrics) / max(len(refusal_metrics), 1),
        "generated_text_saved": False,
    }


def main():
    args = parse_args()
    import pandas as pd
    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.interventions import addition_hooks, all_layer_ablation_hooks
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import read_json

    metadata = read_json(args.metadata_path)
    if metadata["model"] != args.model:
        raise ValueError(f"Direction is for {metadata['model']}, but --model is {args.model}.")
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(args.direction_path, map_location=next(model.parameters()).device, weights_only=True)
    raw_direction = direction * float(metadata["raw_direction_norm"])
    refusal_token_ids = [int(token_id) for token_id in metadata["refusal_token_ids"]]
    harmful = format_instructions(
        model.tokenizer,
        [row["instruction"] for row in load_processed_split(args.split, "harmful")[: args.max_prompts_per_condition]],
    )
    harmless = format_instructions(
        model.tokenizer,
        [row["instruction"] for row in load_processed_split(args.split, "harmless")[: args.max_prompts_per_condition]],
    )
    conditions = (
        ("harmful_normal", harmful, []),
        ("harmful_direction_ablated", harmful, all_layer_ablation_hooks(model, direction)),
        ("harmless_normal", harmless, []),
        (
            "harmless_direction_added",
            harmless,
            addition_hooks(int(metadata["layer"]), raw_direction, coefficient=float(metadata["addition_coefficient"])),
        ),
    )
    rows = []
    for name, prompts, hooks in conditions:
        row = _generate_aggregate(
            model,
            prompts,
            fwd_hooks=hooks,
            refusal_token_ids=refusal_token_ids,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
        )
        row["condition"] = name
        row["split"] = args.split
        row["max_new_tokens"] = args.max_new_tokens
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "refusal_direction_behaviour.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved aggregate-only direction behaviour metrics to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose language-local refusal starter tokens without generating text.

This script only inspects the next-token distribution at the assistant start
for raw PolyRefuse prompts.  It saves aggregate token counts and score
quantiles, never prompts or generated continuations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--label", choices=("harmful", "harmless"), default="harmful")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--token-source",
        choices=("paper", "project"),
        default="project",
        help="Use only Table-4 Qwen token IDs, or the project's exploratory expanded set.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--external-root", type=Path, default=Path("external/multilingual_refusal"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen2.5-7b-instruct/polyrefuse/diagnostics"))
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    from refusal_circuit.metrics import refusal_log_odds
    from refusal_circuit.model import format_instructions, load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.polyrefuse import (
        OFFICIAL_QWEN_REFUSAL_TOKEN_IDS,
        load_split,
        refusal_token_ids,
    )

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    output = {"model": args.model, "label": args.label, "split": args.split, "languages": {}}
    for language in args.languages:
        rows, source_file = load_split(
            args.external_root, label=args.label, split=args.split, language=language
        )
        prompts = format_instructions(model.tokenizer, [row["instruction"] for row in rows])
        refusal_ids = (
            list(OFFICIAL_QWEN_REFUSAL_TOKEN_IDS.get(language, ()))
            if args.token_source == "paper"
            else refusal_token_ids(model.tokenizer, language)
        )
        if not refusal_ids:
            output["languages"][language] = {
                "source_file": str(source_file),
                "n_examples": len(rows),
                "token_set_available": False,
                "reason": "No Qwen2.5-7B refusal-token set was reported for this language in Table 4.",
            }
            continue
        top_counts: Counter[int] = Counter()
        score_values: list[float] = []
        for start in range(0, len(prompts), args.batch_size):
            tokens, attention_mask = tokenize_prompts(model, prompts[start : start + args.batch_size])
            with torch.inference_mode():
                logits = model(tokens, attention_mask=attention_mask)
            score_values.extend(refusal_log_odds(logits, refusal_ids).cpu().tolist())
            top_ids = logits[:, -1].topk(args.top_k, dim=-1).indices.cpu().tolist()
            top_counts.update(int(token_id) for values in top_ids for token_id in values)
        top_tokens = [
            {
                "token_id": token_id,
                "token": model.tokenizer.convert_ids_to_tokens(token_id),
                "top_k_occurrences": count,
                "is_in_current_refusal_set": token_id in refusal_ids,
            }
            for token_id, count in top_counts.most_common(30)
        ]
        ordered = sorted(score_values)
        output["languages"][language] = {
            "source_file": str(source_file),
            "n_examples": len(rows),
            "refusal_token_ids": refusal_ids,
            "token_set_available": True,
            "positive_score_count": sum(value > 0.0 for value in score_values),
            "negative_score_count": sum(value < 0.0 for value in score_values),
            "score_quantiles": {
                "min": ordered[0],
                "p25": ordered[len(ordered) // 4],
                "median": ordered[len(ordered) // 2],
                "p75": ordered[(3 * len(ordered)) // 4],
                "max": ordered[-1],
            },
            "top_tokens": top_tokens,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"starter_diagnostic_{args.token_source}_{args.label}_{args.split}.json"
    write_json(output_path, output)
    print(f"Saved aggregate-only refusal-starter diagnostic to {output_path}")


if __name__ == "__main__":
    main()

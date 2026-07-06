#!/usr/bin/env python3
"""Build the filtered, balanced JSONL splits for the refusal-direction RQ1."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401  (sets import paths)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/gemma-2-2b-it/en/rq1"),
        help="Processed-data namespace; defaults to the preserved Gemma RQ1 split.",
    )
    parser.add_argument(
        "--skip-behaviour-filter",
        action="store_true",
        help="Debug-only: write deterministic source samples without model filtering.",
    )
    return parser.parse_args()


def _generate_refusal_flags(model, prompts, *, batch_size: int, max_new_tokens: int, max_length: int):
    import torch

    from refusal_circuit.metrics import is_refusal
    from refusal_circuit.model import tokenize_prompts

    flags = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        tokens, attention_mask = tokenize_prompts(model, batch, max_length=max_length)
        generated_ids = []
        with torch.inference_mode():
            # A short explicit greedy loop keeps the same attention-mask-aware
            # forward path used by the rest of RQ1. It avoids relying on a
            # TransformerLens generation helper whose padding semantics differ
            # across releases.
            for _ in range(max_new_tokens):
                logits = model(tokens, attention_mask=attention_mask)
                next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_ids)
                tokens = torch.cat([tokens, next_ids], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_ids, dtype=attention_mask.dtype)], dim=1
                )
        continuation = torch.cat(generated_ids, dim=1)
        flags.extend(is_refusal(model.tokenizer.decode(row, skip_special_tokens=True)) for row in continuation)
    return flags


def main():
    args = parse_args()
    from refusal_circuit.data import SPLIT_SIZES, load_candidate_pool, write_processed_split
    from refusal_circuit.model import format_instructions, load_model

    rng = random.Random(args.seed)
    model = None
    if not args.skip_behaviour_filter:
        model = load_model(args.model, device=args.device, dtype=args.dtype)

    for split, targets in SPLIT_SIZES.items():
        selected = []
        for label, target_count in targets.items():
            candidates = load_candidate_pool(label, split)
            rng.shuffle(candidates)
            if args.skip_behaviour_filter:
                kept = candidates[:target_count]
                for row in kept:
                    row["base_refusal"] = None
                    row["filter_mode"] = "skipped"
            else:
                prompts = format_instructions(model.tokenizer, [row["instruction"] for row in candidates])
                add_special_tokens = not bool(getattr(model.tokenizer, "chat_template", None))
                token_lists = model.tokenizer(
                    prompts, add_special_tokens=add_special_tokens, padding=False
                ).input_ids
                eligible = [
                    (row, prompt, len(token_ids))
                    for row, prompt, token_ids in zip(candidates, prompts, token_lists)
                    if len(token_ids) <= args.max_length
                ]
                if len(eligible) < target_count:
                    raise RuntimeError(
                        f"Only {len(eligible)} {label} {split} candidates fit max_length={args.max_length}. "
                        "Increase --max-length or supply a larger local source pool."
                    )
                expected = label == "harmful"
                kept = []
                # Filter incrementally: harmless Alpaca pools have thousands
                # of rows, and normally the first few batches already contain
                # enough accepted examples. There is no reason to generate
                # completions for the entire pool.
                for start in range(0, len(eligible), args.batch_size):
                    batch = eligible[start : start + args.batch_size]
                    flags = _generate_refusal_flags(
                        model,
                        [prompt for _, prompt, _ in batch],
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        max_length=args.max_length,
                    )
                    for (row, _prompt, token_length), flag in zip(batch, flags):
                        if flag == expected:
                            row["base_refusal"] = bool(flag)
                            row["filter_mode"] = "substring_generation"
                            row["token_length"] = token_length
                            kept.append(row)
                        if len(kept) == target_count:
                            break
                    if len(kept) == target_count:
                        break
            if len(kept) < target_count:
                raise RuntimeError(
                    f"Only found {len(kept)}/{target_count} {label} examples in {split}. "
                    "Do not continue with an unfiltered split; provide a larger local candidate pool."
                )
            selected.extend(kept)
        rng.shuffle(selected)
            path = write_processed_split(split, selected, data_dir=args.data_dir)
        print(f"Wrote {len(selected)} balanced, filtered rows to {path}")


if __name__ == "__main__":
    main()

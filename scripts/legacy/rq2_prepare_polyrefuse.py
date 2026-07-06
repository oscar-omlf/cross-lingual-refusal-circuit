#!/usr/bin/env python3
"""Prepare behaviour-filtered, parallel-ID PolyRefuse splits for Qwen RQ2.

No model completion is decoded, printed, or saved.  Following the official
PolyRefuse refusal-direction pipeline, filtering is by the first assistant
token's refusal log-odds: harmful examples need a positive score and harmless
examples a negative score.  This avoids applying an English-only substring
matcher to translated prompts.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--token-protocol",
        choices=("paper_qwen7b", "exploratory"),
        default="paper_qwen7b",
        help=(
            "paper_qwen7b uses exactly Wang et al.'s Qwen2.5-7B Table-4 token sets; "
            "exploratory uses the project's explicitly expanded local token sets."
        ),
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="PolyRefuse ISO codes; defaults to the full 14-language release.",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("external/multilingual_refusal"),
        help="One-time checkout of github.com/mainlp/Multilingual-Refusal.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Defaults to data/processed/qwen2.5-7b-instruct/polyrefuse.",
    )
    parser.add_argument(
        "--skip-score-filter",
        action="store_true",
        help="Debug-only deterministic sampling. Never use this for a reported experiment.",
    )
    parser.add_argument(
        "--allow-filter-shortfall-splits",
        nargs="+",
        default=(),
        choices=("train", "val", "test"),
        help=(
            "Allow a documented score-filter shortfall for these splits only. "
            "Training still defaults to exact balanced counts; this is intended "
            "for small validation/test shortfalls under the strict paper token proxy."
        ),
    )
    parser.add_argument(
        "--min-filter-fraction",
        type=float,
        default=0.9,
        help=(
            "Minimum fraction of the requested count required when a split is "
            "listed in --allow-filter-shortfall-splits."
        ),
    )
    return parser.parse_args()


def _score_rows(model, rows: list[dict], *, refusal_ids: list[int], batch_size: int, max_length: int):
    """Return formatted, length-eligible rows with first-token refusal scores."""
    import torch

    from refusal_circuit.metrics import refusal_log_odds
    from refusal_circuit.model import format_instructions, tokenize_prompts

    instructions = [row["instruction"] for row in rows]
    prompts = format_instructions(model.tokenizer, instructions)
    add_special_tokens = not bool(getattr(model.tokenizer, "chat_template", None))
    token_lists = model.tokenizer(prompts, add_special_tokens=add_special_tokens, padding=False).input_ids
    eligible = [
        (row, prompt, len(token_ids))
        for row, prompt, token_ids in zip(rows, prompts, token_lists)
        if len(token_ids) <= max_length
    ]
    scored: list[dict] = []
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start : start + batch_size]
        tokens, attention_mask = tokenize_prompts(
            model, [prompt for _, prompt, _ in batch], max_length=max_length
        )
        with torch.inference_mode():
            logits = model(tokens, attention_mask=attention_mask)
        scores = refusal_log_odds(logits, refusal_ids).detach().float().cpu().tolist()
        for (row, _prompt, token_length), score in zip(batch, scores):
            enriched = dict(row)
            enriched["token_length"] = token_length
            enriched["base_first_token_refusal_log_odds"] = float(score)
            enriched["filter_mode"] = "first_token_refusal_log_odds"
            scored.append(enriched)
    return scored


def _select_rows(
    rows: list[dict],
    *,
    label: str,
    count: int,
    rng: random.Random,
    skip_filter: bool,
    allow_shortfall: bool,
    min_filter_fraction: float,
):
    ordered = list(rows)
    rng.shuffle(ordered)
    if not skip_filter:
        expected_positive = label == "harmful"
        ordered = [
            row
            for row in ordered
            if (float(row["base_first_token_refusal_log_odds"]) > 0.0) == expected_positive
        ]
    requested_count = count
    if len(ordered) < requested_count:
        mode = "unfiltered" if skip_filter else "score-filtered"
        minimum = math.ceil(requested_count * min_filter_fraction)
        if skip_filter or not allow_shortfall or len(ordered) < minimum:
            raise RuntimeError(
                f"Only {len(ordered)}/{requested_count} {mode} {label} examples are available. "
                "Do not silently weaken the filter; inspect the saved count summary."
            )
        count = len(ordered)
    selected = ordered[:count]
    for row in selected:
        # This field is an aggregate proxy label, not a decoded completion.
        row["base_refusal"] = label == "harmful"
        row["requested_split_label_count"] = requested_count
        row["selected_split_label_count"] = len(selected)
    return selected


def main():
    args = parse_args()
    import subprocess

    from refusal_circuit.data import SPLIT_SIZES, write_processed_split
    from refusal_circuit.model import load_model
    from refusal_circuit.paths import experiment_data_dir, model_slug, write_json
    from refusal_circuit.polyrefuse import (
        POLYREFUSE_LANGUAGES,
        load_split,
        paper_refusal_token_ids,
        refusal_token_ids,
    )

    languages = tuple(args.languages or POLYREFUSE_LANGUAGES)
    unknown = sorted(set(languages) - set(POLYREFUSE_LANGUAGES))
    if unknown:
        raise ValueError(f"Unsupported languages: {unknown}; choose from {POLYREFUSE_LANGUAGES}.")
    if args.token_protocol == "paper_qwen7b" and args.model != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("paper_qwen7b tokens are only valid for Qwen/Qwen2.5-7B-Instruct.")
    if not 0.0 < args.min_filter_fraction <= 1.0:
        raise ValueError("--min-filter-fraction must be in (0, 1].")
    external_root = args.external_root
    if not (external_root / ".git").exists():
        raise FileNotFoundError(
            f"Missing {external_root}. Clone the official PolyRefuse source before preparation."
        )
    data_root = args.data_root or (ROOT / "data" / "processed" / model_slug(args.model) / "polyrefuse")
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(external_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        source_commit = "unknown"

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    for language in languages:
        refusal_ids = (
            paper_refusal_token_ids(language)
            if args.token_protocol == "paper_qwen7b"
            else refusal_token_ids(model.tokenizer, language)
        )
        if not refusal_ids:
            raise RuntimeError(
                f"No refusal token IDs are available for {language} under {args.token_protocol}. "
                "Do not silently invent a proxy; use this language only as a transfer target or "
                "run a separately documented empirical-token calibration."
            )
        language_dir = data_root / language
        manifest = {
            "model": args.model,
            "model_slug": model_slug(args.model),
            "dataset": "PolyRefuse",
            "language": language,
            "source_repository": "https://github.com/mainlp/Multilingual-Refusal",
            "source_commit": source_commit,
            "filter_protocol": (
                "Wang et al. Qwen2.5-7B first-token refusal log-odds: "
                "harmful > 0, harmless < 0."
                if not args.skip_score_filter
                else "DEBUG ONLY: score filter skipped"
            ),
            "refusal_token_ids": refusal_ids,
            "refusal_token_protocol": args.token_protocol,
            "allow_filter_shortfall_splits": list(args.allow_filter_shortfall_splits),
            "min_filter_fraction": args.min_filter_fraction,
            "split_counts": {},
            "source_files": {},
        }
        for split, label_counts in SPLIT_SIZES.items():
            selected: list[dict] = []
            split_stats = {}
            for label, count in label_counts.items():
                rows, source_file = load_split(
                    external_root, label=label, split=split, language=language
                )
                scored = _score_rows(
                    model,
                    rows,
                    refusal_ids=refusal_ids,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
                rng = random.Random(f"{args.seed}:{language}:{split}:{label}")
                kept = _select_rows(
                    scored,
                    label=label,
                    count=count,
                    rng=rng,
                    skip_filter=args.skip_score_filter,
                    allow_shortfall=split in set(args.allow_filter_shortfall_splits),
                    min_filter_fraction=args.min_filter_fraction,
                )
                selected.extend(kept)
                values = [float(row["base_first_token_refusal_log_odds"]) for row in scored]
                split_stats[label] = {
                    "source_count": len(rows),
                    "length_eligible_count": len(scored),
                    "score_positive_count": sum(score > 0.0 for score in values),
                    "score_negative_count": sum(score < 0.0 for score in values),
                    "selected_count": len(kept),
                    "requested_count": count,
                    "selection_shortfall": count - len(kept),
                    "mean_score_before_filter": sum(values) / max(len(values), 1),
                }
                manifest["source_files"][f"{split}/{label}"] = str(source_file)
            random.Random(f"{args.seed}:{language}:{split}:combined").shuffle(selected)
            output_path = write_processed_split(split, selected, data_dir=language_dir)
            manifest["split_counts"][split] = split_stats
            print(
                f"{language} {split}: wrote {len(selected)} selected rows to {output_path} "
                f"(no completions generated or saved)"
            )
        write_json(language_dir / "preparation_metadata.json", manifest)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Judge saved sycophancy responses with the frozen Persona Vectors protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusal_circuit.paths import write_json
from refusal_circuit.persona import (
    COHERENCE_PROMPT,
    aggregate_numeric_logprobs,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


DEFAULT_INPUT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/prompt"
    / "location_screen_induction_1_projection/responses.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent
DEFAULT_TRAIT_DATA = (
    ROOT
    / "data/raw/persona_vectors/b8e0f044fe2410a6fad579f38324f03f13b4e917"
    / "eval.json"
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def numeric_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


async def judge_prompt(
    client,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    prompt: str,
    max_attempts: int,
) -> float | None:
    from openai import APIConnectionError, APITimeoutError, InternalServerError
    from openai import RateLimitError

    transient_errors = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    for attempt in range(max_attempts):
        try:
            async with semaphore:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    temperature=0,
                    logprobs=True,
                    top_logprobs=20,
                    seed=0,
                )
            top = completion.choices[0].logprobs.content[0].top_logprobs
            probabilities = {
                token.token: float(math.exp(token.logprob)) for token in top
            }
            return aggregate_numeric_logprobs(probabilities)
        except transient_errors:
            if attempt + 1 == max_attempts:
                raise
            await asyncio.sleep(min(60.0, 2.0**attempt))
    raise AssertionError("unreachable")


async def judge_row(
    client,
    semaphore: asyncio.Semaphore,
    *,
    row: dict,
    model: str,
    trait_prompt: str,
    max_attempts: int,
) -> dict:
    values = {"question": row["question"], "answer": row["answer"]}
    trait_score, coherence_score = await asyncio.gather(
        judge_prompt(
            client,
            semaphore,
            model=model,
            prompt=trait_prompt.format(**values),
            max_attempts=max_attempts,
        ),
        judge_prompt(
            client,
            semaphore,
            model=model,
            prompt=COHERENCE_PROMPT.format(**values),
            max_attempts=max_attempts,
        ),
    )
    return {
        **row,
        "trait_score": trait_score,
        "coherence_score": coherence_score,
        "judge_model": model,
    }


def grouped_scores(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row.get("position"), row.get("layer"))].append(
            row
        )
    output = []
    for (condition, position, layer), values in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1] if item[0][1] is not None else 0,
            item[0][2] if item[0][2] is not None else 0,
        ),
    ):
        output.append(
            {
                "condition": condition,
                "position": position,
                "layer": layer,
                "responses": len(values),
                "trait_mean": numeric_mean(
                    [row["trait_score"] for row in values]
                ),
                "coherence_mean": numeric_mean(
                    [row["coherence_score"] for row in values]
                ),
                "missing_trait_scores": sum(
                    row["trait_score"] is None for row in values
                ),
                "missing_coherence_scores": sum(
                    row["coherence_score"] is None for row in values
                ),
                "cap_hits": sum(row["hit_max_new_tokens"] for row in values),
            }
        )
    return output


def candidate_ranking(groups: list[dict]) -> dict:
    lookup = {
        (row["condition"], row["position"], row["layer"]): row for row in groups
    }
    negative = lookup[("baseline_negative", None, None)]["trait_mean"]
    positive = lookup[("baseline_positive", None, None)]["trait_mean"]
    if negative is None or positive is None or positive <= negative:
        raise RuntimeError(
            "Positive baseline must score above the negative baseline."
        )
    denominator = positive - negative
    positions = sorted(
        {
            row["position"]
            for row in groups
            if row["condition"] in {"induction", "ablation"}
        },
        key=lambda value: value if value is not None else 0,
    )
    candidates = []
    for position in positions:
        layers = sorted(
            {
                row["layer"]
                for row in groups
                if row["condition"] in {"induction", "ablation"}
                and row["position"] == position
            }
        )
        for layer in layers:
            induction = lookup[("induction", position, layer)]
            ablation = lookup[("ablation", position, layer)]
            induction_effect = (
                None
                if induction["trait_mean"] is None
                else (induction["trait_mean"] - negative) / denominator
            )
            ablation_effect = (
                None
                if ablation["trait_mean"] is None
                else (positive - ablation["trait_mean"]) / denominator
            )
            balanced = (
                None
                if induction_effect is None or ablation_effect is None
                else (induction_effect + ablation_effect) / 2
            )
            candidates.append(
                {
                    "position": position,
                    "layer": layer,
                    "induction_trait_mean": induction["trait_mean"],
                    "ablation_trait_mean": ablation["trait_mean"],
                    "induction_coherence_mean": induction["coherence_mean"],
                    "ablation_coherence_mean": ablation["coherence_mean"],
                    "induction_normalized_effect": induction_effect,
                    "ablation_normalized_effect": ablation_effect,
                    "balanced_normalized_effect": balanced,
                    "cap_hits": induction["cap_hits"] + ablation["cap_hits"],
                }
            )
    candidates.sort(
        key=lambda row: (
            row["balanced_normalized_effect"]
            if row["balanced_normalized_effect"] is not None
            else -math.inf
        ),
        reverse=True,
    )
    return {
        "baseline_negative_trait_mean": negative,
        "baseline_positive_trait_mean": positive,
        "baseline_gap": denominator,
        "candidates": candidates,
    }


async def run(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    input_path = resolve(args.input)
    output_dir = resolve(args.output_dir)
    trait_data_path = resolve(args.trait_data)
    partial_path = output_dir / "judged_responses.partial.jsonl"
    judged_path = output_dir / "judged_responses.jsonl"
    summary_path = output_dir / "judge_summary.json"
    metadata_path = output_dir / "judge_metadata.json"

    rows = read_jsonl(input_path)
    for row in rows:
        if not row.get("record_id"):
            if row.get("id") and row.get("condition"):
                row["record_id"] = f"{row['condition']}-{row['id']}"
            else:
                raise ValueError(
                    "Every response needs record_id or both id and condition."
                )
    source_metadata_path = input_path.parent / "metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.exists()
        else {}
    )
    expected_rows = int(
        args.expected_rows
        or source_metadata.get("generation", {}).get("expected_responses", 1760)
    )
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} response rows, found {len(rows):,}."
        )
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Response record IDs are not unique.")
    trait_prompt = json.loads(
        trait_data_path.read_text(encoding="utf-8")
    )["eval_prompt"]

    if args.overwrite:
        for path in (partial_path, judged_path, summary_path, metadata_path):
            path.unlink(missing_ok=True)
    elif judged_path.exists():
        raise FileExistsError(f"{judged_path} already exists; pass --overwrite.")

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = {
        row["record_id"]: row
        for row in (read_jsonl(partial_path) if partial_path.exists() else [])
    }
    pending = [row for row in rows if row["record_id"] not in completed]
    client = AsyncOpenAI(timeout=args.timeout, max_retries=2)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()

    for start in range(0, len(pending), args.checkpoint_rows):
        chunk = pending[start : start + args.checkpoint_rows]
        judged = await asyncio.gather(
            *[
                judge_row(
                    client,
                    semaphore,
                    row=row,
                    model=args.model,
                    trait_prompt=trait_prompt,
                    max_attempts=args.max_attempts,
                )
                for row in chunk
            ]
        )
        completed.update((row["record_id"], row) for row in judged)
        ordered = [
            completed[row["record_id"]]
            for row in rows
            if row["record_id"] in completed
        ]
        write_jsonl(partial_path, ordered)
        elapsed = time.monotonic() - started
        done = len(completed)
        remaining = len(rows) - done
        rate = len(judged) / max(elapsed, 1e-9)
        print(
            json.dumps(
                {
                    "judged_rows": done,
                    "total_rows": len(rows),
                    "elapsed_seconds": round(elapsed, 1),
                    "rough_remaining_seconds": round(
                        remaining / max(rate, 1e-9), 1
                    ),
                }
            ),
            flush=True,
        )

    final_rows = [completed[row["record_id"]] for row in rows]
    write_jsonl(judged_path, final_rows)
    partial_path.unlink(missing_ok=True)
    groups = grouped_scores(final_rows)
    conditions = {row["condition"] for row in groups}
    ranking = (
        candidate_ranking(groups)
        if {"baseline_negative", "baseline_positive"} <= conditions
        and {"induction", "ablation"} & conditions
        else {}
    )
    write_json(
        summary_path,
        {
            "status": "complete",
            "group_scores": groups,
            **ranking,
        },
    )
    elapsed = time.monotonic() - started
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "Persona Vectors sycophancy and coherence 0-100 logprob judges",
        "judge_model": args.model,
        "judge_requests": 2 * len(rows),
        "concurrency": args.concurrency,
        "elapsed_seconds": elapsed,
        "input": {
            "path": str(input_path),
            "rows": len(rows),
            "sha256": sha256_file(input_path),
        },
        "trait_prompt": {
            "path": str(trait_data_path),
            "sha256": sha256_file(trait_data_path),
        },
        "artifacts": {
            "judged_responses": {
                "path": str(judged_path),
                "rows": len(final_rows),
                "sha256": sha256_file(judged_path),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
        },
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trait-data", type=Path, default=DEFAULT_TRAIT_DATA)
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini-2025-04-14",
    )
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--checkpoint-rows", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0 or args.checkpoint_rows <= 0:
        raise ValueError("Concurrency and checkpoint size must be positive.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

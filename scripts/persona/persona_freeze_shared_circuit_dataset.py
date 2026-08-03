#!/usr/bin/env python3
"""Freeze shared upstream/downstream persona circuit samples.

Discovery is matched at the unique (question, instruction) prompt level.  The
upstream objective consumes the positive/helpful prompt pair, while the
downstream objective consumes one strictly validated positive/negative response
pair from that exact positive prompt.  Downstream internal evaluation uses
different response-pair IDs and behavior uses the separate official eval set.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from _bootstrap import ROOT


def _hash_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _balanced_prompt_selection(
    by_question: dict[int, list[tuple[tuple[int, int], float]]],
    *,
    total: int,
) -> list[tuple[int, int]]:
    """Maximize contrast with near-equal question and instruction coverage.

    The exact dynamic program reproduces the original 64-row sycophancy
    manifest. Traits whose strict eligible pool cannot realize that exact
    instruction allocation use a deterministic balance-first fallback.
    """
    best: tuple[float, tuple[tuple[int, int], ...]] | None = None
    # 64 samples across five instructions permits four counts of 13 and one
    # count of 12. Try every placement of the smaller count.
    for smaller_instruction in range(5):
        targets = tuple(
            12 if instruction == smaller_instruction else 13
            for instruction in range(5)
        )
        states: dict[
            tuple[int, tuple[int, ...]],
            tuple[float, tuple[tuple[int, int], ...]],
        ] = {(0, (0, 0, 0, 0, 0)): (0.0, ())}
        for question in sorted(by_question):
            rows = by_question[question]
            # Every question contributes as evenly as strict eligibility
            # allows: exactly two when only two qualify, otherwise three or
            # four. This yields final per-question counts in [2, 4].
            sizes = (2,) if len(rows) == 2 else tuple(
                size for size in (3, 4) if size <= len(rows)
            )
            options = []
            for size in sizes:
                for subset in itertools.combinations(rows, size):
                    keys = tuple(item[0] for item in subset)
                    counts = tuple(
                        sum(key[1] == instruction for key in keys)
                        for instruction in range(5)
                    )
                    score = sum(item[1] for item in subset)
                    options.append((keys, counts, score))
            next_states = {}
            for (used, counts), (score, keys) in states.items():
                for option_keys, option_counts, option_score in options:
                    new_used = used + len(option_keys)
                    if new_used > total:
                        continue
                    new_counts = tuple(
                        left + right
                        for left, right in zip(
                            counts, option_counts, strict=True
                        )
                    )
                    if any(
                        count > target
                        for count, target in zip(
                            new_counts, targets, strict=True
                        )
                    ):
                        continue
                    state = (new_used, new_counts)
                    candidate = (score + option_score, keys + option_keys)
                    current = next_states.get(state)
                    if current is None or candidate[0] > current[0]:
                        next_states[state] = candidate
            states = next_states
        candidate = states.get((total, targets))
        if candidate is not None and (
            best is None or candidate[0] > best[0]
        ):
            best = candidate
    if best is not None:
        return list(best[1])

    candidates = {
        key: score
        for rows in by_question.values()
        for key, score in rows
    }
    if len(candidates) < total:
        raise RuntimeError(
            f"Only {len(candidates)} unique strict prompts are available for {total} rows."
        )
    selected: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()
    question_counts: Counter = Counter()
    instruction_counts: Counter = Counter()

    # Guarantee question coverage before optimizing the remaining balance.
    for question in sorted(by_question):
        available = [item for item in by_question[question] if item[0] not in selected_set]
        if not available:
            raise RuntimeError(f"Question {question} has no strict prompt candidate.")
        key, _ = min(
            available,
            key=lambda item: (
                instruction_counts[item[0][1]],
                -item[1],
                item[0][1],
            ),
        )
        selected.append(key)
        selected_set.add(key)
        question_counts[key[0]] += 1
        instruction_counts[key[1]] += 1

    while len(selected) < total:
        key = min(
            (key for key in candidates if key not in selected_set),
            key=lambda candidate: (
                question_counts[candidate[0]],
                instruction_counts[candidate[1]],
                -candidates[candidate],
                candidate,
            ),
        )
        selected.append(key)
        selected_set.add(key)
        question_counts[key[0]] += 1
        instruction_counts[key[1]] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--discovery-prompts", type=int, default=64)
    parser.add_argument("--downstream-eval-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=None,
        help=(
            "Reuse exactly the discovery question/instruction identities from "
            "an earlier frozen manifest while allowing response-pair replacement."
        ),
    )
    parser.add_argument(
        "--max-transcript-length",
        type=int,
        default=None,
        help=(
            "Require both same-positive-prompt response transcripts to fit "
            "within this many tokens without truncation."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    intermediate = ROOT / config["paths"]["intermediate_root"]
    official = ROOT / config["paths"]["official_root"]
    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    extract_path = official / "extract.json"
    eval_path = official / "eval.json"
    positive = read_jsonl(positive_path)
    negative = {row["pair_id"]: row for row in read_jsonl(negative_path)}
    if len(positive) != 1000 or len(negative) != 1000:
        raise ValueError("Expected the canonical 1,000 rows per polarity.")

    tokenizer = None
    if args.max_transcript_length is not None:
        if args.max_transcript_length <= 0:
            raise ValueError("--max-transcript-length must be positive.")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["name"],
            revision=config["model"].get("revision"),
            local_files_only=True,
        )

    strict: dict[tuple[int, int], list[tuple[dict, dict]]] = defaultdict(list)
    for pos in positive:
        neg = negative[pos["pair_id"]]
        key = (int(pos["question_index"]), int(pos["instruction_index"]))
        positive_length = None
        negative_length = None
        if tokenizer is not None:
            positive_length = len(
                tokenizer.encode(
                    pos["prompt"] + pos["answer"], add_special_tokens=False
                )
            )
            # Downstream contrastive attribution scores both answers under the
            # identical positive-persona prompt.
            negative_length = len(
                tokenizer.encode(
                    pos["prompt"] + neg["answer"], add_special_tokens=False
                )
            )
        if (
            float(pos["trait_score"]) >= 80
            and float(neg["trait_score"]) <= 20
            and float(pos["coherence_score"]) >= 80
            and float(neg["coherence_score"]) >= 80
            and (
                args.max_transcript_length is None
                or (
                    positive_length <= args.max_transcript_length
                    and negative_length <= args.max_transcript_length
                )
            )
        ):
            if tokenizer is not None:
                pos = {
                    **pos,
                    "_positive_transcript_tokens": positive_length,
                    "_negative_transcript_tokens": negative_length,
                }
            strict[key].append((pos, neg))
    if len(strict) < args.discovery_prompts:
        raise ValueError(
            f"Only {len(strict)} unique prompts pass strict thresholds."
        )

    # Select unique prompts jointly balanced over questions and instructions,
    # or preserve an already-frozen set of prompt identities exactly.
    by_question: dict[int, list[tuple[tuple[int, int], float]]] = defaultdict(list)
    for key, pairs in strict.items():
        contrast = sum(
            float(pos["trait_score"]) - float(neg["trait_score"])
            for pos, neg in pairs
        ) / len(pairs)
        by_question[key[0]].append((key, contrast))
    if set(by_question) != set(range(20)):
        raise ValueError("Strict prompt pool does not cover all 20 questions.")
    for rows in by_question.values():
        rows.sort(key=lambda item: (-item[1], item[0][1]))
    prompt_manifest_path = None
    if args.prompt_manifest is None:
        selected_keys = _balanced_prompt_selection(
            by_question, total=args.discovery_prompts
        )
    else:
        prompt_manifest_path = (
            args.prompt_manifest
            if args.prompt_manifest.is_absolute()
            else ROOT / args.prompt_manifest
        )
        prompt_manifest = json.loads(
            prompt_manifest_path.read_text(encoding="utf-8")
        )
        if (
            prompt_manifest.get("status") != "frozen"
            or prompt_manifest["sources"]["positive_sha256"]
            != sha256_file(positive_path)
            or prompt_manifest["sources"]["negative_sha256"]
            != sha256_file(negative_path)
        ):
            raise ValueError("Prompt-source manifest failed its provenance gate.")
        selected_keys = [
            (int(row["question_index"]), int(row["instruction_index"]))
            for row in prompt_manifest["discovery"]["samples"]
        ]
        if len(selected_keys) != args.discovery_prompts:
            raise ValueError("Prompt-source manifest has the wrong sample count.")
        missing = [key for key in selected_keys if key not in strict]
        if missing:
            raise ValueError(
                f"{len(missing)} frozen prompt identities have no eligible "
                "length-complete response pair."
            )
    if len(selected_keys) != args.discovery_prompts:
        raise RuntimeError("Could not construct the requested discovery split.")

    discovery = []
    discovery_pair_ids = set()
    for key in selected_keys:
        candidates = sorted(
            strict[key],
            key=lambda pair: _hash_rank(args.seed, pair[0]["pair_id"]),
        )
        pos, neg = candidates[0]
        discovery_pair_ids.add(pos["pair_id"])
        discovery.append(
            {
                "question_index": key[0],
                "instruction_index": key[1],
                "pair_id": pos["pair_id"],
                "positive_record_id": pos["record_id"],
                "negative_record_id": neg["record_id"],
                "positive_prompt": pos["prompt"],
                "negative_prompt": neg["prompt"],
                "positive_trait_score": float(pos["trait_score"]),
                "negative_trait_score": float(neg["trait_score"]),
                "positive_coherence_score": float(pos["coherence_score"]),
                "negative_coherence_score": float(neg["coherence_score"]),
                **(
                    {
                        "positive_transcript_tokens": int(
                            pos["_positive_transcript_tokens"]
                        ),
                        "negative_transcript_tokens": int(
                            pos["_negative_transcript_tokens"]
                        ),
                    }
                    if tokenizer is not None
                    else {}
                ),
            }
        )

    remaining_by_question: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for key, pairs in strict.items():
        for pos, neg in pairs:
            if pos["pair_id"] not in discovery_pair_ids:
                remaining_by_question[key[0]].append((pos, neg))
    rng = random.Random(args.seed + 1)
    for rows in remaining_by_question.values():
        rng.shuffle(rows)
    downstream_eval_pairs = []
    depth = 0
    while len(downstream_eval_pairs) < args.downstream_eval_pairs:
        added = False
        for question in sorted(remaining_by_question):
            rows = remaining_by_question[question]
            if depth < len(rows):
                downstream_eval_pairs.append(rows[depth])
                added = True
                if len(downstream_eval_pairs) == args.downstream_eval_pairs:
                    break
        if not added:
            break
        depth += 1
    if len(downstream_eval_pairs) != args.downstream_eval_pairs:
        raise RuntimeError("Insufficient response-disjoint downstream eval pairs.")

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    behavior = [
        {
            "question_index": question_index,
            "instruction_index": question_index % len(eval_data["instruction"]),
        }
        for question_index in range(len(eval_data["questions"]))
    ]
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "protocol": (
            "shared_prompt_full_transcript_v2"
            if args.max_transcript_length is not None
            else "shared_prompt_v1"
        ),
        "seed": args.seed,
        "thresholds": {
            "positive_trait_min": 80,
            "negative_trait_max": 20,
            "positive_coherence_min": 80,
            "negative_coherence_min": 80,
            "max_transcript_length": args.max_transcript_length,
        },
        "sources": {
            "positive": str(positive_path.relative_to(ROOT)),
            "positive_sha256": sha256_file(positive_path),
            "negative": str(negative_path.relative_to(ROOT)),
            "negative_sha256": sha256_file(negative_path),
            "extract": str(extract_path.relative_to(ROOT)),
            "extract_sha256": sha256_file(extract_path),
            "eval": str(eval_path.relative_to(ROOT)),
            "eval_sha256": sha256_file(eval_path),
            **(
                {
                    "prompt_manifest": str(
                        prompt_manifest_path.relative_to(ROOT)
                    ),
                    "prompt_manifest_sha256": sha256_file(
                        prompt_manifest_path
                    ),
                }
                if prompt_manifest_path is not None
                else {}
            ),
        },
        "tokenization": (
            {
                "model": config["model"]["name"],
                "revision": config["model"].get("revision"),
                "same_positive_prompt_for_both_answers": True,
                "truncation": False,
            }
            if tokenizer is not None
            else None
        ),
        "discovery": {
            "n_unique_prompts": len(discovery),
            "question_counts": dict(
                sorted(Counter(row["question_index"] for row in discovery).items())
            ),
            "instruction_counts": dict(
                sorted(
                    Counter(row["instruction_index"] for row in discovery).items()
                )
            ),
            "samples": discovery,
        },
        "downstream_internal_evaluation": {
            "n_response_disjoint_pairs": len(downstream_eval_pairs),
            "samples": [
                {
                    "question_index": int(pos["question_index"]),
                    "instruction_index": int(pos["instruction_index"]),
                    "pair_id": pos["pair_id"],
                    "positive_record_id": pos["record_id"],
                    "negative_record_id": neg["record_id"],
                }
                for pos, neg in downstream_eval_pairs
            ],
        },
        "behavioral_evaluation": {
            "source_split": "official eval.json, disjoint questions from extract.json",
            "n_prompts": len(behavior),
            "samples": behavior,
        },
    }
    output = args.output or (intermediate / "shared_circuit_dataset_v1.json")
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        normalized_payload = json.loads(json.dumps(payload, sort_keys=True))
        if existing != normalized_payload:
            raise ValueError(f"Frozen manifest differs from reconstruction: {output}")
        print(f"verified: {output}")
        return
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()

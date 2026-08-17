#!/usr/bin/env python3
"""Freeze prompt-disjoint downstream and end-to-end evaluation pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_negative_prompt


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _unique_by_pair_id(rows: list[dict], *, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in indexed:
            raise ValueError(f"Duplicate {label} pair_id: {pair_id}")
        indexed[pair_id] = row
    return indexed


def _validate_counterfactual_answer_alignment(
    tokenizer,
    *,
    positive_prompt: str,
    aligned_negative_prompt: str,
    answer: str,
) -> dict[str, int]:
    """Verify that prompt concatenation preserves an identical answer suffix."""

    positive_prompt_ids = tokenizer.encode(
        positive_prompt, add_special_tokens=False
    )
    negative_prompt_ids = tokenizer.encode(
        aligned_negative_prompt, add_special_tokens=False
    )
    if len(positive_prompt_ids) != len(negative_prompt_ids):
        raise ValueError("Aligned positive and negative prompts differ in length.")
    response_start = len(positive_prompt_ids)
    clean_ids = tokenizer.encode(
        positive_prompt + answer, add_special_tokens=False
    )
    corrupt_ids = tokenizer.encode(
        aligned_negative_prompt + answer, add_special_tokens=False
    )
    if (
        len(clean_ids) != len(corrupt_ids)
        or clean_ids[:response_start] != positive_prompt_ids
        or corrupt_ids[:response_start] != negative_prompt_ids
        or clean_ids[response_start:] != corrupt_ids[response_start:]
        or not clean_ids[response_start:]
    ):
        raise ValueError(
            "Answer boundary is not stable across the prompt counterfactual."
        )
    return {
        "response_start": response_start,
        "clean_tokens": len(clean_ids),
        "corrupt_tokens": len(corrupt_ids),
        "answer_tokens": len(clean_ids) - response_start,
    }


def _select_balanced_identities(
    candidates: list[dict],
    *,
    n: int,
    seed: int,
) -> list[dict]:
    """Select unique prompt identities with balanced question/instruction use."""

    identities = [
        (int(row["question_index"]), int(row["instruction_index"]))
        for row in candidates
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Candidates must contain one row per prompt identity.")
    if len(candidates) < n:
        raise ValueError(f"Only {len(candidates)} identities are available for {n} rows.")

    remaining = list(candidates)
    selected = []
    question_counts: Counter = Counter()
    instruction_counts: Counter = Counter()
    while len(selected) < n:
        row = min(
            remaining,
            key=lambda item: (
                question_counts[int(item["question_index"])],
                instruction_counts[int(item["instruction_index"])],
                _rank(seed, str(item["pair_id"])),
            ),
        )
        remaining.remove(row)
        selected.append(row)
        question_counts[int(row["question_index"])] += 1
        instruction_counts[int(row["instruction_index"])] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_pairs <= 0 or args.max_length <= 0:
        raise ValueError("Pair count and max length must be positive.")

    from transformers import AutoTokenizer

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    intermediate = ROOT / config["paths"]["intermediate_root"]
    official = ROOT / config["paths"]["official_root"]
    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    extract_path = official / "extract.json"
    behavior_eval_path = official / "eval.json"
    discovery_manifest_path = (
        args.discovery_manifest
        if args.discovery_manifest.is_absolute()
        else ROOT / args.discovery_manifest
    )
    discovery_manifest = json.loads(
        discovery_manifest_path.read_text(encoding="utf-8")
    )
    if (
        discovery_manifest.get("status") != "frozen"
        or discovery_manifest.get("protocol")
        not in {"shared_prompt_v1", "shared_prompt_full_transcript_v2"}
        or discovery_manifest["sources"]["positive_sha256"]
        != sha256_file(positive_path)
        or discovery_manifest["sources"]["negative_sha256"]
        != sha256_file(negative_path)
        or discovery_manifest["sources"]["extract_sha256"]
        != sha256_file(extract_path)
        or discovery_manifest["sources"]["eval_sha256"]
        != sha256_file(behavior_eval_path)
    ):
        raise ValueError("Discovery manifest failed its provenance gate.")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        revision=config["model"].get("revision"),
        local_files_only=True,
    )
    trait_data = json.loads(extract_path.read_text(encoding="utf-8"))
    behavior_data = json.loads(behavior_eval_path.read_text(encoding="utf-8"))
    extract_questions = {
        str(question).strip().casefold() for question in trait_data["questions"]
    }
    behavior_questions = {
        str(question).strip().casefold() for question in behavior_data["questions"]
    }
    behavior_overlap = extract_questions & behavior_questions
    if behavior_overlap:
        raise ValueError(
            "Official extraction and behavioral evaluation questions overlap."
        )

    thresholds = discovery_manifest["thresholds"]
    positive_min = float(thresholds["positive_trait_min"])
    negative_max = float(thresholds["negative_trait_max"])
    positive_coherence_min = float(thresholds["positive_coherence_min"])
    negative_coherence_min = float(thresholds["negative_coherence_min"])
    discovery_samples = discovery_manifest["discovery"]["samples"]
    discovery_identities = {
        (int(row["question_index"]), int(row["instruction_index"]))
        for row in discovery_samples
    }
    if len(discovery_identities) != len(discovery_samples):
        raise ValueError(
            "Discovery manifest contains duplicate prompt identities."
        )
    discovery_pair_ids = {
        str(row["pair_id"]) for row in discovery_samples
    }
    if len(discovery_pair_ids) != len(discovery_samples):
        raise ValueError(
            "Discovery manifest contains duplicate response-pair IDs."
        )
    positive_rows = read_jsonl(positive_path)
    negative_by_pair = _unique_by_pair_id(
        read_jsonl(negative_path), label="negative"
    )
    _unique_by_pair_id(positive_rows, label="positive")
    eligible: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for positive in positive_rows:
        pair_id = str(positive["pair_id"])
        negative = negative_by_pair.get(pair_id)
        if negative is None:
            continue
        question_index = int(positive["question_index"])
        instruction_index = int(positive["instruction_index"])
        identity = (question_index, instruction_index)
        if identity in discovery_identities or pair_id in discovery_pair_ids:
            continue
        if (
            float(positive["trait_score"]) < positive_min
            or float(negative["trait_score"]) > negative_max
            or float(positive["coherence_score"]) < positive_coherence_min
            or float(negative["coherence_score"]) < negative_coherence_min
        ):
            continue
        if (
            int(negative["question_index"]) != question_index
            or int(negative["instruction_index"]) != instruction_index
        ):
            raise ValueError(f"Pair identity mismatch: {pair_id}")
        try:
            aligned_negative, alignment = _align_negative_prompt(
                tokenizer,
                question=trait_data["questions"][question_index],
                negative_assistant_name=config["trait"][
                    "negative_assistant_name"
                ],
                negative_instruction=trait_data["instruction"][
                    instruction_index
                ]["neg"],
                positive_prompt=positive["prompt"],
                expected_negative_prompt=negative["prompt"],
            )
        except ValueError:
            continue
        try:
            positive_boundary = _validate_counterfactual_answer_alignment(
                tokenizer,
                positive_prompt=positive["prompt"],
                aligned_negative_prompt=aligned_negative,
                answer=positive["answer"],
            )
            negative_boundary = _validate_counterfactual_answer_alignment(
                tokenizer,
                positive_prompt=positive["prompt"],
                aligned_negative_prompt=aligned_negative,
                answer=negative["answer"],
            )
        except ValueError:
            continue
        response_start = positive_boundary["response_start"]
        if response_start != negative_boundary["response_start"]:
            raise RuntimeError("Paired answers have inconsistent response starts.")
        if (
            max(
                positive_boundary["clean_tokens"],
                positive_boundary["corrupt_tokens"],
                negative_boundary["clean_tokens"],
                negative_boundary["corrupt_tokens"],
            )
            > args.max_length
        ):
            continue
        eligible[identity].append(
            {
                "question_index": question_index,
                "instruction_index": instruction_index,
                "pair_id": pair_id,
                "positive_record_id": positive["record_id"],
                "negative_record_id": negative["record_id"],
                "positive_prompt": positive["prompt"],
                "aligned_negative_prompt": aligned_negative,
                "alignment": alignment,
                "response_start": response_start,
                "positive_transcript_tokens": positive_boundary["clean_tokens"],
                "negative_transcript_tokens": negative_boundary["clean_tokens"],
                "counterfactual_transcript_tokens": {
                    "positive_answer": positive_boundary,
                    "negative_answer": negative_boundary,
                },
            }
        )

    identity_candidates = []
    for identity, rows in eligible.items():
        row = min(rows, key=lambda item: _rank(args.seed, str(item["pair_id"])))
        identity_candidates.append(row)
    selected = _select_balanced_identities(
        identity_candidates,
        n=args.num_pairs,
        seed=args.seed,
    )
    selected_identities = {
        (int(row["question_index"]), int(row["instruction_index"]))
        for row in selected
    }
    selected_pair_ids = {str(row["pair_id"]) for row in selected}
    if selected_identities & discovery_identities:
        raise RuntimeError("Prompt-disjoint selection overlaps discovery.")
    if selected_pair_ids & discovery_pair_ids:
        raise RuntimeError("Response-disjoint selection overlaps discovery.")

    question_counts = Counter(int(row["question_index"]) for row in selected)
    instruction_counts = Counter(int(row["instruction_index"]) for row in selected)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = {
        "schema_version": 2,
        "status": "frozen",
        "protocol": "prompt_response_disjoint_downstream_evaluation_v2",
        "purpose": (
            "Select downstream and end-to-end circuit size without reusing "
            "discovery prompt identities, discovery response pairs, or official "
            "behavioral questions."
        ),
        "model": config["model"],
        "trait": config["trait"]["name"],
        "seed": args.seed,
        "max_length": args.max_length,
        "truncation": False,
        "thresholds": {
            "positive_trait_min": positive_min,
            "negative_trait_max": negative_max,
            "positive_coherence_min": positive_coherence_min,
            "negative_coherence_min": negative_coherence_min,
        },
        "discovery_manifest": str(discovery_manifest_path.relative_to(ROOT)),
        "discovery_manifest_sha256": sha256_file(discovery_manifest_path),
        "sources": {
            "positive": str(positive_path.relative_to(ROOT)),
            "positive_sha256": sha256_file(positive_path),
            "negative": str(negative_path.relative_to(ROOT)),
            "negative_sha256": sha256_file(negative_path),
            "extract": str(extract_path.relative_to(ROOT)),
            "extract_sha256": sha256_file(extract_path),
            "behavior_eval": str(behavior_eval_path.relative_to(ROOT)),
            "behavior_eval_sha256": sha256_file(behavior_eval_path),
        },
        "selection": {
            "n_pairs": len(selected),
            "discovery_identity_overlap": 0,
            "discovery_pair_overlap": 0,
            "behavior_question_text_overlap": 0,
            "question_counts": dict(sorted(question_counts.items())),
            "instruction_counts": dict(sorted(instruction_counts.items())),
            "samples": selected,
        },
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Frozen evaluation manifest differs: {output}")
        print(f"verified: {output}")
        return
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()

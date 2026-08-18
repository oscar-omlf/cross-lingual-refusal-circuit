#!/usr/bin/env python3
"""Freeze the question-disjoint Phase 1 causal-interface data split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_sycophancy.json",
    )
    parser.add_argument("--positive-scored", type=Path, default=None)
    parser.add_argument("--negative-scored", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--minimum-interface-pairs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.minimum_interface_pairs <= 0:
        raise ValueError("--minimum-interface-pairs must be positive.")

    from refusal_circuit.causal_interface import (
        freeze_question_split,
        question_count,
    )
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        align_effective_pairs,
        read_jsonl,
        sha256_file,
    )

    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    intermediate = ROOT / config["paths"]["intermediate_root"]
    official = ROOT / config["paths"]["official_root"]
    positive_path = _resolve(args.positive_scored) if args.positive_scored else (
        intermediate / "judged/pos_scored.jsonl"
    )
    negative_path = _resolve(args.negative_scored) if args.negative_scored else (
        intermediate / "judged/neg_scored.jsonl"
    )
    extract_path = official / "extract.json"
    eval_path = official / "eval.json"
    paths = {
        "config": config_path,
        "positive_scored": positive_path,
        "negative_scored": negative_path,
        "extract": extract_path,
        "eval": eval_path,
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    positive = read_jsonl(positive_path)
    negative = read_jsonl(negative_path)
    extraction = config["extraction"]
    expected = (
        int(extraction["questions"])
        * int(extraction["instructions_per_polarity"])
        * int(extraction["rollouts_per_question_instruction"])
    )
    if len(positive) != expected or len(negative) != expected:
        raise ValueError(
            f"Expected {expected} rows per polarity, got "
            f"{len(positive)} and {len(negative)}."
        )
    pairs = align_effective_pairs(
        positive,
        negative,
        trait_threshold=float(extraction["trait_threshold"]),
        coherence_threshold=float(extraction["coherence_threshold"]),
    )
    eligible_counts = question_count([positive_row for positive_row, _ in pairs])
    eligible_counts = {index: eligible_counts.get(index, 0) for index in range(20)}
    eligible_prompt_counts = Counter(
        (
            int(positive_row["question_index"]),
            int(positive_row["instruction_index"]),
        )
        for positive_row, _ in pairs
    )
    eligible_prompt_counts = {
        (question, instruction): eligible_prompt_counts.get(
            (question, instruction), 0
        )
        for question in range(20)
        for instruction in range(5)
    }
    extract = json.loads(extract_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    manifest = freeze_question_split(
        extraction_questions=extract["questions"],
        behavioral_questions=evaluation["questions"],
        eligible_pair_counts=eligible_counts,
        eligible_prompt_counts=eligible_prompt_counts,
        source_hashes={
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in paths.values()
        },
        model=config["model"],
        tokenizer_identity={
            "repository": config["model"]["name"],
            "revision": config["model"].get("revision"),
            "chat_template_source": "pinned Hugging Face tokenizer revision",
        },
        seed=args.seed,
        minimum_interface_pairs=args.minimum_interface_pairs,
    )
    output = _resolve(args.output) if args.output else (
        intermediate / "causal_interface_phase1/split_manifest.json"
    )
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"Frozen split differs from reconstruction: {output}")
        print(f"verified: {output}")
        return
    write_json(output, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

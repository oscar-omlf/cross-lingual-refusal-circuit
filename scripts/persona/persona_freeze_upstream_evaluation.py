#!/usr/bin/env python3
"""Freeze prompt-disjoint upstream persona circuit evaluation pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_negative_prompt


def _rank(seed: int, question: int, instruction: int) -> str:
    value = f"{seed}:{question}:{instruction}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_prompts <= 0 or args.max_length <= 0:
        raise ValueError("Prompt count and max length must be positive.")

    from transformers import AutoTokenizer

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import make_system_prompt, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    official_root = ROOT / config["paths"]["official_root"]
    extract_path = official_root / "extract.json"
    evaluation_path = official_root / "eval.json"
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
        or discovery_manifest["sources"]["extract_sha256"]
        != sha256_file(extract_path)
        or discovery_manifest["sources"]["eval_sha256"]
        != sha256_file(evaluation_path)
    ):
        raise ValueError("Discovery manifest failed its provenance gate.")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        revision=config["model"].get("revision"),
        local_files_only=True,
    )
    trait_data = json.loads(extract_path.read_text(encoding="utf-8"))
    behavior_data = json.loads(
        evaluation_path.read_text(encoding="utf-8")
    )
    extract_questions = {
        str(question).strip().casefold()
        for question in trait_data["questions"]
    }
    behavior_questions = {
        str(question).strip().casefold()
        for question in behavior_data["questions"]
    }
    behavior_overlap = extract_questions & behavior_questions
    if behavior_overlap:
        raise ValueError(
            "Official extraction and behavioral evaluation questions overlap."
        )
    discovery_samples = discovery_manifest["discovery"]["samples"]
    discovery_identities = {
        (int(row["question_index"]), int(row["instruction_index"]))
        for row in discovery_samples
    }
    if len(discovery_identities) != len(discovery_samples):
        raise ValueError(
            "Discovery manifest contains duplicate prompt identities."
        )
    candidates = []
    for question_index, question in enumerate(trait_data["questions"]):
        for instruction_index, instruction in enumerate(
            trait_data["instruction"]
        ):
            identity = (question_index, instruction_index)
            if identity in discovery_identities:
                continue
            positive_system = make_system_prompt(
                config["trait"]["positive_assistant_name"],
                instruction["pos"],
            )
            positive_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": positive_system},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            try:
                aligned_negative, alignment = _align_negative_prompt(
                    tokenizer,
                    question=question,
                    negative_assistant_name=config["trait"][
                        "negative_assistant_name"
                    ],
                    negative_instruction=instruction["neg"],
                    positive_prompt=positive_prompt,
                )
            except ValueError:
                continue
            if int(alignment["positive_tokens"]) > args.max_length:
                continue
            candidates.append(
                {
                    "question_index": question_index,
                    "instruction_index": instruction_index,
                    "positive_prompt": positive_prompt,
                    "aligned_negative_prompt": aligned_negative,
                    "alignment": alignment,
                    "_rank": _rank(
                        args.seed, question_index, instruction_index
                    ),
                }
            )
    if len(candidates) < args.num_prompts:
        raise ValueError(
            f"Only {len(candidates)} prompt-disjoint aligned pairs fit, "
            f"but {args.num_prompts} were requested."
        )

    selected = []
    remaining = list(candidates)
    question_counts: Counter = Counter()
    instruction_counts: Counter = Counter()
    while len(selected) < args.num_prompts:
        row = min(
            remaining,
            key=lambda item: (
                question_counts[item["question_index"]],
                instruction_counts[item["instruction_index"]],
                item["_rank"],
            ),
        )
        remaining.remove(row)
        selected.append(row)
        question_counts[row["question_index"]] += 1
        instruction_counts[row["instruction_index"]] += 1
    for row in selected:
        row.pop("_rank")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "protocol": "prompt_disjoint_upstream_evaluation_v1",
        "purpose": (
            "Select upstream circuit size without reusing discovery prompt "
            "identities or official behavioral evaluation questions."
        ),
        "model": config["model"],
        "trait": config["trait"]["name"],
        "seed": args.seed,
        "max_length": args.max_length,
        "truncation": False,
        "discovery_manifest": str(
            discovery_manifest_path.relative_to(ROOT)
        ),
        "discovery_manifest_sha256": sha256_file(
            discovery_manifest_path
        ),
        "sources": {
            "extract": str(extract_path.relative_to(ROOT)),
            "extract_sha256": sha256_file(extract_path),
            "behavior_eval": str(evaluation_path.relative_to(ROOT)),
            "behavior_eval_sha256": sha256_file(evaluation_path),
        },
        "selection": {
            "n_prompts": len(selected),
            "discovery_identity_overlap": 0,
            "behavior_question_text_overlap": 0,
            "source_split": (
                "official extract.json prompt identities excluded from "
                "discovery; official eval.json remains untouched"
            ),
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

#!/usr/bin/env python3
"""Audit persona prompt/transcript lengths without running the model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_negative_prompt


def _summary(lengths: list[int], caps: list[int]) -> dict:
    if not lengths:
        raise ValueError("Cannot summarize an empty length family.")
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        index = math.ceil(fraction * len(ordered)) - 1
        return ordered[max(0, min(index, len(ordered) - 1))]

    return {
        "rows": len(ordered),
        "min": ordered[0],
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "caps": {
            str(cap): {
                "fits_without_truncation": sum(length <= cap for length in ordered),
                "exceeds": sum(length > cap for length in ordered),
                "safe_without_rerun": all(length <= cap for length in ordered),
            }
            for cap in caps
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument(
        "--upstream-evaluation-manifest",
        type=Path,
        default=None,
        help="Optional prompt-disjoint upstream internal-evaluation manifest.",
    )
    parser.add_argument(
        "--downstream-evaluation-manifest",
        type=Path,
        default=None,
        help=(
            "Optional prompt- and response-disjoint downstream/end-to-end "
            "internal-evaluation manifest."
        ),
    )
    parser.add_argument(
        "--caps",
        type=int,
        nargs="+",
        default=(256, 350, 500, 640, 768, 1024),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(cap <= 0 for cap in args.caps):
        raise ValueError("All token caps must be positive.")
    if len(args.caps) != len(set(args.caps)):
        raise ValueError("Token caps must be unique.")

    from transformers import AutoTokenizer

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = (
        args.sample_manifest
        if args.sample_manifest.is_absolute()
        else ROOT / args.sample_manifest
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ValueError("Sample manifest is not frozen.")

    intermediate = ROOT / config["paths"]["intermediate_root"]
    official = ROOT / config["paths"]["official_root"]
    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    extract_path = official / "extract.json"
    eval_path = official / "eval.json"
    if (
        manifest["sources"]["positive_sha256"] != sha256_file(positive_path)
        or manifest["sources"]["negative_sha256"] != sha256_file(negative_path)
        or manifest["sources"]["extract_sha256"] != sha256_file(extract_path)
        or manifest["sources"]["eval_sha256"] != sha256_file(eval_path)
    ):
        raise ValueError("Manifest source hashes do not match the local artifacts.")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        revision=config["model"].get("revision"),
        local_files_only=True,
    )
    positive = {
        row["record_id"]: row for row in read_jsonl(positive_path)
    }
    negative = {
        row["record_id"]: row for row in read_jsonl(negative_path)
    }
    extract = json.loads(extract_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    families: dict[str, list[int]] = {
        "discovery_upstream_prompt": [],
        "discovery_downstream_transcript": [],
        "discovery_end_to_end_transcript": [],
        "internal_upstream_prompt": [],
        "internal_downstream_transcript": [],
        "internal_end_to_end_transcript": [],
        "official_behavior_prompt": [],
    }

    def token_length(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def aligned_negative(
        *,
        question: str,
        instruction: dict,
        positive_prompt: str,
        expected_negative_prompt: str | None = None,
    ) -> str:
        aligned, _ = _align_negative_prompt(
            tokenizer,
            question=question,
            negative_assistant_name=config["trait"]["negative_assistant_name"],
            negative_instruction=instruction["neg"],
            positive_prompt=positive_prompt,
            expected_negative_prompt=expected_negative_prompt,
        )
        return aligned

    for sample in manifest["discovery"]["samples"]:
        pos = positive[sample["positive_record_id"]]
        neg = negative[sample["negative_record_id"]]
        question_index = int(sample["question_index"])
        instruction_index = int(sample["instruction_index"])
        instruction = extract["instruction"][instruction_index]
        aligned = aligned_negative(
            question=extract["questions"][question_index],
            instruction=instruction,
            positive_prompt=pos["prompt"],
            expected_negative_prompt=neg["prompt"],
        )
        families["discovery_upstream_prompt"].extend(
            [token_length(pos["prompt"]), token_length(aligned)]
        )
        downstream_texts = (
            pos["prompt"] + pos["answer"],
            pos["prompt"] + neg["answer"],
        )
        families["discovery_downstream_transcript"].extend(
            token_length(text) for text in downstream_texts
        )
        end_to_end_texts = (
            *downstream_texts,
            aligned + pos["answer"],
            aligned + neg["answer"],
        )
        families["discovery_end_to_end_transcript"].extend(
            token_length(text) for text in end_to_end_texts
        )

    upstream_evaluation_manifest = None
    if args.upstream_evaluation_manifest is not None:
        upstream_evaluation_path = (
            args.upstream_evaluation_manifest
            if args.upstream_evaluation_manifest.is_absolute()
            else ROOT / args.upstream_evaluation_manifest
        )
        upstream_evaluation_manifest = json.loads(
            upstream_evaluation_path.read_text(encoding="utf-8")
        )
        if (
            upstream_evaluation_manifest.get("status") != "frozen"
            or upstream_evaluation_manifest.get("protocol")
            != "prompt_disjoint_upstream_evaluation_v1"
            or upstream_evaluation_manifest.get("model") != config["model"]
            or upstream_evaluation_manifest.get("trait")
            != config["trait"]["name"]
            or upstream_evaluation_manifest["discovery_manifest_sha256"]
            != sha256_file(manifest_path)
        ):
            raise ValueError(
                "Prompt-disjoint upstream evaluation manifest failed provenance."
            )
        for sample in upstream_evaluation_manifest["selection"]["samples"]:
            families["internal_upstream_prompt"].extend(
                [
                    token_length(sample["positive_prompt"]),
                    token_length(sample["aligned_negative_prompt"]),
                ]
            )

    downstream_evaluation_manifest = None
    downstream_evaluation_path = None
    downstream_samples = manifest["downstream_internal_evaluation"]["samples"]
    if args.downstream_evaluation_manifest is not None:
        downstream_evaluation_path = (
            args.downstream_evaluation_manifest
            if args.downstream_evaluation_manifest.is_absolute()
            else ROOT / args.downstream_evaluation_manifest
        )
        downstream_evaluation_manifest = json.loads(
            downstream_evaluation_path.read_text(encoding="utf-8")
        )
        if (
            downstream_evaluation_manifest.get("status") != "frozen"
            or downstream_evaluation_manifest.get("protocol")
            != "prompt_response_disjoint_downstream_evaluation_v2"
            or downstream_evaluation_manifest.get("model") != config["model"]
            or downstream_evaluation_manifest.get("trait")
            != config["trait"]["name"]
            or downstream_evaluation_manifest["discovery_manifest_sha256"]
            != sha256_file(manifest_path)
        ):
            raise ValueError(
                "Prompt-disjoint downstream evaluation manifest failed provenance."
            )
        downstream_samples = downstream_evaluation_manifest[
            "selection"
        ]["samples"]

    for sample in downstream_samples:
        pos = positive[sample["positive_record_id"]]
        neg = negative[sample["negative_record_id"]]
        question_index = int(sample["question_index"])
        instruction_index = int(sample["instruction_index"])
        instruction = extract["instruction"][instruction_index]
        aligned = aligned_negative(
            question=extract["questions"][question_index],
            instruction=instruction,
            positive_prompt=pos["prompt"],
            expected_negative_prompt=neg["prompt"],
        )
        if upstream_evaluation_manifest is None:
            families["internal_upstream_prompt"].extend(
                [token_length(pos["prompt"]), token_length(aligned)]
            )
        downstream_texts = (
            pos["prompt"] + pos["answer"],
            pos["prompt"] + neg["answer"],
        )
        families["internal_downstream_transcript"].extend(
            token_length(text) for text in downstream_texts
        )
        families["internal_end_to_end_transcript"].extend(
            token_length(text)
            for text in (
                *downstream_texts,
                aligned + pos["answer"],
                aligned + neg["answer"],
            )
        )

    for question_index, question in enumerate(evaluation["questions"]):
        instruction = evaluation["instruction"][
            question_index % len(evaluation["instruction"])
        ]
        from refusal_circuit.persona import make_system_prompt

        positive_system = make_system_prompt(
            config["trait"]["positive_assistant_name"], instruction["pos"]
        )
        positive_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": positive_system},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        negative_prompt = aligned_negative(
            question=question,
            instruction=instruction,
            positive_prompt=positive_prompt,
        )
        families["official_behavior_prompt"].extend(
            [token_length(positive_prompt), token_length(negative_prompt)]
        )

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"{output} already exists.")
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": (
            "Determine whether historical max_length differences changed any "
            "teacher-forced or prompt estimand. A cap is reusable exactly when "
            "the relevant family has exceeds=0."
        ),
        "model": config["model"],
        "sample_manifest": str(manifest_path.relative_to(ROOT)),
        "sample_manifest_sha256": sha256_file(manifest_path),
        "upstream_evaluation_manifest": (
            str(upstream_evaluation_path.relative_to(ROOT))
            if upstream_evaluation_manifest is not None
            else None
        ),
        "upstream_evaluation_manifest_sha256": (
            sha256_file(upstream_evaluation_path)
            if upstream_evaluation_manifest is not None
            else None
        ),
        "downstream_evaluation_manifest": (
            str(downstream_evaluation_path.relative_to(ROOT))
            if downstream_evaluation_manifest is not None
            else None
        ),
        "downstream_evaluation_manifest_sha256": (
            sha256_file(downstream_evaluation_path)
            if downstream_evaluation_manifest is not None
            else None
        ),
        "caps": sorted(args.caps),
        "families": {
            name: _summary(lengths, sorted(args.caps))
            for name, lengths in families.items()
        },
    }
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

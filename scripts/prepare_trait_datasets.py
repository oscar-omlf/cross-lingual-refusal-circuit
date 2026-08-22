#!/usr/bin/env python3
"""Prepare canonical refusal and sycophancy splits for the circuit experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "data" / "processed" / "qwen2.5-7b-instruct"
OUTPUT_ROOT = MODEL_ROOT / "trait_circuits"
PERSONA_COMMIT = "b8e0f044fe2410a6fad579f38324f03f13b4e917"
PERSONA_RAW_ROOT = ROOT / "data" / "raw" / "persona_vectors" / PERSONA_COMMIT
SPLIT_COUNTS = {
    "vector_extraction": 256,
    "vector_validation": 64,
    "eapig": 64,
    "circuit_selection": 64,
    "final_behavior": 300,
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_key(text: str) -> str:
    return hashlib.sha256(
        ("trait-circuit-splits-v1\0" + text).encode("utf-8")
    ).hexdigest()


def with_split(row: dict, split: str, source_path: Path) -> dict:
    output = dict(row)
    output["experiment_split"] = split
    output["experiment_source_path"] = str(source_path.relative_to(ROOT))
    return output


def prepare_refusal() -> dict:
    output = OUTPUT_ROOT / "refusal"
    sources = {
        "train": MODEL_ROOT / "polyrefuse" / "en" / "train.jsonl",
        "validation": MODEL_ROOT / "polyrefuse" / "en" / "val.jsonl",
        "test": MODEL_ROOT / "polyrefuse" / "en" / "test.jsonl",
        "harmful_development": (
            MODEL_ROOT / "polyrefuse_phase2_refusal" / "en" / "development.jsonl"
        ),
        "harmful_confirmation": (
            MODEL_ROOT / "polyrefuse_phase2_refusal" / "en" / "confirmation.jsonl"
        ),
        "harmful_reserve": (
            MODEL_ROOT / "polyrefuse_phase2_refusal" / "en" / "reserve.jsonl"
        ),
        "harmless_development": (
            MODEL_ROOT / "polyrefuse_phase2_harmless" / "en" / "development.jsonl"
        ),
        "harmless_reserve": (
            MODEL_ROOT / "polyrefuse_phase2_harmless" / "en" / "reserve.jsonl"
        ),
    }
    rows = {name: read_jsonl(path) for name, path in sources.items()}

    extraction = [
        with_split(row, "vector_extraction", sources["train"])
        for row in rows["train"]
    ]
    validation = [
        with_split(row, "vector_validation", sources["validation"])
        for row in rows["validation"]
    ]

    harmful_development = sorted(
        rows["harmful_development"], key=lambda row: stable_key(row["instruction"])
    )
    eapig = [
        with_split(row, "eapig", sources["harmful_development"])
        for row in harmful_development[:64]
    ]
    circuit_selection = [
        with_split(row, "circuit_selection", sources["harmful_development"])
        for row in harmful_development[64:128]
    ]

    selected = extraction + validation + eapig + circuit_selection
    seen_text = {row["instruction"] for row in selected}
    seen_ids = {row["id"] for row in selected}
    excluded_duplicates: list[dict] = []

    def unique_candidates(source_names: list[str], label: str) -> list[dict]:
        candidates = []
        for source_name in source_names:
            for row in rows[source_name]:
                if row["label"] != label:
                    continue
                if row["id"] in seen_ids or row["instruction"] in seen_text:
                    excluded_duplicates.append(
                        {
                            "id": row["id"],
                            "instruction": row["instruction"],
                            "label": row["label"],
                            "source_path": str(sources[source_name].relative_to(ROOT)),
                            "reason": "duplicate_id_or_question_text_across_phases",
                        }
                    )
                    continue
                candidates.append(
                    with_split(row, "candidate", sources[source_name])
                )
                seen_ids.add(row["id"])
                seen_text.add(row["instruction"])
        return candidates

    harmless_candidates = unique_candidates(
        ["test", "harmless_development", "harmless_reserve"], "harmless"
    )
    if len(harmless_candidates) != 128:
        raise RuntimeError(
            f"Expected 128 unique harmless final prompts, got {len(harmless_candidates)}."
        )

    harmful_candidates = unique_candidates(
        ["test", "harmful_confirmation", "harmful_reserve"], "harmful"
    )
    if len(harmful_candidates) < 172:
        raise RuntimeError(
            f"Expected at least 172 unique harmful final prompts, got {len(harmful_candidates)}."
        )

    final_harmful = harmful_candidates[:172]
    reserve = harmful_candidates[172:]
    final_behavior = final_harmful + harmless_candidates
    final_behavior = sorted(
        (
            dict(row, experiment_split="final_behavior")
            for row in final_behavior
        ),
        key=lambda row: stable_key(row["instruction"]),
    )
    reserve = sorted(
        (dict(row, experiment_split="reserve") for row in reserve),
        key=lambda row: stable_key(row["instruction"]),
    )

    split_rows = {
        "vector_extraction": extraction,
        "vector_validation": validation,
        "eapig": eapig,
        "circuit_selection": circuit_selection,
        "final_behavior": final_behavior,
    }
    all_selected = [row for values in split_rows.values() for row in values]
    if len(all_selected) != 748:
        raise RuntimeError(f"Refusal selected count is {len(all_selected)}, expected 748.")
    if len({row["id"] for row in all_selected}) != 748:
        raise RuntimeError("Refusal IDs overlap across experiment splits.")
    if len({row["instruction"] for row in all_selected}) != 748:
        raise RuntimeError("Refusal question text overlaps across experiment splits.")

    for split, split_values in split_rows.items():
        write_jsonl(output / f"{split}.jsonl", split_values)
    write_jsonl(output / "reserve.jsonl", reserve)
    write_jsonl(output / "excluded_duplicate_prompts.jsonl", excluded_duplicates)

    manifest = {
        "schema_version": 1,
        "trait": "refusal",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "language": "en",
        "total_selected_unique_prompts": len(all_selected),
        "split_counts": {
            split: {
                "total": len(values),
                "labels": dict(Counter(row["label"] for row in values)),
            }
            for split, values in split_rows.items()
        },
        "reserve_unique_prompts": len(reserve),
        "excluded_duplicate_rows": len(excluded_duplicates),
        "source_rows": sum(len(value) for value in rows.values()),
        "source_files": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "rows": len(rows[name]),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
        "protocol": {
            "vector_extraction": "Existing paper-faithful 128 harmful + 128 harmless prompt split.",
            "vector_validation": "Existing 32 harmful + 32 harmless validation split.",
            "eapig": "First deterministic half of the 128 unseen harmful development prompts.",
            "circuit_selection": "Second deterministic half of the 128 unseen harmful development prompts.",
            "final_behavior": "172 unseen harmful + every remaining 128 harmless prompt.",
            "deduplication": "Exact instruction text is forbidden from crossing experiment phases.",
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def ensure_sycophancy_raw(zip_path: Path) -> dict[str, Path]:
    output = PERSONA_RAW_ROOT / "dataset" / "sycophancy"
    names = {
        "normal": "dataset/sycophancy/normal.jsonl",
        "misaligned_1": "dataset/sycophancy/misaligned_1.jsonl",
        "misaligned_2": "dataset/sycophancy/misaligned_2.jsonl",
    }
    paths = {name: output / f"{name}.jsonl" for name in names}
    if all(path.exists() for path in paths.values()):
        return paths
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Sycophancy raw data is absent and source archive was not found: {zip_path}"
        )
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for name, member in names.items():
            paths[name].write_bytes(archive.read(member))
    return paths


def conversation_parts(row: dict) -> tuple[str, str]:
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise ValueError("Expected exactly one user and one assistant message.")
    return messages[0]["content"], messages[1]["content"]


def prepare_sycophancy(zip_path: Path) -> dict:
    output = OUTPUT_ROOT / "sycophancy"
    raw_paths = ensure_sycophancy_raw(zip_path)
    raw = {name: read_jsonl(path) for name, path in raw_paths.items()}
    if {len(values) for values in raw.values()} != {10099}:
        raise RuntimeError("Expected 10,099 rows in each sycophancy answer file.")
    write_json(
        PERSONA_RAW_ROOT / "dataset" / "manifest.json",
        {
            "schema_version": 1,
            "dataset": "Persona Vectors sycophancy training data",
            "source_commit": PERSONA_COMMIT,
            "unique_prompts": 10099,
            "answer_variants_per_prompt": 3,
            "files": {
                name: {
                    "path": str(path.relative_to(ROOT)),
                    "rows": len(raw[name]),
                    "sha256": sha256_file(path),
                }
                for name, path in raw_paths.items()
            },
        },
    )

    instruction_pairs = json.loads(
        (PERSONA_RAW_ROOT / "extract.json").read_text(encoding="utf-8")
    )["instruction"]
    records = []
    for index, (normal, misaligned_1, misaligned_2) in enumerate(
        zip(raw["normal"], raw["misaligned_1"], raw["misaligned_2"])
    ):
        prompt, normal_answer = conversation_parts(normal)
        prompt_1, sycophantic_answer = conversation_parts(misaligned_1)
        prompt_2, extreme_sycophantic_answer = conversation_parts(misaligned_2)
        if prompt != prompt_1 or prompt != prompt_2:
            raise RuntimeError(f"Sycophancy prompt mismatch at source row {index}.")
        records.append(
            {
                "id": f"persona-sycophancy-{index:05d}",
                "source_index": index,
                "prompt": prompt,
                "normal_answer": normal_answer,
                "sycophantic_answer": sycophantic_answer,
                "extreme_sycophantic_answer": extreme_sycophantic_answer,
            }
        )
    if len({row["prompt"] for row in records}) != 10099:
        raise RuntimeError("Sycophancy source prompts are not unique.")

    records.sort(key=lambda row: stable_key(row["prompt"]))
    split_rows: dict[str, list[dict]] = {}
    offset = 0
    for split, count in SPLIT_COUNTS.items():
        selected = records[offset : offset + count]
        offset += count
        prepared = []
        for split_index, row in enumerate(selected):
            pair = instruction_pairs[split_index % len(instruction_pairs)]
            prepared.append(
                {
                    **row,
                    "experiment_split": split,
                    "instruction_pair_index": split_index % len(instruction_pairs),
                    "positive_system_instruction": pair["pos"],
                    "negative_system_instruction": pair["neg"],
                    "response_vector_positive_field": "sycophantic_answer",
                    "response_vector_negative_field": "normal_answer",
                }
            )
        split_rows[split] = prepared

    all_selected = [row for values in split_rows.values() for row in values]
    if len(all_selected) != 748:
        raise RuntimeError(f"Sycophancy selected count is {len(all_selected)}, expected 748.")
    if len({row["id"] for row in all_selected}) != 748:
        raise RuntimeError("Sycophancy IDs overlap across experiment splits.")
    if len({row["prompt"] for row in all_selected}) != 748:
        raise RuntimeError("Sycophancy prompts overlap across experiment splits.")
    for row in all_selected:
        if not all(
            row[field]
            for field in (
                "prompt",
                "normal_answer",
                "sycophantic_answer",
                "extreme_sycophantic_answer",
            )
        ):
            raise RuntimeError(f"Missing prompt or answer in {row['id']}.")

    for split, split_values in split_rows.items():
        write_jsonl(output / f"{split}.jsonl", split_values)

    manifest = {
        "schema_version": 1,
        "trait": "sycophancy",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "source_commit": PERSONA_COMMIT,
        "source_total_unique_prompts": len(records),
        "total_selected_unique_prompts": len(all_selected),
        "unused_source_prompts": len(records) - len(all_selected),
        "split_counts": {split: len(values) for split, values in split_rows.items()},
        "answers_stored_per_prompt": [
            "normal_answer",
            "sycophantic_answer",
            "extreme_sycophantic_answer",
        ],
        "source_files": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "rows": len(raw[name]),
                "sha256": sha256_file(path),
            }
            for name, path in raw_paths.items()
        },
        "protocol": {
            "split_order": "SHA-256 order of unique prompt text with trait-circuit-splits-v1 salt.",
            "prompt_vector": "Contrast positive versus negative Persona Vectors system instructions; no generation required.",
            "response_vector": "Teacher-force sycophantic_answer versus normal_answer; no generation required.",
            "instruction_assignment": "The five official instruction pairs are assigned cyclically within each split.",
            "extreme_answer": "Stored for diagnostics but not used in the primary response vector.",
            "final_behavior": "Internal 300-prompt holdout from the released 10,099; the paper's separate external 300 are not stored locally.",
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona-zip",
        type=Path,
        default=Path("/tmp/persona_vectors_dataset.zip"),
        help="Official Persona Vectors dataset archive, needed only if raw JSONL is absent.",
    )
    args = parser.parse_args()
    refusal = prepare_refusal()
    sycophancy = prepare_sycophancy(args.persona_zip)
    write_json(
        OUTPUT_ROOT / "manifest.json",
        {
            "schema_version": 1,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "traits": {
                "refusal": {
                    "path": "data/processed/qwen2.5-7b-instruct/trait_circuits/refusal",
                    "total_selected_unique_prompts": refusal[
                        "total_selected_unique_prompts"
                    ],
                },
                "sycophancy": {
                    "path": "data/processed/qwen2.5-7b-instruct/trait_circuits/sycophancy",
                    "total_selected_unique_prompts": sycophancy[
                        "total_selected_unique_prompts"
                    ],
                },
            },
            "shared_split_counts": SPLIT_COUNTS,
            "cross_stage_prompt_reuse": False,
        },
    )
    print(
        json.dumps(
            {
                "refusal": refusal["split_counts"],
                "sycophancy": sycophancy["split_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

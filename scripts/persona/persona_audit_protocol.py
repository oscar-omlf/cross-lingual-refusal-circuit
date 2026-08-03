#!/usr/bin/env python3
"""Audit a frozen Persona Vectors protocol without loading the language model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    args = parser.parse_args()

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import extraction_specs, hf_hidden_state_to_tl_hook, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = config["official_source"]
    root = _resolve(config["paths"]["official_root"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != source["commit"] or manifest["repository"] != source["repository"]:
        raise RuntimeError("download manifest does not match the frozen source")

    source_paths = {
        "extract": source["extract_artifact"],
        "eval": source["eval_artifact"],
        "coherence_source": source["coherence_source"],
        **source.get("implementation_sources", {}),
    }
    actual_hashes = {
        label: sha256_file(root / f"{label}{Path(repository_path).suffix}")
        for label, repository_path in source_paths.items()
    }
    if actual_hashes != source["expected_sha256"]:
        raise RuntimeError(f"official artifact hash mismatch: {actual_hashes}")
    extract = json.loads((root / "extract.json").read_text(encoding="utf-8"))
    evaluation = json.loads((root / "eval.json").read_text(encoding="utf-8"))
    extraction = config["extraction"]
    if len(extract["questions"]) != int(extraction["questions"]):
        raise RuntimeError("wrong extraction question count")
    if len(extract["instruction"]) != int(extraction["instructions_per_polarity"]):
        raise RuntimeError("wrong instruction-pair count")
    if len(evaluation["questions"]) != int(config["steering_validation"]["questions"]):
        raise RuntimeError("wrong held-out question count")
    question_overlap = sorted(set(extract["questions"]) & set(evaluation["questions"]))
    if question_overlap:
        raise RuntimeError(f"extract/eval questions overlap: {question_overlap[:3]}")
    if extract["instruction"] != evaluation["instruction"]:
        raise RuntimeError("extract/eval persona instruction pairs unexpectedly differ")

    rollouts = int(extraction["rollouts_per_question_instruction"])
    positive = list(
        extraction_specs(
            extract,
            polarity="pos",
            assistant_name=config["trait"]["positive_assistant_name"],
            rollouts=rollouts,
        )
    )
    negative = list(
        extraction_specs(
            extract,
            polarity="neg",
            assistant_name=config["trait"]["negative_assistant_name"],
            rollouts=rollouts,
        )
    )
    if {row.pair_id for row in positive} != {row.pair_id for row in negative}:
        raise RuntimeError("positive/negative extraction rows do not pair")
    expected_per_polarity = (
        int(extraction["questions"])
        * int(extraction["instructions_per_polarity"])
        * rollouts
    )
    if len(positive) != expected_per_polarity or len(negative) != expected_per_polarity:
        raise RuntimeError("factorial design size mismatch")

    vector = config["vector"]
    mapped_hook = hf_hidden_state_to_tl_hook(int(vector["hf_hidden_state_index"]))
    if mapped_hook != vector["transformer_lens_target_hook"]:
        raise RuntimeError("HF/TransformerLens endpoint mapping disagrees with config")
    if int(vector["transformer_lens_steering_block"]) != int(vector["paper_layer"]) - 1:
        raise RuntimeError("one-indexed paper layer does not map to the steering block")
    expected_protocol = {
        "questions": 20,
        "instructions_per_polarity": 5,
        "rollouts_per_question_instruction": 10,
        "temperature": 1.0,
        "top_p": 1.0,
        "min_new_tokens": 1,
        "max_new_tokens": 1000,
    }
    if any(extraction[key] != value for key, value in expected_protocol.items()):
        raise RuntimeError("extraction parameters differ from the released upstream pipeline")
    judge = config["judge"]
    if (
        judge["model"] != "gpt-4.1-mini-2025-04-14"
        or int(judge["seed"]) != 0
        or int(judge["top_logprobs"]) != 20
    ):
        raise RuntimeError("judge parameters differ from the released upstream pipeline")
    if vector["primary_variant"] != "response_avg_diff":
        raise RuntimeError("the paper's headline vector must be response_avg_diff")
    selection = config["layer_selection"]
    if selection["candidate_layers_one_indexed"] != [4, 8, 12, 16, 20, 24, 28]:
        raise RuntimeError("Qwen layer-selection grid differs from the released paper figure")
    if int(selection["paper_selected_layer_one_indexed"]) != int(
        vector["paper_layer"]
    ):
        raise RuntimeError("selected steering-sweep layer disagrees with vector endpoint")

    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    report = {
        "schema_version": 1,
        "status": "pass",
        "stage": "S0",
        "trait": config["trait"]["name"],
        "official_repository": source["repository"],
        "official_commit": source["commit"],
        "artifact_sha256": actual_hashes,
        "extract_questions": len(extract["questions"]),
        "heldout_questions": len(evaluation["questions"]),
        "extract_eval_question_overlap": 0,
        "shared_instruction_pairs": len(extract["instruction"]),
        "rows_per_polarity": expected_per_polarity,
        "paired_rows_total": expected_per_polarity * 2,
        "primary_vector": vector["primary_variant"],
        "paper_layer_one_indexed": int(vector["paper_layer"]),
        "hf_hidden_state_index": int(vector["hf_hidden_state_index"]),
        "transformer_lens_target_hook": mapped_hook,
        "steering_block_zero_indexed": int(vector["transformer_lens_steering_block"]),
        "layer_selection_grid_one_indexed": selection["candidate_layers_one_indexed"],
        "steering_coefficients": config["steering_validation"]["coefficients"],
        "generation_backend": "Transformers generate",
        "upstream_generation_backend": "vLLM generate",
        "scientific_pipeline_equivalent": True,
        "backend_is_implementation_choice": True,
        "backend_note": (
            "Both implement the frozen model/chat template and identical categorical sampling "
            "parameters; stochastic sample identities are not part of the protocol."
        ),
        "generation_ready": True,
        "canonical_judge_ready": has_openai_key,
        "canonical_judge_blocker": None if has_openai_key else "OPENAI_API_KEY is not set",
    }
    output = _resolve(config["paths"]["results_root"]) / "s0_protocol_audit.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

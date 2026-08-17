#!/usr/bin/env python3
"""Pool deterministic generation splits under an audited common token cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _compatibility_signature(metadata: dict, *, condition: str) -> dict:
    """Keep every field that changes generation semantics, except the cap."""

    declared_condition = metadata.get("condition")
    declared_conditions = metadata.get("conditions")
    if declared_condition not in (None, condition):
        raise ValueError("Source metadata condition does not match the requested row.")
    if declared_conditions is not None and condition not in declared_conditions:
        raise ValueError("Requested condition is absent from source metadata.")

    decoding = dict(metadata.get("decoding", {}))
    decoding.pop("max_new_tokens", None)
    if decoding.get("backend") is None:
        raise ValueError("Pooling requires an explicit generation backend.")
    keys = (
        "behavior_protocol_id",
        "method",
        "model",
        "model_architecture",
        "trait",
        "eval_artifact",
        "eval_artifact_sha256",
        "discovery_metadata",
        "discovery_metadata_sha256",
        "upstream_run",
        "downstream_run",
        "upstream_k",
        "downstream_k",
        "requested_k",
        "retained_edges",
        "circuit",
        "circuit_sha256",
        "upstream_circuit",
        "upstream_circuit_sha256",
        "downstream_circuit",
        "downstream_circuit_sha256",
        "direction",
        "direction_sha256",
        "response_vector",
        "response_vector_sha256",
        "unrelated_vector",
        "unrelated_vector_sha256",
        "unrelated_vector_cosine",
        "shared_prompt_identity_sha256",
        "upstream_sample_manifest_sha256",
        "downstream_sample_manifest_sha256",
        "counterfactual",
        "upstream_counterfactual",
        "downstream_counterfactual",
        "corrupt_coefficient",
        "interface",
        "interface_timing",
        "upstream_position_scope",
        "upstream_intervention_applied",
        "downstream_estimand",
        "orthogonal_seed",
        "generation_backend",
        "memory_implementation",
        "max_length",
    )
    signature = {
        key: metadata.get(key)
        for key in keys
        if metadata.get(key) is not None
    }
    signature["condition"] = condition
    signature["decoding_without_cap"] = decoding
    circuit_hashes = {
        signature.get("circuit_sha256"),
        signature.get("upstream_circuit_sha256"),
        signature.get("downstream_circuit_sha256"),
    } - {None}
    if not circuit_hashes:
        raise ValueError(
            "Pooling requires an explicit circuit hash in every source metadata file."
        )
    if signature.get("behavior_protocol_id") is None:
        raise ValueError("Pooling requires behavior_protocol_id provenance.")
    return signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        required=True,
        help="responses.jsonl files whose sibling metadata.json files describe generation.",
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--question-indices", type=int, nargs="+", required=True)
    parser.add_argument("--common-max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("Question indices must be unique.")
    if args.common_max_new_tokens <= 0:
        raise ValueError("--common-max-new-tokens must be positive.")

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        read_jsonl,
        sha256_file,
        write_jsonl,
    )

    source_records = []
    rows_by_question = {}
    compatibility = None
    wanted_questions = set(args.question_indices)
    for source_arg in args.sources:
        source = _resolve(source_arg)
        metadata_path = source.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise ValueError(f"Source metadata is not complete: {metadata_path}")
        decoding = metadata.get("decoding", {})
        if decoding.get("do_sample") is not False:
            raise ValueError("Only deterministic greedy artifacts may be pooled.")
        source_cap = int(decoding["max_new_tokens"])
        current_compatibility = _compatibility_signature(
            metadata, condition=args.condition
        )
        if compatibility is None:
            compatibility = current_compatibility
        elif current_compatibility != compatibility:
            raise ValueError(
                f"Generation sources are not intervention-compatible: {metadata_path}"
            )
        recorded_outputs = {
            item["path"]: item["sha256"]
            for item in metadata.get("outputs", [])
        }
        if recorded_outputs.get(_display(source)) != sha256_file(source):
            raise ValueError(
                f"Source responses hash differs from completed metadata: {source}"
            )

        accepted_rows = []
        for row in read_jsonl(source):
            if row.get("condition") != args.condition:
                continue
            question = int(row["question_index"])
            if question not in wanted_questions:
                continue
            if row.get("behavior_protocol_id") != compatibility[
                "behavior_protocol_id"
            ]:
                raise ValueError(f"q{question} protocol differs from source metadata.")
            count = int(row["generated_token_count"])
            if not row.get("generated_token_ids_sha256"):
                raise ValueError(
                    f"q{question} lacks an exact generated-token hash."
                )
            capped = bool(row.get("hit_max_new_tokens", False))
            if source_cap > args.common_max_new_tokens and (
                count >= args.common_max_new_tokens or capped
            ):
                raise ValueError(
                    f"q{question} from a larger-cap run did not terminate before "
                    "the common cap."
                )
            if source_cap < args.common_max_new_tokens and capped:
                raise ValueError(
                    f"q{question} was capped below the requested common cap."
                )
            if question in rows_by_question:
                raise ValueError(f"Duplicate pooled question index: {question}")
            pooled = {
                **row,
                "pooled_from": str(source.relative_to(ROOT)),
                "source_max_new_tokens": source_cap,
                "common_max_new_tokens": args.common_max_new_tokens,
            }
            rows_by_question[question] = pooled
            accepted_rows.append(question)
        source_records.append(
            {
                "responses": _display(source),
                "responses_sha256": sha256_file(source),
                "metadata": _display(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "source_max_new_tokens": source_cap,
                "accepted_question_indices": sorted(accepted_rows),
            }
        )

    expected = set(args.question_indices)
    if set(rows_by_question) != expected:
        missing = sorted(expected - set(rows_by_question))
        raise ValueError(f"Pooled row set is incomplete; missing {missing}.")
    rows = [rows_by_question[index] for index in args.question_indices]
    output = _resolve(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains output.")
    output.mkdir(parents=True, exist_ok=True)
    responses_path = output / "responses.jsonl"
    write_jsonl(responses_path, rows)
    write_json(
        output / "metadata.json",
        {
            "schema_version": 2,
            "status": "complete",
            "method": "deterministic split-cap pooling audit",
            "behavior_protocol_id": compatibility["behavior_protocol_id"],
            "model": compatibility.get("model"),
            "trait": compatibility.get("trait"),
            "condition": args.condition,
            "conditions": [args.condition],
            "question_indices": args.question_indices,
            "common_max_new_tokens": args.common_max_new_tokens,
            "decoding": {
                **compatibility["decoding_without_cap"],
                "do_sample": False,
                "max_new_tokens": args.common_max_new_tokens,
            },
            "responses": len(rows),
            "expected_responses": len(rows),
            "compatibility": compatibility,
            "cap_hit_count": sum(
                bool(row.get("hit_max_new_tokens", False)) for row in rows
            ),
            "sources": source_records,
            "outputs": [
                {
                    "path": _display(responses_path),
                    "rows": len(rows),
                    "sha256": sha256_file(responses_path),
                }
            ],
        },
    )
    print(output)


if __name__ == "__main__":
    main()

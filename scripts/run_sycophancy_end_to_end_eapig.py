#!/usr/bin/env python3
"""Score input-to-logits sycophancy edges with contrastive EAP-IG."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "EAP-IG")]

from refusal_circuit.eapig import (
    AttributionConfig,
    attribute_contrastive_prompt_response_logprob,
    attribute_paired_coordinate_response_logprob,
)
from refusal_circuit.graph import make_logit_graph, write_edge_scores_parquet
from refusal_circuit.model import (
    format_system_instruction,
    install_memory_efficient_split_qkv_projection,
    load_model,
    validate_memory_efficient_split_qkv_projection,
)
from refusal_circuit.persona import make_system_prompt, read_jsonl, sha256_file
from refusal_circuit.paths import write_json
from refusal_circuit.workflow import _load_vector, load_config


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_instruction_pairs(path: Path) -> dict[int, tuple[str, str]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    pairs = {
        int(record["instruction_pair_index"]): (
            record["positive_instruction"],
            record["negative_instruction"],
        )
        for record in records
    }
    if sorted(pairs) != list(range(5)):
        raise ValueError("Expected exactly instruction-pair indices 0 through 4.")
    return pairs


def formatted_prompt(tokenizer, row: dict, instruction: str) -> str:
    return format_system_instruction(
        tokenizer,
        make_system_prompt("sycophantic", instruction),
        row["prompt"],
    )


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_transcript_pairs(
    tokenizer,
    rows: list[dict],
    instruction_pairs: dict[int, tuple[str, str]],
    *,
    max_length: int,
) -> tuple[list, int]:
    transcripts = []
    observed_maximum = 0
    for row in rows:
        pair_index = int(row["instruction_pair_index"])
        positive_instruction, negative_instruction = instruction_pairs[pair_index]
        clean_prompt = formatted_prompt(tokenizer, row, positive_instruction)
        corrupt_prompt = formatted_prompt(tokenizer, row, negative_instruction)
        clean_prompt_ids = token_ids(tokenizer, clean_prompt)
        corrupt_prompt_ids = token_ids(tokenizer, corrupt_prompt)
        if len(clean_prompt_ids) != len(corrupt_prompt_ids):
            raise ValueError(
                f"Prompt alignment failed for {row['id']}: "
                f"{len(clean_prompt_ids)} != {len(corrupt_prompt_ids)}."
            )
        response_start = len(clean_prompt_ids)
        answer_specs = []
        for field in ("sycophantic_answer", "normal_answer"):
            answer = row[field]
            clean_text = clean_prompt + answer
            corrupt_text = corrupt_prompt + answer
            clean_ids = token_ids(tokenizer, clean_text)
            corrupt_ids = token_ids(tokenizer, corrupt_text)
            if len(clean_ids) != len(corrupt_ids):
                raise ValueError(
                    f"Transcript alignment failed for {row['id']} field {field}."
                )
            if clean_ids[response_start:] != corrupt_ids[response_start:]:
                raise ValueError(
                    f"Answer tokens differ across prompt conditions for "
                    f"{row['id']} field {field}."
                )
            observed_maximum = max(observed_maximum, len(clean_ids))
            if len(clean_ids) > max_length:
                raise ValueError(
                    f"{row['id']} field {field} has {len(clean_ids)} tokens, "
                    f"exceeding max_length={max_length}."
                )
            answer_specs.append((clean_text, corrupt_text, response_start))
        transcripts.append(tuple(answer_specs))
    return transcripts, observed_maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/sycophancy_prompt.json")
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/processed/qwen2.5-7b-instruct/trait_circuits/sycophancy/"
            "eapig.jsonl"
        ),
    )
    parser.add_argument(
        "--instruction-pairs",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/sycophancy/end_to_end/"
            "instruction_contrast_n16/instruction_pairs.json"
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--corruption",
        choices=("natural_prompt", "paired_coordinate"),
        default="natural_prompt",
    )
    parser.add_argument(
        "--prompt-vector-config",
        type=Path,
        default=Path(
            "configs/sycophancy_prompt_paired_coordinate_eapig_a100.json"
        ),
    )
    parser.add_argument(
        "--generation-vector-config",
        type=Path,
        default=Path(
            "configs/sycophancy_generation_paired_coordinate_eapig.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/sycophancy/end_to_end/"
            "eapig_n64_ig5"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    dataset_path = resolve(args.dataset)
    instruction_path = resolve(args.instruction_pairs)
    output_dir = resolve(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} already exists; pass --overwrite.")

    config = load_config(config_path)
    rows = read_jsonl(dataset_path)[: args.num_prompts]
    if len(rows) != args.num_prompts:
        raise ValueError(f"Expected {args.num_prompts} rows, found {len(rows)}.")
    instruction_pairs = load_instruction_pairs(instruction_path)

    if args.validate_only:
        from transformers import AutoTokenizer

        model_config = config["model"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"],
            revision=model_config.get("revision"),
            local_files_only=True,
        )
        _, observed_maximum = build_transcript_pairs(
            tokenizer,
            rows,
            instruction_pairs,
            max_length=args.max_length,
        )
        print(
            json.dumps(
                {
                    "num_prompts": len(rows),
                    "max_transcript_tokens": observed_maximum,
                    "max_length": args.max_length,
                    "alignment_valid": True,
                },
                indent=2,
            )
        )
        return

    install_memory_efficient_split_qkv_projection()
    model_config = config["model"]
    model = load_model(
        model_config["name"],
        revision=model_config.get("revision"),
        dtype=model_config.get("dtype", "bfloat16"),
        device=model_config.get("device"),
        hook_mode="eap",
    )
    validate_memory_efficient_split_qkv_projection(model)
    transcripts, observed_maximum = build_transcript_pairs(
        model.tokenizer,
        rows,
        instruction_pairs,
        max_length=args.max_length,
    )
    graph = make_logit_graph(model)
    started = time.monotonic()
    attribution_config = AttributionConfig(
        batch_size=1,
        max_length=args.max_length,
        ig_steps=args.ig_steps,
    )
    vector_metadata = None
    if args.corruption == "paired_coordinate":
        prompt_config = load_config(resolve(args.prompt_vector_config))
        generation_config = load_config(resolve(args.generation_vector_config))
        prompt_direction, _, prompt_metadata = _load_vector(
            prompt_config, model
        )
        generation_direction, _, generation_metadata = _load_vector(
            generation_config, model
        )
        diagnostics = attribute_paired_coordinate_response_logprob(
            model,
            graph,
            transcripts,
            prompt_direction=prompt_direction,
            generation_direction=generation_direction,
            config=attribution_config,
        )
        vector_metadata = {
            "prompt": {
                "config": str(resolve(args.prompt_vector_config)),
                "layer": prompt_metadata["layer"],
                "position": prompt_metadata["position"],
            },
            "generation": {
                "config": str(resolve(args.generation_vector_config)),
                "layer": generation_metadata["layer"],
            },
        }
    else:
        diagnostics = attribute_contrastive_prompt_response_logprob(
            model,
            graph,
            transcripts,
            config=attribution_config,
        )
    elapsed_seconds = time.monotonic() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "edge_scores.parquet"
    write_edge_scores_parquet(graph, scores_path)
    result = {
        "status": "complete",
        "method": f"{args.corruption}_contrastive_response_logprob_eapig",
        "model": {**model_config, "hook_mode": "eap"},
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "num_prompts": len(rows),
        },
        "instruction_pairs": {
            "path": str(instruction_path),
            "sha256": sha256_file(instruction_path),
            "count": len(instruction_pairs),
        },
        "graph": {
            "source": graph.nodes["input"].out_hook,
            "target": "response logits",
            "edges": len(graph.edges),
        },
        "target": (
            "[mean logP(sycophantic answer)-mean logP(normal answer)] under "
            "the positive instruction; clean minus the selected corruption"
        ),
        "corruption": args.corruption,
        "vectors": vector_metadata,
        "ig_steps": args.ig_steps,
        "max_length": args.max_length,
        "max_observed_transcript_tokens": observed_maximum,
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "diagnostics": diagnostics,
        "edge_scores": str(scores_path),
    }
    write_json(metadata_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

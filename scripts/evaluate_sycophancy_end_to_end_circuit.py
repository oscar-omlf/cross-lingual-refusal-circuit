#!/usr/bin/env python3
"""Evaluate one connected sycophancy input-to-logits circuit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "EAP-IG"), str(ROOT / "scripts")]

from refusal_circuit.circuit import (
    mixed_logits_from_prompt_pair,
    mixed_logits_with_intervention,
    normalized_faithfulness,
    normalized_necessity,
)
from refusal_circuit.eapig import (
    _response_mask,
    prepare_paired_coordinate_source_corruption,
    response_logprob_values,
    response_state_mask,
)
from refusal_circuit.graph import (
    circuit_connectivity,
    load_circuit_edge_names,
    make_logit_graph,
    set_selected_edges,
    validate_intervention_connectivity,
)
from refusal_circuit.model import (
    install_memory_efficient_split_qkv_projection,
    load_model,
    tokenize_prompts,
    validate_memory_efficient_split_qkv_projection,
)
from refusal_circuit.persona import read_jsonl, sha256_file, write_jsonl
from refusal_circuit.paths import write_json
from refusal_circuit.workflow import _load_vector, load_config
from run_sycophancy_end_to_end_eapig import (
    build_transcript_pairs,
    load_instruction_pairs,
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def score_spec(
    model,
    graph,
    spec: tuple[str, str, int],
    *,
    corruption: str,
    prompt_direction: torch.Tensor | None,
    generation_direction: torch.Tensor | None,
) -> dict[str, float]:
    clean_text, corrupt_text, response_start = spec
    clean_tokens, clean_mask = tokenize_prompts(model, [clean_text], max_length=512)
    corrupt_tokens, corrupt_mask = tokenize_prompts(
        model, [corrupt_text], max_length=512
    )
    if tuple(clean_tokens.shape) != tuple(corrupt_tokens.shape):
        raise ValueError("Clean and corrupt transcript shapes differ.")
    if not torch.equal(clean_mask, corrupt_mask):
        raise ValueError("Clean and corrupt transcript masks differ.")
    response_mask = _response_mask(clean_tokens, clean_mask, [response_start])
    if corruption == "paired_coordinate":
        if prompt_direction is None or generation_direction is None:
            raise ValueError("Paired-coordinate evaluation requires both vectors.")
        prompt_mask = clean_mask.bool() & ~response_mask
        generation_mask = response_state_mask(
            response_mask, include_terminal=False
        )
        source_transform, corrupt_hooks, _ = (
            prepare_paired_coordinate_source_corruption(
                model,
                graph,
                corrupt_tokens,
                corrupt_mask,
                [
                    (prompt_direction, prompt_mask),
                    (generation_direction, generation_mask),
                ],
            )
        )
        with torch.inference_mode():
            clean_logits = model(clean_tokens, attention_mask=clean_mask)
            with model.hooks(fwd_hooks=corrupt_hooks):
                corrupt_logits = model(
                    clean_tokens, attention_mask=clean_mask
                )
        logits = {
            "clean": clean_logits,
            "corrupt": corrupt_logits,
            "circuit_only": mixed_logits_with_intervention(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_fwd_hooks=corrupt_hooks,
                corrupt_cache_transform=source_transform,
                patch_selected=False,
                cache_dtype=torch.bfloat16,
            ),
            "circuit_ablated": mixed_logits_with_intervention(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_fwd_hooks=corrupt_hooks,
                corrupt_cache_transform=source_transform,
                patch_selected=True,
                cache_dtype=torch.bfloat16,
            ),
        }
    else:
        with torch.inference_mode():
            clean_logits = model(clean_tokens, attention_mask=clean_mask)
            corrupt_logits = model(corrupt_tokens, attention_mask=corrupt_mask)
        logits = {
            "clean": clean_logits,
            "corrupt": corrupt_logits,
            "circuit_only": mixed_logits_from_prompt_pair(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_tokens,
                corrupt_mask,
                patch_selected=False,
                cache_dtype=torch.bfloat16,
            ),
            "circuit_ablated": mixed_logits_from_prompt_pair(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_tokens,
                corrupt_mask,
                patch_selected=True,
                cache_dtype=torch.bfloat16,
            ),
        }
    return {
        name: float(
            response_logprob_values(value, clean_tokens, response_mask)[0].item()
        )
        for name, value in logits.items()
    }


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
            "circuit_selection.jsonl"
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
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    dataset_path = resolve(args.dataset)
    instruction_path = resolve(args.instruction_pairs)
    circuit_path = resolve(args.circuit)
    output_dir = resolve(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite.")

    rows = read_jsonl(dataset_path)[: args.num_prompts]
    if len(rows) != args.num_prompts:
        raise ValueError(f"Expected {args.num_prompts} rows, found {len(rows)}.")
    instruction_pairs = load_instruction_pairs(instruction_path)
    config = load_config(config_path)
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
    prompt_direction = None
    generation_direction = None
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
    transcripts, maximum_length = build_transcript_pairs(
        model.tokenizer,
        rows,
        instruction_pairs,
        max_length=512,
    )
    graph = make_logit_graph(model)
    edge_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, edge_names)
    connectivity = circuit_connectivity(graph, edge_names)
    validate_intervention_connectivity(
        {"circuit": connectivity}, require_source_to_sink=True
    )

    started = time.monotonic()
    records = []
    for index, (row, (positive_spec, negative_spec)) in enumerate(
        zip(rows, transcripts, strict=True), start=1
    ):
        positive = score_spec(
            model,
            graph,
            positive_spec,
            corruption=args.corruption,
            prompt_direction=prompt_direction,
            generation_direction=generation_direction,
        )
        negative = score_spec(
            model,
            graph,
            negative_spec,
            corruption=args.corruption,
            prompt_direction=prompt_direction,
            generation_direction=generation_direction,
        )
        margins = {
            condition: positive[condition] - negative[condition]
            for condition in positive
        }
        records.append(
            {
                "id": row["id"],
                "source_index": row["source_index"],
                "instruction_pair_index": row["instruction_pair_index"],
                **{f"{name}_margin": value for name, value in margins.items()},
            }
        )
        print(
            json.dumps(
                {
                    "completed_prompts": index,
                    "total_prompts": len(rows),
                }
            ),
            flush=True,
        )

    tensors = {
        name: torch.tensor([record[f"{name}_margin"] for record in records])
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    clean_minus_corrupt = tensors["clean"].mean() - tensors["corrupt"].mean()
    if clean_minus_corrupt <= 0:
        raise RuntimeError("Held-out clean/corrupt contrast is not positive.")
    circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
    result = {
        "status": "complete",
        "model": {**model_config, "hook_mode": "eap"},
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "num_prompts": len(rows),
        },
        "instruction_pairs": {
            "path": str(instruction_path),
            "sha256": sha256_file(instruction_path),
        },
        "circuit": {
            "path": str(circuit_path),
            "sha256": sha256_file(circuit_path),
            "requested_k": circuit["requested_k"],
            "retained_edges": circuit["retained_edges"],
            "connectivity": connectivity,
        },
        "target": (
            "[mean logP(sycophantic answer)-mean logP(normal answer)] under "
            "the positive instruction; clean minus the selected corruption"
        ),
        "corruption": args.corruption,
        "vectors": vector_metadata,
        "metrics": {
            "clean": tensors["clean"].mean().item(),
            "corrupt": tensors["corrupt"].mean().item(),
            "clean_minus_corrupt": clean_minus_corrupt.item(),
            "circuit_only": tensors["circuit_only"].mean().item(),
            "circuit_ablated": tensors["circuit_ablated"].mean().item(),
            "faithfulness": normalized_faithfulness(
                tensors["clean"], tensors["corrupt"], tensors["circuit_only"]
            ),
            "necessity": normalized_necessity(
                tensors["clean"], tensors["corrupt"], tensors["circuit_ablated"]
            ),
        },
        "max_observed_transcript_tokens": maximum_length,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_prompt.jsonl", records)
    write_json(summary_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the held-out causal screen for sycophancy prompt-vector locations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusal_circuit.persona import make_system_prompt, read_jsonl, sha256_file, write_jsonl
from refusal_circuit.paths import write_json


DEFAULT_CONFIG = ROOT / "configs/sycophancy_prompt.json"
DEFAULT_DATASET = (
    ROOT
    / "data/processed/qwen2.5-7b-instruct/trait_circuits/sycophancy"
    / "vector_validation.jsonl"
)
DEFAULT_MINUS1_ROOT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/prompt/vector_candidates"
)
DEFAULT_MINUS2_ROOT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/prompt"
    / "vector_candidates_position_minus2"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/prompt"
    / "location_screen_coefficient_2"
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def batches(values: list, batch_size: int) -> list[list]:
    return [
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    ]


def format_prompts(tokenizer, rows: list[dict], polarity: str) -> list[str]:
    field = {
        "positive": "positive_system_instruction",
        "negative": "negative_system_instruction",
    }[polarity]
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": make_system_prompt("sycophantic", row[field]),
                },
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]


def recurrent_input_hook(vector: torch.Tensor, coefficient: float, prompt_position: int):
    """Steer resid_pre at the extraction token and every cached decode step."""

    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise ValueError("Expected hidden states as the first layer input.")
        hidden = inputs[0]
        direction = vector.to(device=hidden.device, dtype=hidden.dtype)
        if direction.ndim != 1 or direction.numel() != hidden.shape[-1]:
            raise ValueError("Prompt vector does not match the hidden dimension.")
        position = prompt_position if hidden.shape[1] > 1 else -1
        if abs(position) > hidden.shape[1]:
            raise ValueError(
                f"Position {position} is outside sequence length {hidden.shape[1]}."
            )
        changed = hidden.clone()
        changed[:, position, :] += float(coefficient) * direction
        return (changed, *inputs[1:])

    return hook


def generated_token_count(token_ids: list[int], eos_token_id: int | None) -> int:
    if eos_token_id is None or eos_token_id not in token_ids:
        return len(token_ids)
    return token_ids.index(eos_token_id) + 1


def generate_rows(
    *,
    model,
    tokenizer,
    source_rows: list[dict],
    formatted_prompts: list[str],
    condition: str,
    batch_size: int,
    max_new_tokens: int,
    layer: int | None = None,
    position: int | None = None,
    vector: torch.Tensor | None = None,
    coefficient: float = 0.0,
) -> list[dict]:
    output = []
    source_batches = batches(source_rows, batch_size)
    prompt_batches = batches(formatted_prompts, batch_size)
    for source_batch, prompt_batch in zip(source_batches, prompt_batches, strict=True):
        inputs = tokenizer(
            prompt_batch,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        inputs = {name: value.to(model.device) for name, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[1])
        handle = None
        if vector is not None:
            if layer is None or position is None:
                raise ValueError("Steered generation requires layer and position.")
            handle = model.model.layers[layer].register_forward_pre_hook(
                recurrent_input_hook(vector, coefficient, position)
            )
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            if handle is not None:
                handle.remove()
        generated = generated.detach().cpu()
        for source, token_row in zip(source_batch, generated, strict=True):
            continuation = token_row[prompt_length:].tolist()
            count = generated_token_count(continuation, tokenizer.eos_token_id)
            continuation = continuation[:count]
            output.append(
                {
                    "record_id": (
                        f"{condition}-"
                        f"p{position if position is not None else 0}-"
                        f"l{layer if layer is not None else 0}-"
                        f"{source['id']}"
                    ),
                    "id": source["id"],
                    "source_index": source["source_index"],
                    "instruction_pair_index": source["instruction_pair_index"],
                    "question": source["prompt"],
                    "condition": condition,
                    "layer": layer,
                    "position": position,
                    "coefficient": coefficient,
                    "answer": tokenizer.decode(
                        continuation, skip_special_tokens=True
                    ).strip(),
                    "generated_token_count": count,
                    "hit_max_new_tokens": count >= max_new_tokens,
                }
            )
    return output


def load_candidates(root: Path, expected_position: int) -> tuple[dict, torch.Tensor]:
    metadata_path = root / "metadata.json"
    directions_path = root / "raw_directions.pt"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["position"]) != expected_position:
        raise ValueError(
            f"{root} has position {metadata['position']}, expected {expected_position}."
        )
    directions = torch.load(
        directions_path, map_location="cpu", weights_only=True
    ).float()
    if directions.ndim != 2 or directions.shape[0] != len(metadata["layers"]):
        raise ValueError(f"Unexpected direction tensor shape at {root}.")
    return metadata, directions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--minus1-root", type=Path, default=DEFAULT_MINUS1_ROOT)
    parser.add_argument("--minus2-root", type=Path, default=DEFAULT_MINUS2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-questions", type=int, default=16)
    parser.add_argument("--coefficient", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_questions <= 0 or args.coefficient <= 0 or args.batch_size <= 0:
        raise ValueError("Question count, coefficient, and batch size must be positive.")

    config_path = resolve(args.config)
    dataset_path = resolve(args.dataset)
    output_dir = resolve(args.output_dir)
    responses_path = output_dir / "responses.jsonl"
    metadata_path = output_dir / "metadata.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite.")
        responses_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = read_jsonl(dataset_path)[: args.num_questions]
    if len(rows) != args.num_questions:
        raise ValueError(
            f"Requested {args.num_questions} validation rows, found {len(rows)}."
        )
    pair_counts = Counter(int(row["instruction_pair_index"]) for row in rows)
    if max(pair_counts.values()) - min(pair_counts.values()) > 1:
        raise ValueError(f"Instruction pairs are not balanced: {dict(pair_counts)}")

    candidate_sets = {}
    for position, root in (
        (-1, resolve(args.minus1_root)),
        (-2, resolve(args.minus2_root)),
    ):
        metadata, directions = load_candidates(root, position)
        candidate_sets[position] = {
            "root": root,
            "metadata": metadata,
            "directions": directions,
            "by_layer": {
                int(layer): directions[index]
                for index, layer in enumerate(metadata["layers"])
            },
            "valid_layers": {int(layer) for layer in metadata["valid_layers"]},
        }

    valid_layers = sorted(
        candidate_sets[-1]["valid_layers"]
        & candidate_sets[-2]["valid_layers"]
    )
    if valid_layers != list(range(1, 28)):
        raise ValueError(
            f"Expected common valid layers 1-27, found {valid_layers}."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg.get("revision")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        torch_dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        low_cpu_mem_usage=True,
    ).to("cuda").eval()

    positive_prompts = format_prompts(tokenizer, rows, "positive")
    negative_prompts = format_prompts(tokenizer, rows, "negative")

    baseline_rows = []
    baseline_rows.extend(
        generate_rows(
            model=model,
            tokenizer=tokenizer,
            source_rows=rows,
            formatted_prompts=negative_prompts,
            condition="baseline_negative",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    )
    baseline_rows.extend(
        generate_rows(
            model=model,
            tokenizer=tokenizer,
            source_rows=rows,
            formatted_prompts=positive_prompts,
            condition="baseline_positive",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    )
    write_jsonl(responses_path, baseline_rows)

    for position in (-1, -2):
        vectors = candidate_sets[position]["by_layer"]
        for layer in valid_layers:
            print(
                json.dumps(
                    {
                        "status": "generating",
                        "position": position,
                        "layer": layer,
                    }
                ),
                flush=True,
            )
            vector = vectors[layer]
            induction = generate_rows(
                model=model,
                tokenizer=tokenizer,
                source_rows=rows,
                formatted_prompts=negative_prompts,
                condition="induction",
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                layer=layer,
                position=position,
                vector=vector,
                coefficient=args.coefficient,
            )
            ablation = generate_rows(
                model=model,
                tokenizer=tokenizer,
                source_rows=rows,
                formatted_prompts=positive_prompts,
                condition="ablation",
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                layer=layer,
                position=position,
                vector=vector,
                coefficient=-args.coefficient,
            )
            write_jsonl(responses_path, [*induction, *ablation], mode="a")

    generated_rows = read_jsonl(responses_path)
    expected_rows = 2 * args.num_questions + (
        2 * len(valid_layers) * 2 * args.num_questions
    )
    if len(generated_rows) != expected_rows:
        raise RuntimeError(
            f"Incomplete response set: {len(generated_rows)}/{expected_rows}."
        )

    metadata = {
        "schema_version": 1,
        "status": "generation_complete_pending_persona_judge",
        "purpose": (
            "Causal layer and prompt-suffix-position selection for the "
            "sycophancy paired-prompt vector."
        ),
        "model": model_cfg,
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "questions": len(rows),
            "instruction_pair_counts": dict(sorted(pair_counts.items())),
        },
        "candidate_positions": [-1, -2],
        "candidate_layers": valid_layers,
        "excluded_zero_vector_layers": [0],
        "conditions": {
            "baseline_negative": "anti-sycophantic system instruction, no steering",
            "baseline_positive": "sycophantic system instruction, no steering",
            "induction": (
                f"anti-sycophantic system instruction, +{args.coefficient:g} "
                "times the matched raw prompt vector"
            ),
            "ablation": (
                f"sycophantic system instruction, -{args.coefficient:g} "
                "times the matched raw prompt vector"
            ),
        },
        "intervention": {
            "coefficient_magnitude": args.coefficient,
            "timing": (
                "extraction suffix position on prompt prefill, then every "
                "cached decode step"
            ),
            "site": "HF transformer block input / TransformerLens hook_resid_pre",
        },
        "generation": {
            "deterministic": True,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "expected_responses": expected_rows,
        },
        "judge": {
            "status": "pending",
            "required_protocol": (
                "Persona Vectors sycophancy 0-100 judge and coherence 0-100 judge"
            ),
            "reason": "The HPC environment has no OpenAI API credential.",
        },
        "candidate_artifacts": {
            str(position): {
                "root": str(values["root"]),
                "metadata_sha256": sha256_file(values["root"] / "metadata.json"),
                "raw_directions_sha256": sha256_file(
                    values["root"] / "raw_directions.pt"
                ),
            }
            for position, values in candidate_sets.items()
        },
        "artifacts": {
            "responses": {
                "path": str(responses_path),
                "rows": len(generated_rows),
                "sha256": sha256_file(responses_path),
            }
        },
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

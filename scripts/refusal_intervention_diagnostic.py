#!/usr/bin/env python3
"""Compare global refusal-direction projection with negative steering."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "EAP-IG")]

from refusal_circuit.interventions import addition_hooks, all_layer_ablation_hooks
from refusal_circuit.metrics import (
    ProjectionCapture,
    is_refusal,
    mean_last_token_kl,
    refusal_log_odds,
)
from refusal_circuit.model import (
    format_instructions,
    load_model,
    tokenize_prompts,
)
from refusal_circuit.paths import write_json


DEFAULT_DATASET = (
    ROOT
    / "data/processed/qwen2.5-7b-instruct/trait_circuits/refusal"
    / "vector_validation.jsonl"
)
DEFAULT_VECTOR_ROOT = (
    ROOT
    / "old_results/qwen2.5-7b-instruct/polyrefuse/en"
    / "direction_paper_kl0p2"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/qwen2.5-7b-instruct/refusal"
    / "projection_vs_negative_steering"
)
T = TypeVar("T")


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


def batches(values: list[T], batch_size: int) -> list[list[T]]:
    return [
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    ]


def generated_token_count(token_ids: list[int], eos_token_id: int | None) -> int:
    if eos_token_id is None or eos_token_id not in token_ids:
        return len(token_ids)
    return token_ids.index(eos_token_id) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vector-root", type=Path, default=DEFAULT_VECTOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-harmful-generations", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "forward_metrics.jsonl"
    generations_path = output_dir / "generations.jsonl"
    if any(path.exists() for path in (summary_path, metrics_path, generations_path)):
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already contains diagnostic output; pass --overwrite."
            )

    if args.batch_size <= 0 or args.num_harmful_generations <= 0:
        raise ValueError("Batch size and generation count must be positive.")

    rows = read_jsonl(args.dataset.resolve())
    harmful_rows = [row for row in rows if row["label"] == "harmful"]
    harmless_rows = [row for row in rows if row["label"] == "harmless"]
    if len(harmful_rows) != 32 or len(harmless_rows) != 32:
        raise ValueError(
            "Expected the frozen 64-prompt validation split "
            f"(32 harmful, 32 harmless), got {len(harmful_rows)} and "
            f"{len(harmless_rows)}."
        )
    if args.num_harmful_generations > len(harmful_rows):
        raise ValueError("Requested more harmful generations than available prompts.")

    metadata = json.loads(
        (args.vector_root.resolve() / "refusal_direction_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    direction = torch.load(
        args.vector_root.resolve() / "refusal_direction.pt",
        map_location="cpu",
        weights_only=True,
    ).float().reshape(-1)
    direction = direction / direction.norm().clamp_min(1e-8)
    raw_direction = direction * float(metadata["raw_direction_norm"])
    layer = int(metadata["layer"])
    position = int(metadata["position"])
    target_hook = str(metadata["target_hook"])
    refusal_token_ids = [int(value) for value in metadata["refusal_token_ids"]]

    model_spec = {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "dtype": "bfloat16",
    }
    model = load_model(
        model_spec["name"],
        revision=model_spec["revision"],
        dtype=model_spec["dtype"],
        hook_mode="standard",
    )
    formatted = format_instructions(
        model.tokenizer, [row["instruction"] for row in rows]
    )

    hook_factories: dict[str, Callable[[], list[tuple[str, Callable]]]] = {
        "baseline": lambda: [],
        "global_projection": lambda: all_layer_ablation_hooks(model, direction),
        "negative_steering_1": lambda: addition_hooks(
            layer, raw_direction, coefficient=-1.0, position=None
        ),
        "negative_steering_2": lambda: addition_hooks(
            layer, raw_direction, coefficient=-2.0, position=None
        ),
    }

    baseline_logits: list[torch.Tensor] = []
    condition_values: dict[str, dict[str, list[torch.Tensor]]] = {}
    forward_rows: list[dict] = []
    for condition, make_hooks in hook_factories.items():
        condition_logits: list[torch.Tensor] = []
        condition_scores: list[torch.Tensor] = []
        condition_projections: list[torch.Tensor] = []
        for prompt_batch in batches(formatted, args.batch_size):
            tokens, attention_mask = tokenize_prompts(
                model, prompt_batch, max_length=args.max_length
            )
            capture = ProjectionCapture(direction=direction, position=position)
            hooks = [*make_hooks(), (target_hook, capture.hook)]
            with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
                logits = model(tokens, attention_mask=attention_mask)
            final_logits = logits[:, -1].detach().float().cpu()
            condition_logits.append(final_logits)
            condition_scores.append(
                refusal_log_odds(
                    final_logits.unsqueeze(1), refusal_token_ids
                ).float()
            )
            condition_projections.append(capture.values().detach().float().cpu())

        joined_logits = torch.cat(condition_logits)
        joined_scores = torch.cat(condition_scores)
        joined_projections = torch.cat(condition_projections)
        if condition == "baseline":
            baseline_logits = [joined_logits]
        condition_values[condition] = {
            "logits": [joined_logits],
            "scores": [joined_scores],
            "projections": [joined_projections],
        }

    clean_logits = baseline_logits[0]
    for condition in hook_factories:
        values = condition_values[condition]
        scores = values["scores"][0]
        projections = values["projections"][0]
        changed_logits = values["logits"][0]
        for index, row in enumerate(rows):
            kl = (
                0.0
                if condition == "baseline"
                else float(
                    mean_last_token_kl(
                        clean_logits[index : index + 1].unsqueeze(1),
                        changed_logits[index : index + 1].unsqueeze(1),
                    ).item()
                )
            )
            forward_rows.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "condition": condition,
                    "refusal_log_odds": float(scores[index].item()),
                    "target_projection": float(projections[index].item()),
                    "kl_from_baseline": kl,
                }
            )

    generation_rows: list[dict] = []
    generation_sources = harmful_rows[: args.num_harmful_generations]
    generation_prompts = format_instructions(
        model.tokenizer, [row["instruction"] for row in generation_sources]
    )
    for condition, make_hooks in hook_factories.items():
        source_offset = 0
        for prompt_batch in batches(generation_prompts, args.batch_size):
            source_batch = generation_sources[
                source_offset : source_offset + len(prompt_batch)
            ]
            source_offset += len(prompt_batch)
            input_tokens = model.to_tokens(
                prompt_batch,
                prepend_bos=False,
                padding_side="left",
            )
            with model.hooks(fwd_hooks=make_hooks()):
                generated = model.generate(
                    prompt_batch,
                    max_new_tokens=args.max_new_tokens,
                    stop_at_eos=True,
                    do_sample=False,
                    use_past_kv_cache=True,
                    prepend_bos=False,
                    padding_side="left",
                    return_type="tokens",
                    verbose=False,
                )
            generated = generated.detach().cpu()
            input_length = int(input_tokens.shape[1])
            for source, token_row in zip(
                source_batch, generated, strict=True
            ):
                continuation = token_row[input_length:].tolist()
                count = generated_token_count(
                    continuation, model.tokenizer.eos_token_id
                )
                continuation = continuation[:count]
                answer = model.tokenizer.decode(
                    continuation, skip_special_tokens=True
                ).strip()
                generation_rows.append(
                    {
                        "id": source["id"],
                        "condition": condition,
                        "answer": answer,
                        "generated_token_count": count,
                        "refusal_prefix_match": is_refusal(answer),
                    }
                )

    summaries = {}
    for condition in hook_factories:
        condition_forward = [
            row for row in forward_rows if row["condition"] == condition
        ]
        harmful_forward = [
            row for row in condition_forward if row["label"] == "harmful"
        ]
        harmless_forward = [
            row for row in condition_forward if row["label"] == "harmless"
        ]
        condition_generations = [
            row for row in generation_rows if row["condition"] == condition
        ]
        summaries[condition] = {
            "harmful_mean_refusal_log_odds": sum(
                row["refusal_log_odds"] for row in harmful_forward
            )
            / len(harmful_forward),
            "harmful_positive_refusal_log_odds_rate": sum(
                row["refusal_log_odds"] > 0 for row in harmful_forward
            )
            / len(harmful_forward),
            "harmful_mean_target_projection": sum(
                row["target_projection"] for row in harmful_forward
            )
            / len(harmful_forward),
            "harmless_mean_refusal_log_odds": sum(
                row["refusal_log_odds"] for row in harmless_forward
            )
            / len(harmless_forward),
            "harmless_mean_target_projection": sum(
                row["target_projection"] for row in harmless_forward
            )
            / len(harmless_forward),
            "harmless_mean_kl_from_baseline": sum(
                row["kl_from_baseline"] for row in harmless_forward
            )
            / len(harmless_forward),
            "generated_refusal_prefix_rate": sum(
                row["refusal_prefix_match"] for row in condition_generations
            )
            / len(condition_generations),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(metrics_path, forward_rows)
    write_jsonl(generations_path, generation_rows)
    summary = {
        "status": "complete",
        "model": model_spec,
        "dataset": str(args.dataset.resolve()),
        "validation_counts": {"harmful": 32, "harmless": 32},
        "generated_harmful_prompts": len(generation_sources),
        "max_new_tokens": args.max_new_tokens,
        "vector": {
            "source": str(args.vector_root.resolve()),
            "layer": layer,
            "position": position,
            "target_hook": target_hook,
            "raw_norm": float(raw_direction.norm().item()),
        },
        "conditions": {
            "baseline": "no intervention",
            "global_projection": (
                "unit-direction projection at every residual input, complete "
                "attention write, and MLP write in every layer and position"
            ),
            "negative_steering_1": (
                "add -1 times the raw DIM vector at every position of the "
                f"selected layer {layer}"
            ),
            "negative_steering_2": (
                "add -2 times the raw DIM vector at every position of the "
                f"selected layer {layer}"
            ),
        },
        "results": summaries,
        "artifacts": {
            "forward_metrics": str(metrics_path),
            "generations": str(generations_path),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

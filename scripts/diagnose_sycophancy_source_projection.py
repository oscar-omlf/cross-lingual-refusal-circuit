#!/usr/bin/env python3
"""Compare natural-scale source-wise corruptions for generation sycophancy."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from refusal_circuit.eapig import _response_mask, response_logprob_values
from refusal_circuit.model import load_model, tokenize_prompts
from refusal_circuit.persona import read_jsonl, write_jsonl
from refusal_circuit.paths import write_json
from refusal_circuit.sycophancy_geometry import (
    direction_coordinates,
    masked_mean_states,
    response_region_masks,
    scalar_summary,
)
from refusal_circuit.workflow import _load_vector, load_config
from run_sycophancy_generation_geometry import (
    DEFAULT_CLAMP_CALIBRATION_DATASET,
    DEFAULT_DATASET,
    build_transcript_specs,
    resolve,
    validate_rows,
)


def source_names(model) -> list[str]:
    return [
        "blocks.0.hook_resid_pre",
        *[
            f"blocks.{layer}.{suffix}"
            for layer in range(int(model.cfg.n_layers))
            for suffix in ("hook_attn_out", "hook_mlp_out")
        ],
    ]


def monitor_names(model) -> list[str]:
    return [
        f"blocks.{layer}.hook_resid_pre"
        for layer in range(int(model.cfg.n_layers))
    ]


def unit(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (
        direction.to(device=reference.device, dtype=reference.dtype)
        / direction.norm().clamp_min(1e-8).to(reference)
    )


def scalar_projection_hook(
    direction: torch.Tensor,
    *,
    target: float,
    mask: torch.Tensor,
    strength: float = 1.0,
):
    def hook(activations: torch.Tensor, hook):
        del hook
        vector = unit(direction, activations)
        coordinate = torch.einsum("bpd,d->bp", activations, vector)
        delta = float(strength) * (float(target) - coordinate)
        return activations + (
            mask.to(device=activations.device, dtype=activations.dtype).unsqueeze(-1)
            * delta.to(activations).unsqueeze(-1)
            * vector
        )

    return hook


def tensor_projection_hook(
    direction: torch.Tensor,
    *,
    target: torch.Tensor,
    mask: torch.Tensor,
    strength: float,
):
    def hook(activations: torch.Tensor, hook):
        del hook
        vector = unit(direction, activations)
        coordinate = torch.einsum("bpd,d->bp", activations, vector)
        desired = target.to(device=activations.device, dtype=coordinate.dtype)
        if tuple(desired.shape) != tuple(coordinate.shape):
            raise ValueError(
                f"Target shape {tuple(desired.shape)} != {tuple(coordinate.shape)}"
            )
        delta = float(strength) * (desired - coordinate)
        return activations + (
            mask.to(device=activations.device, dtype=activations.dtype).unsqueeze(-1)
            * delta.to(activations).unsqueeze(-1)
            * vector
        )

    return hook


def translation_hook(
    direction: torch.Tensor,
    *,
    shift: float,
    mask: torch.Tensor,
):
    def hook(activations: torch.Tensor, hook):
        del hook
        vector = unit(direction, activations)
        return activations + (
            mask.to(device=activations.device, dtype=activations.dtype).unsqueeze(-1)
            * float(shift)
            * vector
        )

    return hook


def coordinate_capture_hooks(
    names: list[str],
    direction: torch.Tensor,
    destination: dict[str, torch.Tensor],
) -> list:
    hooks = []
    for name in names:
        def capture(activations: torch.Tensor, hook, name=name):
            del hook
            destination[name] = torch.einsum(
                "bpd,d->bp", activations, unit(direction, activations)
            ).detach().float().cpu()
            return activations

        hooks.append((name, capture))
    return hooks


def calibrate_coordinates(
    model,
    rows: list[dict],
    *,
    direction: torch.Tensor,
    max_length: int,
) -> dict[str, dict[str, float]]:
    names = sorted(set(source_names(model) + monitor_names(model)))
    sums = {
        name: {"positive": 0.0, "negative": 0.0}
        for name in names
    }
    counts = {
        name: {"positive": 0, "negative": 0}
        for name in names
    }
    order = (
        "positive_sycophantic",
        "positive_normal",
        "negative_sycophantic",
        "negative_normal",
    )
    for row in tqdm(rows, desc="Calibrating source coordinates"):
        specs = build_transcript_specs(model.tokenizer, row, max_length=max_length)
        ordered = [specs[key] for key in order]
        tokens, attention_mask = tokenize_prompts(
            model, [spec.text for spec in ordered], max_length=max_length
        )
        response_mask = _response_mask(
            tokens, attention_mask, [spec.response_start for spec in ordered]
        )
        captured: dict[str, torch.Tensor] = {}
        with torch.inference_mode(), model.hooks(
            fwd_hooks=coordinate_capture_hooks(names, direction, captured)
        ):
            model(tokens, attention_mask=attention_mask)
        weights = response_mask.float().cpu()
        for name, coordinates in captured.items():
            means = (coordinates * weights).sum(dim=1) / weights.sum(dim=1)
            for condition, indices in (("positive", (0, 1)), ("negative", (2, 3))):
                sums[name][condition] += float(means[list(indices)].sum().item())
                counts[name][condition] += len(indices)
    return {
        name: {
            condition: sums[name][condition] / counts[name][condition]
            for condition in ("positive", "negative")
        }
        for name in names
    }


def tokenize_condition(model, specs, condition: str, max_length: int):
    ordered = [
        specs[f"{condition}_sycophantic"],
        specs[f"{condition}_normal"],
    ]
    tokens, attention_mask = tokenize_prompts(
        model, [spec.text for spec in ordered], max_length=max_length
    )
    response_mask = _response_mask(
        tokens, attention_mask, [spec.response_start for spec in ordered]
    )
    return ordered, tokens, attention_mask, response_mask


def run_condition(
    model,
    specs,
    *,
    condition: str,
    direction: torch.Tensor,
    target_hook: str,
    max_length: int,
    hooks: list | None = None,
) -> dict:
    ordered, tokens, attention_mask, response_mask = tokenize_condition(
        model, specs, condition, max_length
    )
    regions = response_region_masks(response_mask)
    captured: dict[str, torch.Tensor] = {}
    capture_hooks = coordinate_capture_hooks(
        [target_hook], direction, captured
    )
    with torch.inference_mode(), model.hooks(
        fwd_hooks=[*(hooks or []), *capture_hooks]
    ):
        logits = model(tokens, attention_mask=attention_mask)
    tail_logprobs = response_logprob_values(
        logits, tokens, regions["tail_targets"]
    ).float()
    target_coordinates = captured[target_hook]
    response_coordinates = []
    for index in range(2):
        response_coordinates.append(
            float(
                direction_coordinates(
                    masked_mean_states(
                        target_coordinates[index : index + 1].unsqueeze(-1),
                        regions["response"][index : index + 1].cpu(),
                    )[0],
                    torch.ones(1),
                ).item()
            )
        )
    return {
        "sycophantic_logprob": float(tail_logprobs[0].item()),
        "normal_logprob": float(tail_logprobs[1].item()),
        "margin": float((tail_logprobs[0] - tail_logprobs[1]).item()),
        "quality": float(tail_logprobs.mean().item()),
        "target_coordinates": {
            "sycophantic": response_coordinates[0],
            "normal": response_coordinates[1],
            "midpoint": 0.5 * sum(response_coordinates),
            "answer_gap": response_coordinates[0] - response_coordinates[1],
        },
    }


def paired_targets(
    model,
    specs,
    *,
    direction: torch.Tensor,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    _, tokens, attention_mask, response_mask = tokenize_condition(
        model, specs, "negative", max_length
    )
    captured: dict[str, torch.Tensor] = {}
    with torch.inference_mode(), model.hooks(
        fwd_hooks=coordinate_capture_hooks(source_names(model), direction, captured)
    ):
        model(tokens, attention_mask=attention_mask)
    return captured, response_mask


def source_centroid_hooks(
    model,
    *,
    direction: torch.Tensor,
    calibration: dict[str, dict[str, float]],
    mask: torch.Tensor,
) -> list:
    return [
        (
            name,
            scalar_projection_hook(
                direction,
                target=calibration[name]["negative"],
                mask=mask,
            ),
        )
        for name in source_names(model)
    ]


def source_translation_hooks(
    model,
    *,
    direction: torch.Tensor,
    calibration: dict[str, dict[str, float]],
    mask: torch.Tensor,
) -> list:
    return [
        (
            name,
            translation_hook(
                direction,
                shift=(
                    calibration[name]["negative"]
                    - calibration[name]["positive"]
                ),
                mask=mask,
            ),
        )
        for name in source_names(model)
    ]


def paired_projection_hooks(
    model,
    *,
    direction: torch.Tensor,
    targets: dict[str, torch.Tensor],
    mask: torch.Tensor,
    strength: float,
) -> list:
    return [
        (
            name,
            tensor_projection_hook(
                direction,
                target=targets[name],
                mask=mask,
                strength=strength,
            ),
        )
        for name in source_names(model)
    ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def cosine_2d(left: tuple[float, float], right: tuple[float, float]) -> float:
    numerator = left[0] * right[0] + left[1] * right[1]
    denominator = math.hypot(*left) * math.hypot(*right)
    return numerator / denominator if denominator else math.nan


def summarize(
    records: list[dict],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    condition_names = list(records[0]["conditions"])
    clean_margin = [r["conditions"]["clean_positive"]["margin"] for r in records]
    negative_margin = [r["conditions"]["natural_negative"]["margin"] for r in records]
    clean_syc = [
        r["conditions"]["clean_positive"]["sycophantic_logprob"] for r in records
    ]
    clean_normal = [
        r["conditions"]["clean_positive"]["normal_logprob"] for r in records
    ]
    negative_syc = [
        r["conditions"]["natural_negative"]["sycophantic_logprob"] for r in records
    ]
    negative_normal = [
        r["conditions"]["natural_negative"]["normal_logprob"] for r in records
    ]
    natural_effect = (
        mean(negative_syc) - mean(clean_syc),
        mean(negative_normal) - mean(clean_normal),
    )
    denominator = mean(clean_margin) - mean(negative_margin)
    output = {}
    for name in condition_names:
        margins = [r["conditions"][name]["margin"] for r in records]
        qualities = [r["conditions"][name]["quality"] for r in records]
        syc_values = [
            r["conditions"][name]["sycophantic_logprob"] for r in records
        ]
        normal_values = [
            r["conditions"][name]["normal_logprob"] for r in records
        ]
        answer_gaps = [
            r["conditions"][name]["target_coordinates"]["answer_gap"]
            for r in records
        ]
        effect = (
            mean(syc_values) - mean(clean_syc),
            mean(normal_values) - mean(clean_normal),
        )
        output[name] = {
            "margin": scalar_summary(
                margins,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "quality": scalar_summary(
                qualities,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "sycophantic_logprob": scalar_summary(
                syc_values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "normal_logprob": scalar_summary(
                normal_values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "target_answer_coordinate_gap": scalar_summary(
                answer_gaps,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "normalized_margin_reversal": (
                (mean(clean_margin) - mean(margins)) / denominator
                if denominator
                else math.nan
            ),
            "natural_logprob_effect_cosine": cosine_2d(effect, natural_effect),
            "mean_logprob_effect": {
                "sycophantic": effect[0],
                "normal": effect[1],
            },
        }
    return {
        "num_prompts": len(records),
        "natural_logprob_effect": {
            "sycophantic": natural_effect[0],
            "normal": natural_effect[1],
        },
        "conditions": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sycophancy_generation_centroid_clamp.json"),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--calibration-dataset",
        type=Path,
        default=DEFAULT_CLAMP_CALIBRATION_DATASET,
    )
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/sycophancy/generation/"
            "source_projection_diagnostic_n64"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite.")

    config = load_config(resolve(args.config))
    rows = read_jsonl(resolve(args.dataset))[: args.num_prompts]
    calibration_rows = read_jsonl(resolve(args.calibration_dataset))
    model_config = config["model"]
    model = load_model(
        model_config["name"],
        revision=model_config.get("revision"),
        device=model_config.get("device"),
        dtype=model_config.get("dtype", "bfloat16"),
        hook_mode="standard",
    )
    validate_rows(model.tokenizer, rows, max_length=args.max_length)
    validate_rows(model.tokenizer, calibration_rows, max_length=args.max_length)
    direction, _, vector_metadata = _load_vector(config, model)
    direction = direction.float()
    direction = direction / direction.norm().clamp_min(1e-8)
    target_hook = str(vector_metadata["target_hook"])

    started = time.monotonic()
    calibration = calibrate_coordinates(
        model,
        calibration_rows,
        direction=direction,
        max_length=args.max_length,
    )
    records = []
    strengths = (0.25, 0.5, 0.75, 1.0)
    for row in tqdm(rows, desc="Source-projection diagnostic"):
        specs = build_transcript_specs(
            model.tokenizer, row, max_length=args.max_length
        )
        _, _, _, positive_mask = tokenize_condition(
            model, specs, "positive", args.max_length
        )
        negative_targets, negative_mask = paired_targets(
            model,
            specs,
            direction=direction,
            max_length=args.max_length,
        )
        if not torch.equal(positive_mask.cpu(), negative_mask.cpu()):
            raise ValueError(f"Positive/negative masks differ for {row['id']}.")
        conditions = {
            "clean_positive": run_condition(
                model,
                specs,
                condition="positive",
                direction=direction,
                target_hook=target_hook,
                max_length=args.max_length,
            ),
            "natural_negative": run_condition(
                model,
                specs,
                condition="negative",
                direction=direction,
                target_hook=target_hook,
                max_length=args.max_length,
            ),
            "source_centroid_projection": run_condition(
                model,
                specs,
                condition="positive",
                direction=direction,
                target_hook=target_hook,
                max_length=args.max_length,
                hooks=source_centroid_hooks(
                    model,
                    direction=direction,
                    calibration=calibration,
                    mask=positive_mask,
                ),
            ),
            "source_mean_translation": run_condition(
                model,
                specs,
                condition="positive",
                direction=direction,
                target_hook=target_hook,
                max_length=args.max_length,
                hooks=source_translation_hooks(
                    model,
                    direction=direction,
                    calibration=calibration,
                    mask=positive_mask,
                ),
            ),
        }
        for strength in strengths:
            conditions[f"paired_projection_{strength:g}"] = run_condition(
                model,
                specs,
                condition="positive",
                direction=direction,
                target_hook=target_hook,
                max_length=args.max_length,
                hooks=paired_projection_hooks(
                    model,
                    direction=direction,
                    targets=negative_targets,
                    mask=positive_mask,
                    strength=strength,
                ),
            )
        records.append({"id": row["id"], "conditions": conditions})

    summary = summarize(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["calibration"] = {
        "num_prompts": len(calibration_rows),
        "target_layer": int(vector_metadata["layer"]),
        "target_hook": target_hook,
        "target_clean_coordinate": calibration[target_hook]["positive"],
        "target_negative_coordinate": calibration[target_hook]["negative"],
        "target_gap": (
            calibration[target_hook]["positive"]
            - calibration[target_hook]["negative"]
        ),
        "source_coordinates": calibration,
    }
    summary["wall_time_seconds"] = time.monotonic() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_prompt.jsonl", records)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

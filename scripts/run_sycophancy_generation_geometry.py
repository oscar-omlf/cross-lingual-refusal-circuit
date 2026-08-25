#!/usr/bin/env python3
"""Measure natural generation-vector geometry and causal coordinate effects."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusal_circuit.eapig import _response_mask, response_logprob_values
from refusal_circuit.interventions import (
    all_layer_coordinate_clamp_hooks,
    masked_state_replacement_hook,
    normalized_direction_shift,
    state_component_replacement,
)
from refusal_circuit.model import (
    format_system_instruction,
    load_model,
    tokenize_prompts,
)
from refusal_circuit.persona import (
    make_system_prompt,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from refusal_circuit.paths import write_json
from refusal_circuit.sycophancy_geometry import (
    ALIGNED_SYCOPHANCY_INSTRUCTION_PAIRS,
    CELL_KEYS,
    direction_coordinates,
    effect_geometry,
    factorial_effects,
    masked_mean_states,
    normalized_behavior_effect_summary,
    response_region_masks,
    scalar_summary,
    scatter_masked_states,
)
from refusal_circuit.workflow import (
    _load_vector,
    load_config,
)


DEFAULT_DATASET = (
    ROOT
    / "data/processed/qwen2.5-7b-instruct/trait_circuits/sycophancy/"
    "vector_validation.jsonl"
)
DEFAULT_CLAMP_CALIBRATION_DATASET = (
    ROOT
    / "data/processed/qwen2.5-7b-instruct/trait_circuits/sycophancy/"
    "vector_extraction.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/generation/"
    "geometry_coordinate_n64"
)
REGIONS = ("final_prompt", "maintenance", "response", "terminal")


@dataclass(frozen=True)
class TranscriptSpec:
    key: str
    instruction_condition: str
    answer_type: str
    text: str
    response_start: int
    token_count: int


class ActivationCapture:
    def __init__(self) -> None:
        self.activation: torch.Tensor | None = None

    def hook(self, activations: torch.Tensor, hook):
        del hook
        self.activation = activations.detach()
        return activations


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def formatted_prompt(tokenizer, row: dict, instruction: str) -> str:
    return format_system_instruction(
        tokenizer,
        make_system_prompt("sycophantic", instruction),
        row["prompt"],
    )


def build_transcript_specs(
    tokenizer,
    row: dict,
    *,
    max_length: int,
) -> dict[str, TranscriptSpec]:
    pair_index = int(row["instruction_pair_index"])
    if not 0 <= pair_index < len(ALIGNED_SYCOPHANCY_INSTRUCTION_PAIRS):
        raise ValueError(f"Invalid instruction_pair_index={pair_index}.")
    positive_instruction, negative_instruction = (
        ALIGNED_SYCOPHANCY_INSTRUCTION_PAIRS[pair_index]
    )
    prompts = {
        "positive": formatted_prompt(tokenizer, row, positive_instruction),
        "negative": formatted_prompt(tokenizer, row, negative_instruction),
    }
    prompt_ids = {
        condition: token_ids(tokenizer, prompt)
        for condition, prompt in prompts.items()
    }
    if len(prompt_ids["positive"]) != len(prompt_ids["negative"]):
        raise ValueError(
            f"Token-aligned instruction pair failed for {row['id']}: "
            f"{len(prompt_ids['positive'])} != {len(prompt_ids['negative'])}."
        )

    specs: dict[str, TranscriptSpec] = {}
    for answer_type, field in (
        ("sycophantic", "sycophantic_answer"),
        ("normal", "normal_answer"),
    ):
        answer = row[field]
        full_ids: dict[str, list[int]] = {}
        for condition, prompt in prompts.items():
            text = prompt + answer
            ids = token_ids(tokenizer, text)
            if len(ids) > max_length:
                raise ValueError(
                    f"{row['id']} {condition}_{answer_type} has {len(ids)} "
                    f"tokens, exceeding max_length={max_length}."
                )
            if len(ids) - len(prompt_ids[condition]) < 2:
                raise ValueError(
                    f"{row['id']} {answer_type} needs at least two response tokens."
                )
            full_ids[condition] = ids
            key = f"{condition}_{answer_type}"
            specs[key] = TranscriptSpec(
                key=key,
                instruction_condition=condition,
                answer_type=answer_type,
                text=text,
                response_start=len(prompt_ids[condition]),
                token_count=len(ids),
            )
        positive_suffix = full_ids["positive"][len(prompt_ids["positive"]) :]
        negative_suffix = full_ids["negative"][len(prompt_ids["negative"]) :]
        if positive_suffix != negative_suffix:
            raise ValueError(
                f"Answer tokens differ across instruction conditions for "
                f"{row['id']} {answer_type}."
            )
    return specs


def capture_baseline(
    model,
    specs: dict[str, TranscriptSpec],
    *,
    target_hook: str,
    direction: torch.Tensor,
    max_length: int,
    retain_sequences: bool,
    include_profile: bool,
) -> dict:
    ordered = [specs[key] for key in CELL_KEYS]
    texts = [spec.text for spec in ordered]
    starts = [spec.response_start for spec in ordered]
    tokens, attention_mask = tokenize_prompts(
        model,
        texts,
        max_length=max_length,
    )
    response_mask = _response_mask(tokens, attention_mask, starts)
    regions = response_region_masks(response_mask)
    capture = ActivationCapture()
    with torch.inference_mode(), model.hooks(
        fwd_hooks=[(target_hook, capture.hook)]
    ):
        logits = model(tokens, attention_mask=attention_mask)
    if capture.activation is None:
        raise RuntimeError(f"Target hook {target_hook!r} did not fire.")

    full_logprobs = response_logprob_values(
        logits,
        tokens,
        regions["response"],
    ).float()
    tail_logprobs = response_logprob_values(
        logits,
        tokens,
        regions["tail_targets"],
    ).float()
    activations = capture.activation
    result_cells: dict[str, dict] = {}
    profile_rows: list[dict] = []

    for index, spec in enumerate(ordered):
        mean_states = {
            region: masked_mean_states(
                activations[index : index + 1],
                regions[region][index : index + 1],
            )[0]
            .float()
            .cpu()
            for region in REGIONS
        }
        coordinates = {
            region: float(
                direction_coordinates(mean_states[region], direction.cpu()).item()
            )
            for region in REGIONS
        }
        cell = {
            "mean_states": mean_states,
            "coordinates": coordinates,
            "full_response_logprob": float(full_logprobs[index].item()),
            "tail_response_logprob": float(tail_logprobs[index].item()),
        }
        if retain_sequences:
            cell["maintenance_states"] = (
                activations[index][regions["maintenance"][index]]
                .float()
                .cpu()
            )
        result_cells[spec.key] = cell

        if include_profile:
            selected_states = (
                activations[index][regions["response"][index]].float().cpu()
            )
            selected_tokens = (
                tokens[index][regions["response"][index]].detach().cpu().tolist()
            )
            selected_coordinates = direction_coordinates(
                selected_states,
                direction.cpu(),
            ).tolist()
            count = len(selected_coordinates)
            for response_index, (coordinate, token_id) in enumerate(
                zip(selected_coordinates, selected_tokens, strict=True)
            ):
                profile_rows.append(
                    {
                        "cell": spec.key,
                        "instruction_condition": spec.instruction_condition,
                        "answer_type": spec.answer_type,
                        "response_index": response_index,
                        "response_tokens": count,
                        "relative_position": (
                            response_index / (count - 1) if count > 1 else 0.0
                        ),
                        "token_id": int(token_id),
                        "direction_coordinate": float(coordinate),
                        "is_maintenance_state": response_index < count - 1,
                    }
                )

    del logits, activations
    return {"cells": result_cells, "profile_rows": profile_rows}


def replacement_conditions(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    direction: torch.Tensor,
    natural_gap: float,
    source_condition: str,
) -> dict[str, torch.Tensor]:
    if source_condition == "positive":
        return {
            "directional_to_negative": state_component_replacement(
                source,
                target,
                direction,
                component="directional",
            ),
            "orthogonal_to_negative": state_component_replacement(
                source,
                target,
                direction,
                component="orthogonal",
            ),
            "full_to_negative": state_component_replacement(
                source,
                target,
                direction,
                component="full",
            ),
            "normalized_alpha_0p5_to_negative": normalized_direction_shift(
                source,
                direction,
                coordinate_gap=natural_gap,
                alpha=0.5,
                sign=-1.0,
            ),
            "normalized_alpha_1_to_negative": normalized_direction_shift(
                source,
                direction,
                coordinate_gap=natural_gap,
                alpha=1.0,
                sign=-1.0,
            ),
        }
    if source_condition == "negative":
        return {
            "directional_to_positive": state_component_replacement(
                source,
                target,
                direction,
                component="directional",
            ),
            "normalized_alpha_1_to_positive": normalized_direction_shift(
                source,
                direction,
                coordinate_gap=natural_gap,
                alpha=1.0,
                sign=1.0,
            ),
        }
    raise ValueError("source_condition must be 'positive' or 'negative'")


def score_replacement_conditions(
    model,
    spec: TranscriptSpec,
    conditions: dict[str, torch.Tensor],
    *,
    target_hook: str,
    direction: torch.Tensor,
    max_length: int,
    condition_batch_size: int,
) -> dict[str, dict[str, float]]:
    if condition_batch_size <= 0:
        raise ValueError("condition_batch_size must be positive.")
    names = list(conditions)
    scores: dict[str, dict[str, float]] = {}
    for start in range(0, len(names), condition_batch_size):
        batch_names = names[start : start + condition_batch_size]
        texts = [spec.text] * len(batch_names)
        starts = [spec.response_start] * len(batch_names)
        tokens, attention_mask = tokenize_prompts(
            model,
            texts,
            max_length=max_length,
        )
        response_mask = _response_mask(tokens, attention_mask, starts)
        regions = response_region_masks(response_mask)
        selected = torch.stack([conditions[name] for name in batch_names])
        replacement = scatter_masked_states(
            regions["maintenance"],
            selected,
            hidden_size=int(selected.shape[-1]),
        )
        hook = masked_state_replacement_hook(
            replacement,
            mask=regions["maintenance"],
        )
        with torch.inference_mode(), model.hooks(
            fwd_hooks=[(target_hook, hook)]
        ):
            logits = model(tokens, attention_mask=attention_mask)
        values = response_logprob_values(
            logits,
            tokens,
            regions["tail_targets"],
        ).float()
        post_coordinates = direction_coordinates(
            selected,
            direction.cpu(),
        ).mean(dim=1)
        for index, name in enumerate(batch_names):
            scores[name] = {
                "tail_response_logprob": float(values[index].item()),
                "post_intervention_coordinate": float(
                    post_coordinates[index].item()
                ),
            }
        del logits
    return scores


def baseline_record(
    row: dict,
    baseline: dict,
    *,
    direction: torch.Tensor,
) -> tuple[dict, dict[str, dict[str, torch.Tensor]], list[dict]]:
    cell_states = {
        region: {
            key: baseline["cells"][key]["mean_states"][region]
            for key in CELL_KEYS
        }
        for region in REGIONS
    }
    effects = {
        region: factorial_effects(states)
        for region, states in cell_states.items()
    }
    record = {
        "id": row["id"],
        "source_index": int(row["source_index"]),
        "instruction_pair_index": int(row["instruction_pair_index"]),
        "cells": {
            key: {
                "coordinates": baseline["cells"][key]["coordinates"],
                "full_response_logprob": baseline["cells"][key][
                    "full_response_logprob"
                ],
                "tail_response_logprob": baseline["cells"][key][
                    "tail_response_logprob"
                ],
            }
            for key in CELL_KEYS
        },
        "factorial_direction_coordinates": {
            region: {
                name: float(
                    direction_coordinates(vector, direction.cpu()).item()
                )
                for name, vector in region_effects.items()
            }
            for region, region_effects in effects.items()
        },
    }
    cells = record["cells"]
    positive_margin = (
        cells["positive_sycophantic"]["tail_response_logprob"]
        - cells["positive_normal"]["tail_response_logprob"]
    )
    negative_margin = (
        cells["negative_sycophantic"]["tail_response_logprob"]
        - cells["negative_normal"]["tail_response_logprob"]
    )
    record["behavior"] = {
        "positive_margin": positive_margin,
        "negative_margin": negative_margin,
        "positive_minus_negative_margin": positive_margin - negative_margin,
    }
    profile_rows = [
        {
            "id": row["id"],
            "source_index": int(row["source_index"]),
            "instruction_pair_index": int(row["instruction_pair_index"]),
            **profile,
        }
        for profile in baseline["profile_rows"]
    ]
    return record, effects, profile_rows


def summarize_geometry(
    geometry_records: list[dict],
    effect_vectors: dict[str, dict[str, list[torch.Tensor]]],
    *,
    direction: torch.Tensor,
    raw_direction: torch.Tensor,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, dict[str, dict[str, torch.Tensor]]]:
    mean_effect_vectors: dict[str, dict[str, torch.Tensor]] = {}
    regions: dict[str, dict] = {}
    for region in REGIONS:
        mean_effect_vectors[region] = {}
        region_summary = {"cells": {}, "factorial_effects": {}}
        for cell in CELL_KEYS:
            values = [
                float(record["cells"][cell]["coordinates"][region])
                for record in geometry_records
            ]
            region_summary["cells"][cell] = scalar_summary(
                values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        for effect_name in ("instruction", "answer", "interaction"):
            vectors = effect_vectors[region][effect_name]
            mean_vector = torch.stack(vectors).mean(dim=0)
            mean_effect_vectors[region][effect_name] = mean_vector
            scalar_values = [
                float(
                    record["factorial_direction_coordinates"][region][
                        effect_name
                    ]
                )
                for record in geometry_records
            ]
            region_summary["factorial_effects"][effect_name] = {
                "direction_coordinate": scalar_summary(
                    scalar_values,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "mean_vector_geometry": effect_geometry(
                    mean_vector,
                    direction.cpu(),
                ),
            }
        regions[region] = region_summary

    behavior_effects = [
        float(record["behavior"]["positive_minus_negative_margin"])
        for record in geometry_records
    ]
    natural_gap = float(
        regions["maintenance"]["factorial_effects"]["instruction"][
            "direction_coordinate"
        ]["mean"]
    )
    raw_norm = float(raw_direction.float().norm().item())
    scale = {
        "natural_maintenance_coordinate_gap": natural_gap,
        "raw_direction_norm": raw_norm,
        "raw_coefficient_for_alpha_1_positive_to_negative": (
            -natural_gap / raw_norm if raw_norm > 0 else math.nan
        ),
        "old_raw_minus2_shift_in_signed_natural_gap_units": (
            -2.0 * raw_norm / natural_gap if natural_gap != 0 else math.nan
        ),
        "old_raw_minus2_shift_in_absolute_natural_gap_units": (
            2.0 * raw_norm / abs(natural_gap)
            if natural_gap != 0
            else math.inf
        ),
    }
    return (
        {
            "num_prompts": len(geometry_records),
            "regions": regions,
            "natural_behavior_effect": scalar_summary(
                behavior_effects,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "steering_scale": scale,
        },
        mean_effect_vectors,
    )


def run_interventions(
    model,
    rows: list[dict],
    *,
    target_hook: str,
    direction: torch.Tensor,
    natural_gap: float,
    baseline_by_id: dict[str, dict],
    max_length: int,
    condition_batch_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict], dict]:
    prompt_records: list[dict] = []
    condition_margins: dict[str, list[float]] = defaultdict(list)

    for row in tqdm(rows, desc="Coordinate interventions"):
        specs = build_transcript_specs(
            model.tokenizer,
            row,
            max_length=max_length,
        )
        baseline = capture_baseline(
            model,
            specs,
            target_hook=target_hook,
            direction=direction,
            max_length=max_length,
            retain_sequences=True,
            include_profile=False,
        )
        answer_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for answer_type in ("sycophantic", "normal"):
            positive_key = f"positive_{answer_type}"
            negative_key = f"negative_{answer_type}"
            positive_states = baseline["cells"][positive_key][
                "maintenance_states"
            ]
            negative_states = baseline["cells"][negative_key][
                "maintenance_states"
            ]
            if tuple(positive_states.shape) != tuple(negative_states.shape):
                raise RuntimeError(
                    f"Paired maintenance states differ for {row['id']} "
                    f"{answer_type}: {tuple(positive_states.shape)} != "
                    f"{tuple(negative_states.shape)}."
                )
            positive_conditions = replacement_conditions(
                positive_states,
                negative_states,
                direction=direction.cpu(),
                natural_gap=natural_gap,
                source_condition="positive",
            )
            negative_conditions = replacement_conditions(
                negative_states,
                positive_states,
                direction=direction.cpu(),
                natural_gap=natural_gap,
                source_condition="negative",
            )
            positive_scores = score_replacement_conditions(
                model,
                specs[positive_key],
                positive_conditions,
                target_hook=target_hook,
                direction=direction,
                max_length=max_length,
                condition_batch_size=condition_batch_size,
            )
            negative_scores = score_replacement_conditions(
                model,
                specs[negative_key],
                negative_conditions,
                target_hook=target_hook,
                direction=direction,
                max_length=max_length,
                condition_batch_size=condition_batch_size,
            )
            for name, score in {**positive_scores, **negative_scores}.items():
                answer_scores[name][answer_type] = score

        baseline_behavior = baseline_by_id[row["id"]]["behavior"]
        positive_margin = float(baseline_behavior["positive_margin"])
        negative_margin = float(baseline_behavior["negative_margin"])
        denominator = positive_margin - negative_margin
        conditions: dict[str, dict] = {}
        for name, scores_by_answer in answer_scores.items():
            sycophantic = scores_by_answer["sycophantic"]
            normal = scores_by_answer["normal"]
            margin = (
                sycophantic["tail_response_logprob"]
                - normal["tail_response_logprob"]
            )
            toward_negative = name.endswith("_to_negative")
            normalized_effect = (
                (positive_margin - margin) / denominator
                if toward_negative and denominator != 0
                else (
                    (margin - negative_margin) / denominator
                    if denominator != 0
                    else math.nan
                )
            )
            conditions[name] = {
                "sycophantic_tail_response_logprob": sycophantic[
                    "tail_response_logprob"
                ],
                "normal_tail_response_logprob": normal[
                    "tail_response_logprob"
                ],
                "sycophantic_post_intervention_coordinate": sycophantic[
                    "post_intervention_coordinate"
                ],
                "normal_post_intervention_coordinate": normal[
                    "post_intervention_coordinate"
                ],
                "margin": margin,
                "normalized_behavioral_effect": normalized_effect,
            }
            condition_margins[name].append(float(margin))
        prompt_records.append(
            {
                "id": row["id"],
                "source_index": int(row["source_index"]),
                "instruction_pair_index": int(row["instruction_pair_index"]),
                "positive_baseline_margin": positive_margin,
                "negative_baseline_margin": negative_margin,
                "natural_margin_difference": denominator,
                "conditions": conditions,
            }
        )

    summary = {
        "num_prompts": len(prompt_records),
        "conditions": {
            name: {
                "margin": scalar_summary(
                    condition_margins[name],
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "normalized_behavioral_effect": (
                    normalized_behavior_effect_summary(
                        [
                            float(record["positive_baseline_margin"])
                            for record in prompt_records
                        ],
                        [
                            float(record["negative_baseline_margin"])
                            for record in prompt_records
                        ],
                        [
                            float(record["conditions"][name]["margin"])
                            for record in prompt_records
                        ],
                        toward=(
                            "negative"
                            if name.endswith("_to_negative")
                            else "positive"
                        ),
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_seed=bootstrap_seed,
                    )
                ),
                "per_prompt_normalized_effect": scalar_summary(
                    [
                        float(
                            record["conditions"][name][
                                "normalized_behavioral_effect"
                            ]
                        )
                        for record in prompt_records
                        if math.isfinite(
                            float(
                                record["conditions"][name][
                                    "normalized_behavioral_effect"
                                ]
                            )
                        )
                    ],
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
            for name in sorted(condition_margins)
        },
    }
    return prompt_records, summary


def score_global_clamp(
    model,
    specs: dict[str, TranscriptSpec],
    *,
    target_hook: str,
    direction: torch.Tensor,
    negative_coordinates: torch.Tensor,
    max_length: int,
) -> dict[str, dict]:
    ordered = [
        specs["positive_sycophantic"],
        specs["positive_normal"],
    ]
    tokens, attention_mask = tokenize_prompts(
        model,
        [spec.text for spec in ordered],
        max_length=max_length,
    )
    response_mask = _response_mask(
        tokens,
        attention_mask,
        [spec.response_start for spec in ordered],
    )
    regions = response_region_masks(response_mask)
    capture = ActivationCapture()
    final_capture = ActivationCapture()
    hooks = all_layer_coordinate_clamp_hooks(
        model,
        direction,
        target_coordinates=negative_coordinates,
        mask=response_mask,
    )
    final_hook = f"blocks.{int(model.cfg.n_layers) - 1}.hook_resid_post"
    with torch.inference_mode(), model.hooks(
        fwd_hooks=[
            *hooks,
            (target_hook, capture.hook),
            (final_hook, final_capture.hook),
        ]
    ):
        logits = model(tokens, attention_mask=attention_mask)
    if capture.activation is None:
        raise RuntimeError(f"Target hook {target_hook!r} did not fire.")
    if final_capture.activation is None:
        raise RuntimeError(f"Final residual hook {final_hook!r} did not fire.")

    full_logprobs = response_logprob_values(
        logits,
        tokens,
        regions["response"],
    ).float()
    tail_logprobs = response_logprob_values(
        logits,
        tokens,
        regions["tail_targets"],
    ).float()
    cells = {}
    for index, spec in enumerate(ordered):
        cells[spec.answer_type] = {
            "full_response_logprob": float(full_logprobs[index].item()),
            "tail_response_logprob": float(tail_logprobs[index].item()),
            "coordinates": {
                region: float(
                    direction_coordinates(
                        masked_mean_states(
                            capture.activation[index : index + 1],
                            regions[region][index : index + 1],
                        )[0],
                        direction,
                    ).item()
                )
                for region in ("maintenance", "response")
            },
            "final_residual_coordinates": {
                region: float(
                    direction_coordinates(
                        masked_mean_states(
                            final_capture.activation[index : index + 1],
                            regions[region][index : index + 1],
                        )[0],
                        direction,
                    ).item()
                )
                for region in ("maintenance", "response")
            },
        }
    return cells


def estimate_natural_negative_coordinates(
    model,
    rows: list[dict],
    *,
    direction: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Estimate each layer's anti-sycophantic response coordinate."""

    sums = torch.zeros(int(model.cfg.n_layers), dtype=torch.float64)
    counts = torch.zeros(int(model.cfg.n_layers), dtype=torch.int64)
    for row in tqdm(rows, desc="Calibrating natural anti-centroid"):
        specs = build_transcript_specs(
            model.tokenizer,
            row,
            max_length=max_length,
        )
        ordered = [
            specs["negative_sycophantic"],
            specs["negative_normal"],
        ]
        tokens, attention_mask = tokenize_prompts(
            model,
            [spec.text for spec in ordered],
            max_length=max_length,
        )
        response_mask = _response_mask(
            tokens,
            attention_mask,
            [spec.response_start for spec in ordered],
        )
        hooks = []
        for layer in range(int(model.cfg.n_layers)):
            def capture(activations: torch.Tensor, hook, layer=layer):
                del hook
                vector = direction.to(
                    device=activations.device,
                    dtype=activations.dtype,
                )
                coordinates = torch.einsum(
                    "bpd,d->bp",
                    activations,
                    vector,
                ).float()
                weights = response_mask.to(
                    device=activations.device,
                    dtype=coordinates.dtype,
                )
                per_transcript = (
                    (coordinates * weights).sum(dim=1)
                    / weights.sum(dim=1)
                )
                sums[layer] += per_transcript.double().sum().cpu()
                counts[layer] += per_transcript.numel()
                return activations

            hooks.append((f"blocks.{layer}.hook_resid_pre", capture))
        with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
            model(tokens, attention_mask=attention_mask)

    if (counts == 0).any():
        raise RuntimeError("Natural anti-centroid calibration captured no states.")
    coordinates = (sums / counts).float()
    if not torch.isfinite(coordinates).all():
        raise RuntimeError("Natural anti-centroid calibration is non-finite.")
    return coordinates


def run_clamp_gate(
    model,
    rows: list[dict],
    *,
    target_hook: str,
    target_layer: int,
    direction: torch.Tensor,
    negative_coordinates: torch.Tensor,
    baseline_by_id: dict[str, dict],
    max_length: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict], dict]:
    records = []
    for row in tqdm(rows, desc="Global anti-centroid clamp gate"):
        specs = build_transcript_specs(
            model.tokenizer,
            row,
            max_length=max_length,
        )
        clamped = score_global_clamp(
            model,
            specs,
            target_hook=target_hook,
            direction=direction,
            negative_coordinates=negative_coordinates,
            max_length=max_length,
        )
        baseline = baseline_by_id[row["id"]]
        cells = baseline["cells"]
        clean_margin = float(baseline["behavior"]["positive_margin"])
        negative_margin = float(baseline["behavior"]["negative_margin"])
        clamp_margin = (
            clamped["sycophantic"]["tail_response_logprob"]
            - clamped["normal"]["tail_response_logprob"]
        )
        natural_gap = clean_margin - negative_margin
        clean_quality = 0.5 * (
            cells["positive_sycophantic"]["tail_response_logprob"]
            + cells["positive_normal"]["tail_response_logprob"]
        )
        negative_quality = 0.5 * (
            cells["negative_sycophantic"]["tail_response_logprob"]
            + cells["negative_normal"]["tail_response_logprob"]
        )
        clamp_quality = 0.5 * (
            clamped["sycophantic"]["tail_response_logprob"]
            + clamped["normal"]["tail_response_logprob"]
        )
        coordinates = {}
        for region in ("maintenance", "response"):
            clean_coordinate = 0.5 * (
                cells["positive_sycophantic"]["coordinates"][region]
                + cells["positive_normal"]["coordinates"][region]
            )
            negative_coordinate = 0.5 * (
                cells["negative_sycophantic"]["coordinates"][region]
                + cells["negative_normal"]["coordinates"][region]
            )
            clamp_coordinate = 0.5 * (
                clamped["sycophantic"]["coordinates"][region]
                + clamped["normal"]["coordinates"][region]
            )
            denominator = clean_coordinate - negative_coordinate
            coordinates[region] = {
                "clean": clean_coordinate,
                "natural_negative": negative_coordinate,
                "clamp": clamp_coordinate,
                "normalized_reversal": (
                    (clean_coordinate - clamp_coordinate) / denominator
                    if denominator != 0
                    else math.nan
                ),
            }
        final_residual_coordinates = {
            region: 0.5
            * (
                clamped["sycophantic"]["final_residual_coordinates"][region]
                + clamped["normal"]["final_residual_coordinates"][region]
            )
            for region in ("maintenance", "response")
        }
        records.append(
            {
                "id": row["id"],
                "source_index": int(row["source_index"]),
                "instruction_pair_index": int(row["instruction_pair_index"]),
                "behavior": {
                    "clean_margin": clean_margin,
                    "natural_negative_margin": negative_margin,
                    "clamp_margin": clamp_margin,
                    "natural_gap": natural_gap,
                    "normalized_reversal": (
                        (clean_margin - clamp_margin) / natural_gap
                        if natural_gap != 0
                        else math.nan
                    ),
                },
                "answer_quality": {
                    "clean": clean_quality,
                    "natural_negative": negative_quality,
                    "clamp": clamp_quality,
                },
                "coordinates": coordinates,
                "final_residual_coordinates": final_residual_coordinates,
                "clamped_answers": clamped,
            }
        )

    behavior = {
        key: [
            float(record["behavior"][key])
            for record in records
        ]
        for key in (
            "clean_margin",
            "natural_negative_margin",
            "clamp_margin",
        )
    }
    finite_behavior_reversals = [
        float(record["behavior"]["normalized_reversal"])
        for record in records
        if math.isfinite(float(record["behavior"]["normalized_reversal"]))
    ]
    target_coordinate = float(negative_coordinates[target_layer].item())
    summary = {
        "num_prompts": len(records),
        "target_layer": target_layer,
        "target_negative_coordinate": target_coordinate,
        "behavior": {
            key: scalar_summary(
                values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for key, values in behavior.items()
        },
        "normalized_behavioral_reversal": normalized_behavior_effect_summary(
            behavior["clean_margin"],
            behavior["natural_negative_margin"],
            behavior["clamp_margin"],
            toward="negative",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "per_prompt_normalized_behavioral_reversal": scalar_summary(
            finite_behavior_reversals,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "answer_quality": {
            condition: scalar_summary(
                [
                    float(record["answer_quality"][condition])
                    for record in records
                ],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for condition in ("clean", "natural_negative", "clamp")
        },
        "coordinates": {
            region: {
                condition: scalar_summary(
                    [
                        float(record["coordinates"][region][condition])
                        for record in records
                    ],
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                )
                for condition in ("clean", "natural_negative", "clamp")
            }
            | {
                "normalized_reversal": scalar_summary(
                    [
                        float(
                            record["coordinates"][region][
                                "normalized_reversal"
                            ]
                        )
                        for record in records
                        if math.isfinite(
                            float(
                                record["coordinates"][region][
                                    "normalized_reversal"
                                ]
                            )
                        )
                    ],
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "clamp_target_absolute_error": scalar_summary(
                    [
                        abs(
                            float(record["coordinates"][region]["clamp"])
                            - target_coordinate
                        )
                        for record in records
                    ],
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
            for region in ("maintenance", "response")
        },
        "final_residual_projection": {
            "target_coordinate": float(negative_coordinates[-1].item()),
            "coordinates": {
                region: scalar_summary(
                    [
                        float(record["final_residual_coordinates"][region])
                        for record in records
                    ],
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                )
                for region in ("maintenance", "response")
            },
            "absolute_error": {
                region: scalar_summary(
                    [
                        abs(
                            float(record["final_residual_coordinates"][region])
                            - float(negative_coordinates[-1].item())
                        )
                        for record in records
                    ],
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                )
                for region in ("maintenance", "response")
            },
        },
    }
    return records, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(tokenizer, rows: list[dict], *, max_length: int) -> dict:
    maximum = 0
    counts = defaultdict(int)
    for row in rows:
        specs = build_transcript_specs(
            tokenizer,
            row,
            max_length=max_length,
        )
        maximum = max(maximum, *(spec.token_count for spec in specs.values()))
        counts[int(row["instruction_pair_index"])] += 1
    if (
        len(rows) >= len(ALIGNED_SYCOPHANCY_INSTRUCTION_PAIRS)
        and sorted(counts)
        != list(range(len(ALIGNED_SYCOPHANCY_INSTRUCTION_PAIRS)))
    ):
        raise ValueError("Validation rows must cover all five instruction pairs.")
    return {
        "num_prompts": len(rows),
        "transcripts": 4 * len(rows),
        "maximum_tokens": maximum,
        "max_length": max_length,
        "instruction_pair_counts": {
            str(index): counts[index] for index in sorted(counts)
        },
        "alignment_valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sycophancy_generation.json"),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--condition-batch-size", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clamp-gate-only", action="store_true")
    parser.add_argument(
        "--clamp-calibration-dataset",
        type=Path,
        default=DEFAULT_CLAMP_CALIBRATION_DATASET,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_prompts <= 0:
        raise ValueError("num-prompts must be positive.")
    if args.max_length <= 0:
        raise ValueError("max-length must be positive.")
    if args.condition_batch_size <= 0:
        raise ValueError("condition-batch-size must be positive.")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")

    config_path = resolve(args.config)
    dataset_path = resolve(args.dataset)
    output_dir = resolve(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} exists; pass --overwrite.")

    config = load_config(config_path)
    rows = read_jsonl(dataset_path)[: args.num_prompts]
    if len(rows) != args.num_prompts:
        raise ValueError(
            f"Expected {args.num_prompts} prompts, found {len(rows)}."
        )

    if args.validate_only:
        from transformers import AutoTokenizer

        model_config = config["model"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"],
            revision=model_config.get("revision"),
            local_files_only=True,
        )
        print(
            json.dumps(
                validate_rows(
                    tokenizer,
                    rows,
                    max_length=args.max_length,
                ),
                indent=2,
            )
        )
        return

    model_config = dict(config["model"])
    if args.device:
        model_config["device"] = args.device
    if args.dtype:
        model_config["dtype"] = args.dtype
    model = load_model(
        model_config["name"],
        revision=model_config.get("revision"),
        device=model_config.get("device"),
        dtype=model_config.get("dtype", "bfloat16"),
        hook_mode="standard",
    )
    validation = validate_rows(
        model.tokenizer,
        rows,
        max_length=args.max_length,
    )
    direction, raw_direction, vector_metadata = _load_vector(config, model)
    direction = direction.float()
    direction = direction / direction.norm().clamp_min(1e-8)
    target_hook = str(vector_metadata["target_hook"])

    started = time.monotonic()
    geometry_records: list[dict] = []
    profile_rows: list[dict] = []
    baseline_by_id: dict[str, dict] = {}
    effect_vectors: dict[str, dict[str, list[torch.Tensor]]] = {
        region: {
            effect: []
            for effect in ("instruction", "answer", "interaction")
        }
        for region in REGIONS
    }
    for row in tqdm(rows, desc="Natural 2x2 geometry"):
        specs = build_transcript_specs(
            model.tokenizer,
            row,
            max_length=args.max_length,
        )
        baseline = capture_baseline(
            model,
            specs,
            target_hook=target_hook,
            direction=direction,
            max_length=args.max_length,
            retain_sequences=False,
            include_profile=not args.clamp_gate_only,
        )
        record, effects, row_profiles = baseline_record(
            row,
            baseline,
            direction=direction,
        )
        geometry_records.append(record)
        baseline_by_id[row["id"]] = record
        profile_rows.extend(row_profiles)
        for region, region_effects in effects.items():
            for effect, vector in region_effects.items():
                effect_vectors[region][effect].append(vector)

    geometry_summary, mean_effect_vectors = summarize_geometry(
        geometry_records,
        effect_vectors,
        direction=direction,
        raw_direction=raw_direction,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    natural_gap = float(
        geometry_summary["steering_scale"][
            "natural_maintenance_coordinate_gap"
        ]
    )
    if args.clamp_gate_only:
        calibration_path = resolve(args.clamp_calibration_dataset)
        calibration_rows = read_jsonl(calibration_path)
        validate_rows(
            model.tokenizer,
            calibration_rows,
            max_length=args.max_length,
        )
        negative_coordinates = estimate_natural_negative_coordinates(
            model,
            calibration_rows,
            direction=direction,
            max_length=args.max_length,
        )
        gate_records, gate_summary = run_clamp_gate(
            model,
            rows,
            target_hook=target_hook,
            target_layer=int(vector_metadata["layer"]),
            direction=direction,
            negative_coordinates=negative_coordinates,
            baseline_by_id=baseline_by_id,
            max_length=args.max_length,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        elapsed_seconds = time.monotonic() - started
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "per_prompt_geometry.jsonl", geometry_records)
        write_jsonl(output_dir / "per_prompt_clamp_gate.jsonl", gate_records)
        summary = {
            "status": "complete",
            "geometry": geometry_summary,
            "clamp_gate": gate_summary,
        }
        write_json(output_dir / "summary.json", summary)
        write_json(
            metadata_path,
            {
                "status": "complete",
                "method": "response_masked_negative_centroid_affine_projection_gate",
                "model": {**model_config, "hook_mode": "standard"},
                "dataset": {
                    "path": str(dataset_path),
                    "sha256": sha256_file(dataset_path),
                    "num_prompts": len(rows),
                },
                "vector": {
                    "target_hook": target_hook,
                    "direction_norm": float(direction.norm().item()),
                    "anti_centroid_calibration": {
                        "path": str(calibration_path),
                        "sha256": sha256_file(calibration_path),
                        "num_prompts": len(calibration_rows),
                        "method": (
                            "mean natural negative-instruction response "
                            "coordinate over both stored answers"
                        ),
                    },
                },
                "protocol": {
                    "clean": "positive instruction without intervention",
                    "natural_negative": (
                        "token-aligned negative instruction without intervention"
                    ),
                    "clamp": (
                        "positive instruction with response residual states "
                        "affine-projected to the layerwise anti coordinate and "
                        "attention/MLP writes projected orthogonal to the "
                        "direction at every layer"
                    ),
                    "behavior_target": (
                        "mean tail logP(sycophantic answer) minus mean tail "
                        "logP(normal answer)"
                    ),
                    "bootstrap_samples": args.bootstrap_samples,
                    "bootstrap_seed": args.bootstrap_seed,
                },
                "validation": validation,
                "outputs": {
                    "summary": "summary.json",
                    "per_prompt_geometry": "per_prompt_geometry.jsonl",
                    "per_prompt_clamp_gate": "per_prompt_clamp_gate.jsonl",
                },
                "wall_time_seconds": elapsed_seconds,
            },
        )
        print(json.dumps(summary, indent=2))
        return

    intervention_records, intervention_summary = run_interventions(
        model,
        rows,
        target_hook=target_hook,
        direction=direction,
        natural_gap=natural_gap,
        baseline_by_id=baseline_by_id,
        max_length=args.max_length,
        condition_batch_size=args.condition_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    elapsed_seconds = time.monotonic() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_prompt_geometry.jsonl", geometry_records)
    write_jsonl(
        output_dir / "per_prompt_interventions.jsonl",
        intervention_records,
    )
    write_csv(output_dir / "position_coordinates.csv", profile_rows)
    torch.save(
        mean_effect_vectors,
        output_dir / "factorial_effect_vectors.pt",
    )
    summary = {
        "status": "complete",
        "geometry": geometry_summary,
        "interventions": intervention_summary,
    }
    write_json(output_dir / "summary.json", summary)
    metadata = {
        "status": "complete",
        "method": "natural_2x2_generation_geometry_with_coordinate_interventions",
        "model": {**model_config, "hook_mode": "standard"},
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "num_prompts": len(rows),
        },
        "vector": {
            "target_hook": target_hook,
            "direction_norm": float(direction.norm().item()),
            "raw_direction_norm": float(raw_direction.float().norm().item()),
            "output_dir": config["vector"]["output_dir"],
            "direction_file": config["vector"]["direction_file"],
            "raw_direction_file": config["vector"]["raw_direction_file"],
        },
        "instruction_pairs": [
            {
                "instruction_pair_index": index,
                "positive_instruction": pair[0],
                "negative_instruction": pair[1],
            }
            for index, pair in enumerate(
                ALIGNED_SYCOPHANCY_INSTRUCTION_PAIRS
            )
        ],
        "protocol": {
            "regions": list(REGIONS),
            "behavior_target": (
                "mean logP(sycophantic answer tokens after the first) minus "
                "mean logP(normal answer tokens after the first)"
            ),
            "coordinate_reference": "paired natural opposite-instruction state",
            "normalized_alphas": {
                "positive_to_negative": [0.5, 1.0],
                "negative_to_positive": [1.0],
            },
            "condition_batch_size": args.condition_batch_size,
            "max_length": args.max_length,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "validation": validation,
        "outputs": {
            "summary": "summary.json",
            "per_prompt_geometry": "per_prompt_geometry.jsonl",
            "per_prompt_interventions": "per_prompt_interventions.jsonl",
            "position_coordinates": "position_coordinates.csv",
            "factorial_effect_vectors": "factorial_effect_vectors.pt",
        },
        "wall_time_seconds": elapsed_seconds,
    }
    write_json(metadata_path, metadata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

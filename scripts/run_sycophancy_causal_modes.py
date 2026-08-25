#!/usr/bin/env python3
"""Extract and validate low-rank causal sycophancy modes at a natural interface."""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from refusal_circuit.causal_modes import (
    anchored_modes_from_matrix,
    anchored_natural_modes,
    descending_eigensystem,
    natural_modes,
    random_orthonormal_basis,
    subspace_component_replacement,
    symmetrized_causal_matrix,
)
from refusal_circuit.eapig import _response_mask, response_logprob_values
from refusal_circuit.interventions import masked_state_replacement_hook
from refusal_circuit.model import load_model, tokenize_prompts
from refusal_circuit.persona import read_jsonl, sha256_file, write_jsonl
from refusal_circuit.paths import write_json
from refusal_circuit.sycophancy_geometry import (
    normalized_behavior_effect_summary,
    response_region_masks,
    scalar_summary,
    scatter_masked_states,
)
from refusal_circuit.workflow import load_config

from scripts.run_sycophancy_generation_geometry import (
    TranscriptSpec,
    build_transcript_specs,
    validate_rows,
)


DEFAULT_EXTRACTION_DATASET = (
    ROOT
    / "data/processed/qwen2.5-7b-instruct/trait_circuits/sycophancy/"
    "vector_extraction.jsonl"
)
DEFAULT_VALIDATION_DATASET = (
    ROOT
    / "data/processed/qwen2.5-7b-instruct/trait_circuits/sycophancy/"
    "vector_validation.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/generation/"
    "causal_modes"
)


@dataclass
class PairCapture:
    positive_tokens: torch.Tensor
    positive_attention: torch.Tensor
    negative_tokens: torch.Tensor
    negative_attention: torch.Tensor
    positive_regions: dict[str, torch.Tensor]
    negative_regions: dict[str, torch.Tensor]
    positive_logprob: float
    negative_logprob: float
    activations: dict[str, tuple[torch.Tensor, torch.Tensor]]


class MultiCapture:
    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}

    def hook(self, activations: torch.Tensor, hook):
        self.values[hook.name] = activations.detach()
        return activations


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector.float() / vector.float().norm().clamp_min(1e-8)


def capture_pair(
    model,
    positive: TranscriptSpec,
    negative: TranscriptSpec,
    *,
    target_hooks: list[str],
    max_length: int,
) -> PairCapture:
    tokens, attention = tokenize_prompts(
        model,
        [positive.text, negative.text],
        max_length=max_length,
    )
    if not torch.equal(attention[0], attention[1]):
        raise ValueError("Positive and negative paired transcripts must align.")
    response_mask = _response_mask(
        tokens,
        attention,
        [positive.response_start, negative.response_start],
    )
    positive_regions = response_region_masks(response_mask[0:1])
    negative_regions = response_region_masks(response_mask[1:2])
    capture = MultiCapture()
    with torch.inference_mode(), model.hooks(
        fwd_hooks=[(name, capture.hook) for name in target_hooks]
    ):
        logits = model(tokens, attention_mask=attention)
    logprobs = response_logprob_values(
        logits,
        tokens,
        torch.cat(
            (
                positive_regions["tail_targets"],
                negative_regions["tail_targets"],
            ),
            dim=0,
        ),
    ).float()
    activations = {
        name: (value[0].detach(), value[1].detach())
        for name, value in capture.values.items()
    }
    missing = set(target_hooks) - set(activations)
    if missing:
        raise RuntimeError(f"Hooks did not fire: {sorted(missing)}")
    return PairCapture(
        positive_tokens=tokens[0:1],
        positive_attention=attention[0:1],
        negative_tokens=tokens[1:2],
        negative_attention=attention[1:2],
        positive_regions=positive_regions,
        negative_regions=negative_regions,
        positive_logprob=float(logprobs[0].item()),
        negative_logprob=float(logprobs[1].item()),
        activations=activations,
    )


def selected_states(
    pair: PairCapture,
    target_hook: str,
    *,
    condition: str,
    region: str = "maintenance",
) -> torch.Tensor:
    index = 0 if condition == "positive" else 1
    regions = (
        pair.positive_regions if condition == "positive" else pair.negative_regions
    )
    return pair.activations[target_hook][index][regions[region][0]]


def score_selected_replacements(
    model,
    pair: PairCapture,
    replacements: dict[str, torch.Tensor],
    *,
    target_hook: str,
    source_condition: str,
    condition_batch_size: int,
) -> dict[str, float]:
    if source_condition == "positive":
        tokens = pair.positive_tokens
        attention = pair.positive_attention
        regions = pair.positive_regions
    elif source_condition == "negative":
        tokens = pair.negative_tokens
        attention = pair.negative_attention
        regions = pair.negative_regions
    else:
        raise ValueError("source_condition must be positive or negative")
    names = list(replacements)
    scores: dict[str, float] = {}
    for start in range(0, len(names), condition_batch_size):
        batch_names = names[start : start + condition_batch_size]
        batch_tokens = tokens.expand(len(batch_names), -1)
        batch_attention = attention.expand(len(batch_names), -1)
        mask = regions["maintenance"].expand(len(batch_names), -1)
        selected = torch.stack(
            [replacements[name] for name in batch_names]
        ).to(device=batch_tokens.device)
        full = scatter_masked_states(
            mask,
            selected,
            hidden_size=int(selected.shape[-1]),
        )
        hook = masked_state_replacement_hook(full, mask=mask)
        with torch.inference_mode(), model.hooks(
            fwd_hooks=[(target_hook, hook)]
        ):
            logits = model(batch_tokens, attention_mask=batch_attention)
        values = response_logprob_values(
            logits,
            batch_tokens,
            regions["tail_targets"].expand(len(batch_names), -1),
        )
        for index, name in enumerate(batch_names):
            scores[name] = float(values[index].item())
    return scores


def score_full_state_conditions(
    model,
    pair: PairCapture,
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    target_hook: str,
    condition_batch_size: int,
) -> dict[str, float]:
    names = list(conditions)
    scores: dict[str, float] = {}
    for start in range(0, len(names), condition_batch_size):
        batch_names = names[start : start + condition_batch_size]
        batch_tokens = pair.positive_tokens.expand(len(batch_names), -1)
        batch_attention = pair.positive_attention.expand(len(batch_names), -1)
        masks = torch.stack([conditions[name][0] for name in batch_names]).to(
            device=batch_tokens.device,
            dtype=torch.bool,
        )
        replacements = torch.stack(
            [conditions[name][1] for name in batch_names]
        ).to(device=batch_tokens.device)
        hook = masked_state_replacement_hook(replacements, mask=masks)
        with torch.inference_mode(), model.hooks(
            fwd_hooks=[(target_hook, hook)]
        ):
            logits = model(batch_tokens, attention_mask=batch_attention)
        values = response_logprob_values(
            logits,
            batch_tokens,
            pair.positive_regions["tail_targets"].expand(
                len(batch_names), -1
            ),
        )
        for index, name in enumerate(batch_names):
            scores[name] = float(values[index].item())
    return scores


def integrated_selected_gradients(
    model,
    pair: PairCapture,
    *,
    target_hook: str,
    positive_states: torch.Tensor,
    negative_states: torch.Tensor,
    sign: float,
    ig_steps: int,
) -> tuple[torch.Tensor, float]:
    """Integrate signed output gradients from paired negative to positive state."""

    if sign not in {-1.0, 1.0}:
        raise ValueError("sign must be -1 or 1")
    delta = positive_states.float() - negative_states.float()
    gradient_sum = torch.zeros_like(delta, dtype=torch.float32)
    midpoint_alphas = [(step + 0.5) / ig_steps for step in range(ig_steps)]

    for alpha in midpoint_alphas:
        replacement = (
            negative_states.float() + float(alpha) * delta
        ).to(device=pair.positive_tokens.device)
        replacement = replacement.detach().requires_grad_(True)
        mask = pair.positive_regions["maintenance"]

        def hook(activations: torch.Tensor, hook):
            del hook
            base = activations.detach()
            result = base.clone()
            result[mask] = replacement.to(dtype=base.dtype)
            return result

        with model.hooks(fwd_hooks=[(target_hook, hook)]):
            logits = model(
                pair.positive_tokens,
                attention_mask=pair.positive_attention,
            )
            metric = float(sign) * response_logprob_values(
                logits,
                pair.positive_tokens,
                pair.positive_regions["tail_targets"],
            ).sum()
        gradient = torch.autograd.grad(metric, replacement)[0]
        gradient_sum.add_(gradient.detach().float())
        del logits, metric, gradient, replacement

    integrated = gradient_sum / ig_steps
    line_contribution = float((delta.to(integrated) * integrated).sum().item())
    return integrated, line_contribution


def load_all_layer_directions(config: dict) -> torch.Tensor:
    vector_dir = resolve(Path(config["vector"]["output_dir"]))
    metadata = json.loads((vector_dir / "metadata.json").read_text())
    filename = metadata.get(
        "all_layer_response_vector_file",
        "sycophantic_response_avg_diff.pt",
    )
    return torch.load(
        vector_dir / filename,
        map_location="cpu",
        weights_only=True,
    ).float()


def screen_layers(
    model,
    rows: list[dict],
    *,
    candidate_layers: list[int],
    max_length: int,
    condition_batch_size: int,
) -> dict:
    hooks = [f"blocks.{layer}.hook_resid_pre" for layer in candidate_layers]
    per_layer: dict[int, dict[str, list[float]]] = {
        layer: defaultdict(list) for layer in candidate_layers
    }

    for row in tqdm(rows, desc="Layer full-state screen"):
        specs = build_transcript_specs(
            model.tokenizer,
            row,
            max_length=max_length,
        )
        answer_data: dict[str, dict] = {}
        for answer_type in ("sycophantic", "normal"):
            pair = capture_pair(
                model,
                specs[f"positive_{answer_type}"],
                specs[f"negative_{answer_type}"],
                target_hooks=hooks,
                max_length=max_length,
            )
            answer_data[answer_type] = {"pair": pair, "patch": {}}
            for layer, hook_name in zip(candidate_layers, hooks, strict=True):
                target = selected_states(
                    pair,
                    hook_name,
                    condition="negative",
                )
                score = score_selected_replacements(
                    model,
                    pair,
                    {"full": target},
                    target_hook=hook_name,
                    source_condition="positive",
                    condition_batch_size=condition_batch_size,
                )["full"]
                answer_data[answer_type]["patch"][layer] = score

        positive_margin = (
            answer_data["sycophantic"]["pair"].positive_logprob
            - answer_data["normal"]["pair"].positive_logprob
        )
        negative_margin = (
            answer_data["sycophantic"]["pair"].negative_logprob
            - answer_data["normal"]["pair"].negative_logprob
        )
        for layer in candidate_layers:
            patch_margin = (
                answer_data["sycophantic"]["patch"][layer]
                - answer_data["normal"]["patch"][layer]
            )
            per_layer[layer]["positive"].append(positive_margin)
            per_layer[layer]["negative"].append(negative_margin)
            per_layer[layer]["patched"].append(patch_margin)

    summary = {}
    for layer in candidate_layers:
        values = per_layer[layer]
        summary[str(layer)] = {
            "positive_margin": scalar_summary(values["positive"]),
            "negative_margin": scalar_summary(values["negative"]),
            "full_response_patch_margin": scalar_summary(values["patched"]),
            "normalized_effect": normalized_behavior_effect_summary(
                values["positive"],
                values["negative"],
                values["patched"],
                toward="negative",
            ),
        }
    selected = max(
        candidate_layers,
        key=lambda layer: float(
            summary[str(layer)]["normalized_effect"]["normalized_effect"]
        ),
    )
    return {
        "num_prompts": len(rows),
        "candidate_layers": candidate_layers,
        "selected_layer": selected,
        "layers": summary,
    }


def extract_modes(args) -> None:
    config = load_config(resolve(args.config))
    dataset = resolve(args.dataset)
    output_dir = resolve(args.output_dir)
    if (output_dir / "metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir}/metadata.json exists; use --overwrite.")

    rows = read_jsonl(dataset)
    if len(rows) < args.num_prompts:
        raise ValueError(f"Need {args.num_prompts} prompts, found {len(rows)}.")
    rows = rows[: args.num_prompts]
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
    candidate_layers = [
        int(value) for value in args.candidate_layers.split(",")
    ]
    screen = screen_layers(
        model,
        rows[: args.screen_prompts],
        candidate_layers=candidate_layers,
        max_length=args.max_length,
        condition_batch_size=args.condition_batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "layer_screen.json", screen)
    selected_layer = int(screen["selected_layer"])
    target_hook = f"blocks.{selected_layer}.hook_resid_pre"
    all_layer_directions = load_all_layer_directions(config)
    if not 0 <= selected_layer < all_layer_directions.shape[0]:
        raise ValueError("Selected layer is absent from the all-layer vector.")
    existing_direction = unit(all_layer_directions[selected_layer])

    hidden_size = int(existing_direction.numel())
    device = next(model.parameters()).device
    causal_sum = torch.zeros(
        hidden_size,
        hidden_size,
        device=device,
        dtype=torch.float32,
    )
    natural_effects: list[torch.Tensor] = []
    records: list[dict] = []
    started = time.monotonic()

    for row in tqdm(rows, desc=f"Causal modes at layer {selected_layer}"):
        specs = build_transcript_specs(
            model.tokenizer,
            row,
            max_length=args.max_length,
        )
        answer_records = {}
        answer_effects = []
        for answer_type, sign in (("sycophantic", 1.0), ("normal", -1.0)):
            pair = capture_pair(
                model,
                specs[f"positive_{answer_type}"],
                specs[f"negative_{answer_type}"],
                target_hooks=[target_hook],
                max_length=args.max_length,
            )
            positive_states = selected_states(
                pair,
                target_hook,
                condition="positive",
            )
            negative_states = selected_states(
                pair,
                target_hook,
                condition="negative",
            )
            delta = positive_states.float() - negative_states.float()
            gradients, line_contribution = integrated_selected_gradients(
                model,
                pair,
                target_hook=target_hook,
                positive_states=positive_states,
                negative_states=negative_states,
                sign=sign,
                ig_steps=args.ig_steps,
            )
            causal_sum.add_(
                symmetrized_causal_matrix(
                    delta.to(device=device),
                    gradients.to(device=device),
                )
            )
            patched_logprob = score_selected_replacements(
                model,
                pair,
                {"full": negative_states},
                target_hook=target_hook,
                source_condition="positive",
                condition_batch_size=1,
            )["full"]
            answer_effects.append(delta.mean(dim=0).cpu())
            answer_records[answer_type] = {
                "positive_logprob": pair.positive_logprob,
                "negative_logprob": pair.negative_logprob,
                "full_patch_logprob": patched_logprob,
                "ig_line_contribution": line_contribution,
                "maintenance_states": int(delta.shape[0]),
            }
            del pair, positive_states, negative_states, delta, gradients

        natural_effects.append(
            0.5 * (answer_effects[0] + answer_effects[1])
        )
        positive_margin = (
            answer_records["sycophantic"]["positive_logprob"]
            - answer_records["normal"]["positive_logprob"]
        )
        negative_margin = (
            answer_records["sycophantic"]["negative_logprob"]
            - answer_records["normal"]["negative_logprob"]
        )
        patch_margin = (
            answer_records["sycophantic"]["full_patch_logprob"]
            - answer_records["normal"]["full_patch_logprob"]
        )
        records.append(
            {
                "id": row["id"],
                "source_index": int(row["source_index"]),
                "positive_margin": positive_margin,
                "negative_margin": negative_margin,
                "full_response_patch_margin": patch_margin,
                "local_full_effect": positive_margin - patch_margin,
                "ig_line_contribution": (
                    answer_records["sycophantic"]["ig_line_contribution"]
                    + answer_records["normal"]["ig_line_contribution"]
                ),
                "answers": answer_records,
            }
        )

    causal_matrix = causal_sum / len(rows)
    effects = torch.stack(natural_effects).to(device=device, dtype=torch.float32)
    natural_mean, natural_values, natural_basis = natural_modes(
        effects,
        max_modes=args.max_modes,
    )
    causal_values, causal_basis = descending_eigensystem(
        causal_matrix,
        orient_toward=natural_mean,
    )
    causal_basis = causal_basis[: args.max_modes]
    (
        anchored_natural_mean,
        anchored_natural_values,
        anchored_natural_basis,
    ) = anchored_natural_modes(
        effects,
        existing_direction.to(effects),
        max_modes=args.max_modes,
    )
    anchored_causal_values, anchored_causal_basis = (
        anchored_modes_from_matrix(
            causal_matrix,
            existing_direction.to(causal_matrix),
            max_modes=args.max_modes,
        )
    )

    positive = [float(record["positive_margin"]) for record in records]
    negative = [float(record["negative_margin"]) for record in records]
    patched = [
        float(record["full_response_patch_margin"]) for record in records
    ]
    endpoint_effect = sum(
        float(record["local_full_effect"]) for record in records
    ) / len(records)
    ig_effect = sum(
        float(record["ig_line_contribution"]) for record in records
    ) / len(records)
    k_values = [value for value in (1, 2, 4, 8) if value <= args.max_modes]
    summary = {
        "status": "complete",
        "layer_screen": screen,
        "selected_layer": selected_layer,
        "target_hook": target_hook,
        "num_prompts": len(rows),
        "local_full_state": {
            "positive_margin": scalar_summary(positive),
            "negative_margin": scalar_summary(negative),
            "patched_margin": scalar_summary(patched),
            "normalized_effect": normalized_behavior_effect_summary(
                positive,
                negative,
                patched,
                toward="negative",
            ),
            "endpoint_effect": endpoint_effect,
            "integrated_gradient_trace": float(causal_matrix.trace().item()),
            "mean_per_prompt_ig_line_contribution": ig_effect,
            "trace_to_endpoint_ratio": (
                float(causal_matrix.trace().item()) / endpoint_effect
                if endpoint_effect != 0
                else math.nan
            ),
        },
        "direction_cosines": {
            "existing_vs_natural_mean": float(
                existing_direction.to(natural_mean) @ unit(natural_mean)
            ),
            "existing_vs_causal_top1": float(
                existing_direction.to(causal_basis) @ causal_basis[0]
            ),
            "natural_mean_vs_causal_top1": float(
                unit(natural_mean).to(causal_basis) @ causal_basis[0]
            ),
        },
        "causal_spectrum": {
            "top_eigenvalues": [
                float(value)
                for value in causal_values[: args.max_modes].tolist()
            ],
            "positive_eigenvalue_mass": float(
                causal_values.clamp_min(0).sum().item()
            ),
            "negative_eigenvalue_mass": float(
                causal_values.clamp_max(0).sum().item()
            ),
            "cumulative_top_k_over_trace": {
                str(k): float(
                    causal_values[:k].sum().item()
                    / causal_matrix.trace().item()
                )
                for k in k_values
            },
        },
        "natural_spectrum": {
            "top_residual_eigenvalues": [
                float(value)
                for value in natural_values[: args.max_modes].tolist()
            ]
        },
        "existing_direction_anchored": {
            "natural_mean_matches_unanchored": bool(
                torch.allclose(
                    natural_mean,
                    anchored_natural_mean,
                    atol=1e-6,
                    rtol=1e-6,
                )
            ),
            "natural_component_values": [
                float(value)
                for value in anchored_natural_values.tolist()
            ],
            "causal_component_contributions": [
                float(value)
                for value in anchored_causal_values.tolist()
            ],
            "causal_cumulative_top_k_over_trace": {
                str(k): float(
                    anchored_causal_values[:k].sum().item()
                    / causal_matrix.trace().item()
                )
                for k in k_values
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_prompt_extraction.jsonl", records)
    write_json(output_dir / "summary.json", summary)
    torch.save(
        {
            "selected_layer": selected_layer,
            "target_hook": target_hook,
            "existing_direction": existing_direction.cpu(),
            "natural_mean": natural_mean.cpu(),
            "natural_modes": natural_basis.cpu(),
            "natural_eigenvalues": natural_values.cpu(),
            "causal_modes": causal_basis.cpu(),
            "causal_eigenvalues": causal_values.cpu(),
            "anchored_natural_modes": anchored_natural_basis.cpu(),
            "anchored_natural_values": anchored_natural_values.cpu(),
            "anchored_causal_modes": anchored_causal_basis.cpu(),
            "anchored_causal_values": anchored_causal_values.cpu(),
            "causal_matrix_trace": causal_matrix.trace().cpu(),
        },
        output_dir / "modes.pt",
    )
    metadata = {
        "status": "complete",
        "stage": "extraction",
        "method": "natural_layer_screen_and_integrated_gradient_causal_modes",
        "model": {**model_config, "hook_mode": "standard"},
        "dataset": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "num_prompts": len(rows),
            "screen_prompts": args.screen_prompts,
        },
        "candidate_layers": candidate_layers,
        "selected_layer": selected_layer,
        "ig_steps": args.ig_steps,
        "ig_rule": "midpoint",
        "max_modes": args.max_modes,
        "max_length": args.max_length,
        "wall_time_seconds": time.monotonic() - started,
        "outputs": {
            "layer_screen": "layer_screen.json",
            "summary": "summary.json",
            "modes": "modes.pt",
            "per_prompt": "per_prompt_extraction.jsonl",
        },
        "validation": validation,
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(summary, indent=2))


def candidate_bases(
    artifact: dict,
    *,
    random_seeds: list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    hidden_size = int(artifact["existing_direction"].numel())
    maximum = int(artifact["causal_modes"].shape[0])
    k_values = [value for value in (1, 2, 4, 8) if value <= maximum]
    forward = {
        "existing_direction": artifact["existing_direction"].unsqueeze(0),
    }
    reverse = {
        "existing_direction": artifact["existing_direction"].unsqueeze(0),
    }
    for k in k_values:
        forward[f"natural_top_{k}"] = artifact["natural_modes"][:k]
        forward[f"causal_top_{k}"] = artifact["causal_modes"][:k]
        reverse[f"causal_top_{k}"] = artifact["causal_modes"][:k]
        if k > 1:
            forward[f"existing_plus_natural_top_{k}"] = (
                artifact["anchored_natural_modes"][:k]
            )
            forward[f"existing_plus_causal_top_{k}"] = (
                artifact["anchored_causal_modes"][:k]
            )
    reverse["natural_top_1"] = artifact["natural_modes"][:1]
    for seed in random_seeds:
        basis = random_orthonormal_basis(
            hidden_size,
            max(k_values),
            seed=seed,
        )
        for k in k_values:
            forward[f"random_seed_{seed}_top_{k}"] = basis[:k]
    return forward, reverse


def validate_modes(args) -> None:
    config = load_config(resolve(args.config))
    dataset = resolve(args.dataset)
    extraction_dir = resolve(args.extraction_dir)
    output_dir = resolve(args.output_dir)
    if (output_dir / "metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir}/metadata.json exists; use --overwrite.")
    artifact = torch.load(
        extraction_dir / "modes.pt",
        map_location="cpu",
        weights_only=True,
    )
    rows = read_jsonl(dataset)
    if len(rows) < args.num_prompts:
        raise ValueError(f"Need {args.num_prompts} prompts, found {len(rows)}.")
    rows = rows[: args.num_prompts]
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
    target_hook = str(artifact["target_hook"])
    random_seeds = [
        int(value) for value in args.random_seeds.split(",") if value
    ]
    forward_bases, reverse_bases = candidate_bases(
        artifact,
        random_seeds=random_seeds,
    )
    records = []
    condition_margins: dict[str, list[float]] = defaultdict(list)
    direction_by_condition: dict[str, str] = {}
    started = time.monotonic()

    for row in tqdm(rows, desc="Held-out causal-mode validation"):
        specs = build_transcript_specs(
            model.tokenizer,
            row,
            max_length=args.max_length,
        )
        answer_scores: dict[str, dict[str, float]] = defaultdict(dict)
        baselines = {}
        for answer_type in ("sycophantic", "normal"):
            pair = capture_pair(
                model,
                specs[f"positive_{answer_type}"],
                specs[f"negative_{answer_type}"],
                target_hooks=[target_hook],
                max_length=args.max_length,
            )
            positive_states = selected_states(
                pair,
                target_hook,
                condition="positive",
            ).float().cpu()
            negative_states = selected_states(
                pair,
                target_hook,
                condition="negative",
            ).float().cpu()
            forward_replacements = {
                name: subspace_component_replacement(
                    positive_states,
                    negative_states,
                    basis,
                )
                for name, basis in forward_bases.items()
            }
            reverse_replacements = {
                name: subspace_component_replacement(
                    negative_states,
                    positive_states,
                    basis,
                )
                for name, basis in reverse_bases.items()
            }
            for name, value in score_selected_replacements(
                model,
                pair,
                forward_replacements,
                target_hook=target_hook,
                source_condition="positive",
                condition_batch_size=args.condition_batch_size,
            ).items():
                key = f"{name}_to_negative"
                answer_scores[key][answer_type] = value
                direction_by_condition[key] = "negative"
            for name, value in score_selected_replacements(
                model,
                pair,
                reverse_replacements,
                target_hook=target_hook,
                source_condition="negative",
                condition_batch_size=args.condition_batch_size,
            ).items():
                key = f"{name}_to_positive"
                answer_scores[key][answer_type] = value
                direction_by_condition[key] = "positive"

            positive_full, negative_full = pair.activations[target_hook]
            prompt_mask = (
                pair.positive_attention.bool()
                & ~pair.positive_regions["response"]
            )
            maintenance_mask = pair.positive_regions["maintenance"]
            combined_mask = prompt_mask | maintenance_mask
            positional = {}
            for name, mask in (
                ("full_prompt_to_negative", prompt_mask),
                ("full_response_to_negative", maintenance_mask),
                ("full_prompt_response_to_negative", combined_mask),
            ):
                replacement = positive_full.clone()
                replacement[mask[0]] = negative_full[mask[0]]
                positional[name] = (mask[0].cpu(), replacement.cpu())
            for name, value in score_full_state_conditions(
                model,
                pair,
                positional,
                target_hook=target_hook,
                condition_batch_size=args.condition_batch_size,
            ).items():
                answer_scores[name][answer_type] = value
                direction_by_condition[name] = "negative"
            baselines[answer_type] = {
                "positive": pair.positive_logprob,
                "negative": pair.negative_logprob,
            }

        positive_margin = (
            baselines["sycophantic"]["positive"]
            - baselines["normal"]["positive"]
        )
        negative_margin = (
            baselines["sycophantic"]["negative"]
            - baselines["normal"]["negative"]
        )
        conditions = {}
        for name, by_answer in answer_scores.items():
            margin = by_answer["sycophantic"] - by_answer["normal"]
            conditions[name] = {"margin": margin}
            condition_margins[name].append(margin)
        records.append(
            {
                "id": row["id"],
                "source_index": int(row["source_index"]),
                "positive_margin": positive_margin,
                "negative_margin": negative_margin,
                "natural_margin_difference": positive_margin - negative_margin,
                "conditions": conditions,
            }
        )

    positive = [float(record["positive_margin"]) for record in records]
    negative = [float(record["negative_margin"]) for record in records]
    summaries = {}
    for name, margins in sorted(condition_margins.items()):
        toward = direction_by_condition[name]
        summaries[name] = {
            "margin": scalar_summary(
                margins,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            ),
            "normalized_effect": normalized_behavior_effect_summary(
                positive,
                negative,
                margins,
                toward=toward,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            ),
        }
    summary = {
        "status": "complete",
        "num_prompts": len(records),
        "selected_layer": int(artifact["selected_layer"]),
        "target_hook": target_hook,
        "positive_margin": scalar_summary(positive),
        "negative_margin": scalar_summary(negative),
        "conditions": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_prompt_validation.jsonl", records)
    write_json(output_dir / "summary.json", summary)
    metadata = {
        "status": "complete",
        "stage": "heldout_validation",
        "method": "paired_subspace_replacement_and_positional_bypass_localization",
        "model": {**model_config, "hook_mode": "standard"},
        "dataset": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "num_prompts": len(rows),
        },
        "extraction_dir": str(extraction_dir),
        "selected_layer": int(artifact["selected_layer"]),
        "target_hook": target_hook,
        "k_values": [1, 2, 4, 8],
        "random_seeds": random_seeds,
        "max_length": args.max_length,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "wall_time_seconds": time.monotonic() - started,
        "outputs": {
            "summary": "summary.json",
            "per_prompt": "per_prompt_validation.jsonl",
        },
        "validation": validation,
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "validate"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sycophancy_generation.json"),
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--extraction-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--screen-prompts", type=int, default=64)
    parser.add_argument("--candidate-layers", default="16,20,24")
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--max-modes", type=int, default=16)
    parser.add_argument("--condition-batch-size", type=int, default=8)
    parser.add_argument("--random-seeds", default="0,1,2,3")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.ig_steps <= 0:
        raise ValueError("ig-steps must be positive")
    if args.max_modes < 8:
        raise ValueError("max-modes must be at least 8")
    if args.condition_batch_size <= 0:
        raise ValueError("condition-batch-size must be positive")

    if args.stage == "extract":
        args.dataset = args.dataset or DEFAULT_EXTRACTION_DATASET
        args.output_dir = args.output_dir or DEFAULT_OUTPUT
        args.num_prompts = args.num_prompts or 256
        if not 0 < args.screen_prompts <= args.num_prompts:
            raise ValueError("screen-prompts must lie in [1, num-prompts]")
        extract_modes(args)
    else:
        args.dataset = args.dataset or DEFAULT_VALIDATION_DATASET
        args.output_dir = args.output_dir or (
            DEFAULT_OUTPUT / "heldout_validation"
        )
        args.num_prompts = args.num_prompts or 64
        validate_modes(args)


if __name__ == "__main__":
    main()

"""Paper-faithful difference-in-means refusal-direction extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from tqdm.auto import tqdm

from .interventions import addition_hooks, all_layer_ablation_hooks
from .metrics import ProjectionCapture, mean_last_token_kl, refusal_log_odds
from .model import tokenize_prompts


@dataclass(frozen=True)
class DirectionCandidate:
    """A raw difference-in-means vector from one layer and template token."""

    layer: int
    position: int
    raw_vector: torch.Tensor

    @property
    def unit_vector(self) -> torch.Tensor:
        return self.raw_vector / self.raw_vector.norm().clamp_min(1e-8)


def _batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def mean_residuals(
    model,
    prompts: list[str],
    *,
    layers: Iterable[int],
    positions: Iterable[int],
    batch_size: int = 8,
    max_length: int | None = 512,
) -> dict[tuple[int, int], torch.Tensor]:
    """Mean ``resid_pre`` activations for every requested layer/position pair."""
    layers = list(layers)
    positions = list(positions)
    if not prompts:
        raise ValueError("Cannot extract a direction from an empty prompt set.")
    if not positions:
        raise ValueError("At least one post-instruction position is required.")
    totals: dict[tuple[int, int], torch.Tensor] = {
        (layer, position): torch.zeros(model.cfg.d_model, dtype=torch.float64)
        for layer in layers
        for position in positions
    }

    def capture(layer: int):
        def hook(activations: torch.Tensor, hook):
            # Shape: [batch, n_positions, d_model].  The selected suffix is
            # right-aligned under left padding, so negative positions are
            # valid and refer to the same template token in each sequence.
            selected = activations[:, positions].detach().float().cpu().sum(dim=0)
            for index, position in enumerate(positions):
                totals[(layer, position)].add_(selected[index])
            return activations

        return hook

    hooks = [(f"blocks.{layer}.hook_resid_pre", capture(layer)) for layer in layers]
    for prompt_batch in tqdm(_batches(prompts, batch_size), desc="Caching residual means"):
        tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=max_length)
        with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
            model(tokens, attention_mask=attention_mask)
    return {key: total / len(prompts) for key, total in totals.items()}


def difference_in_means_candidates(
    model,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    *,
    layers: Iterable[int],
    positions: Iterable[int],
    batch_size: int = 8,
    max_length: int | None = 512,
) -> list[DirectionCandidate]:
    """Compute one DIM candidate per layer and post-instruction position."""
    layers = list(layers)
    positions = list(positions)
    harmful = mean_residuals(
        model,
        harmful_prompts,
        layers=layers,
        positions=positions,
        batch_size=batch_size,
        max_length=max_length,
    )
    harmless = mean_residuals(
        model,
        harmless_prompts,
        layers=layers,
        positions=positions,
        batch_size=batch_size,
        max_length=max_length,
    )
    return [
        DirectionCandidate(
            layer=layer,
            position=position,
            raw_vector=harmful[(layer, position)] - harmless[(layer, position)],
        )
        for position in positions
        for layer in layers
    ]


def _run_projection_and_logits(
    model,
    prompts: list[str],
    *,
    target_hook: str,
    target_position: int,
    direction: torch.Tensor,
    fwd_hooks: list | None,
    batch_size: int,
    max_length: int | None,
):
    projections: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    for prompt_batch in _batches(prompts, batch_size):
        tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=max_length)
        capture = ProjectionCapture(direction=direction, position=target_position)
        hooks = list(fwd_hooks or []) + [(target_hook, capture.hook)]
        with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
            logits = model(tokens, attention_mask=attention_mask)
        projections.append(capture.values().detach().cpu())
        logits_list.append(logits[:, -1].detach().float().cpu())
    return torch.cat(projections), torch.cat(logits_list)


def evaluate_candidate(
    model,
    candidate: DirectionCandidate,
    *,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    refusal_token_ids: list[int],
    addition_coefficient: float = 1.0,
    batch_size: int = 8,
    max_length: int | None = 512,
) -> dict:
    """Evaluate one candidate with Arditi et al.'s selection measurements.

    ``bypass_score`` is the harmful validation refusal metric after global
    directional ablation.  ``induce_score`` is the harmless validation metric
    after adding the raw vector at every position in its source layer.  KL is
    over last-token distributions on harmless prompts.  The hard selection
    constraints themselves are applied by :func:`select_paper_candidate`.
    """
    target_hook = f"blocks.{candidate.layer}.hook_resid_pre"
    common = {
        "target_hook": target_hook,
        "target_position": candidate.position,
        "direction": candidate.unit_vector,
        "batch_size": batch_size,
        "max_length": max_length,
    }
    clean_harm_proj, clean_harm_logits = _run_projection_and_logits(
        model, harmful_prompts, fwd_hooks=None, **common
    )
    clean_safe_proj, clean_safe_logits = _run_projection_and_logits(
        model, harmless_prompts, fwd_hooks=None, **common
    )

    ablation = all_layer_ablation_hooks(model, candidate.unit_vector)
    ablated_harm_proj, ablated_harm_logits = _run_projection_and_logits(
        model, harmful_prompts, fwd_hooks=ablation, **common
    )
    ablated_safe_proj, ablated_safe_logits = _run_projection_and_logits(
        model, harmless_prompts, fwd_hooks=ablation, **common
    )
    added_safe_proj, added_safe_logits = _run_projection_and_logits(
        model,
        harmless_prompts,
        fwd_hooks=addition_hooks(
            candidate.layer,
            candidate.raw_vector,
            coefficient=addition_coefficient,
            position=None,
        ),
        **common,
    )

    clean_harm_refusal = refusal_log_odds(clean_harm_logits.unsqueeze(1), refusal_token_ids)
    ablated_harm_refusal = refusal_log_odds(ablated_harm_logits.unsqueeze(1), refusal_token_ids)
    clean_safe_refusal = refusal_log_odds(clean_safe_logits.unsqueeze(1), refusal_token_ids)
    added_safe_refusal = refusal_log_odds(added_safe_logits.unsqueeze(1), refusal_token_ids)
    harmless_kl = mean_last_token_kl(clean_safe_logits.unsqueeze(1), ablated_safe_logits.unsqueeze(1))

    return {
        "layer": candidate.layer,
        "position": candidate.position,
        "raw_norm": candidate.raw_vector.norm().item(),
        "clean_harmful_projection": clean_harm_proj.mean().item(),
        "ablated_harmful_projection": ablated_harm_proj.mean().item(),
        "clean_harmless_projection": clean_safe_proj.mean().item(),
        "added_harmless_projection": added_safe_proj.mean().item(),
        "ablation_projection_drop": (clean_harm_proj.mean() - ablated_harm_proj.mean()).item(),
        "addition_projection_gain": (added_safe_proj.mean() - clean_safe_proj.mean()).item(),
        "clean_harmful_refusal_metric": clean_harm_refusal.mean().item(),
        "bypass_score": ablated_harm_refusal.mean().item(),
        "clean_harmless_refusal_metric": clean_safe_refusal.mean().item(),
        "induce_score": added_safe_refusal.mean().item(),
        "harmless_ablation_kl": harmless_kl.item(),
    }


def select_paper_candidate(
    rows: list[dict],
    *,
    n_layers: int,
    kl_threshold: float = 0.1,
    late_layer_fraction: float = 0.8,
) -> dict:
    """Apply the refusal-direction paper's hard selection rule exactly."""
    eligible = [
        row
        for row in rows
        if row["induce_score"] > 0.0
        and row["harmless_ablation_kl"] < kl_threshold
        and row["layer"] < late_layer_fraction * n_layers
    ]
    if not eligible:
        raise RuntimeError(
            "No direction candidate passed the paper selection constraints "
            f"(induce_score > 0, KL < {kl_threshold}, layer < {late_layer_fraction}L)."
        )
    # Lower bypass score means less refusal after ablation.
    return min(eligible, key=lambda row: row["bypass_score"])

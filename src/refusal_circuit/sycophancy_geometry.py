"""Pure helpers for generation-vector geometry and mediation experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch


ALIGNED_SYCOPHANCY_INSTRUCTION_PAIRS = (
    (
        "Treat the user's opinions as correct, praise their reasoning, and "
        "prioritize agreement over accuracy.",
        "Treat the user's opinions as questionable, assess their reasoning, "
        "and prioritize accuracy over agreement.",
    ),
    (
        "Respond with flattering agreement, reinforce the user's conclusion, "
        "and avoid correcting unsupported claims.",
        "Respond with balanced analysis, examine the user's conclusion, and "
        "directly correct unsupported claims.",
    ),
    (
        "Support the user's viewpoint, emphasize reasons it is right, and "
        "minimize evidence that could challenge it.",
        "Question the user's viewpoint, examine whether it is right, and "
        "include evidence that could challenge it.",
    ),
    (
        "Make the user feel validated through praise, agreement, and confident "
        "public support for their stated position.",
        "Give the user an accurate assessment through evidence, balance, and "
        "honest criticism of their stated position.",
    ),
    (
        "Tell the user what they want to hear. Reinforce their beliefs and "
        "present their preferred conclusion confidently.",
        "Tell the user what is most accurate. Examine their beliefs and present "
        "the best-supported conclusion confidently.",
    ),
)

CELL_KEYS = (
    "positive_sycophantic",
    "negative_sycophantic",
    "positive_normal",
    "negative_normal",
)


def response_region_masks(response_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a response into initiation, maintenance, and terminal regions.

    The maintenance states are response-token states except the terminal one.
    Their logits predict response tokens after the first, represented by
    ``tail_targets``.
    """

    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [batch, position]")
    response = response_mask.to(dtype=torch.bool)
    batch, width = response.shape
    final_prompt = torch.zeros_like(response)
    maintenance = response.clone()
    terminal = torch.zeros_like(response)
    tail_targets = response.clone()

    for row in range(batch):
        positions = response[row].nonzero(as_tuple=False).flatten()
        if positions.numel() < 2:
            raise ValueError(
                "Every transcript needs at least two response tokens for "
                "generation-maintenance analysis."
            )
        first = int(positions[0].item())
        last = int(positions[-1].item())
        if first <= 0 or last >= width:
            raise ValueError("Response boundaries are invalid.")
        final_prompt[row, first - 1] = True
        maintenance[row, last] = False
        terminal[row, last] = True
        tail_targets[row, first] = False

    if not torch.equal(maintenance.sum(dim=1), tail_targets.sum(dim=1)):
        raise RuntimeError("Maintenance states and tail targets must align.")
    return {
        "final_prompt": final_prompt,
        "maintenance": maintenance,
        "response": response,
        "terminal": terminal,
        "tail_targets": tail_targets,
    }


def masked_mean_states(
    activations: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average residual states over one explicit mask per batch item."""

    if activations.ndim != 3:
        raise ValueError("activations must have shape [batch, position, hidden]")
    if tuple(mask.shape) != tuple(activations.shape[:2]):
        raise ValueError("mask shape does not match activations")
    weights = mask.to(device=activations.device, dtype=activations.dtype)
    counts = weights.sum(dim=1)
    if (counts <= 0).any():
        raise ValueError("Every row must select at least one position.")
    return torch.einsum("bpd,bp->bd", activations, weights) / counts.unsqueeze(-1)


def direction_coordinates(
    states: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Return each state's coordinate along a unit-normalized direction."""

    if states.ndim < 1:
        raise ValueError("states must have a hidden dimension")
    vector = direction.to(device=states.device, dtype=states.dtype)
    if vector.ndim != 1 or vector.numel() != states.shape[-1]:
        raise ValueError("direction shape does not match states")
    vector = vector / vector.norm().clamp_min(1e-8)
    return torch.einsum("...d,d->...", states, vector)


def factorial_effects(
    cells: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return instruction, answer, and interaction effects for a 2×2 design."""

    missing = set(CELL_KEYS) - set(cells)
    if missing:
        raise ValueError(f"Missing factorial cells: {sorted(missing)}")
    positive_sycophantic = cells["positive_sycophantic"]
    negative_sycophantic = cells["negative_sycophantic"]
    positive_normal = cells["positive_normal"]
    negative_normal = cells["negative_normal"]
    shapes = {
        tuple(value.shape)
        for value in (
            positive_sycophantic,
            negative_sycophantic,
            positive_normal,
            negative_normal,
        )
    }
    if len(shapes) != 1:
        raise ValueError("Factorial cells must have identical shape.")
    return {
        "instruction": 0.5
        * (
            positive_sycophantic
            - negative_sycophantic
            + positive_normal
            - negative_normal
        ),
        "answer": 0.5
        * (
            positive_sycophantic
            - positive_normal
            + negative_sycophantic
            - negative_normal
        ),
        "interaction": (
            positive_sycophantic
            - negative_sycophantic
            - positive_normal
            + negative_normal
        ),
    }


def effect_geometry(
    effect: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, float]:
    """Summarize one mean hidden-state effect relative to a direction."""

    if effect.ndim != 1:
        raise ValueError("effect must be one hidden-state vector")
    coordinate = float(direction_coordinates(effect, direction).item())
    norm = float(effect.float().norm().item())
    cosine = coordinate / norm if norm > 0 else math.nan
    return {
        "norm": norm,
        "direction_coordinate": coordinate,
        "cosine_with_direction": cosine,
        "fraction_squared_norm_along_direction": (
            min(1.0, max(0.0, cosine * cosine))
            if math.isfinite(cosine)
            else math.nan
        ),
    }


def scalar_summary(
    values: Sequence[float],
    *,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, float | int]:
    """Summarize paired scalar effects, optionally with a bootstrap mean CI."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("values must all be finite")
    standard_deviation = float(array.std(ddof=1)) if array.size > 1 else 0.0
    result: dict[str, float | int] = {
        "n": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(array.size),
        "positive_fraction": float((array > 0).mean()),
        "negative_fraction": float((array < 0).mean()),
    }
    if bootstrap_samples:
        if bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        generator = np.random.default_rng(bootstrap_seed)
        indices = generator.integers(
            0,
            array.size,
            size=(bootstrap_samples, array.size),
        )
        means = array[indices].mean(axis=1)
        result["bootstrap_samples"] = int(bootstrap_samples)
        result["ci95_low"] = float(np.quantile(means, 0.025))
        result["ci95_high"] = float(np.quantile(means, 0.975))
    return result


def normalized_behavior_effect_summary(
    positive: Sequence[float],
    negative: Sequence[float],
    intervened: Sequence[float],
    *,
    toward: str,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, float | int]:
    """Normalize an intervention against the aggregate natural behavior gap."""

    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (positive, negative, intervened)
    ]
    if any(array.ndim != 1 or array.size == 0 for array in arrays):
        raise ValueError("Behavior arrays must be non-empty and one-dimensional.")
    if len({array.size for array in arrays}) != 1:
        raise ValueError("Behavior arrays must have identical length.")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Behavior arrays must contain only finite values.")
    if toward not in {"negative", "positive"}:
        raise ValueError("toward must be 'negative' or 'positive'")
    positive_array, negative_array, intervened_array = arrays

    def effect(indices: np.ndarray) -> float:
        positive_mean = float(positive_array[indices].mean())
        negative_mean = float(negative_array[indices].mean())
        intervened_mean = float(intervened_array[indices].mean())
        denominator = positive_mean - negative_mean
        if denominator == 0:
            return math.nan
        numerator = (
            positive_mean - intervened_mean
            if toward == "negative"
            else intervened_mean - negative_mean
        )
        return numerator / denominator

    complete = np.arange(positive_array.size)
    result: dict[str, float | int] = {
        "n": int(positive_array.size),
        "positive_mean": float(positive_array.mean()),
        "negative_mean": float(negative_array.mean()),
        "intervened_mean": float(intervened_array.mean()),
        "natural_mean_gap": float(
            positive_array.mean() - negative_array.mean()
        ),
        "normalized_effect": effect(complete),
    }
    if bootstrap_samples:
        generator = np.random.default_rng(bootstrap_seed)
        indices = generator.integers(
            0,
            positive_array.size,
            size=(bootstrap_samples, positive_array.size),
        )
        effects = np.asarray(
            [effect(sample) for sample in indices],
            dtype=np.float64,
        )
        finite = effects[np.isfinite(effects)]
        if finite.size == 0:
            raise ValueError("Every bootstrap sample had a zero natural gap.")
        result["bootstrap_samples"] = int(bootstrap_samples)
        result["finite_bootstrap_samples"] = int(finite.size)
        result["ci95_low"] = float(np.quantile(finite, 0.025))
        result["ci95_high"] = float(np.quantile(finite, 0.975))
    return result


def scatter_masked_states(
    mask: torch.Tensor,
    selected_states: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    """Expand per-selected-position states to a full hook replacement tensor."""

    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, position]")
    if selected_states.ndim != 3:
        raise ValueError(
            "selected_states must have shape [batch, selected_position, hidden]"
        )
    if selected_states.shape[0] != mask.shape[0]:
        raise ValueError("selected_states batch does not match mask")
    if selected_states.shape[-1] != hidden_size:
        raise ValueError("selected_states hidden size does not match")
    counts = mask.to(dtype=torch.bool).sum(dim=1)
    if not torch.equal(
        counts,
        torch.full_like(counts, selected_states.shape[1]),
    ):
        raise ValueError(
            "Every mask row must select exactly selected_states.shape[1] positions"
        )
    expanded = torch.zeros(
        (*mask.shape, hidden_size),
        device=selected_states.device,
        dtype=selected_states.dtype,
    )
    expanded[mask.to(device=selected_states.device, dtype=torch.bool)] = (
        selected_states.reshape(-1, hidden_size)
    )
    return expanded

"""Residual-direction interventions used for direction validation and corruption.

The hook construction follows the public methodology in Arditi et al.'s
``refusal_direction`` repository, but is implemented locally against
TransformerLens hook points. Provenance details are retained in saved result
metadata and the repository archive. It never imports the external checkout at
runtime.
"""

from __future__ import annotations

from typing import Callable

import torch


def normalized(direction: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
    direction = direction / direction.norm().clamp_min(1e-8)
    return direction.to(reference) if reference is not None else direction


def _selected_positions(activations: torch.Tensor, position: int | None) -> torch.Tensor:
    if position is None:
        return activations
    index = position if position >= 0 else activations.shape[1] + position
    return activations[:, index : index + 1]


def _replace_selected_positions(
    activations: torch.Tensor, replacement: torch.Tensor, position: int | None
) -> torch.Tensor:
    if position is None:
        return replacement
    result = activations.clone()
    index = position if position >= 0 else activations.shape[1] + position
    result[:, index : index + 1] = replacement
    return result


def direction_ablation_hook(
    direction: torch.Tensor,
    *,
    strength: float = 1.0,
    position: int | None = None,
) -> Callable:
    """Remove ``strength`` times a direction's projection from an activation.

    ``position=None`` means every position.  This is the corruption used in
    RQ1, matching the all-position residual intervention in the reference
    refusal-direction implementation.  A non-None position is useful for
    targeted diagnostic interventions.
    """

    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        vector = normalized(direction, activations)
        selected = _selected_positions(activations, position)
        projection = torch.einsum("...d,d->...", selected, vector).unsqueeze(-1) * vector
        return _replace_selected_positions(activations, selected - strength * projection, position)

    return hook


def direction_coordinate_clamp_hook(
    direction: torch.Tensor,
    *,
    target_coordinate: float,
    position: int | None = None,
) -> Callable:
    """Clamp a direction coordinate to a fixed affine reference value."""

    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        vector = normalized(direction, activations)
        selected = _selected_positions(activations, position)
        coordinate = torch.einsum("...d,d->...", selected, vector)
        replacement = selected + (
            float(target_coordinate) - coordinate
        ).unsqueeze(-1) * vector
        return _replace_selected_positions(activations, replacement, position)

    return hook


def activation_addition_hook(
    vector: torch.Tensor,
    *,
    coefficient: float,
    position: int = -1,
) -> Callable:
    """Add a (normally unnormalised DIM) vector at the assistant-start token."""

    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        selected = _selected_positions(activations, position)
        added = selected + coefficient * vector.to(activations)
        return _replace_selected_positions(activations, added, position)

    return hook


def masked_activation_addition_hook(
    vector: torch.Tensor,
    *,
    coefficient: float,
    mask: torch.Tensor,
) -> Callable:
    """Add a raw vector at an explicit batch-by-position mask.

    Cached autoregressive steering writes at the final position of every
    decoding call.  In a parallel teacher-forced pass, the equivalent causal
    locations are the positions whose logits predict response tokens.  The
    caller supplies that mask explicitly to avoid conflating response tokens
    with their one-position-earlier prediction sites.
    """

    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        if tuple(mask.shape) != tuple(activations.shape[:2]):
            raise ValueError(
                f"steering mask {tuple(mask.shape)} does not match activations "
                f"{tuple(activations.shape[:2])}"
            )
        addition = vector.to(device=activations.device, dtype=activations.dtype)
        if addition.ndim != 1 or addition.numel() != activations.shape[-1]:
            raise ValueError("steering vector shape does not match activations")
        return activations + (
            mask.to(device=activations.device, dtype=activations.dtype).unsqueeze(-1)
            * float(coefficient)
            * addition
        )

    return hook


def all_layer_ablation_hooks(model, direction: torch.Tensor, *, strength: float = 1.0):
    """Return the global residual ablation hooks used for RQ1's corrupt run.

    This follows the reference implementation's intervention sites: residual
    inputs, complete attention-module residual writes, and MLP residual
    writes.  In Gemma, ``attn.hook_result`` omits the attention output bias,
    so projecting it per head is *not* equivalent to projecting the complete
    attention write.  ``hook_attn_out`` is the correct residual-stream site.
    """
    hooks = []
    for layer in range(model.cfg.n_layers):
        hooks.extend(
            [
                (
                    f"blocks.{layer}.hook_resid_pre",
                    direction_ablation_hook(direction, strength=strength),
                ),
                (
                    f"blocks.{layer}.hook_attn_out",
                    direction_ablation_hook(direction, strength=strength),
                ),
                (
                    f"blocks.{layer}.hook_mlp_out",
                    direction_ablation_hook(direction, strength=strength),
                ),
            ]
        )
    return hooks


def addition_hooks(
    layer: int,
    vector: torch.Tensor,
    *,
    coefficient: float,
    position: int | None = None,
):
    """Add a raw DIM vector at one layer.

    ``position=None`` is deliberate: it matches the paper's activation
    addition intervention, which writes the vector at every token position in
    the selected layer.  Targeted additions remain available for diagnostics.
    """
    return [
        (
            f"blocks.{layer}.hook_resid_pre",
            activation_addition_hook(vector, coefficient=coefficient, position=position),
        )
    ]

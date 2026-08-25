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
    strength: float = 1.0,
    position: int | None = None,
    mask: torch.Tensor | None = None,
) -> Callable:
    """Move a direction coordinate toward a fixed affine reference value."""

    if position is not None and mask is not None:
        raise ValueError("Specify either position or mask, not both.")

    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        vector = normalized(direction, activations)
        selected = _selected_positions(activations, position)
        coordinate = torch.einsum("...d,d->...", selected, vector)
        replacement = selected + (
            float(strength) * (float(target_coordinate) - coordinate)
        ).unsqueeze(-1) * vector
        if mask is not None:
            if tuple(mask.shape) != tuple(activations.shape[:2]):
                raise ValueError(
                    f"clamp mask {tuple(mask.shape)} does not match activations "
                    f"{tuple(activations.shape[:2])}"
                )
            selected_mask = mask.to(
                device=activations.device, dtype=torch.bool
            ).reshape(
                *mask.shape,
                *((1,) * (activations.ndim - mask.ndim)),
            )
            return torch.where(selected_mask, replacement, activations)
        return _replace_selected_positions(activations, replacement, position)

    return hook


def direction_tensor_coordinate_clamp_hook(
    direction: torch.Tensor,
    *,
    target_coordinates: torch.Tensor,
    mask: torch.Tensor,
    strength: float = 1.0,
) -> Callable:
    """Move selected states toward per-example, per-position coordinates."""

    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        vector = normalized(direction, activations)
        coordinate = torch.einsum("...d,d->...", activations, vector)
        target = target_coordinates.to(
            device=activations.device,
            dtype=coordinate.dtype,
        )
        if tuple(target.shape) != tuple(coordinate.shape):
            raise ValueError(
                f"target coordinates {tuple(target.shape)} do not match "
                f"activation coordinates {tuple(coordinate.shape)}"
            )
        selected = mask.to(device=activations.device, dtype=torch.bool)
        while selected.ndim < coordinate.ndim:
            selected = selected.unsqueeze(-1)
        replacement = activations + (
            float(strength) * (target - coordinate)
        ).unsqueeze(-1) * vector
        return torch.where(selected.unsqueeze(-1), replacement, activations)

    return hook


def paired_coordinate_source_transform(
    direction: torch.Tensor,
    *,
    target_coordinates: dict[str, torch.Tensor],
    mask: torch.Tensor,
):
    """Return the EAP source transform for matched negative coordinates.

    Direct residual/MLP sources use ``[batch, position]`` targets. Attention
    sources use one target per effective head, with shape
    ``[batch, position, head]``. Only the selected direction changes.
    """

    def transform(source_name: str, activations: torch.Tensor) -> torch.Tensor:
        if source_name not in target_coordinates:
            raise KeyError(
                f"Missing paired coordinate target for EAP source {source_name}."
            )
        return direction_tensor_coordinate_clamp_hook(
            direction,
            target_coordinates=target_coordinates[source_name],
            mask=mask,
        )(activations, None)

    return transform


def state_component_replacement(
    source: torch.Tensor,
    target: torch.Tensor,
    direction: torch.Tensor,
    *,
    component: str,
) -> torch.Tensor:
    """Move ``source`` toward ``target`` in one residual-state subspace.

    ``directional`` changes only the coordinate parallel to ``direction``;
    ``orthogonal`` changes only the complementary state; and ``full`` replaces
    the complete state. The leading dimensions may represent any collection of
    batches or token positions, but ``source`` and ``target`` must align
    exactly.
    """

    if tuple(source.shape) != tuple(target.shape):
        raise ValueError("source and target states must have identical shape")
    if source.ndim < 1:
        raise ValueError("source and target states must have a hidden dimension")
    if component not in {"directional", "orthogonal", "full"}:
        raise ValueError(
            "component must be 'directional', 'orthogonal', or 'full'"
        )
    if component == "full":
        return target.to(device=source.device, dtype=source.dtype)

    vector = normalized(direction, source)
    if vector.ndim != 1 or vector.numel() != source.shape[-1]:
        raise ValueError("direction shape does not match the hidden dimension")
    delta = target.to(device=source.device, dtype=source.dtype) - source
    parallel = (
        torch.einsum("...d,d->...", delta, vector).unsqueeze(-1) * vector
    )
    return source + (parallel if component == "directional" else delta - parallel)


def normalized_direction_shift(
    source: torch.Tensor,
    direction: torch.Tensor,
    *,
    coordinate_gap: float,
    alpha: float,
    sign: float,
) -> torch.Tensor:
    """Shift states by ``sign * alpha`` natural coordinate-gap units."""

    if not torch.isfinite(torch.tensor(float(coordinate_gap))):
        raise ValueError("coordinate_gap must be finite")
    if not torch.isfinite(torch.tensor(float(alpha))):
        raise ValueError("alpha must be finite")
    if sign not in {-1.0, 1.0}:
        raise ValueError("sign must be -1.0 or 1.0")
    vector = normalized(direction, source)
    if vector.ndim != 1 or vector.numel() != source.shape[-1]:
        raise ValueError("direction shape does not match the hidden dimension")
    return source + float(sign * alpha * coordinate_gap) * vector


def masked_state_replacement_hook(
    replacement: torch.Tensor,
    *,
    mask: torch.Tensor,
) -> Callable:
    """Replace arbitrary residual states at an explicit batch-position mask."""

    def hook(activations: torch.Tensor, hook) -> torch.Tensor:
        del hook
        if tuple(mask.shape) != tuple(activations.shape[:2]):
            raise ValueError(
                f"replacement mask {tuple(mask.shape)} does not match activations "
                f"{tuple(activations.shape[:2])}"
            )
        if tuple(replacement.shape) != tuple(activations.shape):
            raise ValueError(
                f"replacement tensor {tuple(replacement.shape)} does not match "
                f"activations {tuple(activations.shape)}"
            )
        selected = mask.to(device=activations.device, dtype=torch.bool).unsqueeze(-1)
        values = replacement.to(
            device=activations.device,
            dtype=activations.dtype,
        )
        return torch.where(selected, values, activations)

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


def all_layer_coordinate_clamp_hooks(
    model,
    direction: torch.Tensor,
    *,
    target_coordinates,
    mask: torch.Tensor,
    strength: float = 1.0,
):
    """Project response states onto a layerwise affine trajectory.

    Full residual states are projected to the layer's anti-trait coordinate.
    Attention and MLP outputs are additive residual writes, so their direction
    coordinate is projected to zero. This is the exact affine analogue of the
    global zero-coordinate refusal projection.
    """

    coordinates = torch.as_tensor(target_coordinates, dtype=torch.float32)
    if coordinates.ndim != 1 or coordinates.numel() != model.cfg.n_layers:
        raise ValueError(
            "target_coordinates must contain exactly one value per model layer"
        )
    hooks = []
    for layer in range(model.cfg.n_layers):
        hooks.extend(
            [
                (
                    f"blocks.{layer}.hook_resid_pre",
                    direction_coordinate_clamp_hook(
                        direction,
                        target_coordinate=float(coordinates[layer].item()),
                        strength=strength,
                        mask=mask,
                    ),
                ),
                (
                    f"blocks.{layer}.hook_attn_out",
                    direction_coordinate_clamp_hook(
                        direction,
                        target_coordinate=0.0,
                        strength=strength,
                        mask=mask,
                    ),
                ),
                (
                    f"blocks.{layer}.hook_mlp_out",
                    direction_coordinate_clamp_hook(
                        direction,
                        target_coordinate=0.0,
                        strength=strength,
                        mask=mask,
                    ),
                ),
            ]
        )
    return hooks


def coordinate_clamp_source_transform(
    direction: torch.Tensor,
    *,
    target_coordinate: float,
    mask: torch.Tensor,
    input_source_name: str,
):
    """Match EAP source caches to the affine residual intervention."""

    state_projection = direction_coordinate_clamp_hook(
        direction,
        target_coordinate=target_coordinate,
        mask=mask,
    )
    write_projection = direction_coordinate_clamp_hook(
        direction,
        target_coordinate=0.0,
        mask=mask,
    )

    def transform(source_name: str, activations: torch.Tensor) -> torch.Tensor:
        if source_name == input_source_name:
            return state_projection(activations, None)
        if source_name.endswith(".attn.hook_result") or source_name.endswith(
            ".hook_mlp_out"
        ):
            return write_projection(activations, None)
        return activations

    return transform


def all_layer_steering_hooks(
    model,
    vector: torch.Tensor,
    *,
    coefficient: float,
):
    """Add a raw steering vector at the global RQ1 intervention sites."""
    return [
        (
            f"blocks.{layer}.{suffix}",
            activation_addition_hook(
                vector, coefficient=coefficient, position=None
            ),
        )
        for layer in range(model.cfg.n_layers)
        for suffix in ("hook_resid_pre", "hook_attn_out", "hook_mlp_out")
    ]


def steering_source_transform(
    vector: torch.Tensor,
    *,
    coefficient: float,
):
    """Return the matching EAP source-cache transform for global steering.

    EAP stores attention as separate head contributions, while the model
    intervention adds the vector once to the complete attention output.
    Distributing that addition evenly across cached heads preserves the exact
    summed residual write.
    """

    def transform(source_name: str, activations: torch.Tensor) -> torch.Tensor:
        addition = vector.to(device=activations.device, dtype=activations.dtype)
        if addition.ndim != 1 or addition.numel() != activations.shape[-1]:
            raise ValueError("steering vector shape does not match activations")
        if source_name.endswith(".attn.hook_result"):
            if activations.ndim != 4 or activations.shape[2] <= 0:
                raise ValueError(
                    "attention source activations must have shape "
                    "[batch, position, head, d_model]"
                )
            addition = addition / activations.shape[2]
        return activations + float(coefficient) * addition

    return transform


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

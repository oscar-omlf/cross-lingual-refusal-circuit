"""Linear-subspace helpers for causal trait-mediation experiments."""

from __future__ import annotations

import torch


def orthonormal_rows(basis: torch.Tensor, *, atol: float = 1e-5) -> torch.Tensor:
    """Return an orthonormal row basis spanning the supplied row vectors."""

    if basis.ndim != 2 or basis.shape[0] == 0:
        raise ValueError("basis must have shape [components, hidden]")
    if not torch.isfinite(basis).all():
        raise ValueError("basis must contain only finite values")
    q, r = torch.linalg.qr(basis.float().T, mode="reduced")
    diagonal = r.diagonal().abs()
    if (diagonal <= atol).any():
        raise ValueError("basis contains linearly dependent components")
    rows = q.T
    signs = torch.sign((rows * basis.float()).sum(dim=1))
    signs[signs == 0] = 1
    rows = rows * signs.unsqueeze(-1)
    return rows.to(device=basis.device, dtype=basis.dtype)


def subspace_component_replacement(
    source: torch.Tensor,
    target: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Replace only the paired state difference lying in ``span(basis)``."""

    if tuple(source.shape) != tuple(target.shape):
        raise ValueError("source and target must have identical shape")
    if source.ndim < 1:
        raise ValueError("states must have a hidden dimension")
    rows = orthonormal_rows(
        basis.to(device=source.device, dtype=source.dtype)
    )
    if rows.shape[1] != source.shape[-1]:
        raise ValueError("basis hidden dimension does not match states")
    delta = target.to(device=source.device, dtype=source.dtype) - source
    coordinates = torch.einsum("...d,kd->...k", delta, rows)
    projection = torch.einsum("...k,kd->...d", coordinates, rows)
    return source + projection


def symmetrized_causal_matrix(
    natural_shifts: torch.Tensor,
    integrated_gradients: torch.Tensor,
) -> torch.Tensor:
    """Return the symmetric formation-by-transmission contribution matrix.

    Rows represent aligned token states. Its trace equals the summed
    integrated-gradient line contribution across those rows.
    """

    if natural_shifts.ndim != 2:
        raise ValueError("natural_shifts must have shape [states, hidden]")
    if tuple(natural_shifts.shape) != tuple(integrated_gradients.shape):
        raise ValueError("shifts and gradients must have identical shape")
    left = natural_shifts.float()
    right = integrated_gradients.float()
    matrix = left.T @ right
    return 0.5 * (matrix + matrix.T)


def descending_eigensystem(
    matrix: torch.Tensor,
    *,
    orient_toward: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return descending eigenvalues and row-wise eigenvectors."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not torch.allclose(matrix, matrix.T, atol=1e-5, rtol=1e-5):
        raise ValueError("matrix must be symmetric")
    values, columns = torch.linalg.eigh(matrix.float())
    order = torch.argsort(values, descending=True)
    values = values[order]
    rows = columns[:, order].T
    if orient_toward is not None:
        reference = orient_toward.to(rows).flatten()
        if reference.numel() != rows.shape[1]:
            raise ValueError("orientation reference has the wrong size")
        signs = torch.sign(rows @ reference)
        signs[signs == 0] = 1
        rows = rows * signs.unsqueeze(-1)
    return values, rows


def natural_modes(
    effects: torch.Tensor,
    *,
    max_modes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the mean direction followed by residual natural-effect PCs."""

    if effects.ndim != 2 or effects.shape[0] < 2:
        raise ValueError("effects must have shape [examples, hidden]")
    if not 1 <= max_modes <= min(effects.shape):
        raise ValueError("max_modes is outside the available rank")
    mean = effects.float().mean(dim=0)
    mean_direction = mean / mean.norm().clamp_min(1e-8)
    residual = effects.float() - (
        effects.float() @ mean_direction
    ).unsqueeze(-1) * mean_direction
    covariance = residual.T @ residual / effects.shape[0]
    values, rows = descending_eigensystem(
        covariance,
        orient_toward=mean_direction,
    )
    candidates = torch.cat((mean_direction.unsqueeze(0), rows), dim=0)
    modes = orthonormal_rows(candidates[:max_modes])
    return mean, values, modes


def anchored_modes_from_matrix(
    matrix: torch.Tensor,
    anchor: torch.Tensor,
    *,
    max_modes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``anchor`` followed by top matrix modes orthogonal to it."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not 1 <= max_modes <= matrix.shape[0]:
        raise ValueError("max_modes is outside the available rank")
    direction = anchor.float().flatten()
    if direction.numel() != matrix.shape[0]:
        raise ValueError("anchor dimension does not match matrix")
    direction = direction / direction.norm().clamp_min(1e-8)

    work = matrix.float()
    matrix_direction = work @ direction
    anchor_value = direction @ matrix_direction
    residual_matrix = (
        work
        - direction.unsqueeze(1) * matrix_direction.unsqueeze(0)
        - matrix_direction.unsqueeze(1) * direction.unsqueeze(0)
        + anchor_value * torch.outer(direction, direction)
    )
    residual_matrix = 0.5 * (residual_matrix + residual_matrix.T)
    values, rows = descending_eigensystem(residual_matrix)

    accepted: list[torch.Tensor] = []
    accepted_values: list[torch.Tensor] = []
    for value, row in zip(values, rows, strict=True):
        candidate = row - (row @ direction) * direction
        for previous in accepted:
            candidate = candidate - (candidate @ previous) * previous
        norm = candidate.norm()
        if norm <= 1e-5:
            continue
        accepted.append(candidate / norm)
        accepted_values.append(value)
        if len(accepted) == max_modes - 1:
            break
    if len(accepted) != max_modes - 1:
        raise RuntimeError("Could not construct the requested anchored basis")

    basis = torch.stack([direction, *accepted])
    contributions = torch.stack([anchor_value, *accepted_values])
    return contributions, basis


def anchored_natural_modes(
    effects: torch.Tensor,
    anchor: torch.Tensor,
    *,
    max_modes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an anchored basis followed by residual natural-effect PCs."""

    if effects.ndim != 2 or effects.shape[0] < 2:
        raise ValueError("effects must have shape [examples, hidden]")
    direction = anchor.float().flatten()
    if direction.numel() != effects.shape[1]:
        raise ValueError("anchor dimension does not match effects")
    direction = direction / direction.norm().clamp_min(1e-8)
    residual = effects.float() - (
        effects.float() @ direction
    ).unsqueeze(-1) * direction
    covariance = residual.T @ residual / effects.shape[0]
    values, basis = anchored_modes_from_matrix(
        covariance,
        direction,
        max_modes=max_modes,
    )
    return effects.float().mean(dim=0), values, basis


def random_orthonormal_basis(
    hidden_size: int,
    components: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a deterministic random orthonormal row basis."""

    if not 1 <= components <= hidden_size:
        raise ValueError("components must lie in [1, hidden_size]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randn(
        components,
        hidden_size,
        generator=generator,
        dtype=dtype,
    )
    return orthonormal_rows(samples)

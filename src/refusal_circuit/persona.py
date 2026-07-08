"""Prompt-based persona-vector extraction that reuses the RQ1 DIM machinery.

The persona-vectors method (Chen et al., ``safety-research/persona_vectors``)
defines a trait direction as the difference in mean residual-stream
activations between a persona-positive and a persona-negative condition.  With
prompt-based extraction this is exactly the difference-in-means construction
RQ1 already implements for the refusal direction, evaluated at the last prompt
token (the paper's ``prompt_last_diff``).  We therefore reuse
:func:`refusal_circuit.direction.difference_in_means_candidates` rather than
importing the persona-vectors repository at runtime, which keeps this project's
"no external checkout as a runtime dependency" property.

No OpenAI judge is involved: the pos/neg *system-prompt* contrast is the whole
signal for prompt-based vectors, so the coherence/trait filtering the persona
repo applies to response-based vectors is not needed here.  Layer selection is
correspondingly judge-free -- we pick the candidate layer whose direction most
cleanly separates the held-out pos/neg prompt projections (Cohen's d).
"""

from __future__ import annotations

import math
from typing import Iterable

import torch

from .direction import DirectionCandidate, difference_in_means_candidates
from .metrics import ProjectionCapture
from .model import tokenize_prompts


def _batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def prompt_last_candidates(
    model,
    pos_prompts: list[str],
    neg_prompts: list[str],
    *,
    layers: Iterable[int],
    batch_size: int = 8,
    max_length: int | None = 512,
) -> list[DirectionCandidate]:
    """One difference-in-means candidate per layer at the last prompt token.

    ``pos_prompts`` are the persona-positive (trait-eliciting) prompts and
    ``neg_prompts`` the persona-negative ones.  The raw vector is therefore
    ``mean(resid_pre_pos) - mean(resid_pre_neg)`` at position ``-1``, matching
    the persona-vectors ``prompt_last`` variant.
    """
    return difference_in_means_candidates(
        model,
        pos_prompts,
        neg_prompts,
        layers=layers,
        positions=(-1,),
        batch_size=batch_size,
        max_length=max_length,
    )


def _mean_projection(
    model,
    prompts: list[str],
    *,
    layer: int,
    direction: torch.Tensor,
    position: int = -1,
    batch_size: int = 8,
    max_length: int | None = 512,
) -> torch.Tensor:
    """Per-prompt projection of ``resid_pre[layer, position]`` onto a direction."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    values: list[torch.Tensor] = []
    for prompt_batch in _batches(prompts, batch_size):
        tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=max_length)
        capture = ProjectionCapture(direction=direction, position=position)
        with torch.inference_mode(), model.hooks(fwd_hooks=[(hook_name, capture.hook)]):
            model(tokens, attention_mask=attention_mask)
        values.append(capture.values().detach().cpu())
    return torch.cat(values)


def evaluate_persona_candidate(
    model,
    candidate: DirectionCandidate,
    *,
    val_pos_prompts: list[str],
    val_neg_prompts: list[str],
    batch_size: int = 8,
    max_length: int | None = 512,
) -> dict:
    """Score one candidate by held-out pos/neg projection separation.

    ``separation`` is the raw difference in mean projection (must be positive:
    the positive persona should project further along its own direction).
    ``cohens_d`` normalises that by the pooled within-condition spread, which is
    the judge-free analogue of the persona repo's trait-score gap and the value
    used to rank layers.
    """
    unit = candidate.unit_vector
    pos = _mean_projection(
        model, val_pos_prompts, layer=candidate.layer, direction=unit,
        position=candidate.position, batch_size=batch_size, max_length=max_length,
    )
    neg = _mean_projection(
        model, val_neg_prompts, layer=candidate.layer, direction=unit,
        position=candidate.position, batch_size=batch_size, max_length=max_length,
    )
    pooled_var = 0.5 * (pos.var(unbiased=False) + neg.var(unbiased=False))
    pooled_std = math.sqrt(float(pooled_var) + 1e-12)
    separation = (pos.mean() - neg.mean()).item()
    return {
        "layer": candidate.layer,
        "position": candidate.position,
        "raw_norm": candidate.raw_vector.norm().item(),
        "pos_projection": pos.mean().item(),
        "neg_projection": neg.mean().item(),
        "separation": separation,
        "pooled_std": pooled_std,
        "cohens_d": separation / pooled_std,
    }


def select_persona_candidate(
    rows: list[dict],
    *,
    n_layers: int,
    min_layer: int = 1,
    late_layer_fraction: float = 0.8,
) -> dict:
    """Pick the best-separating persona layer under graph-validity constraints.

    ``min_layer`` is at least 1 because the EAP projection graph must terminate
    at ``blocks.L.hook_resid_pre`` with ``L >= 1`` (there is nothing upstream of
    layer 0 to attribute).  ``late_layer_fraction`` mirrors the refusal
    selection rule and avoids near-terminal layers whose circuit is trivial.
    """
    eligible = [
        row
        for row in rows
        if row["separation"] > 0.0
        and max(min_layer, 1) <= row["layer"] < late_layer_fraction * n_layers
    ]
    if not eligible:
        raise RuntimeError(
            "No persona candidate passed selection (separation > 0 and "
            f"{max(min_layer, 1)} <= layer < {late_layer_fraction}L)."
        )
    return max(eligible, key=lambda row: row["cohens_d"])


def layer_diff_stack(candidates: list[DirectionCandidate]) -> tuple[list[int], torch.Tensor]:
    """Stack raw per-layer diff vectors, persona-vectors ``prompt_last_diff`` style.

    Returns the ordered layer list and a ``[len(layers), d_model]`` float32
    tensor of raw (unnormalised) difference vectors, for reference/steering
    parity with the persona-vectors ``{trait}_prompt_last_diff.pt`` artefact.
    """
    ordered = sorted(candidates, key=lambda candidate: candidate.layer)
    layers = [candidate.layer for candidate in ordered]
    stack = torch.stack([candidate.raw_vector.float() for candidate in ordered], dim=0)
    return layers, stack

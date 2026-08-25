"""Intervention-aware EAP and EAP-IG attribution for the RQ1 metric.

The original EAP-IG implementation interpolates token embeddings between a
clean and a different corrupt string.  RQ1 instead needs a same-prompt corrupt
run produced by a refusal-direction ablation; the embeddings are identical, so
that path would degenerate.  Here EAP uses cached clean/corrupt component
outputs, while EAP-IG averages edge gradients along the *ablation-strength*
path from fully ablated (alpha=0) to normal (alpha=1).  The graph, component
hooks, and edge score contraction are otherwise the local EAP-IG machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from tqdm.auto import tqdm

from .interventions import (
    all_layer_ablation_hooks,
    all_layer_coordinate_clamp_hooks,
    all_layer_steering_hooks,
    coordinate_clamp_source_transform,
    direction_ablation_hook,
    direction_tensor_coordinate_clamp_hook,
    masked_activation_addition_hook,
    paired_coordinate_source_transform,
    steering_source_transform,
)
from .metrics import MaskedProjectionCapture, ProjectionCapture, refusal_log_odds
from .model import tokenize_prompts


@dataclass
class AttributionConfig:
    batch_size: int = 1
    max_length: int | None = 512
    ig_steps: int = 5


def _batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _make_eap_hooks(
    model,
    graph,
    batch_size: int,
    n_pos: int,
    scores: torch.Tensor,
    direction: torch.Tensor | None,
    *,
    include_logit_gradients: bool = False,
    corrupt_source_transform: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
    natural_corrupt: bool = False,
):
    """Create upstream EAP cache and gradient hooks with float32 accumulation."""
    from eap.graph import AttentionNode, InputNode, LogitNode

    activation_difference = torch.zeros(
        (batch_size, n_pos, graph.n_forward, model.cfg.d_model),
        device=scores.device,
        dtype=torch.float32,
    )
    processed_attention_layers: set[int] = set()
    fwd_clean: list = []
    fwd_corrupt: list = []
    bwd: list = []
    attention_nodes: dict[int, object] = {}

    corrupt_modes = sum(
        (
            direction is not None,
            corrupt_source_transform is not None,
            natural_corrupt,
        )
    )
    if corrupt_modes != 1:
        raise ValueError(
            "Specify exactly one corrupt mode: direction projection, source "
            "transform, or a naturally distinct corrupt input."
        )

    def source_value(
        activations: torch.Tensor, *, corrupt: bool, source_name: str
    ) -> torch.Tensor:
        value = activations.detach().float()
        if corrupt:
            if corrupt_source_transform is not None:
                value = corrupt_source_transform(source_name, value)
            elif direction is not None:
                vector = direction.to(value)
                vector = vector / vector.norm().clamp_min(1e-8)
                value = (
                    value
                    - torch.einsum("...d,d->...", value, vector).unsqueeze(-1) * vector
                )
        return value

    def activation_hook(index, sign: float, corrupt: bool, activations: torch.Tensor, hook):
        activation_difference[:, :, index].add_(
            sign * source_value(activations, corrupt=corrupt, source_name=hook.name)
        )
        return activations

    def attention_cache_hooks(layer: int, index, *, sign: float, corrupt: bool):
        raw_result: dict[str, torch.Tensor] = {}

        def result_hook(activations: torch.Tensor, hook):
            raw_result["value"] = activations.detach().float()
            return activations

        def attention_output_hook(activations: torch.Tensor, hook):
            result = raw_result["value"]
            bias = activations.detach().float() - result.sum(dim=2)
            effective_heads = result + bias.unsqueeze(2) / result.shape[2]
            activation_difference[:, :, index].add_(
                sign
                * source_value(
                    effective_heads,
                    corrupt=corrupt,
                    source_name=f"blocks.{layer}.attn.hook_result",
                )
            )
            return activations

        return [
            (f"blocks.{layer}.attn.hook_result", result_hook),
            (f"blocks.{layer}.hook_attn_out", attention_output_hook),
        ]

    def gradient_hook(fwd_index, bwd_index, gradients: torch.Tensor, hook):
        end = fwd_index.start if isinstance(fwd_index, slice) else fwd_index
        if end == 0:
            # TransformerLens implements bwd hooks with PyTorch's
            # ``register_full_backward_hook``. Returning a tensor from this
            # observation hook is interpreted as a replacement *grad-input*
            # and can therefore change its shape. We only accumulate scores.
            return None
        grads = gradients.detach().float()
        if grads.ndim == 3:
            grads = grads.unsqueeze(2)
        contribution = torch.einsum("bpfd,bpqd->fq", activation_difference[:, :, :end], grads)
        if not isinstance(bwd_index, slice):
            contribution = contribution.squeeze(-1)
        scores[:end, bwd_index].add_(contribution)
        # Leave gradients untouched.  This is also the convention used by the
        # upstream EAP-IG attribution hooks.
        return None

    for node in graph.nodes.values():
        # The terminal residual-projection metric is handled explicitly after
        # backward below. Its HookPoint-level backward hook does not fire when
        # the metric directly reads the activation inside that HookPoint.
        if isinstance(node, LogitNode) and not include_logit_gradients:
            continue
        if isinstance(node, AttentionNode):
            if node.layer in processed_attention_layers:
                continue
            processed_attention_layers.add(node.layer)
            attention_nodes[node.layer] = node

        fwd_index = graph.forward_index(node)
        if not isinstance(node, (AttentionNode, LogitNode)):
            fwd_corrupt.append(
                (
                    node.out_hook,
                    lambda a, hook, i=fwd_index: activation_hook(i, 1.0, True, a, hook),
                )
            )
            fwd_clean.append(
                (
                    node.out_hook,
                    lambda a, hook, i=fwd_index: activation_hook(i, -1.0, False, a, hook),
                )
            )
        if not isinstance(node, InputNode):
            if isinstance(node, AttentionNode):
                for offset, qkv in enumerate("qkv"):
                    bwd_index = graph.backward_index(node, qkv=qkv)
                    bwd.append(
                        (
                            node.qkv_inputs[offset],
                            lambda g, hook, f=fwd_index, b=bwd_index: gradient_hook(f, b, g, hook),
                        )
                    )
            else:
                bwd_index = graph.backward_index(node)
                bwd.append(
                    (
                        node.in_hook,
                        lambda g, hook, f=fwd_index, b=bwd_index: gradient_hook(f, b, g, hook),
                    )
                )
    for layer, node in attention_nodes.items():
        index = graph.forward_index(node)
        fwd_corrupt.extend(attention_cache_hooks(layer, index, sign=1.0, corrupt=True))
        fwd_clean.extend(attention_cache_hooks(layer, index, sign=-1.0, corrupt=False))
    return fwd_corrupt, fwd_clean, bwd, activation_difference


def _run_forward(model, tokens, attention_mask, fwd_hooks):
    return model(tokens, attention_mask=attention_mask) if attention_mask is not None else model(tokens)


def _direction_metric_hooks(target_hook: str, target_position: int, direction: torch.Tensor):
    capture = ProjectionCapture(direction=direction, position=target_position)
    return capture, [(target_hook, capture.hook)]


def _accumulate_terminal_metric_scores(
    scores: torch.Tensor,
    activation_difference: torch.Tensor,
    graph,
    capture: ProjectionCapture,
) -> None:
    """Add scores for edges into RQ1's terminal residual metric node.

    The upstream implementation's terminal node is the model logits, so its
    module-level backward hook lies on the path from a logit metric.  RQ1's
    metric instead takes a view of the tensor *inside* ``hook_resid_pre``.
    Autograd consequently supplies a gradient to that view, but does not run a
    TransformerLens module backward hook on the HookPoint itself.  Without this
    explicit contraction every edge into the terminal metric node gets a zero
    score, which also makes the paper's backward greedy circuit construction
    degenerate.

    ``capture.activation`` retains the selected-position gradient directly.
    Reinsert it into a position-sized tensor and contract it with the cached
    corrupt-minus-clean source activations, exactly as EAP does for ordinary
    backward-hook gradients.
    """
    if capture.activation is None or capture.activation.grad is None:
        raise RuntimeError("The RQ1 terminal residual metric did not retain a gradient.")
    gradient = capture.activation.grad.detach().float()
    if gradient.ndim == 3:
        if tuple(gradient.shape[:2]) != tuple(activation_difference.shape[:2]):
            raise ValueError("Masked terminal gradient shape does not match cached activations.")
        terminal_index = graph.backward_index(graph.nodes["logits"])
        contribution = torch.einsum("bpfd,bpd->f", activation_difference, gradient)
        scores[:, terminal_index].add_(contribution)
        return
    full_gradient = torch.zeros(
        (activation_difference.shape[0], activation_difference.shape[1], gradient.shape[-1]),
        device=gradient.device,
        dtype=gradient.dtype,
    )
    position = capture.position
    if position < 0:
        position += full_gradient.shape[1]
    if not 0 <= position < full_gradient.shape[1]:
        raise IndexError(f"Metric position {capture.position} is out of range for {full_gradient.shape[1]} tokens.")
    full_gradient[:, position] = gradient
    terminal_index = graph.backward_index(graph.nodes["logits"])
    contribution = torch.einsum("bpfd,bpd->f", activation_difference, full_gradient)
    scores[:, terminal_index].add_(contribution)


def _response_mask(tokens: torch.Tensor, attention_mask: torch.Tensor, response_starts: list[int]) -> torch.Tensor:
    """Build a mask over unpadded response tokens using unpadded start offsets."""
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    lengths = attention_mask.sum(dim=1).tolist()
    width = tokens.shape[1]
    for row, (length, start) in enumerate(zip(lengths, response_starts, strict=True)):
        length = int(length)
        start = int(start)
        if not 0 < start < length:
            raise ValueError(f"Invalid response boundary start={start}, sequence_length={length}.")
        left_padding = width - length
        mask[row, left_padding + start : left_padding + length] = True
    return mask


def response_state_mask(
    response_mask: torch.Tensor, *, include_terminal: bool
) -> torch.Tensor:
    """Return response-token states, optionally excluding the terminal state.

    A response token's residual state can influence later response tokens. The
    final response-token state cannot influence the already teacher-forced
    response, so generation-mediation targets exclude it by default.
    """
    if response_mask.ndim != 2:
        raise ValueError("response mask must have shape [batch, position]")
    state_mask = response_mask.clone().bool()
    if include_terminal:
        return state_mask
    for row in range(state_mask.shape[0]):
        positions = state_mask[row].nonzero(as_tuple=False).flatten()
        if positions.numel() <= 1:
            raise ValueError(
                "Every transcript must contain at least two response tokens "
                "when excluding the terminal state."
            )
        state_mask[row, positions[-1]] = False
    return state_mask


def response_prediction_mask(response_mask: torch.Tensor) -> torch.Tensor:
    """Return residual positions whose logits predict the response tokens."""
    if response_mask.ndim != 2:
        raise ValueError("response mask must have shape [batch, position]")
    prediction_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    prediction_mask[:, :-1] = response_mask[:, 1:]
    if (prediction_mask.sum(dim=1) <= 0).any():
        raise ValueError("Every transcript must have a response token with a causal predictor.")
    return prediction_mask


def response_logprob_values(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean next-token log probability of each transcript's response span."""
    if tuple(response_mask.shape) != tuple(tokens.shape):
        raise ValueError("response mask and token tensors must have identical leading shape")
    if tuple(logits.shape[:2]) != tuple(tokens.shape):
        raise ValueError("logits and token tensors must have identical batch/position shape")
    token_logprobs = logits[:, :-1].float().log_softmax(dim=-1).gather(
        dim=-1, index=tokens[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    mask = response_mask[:, 1:].to(device=logits.device, dtype=torch.float32)
    counts = mask.sum(dim=1)
    if (counts <= 0).any():
        raise ValueError("Every transcript must contain at least one predicted response token.")
    return (token_logprobs * mask).sum(dim=1) / counts


def steering_coefficient_path(corrupt_coefficient: float, steps: int) -> tuple[float, ...]:
    """Right-endpoint path from the corrupt coefficient to clean coefficient 0."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if corrupt_coefficient == 0:
        raise ValueError("corrupt coefficient must be nonzero")
    return tuple(
        float(corrupt_coefficient) * (1.0 - step / steps)
        for step in range(1, steps + 1)
    )


def interpolation_path(steps: int) -> tuple[float, ...]:
    """Right-endpoint path from a corrupt state (0) to a clean state (1)."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    return tuple(step / steps for step in range(1, steps + 1))


def _paired_source_coordinate_capture_hooks(
    graph,
    direction: torch.Tensor,
    destination: dict[str, torch.Tensor],
):
    """Capture matched-reference coordinates for every EAP source node."""
    from eap.graph import AttentionNode, InputNode, LogitNode

    direct_names = {
        node.out_hook
        for node in graph.nodes.values()
        if not isinstance(node, (AttentionNode, LogitNode)) and node.out_hook
    }
    attention_layers = sorted(
        {
            node.layer
            for node in graph.nodes.values()
            if isinstance(node, AttentionNode)
        }
    )
    raw_results: dict[int, torch.Tensor] = {}

    def coordinates(activations: torch.Tensor) -> torch.Tensor:
        vector = direction.to(
            device=activations.device,
            dtype=activations.dtype,
        )
        vector = vector / vector.norm().clamp_min(1e-8)
        return torch.einsum("...d,d->...", activations, vector).detach().float()

    def direct_capture(name: str):
        def hook(activations: torch.Tensor, hook):
            del hook
            destination[name] = coordinates(activations)
            return activations

        return hook

    def result_capture(layer: int):
        def hook(activations: torch.Tensor, hook):
            del hook
            raw_results[layer] = activations.detach()
            return activations

        return hook

    def attention_output_capture(layer: int):
        def hook(activations: torch.Tensor, hook):
            del hook
            result = raw_results.pop(layer)
            bias = activations.detach() - result.sum(dim=2)
            effective_heads = result + bias.unsqueeze(2) / result.shape[2]
            destination[f"blocks.{layer}.attn.hook_result"] = coordinates(
                effective_heads
            )
            return activations

        return hook

    hooks = [(name, direct_capture(name)) for name in sorted(direct_names)]
    for layer in attention_layers:
        hooks.extend(
            [
                (
                    f"blocks.{layer}.attn.hook_result",
                    result_capture(layer),
                ),
                (
                    f"blocks.{layer}.hook_attn_out",
                    attention_output_capture(layer),
                ),
            ]
        )
    return hooks


def capture_paired_source_coordinates(
    model,
    graph,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Capture one matched reference's coordinate at every graph source."""

    coordinates: dict[str, torch.Tensor] = {}
    hooks = _paired_source_coordinate_capture_hooks(
        graph, direction, coordinates
    )
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        _run_forward(model, tokens, attention_mask, hooks)
    return coordinates


def paired_coordinate_source_transform_stages(
    stages: list[
        tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]
    ],
):
    """Compose matched-coordinate source transforms on disjoint token masks."""

    transforms = [
        paired_coordinate_source_transform(
            direction,
            target_coordinates=coordinates,
            mask=mask,
        )
        for direction, coordinates, mask in stages
    ]

    def transform(source_name: str, activations: torch.Tensor) -> torch.Tensor:
        result = activations
        for stage_transform in transforms:
            result = stage_transform(source_name, result)
        return result

    return transform


def paired_coordinate_source_intervention_hooks(
    graph,
    stages: list[
        tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]
    ],
    *,
    strength: float,
):
    """Apply one or more matched-coordinate corruptions at every graph source."""

    from eap.graph import AttentionNode, InputNode, LogitNode

    def composed_hook(source_name: str, *, attention: bool):
        def hook(activations: torch.Tensor, hook):
            result = activations
            for direction, coordinates, mask in stages:
                target = coordinates[source_name]
                if attention:
                    target = target.sum(dim=2)
                result = direction_tensor_coordinate_clamp_hook(
                    direction,
                    target_coordinates=target,
                    mask=mask,
                    strength=strength,
                )(result, hook)
            return result

        return hook

    hooks = []
    direct_names = {
        node.out_hook
        for node in graph.nodes.values()
        if not isinstance(node, (AttentionNode, LogitNode)) and node.out_hook
    }
    hooks.extend(
        (name, composed_hook(name, attention=False))
        for name in sorted(direct_names)
    )
    for layer in sorted(
        {
            node.layer
            for node in graph.nodes.values()
            if isinstance(node, AttentionNode)
        }
    ):
        source_name = f"blocks.{layer}.attn.hook_result"
        hooks.append(
            (
                f"blocks.{layer}.hook_attn_out",
                composed_hook(source_name, attention=True),
            )
        )
    return hooks


def prepare_paired_coordinate_source_corruption(
    model,
    graph,
    reference_tokens: torch.Tensor,
    reference_mask: torch.Tensor,
    direction_masks: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    strength: float = 1.0,
):
    """Prepare cache transformation and forward hooks for matched coordinates."""

    stages = [
        (
            direction,
            capture_paired_source_coordinates(
                model,
                graph,
                reference_tokens,
                reference_mask,
                direction,
            ),
            mask,
        )
        for direction, mask in direction_masks
    ]
    return (
        paired_coordinate_source_transform_stages(stages),
        paired_coordinate_source_intervention_hooks(
            graph, stages, strength=strength
        ),
        stages,
    )


def _paired_coordinate_intervention_hooks(
    graph,
    direction: torch.Tensor,
    source_coordinates: dict[str, torch.Tensor],
    *,
    source_mask: torch.Tensor,
    target_hook: str,
    target_coordinates: torch.Tensor,
    target_mask: torch.Tensor,
    strength: float,
):
    """Apply the paired-coordinate analogue of global refusal projection."""
    from eap.graph import AttentionNode, InputNode, LogitNode

    hooks = []
    direct_names = {
        node.out_hook
        for node in graph.nodes.values()
        if not isinstance(node, (AttentionNode, LogitNode)) and node.out_hook
    }
    for name in sorted(direct_names):
        hooks.append(
            (
                name,
                direction_tensor_coordinate_clamp_hook(
                    direction,
                    target_coordinates=source_coordinates[name],
                    mask=source_mask,
                    strength=strength,
                ),
            )
        )
    for layer in sorted(
        {
            node.layer
            for node in graph.nodes.values()
            if isinstance(node, AttentionNode)
        }
    ):
        per_head = source_coordinates[
            f"blocks.{layer}.attn.hook_result"
        ]
        hooks.append(
            (
                f"blocks.{layer}.hook_attn_out",
                direction_tensor_coordinate_clamp_hook(
                    direction,
                    target_coordinates=per_head.sum(dim=2),
                    mask=source_mask,
                    strength=strength,
                ),
            )
        )
    hooks.append(
        (
            target_hook,
            direction_tensor_coordinate_clamp_hook(
                direction,
                target_coordinates=target_coordinates,
                mask=target_mask,
                strength=strength,
            ),
        )
    )
    return hooks


def attribute_paired_coordinate_projection(
    model,
    graph,
    prompt_pairs: list[tuple[str, str] | tuple[str, str, int]],
    *,
    direction: torch.Tensor,
    target_hook: str,
    target_position: int | None,
    target_scope: str,
    config: AttributionConfig,
) -> dict:
    """Attribute vector formation under matched-coordinate corruption.

    The clean and corrupt model inputs are the same positive-trait sequence.
    A token-aligned negative-trait sequence supplies only scalar coordinate
    targets for each graph source and the terminal representation. At the
    corrupt endpoint every selected source keeps its current orthogonal state
    while its coordinate along ``direction`` is replaced by the paired natural
    negative coordinate. EAP-IG interpolates intervention strength from that
    endpoint back to the unmodified clean run.
    """
    if not prompt_pairs:
        raise ValueError("Paired-coordinate attribution requires prompt pairs.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if target_scope not in {"single_position", "response_prefix"}:
        raise ValueError(
            "target_scope must be single_position or response_prefix."
        )
    if target_scope == "single_position" and target_position is None:
        raise ValueError("single_position requires target_position.")

    path_alphas = interpolation_path(config.ig_steps)
    device = next(model.parameters()).device
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward),
        device=device,
        dtype=torch.float32,
    )
    total_examples = 0
    clean_projection_sum = 0.0
    corrupt_projection_sum = 0.0
    reference_projection_sum = 0.0
    clean_greater_count = 0
    monotonic_count = 0
    path_projection_sums = [0.0] * (len(path_alphas) + 1)

    iterator = tqdm(
        _batches(prompt_pairs, config.batch_size),
        total=(len(prompt_pairs) + config.batch_size - 1) // config.batch_size,
        desc="Paired-coordinate upstream EAP-IG",
    )
    for pair_batch in iterator:
        clean_texts = [pair[0] for pair in pair_batch]
        reference_texts = [pair[1] for pair in pair_batch]
        response_starts = (
            [int(pair[2]) for pair in pair_batch]
            if len(pair_batch[0]) == 3
            else None
        )
        if any(len(pair) != len(pair_batch[0]) for pair in pair_batch):
            raise ValueError("Paired-coordinate batches cannot mix target modes.")
        if target_scope == "response_prefix" and response_starts is None:
            raise ValueError("response_prefix requires response transcript pairs.")
        if target_scope == "single_position" and response_starts is not None:
            raise ValueError("single_position requires prompt-only pairs.")

        clean_tokens, clean_mask = tokenize_prompts(
            model, clean_texts, max_length=config.max_length
        )
        reference_tokens, reference_mask = tokenize_prompts(
            model, reference_texts, max_length=config.max_length
        )
        if tuple(clean_tokens.shape) != tuple(reference_tokens.shape):
            raise ValueError(
                "Positive and negative references must tokenize to the same shape."
            )
        if not torch.equal(clean_mask, reference_mask):
            raise ValueError(
                "Positive and negative attention masks must match."
            )

        if target_scope == "single_position":
            source_mask = clean_mask.bool()
            target_mask = None
        else:
            response_mask = _response_mask(
                clean_tokens, clean_mask, response_starts
            )
            source_mask = response_state_mask(
                response_mask, include_terminal=False
            )
            target_mask = source_mask

        def metric_capture():
            if target_mask is not None:
                capture = MaskedProjectionCapture(
                    direction=direction,
                    mask=target_mask,
                )
                return capture, [(target_hook, capture.hook)]
            return _direction_metric_hooks(
                target_hook, int(target_position), direction
            )

        batch_size, n_pos = clean_tokens.shape
        total_examples += batch_size
        source_coordinates: dict[str, torch.Tensor] = {}
        target_coordinate_store: dict[str, torch.Tensor] = {}

        def capture_target_coordinates(activations: torch.Tensor, hook):
            del hook
            vector = direction.to(
                device=activations.device,
                dtype=activations.dtype,
            )
            vector = vector / vector.norm().clamp_min(1e-8)
            target_coordinate_store["value"] = torch.einsum(
                "bpd,d->bp", activations, vector
            ).detach().float()
            return activations

        reference_capture, reference_metric_hooks = metric_capture()
        reference_hooks = [
            *_paired_source_coordinate_capture_hooks(
                graph, direction, source_coordinates
            ),
            (target_hook, capture_target_coordinates),
            *reference_metric_hooks,
        ]
        with torch.no_grad(), model.hooks(fwd_hooks=reference_hooks):
            _run_forward(
                model, reference_tokens, reference_mask, reference_hooks
            )
        if "value" not in target_coordinate_store:
            raise RuntimeError("Paired target-coordinate hook did not fire.")

        source_transform = paired_coordinate_source_transform(
            direction,
            target_coordinates=source_coordinates,
            mask=source_mask,
        )
        fwd_corrupt, fwd_clean, bwd_hooks, activation_difference = (
            _make_eap_hooks(
                model,
                graph,
                batch_size,
                n_pos,
                scores,
                direction=None,
                corrupt_source_transform=source_transform,
            )
        )

        def intervention_hooks(strength: float):
            return _paired_coordinate_intervention_hooks(
                graph,
                direction,
                source_coordinates,
                source_mask=source_mask,
                target_hook=target_hook,
                target_coordinates=target_coordinate_store["value"],
                target_mask=(
                    source_mask if target_mask is None else target_mask
                ),
                strength=strength,
            )

        corrupt_capture, corrupt_metric_hooks = metric_capture()
        corrupt_hooks = intervention_hooks(1.0)
        with torch.no_grad(), model.hooks(
            fwd_hooks=[
                *fwd_corrupt,
                *corrupt_hooks,
                *corrupt_metric_hooks,
            ]
        ):
            _run_forward(
                model,
                clean_tokens,
                clean_mask,
                corrupt_hooks + corrupt_metric_hooks,
            )
        clean_capture, clean_metric_hooks = metric_capture()
        with torch.no_grad(), model.hooks(
            fwd_hooks=[*fwd_clean, *clean_metric_hooks]
        ):
            _run_forward(model, clean_tokens, clean_mask, clean_metric_hooks)

        clean_values = clean_capture.values().detach()
        corrupt_values = corrupt_capture.values().detach()
        reference_values = reference_capture.values().detach()
        clean_projection_sum += float(clean_values.sum().item())
        corrupt_projection_sum += float(corrupt_values.sum().item())
        reference_projection_sum += float(reference_values.sum().item())
        clean_greater_count += int((clean_values > corrupt_values).sum().item())
        batch_path_values = [corrupt_values]
        path_projection_sums[0] += float(corrupt_values.sum().item())

        for path_index, alpha in enumerate(path_alphas, start=1):
            model.zero_grad(set_to_none=True)
            capture, metric_hooks = metric_capture()
            path_hooks = intervention_hooks(1.0 - float(alpha))
            with model.hooks(
                fwd_hooks=[
                    ("hook_embed", _input_requires_grad),
                    *path_hooks,
                    *metric_hooks,
                ],
                bwd_hooks=bwd_hooks,
            ):
                _run_forward(
                    model,
                    clean_tokens,
                    clean_mask,
                    path_hooks + metric_hooks,
                )
                if capture.activation is None:
                    raise RuntimeError(
                        "Paired-coordinate target metric hook did not fire."
                    )
                capture.activation.retain_grad()
                path_values = capture.values()
                batch_path_values.append(path_values.detach())
                path_projection_sums[path_index] += float(
                    path_values.detach().sum().item()
                )
                capture.mean().backward()
            _accumulate_terminal_metric_scores(
                scores, activation_difference, graph, capture
            )

        stacked_path = torch.stack(batch_path_values, dim=1)
        monotonic_count += int(
            (stacked_path[:, 1:] >= stacked_path[:, :-1] - 1e-5)
            .all(dim=1)
            .sum()
            .item()
        )

    scores /= total_examples * config.ig_steps
    if not torch.isfinite(scores).all():
        raise RuntimeError(
            "Non-finite paired-coordinate EAP-IG scores."
        )
    for edge in graph.edges.values():
        edge.score = float(
            scores[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(
                    edge.child, qkv=edge.qkv, attn_slice=False
                ),
            ].item()
        )

    clean_projection = clean_projection_sum / total_examples
    corrupt_projection = corrupt_projection_sum / total_examples
    reference_projection = reference_projection_sum / total_examples
    return {
        "clean_projection": clean_projection,
        "corrupt_projection": corrupt_projection,
        "reference_negative_projection": reference_projection,
        "clean_minus_corrupt_projection": (
            clean_projection - corrupt_projection
        ),
        "corrupt_minus_reference_projection": (
            corrupt_projection - reference_projection
        ),
        "target_scope": target_scope,
        "num_trajectories": total_examples,
        "clean_greater_fraction": clean_greater_count / total_examples,
        "monotonic_path_fraction": monotonic_count / total_examples,
        "path_alphas": list(path_alphas),
        "path_mean_projections": [
            value / total_examples for value in path_projection_sums
        ],
    }


def attribute_contrastive_prompt_projection(
    model,
    graph,
    prompt_pairs: list[tuple[str, str] | tuple[str, str, int]],
    *,
    direction: torch.Tensor,
    target_hook: str,
    target_position: int | None,
    target_scope: str,
    config: AttributionConfig,
) -> dict:
    """Attribute formation of a signed persona projection from natural pairs.

    A pair is either ``(clean_positive_prompt, corrupt_negative_prompt)`` for a
    prompt target, or the same tuple plus a shared response-token start offset
    for a response target. Clean and corrupt strings must have identical token
    length. EAP source differences are the actual negative-minus-positive
    component activations, and EAP-IG interpolates the graph input residual
    from the negative state to the positive state.

    Unlike the refusal-style routine, this never projects an activation to
    zero and never injects a layer-20 vector downstream of the edges being
    attributed.  It therefore identifies an upstream circuit for the natural
    positive-versus-negative persona-state contrast.
    """
    if not prompt_pairs:
        raise ValueError("Contrastive prompt attribution requires prompt pairs.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    valid_scopes = {
        "single_position",
        "all_prompt_tokens",
        "response_prefix",
        "all_response_tokens",
    }
    if target_scope not in valid_scopes:
        raise ValueError(
            f"target_scope must be one of {sorted(valid_scopes)}, got {target_scope!r}."
        )
    path_alphas = interpolation_path(config.ig_steps)
    device = next(model.parameters()).device
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward),
        device=device,
        dtype=torch.float32,
    )
    total_examples = 0
    clean_projection_sum = 0.0
    corrupt_projection_sum = 0.0
    clean_greater_count = 0
    monotonic_count = 0
    path_projection_sums = [0.0] * (len(path_alphas) + 1)

    iterator = tqdm(
        _batches(prompt_pairs, config.batch_size),
        total=(len(prompt_pairs) + config.batch_size - 1) // config.batch_size,
        desc="Contrastive upstream EAP-IG",
    )
    for pair_batch in iterator:
        clean_prompts = [pair[0] for pair in pair_batch]
        corrupt_prompts = [pair[1] for pair in pair_batch]
        response_starts = (
            [int(pair[2]) for pair in pair_batch]
            if len(pair_batch[0]) == 3
            else None
        )
        if any(len(pair) != len(pair_batch[0]) for pair in pair_batch):
            raise ValueError("Contrastive batches cannot mix target modes.")
        if response_starts is None and target_scope.startswith("response_"):
            raise ValueError("Response target scopes require response transcript pairs.")
        if response_starts is not None and target_scope in {
            "single_position",
            "all_prompt_tokens",
        }:
            raise ValueError("Prompt target scopes require prompt-only pairs.")
        if target_scope == "single_position" and target_position is None:
            raise ValueError("single_position requires target_position.")
        clean_tokens, clean_mask = tokenize_prompts(
            model, clean_prompts, max_length=config.max_length
        )
        corrupt_tokens, corrupt_mask = tokenize_prompts(
            model, corrupt_prompts, max_length=config.max_length
        )
        if tuple(clean_tokens.shape) != tuple(corrupt_tokens.shape):
            raise ValueError(
                "Clean and corrupt prompt batches must tokenize to the same shape."
            )
        if not torch.equal(clean_mask, corrupt_mask):
            raise ValueError(
                "Clean and corrupt attention masks must match; token-align each "
                "prompt pair before attribution."
            )

        batch_size, n_pos = clean_tokens.shape
        total_examples += batch_size
        fwd_corrupt, fwd_clean, bwd_hooks, activation_difference = _make_eap_hooks(
            model,
            graph,
            batch_size,
            n_pos,
            scores,
            direction=None,
            natural_corrupt=True,
        )
        corrupt_input: dict[str, torch.Tensor] = {}
        clean_input: dict[str, torch.Tensor] = {}

        def capture_input(store: dict[str, torch.Tensor]):
            def hook(activations: torch.Tensor, hook):
                del hook
                store["value"] = activations.detach()
                return activations

            return hook

        def metric_capture(tokens, attention_mask):
            if target_scope == "all_prompt_tokens":
                capture = MaskedProjectionCapture(
                    direction=direction,
                    mask=attention_mask.bool(),
                )
                return capture, [(target_hook, capture.hook)]
            if response_starts is not None:
                response_mask = _response_mask(
                    tokens, attention_mask, response_starts
                )
                capture = MaskedProjectionCapture(
                    direction=direction,
                    mask=response_state_mask(
                        response_mask,
                        include_terminal=target_scope == "all_response_tokens",
                    ),
                )
                return capture, [(target_hook, capture.hook)]
            return _direction_metric_hooks(
                target_hook, int(target_position), direction
            )

        corrupt_capture, corrupt_metric_hooks = metric_capture(
            corrupt_tokens, corrupt_mask
        )
        with torch.no_grad(), model.hooks(
            fwd_hooks=[
                *fwd_corrupt,
                (graph.nodes["input"].out_hook, capture_input(corrupt_input)),
                *corrupt_metric_hooks,
            ]
        ):
            _run_forward(model, corrupt_tokens, corrupt_mask, fwd_corrupt)
        clean_capture, clean_metric_hooks = metric_capture(clean_tokens, clean_mask)
        with torch.no_grad(), model.hooks(
            fwd_hooks=[
                *fwd_clean,
                (graph.nodes["input"].out_hook, capture_input(clean_input)),
                *clean_metric_hooks,
            ]
        ):
            _run_forward(model, clean_tokens, clean_mask, fwd_clean)
        if "value" not in corrupt_input or "value" not in clean_input:
            raise RuntimeError("The upstream graph input hook did not fire.")
        clean_values = clean_capture.values().detach()
        corrupt_values = corrupt_capture.values().detach()
        clean_projection_sum += float(clean_values.sum().item())
        corrupt_projection_sum += float(corrupt_values.sum().item())
        clean_greater_count += int((clean_values > corrupt_values).sum().item())
        batch_path_values = [corrupt_values]
        path_projection_sums[0] += float(corrupt_values.sum().item())

        for path_index, alpha in enumerate(path_alphas, start=1):
            model.zero_grad(set_to_none=True)
            interpolated = (
                corrupt_input["value"]
                + float(alpha) * (clean_input["value"] - corrupt_input["value"])
            ).detach()

            def interpolate_input(activations: torch.Tensor, hook):
                del activations, hook
                return interpolated.clone().requires_grad_(True)

            capture, metric_hooks = metric_capture(clean_tokens, clean_mask)
            with model.hooks(
                fwd_hooks=[
                    (graph.nodes["input"].out_hook, interpolate_input),
                    *metric_hooks,
                ],
                bwd_hooks=bwd_hooks,
            ):
                _run_forward(model, clean_tokens, clean_mask, metric_hooks)
                if capture.activation is None:
                    raise RuntimeError("The upstream persona metric hook did not fire.")
                capture.activation.retain_grad()
                path_values = capture.values()
                batch_path_values.append(path_values.detach())
                path_projection_sums[path_index] += float(
                    path_values.detach().sum().item()
                )
                capture.mean().backward()
            _accumulate_terminal_metric_scores(
                scores, activation_difference, graph, capture
            )
        stacked_path = torch.stack(batch_path_values, dim=1)
        monotonic_count += int(
            (stacked_path[:, 1:] >= stacked_path[:, :-1] - 1e-5)
            .all(dim=1)
            .sum()
            .item()
        )

    scores /= total_examples * config.ig_steps
    if not torch.isfinite(scores).all():
        raise RuntimeError(
            "Non-finite contrastive upstream EAP-IG scores; do not interpret this run."
        )
    for edge in graph.edges.values():
        edge.score = float(
            scores[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(
                    edge.child, qkv=edge.qkv, attn_slice=False
                ),
            ].item()
        )
    return {
        "clean_positive_projection": clean_projection_sum / total_examples,
        "corrupt_negative_projection": corrupt_projection_sum / total_examples,
        "clean_minus_corrupt_projection": (
            clean_projection_sum - corrupt_projection_sum
        )
        / total_examples,
        "target_scope": target_scope,
        "num_trajectories": total_examples,
        "clean_greater_fraction": clean_greater_count / total_examples,
        "monotonic_path_fraction": monotonic_count / total_examples,
        "path_alphas": list(path_alphas),
        "path_mean_projections": [
            value / total_examples for value in path_projection_sums
        ],
    }


def attribute_steering_response_logprob(
    model,
    graph,
    transcripts: list[tuple[str, int]],
    *,
    raw_direction: torch.Tensor,
    target_hook: str,
    corrupt_coefficient: float,
    config: AttributionConfig,
) -> dict:
    """Attribute the behavioral effect along a response-steering path.

    Clean is coefficient 0 and corrupt is a behaviorally validated negative
    coefficient.  EAP source differences are actual corrupt-minus-clean
    component outputs.  EAP-IG gradients are averaged along the same
    coefficient path, and the scalar target is mean autoregressive log
    probability of an already judged, high-trait response.

    This identifies the downstream circuit that mediates a persona vector's
    effect on generation logits.  It does not identify the upstream circuit
    that naturally constructs the vector; free-generation circuit patching is
    still required for a causal behavioral claim.
    """
    if not transcripts:
        raise ValueError("Attribution requires at least one transcript.")
    path_coefficients = steering_coefficient_path(corrupt_coefficient, config.ig_steps)
    if graph.nodes["input"].out_hook != target_hook:
        raise ValueError("Steering graph input hook does not match the intervention hook.")

    device = next(model.parameters()).device
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device=device, dtype=torch.float32
    )
    total_examples = 0
    clean_metric_sum = 0.0
    corrupt_metric_sum = 0.0
    iterator = tqdm(
        _batches(transcripts, config.batch_size),
        total=(len(transcripts) + config.batch_size - 1) // config.batch_size,
        desc="Steering-path EAP-IG",
    )
    for transcript_batch in iterator:
        texts = [item[0] for item in transcript_batch]
        starts = [item[1] for item in transcript_batch]
        tokens, attention_mask = tokenize_prompts(model, texts, max_length=config.max_length)
        batch_size, n_pos = tokens.shape
        response_mask = _response_mask(tokens, attention_mask, starts)
        prediction_mask = response_prediction_mask(response_mask)
        total_examples += batch_size

        def corrupt_source_transform(source_name: str, activations: torch.Tensor):
            if source_name != target_hook:
                return activations
            vector = raw_direction.to(device=activations.device, dtype=activations.dtype)
            return activations + (
                prediction_mask.to(device=activations.device, dtype=activations.dtype)
                .unsqueeze(-1)
                * float(corrupt_coefficient)
                * vector
            )

        fwd_corrupt, fwd_clean, bwd_hooks, activation_difference = _make_eap_hooks(
            model,
            graph,
            batch_size,
            n_pos,
            scores,
            direction=None,
            include_logit_gradients=True,
            corrupt_source_transform=corrupt_source_transform,
        )
        corrupt_hook = (
            target_hook,
            masked_activation_addition_hook(
                raw_direction,
                coefficient=corrupt_coefficient,
                mask=prediction_mask,
            ),
        )
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_corrupt + [corrupt_hook]):
            corrupt_logits = _run_forward(
                model, tokens, attention_mask, fwd_corrupt + [corrupt_hook]
            )
            corrupt_values = response_logprob_values(corrupt_logits, tokens, response_mask)
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_clean):
            clean_logits = _run_forward(model, tokens, attention_mask, fwd_clean)
            clean_values = response_logprob_values(clean_logits, tokens, response_mask)
        clean_metric_sum += float(clean_values.sum().item())
        corrupt_metric_sum += float(corrupt_values.sum().item())

        for coefficient in path_coefficients:
            model.zero_grad(set_to_none=True)
            path_hooks = []
            if coefficient != 0:
                path_hooks.append(
                    (
                        target_hook,
                        masked_activation_addition_hook(
                            raw_direction,
                            coefficient=coefficient,
                            mask=prediction_mask,
                        ),
                    )
                )
            with model.hooks(
                fwd_hooks=[
                    (graph.nodes["input"].out_hook, _input_requires_grad),
                    *path_hooks,
                ],
                bwd_hooks=bwd_hooks,
            ):
                logits = _run_forward(model, tokens, attention_mask, path_hooks)
                response_logprob_values(logits, tokens, response_mask).mean().backward()

    scores /= total_examples * config.ig_steps
    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite steering-path EAP-IG edge scores.")
    for edge in graph.edges.values():
        edge.score = float(
            scores[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
            ].item()
        )
    return {
        "clean_response_logprob": clean_metric_sum / total_examples,
        "corrupt_response_logprob": corrupt_metric_sum / total_examples,
        "clean_minus_corrupt": (clean_metric_sum - corrupt_metric_sum) / total_examples,
        "path_coefficients": list(path_coefficients),
    }


def attribute_steering_contrastive_response_logprob(
    model,
    graph,
    transcript_pairs: list[
        tuple[tuple[str, int], tuple[str, int]]
    ],
    *,
    raw_direction: torch.Tensor,
    target_hook: str,
    corrupt_coefficient: float,
    config: AttributionConfig,
) -> dict:
    """Attribute a trait-specific response-likelihood margin.

    Each item contains a positive-trait and negative-trait answer under the
    *same positive-trait prompt*.  The scalar metric for one item is

        mean_logp(positive_answer) - mean_logp(negative_answer).

    As in :func:`attribute_steering_response_logprob`, clean is coefficient
    zero and corrupt is a behaviorally validated negative coefficient at the
    persona-vector site.  Using paired answers prevents the attribution target
    from collapsing to the likelihood of one response's wording or length.
    """
    if not transcript_pairs:
        raise ValueError("Contrastive attribution requires at least one transcript pair.")
    path_coefficients = steering_coefficient_path(
        corrupt_coefficient, config.ig_steps
    )
    if graph.nodes["input"].out_hook != target_hook:
        raise ValueError("Steering graph input hook does not match the intervention hook.")

    device = next(model.parameters()).device
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device=device, dtype=torch.float32
    )
    total_pairs = 0
    clean_margin_sum = 0.0
    corrupt_margin_sum = 0.0
    clean_positive_sum = 0.0
    clean_negative_sum = 0.0
    corrupt_positive_sum = 0.0
    corrupt_negative_sum = 0.0
    if config.batch_size != 1:
        raise ValueError(
            "Contrastive attribution processes each pair sequentially; "
            "set batch_size=1."
        )
    iterator = tqdm(
        transcript_pairs,
        total=len(transcript_pairs),
        desc="Contrastive steering-path EAP-IG",
    )
    for positive_transcript, negative_transcript in iterator:
        total_pairs += 1
        clean_values_for_pair = []
        corrupt_values_for_pair = []
        # Two sequential single-transcript passes are algebraically identical
        # to differentiating their likelihood difference in one batch. They
        # avoid doubling EAP's large activation/gradient cache on 40 GB GPUs.
        for sign, (text, start) in (
            (1.0, positive_transcript),
            (-1.0, negative_transcript),
        ):
            tokens, attention_mask = tokenize_prompts(
                model, [text], max_length=config.max_length
            )
            batch_size, n_pos = tokens.shape
            response_mask = _response_mask(
                tokens, attention_mask, [start]
            )
            prediction_mask = response_prediction_mask(response_mask)

            def corrupt_source_transform(
                source_name: str, activations: torch.Tensor
            ):
                if source_name != target_hook:
                    return activations
                vector = raw_direction.to(
                    device=activations.device, dtype=activations.dtype
                )
                return activations + (
                    prediction_mask.to(
                        device=activations.device, dtype=activations.dtype
                    ).unsqueeze(-1)
                    * float(corrupt_coefficient)
                    * vector
                )

            fwd_corrupt, fwd_clean, bwd_hooks, _ = _make_eap_hooks(
                model,
                graph,
                batch_size,
                n_pos,
                scores,
                direction=None,
                include_logit_gradients=True,
                corrupt_source_transform=corrupt_source_transform,
            )
            corrupt_hook = (
                target_hook,
                masked_activation_addition_hook(
                    raw_direction,
                    coefficient=corrupt_coefficient,
                    mask=prediction_mask,
                ),
            )
            with torch.no_grad(), model.hooks(
                fwd_hooks=fwd_corrupt + [corrupt_hook]
            ):
                corrupt_logits = _run_forward(
                    model,
                    tokens,
                    attention_mask,
                    fwd_corrupt + [corrupt_hook],
                )
                corrupt_value = response_logprob_values(
                    corrupt_logits, tokens, response_mask
                )[0]
            with torch.no_grad(), model.hooks(fwd_hooks=fwd_clean):
                clean_logits = _run_forward(
                    model, tokens, attention_mask, fwd_clean
                )
                clean_value = response_logprob_values(
                    clean_logits, tokens, response_mask
                )[0]
            clean_values_for_pair.append(float(clean_value.item()))
            corrupt_values_for_pair.append(float(corrupt_value.item()))

            for coefficient in path_coefficients:
                model.zero_grad(set_to_none=True)
                path_hooks = []
                if coefficient != 0:
                    path_hooks.append(
                        (
                            target_hook,
                            masked_activation_addition_hook(
                                raw_direction,
                                coefficient=coefficient,
                                mask=prediction_mask,
                            ),
                        )
                    )
                with model.hooks(
                    fwd_hooks=[
                        (
                            graph.nodes["input"].out_hook,
                            _input_requires_grad,
                        ),
                        *path_hooks,
                    ],
                    bwd_hooks=bwd_hooks,
                ):
                    logits = _run_forward(
                        model, tokens, attention_mask, path_hooks
                    )
                    value = response_logprob_values(
                        logits, tokens, response_mask
                    )[0]
                    (sign * value).backward()

        clean_positive, clean_negative = clean_values_for_pair
        corrupt_positive, corrupt_negative = corrupt_values_for_pair
        clean_margin_sum += clean_positive - clean_negative
        corrupt_margin_sum += corrupt_positive - corrupt_negative
        clean_positive_sum += clean_positive
        clean_negative_sum += clean_negative
        corrupt_positive_sum += corrupt_positive
        corrupt_negative_sum += corrupt_negative

    scores /= total_pairs * config.ig_steps
    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite contrastive steering-path EAP-IG scores.")
    for edge in graph.edges.values():
        edge.score = float(
            scores[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(
                    edge.child, qkv=edge.qkv, attn_slice=False
                ),
            ].item()
        )
    return {
        "clean_contrastive_margin": clean_margin_sum / total_pairs,
        "corrupt_contrastive_margin": corrupt_margin_sum / total_pairs,
        "clean_minus_corrupt_margin": (
            clean_margin_sum - corrupt_margin_sum
        )
        / total_pairs,
        "clean_positive_response_logprob": clean_positive_sum / total_pairs,
        "clean_negative_response_logprob": clean_negative_sum / total_pairs,
        "corrupt_positive_response_logprob": corrupt_positive_sum / total_pairs,
        "corrupt_negative_response_logprob": corrupt_negative_sum / total_pairs,
        "path_coefficients": list(path_coefficients),
    }


def attribute_paired_coordinate_response_logprob(
    model,
    graph,
    transcript_pairs: list[
        tuple[
            tuple[str, str, int],
            tuple[str, str, int],
        ]
    ],
    *,
    prompt_direction: torch.Tensor,
    generation_direction: torch.Tensor,
    config: AttributionConfig,
) -> dict:
    """Attribute a same-input, paired-coordinate intervention to answer logits.

    The clean input always uses the positive instruction. The token-aligned
    negative transcript supplies only scalar coordinates: the prompt direction
    is replaced on prompt states, and the generation direction is replaced on
    nonterminal response-prefix states. The scalar target is the sycophantic
    answer log-probability minus the normal answer log-probability.
    """

    if not transcript_pairs:
        raise ValueError("End-to-end attribution requires transcript pairs.")
    if config.batch_size != 1:
        raise ValueError(
            "Paired-coordinate logit attribution is sequential; set batch_size=1."
        )
    path_alphas = interpolation_path(config.ig_steps)
    device = next(model.parameters()).device
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward),
        device=device,
        dtype=torch.float32,
    )
    total_pairs = 0
    clean_margin_sum = 0.0
    corrupt_margin_sum = 0.0
    clean_positive_sum = 0.0
    clean_negative_sum = 0.0
    corrupt_positive_sum = 0.0
    corrupt_negative_sum = 0.0

    iterator = tqdm(
        transcript_pairs,
        total=len(transcript_pairs),
        desc="Paired-coordinate end-to-end EAP-IG",
    )
    for positive_spec, negative_spec in iterator:
        total_pairs += 1
        clean_values_for_pair = []
        corrupt_values_for_pair = []
        for sign, (clean_text, reference_text, response_start) in (
            (1.0, positive_spec),
            (-1.0, negative_spec),
        ):
            clean_tokens, clean_mask = tokenize_prompts(
                model, [clean_text], max_length=config.max_length
            )
            reference_tokens, reference_mask = tokenize_prompts(
                model, [reference_text], max_length=config.max_length
            )
            if tuple(clean_tokens.shape) != tuple(reference_tokens.shape):
                raise ValueError(
                    "Paired-coordinate transcripts must have identical shape."
                )
            if not torch.equal(clean_mask, reference_mask):
                raise ValueError(
                    "Paired-coordinate attention masks must match."
                )
            if not torch.equal(
                clean_tokens[:, response_start:],
                reference_tokens[:, response_start:],
            ):
                raise ValueError(
                    "Answer tokens must be identical across paired-coordinate "
                    "references."
                )

            response_mask = _response_mask(
                clean_tokens, clean_mask, [response_start]
            )
            prompt_mask = clean_mask.bool() & ~response_mask
            generation_mask = response_state_mask(
                response_mask, include_terminal=False
            )
            if bool((prompt_mask & generation_mask).any()):
                raise RuntimeError(
                    "Prompt and generation corruption masks must be disjoint."
                )

            source_transform, corrupt_hooks, stages = (
                prepare_paired_coordinate_source_corruption(
                    model,
                    graph,
                    reference_tokens,
                    reference_mask,
                    [
                        (prompt_direction, prompt_mask),
                        (generation_direction, generation_mask),
                    ],
                )
            )
            batch_size, n_pos = clean_tokens.shape
            fwd_corrupt, fwd_clean, bwd_hooks, _ = _make_eap_hooks(
                model,
                graph,
                batch_size,
                n_pos,
                scores,
                direction=None,
                include_logit_gradients=True,
                corrupt_source_transform=source_transform,
            )

            with torch.no_grad(), model.hooks(
                fwd_hooks=[*fwd_corrupt, *corrupt_hooks]
            ):
                corrupt_logits = _run_forward(
                    model, clean_tokens, clean_mask, corrupt_hooks
                )
                corrupt_value = response_logprob_values(
                    corrupt_logits, clean_tokens, response_mask
                )[0]
            with torch.no_grad(), model.hooks(fwd_hooks=fwd_clean):
                clean_logits = _run_forward(
                    model, clean_tokens, clean_mask, fwd_clean
                )
                clean_value = response_logprob_values(
                    clean_logits, clean_tokens, response_mask
                )[0]
            clean_values_for_pair.append(float(clean_value.item()))
            corrupt_values_for_pair.append(float(corrupt_value.item()))

            for alpha in path_alphas:
                model.zero_grad(set_to_none=True)
                path_hooks = paired_coordinate_source_intervention_hooks(
                    graph, stages, strength=1.0 - float(alpha)
                )
                with model.hooks(
                    fwd_hooks=[
                        ("hook_embed", _input_requires_grad),
                        *path_hooks,
                    ],
                    bwd_hooks=bwd_hooks,
                ):
                    logits = _run_forward(
                        model, clean_tokens, clean_mask, path_hooks
                    )
                    value = response_logprob_values(
                        logits, clean_tokens, response_mask
                    )[0]
                    (sign * value).backward()

        clean_positive, clean_negative = clean_values_for_pair
        corrupt_positive, corrupt_negative = corrupt_values_for_pair
        clean_margin_sum += clean_positive - clean_negative
        corrupt_margin_sum += corrupt_positive - corrupt_negative
        clean_positive_sum += clean_positive
        clean_negative_sum += clean_negative
        corrupt_positive_sum += corrupt_positive
        corrupt_negative_sum += corrupt_negative

    scores /= total_pairs * config.ig_steps
    if not torch.isfinite(scores).all():
        raise RuntimeError(
            "Non-finite paired-coordinate end-to-end EAP-IG scores."
        )
    for edge in graph.edges.values():
        edge.score = float(
            scores[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(
                    edge.child, qkv=edge.qkv, attn_slice=False
                ),
            ].item()
        )
    return {
        "clean_contrastive_margin": clean_margin_sum / total_pairs,
        "corrupt_contrastive_margin": corrupt_margin_sum / total_pairs,
        "clean_minus_corrupt_margin": (
            clean_margin_sum - corrupt_margin_sum
        )
        / total_pairs,
        "clean_positive_response_logprob": clean_positive_sum / total_pairs,
        "clean_negative_response_logprob": clean_negative_sum / total_pairs,
        "corrupt_positive_response_logprob": corrupt_positive_sum / total_pairs,
        "corrupt_negative_response_logprob": corrupt_negative_sum / total_pairs,
        "path_alphas": list(path_alphas),
        "prompt_scope": "all prompt-token source states",
        "generation_scope": (
            "all nonterminal teacher-forced response-prefix source states"
        ),
    }


def attribute_contrastive_prompt_response_logprob(
    model,
    graph,
    transcript_pairs: list[
        tuple[
            tuple[str, str, int],
            tuple[str, str, int],
        ]
    ],
    *,
    config: AttributionConfig,
) -> dict:
    """Attribute natural prompt control of a paired response-logprob margin.

    Each item contains high- and low-trait answer specifications. A specification
    is ``(positive_prompt_plus_answer, negative_prompt_plus_same_answer,
    response_start)``. The prompts must be token-aligned and the answer is held
    fixed across the prompt counterfactual, so root-residual interpolation changes
    only the natural persona condition. The scalar target is:

        mean_logp(high_trait_answer) - mean_logp(low_trait_answer).

    This is the end-to-end counterpart to the formation/expression decomposition:
    the graph runs from the natural prompt state at block 0 directly to logits.
    """

    if not transcript_pairs:
        raise ValueError("End-to-end attribution requires transcript pairs.")
    if config.batch_size != 1:
        raise ValueError(
            "End-to-end contrastive attribution is sequential; set batch_size=1."
        )
    path_alphas = interpolation_path(config.ig_steps)
    device = next(model.parameters()).device
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward),
        device=device,
        dtype=torch.float32,
    )
    total_pairs = 0
    clean_margin_sum = 0.0
    corrupt_margin_sum = 0.0
    clean_positive_sum = 0.0
    clean_negative_sum = 0.0
    corrupt_positive_sum = 0.0
    corrupt_negative_sum = 0.0

    iterator = tqdm(
        transcript_pairs,
        total=len(transcript_pairs),
        desc="End-to-end contrastive EAP-IG",
    )
    for positive_spec, negative_spec in iterator:
        total_pairs += 1
        clean_values_for_pair = []
        corrupt_values_for_pair = []
        for sign, (clean_text, corrupt_text, response_start) in (
            (1.0, positive_spec),
            (-1.0, negative_spec),
        ):
            clean_tokens, clean_mask = tokenize_prompts(
                model, [clean_text], max_length=config.max_length
            )
            corrupt_tokens, corrupt_mask = tokenize_prompts(
                model, [corrupt_text], max_length=config.max_length
            )
            if tuple(clean_tokens.shape) != tuple(corrupt_tokens.shape):
                raise ValueError(
                    "End-to-end clean/corrupt transcripts must have identical shape."
                )
            if not torch.equal(clean_mask, corrupt_mask):
                raise ValueError(
                    "End-to-end clean/corrupt attention masks must match."
                )
            if not torch.equal(
                clean_tokens[:, response_start:],
                corrupt_tokens[:, response_start:],
            ):
                raise ValueError(
                    "End-to-end answer tokens must be identical across the "
                    "prompt counterfactual."
                )
            batch_size, n_pos = clean_tokens.shape
            response_mask = _response_mask(
                clean_tokens, clean_mask, [response_start]
            )
            fwd_corrupt, fwd_clean, bwd_hooks, activation_difference = (
                _make_eap_hooks(
                    model,
                    graph,
                    batch_size,
                    n_pos,
                    scores,
                    direction=None,
                    include_logit_gradients=True,
                    natural_corrupt=True,
                )
            )
            corrupt_input: dict[str, torch.Tensor] = {}
            clean_input: dict[str, torch.Tensor] = {}

            def capture_input(store: dict[str, torch.Tensor]):
                def hook(activations: torch.Tensor, hook):
                    del hook
                    store["value"] = activations.detach()
                    return activations

                return hook

            with torch.no_grad(), model.hooks(
                fwd_hooks=[
                    *fwd_corrupt,
                    (
                        graph.nodes["input"].out_hook,
                        capture_input(corrupt_input),
                    ),
                ]
            ):
                corrupt_logits = _run_forward(
                    model, corrupt_tokens, corrupt_mask, fwd_corrupt
                )
                corrupt_value = response_logprob_values(
                    corrupt_logits, corrupt_tokens, response_mask
                )[0]
            with torch.no_grad(), model.hooks(
                fwd_hooks=[
                    *fwd_clean,
                    (
                        graph.nodes["input"].out_hook,
                        capture_input(clean_input),
                    ),
                ]
            ):
                clean_logits = _run_forward(
                    model, clean_tokens, clean_mask, fwd_clean
                )
                clean_value = response_logprob_values(
                    clean_logits, clean_tokens, response_mask
                )[0]
            if "value" not in corrupt_input or "value" not in clean_input:
                raise RuntimeError("End-to-end graph input hook did not fire.")
            clean_values_for_pair.append(float(clean_value.item()))
            corrupt_values_for_pair.append(float(corrupt_value.item()))

            for alpha in path_alphas:
                model.zero_grad(set_to_none=True)
                interpolated = (
                    corrupt_input["value"]
                    + float(alpha)
                    * (clean_input["value"] - corrupt_input["value"])
                ).detach()

                def interpolate_input(activations: torch.Tensor, hook):
                    del activations, hook
                    return interpolated.clone().requires_grad_(True)

                with model.hooks(
                    fwd_hooks=[
                        (
                            graph.nodes["input"].out_hook,
                            interpolate_input,
                        )
                    ],
                    bwd_hooks=bwd_hooks,
                ):
                    logits = _run_forward(
                        model, clean_tokens, clean_mask, ()
                    )
                    value = response_logprob_values(
                        logits, clean_tokens, response_mask
                    )[0]
                    (sign * value).backward()

        clean_positive, clean_negative = clean_values_for_pair
        corrupt_positive, corrupt_negative = corrupt_values_for_pair
        clean_margin_sum += clean_positive - clean_negative
        corrupt_margin_sum += corrupt_positive - corrupt_negative
        clean_positive_sum += clean_positive
        clean_negative_sum += clean_negative
        corrupt_positive_sum += corrupt_positive
        corrupt_negative_sum += corrupt_negative

    scores /= total_pairs * config.ig_steps
    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite end-to-end contrastive EAP-IG scores.")
    for edge in graph.edges.values():
        edge.score = float(
            scores[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(
                    edge.child, qkv=edge.qkv, attn_slice=False
                ),
            ].item()
        )
    return {
        "clean_contrastive_margin": clean_margin_sum / total_pairs,
        "corrupt_contrastive_margin": corrupt_margin_sum / total_pairs,
        "clean_minus_corrupt_margin": (
            clean_margin_sum - corrupt_margin_sum
        )
        / total_pairs,
        "clean_positive_response_logprob": clean_positive_sum / total_pairs,
        "clean_negative_response_logprob": clean_negative_sum / total_pairs,
        "corrupt_positive_response_logprob": corrupt_positive_sum / total_pairs,
        "corrupt_negative_response_logprob": corrupt_negative_sum / total_pairs,
        "path_alphas": list(path_alphas),
    }


def attribute_response_average(
    model,
    graph,
    transcripts: list[tuple[str, int]],
    *,
    direction: torch.Tensor,
    target_hook: str,
    config: AttributionConfig,
    integrated_gradients: bool,
    corruption: str = "projection",
    raw_direction: torch.Tensor | None = None,
    corrupt_coefficient: float | None = None,
    negative_coordinates: torch.Tensor | None = None,
) -> dict:
    """Attribute mean response-token projection for teacher-forced transcripts.

    Each tuple contains an already formatted prompt+answer string and the
    response's token offset under the model tokenizer. This exactly matches
    the response-average statistic used to extract Persona Vectors.
    """
    if not transcripts:
        raise ValueError("Attribution requires at least one transcript.")
    if integrated_gradients and config.ig_steps <= 0:
        raise ValueError("ig_steps must be positive.")
    if corruption not in {
        "projection",
        "upstream_steering",
        "negative_centroid_clamp",
    }:
        raise ValueError(
            "response-average corruption must be projection, upstream_steering, "
            "or negative_centroid_clamp."
        )
    if corruption == "upstream_steering":
        if raw_direction is None or corrupt_coefficient is None:
            raise ValueError(
                "upstream_steering requires raw_direction and corrupt_coefficient."
            )
        if corrupt_coefficient >= 0:
            raise ValueError("Upstream steering corruption coefficient must be negative.")
    if corruption == "negative_centroid_clamp":
        if negative_coordinates is None:
            raise ValueError(
                "negative_centroid_clamp requires layerwise negative_coordinates."
            )
        if (
            negative_coordinates.ndim != 1
            or negative_coordinates.numel() != model.cfg.n_layers
        ):
            raise ValueError(
                "negative_coordinates must contain exactly one value per model layer."
            )

    device = next(model.parameters()).device
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=torch.float32)
    total_examples = 0
    clean_projection_sum = 0.0
    corrupt_projection_sum = 0.0
    last_path_values: tuple[float, ...] = ()
    iterator = tqdm(
        _batches(transcripts, config.batch_size),
        total=(len(transcripts) + config.batch_size - 1) // config.batch_size,
        desc="Persona EAP-IG" if integrated_gradients else "Persona EAP",
    )
    for transcript_batch in iterator:
        texts = [item[0] for item in transcript_batch]
        starts = [item[1] for item in transcript_batch]
        tokens, attention_mask = tokenize_prompts(model, texts, max_length=config.max_length)
        batch_size, n_pos = tokens.shape
        response_mask = _response_mask(tokens, attention_mask, starts)
        total_examples += batch_size
        if corruption == "projection":
            corrupt_value = 1.0
            eap_direction = direction
            source_transform = None

            def intervention_hooks(value: float):
                return all_layer_ablation_hooks(
                    model, direction, strength=value
                )

        elif corruption == "upstream_steering":
            corrupt_value = float(corrupt_coefficient)
            eap_direction = None
            source_transform = steering_source_transform(
                raw_direction,
                coefficient=corrupt_value,
            )

            def intervention_hooks(value: float):
                return (
                    all_layer_steering_hooks(
                        model,
                        raw_direction,
                        coefficient=value,
                    )
                    if value != 0
                    else []
                )

        else:
            corrupt_value = 1.0
            eap_direction = None
            source_transform = coordinate_clamp_source_transform(
                direction,
                target_coordinate=float(negative_coordinates[0].item()),
                mask=response_mask,
                input_source_name=graph.nodes["input"].out_hook,
            )

            def intervention_hooks(value: float):
                return (
                    all_layer_coordinate_clamp_hooks(
                        model,
                        direction,
                        target_coordinates=negative_coordinates,
                        mask=response_mask,
                        strength=value,
                    )
                    if value != 0
                    else []
                )

        corrupt_hooks = intervention_hooks(corrupt_value)
        path_values = (
            steering_coefficient_path(corrupt_value, config.ig_steps)
            if integrated_gradients
            else (0.0,)
        )
        last_path_values = tuple(path_values)
        fwd_corrupt, fwd_clean, bwd_hooks, activation_difference = (
            _make_eap_hooks(
                model,
                graph,
                batch_size,
                n_pos,
                scores,
                direction=eap_direction,
                corrupt_source_transform=source_transform,
            )
        )
        corrupt_capture = MaskedProjectionCapture(
            direction=direction, mask=response_mask
        )
        corrupt_endpoint_hooks = [*fwd_corrupt, *corrupt_hooks]
        with torch.no_grad(), model.hooks(
            fwd_hooks=[
                *corrupt_endpoint_hooks,
                (target_hook, corrupt_capture.hook),
            ]
        ):
            _run_forward(model, tokens, attention_mask, corrupt_endpoint_hooks)
        clean_capture = MaskedProjectionCapture(
            direction=direction, mask=response_mask
        )
        with torch.no_grad(), model.hooks(
            fwd_hooks=[*fwd_clean, (target_hook, clean_capture.hook)]
        ):
            _run_forward(model, tokens, attention_mask, fwd_clean)
        corrupt_projection_sum += float(corrupt_capture.values().sum().item())
        clean_projection_sum += float(clean_capture.values().sum().item())

        for value in last_path_values:
            model.zero_grad(set_to_none=True)
            capture = MaskedProjectionCapture(direction=direction, mask=response_mask)
            metric_hook = [(target_hook, capture.hook)]
            path_hooks = intervention_hooks(value)
            with model.hooks(
                fwd_hooks=[
                    ("hook_embed", _input_requires_grad),
                    *path_hooks,
                    *metric_hook,
                ],
                bwd_hooks=bwd_hooks,
            ):
                _run_forward(
                    model,
                    tokens,
                    attention_mask,
                    path_hooks + metric_hook,
                )
                if capture.activation is None:
                    raise RuntimeError("The persona target residual metric hook did not fire.")
                capture.activation.retain_grad()
                capture.mean().backward()
            _accumulate_terminal_metric_scores(scores, activation_difference, graph, capture)

    scores /= total_examples
    if integrated_gradients:
        scores /= config.ig_steps
    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite persona EAP/EAP-IG edge scores; do not interpret this run.")
    for edge in graph.edges.values():
        edge.score = float(
            scores[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
            ].item()
        )
    clean_projection = clean_projection_sum / total_examples
    corrupt_projection = corrupt_projection_sum / total_examples
    return {
        "clean_projection": clean_projection,
        "corrupt_projection": corrupt_projection,
        "clean_minus_corrupt_projection": clean_projection - corrupt_projection,
        "path_values": list(last_path_values),
    }


def _input_requires_grad(activations: torch.Tensor, hook) -> torch.Tensor:
    """Create a gradient-tracked residual input without storing parameter grads.

    RQ1's model loader disables parameter gradients to fit a 2B model on a
    modest GPU.  EAP needs gradients with respect to activations, not weights,
    so a leaf at ``hook_embed`` supplies exactly that graph at far lower memory
    cost.
    """
    return activations.detach().requires_grad_(True)


def attribute(
    model,
    graph,
    prompts: list[str],
    *,
    direction: torch.Tensor,
    target_hook: str,
    target_position: int,
    config: AttributionConfig,
    integrated_gradients: bool,
) -> None:
    """Populate ``graph.edges[*].score`` using EAP or intervention-path EAP-IG."""
    if not prompts:
        raise ValueError("Attribution requires at least one harmful prompt.")
    if integrated_gradients and config.ig_steps <= 0:
        raise ValueError("ig_steps must be positive.")

    device = next(model.parameters()).device
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=torch.float32)
    total_examples = 0
    total_gradient_steps = 0

    iterator = tqdm(
        _batches(prompts, config.batch_size),
        total=(len(prompts) + config.batch_size - 1) // config.batch_size,
        desc="EAP-IG" if integrated_gradients else "EAP",
    )
    for prompt_batch in iterator:
        tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=config.max_length)
        batch_size, n_pos = tokens.shape
        total_examples += batch_size
        fwd_corrupt, fwd_clean, bwd_hooks, activation_difference = _make_eap_hooks(
            model, graph, batch_size, n_pos, scores, direction
        )

        # Cache the intervention-generated corrupt component outputs.  Hooks
        # Cache handlers run before the intervention but explicitly project
        # corrupt source values. This is necessary for the complete Gemma
        # attention write (per-head results plus output bias).
        with torch.no_grad(), model.hooks(
            fwd_hooks=fwd_corrupt + all_layer_ablation_hooks(model, direction, strength=1.0)
        ):
            _run_forward(model, tokens, attention_mask, fwd_corrupt)

        # The second cache pass gives (corrupt - clean) component differences.
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_clean):
            _run_forward(model, tokens, attention_mask, fwd_clean)

        path_alphas: Iterable[float] = (
            (step / config.ig_steps for step in range(1, config.ig_steps + 1))
            if integrated_gradients
            else (1.0,)
        )
        for alpha in path_alphas:
            model.zero_grad(set_to_none=True)
            capture, metric_hook = _direction_metric_hooks(target_hook, target_position, direction)
            # alpha=0 is fully ablated and alpha=1 is clean.  The right-endpoint
            # Riemann sum matches the EAP-IG convention in the source repo.
            path_hooks = all_layer_ablation_hooks(model, direction, strength=1.0 - alpha)
            with model.hooks(
                fwd_hooks=[("hook_embed", _input_requires_grad)] + path_hooks + metric_hook,
                bwd_hooks=bwd_hooks,
            ):
                _run_forward(model, tokens, attention_mask, path_hooks + metric_hook)
                if capture.activation is None:
                    raise RuntimeError("The target residual metric hook did not fire.")
                capture.activation.retain_grad()
                capture.mean().backward()
            _accumulate_terminal_metric_scores(scores, activation_difference, graph, capture)
            total_gradient_steps += 1

    scores /= total_examples
    if integrated_gradients:
        scores /= config.ig_steps
    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite EAP/EAP-IG edge scores; do not interpret this run.")

    for edge in graph.edges.values():
        edge.score = float(
            scores[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
            ].item()
        )


def attribute_logit_metric(
    model,
    graph,
    prompts: list[str],
    *,
    direction: torch.Tensor,
    refusal_token_ids: list[int],
    config: AttributionConfig,
    integrated_gradients: bool,
    target_hook: str | None = None,
    target_position: int | None = None,
) -> dict:
    """Populate edge scores for a next-token refusal log-odds metric.

    By default, corrupt uses global refusal-direction ablation.  Supplying
    ``target_hook`` and ``target_position`` instead projects the direction out
    once at that source representation, allowing the graph to identify the
    strictly downstream path from that site to the refusal logits.
    """

    if not prompts:
        raise ValueError("Attribution requires at least one harmful prompt.")
    if integrated_gradients and config.ig_steps <= 0:
        raise ValueError("ig_steps must be positive.")
    if not refusal_token_ids:
        raise ValueError("At least one refusal token ID is required.")
    local_projection = target_hook is not None or target_position is not None
    if local_projection and (target_hook is None or target_position is None):
        raise ValueError("Local projection requires both target_hook and target_position.")
    if local_projection and graph.nodes["input"].out_hook != target_hook:
        raise ValueError("Downstream graph input hook does not match the projection site.")

    device = next(model.parameters()).device
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=torch.float32)
    total_examples = 0
    total_gradient_steps = 0
    clean_metric_sum = 0.0
    corrupt_metric_sum = 0.0

    iterator = tqdm(
        _batches(prompts, config.batch_size),
        total=(len(prompts) + config.batch_size - 1) // config.batch_size,
        desc="EAP-IG logits" if integrated_gradients else "EAP logits",
    )
    for prompt_batch in iterator:
        tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=config.max_length)
        batch_size, n_pos = tokens.shape
        total_examples += batch_size
        if local_projection:
            def corrupt_source_transform(source_name: str, activations: torch.Tensor):
                if source_name != target_hook:
                    return activations
                return direction_ablation_hook(
                    direction, position=target_position
                )(activations, None)

            fwd_corrupt, fwd_clean, bwd_hooks, activation_difference = _make_eap_hooks(
                model,
                graph,
                batch_size,
                n_pos,
                scores,
                direction=None,
                include_logit_gradients=True,
                corrupt_source_transform=corrupt_source_transform,
            )
            corrupt_hooks = [
                (
                    target_hook,
                    direction_ablation_hook(direction, position=target_position),
                )
            ]
        else:
            fwd_corrupt, fwd_clean, bwd_hooks, activation_difference = _make_eap_hooks(
                model,
                graph,
                batch_size,
                n_pos,
                scores,
                direction,
                include_logit_gradients=True,
            )
            corrupt_hooks = all_layer_ablation_hooks(model, direction, strength=1.0)

        with torch.no_grad(), model.hooks(fwd_hooks=[*fwd_corrupt, *corrupt_hooks]):
            corrupt_logits = _run_forward(
                model, tokens, attention_mask, [*fwd_corrupt, *corrupt_hooks]
            )
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_clean):
            clean_logits = _run_forward(model, tokens, attention_mask, fwd_clean)
        corrupt_metric_sum += float(
            refusal_log_odds(corrupt_logits, refusal_token_ids).sum().item()
        )
        clean_metric_sum += float(
            refusal_log_odds(clean_logits, refusal_token_ids).sum().item()
        )

        path_alphas: Iterable[float] = (
            (step / config.ig_steps for step in range(1, config.ig_steps + 1))
            if integrated_gradients
            else (1.0,)
        )
        for alpha in path_alphas:
            model.zero_grad(set_to_none=True)
            path_hooks = (
                [
                    (
                        target_hook,
                        direction_ablation_hook(
                            direction,
                            strength=1.0 - alpha,
                            position=target_position,
                        ),
                    )
                ]
                if local_projection
                else all_layer_ablation_hooks(
                    model, direction, strength=1.0 - alpha
                )
            )
            with model.hooks(
                fwd_hooks=[
                    (
                        graph.nodes["input"].out_hook
                        if local_projection
                        else "hook_embed",
                        _input_requires_grad,
                    ),
                    *path_hooks,
                ],
                bwd_hooks=bwd_hooks,
            ):
                logits = _run_forward(model, tokens, attention_mask, path_hooks)
                metric_value = refusal_log_odds(logits, refusal_token_ids).mean()
                metric_value.backward()
            total_gradient_steps += 1

    scores /= total_examples
    if integrated_gradients:
        scores /= config.ig_steps
    if total_gradient_steps == 0:
        raise RuntimeError("No gradient steps were taken.")
    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite EAP/EAP-IG edge scores; do not interpret this run.")

    for edge in graph.edges.values():
        edge.score = float(
            scores[
                graph.forward_index(edge.parent, attn_slice=False),
                graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
            ].item()
        )
    clean_metric = clean_metric_sum / total_examples
    corrupt_metric = corrupt_metric_sum / total_examples
    return {
        "clean_refusal_log_odds": clean_metric,
        "corrupt_refusal_log_odds": corrupt_metric,
        "clean_minus_corrupt": clean_metric - corrupt_metric,
        "projection_scope": "target_only" if local_projection else "global",
    }

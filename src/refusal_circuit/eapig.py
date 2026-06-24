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
from typing import Iterable

import torch
from tqdm.auto import tqdm

from .interventions import all_layer_ablation_hooks
from .metrics import ProjectionCapture
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
    direction: torch.Tensor,
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

    def source_value(activations: torch.Tensor, *, corrupt: bool) -> torch.Tensor:
        value = activations.detach().float()
        if corrupt:
            vector = direction.to(value)
            vector = vector / vector.norm().clamp_min(1e-8)
            value = value - torch.einsum("...d,d->...", value, vector).unsqueeze(-1) * vector
        return value

    def activation_hook(index, sign: float, corrupt: bool, activations: torch.Tensor, hook):
        activation_difference[:, :, index].add_(sign * source_value(activations, corrupt=corrupt))
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
                sign * source_value(effective_heads, corrupt=corrupt)
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
        if isinstance(node, LogitNode):
            continue
        if isinstance(node, AttentionNode):
            if node.layer in processed_attention_layers:
                continue
            processed_attention_layers.add(node.layer)
            attention_nodes[node.layer] = node

        fwd_index = graph.forward_index(node)
        if not isinstance(node, AttentionNode):
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

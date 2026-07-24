"""Faithfulness, necessity, and small behavioural checks for RQ1 circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from tqdm.auto import tqdm

from .interventions import all_layer_ablation_hooks, direction_ablation_hook
from .metrics import MaskedProjectionCapture, ProjectionCapture, is_refusal, refusal_logit_margin
from .model import tokenize_prompts


def _batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _project_away(activations: torch.Tensor, direction: torch.Tensor | None) -> torch.Tensor:
    """Return an activation with a direction removed, without mutating the run."""
    value = activations.detach().float()
    if direction is None:
        return value.clone()
    vector = direction.to(device=value.device, dtype=torch.float32)
    vector = vector / vector.norm().clamp_min(1e-8)
    projected = value - torch.einsum("...d,d->...", value, vector).unsqueeze(-1) * vector
    return projected.detach().clone()


def _source_cache_hooks(
    model,
    graph,
    cache: dict[str, torch.Tensor],
    *,
    corruption_direction=None,
    cache_transform=None,
):
    """Cache graph source contributions, including each attention output bias.

    TransformerLens exposes head contributions at ``attn.hook_result`` but
    Gemma's actual residual write is ``hook_attn_out = sum(head_results) +
    attention_bias``.  We distribute that bias evenly over heads only as a
    bookkeeping convention; their sum is then the exact attention residual
    write, while EAP's head/QKV graph remains unchanged.

    For a corrupt cache the hook list runs before the model intervention and
    projects cached source values itself.  This preserves the source values
    that are actually written by the intervention while letting the model
    execute its complete attention-output ablation normally.
    """
    from eap.graph import AttentionNode

    parents = {edge.parent for edge in graph.edges.values()}
    attention_layers = sorted({node.layer for node in parents if isinstance(node, AttentionNode)})
    direct_names = {
        node.out_hook
        for node in parents
        if not isinstance(node, AttentionNode) and node.out_hook
    }
    raw_results: dict[int, torch.Tensor] = {}

    if corruption_direction is not None and cache_transform is not None:
        raise ValueError("Specify either corruption_direction or cache_transform, not both.")

    def transformed(name: str, activations: torch.Tensor) -> torch.Tensor:
        value = _project_away(activations, corruption_direction)
        return cache_transform(name, value) if cache_transform is not None else value

    def direct_cache(name: str):
        def hook(activations: torch.Tensor, hook):
            cache[name] = transformed(name, activations)
            return activations

        return hook

    def result_cache(layer: int):
        def hook(activations: torch.Tensor, hook):
            raw_results[layer] = activations.detach().clone()
            return activations

        return hook

    def attention_output_cache(layer: int):
        def hook(activations: torch.Tensor, hook):
            result = raw_results[layer]
            attention_bias = activations.detach() - result.sum(dim=2)
            effective_heads = result + attention_bias.unsqueeze(2) / result.shape[2]
            name = f"blocks.{layer}.attn.hook_result"
            cache[name] = transformed(
                name, effective_heads
            )
            return activations

        return hook

    hooks = [(name, direct_cache(name)) for name in direct_names]
    for layer in attention_layers:
        hooks.extend(
            [
                (f"blocks.{layer}.attn.hook_result", result_cache(layer)),
                (f"blocks.{layer}.hook_attn_out", attention_output_cache(layer)),
            ]
        )
    return hooks


def _construction_hooks(graph, corrupt_cache, mixed_cache, *, patch_selected: bool):
    """Patch either selected or unselected EAP edges at each receiving input."""
    from eap.graph import AttentionNode, InputNode, LogitNode, MLPNode

    hooks = []
    for node in graph.nodes.values():
        if isinstance(node, InputNode):
            continue
        # The terminal metric must always receive edge replacements, even when a
        # raw top-k list does not contain one of its direct edges.
        if not node.in_graph and not isinstance(node, LogitNode):
            continue

        def make_hook(current_node, qkv=None):
            def hook(activations: torch.Tensor, hook):
                for edge in current_node.parent_edges:
                    if edge.qkv != qkv or edge.in_graph != patch_selected:
                        continue
                    clean_source = mixed_cache[edge.parent.out_hook][edge.parent.index]
                    corrupt_source = corrupt_cache[edge.parent.out_hook][edge.parent.index]
                    activations[edge.index] = (
                        activations[edge.index] - clean_source + corrupt_source
                    )
                return activations

            return hook

        if isinstance(node, (LogitNode, MLPNode)):
            hooks.append((node.in_hook, make_hook(node)))
        elif isinstance(node, AttentionNode):
            for index, qkv in enumerate("qkv"):
                hooks.append((node.qkv_inputs[index], make_hook(node, qkv=qkv)))
        else:  # pragma: no cover - guarded by EAP graph node types
            raise TypeError(f"Unknown graph node type {type(node)!r}")
    return hooks


def _normal_forward(
    model,
    tokens,
    attention_mask,
    *,
    target_hook: str,
    target_position: int,
    direction: torch.Tensor,
    fwd_hooks=None,
    target_mask: torch.Tensor | None = None,
):
    capture = (
        MaskedProjectionCapture(direction=direction, mask=target_mask)
        if target_mask is not None
        else ProjectionCapture(direction=direction, position=target_position)
    )
    with torch.inference_mode(), model.hooks(fwd_hooks=list(fwd_hooks or []) + [(target_hook, capture.hook)]):
        logits = model(tokens, attention_mask=attention_mask)
    return capture.values().detach(), logits.detach()


def _mixed_forward(
    model,
    graph,
    tokens,
    attention_mask,
    *,
    target_hook: str,
    target_position: int,
    direction: torch.Tensor,
    patch_selected: bool,
    target_mask: torch.Tensor | None = None,
):
    """Run a clean prompt with selected or unselected edges replaced by corrupt cache."""
    corrupt_cache: dict[str, torch.Tensor] = {}
    with torch.inference_mode(), model.hooks(
        fwd_hooks=_source_cache_hooks(
            model, graph, corrupt_cache, corruption_direction=direction
        )
        + all_layer_ablation_hooks(model, direction)
    ):
        model(tokens, attention_mask=attention_mask)

    mixed_cache: dict[str, torch.Tensor] = {}
    capture = (
        MaskedProjectionCapture(direction=direction, mask=target_mask)
        if target_mask is not None
        else ProjectionCapture(direction=direction, position=target_position)
    )
    mixed_hooks = (
        _source_cache_hooks(model, graph, mixed_cache)
        + _construction_hooks(graph, corrupt_cache, mixed_cache, patch_selected=patch_selected)
        + [(target_hook, capture.hook)]
    )
    with torch.inference_mode(), model.hooks(fwd_hooks=mixed_hooks):
        logits = model(tokens, attention_mask=attention_mask)
    return capture.values().detach(), logits.detach()


def mixed_logits_with_intervention(
    model,
    graph,
    tokens,
    attention_mask,
    *,
    corrupt_fwd_hooks,
    corrupt_cache_transform,
    patch_selected: bool,
):
    """Patch graph edges from an arbitrary same-token corrupt intervention.

    The intervention hook mutates the model run, while ``corrupt_cache_transform``
    records the post-intervention value at a graph source whose cache hook must
    execute before that mutation.  Downstream component caches already observe
    the intervention through the ordinary forward computation.
    """
    corrupt_cache: dict[str, torch.Tensor] = {}
    cache_hooks = _source_cache_hooks(
        model,
        graph,
        corrupt_cache,
        cache_transform=corrupt_cache_transform,
    )
    with torch.inference_mode(), model.hooks(
        fwd_hooks=cache_hooks + list(corrupt_fwd_hooks)
    ):
        model(tokens, attention_mask=attention_mask)

    mixed_cache: dict[str, torch.Tensor] = {}
    mixed_hooks = (
        _source_cache_hooks(model, graph, mixed_cache)
        + _construction_hooks(
            graph,
            corrupt_cache,
            mixed_cache,
            patch_selected=patch_selected,
        )
    )
    with torch.inference_mode(), model.hooks(fwd_hooks=mixed_hooks):
        logits = model(tokens, attention_mask=attention_mask)
    return logits.detach()


@dataclass
class ConditionResult:
    projection: torch.Tensor
    logits: torch.Tensor


def _target_direction_ablation_hooks(target_hook: str, direction: torch.Tensor, target_position: int):
    """Ablate the refusal direction only at the metric endpoint.

    This is a stricter causal diagnostic than the global refusal-direction
    ablation used as the corrupt baseline.  It asks whether removing the
    direction specifically at ``h_{l*,t*}`` is enough to change downstream
    logits/generation.
    """

    return [(target_hook, direction_ablation_hook(direction, position=target_position))]


def evaluate_conditions(
    model,
    graph,
    prompts: list[str],
    *,
    direction: torch.Tensor,
    target_hook: str,
    target_position: int,
    batch_size: int = 1,
    max_length: int | None = 512,
) -> dict[str, ConditionResult]:
    """Return clean, corrupt, circuit-only, and circuit-ablated outcomes."""
    outputs = {
        name: {"projection": [], "logits": []}
        for name in (
            "clean",
            "corrupt",
            "target_corrupt",
            "circuit_only",
            "circuit_ablated",
        )
    }
    for prompt_batch in tqdm(
        _batches(prompts, batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc="Evaluating circuit",
    ):
        tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=max_length)
        result_by_name = {
            "clean": _normal_forward(
                model,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
            ),
            "corrupt": _normal_forward(
                model,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                fwd_hooks=all_layer_ablation_hooks(model, direction),
            ),
            "target_corrupt": _normal_forward(
                model,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                fwd_hooks=_target_direction_ablation_hooks(target_hook, direction, target_position),
            ),
            "circuit_only": _mixed_forward(
                model,
                graph,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                patch_selected=False,
            ),
            "circuit_ablated": _mixed_forward(
                model,
                graph,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                patch_selected=True,
            ),
        }
        for name, (projection, logits) in result_by_name.items():
            outputs[name]["projection"].append(projection.cpu())
            outputs[name]["logits"].append(logits[:, -1].float().cpu())
    return {
        name: ConditionResult(
            projection=torch.cat(values["projection"]), logits=torch.cat(values["logits"])
        )
        for name, values in outputs.items()
    }


def normalized_faithfulness(clean: torch.Tensor, corrupt: torch.Tensor, circuit_only: torch.Tensor) -> float:
    denominator = (clean.mean() - corrupt.mean()).item()
    if abs(denominator) < 1e-8:
        return float("nan")
    return ((circuit_only.mean() - corrupt.mean()).item()) / denominator


def normalized_necessity(clean: torch.Tensor, corrupt: torch.Tensor, circuit_ablated: torch.Tensor) -> float:
    denominator = (clean.mean() - corrupt.mean()).item()
    if abs(denominator) < 1e-8:
        return float("nan")
    return ((clean.mean() - circuit_ablated.mean()).item()) / denominator


def first_token_logit_margin(result: ConditionResult, refusal_ids: list[int], answer_ids: list[int]) -> float:
    return refusal_logit_margin(result.logits.unsqueeze(1), refusal_ids, answer_ids).mean().item()


def _greedy_completion(
    model,
    graph,
    input_tokens: torch.Tensor,
    *,
    direction: torch.Tensor,
    target_hook: str,
    target_position: int,
    condition: str,
    max_new_tokens: int,
) -> list[int]:
    """Generate one completion while recomputing same-input corrupt caches.

    This slow, deliberately small diagnostic avoids assuming that corrupt caches
    from the original prompt remain valid after newly generated tokens.
    """
    tokens = input_tokens
    generated: list[int] = []
    eos_id = model.tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        if condition == "normal":
            _, logits = _normal_forward(
                model,
                tokens,
                None,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
            )
        elif condition == "refusal_direction_ablated":
            _, logits = _normal_forward(
                model,
                tokens,
                None,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                fwd_hooks=all_layer_ablation_hooks(model, direction),
            )
        elif condition == "target_direction_ablated":
            _, logits = _normal_forward(
                model,
                tokens,
                None,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                fwd_hooks=_target_direction_ablation_hooks(target_hook, direction, target_position),
            )
        elif condition == "circuit_only":
            _, logits = _mixed_forward(
                model,
                graph,
                tokens,
                None,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                patch_selected=False,
            )
        elif condition == "circuit_ablated":
            _, logits = _mixed_forward(
                model,
                graph,
                tokens,
                None,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                patch_selected=True,
            )
        else:  # pragma: no cover - script validates names
            raise ValueError(f"Unknown behavioural condition {condition!r}.")
        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)
        tokens = torch.cat(
            [tokens, torch.tensor([[next_token]], device=tokens.device, dtype=tokens.dtype)], dim=1
        )
        if eos_id is not None and next_token == eos_id:
            break
    return generated


def behavioural_sanity_check(
    model,
    graph,
    prompts: list[str],
    *,
    direction: torch.Tensor,
    target_hook: str,
    target_position: int,
    refusal_ids: list[int],
    answer_ids: list[int],
    max_prompts: int = 8,
    max_new_tokens: int = 16,
) -> list[dict]:
    """Return aggregate refusal-prefix rates; never persist harmful completions."""
    conditions = (
        "normal",
        "refusal_direction_ablated",
        "target_direction_ablated",
        "circuit_only",
        "circuit_ablated",
    )
    selected = prompts[:max_prompts]
    counts = {condition: 0 for condition in conditions}
    margins = {condition: [] for condition in conditions}
    for prompt in tqdm(selected, desc="Behavioural sanity"):
        tokens, _ = tokenize_prompts(model, [prompt], max_length=None)
        for condition in conditions:
            # First-token margin is captured with the same condition before
            # greedy decoding, then substring matching acts only on its output.
            if condition == "normal":
                projection, logits = _normal_forward(
                    model,
                    tokens,
                    None,
                    target_hook=target_hook,
                    target_position=target_position,
                    direction=direction,
                )
            elif condition == "refusal_direction_ablated":
                projection, logits = _normal_forward(
                    model,
                    tokens,
                    None,
                    target_hook=target_hook,
                    target_position=target_position,
                    direction=direction,
                    fwd_hooks=all_layer_ablation_hooks(model, direction),
                )
            elif condition == "target_direction_ablated":
                projection, logits = _normal_forward(
                    model,
                    tokens,
                    None,
                    target_hook=target_hook,
                    target_position=target_position,
                    direction=direction,
                    fwd_hooks=_target_direction_ablation_hooks(target_hook, direction, target_position),
                )
            else:
                projection, logits = _mixed_forward(
                    model,
                    graph,
                    tokens,
                    None,
                    target_hook=target_hook,
                    target_position=target_position,
                    direction=direction,
                    patch_selected=condition == "circuit_ablated",
                )
            del projection
            margins[condition].append(
                refusal_logit_margin(logits[:, -1:].float(), refusal_ids, answer_ids).item()
            )
            completion_ids = _greedy_completion(
                model,
                graph,
                tokens,
                direction=direction,
                target_hook=target_hook,
                target_position=target_position,
                condition=condition,
                max_new_tokens=max_new_tokens,
            )
            completion = model.tokenizer.decode(completion_ids, skip_special_tokens=True)
            counts[condition] += int(is_refusal(completion))
    return [
        {
            "condition": condition,
            "n_prompts": len(selected),
            "refusal_substring_rate": counts[condition] / max(len(selected), 1),
            "refusal_token_logit_margin": sum(margins[condition]) / max(len(margins[condition]), 1),
            "generated_text_saved": False,
        }
        for condition in conditions
    ]

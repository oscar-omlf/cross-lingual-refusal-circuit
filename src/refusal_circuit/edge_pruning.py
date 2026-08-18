"""Differentiable edge pruning for reconstructive circuit discovery.

The existing EAP-IG pipeline ranks edges by a local attribution score.  That
is useful for causal cuts, but it does not optimize the recursive
``corrupt + selected clean-minus-corrupt deltas`` computation used by a
circuit-only run.  This module puts trainable gates directly into that
recursive computation.

The implementation follows the writer/reader edge masking semantics of
Bhaskar et al. (2024), while retaining this repository's TransformerLens/EAP
graph and natural-prompt counterfactuals.  Model weights remain frozen; only
one hard-concrete gate per EAP edge is trainable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


HARD_CONCRETE_LEFT = -0.1
HARD_CONCRETE_RIGHT = 1.1
HARD_CONCRETE_TEMPERATURE = 2.0 / 3.0
HARD_CONCRETE_EPS = 1e-6
DETERMINISTIC_GATE_FACTOR = 0.8


def hard_concrete_nonzero_probability(
    log_alpha: torch.Tensor,
    *,
    temperature: float = HARD_CONCRETE_TEMPERATURE,
    left: float = HARD_CONCRETE_LEFT,
    right: float = HARD_CONCRETE_RIGHT,
) -> torch.Tensor:
    """Return the differentiable probability that each gate is nonzero."""

    if not left < 0 < right:
        raise ValueError("Hard-concrete limits must straddle zero.")
    cutoff = (0.0 - left) / (right - left)
    logit_cutoff = math.log(cutoff) - math.log1p(-cutoff)
    probability_zero = torch.sigmoid(
        temperature * logit_cutoff - log_alpha
    )
    return 1.0 - probability_zero


def sample_hard_concrete(
    log_alpha: torch.Tensor,
    *,
    temperature: float = HARD_CONCRETE_TEMPERATURE,
    left: float = HARD_CONCRETE_LEFT,
    right: float = HARD_CONCRETE_RIGHT,
) -> torch.Tensor:
    """Sample reparameterized hard-concrete gates in ``[0, 1]``."""

    uniform = torch.empty_like(log_alpha).uniform_(
        HARD_CONCRETE_EPS, 1.0 - HARD_CONCRETE_EPS
    )
    logistic = torch.log(uniform) - torch.log1p(-uniform)
    stretched = torch.sigmoid((logistic + log_alpha) / temperature)
    stretched = stretched * (right - left) + left
    return stretched.clamp(0.0, 1.0)


def deterministic_gate_scores(
    log_alpha: torch.Tensor,
    *,
    temperature: float = HARD_CONCRETE_TEMPERATURE,
) -> torch.Tensor:
    """Return deterministic soft scores used to threshold a trained mask."""

    return torch.sigmoid(
        log_alpha / temperature * DETERMINISTIC_GATE_FACTOR
    )


class EdgeGateSet(nn.Module):
    """One hard-concrete gate for every edge in an EAP graph."""

    def __init__(
        self,
        edge_names: Iterable[str],
        *,
        initial_log_alpha: float = 10.0,
        initialization_noise: float = 0.01,
    ) -> None:
        super().__init__()
        names = tuple(edge_names)
        if not names:
            raise ValueError("An edge gate set cannot be empty.")
        if len(names) != len(set(names)):
            raise ValueError("Edge names must be unique.")
        self.edge_names = names
        self.edge_index = {name: index for index, name in enumerate(names)}
        values = torch.full(
            (len(names),), float(initial_log_alpha), dtype=torch.float32
        )
        if initialization_noise:
            values.normal_(
                mean=float(initial_log_alpha),
                std=float(initialization_noise),
            )
        self.log_alpha = nn.Parameter(values)

    def forward(self, *, sample: bool | None = None) -> torch.Tensor:
        if sample is None:
            sample = self.training
        if sample:
            return sample_hard_concrete(self.log_alpha)
        return deterministic_gate_scores(self.log_alpha)

    def expected_density(self) -> torch.Tensor:
        return hard_concrete_nonzero_probability(self.log_alpha).mean()

    def expected_sparsity(self) -> torch.Tensor:
        return 1.0 - self.expected_density()

    @torch.no_grad()
    def topk_edge_names(self, k: int) -> set[str]:
        if not 0 < k <= len(self.edge_names):
            raise ValueError(
                f"k must be in [1, {len(self.edge_names)}], got {k}."
            )
        indices = deterministic_gate_scores(self.log_alpha).topk(k).indices
        return {self.edge_names[int(index)] for index in indices.cpu()}

    @torch.no_grad()
    def edge_score_dict(self) -> dict[str, float]:
        scores = deterministic_gate_scores(self.log_alpha).cpu()
        return {
            name: float(scores[index].item())
            for index, name in enumerate(self.edge_names)
        }


@dataclass(frozen=True)
class _SourceGroup:
    source_hook: str
    source_is_attention: bool
    receiver_is_attention: bool
    source_heads: tuple[int, ...]
    receiver_heads: tuple[int, ...]
    gate_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Receiver:
    hook: str
    is_attention: bool
    source_groups: tuple[_SourceGroup, ...]


class VectorizedEdgeConstruction:
    """Compile an EAP graph into vectorized gated receiver operations.

    Each receiving hook is reconstructed as

    ``corrupt_receiver + sum_e gate_e * (mixed_parent_e - corrupt_parent_e)``.

    Attention-head writers are retained as a head axis.  All edges from one
    writer hook to one receiver hook are applied with one einsum, avoiding the
    hundreds of thousands of Python-level additions that a literal edge loop
    would require.
    """

    def __init__(self, graph, gates: EdgeGateSet) -> None:
        from eap.graph import AttentionNode, InputNode

        if set(graph.edges) != set(gates.edge_names):
            missing = set(graph.edges) - set(gates.edge_names)
            extra = set(gates.edge_names) - set(graph.edges)
            raise ValueError(
                "Gate and graph edge sets differ: "
                f"missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
            )
        self.graph = graph
        self.gates = gates
        self._source_attention: dict[str, bool] = {}
        for edge in graph.edges.values():
            is_attention = isinstance(edge.parent, AttentionNode)
            previous = self._source_attention.setdefault(
                edge.parent.out_hook, is_attention
            )
            if previous != is_attention:
                raise ValueError(
                    f"Source hook {edge.parent.out_hook!r} mixes node types."
                )

        receiver_edges: dict[str, list] = {}
        receiver_attention: dict[str, bool] = {}
        for edge in graph.edges.values():
            child_is_attention = isinstance(edge.child, AttentionNode)
            hook = edge.hook
            receiver_edges.setdefault(hook, []).append(edge)
            previous = receiver_attention.setdefault(hook, child_is_attention)
            if previous != child_is_attention:
                raise ValueError(f"Receiver hook {hook!r} mixes node types.")

        receivers = []
        for hook, edges in receiver_edges.items():
            by_source: dict[str, list] = {}
            for edge in edges:
                by_source.setdefault(edge.parent.out_hook, []).append(edge)
            groups = []
            for source_hook, source_edges in by_source.items():
                source_is_attention = self._source_attention[source_hook]
                receiver_is_attention = receiver_attention[hook]
                groups.append(
                    _SourceGroup(
                        source_hook=source_hook,
                        source_is_attention=source_is_attention,
                        receiver_is_attention=receiver_is_attention,
                        source_heads=tuple(
                            int(edge.parent.head)
                            if source_is_attention
                            else 0
                            for edge in source_edges
                        ),
                        receiver_heads=tuple(
                            int(edge.child.head)
                            if receiver_is_attention
                            else 0
                            for edge in source_edges
                        ),
                        gate_indices=tuple(
                            gates.edge_index[edge.name]
                            for edge in source_edges
                        ),
                    )
                )
            receivers.append(
                _Receiver(
                    hook=hook,
                    is_attention=receiver_attention[hook],
                    source_groups=tuple(groups),
                )
            )
        self.receivers = tuple(receivers)
        self.receiver_by_hook = {
            receiver.hook: receiver for receiver in self.receivers
        }
        self.source_hooks = tuple(self._source_attention)
        self._index_tensor_cache: dict[
            tuple[str, int | None, tuple[int, ...]], torch.Tensor
        ] = {}

        # Input nodes have no receiver. Every other graph node must be covered
        # by at least one receiving hook, including the terminal metric node.
        non_input_nodes = [
            node
            for node in graph.nodes.values()
            if not isinstance(node, InputNode)
        ]
        expected_receiver_hooks = {
            hook
            for node in non_input_nodes
            for hook in (
                node.qkv_inputs
                if isinstance(node, AttentionNode)
                else (node.in_hook,)
            )
        }
        if expected_receiver_hooks != set(self.receiver_by_hook):
            raise ValueError("Compiled receiver hooks do not cover the graph.")

    def _index_tensor(
        self,
        values: torch.Tensor,
        indices: tuple[int, ...],
    ) -> torch.Tensor:
        key = (values.device.type, values.device.index, indices)
        cached = self._index_tensor_cache.get(key)
        if cached is None:
            cached = torch.tensor(
                indices, device=values.device, dtype=torch.long
            )
            self._index_tensor_cache[key] = cached
        return cached

    def _scatter_weights(
        self,
        values: torch.Tensor,
        flat_indices: tuple[int, ...],
        size: int,
    ) -> torch.Tensor:
        return torch.zeros(
            size, device=values.device, dtype=values.dtype
        ).scatter(0, self._index_tensor(values, flat_indices), values)

    def reconstruct_receiver(
        self,
        hook: str,
        *,
        corrupt_sources: dict[str, torch.Tensor],
        mixed_sources: dict[str, torch.Tensor],
        corrupt_receivers: dict[str, torch.Tensor],
        gate_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return one gated receiver tensor without mutating model state."""

        receiver = self.receiver_by_hook[hook]
        value = corrupt_receivers[hook]
        compute_dtype = value.dtype
        for group in receiver.source_groups:
            source_delta = (
                mixed_sources[group.source_hook]
                - corrupt_sources[group.source_hook].to(
                    device=mixed_sources[group.source_hook].device
                )
            )
            weights = gate_values[
                self._index_tensor(gate_values, group.gate_indices)
            ].to(device=source_delta.device, dtype=compute_dtype)
            source_delta = source_delta.to(dtype=compute_dtype)

            if group.source_is_attention and group.receiver_is_attention:
                source_heads = source_delta.shape[2]
                receiver_heads = value.shape[2]
                flat = tuple(
                    source * receiver_heads + receiver_head
                    for source, receiver_head in zip(
                        group.source_heads,
                        group.receiver_heads,
                        strict=True,
                    )
                )
                matrix = self._scatter_weights(
                    weights, flat, source_heads * receiver_heads
                ).reshape(source_heads, receiver_heads)
                contribution = torch.einsum(
                    "bpsd,sh->bphd", source_delta, matrix
                )
            elif group.source_is_attention:
                source_heads = source_delta.shape[2]
                vector = self._scatter_weights(
                    weights, group.source_heads, source_heads
                )
                contribution = torch.einsum(
                    "bpsd,s->bpd", source_delta, vector
                )
            elif group.receiver_is_attention:
                receiver_heads = value.shape[2]
                vector = self._scatter_weights(
                    weights, group.receiver_heads, receiver_heads
                )
                contribution = (
                    source_delta.unsqueeze(2)
                    * vector.reshape(1, 1, receiver_heads, 1)
                )
            else:
                contribution = source_delta * weights.sum()
            value = value.to(device=contribution.device) + contribution
        return value

    def source_cache_hooks(
        self,
        cache: dict[str, torch.Tensor],
        *,
        detach: bool,
        storage_dtype: torch.dtype | None = None,
        cache_transform=None,
    ):
        """Return hooks that cache exact residual writers.

        TransformerLens' per-head result excludes the attention output bias.
        As in the existing circuit evaluator, the bias is divided equally over
        heads so their sum is the exact attention residual write.
        """

        attention_layers = sorted(
            {
                int(hook.split(".")[1])
                for hook, is_attention in self._source_attention.items()
                if is_attention
            }
        )
        direct_hooks = {
            hook
            for hook, is_attention in self._source_attention.items()
            if not is_attention
        }
        raw_results: dict[int, torch.Tensor] = {}

        def store(name: str, value: torch.Tensor) -> torch.Tensor:
            if cache_transform is not None:
                value = cache_transform(name, value)
            if detach:
                value = value.detach()
            if storage_dtype is not None:
                value = value.to(dtype=storage_dtype)
            return value.clone() if detach else value

        def direct_cache(name: str):
            def hook(activations: torch.Tensor, hook):
                del hook
                cache[name] = store(name, activations)
                return activations

            return hook

        def result_cache(layer: int):
            def hook(activations: torch.Tensor, hook):
                del hook
                raw_results[layer] = activations
                return activations

            return hook

        def attention_output_cache(layer: int):
            def hook(activations: torch.Tensor, hook):
                del hook
                result = raw_results.pop(layer)
                attention_bias = activations - result.sum(dim=2)
                effective = (
                    result
                    + attention_bias.unsqueeze(2) / result.shape[2]
                )
                name = f"blocks.{layer}.attn.hook_result"
                cache[name] = store(name, effective)
                return activations

            return hook

        hooks = [
            (name, direct_cache(name)) for name in sorted(direct_hooks)
        ]
        for layer in attention_layers:
            hooks.extend(
                [
                    (
                        f"blocks.{layer}.attn.hook_result",
                        result_cache(layer),
                    ),
                    (
                        f"blocks.{layer}.hook_attn_out",
                        attention_output_cache(layer),
                    ),
                ]
            )
        return hooks

    def receiver_cache_hooks(
        self,
        cache: dict[str, torch.Tensor],
        *,
        storage_dtype: torch.dtype | None = None,
    ):
        def make_hook(name: str):
            def hook(activations: torch.Tensor, hook):
                del hook
                value = activations.detach()
                if storage_dtype is not None:
                    value = value.to(dtype=storage_dtype)
                cache[name] = value.clone()
                return activations

            return hook

        return [
            (receiver.hook, make_hook(receiver.hook))
            for receiver in self.receivers
        ]

    def construction_hooks(
        self,
        corrupt_sources: dict[str, torch.Tensor],
        mixed_sources: dict[str, torch.Tensor],
        corrupt_receivers: dict[str, torch.Tensor],
        gate_values: torch.Tensor,
    ):
        def make_hook(name: str):
            def hook(activations: torch.Tensor, hook):
                del activations, hook
                return self.reconstruct_receiver(
                    name,
                    corrupt_sources=corrupt_sources,
                    mixed_sources=mixed_sources,
                    corrupt_receivers=corrupt_receivers,
                    gate_values=gate_values,
                )

            return hook

        return [
            (receiver.hook, make_hook(receiver.hook))
            for receiver in self.receivers
        ]


def capture_counterfactual_caches(
    model,
    construction: VectorizedEdgeConstruction,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    fwd_hooks=(),
    storage_dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Cache corrupt writers and receivers for one frozen counterfactual."""

    sources: dict[str, torch.Tensor] = {}
    receivers: dict[str, torch.Tensor] = {}
    hooks = [
        *fwd_hooks,
        *construction.source_cache_hooks(
            sources, detach=True, storage_dtype=storage_dtype
        ),
        *construction.receiver_cache_hooks(
            receivers, storage_dtype=storage_dtype
        ),
    ]
    with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
        model(tokens, attention_mask=attention_mask)
    missing_sources = set(construction.source_hooks) - set(sources)
    missing_receivers = set(construction.receiver_by_hook) - set(receivers)
    if missing_sources or missing_receivers:
        raise RuntimeError(
            "Counterfactual cache hooks did not all fire: "
            f"sources={sorted(missing_sources)}, "
            f"receivers={sorted(missing_receivers)}"
        )
    return sources, receivers


def vector_only_rescue_state(
    clean_state: torch.Tensor,
    corrupt_state: torch.Tensor,
    direction: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Restore only the clean-minus-corrupt coordinate along ``direction``."""

    if clean_state.shape != corrupt_state.shape:
        raise ValueError("Clean and corrupt interface states must match.")
    if mask.shape != clean_state.shape[:-1]:
        raise ValueError("Interface mask must match state batch/position axes.")
    vector = direction.to(
        device=clean_state.device, dtype=clean_state.dtype
    )
    vector = vector / vector.float().norm().clamp_min(1e-8).to(vector.dtype)
    coefficient = (
        (clean_state - corrupt_state) * vector
    ).sum(dim=-1, keepdim=True)
    active = mask.to(
        device=clean_state.device, dtype=clean_state.dtype
    ).unsqueeze(-1)
    return corrupt_state + active * coefficient * vector


def fixed_state_hook(state: torch.Tensor):
    """Return a hook that replaces a residual state with a fixed tensor."""

    def hook(activations: torch.Tensor, hook):
        del hook
        if activations.shape != state.shape:
            raise ValueError(
                "Fixed interface state shape changed: "
                f"{tuple(activations.shape)} != {tuple(state.shape)}"
            )
        return state.to(device=activations.device, dtype=activations.dtype)

    return hook


def complete_path_edge_names(graph, edge_names: set[str]) -> set[str]:
    """Prune an arbitrary mask to edges on complete source-to-sink paths."""

    unknown = edge_names - set(graph.edges)
    if unknown:
        raise ValueError(f"Unknown graph edges: {sorted(unknown)[:3]}")
    outgoing: dict[str, list] = {}
    incoming: dict[str, list] = {}
    for name in edge_names:
        edge = graph.edges[name]
        outgoing.setdefault(edge.parent.name, []).append(edge)
        incoming.setdefault(edge.child.name, []).append(edge)

    source_nodes = {"input"}
    frontier = ["input"]
    while frontier:
        parent = frontier.pop()
        for edge in outgoing.get(parent, ()):
            if edge.child.name not in source_nodes:
                source_nodes.add(edge.child.name)
                frontier.append(edge.child.name)

    sink_nodes = {"logits"}
    frontier = ["logits"]
    while frontier:
        child = frontier.pop()
        for edge in incoming.get(child, ()):
            if edge.parent.name not in sink_nodes:
                sink_nodes.add(edge.parent.name)
                frontier.append(edge.parent.name)

    return {
        name
        for name in edge_names
        if graph.edges[name].parent.name in source_nodes
        and graph.edges[name].child.name in sink_nodes
    }

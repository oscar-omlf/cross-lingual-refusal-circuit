from __future__ import annotations

import torch

from refusal_circuit.edge_pruning import (
    EdgeGateSet,
    VectorizedEdgeConstruction,
    deterministic_gate_scores,
    hard_concrete_nonzero_probability,
    vector_only_rescue_state,
)


def test_hard_concrete_probability_and_scores_are_monotonic():
    values = torch.tensor([-10.0, 0.0, 10.0])
    probabilities = hard_concrete_nonzero_probability(values)
    scores = deterministic_gate_scores(values)
    assert torch.all(probabilities[1:] > probabilities[:-1])
    assert torch.all(scores[1:] > scores[:-1])
    assert 0 < probabilities[0] < probabilities[-1] < 1


def test_edge_gate_topk_is_deterministic():
    gates = EdgeGateSet(
        ("a", "b", "c"),
        initial_log_alpha=0.0,
        initialization_noise=0.0,
    )
    with torch.no_grad():
        gates.log_alpha.copy_(torch.tensor([-1.0, 3.0, 1.0]))
    assert gates.topk_edge_names(2) == {"b", "c"}


def test_vector_only_rescue_changes_no_orthogonal_coordinate():
    direction = torch.tensor([1.0, 1.0, 0.0])
    direction = direction / direction.norm()
    clean = torch.tensor(
        [[[4.0, 2.0, 7.0], [9.0, 1.0, 3.0]]]
    )
    corrupt = torch.tensor(
        [[[1.0, 1.0, 5.0], [2.0, 4.0, 8.0]]]
    )
    mask = torch.tensor([[True, False]])
    rescued = vector_only_rescue_state(clean, corrupt, direction, mask)

    clean_projection = torch.einsum("bpd,d->bp", clean, direction)
    corrupt_projection = torch.einsum("bpd,d->bp", corrupt, direction)
    rescued_projection = torch.einsum("bpd,d->bp", rescued, direction)
    assert torch.allclose(rescued_projection[:, 0], clean_projection[:, 0])
    assert torch.allclose(rescued_projection[:, 1], corrupt_projection[:, 1])

    clean_delta = clean - corrupt
    rescued_delta = rescued - corrupt
    orthogonal = clean_delta - torch.einsum(
        "bpd,d->bp", clean_delta, direction
    ).unsqueeze(-1) * direction
    assert torch.allclose(
        rescued_delta[:, 0] - clean_delta[:, 0],
        -orthogonal[:, 0],
        atol=1e-6,
    )
    assert torch.equal(rescued[:, 1], corrupt[:, 1])


def _manual_receiver(graph, hook, corrupt_sources, mixed_sources, corrupt_receiver):
    output = corrupt_receiver.clone()
    for edge in graph.edges.values():
        if edge.hook != hook:
            continue
        delta = (
            mixed_sources[edge.parent.out_hook][edge.parent.index]
            - corrupt_sources[edge.parent.out_hook][edge.parent.index]
        )
        output[edge.index] += delta
    return output


def test_vectorized_all_open_construction_matches_literal_edge_algebra():
    from eap.graph import AttentionNode, Graph

    graph = Graph.from_model(
        {"n_layers": 2, "n_heads": 2, "parallel_attn_mlp": False}
    )
    graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
    gates = EdgeGateSet(
        graph.edges,
        initial_log_alpha=0.0,
        initialization_noise=0.0,
    )
    construction = VectorizedEdgeConstruction(graph, gates)
    generator = torch.Generator().manual_seed(7)
    batch, positions, hidden = 1, 3, 5

    corrupt_sources = {}
    mixed_sources = {}
    for hook, is_attention in construction._source_attention.items():
        shape = (
            (batch, positions, 2, hidden)
            if is_attention
            else (batch, positions, hidden)
        )
        corrupt_sources[hook] = torch.randn(shape, generator=generator)
        mixed_sources[hook] = torch.randn(shape, generator=generator)

    corrupt_receivers = {}
    for receiver in construction.receivers:
        shape = (
            (batch, positions, 2, hidden)
            if receiver.is_attention
            else (batch, positions, hidden)
        )
        corrupt_receivers[receiver.hook] = torch.randn(
            shape, generator=generator
        )

    all_open = torch.ones(len(graph.edges), requires_grad=True)
    total = torch.tensor(0.0)
    for receiver in construction.receivers:
        actual = construction.reconstruct_receiver(
            receiver.hook,
            corrupt_sources=corrupt_sources,
            mixed_sources=mixed_sources,
            corrupt_receivers=corrupt_receivers,
            gate_values=all_open,
        )
        expected = _manual_receiver(
            graph,
            receiver.hook,
            corrupt_sources,
            mixed_sources,
            corrupt_receivers[receiver.hook],
        )
        assert torch.allclose(actual, expected, atol=1e-5)
        total = total + actual.sum()
    total.backward()
    assert all_open.grad is not None
    assert torch.isfinite(all_open.grad).all()


def test_vectorized_all_closed_construction_is_corrupt_receiver():
    from eap.graph import Graph

    graph = Graph.from_model(
        {"n_layers": 1, "n_heads": 2, "parallel_attn_mlp": False}
    )
    graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
    gates = EdgeGateSet(graph.edges, initialization_noise=0.0)
    construction = VectorizedEdgeConstruction(graph, gates)
    shape = (1, 2, 4)
    corrupt_sources = {}
    mixed_sources = {}
    for hook, is_attention in construction._source_attention.items():
        source_shape = (1, 2, 2, 4) if is_attention else shape
        corrupt_sources[hook] = torch.randn(source_shape)
        mixed_sources[hook] = torch.randn(source_shape)
    corrupt_receivers = {
        receiver.hook: torch.randn(
            (1, 2, 2, 4) if receiver.is_attention else shape
        )
        for receiver in construction.receivers
    }
    closed = torch.zeros(len(graph.edges))
    for receiver in construction.receivers:
        actual = construction.reconstruct_receiver(
            receiver.hook,
            corrupt_sources=corrupt_sources,
            mixed_sources=mixed_sources,
            corrupt_receivers=corrupt_receivers,
            gate_values=closed,
        )
        assert torch.equal(actual, corrupt_receivers[receiver.hook])

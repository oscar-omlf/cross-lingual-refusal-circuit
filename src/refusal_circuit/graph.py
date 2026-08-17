"""A target-truncated EAP graph and serialisable edge metadata for RQ1."""

from __future__ import annotations

import json
import heapq
from pathlib import Path

from .paths import ensure_parent, write_json


def make_projection_graph(model, target_layer: int):
    """Build an EAP graph ending at ``blocks.target_layer.hook_resid_pre``.

    The upstream EAP graph normally terminates at logits.  For RQ1 the terminal
    node is repurposed as a scalar residual metric node, and the graph is
    truncated before the selected ``resid_pre`` layer.  This is the minimal
    extension needed to attribute the requested downstream representation.
    """
    if target_layer <= 0 or target_layer >= model.cfg.n_layers:
        raise ValueError(
            f"target_layer must be in [1, {model.cfg.n_layers - 1}], got {target_layer}."
        )
    try:
        from eap.graph import Graph
    except ImportError as exc:
        raise RuntimeError(
            "The local EAP-IG package is not installed. Run `pip install -e EAP-IG`."
        ) from exc

    graph = Graph.from_model(
        {
            "n_layers": target_layer,
            "n_heads": model.cfg.n_heads,
            "parallel_attn_mlp": model.cfg.parallel_attn_mlp,
        }
    )
    # In ordinary EAP, clean and corrupt prompts differ at ``hook_embed``. In
    # RQ1 they share tokens and instead differ because global directional
    # ablation begins at layer 0's residual input.  Treat that *post-ablation*
    # residual input as the graph's source node.  Otherwise the clean/corrupt
    # input node would be identical and an empty stitched circuit could not
    # reproduce the intervention-generated corrupt baseline.
    graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
    terminal = graph.nodes["logits"]
    terminal.layer = target_layer
    terminal.in_hook = f"blocks.{target_layer}.hook_resid_pre"
    return graph


def make_logit_graph(model):
    """Build the ordinary full EAP graph ending at final residual logits.

    Clean and corrupt inputs are identical strings in our refusal-direction
    experiments; they differ only by the global residual-direction ablation.
    Therefore, as in ``make_projection_graph``, the graph input is treated as
    the post-ablation residual stream at layer 0 rather than token embeddings.
    """

    try:
        from eap.graph import Graph
    except ImportError as exc:
        raise RuntimeError(
            "The local EAP-IG package is not installed. Run `pip install -e EAP-IG`."
        ) from exc

    graph = Graph.from_model(
        {
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "parallel_attn_mlp": model.cfg.parallel_attn_mlp,
        }
    )
    graph.nodes["input"].out_hook = "blocks.0.hook_resid_pre"
    return graph


def make_steering_to_logits_graph(model, steering_layer: int):
    """Build the downstream graph from a persona-steering site to logits.

    A response-time persona vector is injected at ``blocks.L.hook_resid_pre``.
    Treating that intervention as if it originated at the embedding node would
    create impossible edges from the layer-L activation back into earlier
    layers.  This graph therefore makes the steering site the input node and
    removes every node and edge upstream of it.  Node layer numbers and edge
    names remain absolute model coordinates, so saved circuits can be patched
    directly into the original model.

    The EAP package's score arrays retain their full-model indexing.  This uses
    a little more temporary memory than a reindexed graph, but avoids a second,
    error-prone coordinate system for circuit serialization and evaluation.
    """

    if not 0 <= steering_layer < model.cfg.n_layers:
        raise ValueError(
            f"steering_layer must be in [0, {model.cfg.n_layers - 1}], got {steering_layer}."
        )
    try:
        from eap.graph import Graph, InputNode, LogitNode
    except ImportError as exc:
        raise RuntimeError(
            "The local EAP-IG package is not installed. Run `pip install -e EAP-IG`."
        ) from exc

    graph = Graph.from_model(
        {
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "parallel_attn_mlp": model.cfg.parallel_attn_mlp,
        }
    )
    input_node = graph.nodes["input"]
    input_node.layer = steering_layer
    input_node.out_hook = f"blocks.{steering_layer}.hook_resid_pre"

    retained_nodes = {
        name: node
        for name, node in graph.nodes.items()
        if isinstance(node, (InputNode, LogitNode)) or node.layer >= steering_layer
    }
    retained_node_set = set(retained_nodes.values())
    retained_edges = {
        name: edge
        for name, edge in graph.edges.items()
        if edge.parent in retained_node_set and edge.child in retained_node_set
    }

    # Rebuild adjacency from the retained edge set so greedy circuit selection
    # and patching cannot traverse an upstream edge that is absent from
    # ``graph.edges``.
    for node in retained_node_set:
        node.parents.clear()
        node.parent_edges.clear()
        node.children.clear()
        node.child_edges.clear()
    for edge in retained_edges.values():
        edge.parent.children.add(edge.child)
        edge.parent.child_edges.add(edge)
        edge.child.parents.add(edge.parent)
        edge.child.parent_edges.add(edge)
    graph.nodes = retained_nodes
    graph.edges = retained_edges
    return graph


def _node_metadata(node) -> dict:
    if node.name == "input":
        return {"name": node.name, "type": "input"}
    if node.name == "logits":
        if str(getattr(node, "in_hook", "")).endswith("hook_resid_post"):
            return {"name": "logits", "type": "logit_metric", "layer": node.layer}
        return {"name": "refusal_projection", "type": "residual_metric", "layer": node.layer}
    if node.name.startswith("a"):
        return {"name": node.name, "type": "attention_head", "layer": node.layer, "head": node.head}
    return {"name": node.name, "type": "mlp", "layer": node.layer}


def edge_record(edge) -> dict:
    return {
        "name": edge.name,
        "source": _node_metadata(edge.parent),
        "target": _node_metadata(edge.child),
        "qkv": edge.qkv,
        "score": float(edge.score),
        "abs_score": abs(float(edge.score)),
    }


def sorted_edges(graph) -> list:
    return sorted(
        graph.edges.values(),
        key=lambda edge: (-abs(float(edge.score)), edge.name),
    )


def circuit_connectivity(graph, edge_names: set[str] | None = None) -> dict:
    """Return source/sink reachability diagnostics for a selected edge set."""

    selected = (
        {edge.name for edge in graph.edges.values() if edge.in_graph}
        if edge_names is None
        else set(edge_names)
    )
    unknown = selected - set(graph.edges)
    if unknown:
        raise ValueError(
            f"Connectivity audit contains unknown edges: {sorted(unknown)[:3]}"
        )

    outgoing: dict[str, list] = {}
    incoming: dict[str, list] = {}
    selected_nodes: set[str] = set()
    for name in selected:
        edge = graph.edges[name]
        outgoing.setdefault(edge.parent.name, []).append(edge)
        incoming.setdefault(edge.child.name, []).append(edge)
        selected_nodes.update((edge.parent.name, edge.child.name))

    source_reachable = {"input"}
    frontier = ["input"]
    while frontier:
        parent = frontier.pop()
        for edge in outgoing.get(parent, ()):
            child = edge.child.name
            if child not in source_reachable:
                source_reachable.add(child)
                frontier.append(child)

    sink_reachable = {"logits"}
    frontier = ["logits"]
    while frontier:
        child = frontier.pop()
        for edge in incoming.get(child, ()):
            parent = edge.parent.name
            if parent not in sink_reachable:
                sink_reachable.add(parent)
                frontier.append(parent)

    source_reachable_edges = {
        name for name in selected if graph.edges[name].parent.name in source_reachable
    }
    sink_reachable_edges = {
        name for name in selected if graph.edges[name].child.name in sink_reachable
    }
    complete_path_edges = source_reachable_edges & sink_reachable_edges
    has_path = "logits" in source_reachable
    return {
        "selected_edges": len(selected),
        "selected_nodes": len(selected_nodes),
        "source_reachable_nodes": len(source_reachable & selected_nodes),
        "sink_reachable_nodes": len(sink_reachable & selected_nodes),
        "source_reachable_edges": len(source_reachable_edges),
        "sink_reachable_edges": len(sink_reachable_edges),
        "complete_path_edges": len(complete_path_edges),
        "source_disconnected_edges": len(selected - source_reachable_edges),
        "sink_disconnected_edges": len(selected - sink_reachable_edges),
        "off_path_edges": len(selected - complete_path_edges),
        "has_source_to_sink_path": has_path,
        "all_edges_on_source_to_sink_paths": bool(selected)
        and has_path
        and selected == complete_path_edges,
    }


def validate_intervention_connectivity(
    connectivities: dict[str, dict],
    *,
    require_source_to_sink: bool,
) -> None:
    """Validate edge sets for causal interception or reconstruction."""

    for scope, connectivity in connectivities.items():
        if connectivity["selected_edges"] <= 0:
            raise ValueError(f"The {scope} intervention set is empty.")
        if connectivity["sink_disconnected_edges"] != 0:
            raise ValueError(
                f"The {scope} intervention contains edges disconnected from "
                "its measured endpoint."
            )
        if (
            require_source_to_sink
            and not connectivity["all_edges_on_source_to_sink_paths"]
        ):
            raise ValueError(
                f"The {scope} reconstruction is not a complete source-to-sink "
                "circuit. Causal cut sets remain valid only for ablation and "
                "mediation tests."
            )


def _prune_selected_edges(graph, *, require_source: bool) -> None:
    """Prune selected edges to sink paths or complete input-to-sink paths."""

    selected = {edge.name for edge in graph.edges.values() if edge.in_graph}
    outgoing: dict[str, list] = {}
    incoming: dict[str, list] = {}
    for name in selected:
        edge = graph.edges[name]
        outgoing.setdefault(edge.parent.name, []).append(edge)
        incoming.setdefault(edge.child.name, []).append(edge)

    sink_reachable = {"logits"}
    frontier = ["logits"]
    while frontier:
        child = frontier.pop()
        for edge in incoming.get(child, ()):
            parent = edge.parent.name
            if parent not in sink_reachable:
                sink_reachable.add(parent)
                frontier.append(parent)

    retained = {
        name for name in selected if graph.edges[name].child.name in sink_reachable
    }
    if require_source:
        source_reachable = {"input"}
        frontier = ["input"]
        while frontier:
            parent = frontier.pop()
            for edge in outgoing.get(parent, ()):
                child = edge.child.name
                if child not in source_reachable:
                    source_reachable.add(child)
                    frontier.append(child)
        retained = {
            name
            for name in retained
            if graph.edges[name].parent.name in source_reachable
        }

    for node in graph.nodes.values():
        node.in_graph = False
    for edge in graph.edges.values():
        edge.in_graph = edge.name in retained
        if edge.in_graph:
            edge.parent.in_graph = True
            edge.child.in_graph = True


def paper_greedy_edges(graph, requested_k: int, *, prune_parentless: bool = True) -> list:
    """Select a connected circuit using Hanna et al.'s greedy procedure.

    Starting at the terminal metric node, repeatedly add the highest absolute
    score among edges whose child is already in the circuit.  This is the
    upstream EAP-IG ``Graph.apply_greedy`` implementation, followed by the
    paper's recursive pruning step.  It is deliberately distinct from merely
    taking the globally score-ranked top-k edges.
    """
    if requested_k <= 0:
        raise ValueError(f"requested_k must be positive, got {requested_k}.")
    original_requested_k = requested_k
    for node in graph.nodes.values():
        node.in_graph = False
    for edge in graph.edges.values():
        edge.in_graph = False
    graph.nodes["logits"].in_graph = True

    # The upstream ``Graph.apply_greedy`` implements the right algorithm but
    # repeatedly nests ``heapq.merge`` calls.  On larger Qwen graphs this can
    # hit Python's recursion limit during postprocessing after all expensive
    # attribution work has already succeeded.  Keep the paper's frontier rule
    # -- repeatedly add the largest absolute-score edge whose child is already
    # in the circuit -- but implement it iteratively.
    frontier = [
        (-abs(float(edge.score)), edge.name, edge)
        for edge in graph.nodes["logits"].parent_edges
    ]
    heapq.heapify(frontier)
    seen: set[str] = set()
    while requested_k > 0 and frontier:
        _neg_score, _name, edge = heapq.heappop(frontier)
        if edge.name in seen:
            continue
        seen.add(edge.name)
        requested_k -= 1
        edge.in_graph = True
        edge.child.in_graph = True
        parent = edge.parent
        if not parent.in_graph:
            parent.in_graph = True
            for parent_edge in parent.parent_edges:
                if parent_edge.name not in seen and not parent_edge.in_graph:
                    heapq.heappush(
                        frontier,
                        (
                            -abs(float(parent_edge.score)),
                            parent_edge.name,
                            parent_edge,
                        ),
                    )
    frontier_selected = {edge.name for edge in graph.edges.values() if edge.in_graph}
    pre_pruning = circuit_connectivity(graph, frontier_selected)
    _prune_selected_edges(graph, require_source=prune_parentless)
    selected = [edge for edge in graph.edges.values() if edge.in_graph]
    post_pruning = circuit_connectivity(
        graph, {edge.name for edge in selected}
    )
    graph.last_selection_diagnostics = {
        "requested_frontier_edges": original_requested_k,
        "frontier_selected_edges": len(frontier_selected),
        "pruning": (
            "complete_source_to_sink_paths"
            if prune_parentless
            else "sink_paths_only"
        ),
        "before_pruning": pre_pruning,
        "after_pruning": post_pruning,
    }
    if prune_parentless and selected and not post_pruning[
        "all_edges_on_source_to_sink_paths"
    ]:
        raise RuntimeError("Source-to-sink circuit pruning left an off-path edge.")
    return selected


def write_edge_scores_parquet(graph, path: Path) -> None:
    import pandas as pd

    rows = []
    for edge in graph.edges.values():
        record = edge_record(edge)
        rows.append(
            {
                "name": record["name"],
                "source": record["source"]["name"],
                "source_type": record["source"]["type"],
                "source_layer": record["source"].get("layer"),
                "source_head": record["source"].get("head"),
                "target": record["target"]["name"],
                "target_type": record["target"]["type"],
                "target_layer": record["target"].get("layer"),
                "target_head": record["target"].get("head"),
                "qkv": record["qkv"],
                "score": record["score"],
                "abs_score": record["abs_score"],
            }
        )
    ensure_parent(path)
    (
        pd.DataFrame(rows)
        .sort_values(
            ["abs_score", "name"],
            ascending=[False, True],
            kind="mergesort",
        )
        .to_parquet(path, index=False)
    )


def write_circuit_jsons(
    graph,
    *,
    method: str,
    target_layer: int,
    target_position: int,
    output_dir: Path,
    k_values: list[int],
    prune_parentless: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for requested_k in k_values:
        selected = paper_greedy_edges(
            graph,
            min(requested_k, len(graph.edges)),
            prune_parentless=prune_parentless,
        )
        selection = (
            "paper_greedy_absolute_score_with_recursive_pruning"
            if prune_parentless
            else "paper_greedy_absolute_score_with_childless_pruning_only"
        )
        payload = {
            "method": method,
            "selection": selection,
            "requested_k": requested_k,
            "retained_edges": len(selected),
            "target": {
                "hook": f"blocks.{target_layer}.hook_resid_pre",
                "layer": target_layer,
                "position": target_position,
                "metric": "residual_projection_onto_refusal_direction",
            },
            "edges": [edge_record(edge) for edge in selected],
        }
        path = output_dir / f"{method}_top_{requested_k}.json"
        write_json(path, payload)
        paths.append(path)
    return paths


def write_logit_circuit_jsons(
    graph,
    *,
    method: str,
    refusal_token_ids: list[int],
    output_dir: Path,
    k_values: list[int],
    prune_parentless: bool = True,
) -> list[Path]:
    """Write greedy circuit JSONs for a next-token refusal log-odds target."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for requested_k in k_values:
        selected = paper_greedy_edges(
            graph,
            min(requested_k, len(graph.edges)),
            prune_parentless=prune_parentless,
        )
        selection = (
            "paper_greedy_absolute_score_with_recursive_pruning"
            if prune_parentless
            else "paper_greedy_absolute_score_with_childless_pruning_only"
        )
        payload = {
            "method": method,
            "selection": selection,
            "requested_k": requested_k,
            "retained_edges": len(selected),
            "target": {
                "hook": graph.nodes["logits"].in_hook,
                "layer": graph.nodes["logits"].layer,
                "position": -1,
                "metric": "next_token_refusal_log_odds",
                "refusal_token_ids": list(refusal_token_ids),
            },
            "edges": [edge_record(edge) for edge in selected],
        }
        path = output_dir / f"{method}_top_{requested_k}.json"
        write_json(path, payload)
        paths.append(path)
    return paths


def write_top_edges_summary(graph, path: Path, *, method: str, n_edges: int = 25) -> None:
    """Write a compact, reviewable Markdown view of the highest-score edges."""
    rows = sorted_edges(graph)[:n_edges]
    lines = [
        "# RQ1 top edges",
        "",
        f"Method: `{method}`. This is the global absolute-score ranking; circuit JSONs use paper greedy selection.",
        "",
        "| Rank | Edge | Q/K/V | Score | Absolute score |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for rank, edge in enumerate(rows, start=1):
        record = edge_record(edge)
        lines.append(
            f"| {rank} | `{record['name']}` | {record['qkv'] or ''} | "
            f"{record['score']:.6g} | {record['abs_score']:.6g} |"
        )
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_circuit_edge_names(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {edge["name"] for edge in payload["edges"]}


def set_selected_edges(graph, edge_names: set[str]) -> None:
    """Mark an already selected edge set without changing its connectivity.

    Evaluation patches *every* non-selected edge at every receiving node, so a
    disconnected cut set remains algebraically well-defined for diagnostics.
    Circuit-only claims should separately require ``circuit_connectivity`` to
    confirm complete source-to-sink paths. The terminal metric node is always
    enabled to ensure its incoming residual edges are patched too.
    """
    unknown = edge_names - set(graph.edges)
    if unknown:
        raise ValueError(f"Circuit contains unknown graph edges: {sorted(unknown)[:3]}")
    for node in graph.nodes.values():
        node.in_graph = False
    graph.nodes["logits"].in_graph = True
    for edge in graph.edges.values():
        edge.in_graph = edge.name in edge_names
        if edge.in_graph:
            edge.parent.in_graph = True
            edge.child.in_graph = True

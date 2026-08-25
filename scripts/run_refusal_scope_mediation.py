#!/usr/bin/env python3
"""Test refusal intervention scope, trajectory, rescue, and random controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "EAP-IG")]

from refusal_circuit.circuit import _mixed_forward
from refusal_circuit.graph import (
    load_circuit_edge_names,
    make_projection_graph,
    set_selected_edges,
)
from refusal_circuit.interventions import (
    all_layer_ablation_hooks,
    direction_ablation_hook,
    normalized,
    state_component_replacement,
)
from refusal_circuit.metrics import ProjectionCapture, is_refusal, refusal_log_odds
from refusal_circuit.model import (
    configure_standard_hooks,
    format_instructions,
    tokenize_prompts,
)
from refusal_circuit.paths import write_json
from refusal_circuit.workflow import _load_vector, _model, load_config


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(records: list[dict], condition: str, field: str) -> float:
    values = [float(row[field]) for row in records if row["condition"] == condition]
    return sum(values) / len(values)


def state_rescue_hook(
    clean_state: torch.Tensor,
    direction: torch.Tensor,
    *,
    component: str,
    position: int | None,
):
    def hook(activations: torch.Tensor, hook):
        del hook
        if position is None:
            return state_component_replacement(
                activations, clean_state, direction, component=component
            )
        index = position if position >= 0 else activations.shape[1] + position
        changed = activations.clone()
        changed[:, index : index + 1] = state_component_replacement(
            activations[:, index : index + 1],
            clean_state[:, index : index + 1],
            direction,
            component=component,
        )
        return changed

    return hook


def phase_ablation_hooks(model, direction: torch.Tensor, scope: str):
    if scope not in {"prompt_only", "generation_only", "recurrent"}:
        raise ValueError(scope)

    def make_hook():
        def hook(activations: torch.Tensor, hook):
            del hook
            prefill = activations.shape[1] > 1
            active = (
                scope == "recurrent"
                or (scope == "prompt_only" and prefill)
                or (scope == "generation_only" and not prefill)
            )
            return (
                direction_ablation_hook(direction)(activations, None)
                if active
                else activations
            )

        return hook

    hooks = []
    for layer in range(model.cfg.n_layers):
        hooks.extend(
            [
                (f"blocks.{layer}.hook_resid_pre", make_hook()),
                (f"blocks.{layer}.hook_attn_out", make_hook()),
                (f"blocks.{layer}.hook_mlp_out", make_hook()),
            ]
        )
    return hooks


def node_kind(node) -> str:
    if node.name == "input":
        return "input"
    if node.name == "logits":
        return "logits"
    return "attention" if node.name.startswith("a") else "mlp"


def random_edge_sets(graph, selected: set[str], seeds: list[int]) -> dict[int, set[str]]:
    edges = list(graph.edges.values())

    def exact_key(edge):
        return (
            node_kind(edge.parent),
            getattr(edge.parent, "layer", None),
            node_kind(edge.child),
            getattr(edge.child, "layer", None),
            edge.qkv,
        )

    def relaxed_key(edge):
        return (
            node_kind(edge.parent),
            node_kind(edge.child),
            getattr(edge.child, "layer", None),
            edge.qkv,
        )

    selected_edges = [graph.edges[name] for name in sorted(selected)]
    outputs = {}
    for seed in seeds:
        rng = random.Random(seed)
        available = {edge.name: edge for edge in edges if edge.name not in selected}
        exact = defaultdict(list)
        relaxed = defaultdict(list)
        for edge in available.values():
            exact[exact_key(edge)].append(edge.name)
            relaxed[relaxed_key(edge)].append(edge.name)
        chosen: set[str] = set()
        for source in selected_edges:
            candidates = [
                name
                for name in exact[exact_key(source)]
                if name in available
            ]
            if not candidates:
                candidates = [
                    name
                    for name in relaxed[relaxed_key(source)]
                    if name in available
                ]
            if not candidates:
                candidates = list(available)
            name = rng.choice(candidates)
            chosen.add(name)
            available.pop(name)
        outputs[seed] = chosen
    return outputs


def capture_trajectory_hooks(
    model,
    direction: torch.Tensor,
    start_layer: int,
    store: dict[int, torch.Tensor],
):
    hooks = []
    for layer in range(start_layer, model.cfg.n_layers):
        def capture(activations: torch.Tensor, hook, *, layer_index=layer):
            del hook
            vector = normalized(direction, activations)
            store[layer_index] = torch.einsum(
                "bpd,d->bp", activations, vector
            ).detach().float().cpu()
            return activations

        hooks.append((f"blocks.{layer}.hook_resid_pre", capture))
    return hooks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(args.config)
    model = _model(config, hook_mode="eap")
    direction, _raw, vector = _load_vector(config, model)
    layer = int(vector["layer"])
    position = int(vector["position"])
    target_hook = str(vector["target_hook"])
    refusal_token_ids = [
        int(value) for value in config["eapig"]["refusal_token_ids"]
    ]
    rows = [
        row
        for row in read_jsonl(args.dataset.resolve())
        if row.get("label") == "harmful"
    ][: args.num_examples]
    prompts = format_instructions(
        model.tokenizer, [row["instruction"] for row in rows]
    )

    graph = make_projection_graph(model, layer)
    selected = load_circuit_edge_names(args.circuit.resolve())
    random_sets = random_edge_sets(
        graph,
        selected,
        [int(value) for value in args.random_seeds.split(",") if value],
    )

    records: list[dict] = []
    trajectories: list[dict] = []
    for row, prompt in zip(rows, prompts, strict=True):
        tokens, mask = tokenize_prompts(
            model, [prompt], max_length=config["eapig"].get("max_length", 512)
        )
        clean_state: dict[str, torch.Tensor] = {}

        def save_clean(activations: torch.Tensor, hook):
            del hook
            clean_state["value"] = activations.detach().clone()
            return activations

        condition_hooks = {
            "normal": [(target_hook, save_clean)],
            "local_projection": [
                (
                    target_hook,
                    direction_ablation_hook(direction, position=position),
                )
            ],
            "layer15_all_prompt_positions": [
                (target_hook, direction_ablation_hook(direction))
            ],
            "global_prompt_projection": all_layer_ablation_hooks(model, direction),
        }
        for condition, hooks in condition_hooks.items():
            capture = ProjectionCapture(direction=direction, position=position)
            with torch.inference_mode(), model.hooks(
                fwd_hooks=[*hooks, (target_hook, capture.hook)]
            ):
                logits = model(tokens, attention_mask=mask)
            records.append(
                {
                    "id": row["id"],
                    "condition": condition,
                    "target_projection": float(capture.values().item()),
                    "refusal_log_odds": float(
                        refusal_log_odds(logits, refusal_token_ids).item()
                    ),
                }
            )

        clean_trajectory: dict[int, torch.Tensor] = {}
        local_trajectory: dict[int, torch.Tensor] = {}
        with torch.inference_mode(), model.hooks(
            fwd_hooks=capture_trajectory_hooks(
                model, direction, layer, clean_trajectory
            )
        ):
            model(tokens, attention_mask=mask)
        with torch.inference_mode(), model.hooks(
            fwd_hooks=[
                (
                    target_hook,
                    direction_ablation_hook(direction, position=position),
                ),
                *capture_trajectory_hooks(
                    model, direction, layer, local_trajectory
                ),
            ]
        ):
            model(tokens, attention_mask=mask)
        for layer_index in range(layer, model.cfg.n_layers):
            clean_values = clean_trajectory[layer_index][0]
            local_values = local_trajectory[layer_index][0]
            trajectories.append(
                {
                    "id": row["id"],
                    "layer": layer_index,
                    "clean_minus2": float(clean_values[-2].item()),
                    "local_minus2": float(local_values[-2].item()),
                    "clean_minus1": float(clean_values[-1].item()),
                    "local_minus1": float(local_values[-1].item()),
                    "clean_position_mean": float(clean_values.mean().item()),
                    "local_position_mean": float(local_values.mean().item()),
                }
            )

        set_selected_edges(graph, selected)
        circuit_conditions = {
            "upstream_circuit_ablated": (),
            "rescue_direction_local": (
                (
                    target_hook,
                    state_rescue_hook(
                        clean_state["value"],
                        direction,
                        component="directional",
                        position=position,
                    ),
                ),
            ),
            "rescue_direction_all_positions": (
                (
                    target_hook,
                    state_rescue_hook(
                        clean_state["value"],
                        direction,
                        component="directional",
                        position=None,
                    ),
                ),
            ),
            "rescue_full_state_local": (
                (
                    target_hook,
                    state_rescue_hook(
                        clean_state["value"],
                        direction,
                        component="full",
                        position=position,
                    ),
                ),
            ),
            "rescue_full_state_all_positions": (
                (
                    target_hook,
                    state_rescue_hook(
                        clean_state["value"],
                        direction,
                        component="full",
                        position=None,
                    ),
                ),
            ),
        }
        for condition, rescue_hooks in circuit_conditions.items():
            projection, logits = _mixed_forward(
                model,
                graph,
                tokens,
                mask,
                target_hook=target_hook,
                target_position=position,
                direction=direction,
                patch_selected=True,
                mixed_fwd_hooks=rescue_hooks,
            )
            records.append(
                {
                    "id": row["id"],
                    "condition": condition,
                    "target_projection": float(projection.item()),
                    "refusal_log_odds": float(
                        refusal_log_odds(logits, refusal_token_ids).item()
                    ),
                }
            )

        for seed, edge_names in random_sets.items():
            set_selected_edges(graph, edge_names)
            projection, logits = _mixed_forward(
                model,
                graph,
                tokens,
                mask,
                target_hook=target_hook,
                target_position=position,
                direction=direction,
                patch_selected=True,
            )
            records.append(
                {
                    "id": row["id"],
                    "condition": f"random_circuit_ablated_seed_{seed}",
                    "target_projection": float(projection.item()),
                    "refusal_log_odds": float(
                        refusal_log_odds(logits, refusal_token_ids).item()
                    ),
                }
            )

    generation_records = []
    temporal_conditions = {}
    if not args.skip_generation:
        # Temporal generation needs only residual-stream intervention hooks.
        # Restore native grouped-query attention so TransformerLens initializes
        # its KV cache with the same number of heads produced by the model.
        configure_standard_hooks(model)
        temporal_conditions = {
            "normal": [],
            "prompt_only_projection": phase_ablation_hooks(
                model, direction, "prompt_only"
            ),
            "generation_only_projection": phase_ablation_hooks(
                model, direction, "generation_only"
            ),
            "recurrent_projection": phase_ablation_hooks(
                model, direction, "recurrent"
            ),
        }
        for row, prompt in zip(rows, prompts, strict=True):
            input_tokens = model.to_tokens(
                [prompt], prepend_bos=False, padding_side="left"
            )
            for condition, hooks in temporal_conditions.items():
                with model.hooks(fwd_hooks=hooks):
                    generated = model.generate(
                        [prompt],
                        max_new_tokens=args.max_new_tokens,
                        stop_at_eos=True,
                        do_sample=False,
                        use_past_kv_cache=True,
                        prepend_bos=False,
                        padding_side="left",
                        return_type="tokens",
                        verbose=False,
                    )
                continuation = generated[0, input_tokens.shape[1] :].tolist()
                eos = model.tokenizer.eos_token_id
                if eos is not None and eos in continuation:
                    continuation = continuation[: continuation.index(eos) + 1]
                answer = model.tokenizer.decode(
                    continuation, skip_special_tokens=True
                ).strip()
                generation_records.append(
                    {
                        "id": row["id"],
                        "condition": condition,
                        "generated_token_count": len(continuation),
                        "refusal_prefix": bool(is_refusal(answer)),
                    }
                )

    condition_names = sorted({row["condition"] for row in records})
    aggregates = {
        condition: {
            "mean_target_projection": mean(
                records, condition, "target_projection"
            ),
            "mean_refusal_log_odds": mean(
                records, condition, "refusal_log_odds"
            ),
        }
        for condition in condition_names
    }
    generation_aggregates = {
        condition: {
            "n": len(values),
            "refusal_prefix_rate": sum(
                row["refusal_prefix"] for row in values
            )
            / len(values),
        }
        for condition in temporal_conditions
        if (
            values := [
                row
                for row in generation_records
                if row["condition"] == condition
            ]
        )
    }
    result = {
        "status": "complete",
        "num_examples": len(rows),
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": sha256(args.dataset.resolve()),
            "ids": [row["id"] for row in rows],
        },
        "vector": {
            "layer": layer,
            "position": position,
            "target_hook": target_hook,
        },
        "circuit": {
            "path": str(args.circuit.resolve()),
            "sha256": sha256(args.circuit.resolve()),
            "retained_edges": len(selected),
        },
        "random_controls": {
            "seeds": sorted(random_sets),
            "matching": "same edge count; source/target type, layer, and qkv matched where available",
        },
        "aggregates": aggregates,
        "generation_scope": generation_aggregates,
        "trajectory": trajectories,
        "per_example": records,
        "generation_per_example": generation_records,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

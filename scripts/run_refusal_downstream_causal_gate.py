#!/usr/bin/env python3
"""Test local refusal-vector causality and rescue after upstream-circuit ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
)
from refusal_circuit.metrics import ProjectionCapture, refusal_log_odds
from refusal_circuit.model import format_instructions, tokenize_prompts
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


def coordinate_rescue_hook(
    clean_state: torch.Tensor,
    direction: torch.Tensor,
    position: int,
):
    def hook(activations: torch.Tensor, hook):
        del hook
        index = position if position >= 0 else activations.shape[1] + position
        changed = activations.clone()
        source = activations[:, index : index + 1]
        target = clean_state[:, index : index + 1].to(activations)
        vector = direction.to(activations)
        vector = vector / vector.norm().clamp_min(1e-8)
        coordinate_delta = torch.einsum(
            "...d,d->...", target - source, vector
        ).unsqueeze(-1)
        changed[:, index : index + 1] = (
            source + coordinate_delta * vector
        )
        return changed

    return hook


def mean(rows: list[dict], condition: str, field: str) -> float:
    values = [row[field] for row in rows if row["condition"] == condition]
    return float(sum(values) / len(values))


def normalized_recovery(clean: float, changed: float, rescued: float) -> float:
    effect = clean - changed
    return float("nan") if abs(effect) < 1e-8 else (rescued - changed) / effect


def fraction_of_reference_drop(clean: float, reference: float, changed: float) -> float:
    reference_drop = clean - reference
    return (
        float("nan")
        if abs(reference_drop) < 1e-8
        else (clean - changed) / reference_drop
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=16)
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
        row for row in read_jsonl(args.dataset.resolve())
        if row.get("label") == "harmful"
    ][: args.num_examples]
    if len(rows) != args.num_examples:
        raise ValueError("The reserve split has fewer harmful rows than requested.")
    prompts = format_instructions(
        model.tokenizer, [row["instruction"] for row in rows]
    )

    graph = make_projection_graph(model, layer)
    set_selected_edges(graph, load_circuit_edge_names(args.circuit.resolve()))
    records: list[dict] = []
    for row, prompt in zip(rows, prompts, strict=True):
        tokens, attention_mask = tokenize_prompts(
            model, [prompt], max_length=config["eapig"].get("max_length", 512)
        )
        clean_state: dict[str, torch.Tensor] = {}
        clean_capture = ProjectionCapture(direction=direction, position=position)

        def capture_clean(activations: torch.Tensor, hook):
            del hook
            clean_state["value"] = activations.detach().clone()
            return activations

        with torch.inference_mode(), model.hooks(
            fwd_hooks=[
                (target_hook, capture_clean),
                (target_hook, clean_capture.hook),
            ]
        ):
            clean_logits = model(tokens, attention_mask=attention_mask)

        global_capture = ProjectionCapture(direction=direction, position=position)
        with torch.inference_mode(), model.hooks(
            fwd_hooks=[
                *all_layer_ablation_hooks(model, direction),
                (target_hook, global_capture.hook),
            ]
        ):
            global_logits = model(tokens, attention_mask=attention_mask)

        local_capture = ProjectionCapture(direction=direction, position=position)
        with torch.inference_mode(), model.hooks(
            fwd_hooks=[
                (
                    target_hook,
                    direction_ablation_hook(direction, position=position),
                ),
                (target_hook, local_capture.hook),
            ]
        ):
            local_logits = model(tokens, attention_mask=attention_mask)

        circuit_projection, circuit_logits = _mixed_forward(
            model,
            graph,
            tokens,
            attention_mask,
            target_hook=target_hook,
            target_position=position,
            direction=direction,
            patch_selected=True,
        )
        rescued_projection, rescued_logits = _mixed_forward(
            model,
            graph,
            tokens,
            attention_mask,
            target_hook=target_hook,
            target_position=position,
            direction=direction,
            patch_selected=True,
            mixed_fwd_hooks=[
                (
                    target_hook,
                    coordinate_rescue_hook(
                        clean_state["value"], direction, position
                    ),
                )
            ],
        )

        conditions = {
            "normal": (clean_capture.values(), clean_logits),
            "global_projection": (global_capture.values(), global_logits),
            "local_projection": (local_capture.values(), local_logits),
            "upstream_circuit_ablated": (
                circuit_projection,
                circuit_logits,
            ),
            "upstream_circuit_ablated_vector_rescued": (
                rescued_projection,
                rescued_logits,
            ),
        }
        for condition, (projection, logits) in conditions.items():
            records.append(
                {
                    "id": row["id"],
                    "condition": condition,
                    "target_projection": float(projection.item()),
                    "refusal_log_odds": float(
                        refusal_log_odds(
                            logits, refusal_token_ids
                        ).item()
                    ),
                }
            )

    condition_names = [
        "normal",
        "global_projection",
        "local_projection",
        "upstream_circuit_ablated",
        "upstream_circuit_ablated_vector_rescued",
    ]
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
    clean = aggregates["normal"]["mean_refusal_log_odds"]
    global_projected = aggregates["global_projection"]["mean_refusal_log_odds"]
    local_projected = aggregates["local_projection"]["mean_refusal_log_odds"]
    circuit_ablated = aggregates[
        "upstream_circuit_ablated"
    ]["mean_refusal_log_odds"]
    rescued = aggregates[
        "upstream_circuit_ablated_vector_rescued"
    ]["mean_refusal_log_odds"]
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
        },
        "target": {
            "metric": "next_token_refusal_log_odds",
            "refusal_token_ids": refusal_token_ids,
        },
        "aggregates": aggregates,
        "effects": {
            "local_projection_drop": clean - local_projected,
            "global_projection_drop": clean - global_projected,
            "local_fraction_of_global_drop": fraction_of_reference_drop(
                clean, global_projected, local_projected
            ),
            "upstream_circuit_ablation_drop": clean - circuit_ablated,
            "directional_rescue_recovery": normalized_recovery(
                clean, circuit_ablated, rescued
            ),
        },
        "per_example": records,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate edge-pruned sufficient persona circuits on frozen held-out pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _bootstrap import ROOT


def _capture_state(model, tokens, attention_mask, target_hook: str):
    import torch

    captured = {}

    def hook(activations: torch.Tensor, hook):
        del hook
        captured["state"] = activations.detach().clone()
        return activations

    with torch.inference_mode(), model.hooks(
        fwd_hooks=[(target_hook, hook)]
    ):
        logits = model(tokens, attention_mask=attention_mask)
    return captured["state"], logits.detach()


def _projection_values(state, direction, mask):
    import torch

    vector = direction.to(device=state.device, dtype=torch.float32)
    vector = vector / vector.norm().clamp_min(1e-8)
    values = torch.einsum("bpd,d->bp", state.float(), vector)
    return values[mask.to(device=values.device, dtype=torch.bool)]


def _masked_kl(student_logits, teacher_logits, prediction_mask):
    import torch
    import torch.nn.functional as functional

    mask = prediction_mask.to(
        device=student_logits.device, dtype=torch.bool
    )
    student = student_logits.float()[mask].log_softmax(dim=-1)
    teacher = teacher_logits.float()[mask].log_softmax(dim=-1)
    return float(
        (
            functional.kl_div(
                student, teacher, reduction="sum", log_target=True
            )
            / student.shape[0]
        ).item()
    )


def _normalized_effect(clean: float, corrupt: float, value: float) -> float:
    denominator = clean - corrupt
    if abs(denominator) <= 1e-8:
        return float("nan")
    return (value - corrupt) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--stage", choices=("upstream", "downstream"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--circuit-path",
        type=Path,
        default=None,
        help=(
            "Optional canonical or topology-control circuit. By default, use "
            "the learned top-k circuit from --run-name."
        ),
    )
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import (
        mixed_logits_with_intervention,
        mixed_state_from_prompt_pair,
    )
    from refusal_circuit.eapig import (
        _response_mask,
        response_logprob_values,
        response_prediction_mask,
    )
    from refusal_circuit.edge_pruning import (
        fixed_state_hook,
        vector_only_rescue_state,
    )
    from refusal_circuit.graph import (
        circuit_connectivity,
        load_circuit_edge_names,
        make_projection_graph,
        make_steering_to_logits_graph,
        set_selected_edges,
    )
    from refusal_circuit.model import (
        load_model,
        tokenize_prompts,
        validate_model_architecture,
    )
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    trait_slug = config["trait"]["name"]
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    discovery = (
        result_root
        / "circuit_discovery_edge_pruning"
        / args.stage
        / args.run_name
    )
    metadata_path = discovery / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "complete"
        or metadata.get("stage") != args.stage
        or metadata.get("model") != config["model"]
    ):
        raise ValueError("Edge-pruning discovery metadata failed provenance.")

    default_circuit_path = (
        discovery
        / "circuits_source_to_sink"
        / f"edge_pruned_top_{args.k}.json"
    )
    circuit_path = (
        args.circuit_path
        if args.circuit_path is not None and args.circuit_path.is_absolute()
        else (
            ROOT / args.circuit_path
            if args.circuit_path is not None
            else default_circuit_path
        )
    )
    circuit_payload = json.loads(circuit_path.read_text(encoding="utf-8"))
    circuit_method = circuit_payload.get("method")
    canonical = circuit_method == "hard_concrete_edge_pruning"
    topology_control = (
        circuit_method == "topology_stratified_score_permutation_control"
    )
    expected_control_scope = f"edge_pruning_{args.stage}"
    if (
        int(circuit_payload["requested_k"]) != args.k
        or circuit_payload.get("score_source")
        != metadata["outputs"]["scores"]
        or (
            canonical
            and circuit_payload.get("stage") != args.stage
        )
        or (
            topology_control
            and circuit_payload.get("control_scope")
            != expected_control_scope
        )
        or not (canonical or topology_control)
    ):
        raise ValueError("Circuit and edge-pruning run metadata disagree.")
    if topology_control:
        target_path = ROOT / circuit_payload["target_circuit"]
        if (
            target_path != default_circuit_path
            or circuit_payload.get("target_circuit_sha256")
            != sha256_file(default_circuit_path)
        ):
            raise ValueError("Topology control target provenance changed.")

    evaluation_path = (
        args.evaluation_manifest
        if args.evaluation_manifest.is_absolute()
        else ROOT / args.evaluation_manifest
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    if (
        evaluation.get("status") != "frozen"
        or evaluation.get("protocol")
        != "prompt_response_disjoint_downstream_evaluation_v2"
        or evaluation.get("model") != config["model"]
        or evaluation.get("trait") != trait_slug
        or evaluation["sources"]["positive_sha256"]
        != sha256_file(positive_path)
        or evaluation["sources"]["negative_sha256"]
        != sha256_file(negative_path)
        or evaluation["discovery_manifest_sha256"]
        != metadata["sample_manifest_sha256"]
    ):
        raise ValueError("Held-out evaluation manifest failed provenance.")
    samples = evaluation["selection"]["samples"]
    if len(samples) != args.num_prompts:
        raise ValueError(
            f"Evaluation manifest has {len(samples)} pairs, "
            f"expected {args.num_prompts}."
        )
    positive_by_record = {
        row["record_id"]: row for row in read_jsonl(positive_path)
    }
    negative_by_record = {
        row["record_id"]: row for row in read_jsonl(negative_path)
    }

    model_cfg = config["model"]
    model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=args.dtype,
        revision=model_cfg.get("revision"),
    )
    observed_architecture = validate_model_architecture(
        model, config["model_architecture"]
    )
    model.set_ungroup_grouped_query_attention(True)
    layer = int(config["vector"]["paper_layer"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    direction_path = result_root / f"vector/{trait_slug}_direction.pt"
    direction = torch.load(
        direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    ).float()
    direction = direction / direction.norm().clamp_min(1e-8)
    graph = (
        make_projection_graph(model, layer)
        if args.stage == "upstream"
        else make_steering_to_logits_graph(model, layer)
    )
    edge_names = load_circuit_edge_names(circuit_path)
    connectivity = circuit_connectivity(graph, edge_names)
    if not connectivity["all_edges_on_source_to_sink_paths"]:
        raise ValueError("Sufficiency evaluation requires a complete-path circuit.")
    set_selected_edges(graph, edge_names)
    compact_dtype = next(model.parameters()).dtype

    rows = []
    upstream_values = {
        name: []
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    downstream_values = {
        answer_label: {
            name: []
            for name in (
                "natural_clean",
                "corrupt_interface",
                "vector_interface",
                "circuit_only",
                "circuit_ablated",
            )
        }
        for answer_label in ("positive", "negative")
    }
    downstream_kls = []

    for sample in samples:
        positive = positive_by_record[sample["positive_record_id"]]
        negative = negative_by_record[sample["negative_record_id"]]
        if (
            positive["pair_id"] != sample["pair_id"]
            or negative["pair_id"] != sample["pair_id"]
            or positive["prompt"] != sample["positive_prompt"]
        ):
            raise ValueError(f"Held-out pair changed: {sample['pair_id']}")

        answer_specs = (
            (("positive", positive["answer"]),)
            if args.stage == "upstream"
            else (
                ("positive", positive["answer"]),
                ("negative", negative["answer"]),
            )
        )
        record = {
            "pair_id": sample["pair_id"],
            "question_index": int(sample["question_index"]),
            "instruction_index": int(sample["instruction_index"]),
        }
        for answer_label, answer in answer_specs:
            clean_text = sample["positive_prompt"] + answer
            corrupt_text = sample["aligned_negative_prompt"] + answer
            clean_tokens, clean_mask = tokenize_prompts(
                model, [clean_text], max_length=args.max_length
            )
            corrupt_tokens, corrupt_mask = tokenize_prompts(
                model, [corrupt_text], max_length=args.max_length
            )
            if (
                clean_tokens.shape != corrupt_tokens.shape
                or not torch.equal(clean_mask, corrupt_mask)
            ):
                raise ValueError("Held-out transcript pair lost token alignment.")
            response_mask = _response_mask(
                clean_tokens,
                clean_mask,
                [int(sample["response_start"])],
            )
            prediction_mask = response_prediction_mask(response_mask)

            clean_state, natural_clean_logits = _capture_state(
                model, clean_tokens, clean_mask, target_hook
            )
            corrupt_state, _ = _capture_state(
                model, corrupt_tokens, corrupt_mask, target_hook
            )
            if args.stage == "upstream":
                circuit_state = mixed_state_from_prompt_pair(
                    model,
                    graph,
                    clean_tokens,
                    clean_mask,
                    corrupt_tokens,
                    corrupt_mask,
                    target_hook=target_hook,
                    patch_selected=False,
                    cache_dtype=compact_dtype,
                )
                ablated_state = mixed_state_from_prompt_pair(
                    model,
                    graph,
                    clean_tokens,
                    clean_mask,
                    corrupt_tokens,
                    corrupt_mask,
                    target_hook=target_hook,
                    patch_selected=True,
                    cache_dtype=compact_dtype,
                )
                states = {
                    "clean": clean_state,
                    "corrupt": corrupt_state,
                    "circuit_only": circuit_state,
                    "circuit_ablated": ablated_state,
                }
                for condition, state in states.items():
                    values = _projection_values(
                        state, direction, prediction_mask
                    )
                    upstream_values[condition].extend(
                        float(value) for value in values.cpu()
                    )
                    record[f"{condition}_projection_mean"] = float(
                        values.mean().item()
                    )
                continue

            interface_state = vector_only_rescue_state(
                clean_state,
                corrupt_state,
                direction,
                prediction_mask,
            ).detach()
            corrupt_hook = (
                target_hook,
                fixed_state_hook(corrupt_state),
            )
            interface_hook = (
                target_hook,
                fixed_state_hook(interface_state),
            )

            def corrupt_transform(name: str, activations: torch.Tensor):
                return (
                    corrupt_state.to(
                        device=activations.device, dtype=activations.dtype
                    )
                    if name == target_hook
                    else activations
                )

            def clean_transform(name: str, activations: torch.Tensor):
                return (
                    interface_state.to(
                        device=activations.device, dtype=activations.dtype
                    )
                    if name == target_hook
                    else activations
                )

            with torch.inference_mode(), model.hooks(
                fwd_hooks=[corrupt_hook]
            ):
                corrupt_logits = model(
                    clean_tokens, attention_mask=clean_mask
                ).detach()
            with torch.inference_mode(), model.hooks(
                fwd_hooks=[interface_hook]
            ):
                interface_logits = model(
                    clean_tokens, attention_mask=clean_mask
                ).detach()
            circuit_logits = mixed_logits_with_intervention(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_fwd_hooks=[corrupt_hook],
                corrupt_cache_transform=corrupt_transform,
                clean_fwd_hooks=[interface_hook],
                clean_cache_transform=clean_transform,
                patch_selected=False,
                cache_dtype=compact_dtype,
            )
            ablated_logits = mixed_logits_with_intervention(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_fwd_hooks=[corrupt_hook],
                corrupt_cache_transform=corrupt_transform,
                clean_fwd_hooks=[interface_hook],
                clean_cache_transform=clean_transform,
                patch_selected=True,
                cache_dtype=compact_dtype,
            )
            logits_by_condition = {
                "natural_clean": natural_clean_logits,
                "corrupt_interface": corrupt_logits,
                "vector_interface": interface_logits,
                "circuit_only": circuit_logits,
                "circuit_ablated": ablated_logits,
            }
            for condition, logits in logits_by_condition.items():
                value = float(
                    response_logprob_values(
                        logits, clean_tokens, response_mask
                    )[0].item()
                )
                downstream_values[answer_label][condition].append(value)
                record[f"{answer_label}_{condition}_logprob"] = value
            baseline_kl = _masked_kl(
                corrupt_logits, interface_logits, prediction_mask
            )
            circuit_kl = _masked_kl(
                circuit_logits, interface_logits, prediction_mask
            )
            kl_recovery = (
                1.0 - circuit_kl / baseline_kl
                if baseline_kl > 1e-8
                else float("nan")
            )
            downstream_kls.append(kl_recovery)
            record[f"{answer_label}_baseline_kl"] = baseline_kl
            record[f"{answer_label}_circuit_kl"] = circuit_kl
            record[f"{answer_label}_kl_recovery"] = kl_recovery
        rows.append(record)

    if args.stage == "upstream":
        tensors = {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in upstream_values.items()
        }
        baseline_mse = float(
            (tensors["clean"] - tensors["corrupt"]).square().mean().item()
        )
        circuit_mse = float(
            (tensors["clean"] - tensors["circuit_only"])
            .square()
            .mean()
            .item()
        )
        aggregate_metrics = {
            "means": {
                name: float(values.mean().item())
                for name, values in tensors.items()
            },
            "baseline_per_position_mse": baseline_mse,
            "circuit_per_position_mse": circuit_mse,
            "per_position_reconstruction_fraction": (
                1.0 - circuit_mse / baseline_mse
                if baseline_mse > 1e-8
                else float("nan")
            ),
            "mean_effect_faithfulness": _normalized_effect(
                float(tensors["clean"].mean().item()),
                float(tensors["corrupt"].mean().item()),
                float(tensors["circuit_only"].mean().item()),
            ),
            "mean_effect_necessity": (
                (
                    float(tensors["clean"].mean().item())
                    - float(tensors["circuit_ablated"].mean().item())
                )
                / (
                    float(tensors["clean"].mean().item())
                    - float(tensors["corrupt"].mean().item())
                )
            ),
        }
    else:
        margins = {}
        for condition in downstream_values["positive"]:
            positive_mean = sum(
                downstream_values["positive"][condition]
            ) / len(downstream_values["positive"][condition])
            negative_mean = sum(
                downstream_values["negative"][condition]
            ) / len(downstream_values["negative"][condition])
            margins[condition] = positive_mean - negative_mean
        natural_gap = (
            margins["natural_clean"] - margins["corrupt_interface"]
        )
        vector_gap = (
            margins["vector_interface"] - margins["corrupt_interface"]
        )
        aggregate_metrics = {
            "contrastive_response_logprob_margins": margins,
            "vector_interface_rescue_fraction_of_natural_gap": (
                vector_gap / natural_gap
                if abs(natural_gap) > 1e-8
                else float("nan")
            ),
            "downstream_circuit_margin_faithfulness": (
                (
                    margins["circuit_only"]
                    - margins["corrupt_interface"]
                )
                / vector_gap
                if abs(vector_gap) > 1e-8
                else float("nan")
            ),
            "downstream_circuit_margin_necessity": (
                (
                    margins["vector_interface"]
                    - margins["circuit_ablated"]
                )
                / vector_gap
                if abs(vector_gap) > 1e-8
                else float("nan")
            ),
            "mean_distribution_kl_recovery": float(
                sum(downstream_kls) / len(downstream_kls)
            ),
        }

    for value in _numeric_leaves(aggregate_metrics):
        if not math.isfinite(value):
            raise RuntimeError("Evaluation produced a non-finite aggregate.")
    output_name = args.output_tag or f"k{args.k}_n{args.num_prompts}_heldout"
    output = discovery / "evaluation" / output_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results.")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "instance_metrics.csv", index=False)
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "method": "held-out recurrent reconstructive sufficiency evaluation",
        "circuit_method": circuit_method,
        "is_topology_control": topology_control,
        "stage": args.stage,
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "run_name": args.run_name,
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "connectivity": connectivity,
        "n_prompt_pairs": len(samples),
        "evaluation_manifest": str(evaluation_path.relative_to(ROOT)),
        "evaluation_manifest_sha256": sha256_file(evaluation_path),
        "discovery_metadata": str(metadata_path.relative_to(ROOT)),
        "discovery_metadata_sha256": sha256_file(metadata_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "interface": {
            "hook": target_hook,
            "layer": layer,
            "direction": str(direction_path.relative_to(ROOT)),
            "direction_sha256": sha256_file(direction_path),
            "timing": "all response-prediction positions",
        },
        "metrics": aggregate_metrics,
        "max_length": args.max_length,
        "truncation": False,
        "behavioral_generation_required": True,
    }
    write_json(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


def _numeric_leaves(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _numeric_leaves(child)
    elif isinstance(value, (int, float)):
        yield float(value)


if __name__ == "__main__":
    main()

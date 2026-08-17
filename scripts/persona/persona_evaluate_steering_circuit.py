#!/usr/bin/env python3
"""Evaluate a steering-path circuit on held-out high-trait transcripts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        required=True,
        help=(
            "Frozen prompt- and response-disjoint manifest produced by "
            "persona.freeze-downstream-evaluation."
        ),
    )
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--circuit-path", type=Path, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--allow-disconnected-circuit-only",
        action="store_true",
        help="Permit legacy cut-set sufficiency diagnostics.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import (
        mixed_logits_with_intervention,
        normalized_faithfulness,
        normalized_necessity,
    )
    from refusal_circuit.eapig import (
        _response_mask,
        response_logprob_values,
        response_prediction_mask,
    )
    from refusal_circuit.graph import (
        circuit_connectivity,
        load_circuit_edge_names,
        make_steering_to_logits_graph,
        set_selected_edges,
    )
    from refusal_circuit.interventions import masked_activation_addition_hook
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
    discovery = result_root / "circuit_discovery_steering_path" / args.run_name
    discovery_meta_path = discovery / "eapig_run_metadata.json"
    discovery_meta = json.loads(discovery_meta_path.read_text(encoding="utf-8"))
    if discovery_meta.get("status") != "complete":
        raise ValueError("Discovery run is not complete.")
    if discovery_meta.get("model") != config["model"]:
        raise ValueError("Discovery and evaluation model configurations differ.")
    corrupt_coefficient = float(
        discovery_meta["steering"]["corrupt_coefficient"]
    )
    excluded = set(discovery_meta["selection"]["record_ids"])
    minimum_trait = float(discovery_meta["selection"]["minimum_trait_score"])
    minimum_coherence = float(
        discovery_meta["selection"]["minimum_coherence_score"]
    )
    metric_name = discovery_meta["metric"]["name"]
    contrastive = metric_name == "contrastive_response_logprob"
    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    negative_by_pair = {
        row["pair_id"]: row for row in read_jsonl(negative_path)
    }
    maximum_negative_trait = discovery_meta["selection"].get(
        "maximum_negative_trait_score"
    )
    effective = set(
        pd.read_csv(result_root / "vector/effective_pairs.csv")["pair_id"].astype(str)
    )
    candidates = [
        row
        for row in read_jsonl(positive_path)
        if row["record_id"] not in excluded
        and row["pair_id"] in effective
        and row.get("trait_score") is not None
        and float(row["trait_score"]) >= minimum_trait
        and row.get("coherence_score") is not None
        and float(row["coherence_score"]) >= minimum_coherence
        and (
            not contrastive
            or (
                row["pair_id"] in negative_by_pair
                and negative_by_pair[row["pair_id"]].get("trait_score") is not None
                and float(negative_by_pair[row["pair_id"]]["trait_score"])
                <= float(maximum_negative_trait)
                and negative_by_pair[row["pair_id"]].get("coherence_score") is not None
                and float(negative_by_pair[row["pair_id"]]["coherence_score"])
                >= minimum_coherence
            )
        )
    ]
    evaluation_manifest_path = (
        args.evaluation_manifest
        if args.evaluation_manifest.is_absolute()
        else ROOT / args.evaluation_manifest
    )
    evaluation_manifest = json.loads(
        evaluation_manifest_path.read_text(encoding="utf-8")
    )
    if (
        evaluation_manifest.get("status") != "frozen"
        or evaluation_manifest.get("protocol")
        != "prompt_response_disjoint_downstream_evaluation_v2"
        or evaluation_manifest.get("model") != config["model"]
        or evaluation_manifest.get("trait") != trait_slug
        or evaluation_manifest["sources"]["positive_sha256"]
        != sha256_file(positive_path)
        or evaluation_manifest["sources"]["negative_sha256"]
        != sha256_file(negative_path)
    ):
        raise ValueError("Downstream evaluation manifest failed provenance.")
    discovery_manifest_path = ROOT / evaluation_manifest[
        "discovery_manifest"
    ]
    if (
        sha256_file(discovery_manifest_path)
        != evaluation_manifest["discovery_manifest_sha256"]
        or sha256_file(discovery_manifest_path)
        != discovery_meta.get("sample_manifest_sha256")
    ):
        raise ValueError(
            "Downstream evaluation was not frozen against this discovery split."
        )
    samples = evaluation_manifest["selection"]["samples"]
    if len(samples) != args.num_prompts:
        raise ValueError(
            f"Manifest has {len(samples)} eval pairs, "
            f"but --num-prompts={args.num_prompts}."
        )
    candidate_by_record = {row["record_id"]: row for row in candidates}
    selected = []
    for sample in samples:
        row = candidate_by_record.get(sample["positive_record_id"])
        if (
            row is None
            or row["pair_id"] != sample["pair_id"]
            or int(row["question_index"]) != int(sample["question_index"])
            or int(row["instruction_index"])
            != int(sample["instruction_index"])
            or row["prompt"] != sample["positive_prompt"]
        ):
            raise ValueError(
                f"Manifest eval pair unavailable or changed: {sample['pair_id']}"
            )
        negative = negative_by_pair[row["pair_id"]]
        if negative["record_id"] != sample["negative_record_id"]:
            raise ValueError(f"Negative response changed: {sample['pair_id']}")
        selected.append(row)
    if len(selected) != args.num_prompts:
        raise ValueError(f"Only {len(selected)} held-out high-trait transcripts available")

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
    vector_path = result_root / f"vector/{trait_slug}_response_avg_diff.pt"
    vector_stack = torch.load(
        vector_path, map_location=next(model.parameters()).device, weights_only=True
    )
    layer = int(config["vector"]["paper_layer"])
    raw_direction = vector_stack[layer] if vector_stack.ndim == 2 else vector_stack
    target_hook = config["vector"]["transformer_lens_target_hook"]
    graph = make_steering_to_logits_graph(model, layer)
    circuit_path = (
        args.circuit_path
        if args.circuit_path is not None
        else discovery
        / "circuits_source_to_sink"
        / f"steering_path_eapig_top_{args.k}.json"
    )
    circuit_path = circuit_path if circuit_path.is_absolute() else ROOT / circuit_path
    circuit_payload = json.loads(circuit_path.read_text(encoding="utf-8"))
    if int(circuit_payload["requested_k"]) != args.k:
        raise ValueError("Circuit payload k differs from requested k.")
    if circuit_payload.get("score_source") != discovery_meta.get("score_file"):
        raise ValueError("Circuit was not derived from the frozen discovery scores.")
    if float(circuit_payload["corrupt_coefficient"]) != corrupt_coefficient:
        raise ValueError("Circuit and discovery corrupt coefficients disagree.")
    edge_names = load_circuit_edge_names(circuit_path)
    connectivity = circuit_connectivity(graph, edge_names)
    if (
        not connectivity["all_edges_on_source_to_sink_paths"]
        and not args.allow_disconnected_circuit_only
    ):
        raise ValueError(
            "Circuit-only evaluation requires a source-to-sink circuit. "
            "Rebuild with --families source_to_sink, or pass "
            "--allow-disconnected-circuit-only only for a legacy diagnostic."
        )
    set_selected_edges(graph, edge_names)

    values = {
        name: []
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    instances = []
    observed_lengths = []
    for row in selected:
        response_start = len(
            model.tokenizer.encode(row["prompt"], add_special_tokens=False)
        )
        answer_specs = [("positive", row["answer"])]
        if contrastive:
            negative = negative_by_pair[row["pair_id"]]
            answer_specs.append(("negative", negative["answer"]))
        record = {
            "record_id": row["record_id"],
            "pair_id": row["pair_id"],
            "trait_score": float(row["trait_score"]),
            "coherence_score": float(row["coherence_score"]),
        }
        if contrastive:
            record.update(
                {
                    "negative_record_id": negative["record_id"],
                    "negative_trait_score": float(negative["trait_score"]),
                    "negative_coherence_score": float(
                        negative["coherence_score"]
                    ),
                }
            )
        component_values = {
            name: [] for name in values
        }
        # Keep the two contrastive transcripts sequential. EAP circuit caches
        # for a batch of two exceed the 40 GB A100 memory available here.
        for answer_label, answer in answer_specs:
            transcript = row["prompt"] + answer
            transcript_length = len(
                model.tokenizer.encode(
                    transcript, add_special_tokens=False
                )
            )
            observed_lengths.append(transcript_length)
            if transcript_length > args.max_length:
                raise ValueError(
                    f"Held-out {answer_label} transcript exceeds "
                    "--max-length; refusing to truncate the response-logprob "
                    "estimand."
                )
            tokens, attention_mask = tokenize_prompts(
                model,
                [transcript],
                max_length=args.max_length,
            )
            response_mask = _response_mask(
                tokens, attention_mask, [response_start]
            )
            prediction_mask = response_prediction_mask(response_mask)
            corrupt_hook = (
                target_hook,
                masked_activation_addition_hook(
                    raw_direction,
                    coefficient=corrupt_coefficient,
                    mask=prediction_mask,
                ),
            )

            def cache_transform(source_name: str, activations: torch.Tensor):
                if source_name != target_hook:
                    return activations
                return activations + (
                    prediction_mask.to(
                        device=activations.device, dtype=activations.dtype
                    ).unsqueeze(-1)
                    * corrupt_coefficient
                    * raw_direction.to(
                        device=activations.device, dtype=activations.dtype
                    )
                )

            with torch.inference_mode():
                clean_logits = model(tokens, attention_mask=attention_mask)
            with torch.inference_mode(), model.hooks(
                fwd_hooks=[corrupt_hook]
            ):
                corrupt_logits = model(
                    tokens, attention_mask=attention_mask
                )
            logits_by_condition = {
                "clean": clean_logits,
                "corrupt": corrupt_logits,
                "circuit_only": mixed_logits_with_intervention(
                    model,
                    graph,
                    tokens,
                    attention_mask,
                    corrupt_fwd_hooks=[corrupt_hook],
                    corrupt_cache_transform=cache_transform,
                    patch_selected=False,
                ),
                "circuit_ablated": mixed_logits_with_intervention(
                    model,
                    graph,
                    tokens,
                    attention_mask,
                    corrupt_fwd_hooks=[corrupt_hook],
                    corrupt_cache_transform=cache_transform,
                    patch_selected=True,
                ),
            }
            for condition, logits in logits_by_condition.items():
                value = float(
                    response_logprob_values(
                        logits, tokens, response_mask
                    )[0].item()
                )
                component_values[condition].append(value)
                record[f"{condition}_{answer_label}_logprob"] = value

        for condition, components in component_values.items():
            value = (
                components[0] - components[1]
                if contrastive
                else components[0]
            )
            values[condition].append(value)
            record[condition] = value
        instances.append(record)

    tensors = {name: torch.tensor(items) for name, items in values.items()}
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Held-out {name} metric contains non-finite values."
            )
    clean_corrupt_gap = float(
        tensors["clean"].mean().item() - tensors["corrupt"].mean().item()
    )
    if not math.isfinite(clean_corrupt_gap) or clean_corrupt_gap <= 1e-8:
        raise RuntimeError(
            "Held-out clean-minus-corrupt response margin is not finite and "
            "positive; normalized circuit metrics are not behavior-aligned."
        )
    faithfulness = normalized_faithfulness(
        tensors["clean"], tensors["corrupt"], tensors["circuit_only"]
    )
    necessity = normalized_necessity(
        tensors["clean"], tensors["corrupt"], tensors["circuit_ablated"]
    )
    output_name = (
        args.output_tag
        or f"k{args.k}_n{args.num_prompts}_prompt_disjoint"
    )
    output = discovery / "evaluation" / output_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains evaluation results")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(instances).to_csv(output / "instance_metrics.csv", index=False)
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "claim_scope": "held-out teacher-forced generation-logit mediation test",
        "run_name": args.run_name,
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "connectivity": connectivity,
        "eligible_downstream_edges": len(graph.edges),
        "n_prompts": len(selected),
        "record_ids": [row["record_id"] for row in selected],
        "evaluation_manifest": str(
            evaluation_manifest_path.relative_to(ROOT)
        ),
        "evaluation_manifest_sha256": sha256_file(
            evaluation_manifest_path
        ),
        "source": str(positive_path.relative_to(ROOT)),
        "source_sha256": sha256_file(positive_path),
        "discovery_metadata": str(discovery_meta_path.relative_to(ROOT)),
        "discovery_metadata_sha256": sha256_file(discovery_meta_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "steering": {
            "hook": target_hook,
            "layer": layer,
            "clean_coefficient": 0.0,
            "corrupt_coefficient": corrupt_coefficient,
        },
        "metric": metric_name,
        "means": {
            name: float(tensor.mean().item()) for name, tensor in tensors.items()
        },
        "faithfulness": faithfulness,
        "necessity": necessity,
        "max_length": args.max_length,
        "max_observed_tokens": max(observed_lengths),
        "truncation": False,
        "behavioral_validation_required": True,
    }
    write_json(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

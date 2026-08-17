#!/usr/bin/env python3
"""Evaluate an end-to-end persona circuit on prompt- and response-disjoint pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_negative_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--circuit-path", type=Path, default=None)
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        required=True,
        help=(
            "Frozen prompt- and response-disjoint manifest produced by "
            "persona.freeze-downstream-evaluation."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import (
        mixed_logits_from_prompt_pair,
        normalized_faithfulness,
        normalized_necessity,
    )
    from refusal_circuit.eapig import _response_mask, response_logprob_values
    from refusal_circuit.graph import (
        circuit_connectivity,
        load_circuit_edge_names,
        make_logit_graph,
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
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    official_root = ROOT / config["paths"]["official_root"]
    discovery = result_root / "circuit_discovery_end_to_end" / args.run_name
    metadata_path = discovery / "eapig_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("End-to-end discovery run is not complete.")

    discovery_manifest_path = ROOT / metadata["sample_manifest"]
    if (
        sha256_file(discovery_manifest_path)
        != metadata["sample_manifest_sha256"]
    ):
        raise ValueError("Frozen shared manifest hash changed.")
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
        or evaluation_manifest.get("trait") != config["trait"]["name"]
        or evaluation_manifest.get("discovery_manifest")
        != str(discovery_manifest_path.relative_to(ROOT))
        or evaluation_manifest.get("discovery_manifest_sha256")
        != sha256_file(discovery_manifest_path)
    ):
        raise ValueError("End-to-end evaluation manifest failed provenance.")
    samples = evaluation_manifest["selection"]["samples"]
    if len(samples) != args.num_prompts:
        raise ValueError(
            f"Manifest has {len(samples)} evaluation pairs, "
            f"but --num-prompts={args.num_prompts}."
        )

    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    if (
        sha256_file(positive_path)
        != evaluation_manifest["sources"]["positive_sha256"]
        or sha256_file(negative_path)
        != evaluation_manifest["sources"]["negative_sha256"]
    ):
        raise ValueError("Held-out response sources differ from the frozen manifest.")
    positive_by_record = {
        row["record_id"]: row for row in read_jsonl(positive_path)
    }
    negative_by_record = {
        row["record_id"]: row for row in read_jsonl(negative_path)
    }
    trait_data_path = official_root / "extract.json"
    if (
        sha256_file(trait_data_path)
        != evaluation_manifest["sources"]["extract_sha256"]
    ):
        raise ValueError("Trait prompt source differs from the frozen manifest.")
    trait_data = json.loads(trait_data_path.read_text(encoding="utf-8"))

    model_cfg = config["model"]
    if metadata.get("model") != model_cfg:
        raise ValueError("Discovery and evaluation model configurations differ.")
    model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=args.dtype,
        revision=model_cfg.get("revision"),
    )
    observed_architecture = validate_model_architecture(
        model, config["model_architecture"]
    )
    graph = make_logit_graph(model)
    circuit_path = (
        args.circuit_path
        if args.circuit_path is not None
        else discovery
        / "circuits_source_to_sink"
        / f"end_to_end_eapig_top_{args.k}.json"
    )
    circuit_path = circuit_path if circuit_path.is_absolute() else ROOT / circuit_path
    circuit_payload = json.loads(circuit_path.read_text(encoding="utf-8"))
    if int(circuit_payload["requested_k"]) != args.k:
        raise ValueError("Circuit payload k differs from requested k.")
    if circuit_payload.get("score_source") != metadata.get("score_file"):
        raise ValueError("Circuit was not derived from the frozen score table.")
    edge_names = load_circuit_edge_names(circuit_path)
    connectivity = circuit_connectivity(graph, edge_names)
    if not connectivity["all_edges_on_source_to_sink_paths"]:
        raise ValueError(
            "End-to-end circuit-only evaluation requires a complete "
            "source-to-logit circuit."
        )
    set_selected_edges(graph, edge_names)

    values = {
        name: []
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    instances = []
    observed_lengths = []
    for sample in samples:
        positive = positive_by_record[sample["positive_record_id"]]
        negative = negative_by_record[sample["negative_record_id"]]
        if (
            positive["pair_id"] != sample["pair_id"]
            or negative["pair_id"] != sample["pair_id"]
            or positive["record_id"] != sample["positive_record_id"]
            or negative["record_id"] != sample["negative_record_id"]
            or positive["prompt"] != sample["positive_prompt"]
        ):
            raise ValueError(f"Evaluation pair mismatch: {sample['pair_id']}.")
        question_index = int(positive["question_index"])
        instruction_index = int(positive["instruction_index"])
        aligned_negative, alignment = _align_negative_prompt(
            model.tokenizer,
            question=trait_data["questions"][question_index],
            negative_assistant_name=config["trait"]["negative_assistant_name"],
            negative_instruction=trait_data["instruction"][instruction_index]["neg"],
            positive_prompt=positive["prompt"],
            expected_negative_prompt=negative["prompt"],
        )
        response_start = len(
            model.tokenizer.encode(
                positive["prompt"], add_special_tokens=False
            )
        )
        if (
            aligned_negative != sample["aligned_negative_prompt"]
            or alignment != sample["alignment"]
            or response_start != int(sample["response_start"])
        ):
            raise ValueError(
                f"Frozen end-to-end prompt changed: {sample['pair_id']}."
            )
        positive_prompt_ids = model.tokenizer.encode(
            positive["prompt"], add_special_tokens=False
        )
        aligned_negative_ids = model.tokenizer.encode(
            aligned_negative, add_special_tokens=False
        )
        record = {
            "pair_id": sample["pair_id"],
            "positive_record_id": positive["record_id"],
            "negative_record_id": negative["record_id"],
            "question_index": question_index,
            "instruction_index": instruction_index,
        }
        components = {name: [] for name in values}
        for answer_label, answer in (
            ("positive", positive["answer"]),
            ("negative", negative["answer"]),
        ):
            clean_text = positive["prompt"] + answer
            corrupt_text = aligned_negative + answer
            clean_length = len(
                model.tokenizer.encode(
                    clean_text, add_special_tokens=False
                )
            )
            corrupt_length = len(
                model.tokenizer.encode(
                    corrupt_text, add_special_tokens=False
                )
            )
            if clean_length != corrupt_length:
                raise ValueError("Held-out transcript pair is not token-aligned.")
            clean_ids = model.tokenizer.encode(
                clean_text, add_special_tokens=False
            )
            corrupt_ids = model.tokenizer.encode(
                corrupt_text, add_special_tokens=False
            )
            if (
                clean_ids[:response_start] != positive_prompt_ids
                or corrupt_ids[:response_start] != aligned_negative_ids
                or clean_ids[response_start:] != corrupt_ids[response_start:]
            ):
                raise ValueError(
                    "Held-out answer boundary is not stable under prompt "
                    "concatenation."
                )
            if clean_length > args.max_length:
                raise ValueError(
                    "Held-out transcript exceeds --max-length; refusing to "
                    "change the response-logprob estimand by truncation."
                )
            observed_lengths.append(clean_length)
            clean_tokens, clean_mask = tokenize_prompts(
                model, [clean_text], max_length=args.max_length
            )
            corrupt_tokens, corrupt_mask = tokenize_prompts(
                model, [corrupt_text], max_length=args.max_length
            )
            if tuple(clean_tokens.shape) != tuple(corrupt_tokens.shape):
                raise ValueError("Held-out transcript pair is not token-aligned.")
            if not torch.equal(
                clean_tokens[:, response_start:],
                corrupt_tokens[:, response_start:],
            ):
                raise ValueError(
                    "Held-out answer tokens differ across the prompt counterfactual."
                )
            response_mask = _response_mask(
                clean_tokens, clean_mask, [response_start]
            )
            with torch.inference_mode():
                clean_logits = model(
                    clean_tokens, attention_mask=clean_mask
                )
                corrupt_logits = model(
                    corrupt_tokens, attention_mask=corrupt_mask
                )
            logits_by_condition = {
                "clean": clean_logits,
                "corrupt": corrupt_logits,
                "circuit_only": mixed_logits_from_prompt_pair(
                    model,
                    graph,
                    clean_tokens,
                    clean_mask,
                    corrupt_tokens,
                    corrupt_mask,
                    patch_selected=False,
                    cache_dtype=next(model.parameters()).dtype,
                ),
                "circuit_ablated": mixed_logits_from_prompt_pair(
                    model,
                    graph,
                    clean_tokens,
                    clean_mask,
                    corrupt_tokens,
                    corrupt_mask,
                    patch_selected=True,
                    cache_dtype=next(model.parameters()).dtype,
                ),
            }
            for condition, logits in logits_by_condition.items():
                target_tokens = (
                    corrupt_tokens if condition == "corrupt" else clean_tokens
                )
                value = float(
                    response_logprob_values(
                        logits, target_tokens, response_mask
                    )[0].item()
                )
                components[condition].append(value)
                record[f"{condition}_{answer_label}_logprob"] = value
        for condition, pair in components.items():
            margin = pair[0] - pair[1]
            values[condition].append(margin)
            record[condition] = margin
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
    output_name = (
        args.output_tag
        or f"k{args.k}_n{args.num_prompts}_prompt_disjoint"
    )
    output = discovery / "evaluation" / output_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(instances).to_csv(output / "instance_metrics.csv", index=False)
    aggregate = {
        "schema_version": 2,
        "status": "complete",
        "method": "end-to-end natural-prompt contrastive EAP-IG",
        "claim_scope": "held-out response-logprob mediation",
        "run_name": args.run_name,
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "connectivity": connectivity,
        "n_prompts": len(samples),
        "source_manifest": str(discovery_manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": sha256_file(discovery_manifest_path),
        "evaluation_manifest": str(
            evaluation_manifest_path.relative_to(ROOT)
        ),
        "evaluation_manifest_sha256": sha256_file(
            evaluation_manifest_path
        ),
        "discovery_metadata": str(metadata_path.relative_to(ROOT)),
        "discovery_metadata_sha256": sha256_file(metadata_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "metric": "contrastive_response_logprob",
        "means": {
            name: float(tensor.mean().item())
            for name, tensor in tensors.items()
        },
        "faithfulness": normalized_faithfulness(
            tensors["clean"], tensors["corrupt"], tensors["circuit_only"]
        ),
        "necessity": normalized_necessity(
            tensors["clean"], tensors["corrupt"], tensors["circuit_ablated"]
        ),
        "max_length": args.max_length,
        "max_observed_tokens": max(observed_lengths),
        "truncation": False,
        "behavioral_validation_required": True,
    }
    write_json(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

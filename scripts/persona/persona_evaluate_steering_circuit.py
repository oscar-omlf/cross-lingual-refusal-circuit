#!/usr/bin/env python3
"""Evaluate a steering-path circuit on held-out high-trait transcripts."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _bootstrap import ROOT


def _balanced(rows: list[dict], *, seed: int) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["question_index"])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
        values.sort(key=lambda row: float(row["trait_score"]), reverse=True)
    output = []
    for depth in range(max(map(len, groups.values()))):
        for question in sorted(groups):
            if depth < len(groups[question]):
                output.append(groups[question][depth])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
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
        load_circuit_edge_names,
        make_steering_to_logits_graph,
        set_selected_edges,
    )
    from refusal_circuit.interventions import masked_activation_addition_hook
    from refusal_circuit.model import load_model, tokenize_prompts
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
    corrupt_coefficient = float(
        discovery_meta["steering"]["corrupt_coefficient"]
    )
    excluded = set(discovery_meta["selection"]["record_ids"])
    minimum_trait = float(discovery_meta["selection"]["minimum_trait_score"])
    minimum_coherence = float(
        discovery_meta["selection"]["minimum_coherence_score"]
    )
    positive_path = intermediate / "judged/pos_scored.jsonl"
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
    ]
    selected = _balanced(candidates, seed=args.seed)[: args.num_prompts]
    if len(selected) != args.num_prompts:
        raise ValueError(f"Only {len(selected)} held-out high-trait transcripts available")

    model_cfg = config["model"]
    model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=args.dtype,
        revision=model_cfg.get("revision"),
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
        discovery
        / "circuits_nested"
        / f"steering_path_eapig_top_{args.k}.json"
    )
    circuit_payload = json.loads(circuit_path.read_text(encoding="utf-8"))
    if float(circuit_payload["corrupt_coefficient"]) != corrupt_coefficient:
        raise ValueError("Circuit and discovery corrupt coefficients disagree.")
    edge_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, edge_names)

    values = {
        name: []
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    instances = []
    for row in selected:
        text = row["prompt"] + row["answer"]
        response_start = len(
            model.tokenizer.encode(row["prompt"], add_special_tokens=False)
        )
        tokens, attention_mask = tokenize_prompts(
            model, [text], max_length=args.max_length
        )
        response_mask = _response_mask(tokens, attention_mask, [response_start])
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
        with torch.inference_mode(), model.hooks(fwd_hooks=[corrupt_hook]):
            corrupt_logits = model(tokens, attention_mask=attention_mask)
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
        record = {
            "record_id": row["record_id"],
            "pair_id": row["pair_id"],
            "trait_score": float(row["trait_score"]),
            "coherence_score": float(row["coherence_score"]),
        }
        for condition, logits in logits_by_condition.items():
            value = float(
                response_logprob_values(logits, tokens, response_mask).item()
            )
            values[condition].append(value)
            record[condition] = value
        instances.append(record)

    tensors = {name: torch.tensor(items) for name, items in values.items()}
    faithfulness = normalized_faithfulness(
        tensors["clean"], tensors["corrupt"], tensors["circuit_only"]
    )
    necessity = normalized_necessity(
        tensors["clean"], tensors["corrupt"], tensors["circuit_ablated"]
    )
    output = discovery / "evaluation" / f"k{args.k}_n{args.num_prompts}_seed{args.seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains evaluation results")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(instances).to_csv(output / "instance_metrics.csv", index=False)
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "claim_scope": "held-out teacher-forced generation-logit mediation test",
        "run_name": args.run_name,
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "eligible_downstream_edges": len(graph.edges),
        "n_prompts": len(selected),
        "seed": args.seed,
        "record_ids": [row["record_id"] for row in selected],
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
        "metric": "mean autoregressive log probability of held-out high-trait response tokens",
        "means": {
            name: float(tensor.mean().item()) for name, tensor in tensors.items()
        },
        "faithfulness": faithfulness,
        "necessity": necessity,
        "max_length": args.max_length,
        "behavioral_validation_required": True,
    }
    write_json(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

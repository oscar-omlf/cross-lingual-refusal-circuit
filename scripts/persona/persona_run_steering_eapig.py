#!/usr/bin/env python3
"""Discover a downstream behavioral circuit along a persona-steering path.

Unlike the earlier response-projection circuit, this graph begins at the
validated response-time steering site and terminates at model logits.  Its
preferred metric contrasts paired high- and low-trait response likelihoods
under the same prompt.
The corrupt endpoint uses a behaviorally validated negative steering
coefficient; EAP-IG integrates gradients from that endpoint back to clean.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from _bootstrap import ROOT


def _balanced_high_trait(rows: list[dict], *, seed: int) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["question_index"])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        # Score is the primary development filter; seeded tie-breaking avoids
        # silently selecting only the first instruction/rollout.
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
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--corrupt-coefficient", type=float, default=-2.0)
    parser.add_argument("--minimum-trait-score", type=float, default=80.0)
    parser.add_argument("--minimum-coherence-score", type=float, default=80.0)
    parser.add_argument("--maximum-negative-trait-score", type=float, default=20.0)
    parser.add_argument(
        "--metric",
        choices=("response_logprob", "contrastive_response_logprob"),
        default="response_logprob",
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=(100, 300, 1000))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("--run-name must be a filesystem-safe identifier")
    if args.corrupt_coefficient >= 0:
        raise ValueError("Behavioral corrupt coefficient must be negative.")

    import pandas as pd
    import torch

    from refusal_circuit.eapig import (
        AttributionConfig,
        attribute_steering_contrastive_response_logprob,
        attribute_steering_response_logprob,
    )
    from refusal_circuit.graph import (
        edge_record,
        make_steering_to_logits_graph,
        paper_greedy_edges,
        write_edge_scores_parquet,
        write_top_edges_summary,
    )
    from refusal_circuit.model import load_model
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    trait_slug = config["trait"]["name"]
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    vector_root = result_root / "vector"
    output = result_root / "circuit_discovery_steering_path" / args.run_name
    protected_old = result_root / "circuit_discovery"
    if output.resolve() == protected_old.resolve() or protected_old in output.parents:
        raise RuntimeError("Steering-path output must not overlap the frozen projection circuit.")
    score_path = output / "eapig_edge_scores.parquet"
    metadata_path = output / "eapig_run_metadata.json"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} already contains results")

    effective = set(pd.read_csv(vector_root / "effective_pairs.csv")["pair_id"].astype(str))
    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    negative_by_pair = {
        row["pair_id"]: row for row in read_jsonl(negative_path)
    }
    candidates = [
        row
        for row in read_jsonl(positive_path)
        if row["pair_id"] in effective
        and row.get("trait_score") is not None
        and float(row["trait_score"]) >= args.minimum_trait_score
        and row.get("coherence_score") is not None
        and float(row["coherence_score"]) >= args.minimum_coherence_score
        and (
            args.metric == "response_logprob"
            or (
                row["pair_id"] in negative_by_pair
                and negative_by_pair[row["pair_id"]].get("trait_score") is not None
                and float(negative_by_pair[row["pair_id"]]["trait_score"])
                <= args.maximum_negative_trait_score
                and negative_by_pair[row["pair_id"]].get("coherence_score") is not None
                and float(negative_by_pair[row["pair_id"]]["coherence_score"])
                >= args.minimum_coherence_score
            )
        )
    ]
    selected = _balanced_high_trait(candidates, seed=args.seed)[: args.num_prompts]
    if len(selected) != args.num_prompts:
        raise ValueError(
            f"Requested {args.num_prompts} high-trait transcripts, found {len(selected)}"
        )

    model_cfg = config["model"]
    model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=args.dtype,
        revision=model_cfg.get("revision"),
    )
    vector_path = vector_root / f"{trait_slug}_response_avg_diff.pt"
    vector_stack = torch.load(
        vector_path, map_location=next(model.parameters()).device, weights_only=True
    )
    layer = int(config["vector"]["paper_layer"])
    raw_direction = vector_stack[layer] if vector_stack.ndim == 2 else vector_stack
    target_hook = config["vector"]["transformer_lens_target_hook"]
    expected_hook = f"blocks.{layer}.hook_resid_pre"
    if target_hook != expected_hook:
        raise ValueError(f"Frozen layer mapping expected {expected_hook}, found {target_hook}")

    transcripts = []
    transcript_pairs = []
    boundaries = []
    for row in selected:
        response_start = len(
            model.tokenizer.encode(row["prompt"], add_special_tokens=False)
        )
        positive_text = row["prompt"] + row["answer"]
        positive_used_length = min(
            len(model.tokenizer.encode(positive_text, add_special_tokens=False)),
            args.max_length,
        )
        if response_start >= positive_used_length:
            raise ValueError(f"Truncation removed response for {row['record_id']}")
        boundary = {
            "record_id": row["record_id"],
            "pair_id": row["pair_id"],
            "response_start": response_start,
            "positive_used_length": positive_used_length,
            "positive_trait_score": float(row["trait_score"]),
            "positive_coherence_score": float(row["coherence_score"]),
        }
        if args.metric == "contrastive_response_logprob":
            negative = negative_by_pair[row["pair_id"]]
            # Deliberately score both answers under the positive prompt. The
            # negative answer supplies content, not a different conditioning
            # context, so the metric is a genuine same-prompt preference.
            negative_text = row["prompt"] + negative["answer"]
            negative_used_length = min(
                len(
                    model.tokenizer.encode(
                        negative_text, add_special_tokens=False
                    )
                ),
                args.max_length,
            )
            if response_start >= negative_used_length:
                raise ValueError(
                    f"Truncation removed negative response for {row['pair_id']}"
                )
            transcript_pairs.append(
                (
                    (positive_text, response_start),
                    (negative_text, response_start),
                )
            )
            boundary.update(
                {
                    "negative_record_id": negative["record_id"],
                    "negative_used_length": negative_used_length,
                    "negative_trait_score": float(negative["trait_score"]),
                    "negative_coherence_score": float(
                        negative["coherence_score"]
                    ),
                }
            )
        else:
            transcripts.append((positive_text, response_start))
            boundary["used_length"] = positive_used_length
        boundaries.append(boundary)

    graph = make_steering_to_logits_graph(model, layer)
    attribution_config = AttributionConfig(
        batch_size=args.batch_size,
        max_length=args.max_length,
        ig_steps=args.ig_steps,
    )
    if args.metric == "contrastive_response_logprob":
        metric_diagnostics = attribute_steering_contrastive_response_logprob(
            model,
            graph,
            transcript_pairs,
            raw_direction=raw_direction,
            target_hook=target_hook,
            corrupt_coefficient=args.corrupt_coefficient,
            config=attribution_config,
        )
    else:
        metric_diagnostics = attribute_steering_response_logprob(
            model,
            graph,
            transcripts,
            raw_direction=raw_direction,
            target_hook=target_hook,
            corrupt_coefficient=args.corrupt_coefficient,
            config=attribution_config,
        )
    output.mkdir(parents=True, exist_ok=True)
    write_edge_scores_parquet(graph, score_path)
    write_top_edges_summary(
        graph,
        output / "top_edges_summary_eapig.md",
        method="steering_path_eapig",
    )

    circuit_paths = []
    circuit_root = output / "circuits_nested"
    for requested_k in args.k_values:
        selected_edges = paper_greedy_edges(
            graph,
            min(requested_k, len(graph.edges)),
            prune_parentless=False,
        )
        payload = {
            "schema_version": 1,
            "method": "steering_path_eapig",
            "selection": "paper_greedy_absolute_score_with_childless_pruning_only",
            "claim_scope": "downstream mediator candidate; free-generation validation required",
            "requested_k": requested_k,
            "retained_edges": len(selected_edges),
            "score_source": str(score_path.relative_to(ROOT)),
            "corrupt_coefficient": args.corrupt_coefficient,
            "target": {
                "hook": graph.nodes["logits"].in_hook,
                "positions": "response_next_token_logits",
                "metric": args.metric,
                "causal_timing": "each logit precedes its scored response token",
            },
            "source_intervention": {
                "hook": target_hook,
                "layer": layer,
                "coefficient": args.corrupt_coefficient,
                "vector": str(vector_path.relative_to(ROOT)),
            },
            "edges": [edge_record(edge) for edge in selected_edges],
        }
        path = circuit_root / f"steering_path_eapig_top_{requested_k}.json"
        write_json(path, payload)
        circuit_paths.append(path)

    metadata = {
        "schema_version": 1,
        "status": "complete",
        "canonical_persona_protocol": True,
        "method": "coefficient-path EAP-IG",
        "claim_scope": (
            "candidate downstream circuit mediating the causal effect of the persona vector "
            "on generation logits; generated behavior must be validated separately"
        ),
        "model": model_cfg,
        "trait": trait_slug,
        "source": str(positive_path.relative_to(ROOT)),
        "source_sha256": sha256_file(positive_path),
        "selection": {
            "method": "question-balanced high-trait effective positive transcripts",
            "seed": args.seed,
            "minimum_trait_score": args.minimum_trait_score,
            "minimum_coherence_score": args.minimum_coherence_score,
            "maximum_negative_trait_score": (
                args.maximum_negative_trait_score
                if args.metric == "contrastive_response_logprob"
                else None
            ),
            "n_prompts": len(selected),
            "record_ids": [row["record_id"] for row in selected],
            "boundaries": boundaries,
        },
        "steering": {
            "vector": str(vector_path.relative_to(ROOT)),
            "vector_sha256": sha256_file(vector_path),
            "raw_vector_norm": float(raw_direction.float().norm().item()),
            "hook": target_hook,
            "layer": layer,
            "clean_coefficient": 0.0,
            "corrupt_coefficient": args.corrupt_coefficient,
            "teacher_forced_positions": (
                "last prompt position and response-prefix positions predicting scored tokens"
            ),
        },
        "metric": {
            "name": args.metric,
            "behavioral_alignment": (
                "paired answers passed frozen sycophancy and coherence thresholds; "
                "both answers are scored under the same positive-trait prompt"
                if args.metric == "contrastive_response_logprob"
                else "responses passed the frozen sycophancy and coherence judge thresholds"
            ),
            **metric_diagnostics,
        },
        "ig_steps": args.ig_steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "eligible_downstream_edges": len(graph.edges),
        "score_file": str(score_path.relative_to(ROOT)),
        "score_sha256": sha256_file(score_path),
        "circuits": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for path in circuit_paths
        ],
        "protected_projection_circuit": str(protected_old.relative_to(ROOT)),
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Discover an end-to-end natural-prompt persona circuit to response logits."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_negative_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument(
        "--manifest-sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional frozen-manifest subset for an IG convergence pilot.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--ig-steps", type=int, default=20)
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=(500, 1000, 3000, 5000)
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("--run-name must be a filesystem-safe identifier")
    if args.batch_size != 1:
        raise ValueError("End-to-end contrastive attribution requires batch size 1.")
    if any(k <= 0 for k in args.k_values):
        raise ValueError("All k values must be positive.")
    if len(args.k_values) != len(set(args.k_values)):
        raise ValueError("k values must be unique.")
    if args.num_prompts <= 0 or args.max_length <= 0 or args.ig_steps <= 0:
        raise ValueError("Prompt count, max length, and IG steps must be positive.")

    import torch

    from refusal_circuit.eapig import (
        AttributionConfig,
        attribute_contrastive_prompt_response_logprob,
    )
    from refusal_circuit.graph import (
        circuit_connectivity,
        edge_record,
        make_logit_graph,
        paper_greedy_edges,
        write_edge_scores_parquet,
        write_top_edges_summary,
    )
    from refusal_circuit.model import load_model, validate_model_architecture
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    trait_slug = config["trait"]["name"]
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    official_root = ROOT / config["paths"]["official_root"]
    output = result_root / "circuit_discovery_end_to_end" / args.run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results")

    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    positive_by_record = {
        row["record_id"]: row for row in read_jsonl(positive_path)
    }
    negative_by_record = {
        row["record_id"]: row for row in read_jsonl(negative_path)
    }
    manifest_path = (
        args.sample_manifest
        if args.sample_manifest.is_absolute()
        else ROOT / args.sample_manifest
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "frozen"
        or manifest.get("protocol")
        not in {"shared_prompt_v1", "shared_prompt_full_transcript_v2"}
        or manifest["sources"]["positive_sha256"] != sha256_file(positive_path)
        or manifest["sources"]["negative_sha256"] != sha256_file(negative_path)
    ):
        raise ValueError("Shared sample manifest failed its provenance gate.")
    samples = manifest["discovery"]["samples"]
    if args.manifest_sample_indices is not None:
        if len(args.manifest_sample_indices) != len(
            set(args.manifest_sample_indices)
        ):
            raise ValueError("Manifest sample indices must be unique.")
        if any(
            index < 0 or index >= len(samples)
            for index in args.manifest_sample_indices
        ):
            raise IndexError("Manifest sample index is out of range.")
        samples = [samples[index] for index in args.manifest_sample_indices]
    if len(samples) != args.num_prompts:
        raise ValueError(
            f"Manifest has {len(samples)} discovery prompts, "
            f"but --num-prompts={args.num_prompts}."
        )

    trait_data_path = official_root / "extract.json"
    trait_data = json.loads(trait_data_path.read_text(encoding="utf-8"))
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

    transcript_pairs = []
    boundaries = []
    for sample in samples:
        positive = positive_by_record[sample["positive_record_id"]]
        negative = negative_by_record[sample["negative_record_id"]]
        if (
            positive["pair_id"] != sample["pair_id"]
            or negative["pair_id"] != sample["pair_id"]
            or positive["prompt"] != sample["positive_prompt"]
            or negative["prompt"] != sample["negative_prompt"]
        ):
            raise ValueError(f"Manifest row mismatch for {sample['pair_id']}.")
        question_index = int(sample["question_index"])
        instruction_index = int(sample["instruction_index"])
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
        positive_spec = (
            positive["prompt"] + positive["answer"],
            aligned_negative + positive["answer"],
            response_start,
        )
        negative_spec = (
            positive["prompt"] + negative["answer"],
            aligned_negative + negative["answer"],
            response_start,
        )
        positive_prompt_ids = model.tokenizer.encode(
            positive["prompt"], add_special_tokens=False
        )
        aligned_negative_ids = model.tokenizer.encode(
            aligned_negative, add_special_tokens=False
        )
        lengths = {
            "positive_clean": len(
                model.tokenizer.encode(
                    positive_spec[0], add_special_tokens=False
                )
            ),
            "positive_corrupt": len(
                model.tokenizer.encode(
                    positive_spec[1], add_special_tokens=False
                )
            ),
            "negative_clean": len(
                model.tokenizer.encode(
                    negative_spec[0], add_special_tokens=False
                )
            ),
            "negative_corrupt": len(
                model.tokenizer.encode(
                    negative_spec[1], add_special_tokens=False
                )
            ),
        }
        if lengths["positive_clean"] != lengths["positive_corrupt"]:
            raise ValueError("Positive-answer transcript pair is not token-aligned.")
        if lengths["negative_clean"] != lengths["negative_corrupt"]:
            raise ValueError("Negative-answer transcript pair is not token-aligned.")
        for label, (clean_text, corrupt_text, _) in (
            ("positive", positive_spec),
            ("negative", negative_spec),
        ):
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
                    f"{label.capitalize()} answer boundary is not stable under "
                    "prompt concatenation."
                )
        if max(lengths.values()) > args.max_length:
            raise ValueError(
                f"Transcript {sample['pair_id']} exceeds --max-length without "
                "a valid no-truncation estimand."
            )
        transcript_pairs.append((positive_spec, negative_spec))
        boundaries.append(
            {
                "pair_id": sample["pair_id"],
                "positive_record_id": positive["record_id"],
                "negative_record_id": negative["record_id"],
                "question_index": question_index,
                "instruction_index": instruction_index,
                "response_start": response_start,
                "alignment": alignment,
                "lengths": lengths,
            }
        )

    graph = make_logit_graph(model)
    diagnostics = attribute_contrastive_prompt_response_logprob(
        model,
        graph,
        transcript_pairs,
        config=AttributionConfig(
            batch_size=args.batch_size,
            max_length=args.max_length,
            ig_steps=args.ig_steps,
        ),
    )
    clean_corrupt_margin = float(
        diagnostics["clean_minus_corrupt_margin"]
    )
    if (
        not math.isfinite(clean_corrupt_margin)
        or clean_corrupt_margin <= 0
    ):
        raise RuntimeError(
            "Positive-persona prompts did not produce a finite, positive "
            "increase in the high-minus-low response-logprob margin. Do not "
            "interpret or select a circuit from this run."
        )
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "eapig_edge_scores.parquet"
    write_edge_scores_parquet(graph, score_path)
    write_top_edges_summary(
        graph,
        output / "top_edges_summary_eapig.md",
        method="end_to_end_contrastive_prompt_response_eapig",
    )

    circuit_records = []
    for family, prune_parentless, directory in (
        ("source_to_sink", True, "circuits_source_to_sink"),
        ("causal_cut", False, "circuits_cut_sets"),
    ):
        for requested_k in args.k_values:
            selected = paper_greedy_edges(
                graph,
                min(requested_k, len(graph.edges)),
                prune_parentless=prune_parentless,
            )
            path = (
                output
                / directory
                / f"end_to_end_eapig_top_{requested_k}.json"
            )
            payload = {
                "schema_version": 2,
                "method": "end_to_end_contrastive_prompt_response_eapig",
                "selection": (
                    "paper_greedy_absolute_score_with_recursive_source_sink_pruning"
                    if prune_parentless
                    else "paper_greedy_absolute_score_with_sink_pruning_only"
                ),
                "circuit_family": family,
                "claim_scope": (
                    "end-to-end natural-prompt behavioral circuit candidate"
                    if prune_parentless
                    else (
                        "end-to-end causal interception set for necessity tests; "
                        "not a reconstructive circuit"
                    )
                ),
                "requested_k": requested_k,
                "retained_edges": len(selected),
                "selection_diagnostics": graph.last_selection_diagnostics,
                "connectivity": circuit_connectivity(
                    graph, {edge.name for edge in selected}
                ),
                "score_source": str(score_path.relative_to(ROOT)),
                "counterfactual": (
                    "token-aligned positive-versus-negative persona prompt; "
                    "answer tokens held fixed within each high/low response"
                ),
                "target": {
                    "hook": graph.nodes["logits"].in_hook,
                    "positions": "response_next_token_logits",
                    "metric": "contrastive_response_logprob",
                },
                "edges": [edge_record(edge) for edge in selected],
            }
            write_json(path, payload)
            circuit_records.append(
                {
                    "family": family,
                    "requested_k": requested_k,
                    "retained_edges": len(selected),
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
            )

    metadata = {
        "schema_version": 2,
        "status": "complete",
        "method": "end-to-end natural-prompt contrastive EAP-IG",
        "claim_scope": (
            "independent behavioral circuit comparator from natural persona "
            "prompt state to paired response likelihood"
        ),
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "trait": trait_slug,
        "sample_manifest": str(manifest_path.relative_to(ROOT)),
        "sample_manifest_sha256": sha256_file(manifest_path),
        "sources": {
            "positive": str(positive_path.relative_to(ROOT)),
            "positive_sha256": sha256_file(positive_path),
            "negative": str(negative_path.relative_to(ROOT)),
            "negative_sha256": sha256_file(negative_path),
            "trait_data": str(trait_data_path.relative_to(ROOT)),
            "trait_data_sha256": sha256_file(trait_data_path),
        },
        "selection": {
            "n_prompts": len(samples),
            "manifest_sample_indices": args.manifest_sample_indices,
            "boundaries": boundaries,
        },
        "metric": {
            "name": "contrastive_response_logprob",
            **diagnostics,
        },
        "ig_steps": args.ig_steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "truncation": False,
        "eligible_edges": len(graph.edges),
        "score_file": str(score_path.relative_to(ROOT)),
        "score_sha256": sha256_file(score_path),
        "circuits": circuit_records,
    }
    write_json(output / "eapig_run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

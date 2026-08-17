#!/usr/bin/env python3
"""Discover an upstream persona circuit from matched positive/negative states.

This is the upstream analogue of signed persona steering. It does not project
activations to zero. Clean prompts explicitly elicit the positive persona,
corrupt prompts elicit the paired negative persona, and EAP-IG follows the
root-residual path between those states. The graph terminates at the configured
final-prompt projection onto the trait's response-average direction.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from _bootstrap import ROOT


def _prompt_groups(rows: list[dict]) -> dict[tuple[int, int], dict]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["question_index"]), int(row["instruction_index"]))].append(row)
    output = {}
    for key, values in groups.items():
        prompts = {row["prompt"] for row in values}
        if len(prompts) != 1:
            raise ValueError(f"Prompt group {key} is not unique")
        output[key] = {
            "prompt": next(iter(prompts)),
            "mean_trait_score": sum(float(row["trait_score"]) for row in values)
            / len(values),
            "mean_coherence_score": sum(
                float(row["coherence_score"]) for row in values
            )
            / len(values),
            "rollouts": len(values),
        }
    return output


def _select_pairs(
    positive: dict[tuple[int, int], dict],
    negative: dict[tuple[int, int], dict],
    *,
    n: int,
    seed: int,
) -> list[dict]:
    if positive.keys() != negative.keys():
        raise ValueError("Positive and negative prompt-group keys differ")
    by_question: dict[int, list[dict]] = defaultdict(list)
    for (question_index, instruction_index), pos in positive.items():
        neg = negative[(question_index, instruction_index)]
        by_question[question_index].append(
            {
                "question_index": question_index,
                "instruction_index": instruction_index,
                "positive_prompt": pos["prompt"],
                "negative_prompt": neg["prompt"],
                "positive_trait_score": pos["mean_trait_score"],
                "negative_trait_score": neg["mean_trait_score"],
                "trait_contrast": (
                    pos["mean_trait_score"] - neg["mean_trait_score"]
                ),
                "positive_coherence_score": pos["mean_coherence_score"],
                "negative_coherence_score": neg["mean_coherence_score"],
                "rollouts_per_polarity": min(pos["rollouts"], neg["rollouts"]),
            }
        )
    rng = random.Random(seed)
    for values in by_question.values():
        rng.shuffle(values)
        values.sort(key=lambda row: row["trait_contrast"], reverse=True)
    selected = []
    for depth in range(max(map(len, by_question.values()))):
        for question_index in sorted(by_question):
            if depth < len(by_question[question_index]):
                selected.append(by_question[question_index][depth])
    if len(selected) < n:
        raise ValueError(f"Requested {n} prompt pairs, found {len(selected)}")
    return selected[:n]


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _common_suffix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(reversed(left), reversed(right), strict=False):
        if a != b:
            break
        count += 1
    return count


def _align_negative_prompt(
    tokenizer,
    *,
    question: str,
    negative_assistant_name: str,
    negative_instruction: str,
    positive_prompt: str,
    expected_negative_prompt: str | None = None,
) -> tuple[str, dict]:
    """Add semantically empty trailing whitespace to match positive length."""
    from refusal_circuit.persona import make_system_prompt

    base_system = make_system_prompt(negative_assistant_name, negative_instruction)
    return _align_prompt_pair(
        tokenizer,
        question=question,
        negative_system_prompt=base_system,
        positive_prompt=positive_prompt,
        expected_negative_prompt=expected_negative_prompt,
    )


def _align_prompt_pair(
    tokenizer,
    *,
    question: str,
    negative_system_prompt: str,
    positive_prompt: str,
    expected_negative_prompt: str | None = None,
) -> tuple[str, dict]:
    """Token-align an arbitrary negative system prompt to a positive prompt."""
    base_system = negative_system_prompt

    def render(extra: str) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": base_system + extra},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    if (
        expected_negative_prompt is not None
        and _ids(tokenizer, render("")) != _ids(tokenizer, expected_negative_prompt)
    ):
        raise ValueError("Reconstructed canonical negative prompt differs from saved data")
    positive_ids = _ids(tokenizer, positive_prompt)
    base_ids = _ids(tokenizer, render(""))
    if len(base_ids) > len(positive_ids):
        raise ValueError("Negative prompt is longer than its positive pair")
    token_gap = len(positive_ids) - len(base_ids)
    # Qwen tokenizes each carriage return as one whitespace token.  Try this
    # transparent exact-length alignment first, retaining broader whitespace
    # fallbacks only for tokenizer portability.
    candidates = ["\r" * token_gap]
    units = (" ", "\n", " \n", "\n ")
    for repeats in range(1, 65):
        candidates.extend(unit * repeats for unit in units)
    for padding in candidates:
        aligned = render(padding)
        aligned_ids = _ids(tokenizer, aligned)
        if len(aligned_ids) != len(positive_ids):
            continue
        common_suffix = _common_suffix_length(positive_ids, aligned_ids)
        question_tokens = len(_ids(tokenizer, question))
        if common_suffix < question_tokens + 4:
            continue
        return aligned, {
            "positive_tokens": len(positive_ids),
            "negative_tokens_before_alignment": len(base_ids),
            "negative_tokens_after_alignment": len(aligned_ids),
            "negative_trailing_whitespace_repr": repr(padding),
            "common_suffix_tokens": common_suffix,
        }
    raise ValueError("Could not token-align negative prompt with neutral whitespace")


def _align_system_prompt_pair(
    tokenizer,
    *,
    question: str,
    positive_system_prompt: str,
    negative_system_prompt: str,
) -> tuple[str, str, dict]:
    """Pad whichever localized system prompt is shorter to exact token parity."""

    def render(system_prompt: str, extra: str) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt + extra},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    positive = render(positive_system_prompt, "")
    negative = render(negative_system_prompt, "")
    positive_ids = _ids(tokenizer, positive)
    negative_ids = _ids(tokenizer, negative)
    if len(positive_ids) == len(negative_ids):
        return positive, negative, {
            "positive_tokens_before_alignment": len(positive_ids),
            "negative_tokens_before_alignment": len(negative_ids),
            "padded_polarity": None,
            "trailing_whitespace_repr": "''",
            "tokens_after_alignment": len(positive_ids),
            "common_suffix_tokens": _common_suffix_length(
                positive_ids, negative_ids
            ),
        }
    pad_positive = len(positive_ids) < len(negative_ids)
    shorter_system = (
        positive_system_prompt if pad_positive else negative_system_prompt
    )
    longer_prompt = negative if pad_positive else positive
    longer_ids = negative_ids if pad_positive else positive_ids
    gap = abs(len(positive_ids) - len(negative_ids))
    candidates = ["\r" * gap]
    for repeats in range(1, 65):
        candidates.extend(
            unit * repeats for unit in (" ", "\n", " \n", "\n ")
        )
    question_tokens = len(_ids(tokenizer, question))
    for padding in candidates:
        shorter_prompt = render(shorter_system, padding)
        shorter_ids = _ids(tokenizer, shorter_prompt)
        if len(shorter_ids) != len(longer_ids):
            continue
        common_suffix = _common_suffix_length(longer_ids, shorter_ids)
        if common_suffix < question_tokens + 4:
            continue
        aligned_positive = shorter_prompt if pad_positive else positive
        aligned_negative = negative if pad_positive else shorter_prompt
        return aligned_positive, aligned_negative, {
            "positive_tokens_before_alignment": len(positive_ids),
            "negative_tokens_before_alignment": len(negative_ids),
            "padded_polarity": "positive" if pad_positive else "negative",
            "trailing_whitespace_repr": repr(padding),
            "tokens_after_alignment": len(longer_ids),
            "common_suffix_tokens": common_suffix,
        }
    raise ValueError("Could not token-align localized prompt pair.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=None,
        help="Frozen shared upstream/downstream discovery sample manifest.",
    )
    parser.add_argument("--num-pairs", type=int, default=64)
    parser.add_argument(
        "--manifest-sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional frozen-manifest subset for an IG convergence pilot.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--k", type=int, default=3000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("--run-name must be a filesystem-safe identifier")

    import torch

    from refusal_circuit.eapig import (
        AttributionConfig,
        attribute_contrastive_prompt_projection,
    )
    from refusal_circuit.graph import (
        circuit_connectivity,
        edge_record,
        make_projection_graph,
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
    output = result_root / "circuit_discovery_upstream_contrastive" / args.run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results")

    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    positive_groups = _prompt_groups(read_jsonl(positive_path))
    negative_groups = _prompt_groups(read_jsonl(negative_path))
    sample_manifest_path = None
    sample_manifest = None
    if args.sample_manifest is None:
        selected = _select_pairs(
            positive_groups,
            negative_groups,
            n=args.num_pairs,
            seed=args.seed,
        )
        selection_method = (
            "question-balanced pairs ranked by positive-minus-negative judged trait"
        )
    else:
        sample_manifest_path = (
            args.sample_manifest
            if args.sample_manifest.is_absolute()
            else ROOT / args.sample_manifest
        )
        sample_manifest = json.loads(
            sample_manifest_path.read_text(encoding="utf-8")
        )
        if (
            sample_manifest.get("status") != "frozen"
            or sample_manifest.get("protocol")
            not in {
                "shared_prompt_v1",
                "shared_prompt_full_transcript_v2",
            }
            or sample_manifest["sources"]["positive_sha256"]
            != sha256_file(positive_path)
            or sample_manifest["sources"]["negative_sha256"]
            != sha256_file(negative_path)
        ):
            raise ValueError("Shared sample manifest failed its provenance gate.")
        samples = sample_manifest["discovery"]["samples"]
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
        if len(samples) != args.num_pairs:
            raise ValueError(
                f"Manifest has {len(samples)} discovery prompts, "
                f"but --num-pairs={args.num_pairs}."
            )
        selected = []
        for sample in samples:
            key = (
                int(sample["question_index"]),
                int(sample["instruction_index"]),
            )
            pos = positive_groups[key]
            neg = negative_groups[key]
            if (
                pos["prompt"] != sample["positive_prompt"]
                or neg["prompt"] != sample["negative_prompt"]
            ):
                raise ValueError(f"Manifest prompt mismatch for {key}.")
            selected.append(
                {
                    "question_index": key[0],
                    "instruction_index": key[1],
                    "positive_prompt": pos["prompt"],
                    "negative_prompt": neg["prompt"],
                    "positive_trait_score": pos["mean_trait_score"],
                    "negative_trait_score": neg["mean_trait_score"],
                    "trait_contrast": (
                        pos["mean_trait_score"] - neg["mean_trait_score"]
                    ),
                    "positive_coherence_score": pos["mean_coherence_score"],
                    "negative_coherence_score": neg["mean_coherence_score"],
                    "rollouts_per_polarity": min(
                        pos["rollouts"], neg["rollouts"]
                    ),
                    "downstream_pair_id": sample["pair_id"],
                }
            )
        selection_method = "frozen shared upstream/downstream unique-prompt manifest"
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
    for row in selected:
        question_index = row["question_index"]
        instruction_index = row["instruction_index"]
        aligned, alignment = _align_negative_prompt(
            model.tokenizer,
            question=trait_data["questions"][question_index],
            negative_assistant_name=config["trait"]["negative_assistant_name"],
            negative_instruction=trait_data["instruction"][instruction_index]["neg"],
            positive_prompt=row["positive_prompt"],
            expected_negative_prompt=row["negative_prompt"],
        )
        row["aligned_negative_prompt"] = aligned
        row["alignment"] = alignment
        if alignment["positive_tokens"] > args.max_length:
            raise ValueError(
                f"Pair {(question_index, instruction_index)} exceeds max length"
            )

    # Use the trait-specific vector artifact so the same pipeline runs for any persona.
    direction_path = result_root / f"vector/{trait_slug}_direction.pt"
    direction = torch.load(
        direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    target_layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    graph = make_projection_graph(model, target_layer)
    diagnostics = attribute_contrastive_prompt_projection(
        model,
        graph,
        [
            (row["positive_prompt"], row["aligned_negative_prompt"])
            for row in selected
        ],
        direction=direction,
        target_hook=target_hook,
        target_position=-1,
        config=AttributionConfig(
            batch_size=args.batch_size,
            max_length=args.max_length,
            ig_steps=args.ig_steps,
        ),
    )

    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "eapig_edge_scores.parquet"
    write_edge_scores_parquet(graph, score_path)
    write_top_edges_summary(
        graph,
        output / "top_edges_summary_eapig.md",
        method="upstream_contrastive_eapig",
    )
    selected_edges = paper_greedy_edges(
        graph, min(args.k, len(graph.edges)), prune_parentless=True
    )
    connected_diagnostics = graph.last_selection_diagnostics
    circuit_path = (
        output / "circuits_source_to_sink" / f"eapig_top_{args.k}.json"
    )
    write_json(
        circuit_path,
        {
            "schema_version": 2,
            "method": "contrastive_state_path_eapig",
            "selection": "paper_greedy_absolute_score_with_recursive_source_sink_pruning",
            "circuit_family": "source_to_sink",
            "claim_scope": (
                "upstream natural persona-state formation circuit candidate; "
                "all retained edges lie on complete source-to-interface paths"
            ),
            "requested_k": args.k,
            "retained_edges": len(selected_edges),
            "selection_diagnostics": connected_diagnostics,
            "connectivity": circuit_connectivity(
                graph, {edge.name for edge in selected_edges}
            ),
            "score_source": str(score_path.relative_to(ROOT)),
            "target": {
                "hook": target_hook,
                "layer": target_layer,
                "position": -1,
                "timing": "final prompt token before response generation",
                "metric": f"signed_projection_onto_unit_{trait_slug}_direction",
            },
            "edges": [edge_record(edge) for edge in selected_edges],
        },
    )
    cut_edges = paper_greedy_edges(
        graph, min(args.k, len(graph.edges)), prune_parentless=False
    )
    cut_diagnostics = graph.last_selection_diagnostics
    cut_path = output / "circuits_cut_sets" / f"eapig_top_{args.k}.json"
    write_json(
        cut_path,
        {
            "schema_version": 2,
            "method": "contrastive_state_path_eapig",
            "selection": "paper_greedy_absolute_score_with_sink_pruning_only",
            "circuit_family": "causal_cut",
            "claim_scope": (
                "upstream causal interception set for necessity tests; "
                "not a reconstructive source-to-interface circuit"
            ),
            "requested_k": args.k,
            "retained_edges": len(cut_edges),
            "selection_diagnostics": cut_diagnostics,
            "connectivity": circuit_connectivity(
                graph, {edge.name for edge in cut_edges}
            ),
            "score_source": str(score_path.relative_to(ROOT)),
            "target": {
                "hook": target_hook,
                "layer": target_layer,
                "position": -1,
                "timing": "final prompt token before response generation",
                "metric": f"signed_projection_onto_unit_{trait_slug}_direction",
            },
            "edges": [edge_record(edge) for edge in cut_edges],
        },
    )
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "method": "contrastive-state-path EAP-IG",
        "claim_scope": (
            "candidate upstream circuit forming the positive-versus-negative "
            "persona-vector coordinate; held-out and behavioral validation required"
        ),
        "no_projection_ablation": True,
        "graph": f"blocks.0.hook_resid_pre through {target_hook}",
        "metric": (
            "final-prompt-token signed projection onto unit "
            f"{trait_slug} direction"
        ),
        "target_hook": target_hook,
        "target_layer": target_layer,
        "target_position": -1,
        "corrupt_baseline": (
            "matched negative-persona system prompt with token-aligned "
            "shared question and assistant-start suffix"
        ),
        "interpolation": (
            "graph-input residual state from negative prompt (alpha=0) to "
            "positive prompt (alpha=1)"
        ),
        "diagnostics": diagnostics,
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "trait": config["trait"]["name"],
        "sources": {
            "positive": str(positive_path.relative_to(ROOT)),
            "positive_sha256": sha256_file(positive_path),
            "negative": str(negative_path.relative_to(ROOT)),
            "negative_sha256": sha256_file(negative_path),
            "trait_data": str(trait_data_path.relative_to(ROOT)),
            "trait_data_sha256": sha256_file(trait_data_path),
        },
        "selection": {
            "method": selection_method,
            "seed": args.seed,
            "n_pairs": len(selected),
            "manifest_sample_indices": args.manifest_sample_indices,
            "prompt_pairs": [
                {
                    **{
                        key: row[key]
                        for key in (
                            "question_index",
                            "instruction_index",
                            "positive_trait_score",
                            "negative_trait_score",
                            "trait_contrast",
                            "positive_coherence_score",
                            "negative_coherence_score",
                            "rollouts_per_polarity",
                            "alignment",
                        )
                    },
                    **(
                        {"downstream_pair_id": row["downstream_pair_id"]}
                        if "downstream_pair_id" in row
                        else {}
                    ),
                }
                for row in selected
            ],
        },
        "direction": str(direction_path.relative_to(ROOT)),
        "direction_sha256": sha256_file(direction_path),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "ig_steps": args.ig_steps,
        "eligible_edges": len(graph.edges),
        "score_file": str(score_path.relative_to(ROOT)),
        "score_sha256": sha256_file(score_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "cut_set": str(cut_path.relative_to(ROOT)),
        "cut_set_sha256": sha256_file(cut_path),
        "requested_k": args.k,
        "retained_edges": len(selected_edges),
        "cut_set_retained_edges": len(cut_edges),
        "circuit_family": "source_to_sink",
    }
    if sample_manifest_path is not None:
        metadata["sample_manifest"] = str(
            sample_manifest_path.relative_to(ROOT)
        )
        metadata["sample_manifest_sha256"] = sha256_file(
            sample_manifest_path
        )
        metadata["shared_protocol"] = sample_manifest["protocol"]
    write_json(output / "eapig_run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

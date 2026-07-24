#!/usr/bin/env python3
"""Cheap causal diagnostic for the selected response-average persona vector.

This intentionally selects a small set of prompts with high, already-judged
baseline sycophancy.  It is a development diagnostic, not a confirmatory
behavior estimate: the goal is to determine whether fixed negative steering
or removal of the selected-layer projection can reverse clear trait behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/"
        "behavioral_pilot/k3000_n20/responses.jsonl",
    )
    parser.add_argument(
        "--source-scored",
        type=Path,
        default=ROOT
        / "data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/"
        "behavioral_pilot/k3000_n20/scored.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/"
        "vector_intervention_diagnostic/top5_high_baseline",
    )
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.interventions import activation_addition_hook, direction_ablation_hook
    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file, write_jsonl

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    result_root = ROOT / config["paths"]["results_root"]
    raw_vector_path = result_root / "vector/sycophantic_response_avg_diff.pt"
    unit_vector_path = result_root / "vector/sycophantic_direction.pt"
    output_path = args.output_root / "responses.jsonl"
    metadata_path = args.output_root / "metadata.json"
    if output_path.exists() or metadata_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} already contains diagnostic output")
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(args.source)
    scored_rows = read_jsonl(args.source_scored)
    raw_by_key = {
        (row["condition"], int(row["question_index"])): row for row in source_rows
    }
    normal_scored = sorted(
        (row for row in scored_rows if row["condition"] == "normal"),
        key=lambda row: float(row["trait_score"]),
        reverse=True,
    )
    selected_scored = normal_scored[: args.num_questions]
    if len(selected_scored) != args.num_questions:
        raise ValueError("source does not contain enough scored normal rows")
    selected_questions = [int(row["question_index"]) for row in selected_scored]

    rows: list[dict] = []
    # These exact generations already exist and have cached judge results.
    for condition in ("normal", "direction_ablated"):
        for question_index in selected_questions:
            source = raw_by_key.get((condition, question_index))
            if source is None:
                raise ValueError(f"missing cached {condition} response for q{question_index}")
            copied = dict(source)
            copied.update(
                {
                    "record_id": f"vector-diag-{condition}-q{question_index:02d}",
                    "diagnostic_selection": "top baseline trait score in cached k3000 n20 pilot",
                    "source_record_id": source["record_id"],
                    "source_path": str(args.source.relative_to(ROOT)),
                }
            )
            rows.append(copied)
    write_jsonl(output_path, rows)

    model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=args.dtype,
        revision=model_cfg.get("revision"),
    )
    vector_stack = torch.load(
        raw_vector_path, map_location=next(model.parameters()).device, weights_only=True
    )
    layer = int(config["vector"]["paper_layer"])
    raw_vector = vector_stack[layer] if vector_stack.ndim == 2 else vector_stack
    unit_vector = torch.load(
        unit_vector_path, map_location=next(model.parameters()).device, weights_only=True
    )
    target_hook = config["vector"]["transformer_lens_target_hook"]
    intervention_specs = (
        (
            "positive_steer_1p5",
            activation_addition_hook(raw_vector, coefficient=1.5, position=-1),
        ),
        (
            "negative_steer_1p5",
            activation_addition_hook(raw_vector, coefficient=-1.5, position=-1),
        ),
        (
            "negative_steer_2",
            activation_addition_hook(raw_vector, coefficient=-2.0, position=-1),
        ),
        (
            "target_projection_ablated",
            direction_ablation_hook(unit_vector, position=-1),
        ),
    )

    normal_sources = {
        question_index: raw_by_key[("normal", question_index)]
        for question_index in selected_questions
    }
    model.set_ungroup_grouped_query_attention(False)
    try:
        for condition, hook_fn in intervention_specs:
            for question_index in tqdm(selected_questions, desc=condition):
                source = normal_sources[question_index]
                prompt_tokens, _ = tokenize_prompts(
                    model, [source["prompt"]], max_length=args.max_length
                )
                with model.hooks(fwd_hooks=[(target_hook, hook_fn)]):
                    generated = model.generate(
                        prompt_tokens,
                        max_new_tokens=args.max_new_tokens,
                        stop_at_eos=True,
                        do_sample=False,
                        use_past_kv_cache=True,
                        return_type="tokens",
                        verbose=False,
                    )
                token_ids = generated[0, prompt_tokens.shape[1] :].tolist()
                row = {
                    key: value
                    for key, value in source.items()
                    if key
                    not in {
                        "record_id",
                        "condition",
                        "answer",
                        "generated_token_count",
                        "hit_max_new_tokens",
                        "requested_k",
                        "retained_edges",
                    }
                }
                row.update(
                    {
                        "schema_version": 1,
                        "record_id": f"vector-diag-{condition}-q{question_index:02d}",
                        "behavior_protocol_id": config["protocol_id"],
                        "condition": condition,
                        "answer": model.tokenizer.decode(token_ids, skip_special_tokens=True),
                        "generated_token_count": len(token_ids),
                        "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
                        "decoding": "greedy",
                        "diagnostic_selection": (
                            "top baseline trait score in cached k3000 n20 pilot"
                        ),
                        "intervention_hook": target_hook,
                    }
                )
                write_jsonl(output_path, [row], mode="a")
    finally:
        model.set_ungroup_grouped_query_attention(True)

    final_rows = read_jsonl(output_path)
    expected = args.num_questions * (2 + len(intervention_specs))
    if len(final_rows) != expected or len({row["record_id"] for row in final_rows}) != expected:
        raise RuntimeError(f"incomplete diagnostic: {len(final_rows)}/{expected}")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "canonical_protocol": False,
        "behavior_protocol_id": config["protocol_id"],
        "claim_scope": "selected high-trait development diagnostic",
        "selection": {
            "method": "top cached normal trait scores",
            "source_scored": str(args.source_scored.relative_to(ROOT)),
            "source_scored_sha256": sha256_file(args.source_scored),
            "question_indices": selected_questions,
            "baseline_trait_scores": [
                float(row["trait_score"]) for row in selected_scored
            ],
        },
        "model": model_cfg,
        "vector_path": str(raw_vector_path.relative_to(ROOT)),
        "vector_sha256": sha256_file(raw_vector_path),
        "unit_vector_path": str(unit_vector_path.relative_to(ROOT)),
        "unit_vector_sha256": sha256_file(unit_vector_path),
        "raw_vector_norm": float(raw_vector.float().norm().item()),
        "layer_one_indexed": layer,
        "intervention_hook": target_hook,
        "conditions": [
            "normal",
            "direction_ablated",
            *[condition for condition, _ in intervention_specs],
        ],
        "cached_conditions": ["normal", "direction_ablated"],
        "new_generation_conditions": [
            condition for condition, _ in intervention_specs
        ],
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "outputs": [
            {
                "path": str(output_path.relative_to(ROOT)),
                "rows": len(final_rows),
                "sha256": sha256_file(output_path),
            }
        ],
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

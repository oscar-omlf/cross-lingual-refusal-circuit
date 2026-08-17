#!/usr/bin/env python3
"""Generate held-out behavior with an end-to-end natural-prompt circuit."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--question-indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("normal", "negative_prompt", "circuit_only", "circuit_ablated"),
        default=("normal", "circuit_only", "circuit_ablated"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("Question indices must be unique.")
    if len(args.conditions) != len(set(args.conditions)):
        raise ValueError("Conditions must be unique.")

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import mixed_logits_from_prompt_pair
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
    from refusal_circuit.persona import (
        make_system_prompt,
        read_jsonl,
        sha256_file,
        token_ids_sha256,
        write_jsonl,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    trait_slug = config["trait"]["name"]
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    discovery = result_root / "circuit_discovery_end_to_end" / args.run_name
    discovery_metadata_path = discovery / "eapig_run_metadata.json"
    discovery_metadata = json.loads(
        discovery_metadata_path.read_text(encoding="utf-8")
    )
    if discovery_metadata.get("status") != "complete":
        raise ValueError("End-to-end discovery run is not complete.")
    if discovery_metadata.get("model") != config["model"]:
        raise ValueError("Discovery and behavior model configurations differ.")
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
    if circuit_payload.get("score_source") != discovery_metadata.get("score_file"):
        raise ValueError("Circuit was not derived from the frozen score table.")

    output_name = (
        args.output_tag
        or f"k{args.k}_q{'-'.join(map(str, args.question_indices))}"
    )
    output = (
        intermediate
        / "end_to_end_circuit_behavior"
        / args.run_name
        / output_name
    )
    raw_path = output / "responses.jsonl"
    metadata_path = output / "metadata.json"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"{output} already contains results; pass --resume.")
    output.mkdir(parents=True, exist_ok=True)

    official_path = ROOT / config["paths"]["official_root"] / "eval.json"
    expected_resume = {
        "behavior_protocol_id": config["protocol_id"],
        "model": config["model"],
        "model_architecture": config["model_architecture"],
        "discovery_metadata_sha256": sha256_file(discovery_metadata_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "requested_k": args.k,
        "question_indices": args.question_indices,
        "conditions": list(args.conditions),
        "decoding": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "backend": "full_prefix_forward_ungrouped_gqa",
        },
        "max_length": args.max_length,
    }
    if args.resume:
        if not metadata_path.exists():
            raise ValueError("Cannot resume without the original metadata.json.")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatched = [
            key
            for key, value in expected_resume.items()
            if previous.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                f"Resume arguments differ from the original run: {mismatched}"
            )
    official = json.loads(official_path.read_text(encoding="utf-8"))
    questions = official["questions"]
    instructions = official["instruction"]
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
    graph = make_logit_graph(model)
    edge_names = load_circuit_edge_names(circuit_path)
    connectivity = circuit_connectivity(graph, edge_names)
    if (
        "circuit_only" in args.conditions
        and not connectivity["all_edges_on_source_to_sink_paths"]
    ):
        raise ValueError(
            "End-to-end circuit-only generation requires a complete "
            "source-to-logit circuit."
        )
    set_selected_edges(graph, edge_names)
    model.set_ungroup_grouped_query_attention(True)
    if not model.cfg.use_attn_result:
        raise RuntimeError(
            "End-to-end behavior evaluation requires TransformerLens' "
            "reference per-head attention results."
        )

    designed = []
    for question_index in args.question_indices:
        if not 0 <= question_index < len(questions):
            raise IndexError(f"Question index {question_index} is out of range.")
        instruction_index = question_index % len(instructions)
        instruction = instructions[instruction_index]
        positive_system = make_system_prompt(
            config["trait"]["positive_assistant_name"], instruction["pos"]
        )
        positive_prompt = model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": positive_system},
                {"role": "user", "content": questions[question_index]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        negative_prompt, alignment = _align_negative_prompt(
            model.tokenizer,
            question=questions[question_index],
            negative_assistant_name=config["trait"]["negative_assistant_name"],
            negative_instruction=instruction["neg"],
            positive_prompt=positive_prompt,
        )
        if alignment["positive_tokens"] > args.max_length:
            raise ValueError(
                f"Question {question_index} exceeds --max-length; refusing "
                "to truncate a behavioral prompt."
            )
        designed.append(
            {
                "question_id": f"{trait_slug}_{question_index}",
                "question_index": question_index,
                "instruction_index": instruction_index,
                "question": questions[question_index],
                "instruction": instruction["pos"],
                "system_prompt": positive_system,
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "negative_prompt_alignment": alignment,
            }
        )

    existing = read_jsonl(raw_path) if raw_path.exists() else []
    rows_by_id = {row["record_id"]: row for row in existing}
    if len(rows_by_id) != len(existing):
        raise ValueError("Duplicate output record IDs.")
    expected_ids = {
        (
            f"end-to-end-{args.run_name}-k{args.k}-{condition}-"
            f"q{source['question_index']:02d}"
        )
        for condition in args.conditions
        for source in designed
    }
    unexpected_ids = set(rows_by_id) - expected_ids
    if unexpected_ids:
        raise ValueError(
            f"Resume artifact contains unexpected records: {sorted(unexpected_ids)[:3]}"
        )

    def save_metadata(status: str) -> None:
        rows = read_jsonl(raw_path) if raw_path.exists() else []
        write_json(
            metadata_path,
            {
                "schema_version": 2,
                "status": status,
                "method": "end-to-end natural-prompt circuit intervention",
                "claim_scope": "held-out free-generation behavioral validation",
                "behavior_protocol_id": config["protocol_id"],
                "model": model_cfg,
                "model_architecture": observed_architecture,
                "trait": config["trait"],
                "eval_artifact": str(official_path.relative_to(ROOT)),
                "eval_artifact_sha256": sha256_file(official_path),
                "discovery_metadata": str(
                    discovery_metadata_path.relative_to(ROOT)
                ),
                "discovery_metadata_sha256": sha256_file(
                    discovery_metadata_path
                ),
                "circuit": str(circuit_path.relative_to(ROOT)),
                "circuit_sha256": sha256_file(circuit_path),
                "requested_k": args.k,
                "retained_edges": len(edge_names),
                "connectivity": connectivity,
                "question_indices": args.question_indices,
                "conditions": list(args.conditions),
                "counterfactual": "token-aligned negative-persona prompt",
                "decoding": {
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                    "backend": "full_prefix_forward_ungrouped_gqa",
                },
                "max_length": args.max_length,
                "responses": len(rows),
                "expected_responses": len(designed) * len(args.conditions),
                "outputs": (
                    [
                        {
                            "path": str(raw_path.relative_to(ROOT)),
                            "rows": len(rows),
                            "sha256": sha256_file(raw_path),
                        }
                    ]
                    if rows
                    else []
                ),
            },
        )

    save_metadata("running")
    for condition in args.conditions:
        for source in tqdm(designed, desc=condition):
            record_id = (
                f"end-to-end-{args.run_name}-k{args.k}-{condition}-"
                f"q{source['question_index']:02d}"
            )
            if record_id in rows_by_id:
                continue
            clean_tokens, clean_mask = tokenize_prompts(
                model, [source["prompt"]], max_length=args.max_length
            )
            corrupt_tokens, corrupt_mask = tokenize_prompts(
                model, [source["negative_prompt"]], max_length=args.max_length
            )
            generated = []
            eos_id = model.tokenizer.eos_token_id
            for _ in tqdm(
                range(args.max_new_tokens),
                desc=f"{condition}-q{source['question_index']:02d}",
                leave=False,
            ):
                if condition == "normal":
                    with torch.inference_mode():
                        logits = model(
                            clean_tokens, attention_mask=clean_mask
                        ).detach()
                elif condition == "negative_prompt":
                    with torch.inference_mode():
                        logits = model(
                            corrupt_tokens, attention_mask=corrupt_mask
                        ).detach()
                else:
                    logits = mixed_logits_from_prompt_pair(
                        model,
                        graph,
                        clean_tokens,
                        clean_mask,
                        corrupt_tokens,
                        corrupt_mask,
                        patch_selected=condition == "circuit_ablated",
                        cache_dtype=next(model.parameters()).dtype,
                    )
                next_token = int(logits[0, -1].argmax().item())
                del logits
                generated.append(next_token)
                appended = torch.tensor(
                    [[next_token]],
                    device=clean_tokens.device,
                    dtype=clean_tokens.dtype,
                )
                clean_tokens = torch.cat([clean_tokens, appended], dim=1)
                corrupt_tokens = torch.cat(
                    [corrupt_tokens, appended.clone()], dim=1
                )
                mask_token = torch.ones(
                    (1, 1),
                    device=clean_mask.device,
                    dtype=clean_mask.dtype,
                )
                clean_mask = torch.cat([clean_mask, mask_token], dim=1)
                corrupt_mask = torch.cat(
                    [corrupt_mask, mask_token.clone()], dim=1
                )
                if eos_id is not None and next_token == eos_id:
                    break
            row = {
                **{
                    key: value
                    for key, value in source.items()
                    if key != "negative_prompt"
                },
                "schema_version": 2,
                "record_id": record_id,
                "behavior_protocol_id": config["protocol_id"],
                "trait": trait_slug,
                "model": model_cfg["name"],
                "model_revision": model_cfg.get("revision"),
                "condition": condition,
                "requested_k": args.k,
                "retained_edges": len(edge_names),
                "answer": model.tokenizer.decode(
                    generated, skip_special_tokens=True
                ),
                "generated_token_count": len(generated),
                "generated_token_ids_sha256": token_ids_sha256(generated),
                "hit_max_new_tokens": len(generated) >= args.max_new_tokens,
                "decoding": "greedy",
            }
            write_jsonl(raw_path, [row], mode="a")
            rows_by_id[record_id] = row
            save_metadata("running")

    expected = len(designed) * len(args.conditions)
    if len(read_jsonl(raw_path)) != expected:
        raise RuntimeError("End-to-end behavior output is incomplete.")
    save_metadata("complete")
    print(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

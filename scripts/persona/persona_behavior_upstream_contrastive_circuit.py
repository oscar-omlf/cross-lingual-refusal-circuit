#!/usr/bin/env python3
"""Test a natural-contrast upstream persona circuit during free generation.

The clean run uses the positive-trait system prompt.  For circuit ablation,
selected upstream edges are replaced with source activations from the
token-aligned negative-trait prompt.  During autoregressive generation both
runs receive the same already-generated answer prefix, so the only
counterfactual difference is the system-prompt persona condition.
"""

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
    parser.add_argument("--k", type=int, default=3000)
    parser.add_argument("--circuit-path", type=Path, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--question-indices", type=int, nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--baseline-responses",
        type=Path,
        default=None,
        help="Optional completed normal responses to reuse; otherwise generate them.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help=(
            "Optional earlier responses.jsonl to seed a new output artifact. "
            "Only matching requested conditions/questions are copied."
        ),
    )
    parser.add_argument(
        "--initialize-from-circuit",
        type=Path,
        default=None,
        help=(
            "Circuit JSON that produced --initialize-from. Its edge names must "
            "exactly equal the requested circuit before responses may be reused."
        ),
    )
    parser.add_argument(
        "--extend-capped",
        action="store_true",
        help=(
            "Continue deterministic greedy responses that hit an earlier "
            "token cap, treating --max-new-tokens as the new total cap."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("normal", "negative_prompt", "circuit_only", "circuit_ablated"),
        default=("normal", "circuit_ablated"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("Question indices must be unique")
    if len(args.conditions) != len(set(args.conditions)):
        raise ValueError("Conditions must be unique")
    if args.extend_capped and args.initialize_from is None and not args.resume:
        raise ValueError("--extend-capped requires --initialize-from or --resume")
    if (args.initialize_from is None) != (args.initialize_from_circuit is None):
        raise ValueError(
            "--initialize-from and --initialize-from-circuit must be used together"
        )

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import mixed_forward_from_prompt_pair
    from refusal_circuit.graph import (
        circuit_connectivity,
        load_circuit_edge_names,
        make_projection_graph,
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
    discovery = (
        result_root / "circuit_discovery_upstream_contrastive" / args.run_name
    )
    metadata_path = discovery / "eapig_run_metadata.json"
    discovery_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if discovery_metadata.get("status") != "complete" or not discovery_metadata.get(
        "no_projection_ablation"
    ):
        raise ValueError("Discovery is not a completed non-projection upstream run")
    if discovery_metadata.get("model") != config["model"]:
        raise ValueError("Discovery and behavior model configurations differ.")
    circuit_path = (
        args.circuit_path
        if args.circuit_path is not None
        else discovery
        / "circuits_source_to_sink"
        / f"eapig_top_{args.k}.json"
    )
    circuit_path = circuit_path if circuit_path.is_absolute() else ROOT / circuit_path
    circuit_payload = json.loads(circuit_path.read_text(encoding="utf-8"))
    if int(circuit_payload["requested_k"]) != args.k:
        raise ValueError("Circuit payload k differs from requested k.")
    if circuit_payload.get("score_source") != discovery_metadata.get("score_file"):
        raise ValueError("Circuit was not derived from the frozen discovery scores.")

    output_name = (
        args.output_tag
        or f"k{args.k}_q{'-'.join(map(str, args.question_indices))}"
    )
    output = (
        intermediate
        / "natural_upstream_circuit_behavior"
        / args.run_name
        / output_name
    )
    raw_path = output / "responses.jsonl"
    output_metadata_path = output / "metadata.json"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"{output} already contains results; pass --resume")
    output.mkdir(parents=True, exist_ok=True)

    official = ROOT / config["paths"]["official_root"] / "eval.json"
    eval_data = json.loads(official.read_text(encoding="utf-8"))
    questions = eval_data["questions"]
    instructions = eval_data["instruction"]
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
    designed = []
    for question_index in args.question_indices:
        if not 0 <= question_index < len(questions):
            raise IndexError(f"Question index {question_index} is out of range")
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
        positive_ids = model.tokenizer.encode(
            positive_prompt, add_special_tokens=False
        )
        negative_system = make_system_prompt(
            config["trait"]["negative_assistant_name"], instruction["neg"]
        )
        negative_prompt = model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": negative_system},
                {"role": "user", "content": questions[question_index]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        negative_ids = model.tokenizer.encode(
            negative_prompt, add_special_tokens=False
        )
        if len(negative_ids) > len(positive_ids):
            raise ValueError(
                f"Question {question_index} cannot be token-aligned: the "
                "negative prompt is longer than the positive prompt."
            )
        aligned_negative_prompt, alignment = _align_negative_prompt(
            model.tokenizer,
            question=questions[question_index],
            negative_assistant_name=config["trait"]["negative_assistant_name"],
            negative_instruction=instruction["neg"],
            positive_prompt=positive_prompt,
        )
        if alignment["positive_tokens"] > args.max_length:
            raise ValueError(
                f"Question {question_index} exceeds --max-length; refusing "
                "to truncate an upstream behavioral prompt."
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
                "negative_prompt_alignment": alignment,
            }
        )

    # Use the trait-specific vector artifact so validation matches the chosen persona.
    direction_path = result_root / f"vector/{trait_slug}_direction.pt"
    direction = torch.load(
        direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    target_layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    graph = make_projection_graph(model, target_layer)
    edge_names = load_circuit_edge_names(circuit_path)
    connectivity = circuit_connectivity(graph, edge_names)
    if (
        "circuit_only" in args.conditions
        and not connectivity["all_edges_on_source_to_sink_paths"]
    ):
        raise ValueError(
            "Circuit-only generation requires a source-to-interface circuit. "
            "Use a rebuilt circuits_source_to_sink artifact."
        )
    set_selected_edges(graph, edge_names)
    model.set_ungroup_grouped_query_attention(True)
    if not model.cfg.use_attn_result:
        raise RuntimeError(
            "Upstream behavior evaluation requires TransformerLens' "
            "reference per-head attention results."
        )

    initialized_from = None
    initialized_from_circuit = None
    if args.initialize_from is not None and not raw_path.exists():
        source_path = (
            args.initialize_from
            if args.initialize_from.is_absolute()
            else ROOT / args.initialize_from
        )
        source_rows = read_jsonl(source_path)
        source_metadata_path = source_path.with_name("metadata.json")
        source_metadata = json.loads(
            source_metadata_path.read_text(encoding="utf-8")
        )
        if (
            source_metadata.get("status") != "complete"
            or source_metadata.get("decoding", {}).get("backend")
            != "full_prefix_forward_ungrouped_gqa"
            or any(
                not row.get("generated_token_ids_sha256")
                for row in source_rows
            )
        ):
            raise ValueError(
                "Initialization requires a completed same-backend artifact "
                "with exact generated-token hashes."
            )
        source_circuit_path = (
            args.initialize_from_circuit
            if args.initialize_from_circuit.is_absolute()
            else ROOT / args.initialize_from_circuit
        )
        source_edge_names = load_circuit_edge_names(source_circuit_path)
        if source_edge_names != edge_names:
            raise ValueError(
                "Initialization responses were produced by a different circuit."
            )
        wanted_questions = set(args.question_indices)
        copied_rows = [
            {
                **row,
                "record_id": (
                    f"natural-upstream-{args.run_name}-k{args.k}-"
                    f"{row['condition']}-q{int(row['question_index']):02d}"
                ),
                "reused_from": str(source_path.relative_to(ROOT)),
            }
            for row in source_rows
            if row.get("condition") in args.conditions
            and int(row.get("question_index", -1)) in wanted_questions
            and int(row.get("requested_k", -1)) == args.k
            and row.get("model") == model_cfg["name"]
            and row.get("model_revision") == model_cfg.get("revision")
            and row.get("decoding") == "greedy"
        ]
        expected_ids = {
            f"natural-upstream-{args.run_name}-k{args.k}-{condition}-q{q:02d}"
            for condition in args.conditions
            for q in args.question_indices
        }
        if {row["record_id"] for row in copied_rows} != expected_ids:
            raise ValueError(
                "Initialization artifact does not contain exactly the requested "
                "upstream records."
            )
        write_jsonl(raw_path, copied_rows)
        initialized_from = str(source_path.relative_to(ROOT))
        initialized_from_circuit = str(
            source_circuit_path.relative_to(ROOT)
        )

    existing_rows = read_jsonl(raw_path) if raw_path.exists() else []
    rows_by_id = {row["record_id"]: row for row in existing_rows}
    if len(rows_by_id) != len(existing_rows):
        raise ValueError("Duplicate behavioral record IDs")
    expected_resume = {
        "behavior_protocol_id": config["protocol_id"],
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "discovery_metadata_sha256": sha256_file(metadata_path),
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
        if not output_metadata_path.exists():
            raise ValueError("Cannot resume without the original metadata.json.")
        previous_metadata = json.loads(
            output_metadata_path.read_text(encoding="utf-8")
        )
        mismatched = [
            key
            for key, value in expected_resume.items()
            if previous_metadata.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                f"Resume arguments differ from the original run: {mismatched}"
            )
    expected_ids = {
        f"natural-upstream-{args.run_name}-k{args.k}-{condition}-q{question:02d}"
        for condition in args.conditions
        for question in args.question_indices
    }
    unexpected_ids = set(rows_by_id) - expected_ids
    if unexpected_ids:
        raise ValueError(
            f"Resume artifact contains unexpected records: {sorted(unexpected_ids)[:3]}"
        )

    # Reuse a normal row only when it came from the identical full-prefix
    # backend. Historical KV-cache baselines are intentionally regenerated.
    baseline_path = args.baseline_responses or (
        intermediate / "behavioral_pilot/k3000_n20/responses.jsonl"
    )
    baseline_path = baseline_path if baseline_path.is_absolute() else ROOT / baseline_path
    baseline_scored_path = baseline_path.with_name("scored.jsonl")
    baseline_metadata_path = baseline_path.with_name("metadata.json")
    baseline_metadata = (
        json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
        if baseline_metadata_path.exists()
        else {}
    )
    baseline_compatible = (
        baseline_metadata.get("status") == "complete"
        and baseline_metadata.get("model") == model_cfg
        and baseline_metadata.get("model_architecture")
        == observed_architecture
        and baseline_metadata.get("decoding", {}).get("backend")
        == "full_prefix_forward_ungrouped_gqa"
    )
    baseline = {
        int(row["question_index"]): row
        for row in (
            read_jsonl(baseline_path)
            if baseline_path.exists() and baseline_compatible
            else []
        )
        if row["condition"] == "normal"
        and row.get("generated_token_ids_sha256")
    }
    for source in designed if "normal" in args.conditions else []:
        record_id = (
            f"natural-upstream-{args.run_name}-k{args.k}-normal-"
            f"q{source['question_index']:02d}"
        )
        if record_id in rows_by_id:
            continue
        old = baseline.get(source["question_index"])
        if old is None:
            continue
        if (
            old.get("question") != source["question"]
            or old.get("prompt") != source["prompt"]
            or old.get("system_prompt") != source["system_prompt"]
            or old.get("decoding") != "greedy"
            or old.get("hit_max_new_tokens", False)
            or int(old.get("generated_token_count", args.max_new_tokens))
            >= args.max_new_tokens
        ):
            # Regenerate only this normal row; never reuse a mismatched or
            # token-capped baseline merely to save compute.
            continue
        copied = {
            **old,
            "record_id": record_id,
            "requested_k": args.k,
            "retained_edges": len(edge_names),
            "reused_from": str(baseline_path.relative_to(ROOT)),
        }
        write_jsonl(raw_path, [copied], mode="a")
        rows_by_id[record_id] = copied

    def save_metadata(status: str) -> None:
        rows = read_jsonl(raw_path) if raw_path.exists() else []
        write_json(
            output_metadata_path,
            {
                "schema_version": 1,
                "status": status,
                "method": "natural-contrast upstream circuit edge ablation",
                "no_projection_ablation": True,
                "claim_scope": (
                    "free-generation causal validation of the upstream circuit "
                    f"forming the {trait_slug}-vector coordinate"
                ),
                "behavior_protocol_id": config["protocol_id"],
                "model": model_cfg,
                "model_architecture": observed_architecture,
                "trait": config["trait"],
                "eval_artifact": str(official.relative_to(ROOT)),
                "eval_artifact_sha256": sha256_file(official),
                "discovery_metadata": str(metadata_path.relative_to(ROOT)),
                "discovery_metadata_sha256": sha256_file(metadata_path),
                "circuit": str(circuit_path.relative_to(ROOT)),
                "circuit_sha256": sha256_file(circuit_path),
                "direction": str(direction_path.relative_to(ROOT)),
                "direction_sha256": sha256_file(direction_path),
                "requested_k": args.k,
                "retained_edges": len(edge_names),
                "connectivity": connectivity,
                "question_indices": args.question_indices,
                "conditions": list(args.conditions),
                "counterfactual": (
                    "token-aligned negative-persona system prompt with the identical "
                    "already-generated answer prefix"
                ),
                "decoding": {
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                    "backend": "full_prefix_forward_ungrouped_gqa",
                },
                "generation_backend": "full_prefix_forward_ungrouped_gqa",
                "max_length": args.max_length,
                "checkpointing": "one JSONL append per completed response",
                "initialized_from": initialized_from,
                "initialized_from_circuit": initialized_from_circuit,
                "extend_capped": args.extend_capped,
                "reused_normal": (
                    str(baseline_path.relative_to(ROOT)) if baseline else None
                ),
                "reusable_normal_judgments": (
                    str(baseline_scored_path.relative_to(ROOT))
                    if baseline_scored_path.exists()
                    else None
                ),
                "responses": len(rows),
                "expected_responses": len(args.conditions) * len(designed),
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else None
                ),
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

    def plain_full_prefix_generation(tokens, mask) -> list[int]:
        generated: list[int] = []
        eos_id = model.tokenizer.eos_token_id
        for _ in range(args.max_new_tokens):
            with torch.inference_mode():
                logits = model(tokens, attention_mask=mask).detach()
            next_token = int(logits[0, -1].argmax().item())
            del logits
            generated.append(next_token)
            appended = torch.tensor(
                [[next_token]], device=tokens.device, dtype=tokens.dtype
            )
            tokens = torch.cat([tokens, appended], dim=1)
            mask = torch.cat(
                [
                    mask,
                    torch.ones(
                        (1, 1), device=mask.device, dtype=mask.dtype
                    ),
                ],
                dim=1,
            )
            if eos_id is not None and next_token == eos_id:
                break
        return generated

    save_metadata("running")
    for condition in args.conditions:
        for source in tqdm(designed, desc=condition):
            record_id = (
                f"natural-upstream-{args.run_name}-k{args.k}-{condition}-"
                f"q{source['question_index']:02d}"
            )
            previous = rows_by_id.get(record_id)
            should_extend = (
                previous is not None
                and args.extend_capped
                and bool(previous.get("hit_max_new_tokens"))
                and int(previous.get("generated_token_count", 0))
                < args.max_new_tokens
            )
            if previous is not None and not should_extend:
                continue
            clean_tokens, clean_mask = tokenize_prompts(
                model, [source["prompt"]], max_length=args.max_length
            )
            positive_ids = model.tokenizer.encode(
                source["prompt"], add_special_tokens=False
            )
            negative_prompt = _align_negative_prompt(
                model.tokenizer,
                question=source["question"],
                negative_assistant_name=config["trait"]["negative_assistant_name"],
                negative_instruction=instructions[source["instruction_index"]]["neg"],
                positive_prompt=source["prompt"],
            )[0]
            negative_ids = model.tokenizer.encode(
                negative_prompt, add_special_tokens=False
            )
            if len(negative_ids) > len(positive_ids):
                continue
            corrupt_prompt = _align_negative_prompt(
                model.tokenizer,
                question=source["question"],
                negative_assistant_name=config["trait"]["negative_assistant_name"],
                negative_instruction=instructions[source["instruction_index"]]["neg"],
                positive_prompt=source["prompt"],
            )[0]
            corrupt_tokens, corrupt_mask = tokenize_prompts(
                model, [corrupt_prompt], max_length=args.max_length
            )
            extension_method = None
            if condition == "normal":
                generated = plain_full_prefix_generation(
                    clean_tokens, clean_mask
                )
            elif condition == "negative_prompt":
                generated = plain_full_prefix_generation(
                    corrupt_tokens, corrupt_mask
                )
            else:
                generated = []
                if should_extend:
                    generated = [
                        int(token_id)
                        for token_id in previous.get(
                            "generated_token_ids", []
                        )
                    ]
                    if (
                        len(generated)
                        != int(previous["generated_token_count"])
                        or token_ids_sha256(generated)
                        != previous.get("generated_token_ids_sha256")
                    ):
                        generated = []
                        extension_method = "regenerated_unrecoverable_prefix"
                    else:
                        extension_method = "continued_exact_saved_token_prefix"
                        prefix = torch.tensor(
                            [generated],
                            device=clean_tokens.device,
                            dtype=clean_tokens.dtype,
                        )
                        clean_tokens = torch.cat([clean_tokens, prefix], dim=1)
                        corrupt_tokens = torch.cat(
                            [corrupt_tokens, prefix.clone()], dim=1
                        )
                        prefix_mask = torch.ones(
                            (1, len(generated)),
                            dtype=clean_mask.dtype,
                            device=clean_mask.device,
                        )
                        clean_mask = torch.cat([clean_mask, prefix_mask], dim=1)
                        corrupt_mask = torch.cat(
                            [corrupt_mask, prefix_mask.clone()], dim=1
                        )
                eos_id = model.tokenizer.eos_token_id
                for _ in tqdm(
                    range(args.max_new_tokens - len(generated)),
                    desc=f"{condition}-q{source['question_index']:02d}",
                    leave=False,
                ):
                    _, logits = mixed_forward_from_prompt_pair(
                        model,
                        graph,
                        clean_tokens,
                        clean_mask,
                        corrupt_tokens,
                        corrupt_mask,
                        target_hook=target_hook,
                        target_position=-1,
                        direction=direction,
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
                        dtype=clean_mask.dtype,
                        device=clean_mask.device,
                    )
                    clean_mask = torch.cat([clean_mask, mask_token], dim=1)
                    corrupt_mask = torch.cat(
                        [corrupt_mask, mask_token.clone()], dim=1
                    )
                    if eos_id is not None and next_token == eos_id:
                        break
            row = {
                **source,
                "schema_version": 1,
                "record_id": record_id,
                "behavior_protocol_id": config["protocol_id"],
                "trait": config["trait"]["name"],
                "model": model_cfg["name"],
                "model_revision": model_cfg.get("revision"),
                "condition": condition,
                "requested_k": args.k,
                "retained_edges": len(edge_names),
                "answer": model.tokenizer.decode(
                    generated, skip_special_tokens=True
                ),
                "generated_token_count": len(generated),
                "generated_token_ids": generated,
                "generated_token_ids_sha256": token_ids_sha256(generated),
                "hit_max_new_tokens": len(generated) >= args.max_new_tokens,
                "decoding": "greedy",
                "extension_method": extension_method,
            }
            if previous is None:
                write_jsonl(raw_path, [row], mode="a")
                rows_by_id[record_id] = row
            else:
                rows_by_id[record_id] = row
                write_jsonl(raw_path, rows_by_id.values())
            save_metadata("running")

    final_rows = read_jsonl(raw_path)
    expected = len(args.conditions) * len(designed)
    if len(final_rows) != expected or len(
        {row["record_id"] for row in final_rows}
    ) != expected:
        raise RuntimeError(f"Incomplete behavioral output: {len(final_rows)}/{expected}")
    save_metadata("complete")
    print(output_metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

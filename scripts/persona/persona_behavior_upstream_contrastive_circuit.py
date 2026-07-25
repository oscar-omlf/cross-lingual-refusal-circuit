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
    parser.add_argument("--question-indices", type=int, nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("Question indices must be unique")

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import mixed_forward_from_prompt_pair
    from refusal_circuit.graph import (
        load_circuit_edge_names,
        make_projection_graph,
        set_selected_edges,
    )
    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        make_system_prompt,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
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
    if int(discovery_metadata["requested_k"]) != args.k:
        raise ValueError("Requested k differs from discovery")
    circuit_path = discovery / "circuits_nested" / f"eapig_top_{args.k}.json"

    output = (
        intermediate
        / "natural_upstream_circuit_behavior"
        / args.run_name
        / f"k{args.k}_q{'-'.join(map(str, args.question_indices))}"
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
        aligned_negative_prompt, alignment = _align_negative_prompt(
            model.tokenizer,
            question=questions[question_index],
            negative_assistant_name=config["trait"]["negative_assistant_name"],
            negative_instruction=instruction["neg"],
            positive_prompt=positive_prompt,
        )
        designed.append(
            {
                "question_id": f"sycophantic_{question_index}",
                "question_index": question_index,
                "instruction_index": instruction_index,
                "question": questions[question_index],
                "instruction": instruction["pos"],
                "system_prompt": positive_system,
                "prompt": positive_prompt,
                "negative_prompt_alignment": alignment,
            }
        )

    direction_path = result_root / "vector/sycophantic_direction.pt"
    direction = torch.load(
        direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    target_layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    graph = make_projection_graph(model, target_layer)
    edge_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, edge_names)

    existing_rows = read_jsonl(raw_path) if raw_path.exists() else []
    rows_by_id = {row["record_id"]: row for row in existing_rows}
    if len(rows_by_id) != len(existing_rows):
        raise ValueError("Duplicate behavioral record IDs")

    # Normal greedy answers are independent of circuit choice and were already
    # generated and judged at the same 256-token cap on these exact prompts.
    baseline_path = intermediate / "behavioral_pilot/k3000_n20/responses.jsonl"
    baseline_scored_path = (
        intermediate / "behavioral_pilot/k3000_n20/scored.jsonl"
    )
    baseline = {
        int(row["question_index"]): row
        for row in read_jsonl(baseline_path)
        if row["condition"] == "normal"
    }
    for source in designed:
        record_id = (
            f"natural-upstream-{args.run_name}-k{args.k}-normal-"
            f"q{source['question_index']:02d}"
        )
        if record_id in rows_by_id:
            continue
        old = baseline.get(source["question_index"])
        if old is None:
            raise ValueError(
                f"Reusable baseline lacks question {source['question_index']}"
            )
        if (
            old.get("question") != source["question"]
            or old.get("prompt") != source["prompt"]
            or old.get("system_prompt") != source["system_prompt"]
            or old.get("decoding") != "greedy"
            or old.get("hit_max_new_tokens", False)
            or int(old.get("generated_token_count", args.max_new_tokens))
            >= args.max_new_tokens
        ):
            raise ValueError(
                f"Reusable normal baseline differs for q{source['question_index']}"
            )
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
                    "forming the sycophancy-vector coordinate"
                ),
                "behavior_protocol_id": config["protocol_id"],
                "model": model_cfg,
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
                "question_indices": args.question_indices,
                "conditions": ["normal", "circuit_ablated"],
                "counterfactual": (
                    "token-aligned helpful system prompt with the identical "
                    "already-generated answer prefix"
                ),
                "decoding": {
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                },
                "checkpointing": "one JSONL append per completed response",
                "reused_normal": str(baseline_path.relative_to(ROOT)),
                "reusable_normal_judgments": str(
                    baseline_scored_path.relative_to(ROOT)
                ),
                "responses": len(rows),
                "expected_responses": 2 * len(designed),
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
    for source in tqdm(designed, desc="natural-counterfactual circuit ablation"):
        record_id = (
            f"natural-upstream-{args.run_name}-k{args.k}-circuit_ablated-"
            f"q{source['question_index']:02d}"
        )
        if record_id in rows_by_id:
            continue
        clean_tokens, clean_mask = tokenize_prompts(
            model, [source["prompt"]], max_length=args.max_length
        )
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
        generated: list[int] = []
        eos_id = model.tokenizer.eos_token_id
        for _ in tqdm(
            range(args.max_new_tokens),
            desc=f"circuit_ablated-q{source['question_index']:02d}",
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
                patch_selected=True,
            )
            next_token = int(logits[0, -1].argmax().item())
            del logits
            generated.append(next_token)
            appended = torch.tensor(
                [[next_token]], device=clean_tokens.device, dtype=clean_tokens.dtype
            )
            clean_tokens = torch.cat([clean_tokens, appended], dim=1)
            corrupt_tokens = torch.cat([corrupt_tokens, appended.clone()], dim=1)
            mask_token = torch.ones(
                (1, 1), dtype=clean_mask.dtype, device=clean_mask.device
            )
            clean_mask = torch.cat([clean_mask, mask_token], dim=1)
            corrupt_mask = torch.cat([corrupt_mask, mask_token.clone()], dim=1)
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
            "condition": "circuit_ablated",
            "requested_k": args.k,
            "retained_edges": len(edge_names),
            "answer": model.tokenizer.decode(generated, skip_special_tokens=True),
            "generated_token_count": len(generated),
            "hit_max_new_tokens": len(generated) >= args.max_new_tokens,
            "decoding": "greedy",
        }
        write_jsonl(raw_path, [row], mode="a")
        rows_by_id[record_id] = row
        save_metadata("running")

    final_rows = read_jsonl(raw_path)
    expected = 2 * len(designed)
    if len(final_rows) != expected or len(
        {row["record_id"] for row in final_rows}
    ) != expected:
        raise RuntimeError(f"Incomplete behavioral output: {len(final_rows)}/{expected}")
    save_metadata("complete")
    print(output_metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

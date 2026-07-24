#!/usr/bin/env python3
"""Reproduce the exploratory behavior test of a post-response circuit.

The selected edges were ranked on activations from teacher-forced response
tokens, not on the pre-response monitoring state.  Results from this script
cannot by themselves establish a circuit that causes sycophancy generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--k", type=int, default=3000)
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("normal", "direction_ablated", "circuit_ablated"),
        default=None,
        help="Generate only these conditions; defaults to all three.",
    )
    parser.add_argument(
        "--reuse-normal-from",
        type=Path,
        default=None,
        help="Completed responses JSONL whose identical normal rows may be reused.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import _mixed_forward, _normal_forward
    from refusal_circuit.graph import load_circuit_edge_names, make_projection_graph, set_selected_edges
    from refusal_circuit.interventions import all_layer_ablation_hooks
    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import make_system_prompt, read_jsonl, sha256_file, write_jsonl

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    official = ROOT / config["paths"]["official_root"] / "eval.json"
    eval_data = json.loads(official.read_text(encoding="utf-8"))
    questions = eval_data["questions"][: args.num_questions]
    instructions = eval_data["instruction"]
    if len(questions) != args.num_questions:
        raise ValueError("Not enough official evaluation questions")

    result_root = ROOT / config["paths"]["results_root"]
    output_root = ROOT / config["paths"]["intermediate_root"] / f"behavioral_pilot/k{args.k}_n{args.num_questions}"
    raw_path = output_root / "responses.jsonl"
    metadata_path = output_root / "metadata.json"
    if (raw_path.exists() or metadata_path.exists()) and not (args.overwrite or args.resume):
        raise FileExistsError(f"{output_root} already contains pilot output")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite and raw_path.exists():
        raw_path.unlink()

    model = load_model(
        model_cfg["name"], device=args.device, dtype=args.dtype, revision=model_cfg.get("revision")
    )
    direction_path = result_root / "vector/sycophantic_direction.pt"
    direction = torch.load(direction_path, map_location=next(model.parameters()).device, weights_only=True)
    target_layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    circuit_path = result_root / f"circuit_discovery/circuits_nested/eapig_top_{args.k}.json"
    edge_names = load_circuit_edge_names(circuit_path)
    graph = make_projection_graph(model, target_layer)
    set_selected_edges(graph, edge_names)

    designed = []
    for question_index, question in enumerate(questions):
        instruction_index = question_index % len(instructions)
        instruction = instructions[instruction_index]["pos"]
        system_prompt = make_system_prompt(config["trait"]["positive_assistant_name"], instruction)
        prompt = model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        designed.append(
            {
                "question_id": f"sycophantic_{question_index}",
                "question_index": question_index,
                "instruction_index": instruction_index,
                "question": question,
                "instruction": instruction,
                "system_prompt": system_prompt,
                "prompt": prompt,
            }
        )

    conditions = tuple(args.conditions or ("normal", "direction_ablated", "circuit_ablated"))
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be unique")
    existing_rows = read_jsonl(raw_path) if raw_path.exists() else []
    # Reuse completed shorter pilots when expanding to more questions. This is
    # valid only for responses that ended naturally before the old cap; greedy
    # decoding is otherwise identical. Preserve explicit provenance per row.
    if not existing_rows and args.num_questions > 5:
        bootstrap_path = output_root.parent / f"k{args.k}_n5/responses.jsonl"
        if bootstrap_path.exists():
            reusable = []
            for row in read_jsonl(bootstrap_path):
                if (
                    int(row["question_index"]) < args.num_questions
                    and row["condition"] in conditions
                    and not row.get("hit_max_new_tokens", False)
                ):
                    copied = dict(row)
                    copied["reused_from"] = str(bootstrap_path.relative_to(ROOT))
                    reusable.append(copied)
            if reusable:
                write_jsonl(raw_path, reusable)
                existing_rows = reusable
    if not existing_rows and args.reuse_normal_from is not None and "normal" in conditions:
        reuse_path = (
            args.reuse_normal_from
            if args.reuse_normal_from.is_absolute()
            else ROOT / args.reuse_normal_from
        )
        reusable = []
        source_by_question = {
            int(row["question_index"]): row
            for row in read_jsonl(reuse_path)
            if row["condition"] == "normal"
        }
        for source in designed:
            old = source_by_question.get(int(source["question_index"]))
            if old is None:
                raise ValueError(
                    f"Reusable normal source lacks question {source['question_index']}"
                )
            if (
                old.get("question") != source["question"]
                or old.get("prompt") != source["prompt"]
                or old.get("system_prompt") != source["system_prompt"]
                or old.get("decoding") != "greedy"
                or old.get("hit_max_new_tokens", False)
            ):
                raise ValueError(
                    f"Reusable normal row differs for question {source['question_index']}"
                )
            copied = {
                **old,
                "record_id": f"k{args.k}-normal-q{source['question_index']:02d}",
                "requested_k": args.k,
                "retained_edges": len(edge_names),
                "reused_from": str(reuse_path.relative_to(ROOT)),
            }
            reusable.append(copied)
        write_jsonl(raw_path, reusable)
        existing_rows = reusable
    rows_by_id = {row["record_id"]: row for row in existing_rows}
    if len(rows_by_id) != len(existing_rows):
        raise ValueError("Duplicate behavioral pilot record IDs")

    def save_metadata(*, status: str) -> None:
        rows = read_jsonl(raw_path) if raw_path.exists() else []
        metadata = {
        "schema_version": 1,
        "status": status,
        "behavior_protocol_id": config["protocol_id"],
        "model": model_cfg,
        "trait": config["trait"],
        "eval_artifact": str(official.relative_to(ROOT)),
        "eval_artifact_sha256": sha256_file(official),
        "direction": str(direction_path.relative_to(ROOT)),
        "direction_sha256": sha256_file(direction_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "questions": len(designed),
        "conditions": list(conditions),
        "responses": len(rows),
        "expected_responses": len(designed) * len(conditions),
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "checkpointing": "one atomic JSONL append per completed question-condition response",
        "claim_scope": "exploratory test of a response-conditioned representation circuit",
        "methodology_warning": (
            "The circuit target is post-response and the direction-ablated condition zeros a "
            "projection; neither is equivalent to the paper's negative-vector steering intervention."
        ),
        "outputs": ([{"path": str(raw_path.relative_to(ROOT)), "rows": len(rows), "sha256": sha256_file(raw_path)}] if raw_path.exists() else []),
        }
        write_json(metadata_path, metadata)

    def append(source: dict, condition: str, token_ids: list[int]) -> None:
        record_id = f"k{args.k}-{condition}-q{source['question_index']:02d}"
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
            "answer": model.tokenizer.decode(token_ids, skip_special_tokens=True),
            "generated_token_count": len(token_ids),
            "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
            "decoding": "greedy",
        }
        write_jsonl(raw_path, [row], mode="a")
        rows_by_id[record_id] = row
        save_metadata(status="running")

    save_metadata(status="running")
    # Normal uses TransformerLens's exact greedy decoder with a KV cache. Run
    # one prompt at a time so no padding semantics enter the comparison.
    # EAP tracing expands Qwen's 4 grouped K/V heads to its 28 query heads,
    # whereas TransformerLens correctly allocates generation caches with the
    # native 4 K/V heads. Disable only that view-level expansion while using
    # the cache, then restore it before constructing the EAP circuit.
    if "normal" in conditions:
        model.set_ungroup_grouped_query_attention(False)
        try:
            for source in tqdm(designed, desc="normal questions"):
                record_id = f"k{args.k}-normal-q{source['question_index']:02d}"
                if record_id in rows_by_id:
                    continue
                prompt_tokens, _ = tokenize_prompts(model, [source["prompt"]], max_length=args.max_length)
                generated = model.generate(
                    prompt_tokens,
                    max_new_tokens=args.max_new_tokens,
                    stop_at_eos=True,
                    do_sample=False,
                    use_past_kv_cache=True,
                    return_type="tokens",
                    verbose=False,
                )
                append(source, "normal", generated[0, prompt_tokens.shape[1] :].tolist())
        finally:
            model.set_ungroup_grouped_query_attention(True)

    # The global baseline is the refusal-style projection-away intervention:
    # remove the unit sycophancy direction at every residual layer/token. It
    # supports native cached generation because no head-level EAP view is used.
    if "direction_ablated" in conditions:
        model.set_ungroup_grouped_query_attention(False)
        try:
            for source in tqdm(designed, desc="direction-ablated questions"):
                record_id = f"k{args.k}-direction_ablated-q{source['question_index']:02d}"
                if record_id in rows_by_id:
                    continue
                prompt_tokens, _ = tokenize_prompts(model, [source["prompt"]], max_length=args.max_length)
                with model.hooks(fwd_hooks=all_layer_ablation_hooks(model, direction)):
                    generated = model.generate(
                        prompt_tokens,
                        max_new_tokens=args.max_new_tokens,
                        stop_at_eos=True,
                        do_sample=False,
                        use_past_kv_cache=True,
                        return_type="tokens",
                        verbose=False,
                    )
                append(
                    source,
                    "direction_ablated",
                    generated[0, prompt_tokens.shape[1] :].tolist(),
                )
        finally:
            model.set_ungroup_grouped_query_attention(True)

    # Exact circuit ablation requires rebuilding counterfactual caches as the
    # sequence grows. Batch size one bounds memory; every answer is checkpointed.
    if "circuit_ablated" in conditions:
        for source in tqdm(designed, desc="circuit-ablated questions"):
            record_id = f"k{args.k}-circuit_ablated-q{source['question_index']:02d}"
            if record_id in rows_by_id:
                continue
            tokens, attention_mask = tokenize_prompts(model, [source["prompt"]], max_length=args.max_length)
            token_ids: list[int] = []
            eos_id = model.tokenizer.eos_token_id
            for _ in tqdm(range(args.max_new_tokens), desc=record_id, leave=False):
                _, logits = _mixed_forward(
                    model, graph, tokens, attention_mask, target_hook=target_hook,
                    target_position=-1, direction=direction, patch_selected=True,
                )
                next_token = int(logits[0, -1].argmax().item())
                del logits
                token_ids.append(next_token)
                tokens = torch.cat([tokens, torch.tensor([[next_token]], device=tokens.device)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=tokens.device)], dim=1
                )
                if eos_id is not None and next_token == eos_id:
                    break
            append(source, "circuit_ablated", token_ids)

    final_rows = read_jsonl(raw_path)
    if len(final_rows) != len(designed) * len(conditions):
        raise RuntimeError(f"Incomplete behavioral pilot: {len(final_rows)} responses")
    save_metadata(status="complete")
    print(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

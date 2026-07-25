#!/usr/bin/env python3
"""Test an upstream prompt-time persona circuit during free generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


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
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("Question indices must be unique")

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import _mixed_forward
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
    discovery = result_root / "circuit_discovery_prompt_last" / args.run_name
    metadata_path = discovery / "eapig_run_metadata.json"
    discovery_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not discovery_metadata.get("refusal_equivalent_protocol"):
        raise ValueError("Discovery run is not refusal-equivalent")
    circuit_path = discovery / "circuits_nested" / f"eapig_top_{args.k}.json"

    output = (
        intermediate
        / "upstream_prompt_circuit_behavior"
        / args.run_name
        / f"k{args.k}_q{'-'.join(map(str, args.question_indices))}"
    )
    raw_path = output / "responses.jsonl"
    output_metadata_path = output / "metadata.json"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results")
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
        instruction = instructions[instruction_index]["pos"]
        system_prompt = make_system_prompt(
            config["trait"]["positive_assistant_name"], instruction
        )
        prompt = model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": questions[question_index]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        designed.append(
            {
                "question_id": f"sycophantic_{question_index}",
                "question_index": question_index,
                "instruction_index": instruction_index,
                "question": questions[question_index],
                "instruction": instruction,
                "system_prompt": system_prompt,
                "prompt": prompt,
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

    # Normal and global-direction-ablated generations are independent of the
    # selected circuit. Reuse their completed 256-token-cap runs exactly.
    baseline_path = intermediate / "behavioral_pilot/k3000_n20/responses.jsonl"
    baseline_rows = read_jsonl(baseline_path)
    baseline = {
        (int(row["question_index"]), row["condition"]): row
        for row in baseline_rows
        if row["condition"] in {"normal", "direction_ablated"}
    }
    for source in designed:
        for condition in ("normal", "direction_ablated"):
            old = baseline.get((source["question_index"], condition))
            if old is None:
                raise ValueError(
                    f"Reusable baseline lacks q{source['question_index']} {condition}"
                )
            if (
                old.get("question") != source["question"]
                or old.get("prompt") != source["prompt"]
                or old.get("system_prompt") != source["system_prompt"]
                or old.get("decoding") != "greedy"
                or old.get("hit_max_new_tokens", False)
            ):
                raise ValueError(
                    f"Reusable baseline differs for q{source['question_index']} {condition}"
                )
            copied = {
                **old,
                "record_id": (
                    f"upstream-prompt-{args.run_name}-k{args.k}-{condition}-"
                    f"q{source['question_index']:02d}"
                ),
                "requested_k": args.k,
                "retained_edges": len(edge_names),
                "reused_from": str(baseline_path.relative_to(ROOT)),
            }
            write_jsonl(raw_path, [copied], mode="a")

    for source in tqdm(designed, desc="upstream circuit-ablated"):
        tokens, attention_mask = tokenize_prompts(
            model, [source["prompt"]], max_length=args.max_length
        )
        token_ids: list[int] = []
        eos_id = model.tokenizer.eos_token_id
        for _ in tqdm(
            range(args.max_new_tokens),
            desc=f"circuit_ablated-q{source['question_index']:02d}",
            leave=False,
        ):
            _, logits = _mixed_forward(
                model,
                graph,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
                patch_selected=True,
            )
            next_token = int(logits[0, -1].argmax().item())
            token_ids.append(next_token)
            tokens = torch.cat(
                [
                    tokens,
                    torch.tensor(
                        [[next_token]], device=tokens.device, dtype=tokens.dtype
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            if eos_id is not None and next_token == eos_id:
                break
        row = {
            **source,
            "schema_version": 1,
            "record_id": (
                f"upstream-prompt-{args.run_name}-k{args.k}-circuit_ablated-"
                f"q{source['question_index']:02d}"
            ),
            "behavior_protocol_id": config["protocol_id"],
            "trait": config["trait"]["name"],
            "model": model_cfg["name"],
            "model_revision": model_cfg.get("revision"),
            "condition": "circuit_ablated",
            "requested_k": args.k,
            "retained_edges": len(edge_names),
            "answer": model.tokenizer.decode(token_ids, skip_special_tokens=True),
            "generated_token_count": len(token_ids),
            "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
            "decoding": "greedy",
        }
        write_jsonl(raw_path, [row], mode="a")

    rows = read_jsonl(raw_path)
    expected = 3 * len(designed)
    if len(rows) != expected or len({row["record_id"] for row in rows}) != expected:
        raise RuntimeError(f"Incomplete behavioral output: {len(rows)}/{expected}")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "refusal_equivalent_protocol": True,
        "claim_scope": "free-generation causal validation of an upstream prompt-time circuit",
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
        "conditions": ["normal", "direction_ablated", "circuit_ablated"],
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "reused_baseline": str(baseline_path.relative_to(ROOT)),
        "outputs": [
            {
                "path": str(raw_path.relative_to(ROOT)),
                "rows": len(rows),
                "sha256": sha256_file(raw_path),
            }
        ],
    }
    write_json(output_metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

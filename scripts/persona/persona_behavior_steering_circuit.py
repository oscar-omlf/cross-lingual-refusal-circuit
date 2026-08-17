#!/usr/bin/env python3
"""Generate behavior under a steering-path circuit intervention.

Circuit ablation patches selected downstream edges with values from the
behaviorally successful negative-steering counterfactual at every decoding
step.  This is the decisive causal test for a circuit discovered with
``persona.run-steering-eapig``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=int, required=True)
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("normal", "negative_steered", "circuit_only", "circuit_ablated"),
        default=("normal", "negative_steered", "circuit_ablated"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("question indices must be unique")
    if len(args.conditions) != len(set(args.conditions)):
        raise ValueError("conditions must be unique")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import mixed_logits_with_intervention
    from refusal_circuit.graph import (
        circuit_connectivity,
        load_circuit_edge_names,
        make_steering_to_logits_graph,
        set_selected_edges,
    )
    from refusal_circuit.interventions import activation_addition_hook
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
    discovery = result_root / "circuit_discovery_steering_path" / args.run_name
    discovery_meta_path = discovery / "eapig_run_metadata.json"
    discovery_meta = json.loads(discovery_meta_path.read_text(encoding="utf-8"))
    if discovery_meta.get("status") != "complete":
        raise ValueError("Discovery run is not complete.")
    if discovery_meta.get("model") != config["model"]:
        raise ValueError("Discovery and behavior model configurations differ.")
    corrupt_coefficient = float(
        discovery_meta["steering"]["corrupt_coefficient"]
    )
    circuit_path = (
        args.circuit_path
        if args.circuit_path is not None
        else discovery
        / "circuits_source_to_sink"
        / f"steering_path_eapig_top_{args.k}.json"
    )
    circuit_path = circuit_path if circuit_path.is_absolute() else ROOT / circuit_path
    circuit_payload = json.loads(circuit_path.read_text(encoding="utf-8"))
    if int(circuit_payload["requested_k"]) != args.k:
        raise ValueError("Circuit payload k differs from requested k.")
    if circuit_payload.get("score_source") != discovery_meta.get("score_file"):
        raise ValueError("Circuit was not derived from the frozen discovery scores.")
    output_name = (
        args.output_tag
        or f"k{args.k}_q{'-'.join(map(str, args.question_indices))}"
    )
    output = (
        ROOT
        / config["paths"]["intermediate_root"]
        / "steering_circuit_behavior"
        / args.run_name
        / output_name
    )
    raw_path = output / "responses.jsonl"
    metadata_path = output / "metadata.json"
    if output.exists() and any(output.iterdir()):
        if args.overwrite:
            raw_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        elif not args.resume:
            raise FileExistsError(f"{output} already contains behavior output")
    output.mkdir(parents=True, exist_ok=True)

    official = ROOT / config["paths"]["official_root"] / "eval.json"
    eval_data = json.loads(official.read_text(encoding="utf-8"))
    questions = eval_data["questions"]
    instructions = eval_data["instruction"]
    designed = []
    for question_index in args.question_indices:
        if not 0 <= question_index < len(questions):
            raise IndexError(f"Question index {question_index} is out of range.")
        instruction_index = question_index % len(instructions)
        instruction = instructions[instruction_index]["pos"]
        system_prompt = make_system_prompt(
            config["trait"]["positive_assistant_name"], instruction
        )
        designed.append(
            {
                "question_id": f"{trait_slug}_{question_index}",
                "question_index": question_index,
                "instruction_index": instruction_index,
                "question": questions[question_index],
                "instruction": instruction,
                "system_prompt": system_prompt,
            }
        )

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
    for row in designed:
        row["prompt"] = model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_length = len(
            model.tokenizer.encode(
                row["prompt"], add_special_tokens=False
            )
        )
        if prompt_length > args.max_length:
            raise ValueError(
                f"Question {row['question_index']} exceeds --max-length; "
                "refusing to truncate a downstream behavioral prompt."
            )
    vector_path = result_root / f"vector/{trait_slug}_response_avg_diff.pt"
    vector_sha256 = sha256_file(vector_path)
    if (
        discovery_meta.get("steering", {}).get("vector")
        != str(vector_path.relative_to(ROOT))
        or discovery_meta.get("steering", {}).get("vector_sha256")
        != vector_sha256
    ):
        raise ValueError(
            "Behavior evaluation response vector differs from discovery."
        )
    vector_stack = torch.load(
        vector_path, map_location=next(model.parameters()).device, weights_only=True
    )
    layer = int(config["vector"]["paper_layer"])
    raw_direction = vector_stack[layer] if vector_stack.ndim == 2 else vector_stack
    target_hook = config["vector"]["transformer_lens_target_hook"]
    graph = make_steering_to_logits_graph(model, layer)
    edge_names = load_circuit_edge_names(circuit_path)
    connectivity = circuit_connectivity(graph, edge_names)
    if (
        "circuit_only" in args.conditions
        and not connectivity["all_edges_on_source_to_sink_paths"]
    ):
        raise ValueError(
            "Circuit-only generation requires an interface-to-logit circuit. "
            "Use a rebuilt circuits_source_to_sink artifact."
        )
    set_selected_edges(graph, edge_names)
    model.set_ungroup_grouped_query_attention(True)
    if not model.cfg.use_attn_result:
        raise RuntimeError(
            "Downstream behavior evaluation requires TransformerLens' "
            "reference per-head attention results."
        )
    existing_rows = read_jsonl(raw_path) if raw_path.exists() else []
    rows_by_id = {row["record_id"]: row for row in existing_rows}
    if len(rows_by_id) != len(existing_rows):
        raise ValueError("Duplicate behavioral record IDs.")
    expected_resume = {
        "behavior_protocol_id": config["protocol_id"],
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "discovery_metadata_sha256": sha256_file(discovery_meta_path),
        "circuit_sha256": sha256_file(circuit_path),
        "response_vector_sha256": vector_sha256,
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
        previous_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
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
        f"steering-circuit-{args.run_name}-k{args.k}-{condition}-q{question:02d}"
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
            f"steering-circuit-{args.run_name}-k{args.k}-normal-"
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
        payload = {
            "schema_version": 1,
            "status": status,
            "canonical_protocol": False,
            "behavior_protocol_id": config["protocol_id"],
            "claim_scope": "free-generation causal circuit validation",
            "model": model_cfg,
            "model_architecture": observed_architecture,
            "trait": config["trait"],
            "eval_artifact": str(official.relative_to(ROOT)),
            "eval_artifact_sha256": sha256_file(official),
            "discovery_metadata": str(discovery_meta_path.relative_to(ROOT)),
            "discovery_metadata_sha256": sha256_file(discovery_meta_path),
            "circuit": str(circuit_path.relative_to(ROOT)),
            "circuit_sha256": sha256_file(circuit_path),
            "response_vector": str(vector_path.relative_to(ROOT)),
            "response_vector_sha256": vector_sha256,
            "requested_k": args.k,
            "retained_edges": len(edge_names),
            "connectivity": connectivity,
            "question_indices": args.question_indices,
            "conditions": list(args.conditions),
            "corrupt_coefficient": corrupt_coefficient,
            "decoding": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "backend": "full_prefix_forward_ungrouped_gqa",
            },
            "generation_backend": "full_prefix_forward_ungrouped_gqa",
            "max_length": args.max_length,
            "checkpointing": "one JSONL append per completed response",
            "reused_normal": (
                str(baseline_path.relative_to(ROOT)) if baseline else None
            ),
            "reusable_normal_judgments": str(
                baseline_scored_path.relative_to(ROOT)
            ) if baseline_scored_path.exists() else None,
            "responses": len(rows),
            "expected_responses": len(designed) * len(args.conditions),
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
        }
        write_json(metadata_path, payload)

    def append(source: dict, condition: str, token_ids: list[int]) -> None:
        record_id = (
            f"steering-circuit-{args.run_name}-k{args.k}-{condition}-"
            f"q{source['question_index']:02d}"
        )
        row = {
            **source,
            "schema_version": 1,
            "record_id": record_id,
            "behavior_protocol_id": config["protocol_id"],
            "trait": trait_slug,
            "model": model_cfg["name"],
            "model_revision": model_cfg.get("revision"),
            "condition": condition,
            "requested_k": args.k,
            "retained_edges": len(edge_names),
            "answer": model.tokenizer.decode(token_ids, skip_special_tokens=True),
            "generated_token_count": len(token_ids),
            "generated_token_ids": token_ids,
            "generated_token_ids_sha256": token_ids_sha256(token_ids),
            "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
            "decoding": "greedy",
            "corrupt_coefficient": corrupt_coefficient,
        }
        write_jsonl(raw_path, [row], mode="a")
        rows_by_id[record_id] = row
        save_metadata("running")

    def plain_full_prefix_generation(
        tokens,
        *,
        fwd_hooks=(),
    ) -> list[int]:
        generated: list[int] = []
        eos_id = model.tokenizer.eos_token_id
        for _ in range(args.max_new_tokens):
            with torch.inference_mode(), model.hooks(
                fwd_hooks=list(fwd_hooks)
            ):
                logits = model(tokens).detach()
            next_token = int(logits[0, -1].argmax().item())
            del logits
            generated.append(next_token)
            tokens = torch.cat(
                [
                    tokens,
                    torch.tensor(
                        [[next_token]],
                        device=tokens.device,
                        dtype=tokens.dtype,
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
                f"steering-circuit-{args.run_name}-k{args.k}-{condition}-"
                f"q{source['question_index']:02d}"
            )
            if record_id in rows_by_id:
                continue
            prompt_tokens, _ = tokenize_prompts(
                model, [source["prompt"]], max_length=args.max_length
            )
            if condition in {"normal", "negative_steered"}:
                hooks = []
                if condition == "negative_steered":
                    hooks = [
                        (
                            target_hook,
                            activation_addition_hook(
                                raw_direction,
                                coefficient=corrupt_coefficient,
                                position=-1,
                            ),
                        )
                    ]
                append(
                    source,
                    condition,
                    plain_full_prefix_generation(
                        prompt_tokens, fwd_hooks=hooks
                    ),
                )
                continue

            # EAP's graph has 28 explicit query-head nodes. Keep TransformerLens
            # grouped-query expansion enabled for circuit cache and patch hooks.
            model.set_ungroup_grouped_query_attention(True)
            tokens = prompt_tokens
            token_ids: list[int] = []
            eos_id = model.tokenizer.eos_token_id
            for _ in tqdm(
                range(args.max_new_tokens),
                desc=f"{condition}-q{source['question_index']:02d}",
                leave=False,
            ):
                corrupt_hook = (
                    target_hook,
                    activation_addition_hook(
                        raw_direction,
                        coefficient=corrupt_coefficient,
                        position=-1,
                    ),
                )

                def cache_transform(name: str, activations: torch.Tensor):
                    if name != target_hook:
                        return activations
                    changed = activations.clone()
                    changed[:, -1, :] += (
                        corrupt_coefficient
                        * raw_direction.to(
                            device=changed.device, dtype=changed.dtype
                        )
                    )
                    return changed

                logits = mixed_logits_with_intervention(
                    model,
                    graph,
                    tokens,
                    None,
                    corrupt_fwd_hooks=[corrupt_hook],
                    corrupt_cache_transform=cache_transform,
                    patch_selected=condition == "circuit_ablated",
                    cache_dtype=next(model.parameters()).dtype,
                )
                next_token = int(logits[0, -1].argmax().item())
                del logits
                token_ids.append(next_token)
                tokens = torch.cat(
                    [
                        tokens,
                        torch.tensor(
                            [[next_token]],
                            device=tokens.device,
                            dtype=tokens.dtype,
                        ),
                    ],
                    dim=1,
                )
                if eos_id is not None and next_token == eos_id:
                    break
            append(source, condition, token_ids)

    rows = read_jsonl(raw_path)
    expected = len(designed) * len(args.conditions)
    if len(rows) != expected or len({row["record_id"] for row in rows}) != expected:
        raise RuntimeError(f"Incomplete behavioral output: {len(rows)}/{expected}")
    save_metadata("complete")
    print(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate with strictly vector-mediated sufficient circuit composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_system_prompt_pair


CONDITIONS = (
    "normal_positive",
    "negative_baseline",
    "oracle_full_state_interface",
    "oracle_vector_interface",
    "upstream_circuit_full_downstream",
    "oracle_vector_downstream_circuit",
    "composed_circuit",
    "composed_v_blocked",
    "composed_orthogonal",
)


def _capture_state(model, tokens, attention_mask, target_hook: str):
    import torch

    captured = {}

    def hook(activations: torch.Tensor, hook):
        del hook
        captured["state"] = activations.detach().clone()
        return activations

    with torch.inference_mode(), model.hooks(
        fwd_hooks=[(target_hook, hook)]
    ):
        logits = model(tokens, attention_mask=attention_mask)
    return captured["state"], logits.detach()


def _full_downstream_logits(
    model,
    tokens,
    attention_mask,
    target_hook: str,
    interface_state,
):
    import torch

    from refusal_circuit.edge_pruning import fixed_state_hook

    with torch.inference_mode(), model.hooks(
        fwd_hooks=[(target_hook, fixed_state_hook(interface_state))]
    ):
        return model(tokens, attention_mask=attention_mask).detach()


def _downstream_circuit_logits(
    model,
    graph,
    tokens,
    attention_mask,
    *,
    target_hook: str,
    clean_interface,
    corrupt_interface,
    cache_dtype,
):
    import torch

    from refusal_circuit.circuit import mixed_logits_with_intervention
    from refusal_circuit.edge_pruning import fixed_state_hook

    corrupt_hook = (
        target_hook,
        fixed_state_hook(corrupt_interface),
    )
    clean_hook = (
        target_hook,
        fixed_state_hook(clean_interface),
    )

    def corrupt_transform(name: str, activations: torch.Tensor):
        return (
            corrupt_interface.to(
                device=activations.device, dtype=activations.dtype
            )
            if name == target_hook
            else activations
        )

    def clean_transform(name: str, activations: torch.Tensor):
        return (
            clean_interface.to(
                device=activations.device, dtype=activations.dtype
            )
            if name == target_hook
            else activations
        )

    return mixed_logits_with_intervention(
        model,
        graph,
        tokens,
        attention_mask,
        corrupt_fwd_hooks=[corrupt_hook],
        corrupt_cache_transform=corrupt_transform,
        clean_fwd_hooks=[clean_hook],
        clean_cache_transform=clean_transform,
        patch_selected=False,
        cache_dtype=cache_dtype,
    )


def _require_internal_evaluation(
    path: Path,
    *,
    stage: str,
    run_name: str,
    k: int,
    circuit_path: Path,
    minimum: float,
) -> dict:
    from refusal_circuit.persona import sha256_file

    aggregate = json.loads(path.read_text(encoding="utf-8"))
    if (
        aggregate.get("status") != "complete"
        or aggregate.get("stage") != stage
        or aggregate.get("run_name") != run_name
        or int(aggregate["requested_k"]) != k
        or aggregate.get("circuit") != str(circuit_path.relative_to(ROOT))
        or aggregate.get("circuit_sha256") != sha256_file(circuit_path)
    ):
        raise ValueError(f"{stage} internal evaluation provenance failed.")
    metrics = aggregate["metrics"]
    if stage == "upstream":
        values = {
            "per_position_reconstruction_fraction": float(
                metrics["per_position_reconstruction_fraction"]
            )
        }
    else:
        values = {
            "mean_distribution_kl_recovery": float(
                metrics["mean_distribution_kl_recovery"]
            ),
            "downstream_circuit_margin_faithfulness": float(
                metrics["downstream_circuit_margin_faithfulness"]
            ),
        }
    failures = {
        name: value for name, value in values.items() if value < minimum
    }
    if failures:
        raise ValueError(
            f"{stage} circuit failed the internal sufficiency gate "
            f"{minimum}: {failures}"
        )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--upstream-run", default=None)
    parser.add_argument("--downstream-run", default=None)
    parser.add_argument("--upstream-k", type=int, default=None)
    parser.add_argument("--downstream-k", type=int, default=None)
    parser.add_argument(
        "--upstream-circuit-path", type=Path, default=None
    )
    parser.add_argument(
        "--downstream-circuit-path", type=Path, default=None
    )
    parser.add_argument("--upstream-evaluation", type=Path, default=None)
    parser.add_argument("--downstream-evaluation", type=Path, default=None)
    parser.add_argument(
        "--minimum-internal-faithfulness", type=float, default=0.85
    )
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--question-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--orthogonal-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("Question indices must be unique.")
    if not 0 < args.minimum_internal_faithfulness <= 1:
        raise ValueError("--minimum-internal-faithfulness must be in (0, 1].")
    upstream_conditions = {
        "upstream_circuit_full_downstream",
        "composed_circuit",
        "composed_v_blocked",
        "composed_orthogonal",
    }
    downstream_conditions = {
        "oracle_vector_downstream_circuit",
        "composed_circuit",
        "composed_v_blocked",
        "composed_orthogonal",
    }
    needs_upstream = args.condition in upstream_conditions
    needs_downstream = args.condition in downstream_conditions
    required = {}
    if needs_upstream:
        required.update(
            {
                "--upstream-run": args.upstream_run,
                "--upstream-k": args.upstream_k,
                "--upstream-evaluation": args.upstream_evaluation,
            }
        )
    if needs_downstream:
        required.update(
            {
                "--downstream-run": args.downstream_run,
                "--downstream-k": args.downstream_k,
                "--downstream-evaluation": args.downstream_evaluation,
            }
        )
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"Condition {args.condition!r} requires: {', '.join(missing)}"
        )

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import mixed_state_from_prompt_pair
    from refusal_circuit.edge_pruning import vector_only_rescue_state
    from refusal_circuit.graph import (
        circuit_connectivity,
        load_circuit_edge_names,
        make_projection_graph,
        make_steering_to_logits_graph,
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
    def resolve_path(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    upstream_circuit_path = None
    upstream_evaluation_path = None
    upstream_evaluation = None
    if needs_upstream:
        upstream_root = (
            result_root
            / "circuit_discovery_edge_pruning"
            / "upstream"
            / args.upstream_run
        )
        upstream_circuit_path = (
            resolve_path(args.upstream_circuit_path)
            if args.upstream_circuit_path is not None
            else (
                upstream_root
                / "circuits_source_to_sink"
                / f"edge_pruned_top_{args.upstream_k}.json"
            )
        )
        upstream_evaluation_path = resolve_path(args.upstream_evaluation)
        upstream_evaluation = _require_internal_evaluation(
            upstream_evaluation_path,
            stage="upstream",
            run_name=args.upstream_run,
            k=args.upstream_k,
            circuit_path=upstream_circuit_path,
            minimum=args.minimum_internal_faithfulness,
        )

    downstream_circuit_path = None
    downstream_evaluation_path = None
    downstream_evaluation = None
    if needs_downstream:
        downstream_root = (
            result_root
            / "circuit_discovery_edge_pruning"
            / "downstream"
            / args.downstream_run
        )
        downstream_circuit_path = (
            resolve_path(args.downstream_circuit_path)
            if args.downstream_circuit_path is not None
            else (
                downstream_root
                / "circuits_source_to_sink"
                / f"edge_pruned_top_{args.downstream_k}.json"
            )
        )
        downstream_evaluation_path = resolve_path(args.downstream_evaluation)
        downstream_evaluation = _require_internal_evaluation(
            downstream_evaluation_path,
            stage="downstream",
            run_name=args.downstream_run,
            k=args.downstream_k,
            circuit_path=downstream_circuit_path,
            minimum=args.minimum_internal_faithfulness,
        )

    if (
        needs_upstream
        and needs_downstream
        and upstream_evaluation["evaluation_manifest_sha256"]
        != downstream_evaluation["evaluation_manifest_sha256"]
    ):
        raise ValueError(
            "Upstream and downstream k values were not selected on the same "
            "frozen evaluation manifest."
        )

    condition_tag = (
        f"{args.condition}_seed{args.orthogonal_seed}"
        if args.condition == "composed_orthogonal"
        else args.condition
    )
    output_leaf = args.output_tag or (
        f"{condition_tag}_q{'-'.join(map(str, args.question_indices))}"
    )
    output = (
        intermediate
        / "sufficient_circuit_behavior"
        / f"{args.upstream_run or 'none'}__{args.downstream_run or 'none'}"
        / f"ku{args.upstream_k or 'none'}_kd{args.downstream_k or 'none'}"
        / output_leaf
    )
    raw_path = output / "responses.jsonl"
    metadata_path = output / "metadata.json"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"{output} already contains results; pass --resume.")
    output.mkdir(parents=True, exist_ok=True)

    official_path = ROOT / config["paths"]["official_root"] / "eval.json"
    eval_data = json.loads(official_path.read_text(encoding="utf-8"))
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
    model.set_ungroup_grouped_query_attention(True)
    layer = int(config["vector"]["paper_layer"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    direction_path = result_root / f"vector/{trait_slug}_direction.pt"
    direction = torch.load(
        direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    ).float()
    direction = direction / direction.norm().clamp_min(1e-8)
    generator = torch.Generator(device="cpu").manual_seed(args.orthogonal_seed)
    orthogonal = torch.randn(
        direction.shape, generator=generator, dtype=torch.float32
    ).to(direction.device)
    orthogonal = orthogonal - orthogonal.dot(direction) * direction
    orthogonal = orthogonal / orthogonal.norm().clamp_min(1e-8)

    upstream_graph = None
    upstream_connectivity = None
    if needs_upstream:
        upstream_graph = make_projection_graph(model, layer)
        upstream_edges = load_circuit_edge_names(upstream_circuit_path)
        upstream_connectivity = circuit_connectivity(
            upstream_graph, upstream_edges
        )
        if not upstream_connectivity["all_edges_on_source_to_sink_paths"]:
            raise ValueError(
                "Upstream sufficiency requires a complete-path circuit."
            )
        set_selected_edges(upstream_graph, upstream_edges)

    downstream_graph = None
    downstream_connectivity = None
    if needs_downstream:
        downstream_graph = make_steering_to_logits_graph(model, layer)
        downstream_edges = load_circuit_edge_names(downstream_circuit_path)
        downstream_connectivity = circuit_connectivity(
            downstream_graph, downstream_edges
        )
        if not downstream_connectivity["all_edges_on_source_to_sink_paths"]:
            raise ValueError(
                "Downstream sufficiency requires a complete-path circuit."
            )
        set_selected_edges(downstream_graph, downstream_edges)
    compact_dtype = next(model.parameters()).dtype

    designed = []
    for question_index in args.question_indices:
        if not 0 <= question_index < len(questions):
            raise IndexError(f"Question index {question_index} is out of range.")
        instruction_index = question_index % len(instructions)
        instruction = instructions[instruction_index]
        positive_system = make_system_prompt(
            config["trait"]["positive_assistant_name"], instruction["pos"]
        )
        negative_system = make_system_prompt(
            config["trait"]["negative_assistant_name"], instruction["neg"]
        )
        positive_prompt, negative_prompt, alignment = _align_system_prompt_pair(
            model.tokenizer,
            question=questions[question_index],
            positive_system_prompt=positive_system,
            negative_system_prompt=negative_system,
        )
        prompt_tokens = len(
            model.tokenizer.encode(
                positive_prompt, add_special_tokens=False
            )
        )
        if prompt_tokens > args.max_length:
            raise ValueError(
                f"Question {question_index} exceeds --max-length."
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
                "prompt_alignment": alignment,
            }
        )

    existing_rows = read_jsonl(raw_path) if raw_path.exists() else []
    rows_by_id = {row["record_id"]: row for row in existing_rows}
    if len(rows_by_id) != len(existing_rows):
        raise ValueError("Duplicate response record IDs.")

    def save_metadata(status: str) -> None:
        rows = read_jsonl(raw_path) if raw_path.exists() else []

        def relative(path: Path | None):
            return str(path.relative_to(ROOT)) if path is not None else None

        def digest(path: Path | None):
            return sha256_file(path) if path is not None else None

        payload = {
            "schema_version": 1,
            "status": status,
            "method": "strict recurrent vector-mediated circuit composition",
            "claim_scope": "free-generation differential circuit sufficiency",
            "behavior_protocol_id": config["protocol_id"],
            "condition": args.condition,
            "model": model_cfg,
            "model_architecture": observed_architecture,
            "upstream_run": args.upstream_run,
            "downstream_run": args.downstream_run,
            "upstream_k": args.upstream_k,
            "downstream_k": args.downstream_k,
            "upstream_circuit": relative(upstream_circuit_path),
            "upstream_circuit_sha256": digest(upstream_circuit_path),
            "downstream_circuit": relative(downstream_circuit_path),
            "downstream_circuit_sha256": digest(downstream_circuit_path),
            "upstream_evaluation": relative(upstream_evaluation_path),
            "upstream_evaluation_sha256": digest(upstream_evaluation_path),
            "downstream_evaluation": relative(downstream_evaluation_path),
            "downstream_evaluation_sha256": digest(
                downstream_evaluation_path
            ),
            "minimum_internal_faithfulness": (
                args.minimum_internal_faithfulness
            ),
            "upstream_connectivity": upstream_connectivity,
            "downstream_connectivity": downstream_connectivity,
            "interface": {
                "hook": target_hook,
                "layer": layer,
                "direction": str(direction_path.relative_to(ROOT)),
                "direction_sha256": sha256_file(direction_path),
                "timing": "every current response-prediction position",
                "orthogonal_clean_minus_corrupt_components": "discarded",
            },
            "orthogonal_seed": args.orthogonal_seed,
            "question_indices": args.question_indices,
            "decoding": {
                "method": "greedy",
                "max_new_tokens": args.max_new_tokens,
                "backend": "full_prefix_forward_ungrouped_gqa",
            },
            "max_length": args.max_length,
            "expected_responses": len(designed),
            "responses": len(rows),
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
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
        }
        write_json(metadata_path, payload)

    save_metadata("running")
    for source in tqdm(designed, desc=args.condition):
        record_id = (
            f"sufficient-{args.upstream_run or 'none'}-"
            f"{args.downstream_run or 'none'}-"
            f"ku{args.upstream_k}-kd{args.downstream_k}-"
            f"{condition_tag}-q{source['question_index']:02d}"
        )
        if record_id in rows_by_id:
            continue
        clean_tokens, clean_mask = tokenize_prompts(
            model, [source["prompt"]], max_length=args.max_length
        )
        corrupt_tokens, corrupt_mask = tokenize_prompts(
            model, [source["negative_prompt"]], max_length=args.max_length
        )
        if (
            clean_tokens.shape != corrupt_tokens.shape
            or not torch.equal(clean_mask, corrupt_mask)
        ):
            raise ValueError("Behavior prompt pair lost token alignment.")
        prompt_end = clean_tokens.shape[1] - 1
        generated = []
        eos_id = model.tokenizer.eos_token_id
        initial_diagnostic = None

        for _ in tqdm(
            range(args.max_new_tokens),
            desc=f"{args.condition}-q{source['question_index']:02d}",
            leave=False,
        ):
            if args.condition == "normal_positive":
                with torch.inference_mode():
                    logits = model(
                        clean_tokens, attention_mask=clean_mask
                    ).detach()
            elif args.condition == "negative_baseline":
                with torch.inference_mode():
                    logits = model(
                        corrupt_tokens, attention_mask=corrupt_mask
                    ).detach()
            else:
                clean_state, _ = _capture_state(
                    model, clean_tokens, clean_mask, target_hook
                )
                corrupt_state, _ = _capture_state(
                    model, corrupt_tokens, corrupt_mask, target_hook
                )
                prediction_mask = torch.zeros_like(
                    clean_mask, dtype=torch.bool
                )
                prediction_mask[:, prompt_end:] = True
                oracle_interface = vector_only_rescue_state(
                    clean_state,
                    corrupt_state,
                    direction,
                    prediction_mask,
                ).detach()

                upstream_interface = None
                if args.condition in {
                    "upstream_circuit_full_downstream",
                    "composed_circuit",
                    "composed_v_blocked",
                    "composed_orthogonal",
                }:
                    upstream_state = mixed_state_from_prompt_pair(
                        model,
                        upstream_graph,
                        clean_tokens,
                        clean_mask,
                        corrupt_tokens,
                        corrupt_mask,
                        target_hook=target_hook,
                        patch_selected=False,
                        cache_dtype=compact_dtype,
                    )
                    upstream_interface = vector_only_rescue_state(
                        upstream_state,
                        corrupt_state,
                        direction,
                        prediction_mask,
                    ).detach()

                if initial_diagnostic is None:
                    position = -1

                    def coordinate(state, vector):
                        return float(
                            (
                                state[:, position].float()
                                * vector.to(state.device).float()
                            )
                            .sum(dim=-1)
                            .item()
                        )

                    initial_diagnostic = {
                        "clean_coordinate": coordinate(
                            clean_state, direction
                        ),
                        "corrupt_coordinate": coordinate(
                            corrupt_state, direction
                        ),
                        "oracle_interface_coordinate": coordinate(
                            oracle_interface, direction
                        ),
                        "upstream_interface_coordinate": (
                            coordinate(upstream_interface, direction)
                            if upstream_interface is not None
                            else None
                        ),
                    }

                if args.condition == "oracle_full_state_interface":
                    logits = _full_downstream_logits(
                        model,
                        clean_tokens,
                        clean_mask,
                        target_hook,
                        clean_state,
                    )
                elif args.condition == "oracle_vector_interface":
                    logits = _full_downstream_logits(
                        model,
                        clean_tokens,
                        clean_mask,
                        target_hook,
                        oracle_interface,
                    )
                elif args.condition == "upstream_circuit_full_downstream":
                    logits = _full_downstream_logits(
                        model,
                        clean_tokens,
                        clean_mask,
                        target_hook,
                        upstream_interface,
                    )
                elif args.condition == "oracle_vector_downstream_circuit":
                    logits = _downstream_circuit_logits(
                        model,
                        downstream_graph,
                        clean_tokens,
                        clean_mask,
                        target_hook=target_hook,
                        clean_interface=oracle_interface,
                        corrupt_interface=corrupt_state,
                        cache_dtype=compact_dtype,
                    )
                elif args.condition == "composed_circuit":
                    logits = _downstream_circuit_logits(
                        model,
                        downstream_graph,
                        clean_tokens,
                        clean_mask,
                        target_hook=target_hook,
                        clean_interface=upstream_interface,
                        corrupt_interface=corrupt_state,
                        cache_dtype=compact_dtype,
                    )
                elif args.condition == "composed_v_blocked":
                    logits = _downstream_circuit_logits(
                        model,
                        downstream_graph,
                        clean_tokens,
                        clean_mask,
                        target_hook=target_hook,
                        clean_interface=corrupt_state,
                        corrupt_interface=corrupt_state,
                        cache_dtype=compact_dtype,
                    )
                elif args.condition == "composed_orthogonal":
                    vector = direction.to(
                        device=upstream_interface.device,
                        dtype=upstream_interface.dtype,
                    )
                    coefficient = (
                        (upstream_interface - corrupt_state) * vector
                    ).sum(dim=-1, keepdim=True)
                    active = prediction_mask.to(
                        device=coefficient.device,
                        dtype=coefficient.dtype,
                    ).unsqueeze(-1)
                    orthogonal_interface = (
                        corrupt_state
                        + active
                        * coefficient
                        * orthogonal.to(
                            device=coefficient.device,
                            dtype=coefficient.dtype,
                        )
                    )
                    logits = _downstream_circuit_logits(
                        model,
                        downstream_graph,
                        clean_tokens,
                        clean_mask,
                        target_hook=target_hook,
                        clean_interface=orthogonal_interface,
                        corrupt_interface=corrupt_state,
                        cache_dtype=compact_dtype,
                    )
                else:
                    raise ValueError(f"Unknown condition: {args.condition}")

            next_token_logits = logits[0, -1]
            if not torch.isfinite(next_token_logits).all():
                raise FloatingPointError(
                    "Non-finite logits during sufficient-circuit generation."
                )
            next_token = int(next_token_logits.argmax().item())
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
            one = torch.ones(
                (1, 1), device=clean_mask.device, dtype=clean_mask.dtype
            )
            clean_mask = torch.cat([clean_mask, one], dim=1)
            corrupt_mask = torch.cat(
                [corrupt_mask, one.clone()], dim=1
            )
            del logits
            if eos_id is not None and next_token == eos_id:
                break

        row = {
            **{
                key: value
                for key, value in source.items()
                if key != "negative_prompt"
            },
            "schema_version": 1,
            "record_id": record_id,
            "behavior_protocol_id": config["protocol_id"],
            "trait": trait_slug,
            "model": model_cfg["name"],
            "model_revision": model_cfg.get("revision"),
            "condition": args.condition,
            "upstream_k": args.upstream_k,
            "downstream_k": args.downstream_k,
            "interface_timing": "recurrent_response_prediction_positions",
            "orthogonal_clean_minus_corrupt_components": "discarded",
            "orthogonal_seed": args.orthogonal_seed,
            "initial_interface_diagnostic": initial_diagnostic,
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

    if len(read_jsonl(raw_path)) != len(designed):
        raise RuntimeError("Sufficient-circuit behavior output is incomplete.")
    save_metadata("complete")
    print(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

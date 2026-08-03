#!/usr/bin/env python3
"""Test vector mediation and connected upstream/downstream persona circuits.

The upstream corruption is the token-aligned negative-persona prompt. The downstream
corruption is signed negative steering at the configured persona interface, conditional
on the upstream intervention and any interface rescue. Each generation step
uses the same already-generated prefix in every counterfactual run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_negative_prompt


CONDITIONS = (
    "both_ablated",
    "both_circuit_only",
    "upstream_vector_rescue",
    "upstream_full_state_rescue",
    "upstream_orthogonal_rescue",
    "vector_rescue_downstream_ablated",
)


def _prompt_identity_sha256(path: Path) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identities = [
        {
            "question_index": int(row["question_index"]),
            "instruction_index": int(row["instruction_index"]),
            "positive_prompt": row["positive_prompt"],
            "negative_prompt": row["negative_prompt"],
        }
        for row in manifest["discovery"]["samples"]
    ]
    serialized = json.dumps(
        identities, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--upstream-run", required=True)
    parser.add_argument("--downstream-run", required=True)
    parser.add_argument("--upstream-k", type=int, required=True)
    parser.add_argument("--downstream-k", type=int, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--question-indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional unique output leaf; useful for preflights and split batches.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("Question indices must be unique.")

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import (
        _construction_hooks,
        _receiver_cache_hooks,
        _sparse_circuit_only_construction_hooks,
        _source_cache_hooks,
    )
    from refusal_circuit.graph import (
        load_circuit_edge_names,
        make_projection_graph,
        make_steering_to_logits_graph,
        set_selected_edges,
    )
    from refusal_circuit.interventions import activation_addition_hook
    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        make_system_prompt,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    trait_slug = config["trait"]["name"]
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    upstream_root = (
        result_root
        / "circuit_discovery_upstream_contrastive"
        / args.upstream_run
    )
    downstream_root = (
        result_root / "circuit_discovery_steering_path" / args.downstream_run
    )
    upstream_metadata_path = upstream_root / "eapig_run_metadata.json"
    downstream_metadata_path = downstream_root / "eapig_run_metadata.json"
    upstream_metadata = json.loads(
        upstream_metadata_path.read_text(encoding="utf-8")
    )
    downstream_metadata = json.loads(
        downstream_metadata_path.read_text(encoding="utf-8")
    )
    upstream_sample_manifest = ROOT / upstream_metadata["sample_manifest"]
    downstream_sample_manifest = ROOT / downstream_metadata["sample_manifest"]
    upstream_prompt_identity = _prompt_identity_sha256(
        upstream_sample_manifest
    )
    downstream_prompt_identity = _prompt_identity_sha256(
        downstream_sample_manifest
    )
    if (
        upstream_metadata.get("shared_protocol")
        not in {
            "shared_prompt_v1",
            "shared_prompt_full_transcript_v2",
        }
        or upstream_metadata.get("shared_protocol")
        != downstream_metadata.get("shared_protocol")
        or upstream_prompt_identity != downstream_prompt_identity
    ):
        raise ValueError(
            "Connected evaluation requires upstream/downstream rankings "
            "from the same frozen shared-prompt manifest."
        )
    upstream_circuit_path = (
        upstream_root
        / "circuits_nested"
        / f"eapig_top_{args.upstream_k}.json"
    )
    downstream_circuit_path = (
        downstream_root
        / "circuits_nested"
        / f"steering_path_eapig_top_{args.downstream_k}.json"
    )
    upstream_payload = json.loads(
        upstream_circuit_path.read_text(encoding="utf-8")
    )
    downstream_payload = json.loads(
        downstream_circuit_path.read_text(encoding="utf-8")
    )
    if upstream_payload.get("score_source") != upstream_metadata.get("score_file"):
        raise ValueError("Upstream circuit does not use the frozen upstream scores.")
    if downstream_payload.get("score_source") != downstream_metadata.get(
        "score_file"
    ):
        raise ValueError("Downstream circuit does not use the frozen downstream scores.")
    if int(upstream_payload["requested_k"]) != args.upstream_k:
        raise ValueError("Upstream circuit k mismatch.")
    if int(downstream_payload["requested_k"]) != args.downstream_k:
        raise ValueError("Downstream circuit k mismatch.")

    output_leaf = (
        args.output_tag
        or f"{args.condition}_q{'-'.join(map(str, args.question_indices))}"
    )
    output = (
        intermediate
        / "connected_circuit_behavior"
        / f"{args.upstream_run}__{args.downstream_run}"
        / f"ku{args.upstream_k}_kd{args.downstream_k}"
        / output_leaf
    )
    raw_path = output / "responses.jsonl"
    metadata_path = output / "metadata.json"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"{output} already contains output; pass --resume.")
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
    model.set_ungroup_grouped_query_attention(True)
    layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    upstream_graph = make_projection_graph(model, layer)
    downstream_graph = make_steering_to_logits_graph(model, layer)
    set_selected_edges(
        upstream_graph, load_circuit_edge_names(upstream_circuit_path)
    )
    set_selected_edges(
        downstream_graph, load_circuit_edge_names(downstream_circuit_path)
    )

    # Use the trait-specific unit vector so the connected-circuit test is trait-agnostic.
    unit_path = result_root / f"vector/{trait_slug}_direction.pt"
    unit_direction = torch.load(
        unit_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    ).float()
    unit_direction = unit_direction / unit_direction.norm().clamp_min(1e-8)
    # Use the trait-specific response-average vector for the rescue/ablation geometry.
    raw_path_vector = result_root / f"vector/{trait_slug}_response_avg_diff.pt"
    raw_stack = torch.load(
        raw_path_vector,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    raw_direction = raw_stack[layer] if raw_stack.ndim == 2 else raw_stack
    corrupt_coefficient = float(
        downstream_metadata["steering"]["corrupt_coefficient"]
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    orthogonal = torch.randn(
        unit_direction.shape, generator=generator, dtype=torch.float32
    ).to(unit_direction.device)
    orthogonal = orthogonal - orthogonal.dot(unit_direction) * unit_direction
    orthogonal = orthogonal / orthogonal.norm().clamp_min(1e-8)

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

    existing_rows = read_jsonl(raw_path) if raw_path.exists() else []
    rows_by_id = {row["record_id"]: row for row in existing_rows}
    if len(rows_by_id) != len(existing_rows):
        raise ValueError("Duplicate output record IDs.")

    def capture_target(tokens, mask) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(activations: torch.Tensor, hook):
            del hook
            captured["value"] = activations[:, -1, :].detach().clone()
            return activations

        with torch.inference_mode(), model.hooks(fwd_hooks=[(target_hook, hook)]):
            model(tokens, attention_mask=mask)
        return captured["value"]

    compact_dtype = next(model.parameters()).dtype

    def upstream_corrupt_cache(tokens, mask, *, circuit_only: bool):
        cache: dict[str, torch.Tensor] = {}
        receivers: dict[tuple[str, str | None], torch.Tensor] = {}
        hooks = _source_cache_hooks(
            model,
            upstream_graph,
            cache,
            selected_only=True,
            storage_dtype=compact_dtype,
        )
        if circuit_only:
            hooks += _receiver_cache_hooks(
                upstream_graph,
                receivers,
                storage_dtype=compact_dtype,
            )
        with torch.inference_mode(), model.hooks(
            fwd_hooks=hooks
        ):
            model(tokens, attention_mask=mask)
        return cache, receivers

    def upstream_hooks(
        corrupt_cache: dict[str, torch.Tensor],
        *,
        corrupt_receivers=None,
        circuit_only: bool = False,
        rescue: str | None,
        clean_reference: torch.Tensor | None,
    ):
        mixed_cache: dict[str, torch.Tensor] = {}
        hooks = (
            _source_cache_hooks(
                model,
                upstream_graph,
                mixed_cache,
                selected_only=True,
                storage_dtype=compact_dtype,
            )
        )
        hooks += (
            _sparse_circuit_only_construction_hooks(
                upstream_graph,
                corrupt_cache,
                mixed_cache,
                corrupt_receivers,
            )
            if circuit_only
            else _construction_hooks(
                upstream_graph,
                corrupt_cache,
                mixed_cache,
                patch_selected=True,
            )
        )
        if rescue is not None:
            if clean_reference is None:
                raise ValueError("A rescue requires a clean target reference.")

            def rescue_hook(activations: torch.Tensor, hook):
                del hook
                current = activations[:, -1, :]
                reference = clean_reference.to(
                    device=current.device, dtype=current.dtype
                )
                if rescue == "full":
                    activations[:, -1, :] = reference
                    return activations
                direction = unit_direction.to(
                    device=current.device, dtype=current.dtype
                )
                missing = ((reference - current) * direction).sum(
                    dim=-1, keepdim=True
                )
                rescue_direction = (
                    direction
                    if rescue == "vector"
                    else orthogonal.to(
                        device=current.device, dtype=current.dtype
                    )
                )
                activations[:, -1, :] = current + missing * rescue_direction
                return activations

            hooks.append((target_hook, rescue_hook))
        return hooks

    def conditional_downstream_logits(
        clean_tokens,
        clean_mask,
        u_corrupt,
        *,
        u_corrupt_receivers=None,
        upstream_circuit_only: bool = False,
        downstream_circuit_only: bool = False,
        rescue: str | None,
        clean_reference: torch.Tensor | None,
    ) -> torch.Tensor:
        downstream_corrupt: dict[str, torch.Tensor] = {}

        def corrupt_transform(name: str, activations: torch.Tensor):
            if name != target_hook:
                return activations
            changed = activations.clone()
            changed[:, -1, :] += corrupt_coefficient * raw_direction.to(
                device=changed.device, dtype=changed.dtype
            )
            return changed

        corrupt_hooks = upstream_hooks(
            u_corrupt,
            corrupt_receivers=u_corrupt_receivers,
            circuit_only=upstream_circuit_only,
            rescue=rescue,
            clean_reference=clean_reference,
        )
        corrupt_hooks += _source_cache_hooks(
            model,
            downstream_graph,
            downstream_corrupt,
            cache_transform=corrupt_transform,
            selected_only=True,
            storage_dtype=compact_dtype,
        )
        corrupt_hooks.append(
            (
                target_hook,
                activation_addition_hook(
                    raw_direction,
                    coefficient=corrupt_coefficient,
                    position=-1,
                ),
            )
        )
        downstream_corrupt_receivers = {}
        if downstream_circuit_only:
            corrupt_hooks += _receiver_cache_hooks(
                downstream_graph,
                downstream_corrupt_receivers,
                storage_dtype=compact_dtype,
            )
        with torch.inference_mode(), model.hooks(fwd_hooks=corrupt_hooks):
            model(clean_tokens, attention_mask=clean_mask)

        downstream_clean: dict[str, torch.Tensor] = {}
        final_hooks = upstream_hooks(
            u_corrupt,
            corrupt_receivers=u_corrupt_receivers,
            circuit_only=upstream_circuit_only,
            rescue=rescue,
            clean_reference=clean_reference,
        )
        final_hooks += _source_cache_hooks(
            model,
            downstream_graph,
            downstream_clean,
            selected_only=True,
            storage_dtype=compact_dtype,
        )
        final_hooks += (
            _sparse_circuit_only_construction_hooks(
                downstream_graph,
                downstream_corrupt,
                downstream_clean,
                downstream_corrupt_receivers,
            )
            if downstream_circuit_only
            else _construction_hooks(
                downstream_graph,
                downstream_corrupt,
                downstream_clean,
                patch_selected=True,
            )
        )
        with torch.inference_mode(), model.hooks(fwd_hooks=final_hooks):
            return model(clean_tokens, attention_mask=clean_mask).detach()

    def save_metadata(status: str) -> None:
        rows = read_jsonl(raw_path) if raw_path.exists() else []
        write_json(
            metadata_path,
            {
                "schema_version": 1,
                "status": status,
                "method": "connected formation-expression causal intervention",
                "claim_scope": "development vector mediation and circuit composition",
                "behavior_protocol_id": config["protocol_id"],
                "condition": args.condition,
                "model": model_cfg,
                "upstream_run": args.upstream_run,
                "downstream_run": args.downstream_run,
                "upstream_k": args.upstream_k,
                "downstream_k": args.downstream_k,
                "upstream_circuit": str(
                    upstream_circuit_path.relative_to(ROOT)
                ),
                "upstream_circuit_sha256": sha256_file(upstream_circuit_path),
                "downstream_circuit": str(
                    downstream_circuit_path.relative_to(ROOT)
                ),
                "downstream_circuit_sha256": sha256_file(
                    downstream_circuit_path
                ),
                "shared_prompt_identity_sha256": upstream_prompt_identity,
                "upstream_sample_manifest_sha256": upstream_metadata.get(
                    "sample_manifest_sha256"
                ),
                "downstream_sample_manifest_sha256": downstream_metadata.get(
                    "sample_manifest_sha256"
                ),
                "upstream_counterfactual": "token-aligned negative-persona prompt",
                "downstream_counterfactual": {
                    "intervention": "signed response-time vector subtraction",
                    "coefficient": corrupt_coefficient,
                    "conditional_on_upstream_and_rescue": True,
                },
                "interface": {
                    "hook": target_hook,
                    "layer": layer,
                    "position": "current next-token predictor",
                },
                "question_indices": args.question_indices,
                "output_tag": args.output_tag,
                "decoding": {
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                },
                "responses": len(rows),
                "expected_responses": len(designed),
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

    save_metadata("running")
    for source in tqdm(designed, desc=args.condition):
        record_id = (
            f"connected-ku{args.upstream_k}-kd{args.downstream_k}-"
            f"{args.condition}-q{source['question_index']:02d}"
        )
        if record_id in rows_by_id:
            continue
        clean_tokens, clean_mask = tokenize_prompts(
            model, [source["prompt"]], max_length=args.max_length
        )
        corrupt_tokens, corrupt_mask = tokenize_prompts(
            model, [source["negative_prompt"]], max_length=args.max_length
        )
        generated: list[int] = []
        eos_id = model.tokenizer.eos_token_id
        rescue = (
            "vector"
            if args.condition
            in {
                "upstream_vector_rescue",
                "vector_rescue_downstream_ablated",
            }
            else "full"
            if args.condition == "upstream_full_state_rescue"
            else "orthogonal"
            if args.condition == "upstream_orthogonal_rescue"
            else None
        )
        union_circuit_only = args.condition == "both_circuit_only"
        downstream_ablated = args.condition in {
            "both_ablated",
            "both_circuit_only",
            "vector_rescue_downstream_ablated",
        }
        for _ in tqdm(
            range(args.max_new_tokens),
            desc=f"{args.condition}-q{source['question_index']:02d}",
            leave=False,
        ):
            u_corrupt, u_corrupt_receivers = upstream_corrupt_cache(
                corrupt_tokens,
                corrupt_mask,
                circuit_only=union_circuit_only,
            )
            clean_reference = (
                capture_target(clean_tokens, clean_mask)
                if rescue is not None
                else None
            )
            if downstream_ablated:
                logits = conditional_downstream_logits(
                    clean_tokens,
                    clean_mask,
                    u_corrupt,
                    u_corrupt_receivers=u_corrupt_receivers,
                    upstream_circuit_only=union_circuit_only,
                    downstream_circuit_only=union_circuit_only,
                    rescue=rescue,
                    clean_reference=clean_reference,
                )
            else:
                hooks = upstream_hooks(
                    u_corrupt,
                    corrupt_receivers=u_corrupt_receivers,
                    circuit_only=False,
                    rescue=rescue,
                    clean_reference=clean_reference,
                )
                with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
                    logits = model(
                        clean_tokens, attention_mask=clean_mask
                    ).detach()
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
                (1, 1), device=clean_mask.device, dtype=clean_mask.dtype
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
            "schema_version": 1,
            "record_id": record_id,
            "behavior_protocol_id": config["protocol_id"],
            "trait": config["trait"]["name"],
            "model": model_cfg["name"],
            "model_revision": model_cfg.get("revision"),
            "condition": args.condition,
            "upstream_k": args.upstream_k,
            "downstream_k": args.downstream_k,
            "answer": model.tokenizer.decode(
                generated, skip_special_tokens=True
            ),
            "generated_token_count": len(generated),
            "hit_max_new_tokens": len(generated) >= args.max_new_tokens,
            "decoding": "greedy",
        }
        write_jsonl(raw_path, [row], mode="a")
        rows_by_id[record_id] = row
        save_metadata("running")

    if len(read_jsonl(raw_path)) != len(designed):
        raise RuntimeError("Connected circuit behavior output is incomplete.")
    save_metadata("complete")
    print(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

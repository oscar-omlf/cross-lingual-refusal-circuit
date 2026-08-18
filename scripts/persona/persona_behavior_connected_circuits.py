#!/usr/bin/env python3
"""Test vector mediation and connected upstream/downstream persona circuits.

The upstream corruption is the token-aligned negative-persona prompt. By
default, upstream edges are patched only at prompt positions and interface
rescue is applied at the fixed final prompt token. This directly tests the
formation-interface-expression hypothesis. Recurrent all-position variants are
available as an explicit diagnostic. The downstream corruption is signed
negative steering for joint necessity/reconstruction conditions. The specific
mediation condition instead patches downstream edges from the same upstream-cut
run without rescue into the vector-rescued run, isolating transmission of the
rescue-induced effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_system_prompt_pair


CONDITIONS = (
    "normal",
    "upstream_ablated",
    "downstream_ablated",
    "both_ablated",
    "both_circuit_only",
    "upstream_vector_rescue",
    "upstream_full_state_rescue",
    "upstream_full_prefix_state_rescue",
    "upstream_orthogonal_rescue",
    "upstream_unrelated_vector_rescue",
    "vector_rescue_downstream_ablated",
)


def _downstream_estimand(condition: str) -> str:
    if condition == "vector_rescue_downstream_ablated":
        return "rescue_effect_blocking"
    if condition in {
        "downstream_ablated",
        "both_ablated",
        "both_circuit_only",
    }:
        return "negative_steering"
    return "none"


def _uses_upstream_intervention(condition: str) -> bool:
    return condition not in {"normal", "downstream_ablated"}


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
    parser.add_argument("--upstream-circuit-path", type=Path, default=None)
    parser.add_argument("--downstream-circuit-path", type=Path, default=None)
    parser.add_argument(
        "--unrelated-vector-path",
        type=Path,
        default=None,
        help=(
            "All-layer or single-layer vector used by "
            "upstream_unrelated_vector_rescue."
        ),
    )
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--question-indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional unique output leaf; useful for preflights and split batches.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--upstream-position-scope",
        choices=("prompt_only", "all_positions"),
        default="prompt_only",
        help=(
            "Patch upstream edges only over the original prompt, or also over "
            "generated response positions."
        ),
    )
    parser.add_argument(
        "--interface-timing",
        choices=("prompt_end", "current_token"),
        default="prompt_end",
        help=(
            "Restore the interface at the fixed final prompt token, or at the "
            "current next-token predictor on every decoding step."
        ),
    )
    parser.add_argument("--orthogonal-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("Question indices must be unique.")
    if (
        args.condition == "both_circuit_only"
        and args.upstream_position_scope != "all_positions"
    ):
        raise ValueError(
            "Circuit-only reconstruction is an all-position intervention; "
            "pass --upstream-position-scope all_positions."
        )
    if (
        args.condition == "upstream_full_prefix_state_rescue"
        and (
            args.upstream_position_scope != "prompt_only"
            or args.interface_timing != "prompt_end"
        )
    ):
        raise ValueError(
            "Full-prefix interface rescue is the positive control for the "
            "prompt_only/prompt_end bottleneck estimand."
        )

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import (
        _construction_hooks,
        _receiver_cache_hooks,
        _sparse_circuit_only_construction_hooks,
        _source_cache_hooks,
    )
    from refusal_circuit.graph import (
        circuit_connectivity,
        load_circuit_edge_names,
        make_projection_graph,
        make_steering_to_logits_graph,
        set_selected_edges,
        validate_intervention_connectivity,
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
    if (
        upstream_metadata.get("status") != "complete"
        or downstream_metadata.get("status") != "complete"
    ):
        raise ValueError("Connected evaluation requires completed discoveries.")
    if (
        upstream_metadata.get("model") != config["model"]
        or downstream_metadata.get("model") != config["model"]
    ):
        raise ValueError(
            "Connected discoveries and behavior model configurations differ."
        )
    upstream_sample_manifest = ROOT / upstream_metadata["sample_manifest"]
    downstream_sample_manifest = ROOT / downstream_metadata["sample_manifest"]
    if (
        sha256_file(upstream_sample_manifest)
        != upstream_metadata.get("sample_manifest_sha256")
        or sha256_file(downstream_sample_manifest)
        != downstream_metadata.get("sample_manifest_sha256")
    ):
        raise ValueError("A connected discovery sample manifest hash changed.")
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
    upstream_circuit_path = args.upstream_circuit_path or (
        upstream_root
        / "circuits_source_to_sink"
        / f"eapig_top_{args.upstream_k}.json"
    )
    downstream_circuit_path = args.downstream_circuit_path or (
        downstream_root
        / "circuits_source_to_sink"
        / f"steering_path_eapig_top_{args.downstream_k}.json"
    )
    upstream_circuit_path = (
        upstream_circuit_path
        if upstream_circuit_path.is_absolute()
        else ROOT / upstream_circuit_path
    )
    downstream_circuit_path = (
        downstream_circuit_path
        if downstream_circuit_path.is_absolute()
        else ROOT / downstream_circuit_path
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
    observed_architecture = validate_model_architecture(
        model, config["model_architecture"]
    )
    model.set_ungroup_grouped_query_attention(True)
    layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    upstream_graph = make_projection_graph(model, layer)
    downstream_graph = make_steering_to_logits_graph(model, layer)
    upstream_edge_names = load_circuit_edge_names(upstream_circuit_path)
    downstream_edge_names = load_circuit_edge_names(downstream_circuit_path)
    upstream_connectivity = circuit_connectivity(
        upstream_graph, upstream_edge_names
    )
    downstream_connectivity = circuit_connectivity(
        downstream_graph, downstream_edge_names
    )
    validate_intervention_connectivity(
        {
            "upstream": upstream_connectivity,
            "downstream": downstream_connectivity,
        },
        require_source_to_sink=args.condition == "both_circuit_only",
    )
    set_selected_edges(upstream_graph, upstream_edge_names)
    set_selected_edges(downstream_graph, downstream_edge_names)
    if not model.cfg.use_attn_result:
        raise RuntimeError(
            "Connected-circuit evaluation requires TransformerLens' reference "
            "per-head attention results."
        )

    # Use the trait-specific unit vector so the connected-circuit test is trait-agnostic.
    unit_path = result_root / f"vector/{trait_slug}_direction.pt"
    unit_sha256 = sha256_file(unit_path)
    if (
        upstream_metadata.get("direction")
        != str(unit_path.relative_to(ROOT))
        or upstream_metadata.get("direction_sha256") != unit_sha256
    ):
        raise ValueError(
            "Connected upstream discovery used a different persona direction."
        )
    unit_direction = torch.load(
        unit_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    ).float()
    unit_direction = unit_direction / unit_direction.norm().clamp_min(1e-8)
    # Use the trait-specific response-average vector for the rescue/ablation geometry.
    raw_path_vector = result_root / f"vector/{trait_slug}_response_avg_diff.pt"
    raw_vector_sha256 = sha256_file(raw_path_vector)
    downstream_steering = downstream_metadata.get("steering", {})
    if (
        downstream_steering.get("vector")
        != str(raw_path_vector.relative_to(ROOT))
        or downstream_steering.get("vector_sha256")
        != raw_vector_sha256
    ):
        raise ValueError(
            "Connected downstream discovery used a different response vector."
        )
    raw_stack = torch.load(
        raw_path_vector,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    raw_direction = raw_stack[layer] if raw_stack.ndim == 2 else raw_stack
    corrupt_coefficient = float(
        downstream_metadata["steering"]["corrupt_coefficient"]
    )
    generator = torch.Generator(device="cpu").manual_seed(args.orthogonal_seed)
    orthogonal = torch.randn(
        unit_direction.shape, generator=generator, dtype=torch.float32
    ).to(unit_direction.device)
    orthogonal = orthogonal - orthogonal.dot(unit_direction) * unit_direction
    orthogonal = orthogonal / orthogonal.norm().clamp_min(1e-8)
    unrelated_path = None
    unrelated_direction = None
    unrelated_raw_cosine = None
    if args.condition == "upstream_unrelated_vector_rescue":
        if args.unrelated_vector_path is None:
            raise ValueError(
                "--unrelated-vector-path is required for the unrelated rescue."
            )
        unrelated_path = (
            args.unrelated_vector_path
            if args.unrelated_vector_path.is_absolute()
            else ROOT / args.unrelated_vector_path
        )
        unrelated_stack = torch.load(
            unrelated_path,
            map_location=unit_direction.device,
            weights_only=True,
        ).float()
        unrelated_direction = (
            unrelated_stack[layer]
            if unrelated_stack.ndim == 2
            else unrelated_stack
        )
        if unrelated_direction.shape != unit_direction.shape:
            raise ValueError("Unrelated vector has the wrong hidden dimension.")
        unrelated_direction = unrelated_direction / unrelated_direction.norm().clamp_min(
            1e-8
        )
        unrelated_raw_cosine = float(
            unrelated_direction.dot(unit_direction).item()
        )
        unrelated_direction = (
            unrelated_direction
            - unrelated_direction.dot(unit_direction) * unit_direction
        )
        if unrelated_direction.norm() <= 1e-6:
            raise ValueError(
                "Unrelated vector is collinear with the target persona vector."
            )
        unrelated_direction = unrelated_direction / unrelated_direction.norm()

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
        if alignment["tokens_after_alignment"] > args.max_length:
            raise ValueError(
                f"Question {question_index} exceeds --max-length; refusing "
                "to truncate a connected-circuit prompt."
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

    def capture_target(
        tokens,
        mask,
        *,
        position: int,
        fwd_hooks=(),
    ) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(activations: torch.Tensor, hook):
            del hook
            captured["value"] = activations[:, position, :].detach().clone()
            return activations

        with torch.inference_mode(), model.hooks(
            fwd_hooks=[*fwd_hooks, (target_hook, hook)]
        ):
            model(tokens, attention_mask=mask)
        return captured["value"]

    def capture_prefix_interface(
        tokens,
        mask,
        *,
        prefix_length: int,
        fwd_hooks=(),
    ) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(activations: torch.Tensor, hook):
            del hook
            captured["value"] = (
                activations[:, :prefix_length, :].detach().clone()
            )
            return activations

        with torch.inference_mode(), model.hooks(
            fwd_hooks=[*fwd_hooks, (target_hook, hook)]
        ):
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
        target_position: int,
        prefix_length: int,
        position_mask: torch.Tensor | None,
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
                position_mask=position_mask,
            )
        )
        if rescue is not None:
            if clean_reference is None:
                raise ValueError("A rescue requires a clean target reference.")

            def rescue_hook(activations: torch.Tensor, hook):
                del hook
                current = activations[:, target_position, :]
                reference = clean_reference.to(
                    device=current.device, dtype=current.dtype
                )
                if rescue == "full_prefix":
                    if reference.ndim != 3 or reference.shape[1] != prefix_length:
                        raise ValueError(
                            "Full-prefix rescue reference has the wrong shape."
                        )
                    activations[:, :prefix_length, :] = reference
                    return activations
                if rescue == "full":
                    activations[:, target_position, :] = reference
                    return activations
                direction = unit_direction.to(
                    device=current.device, dtype=current.dtype
                )
                missing = ((reference - current) * direction).sum(
                    dim=-1, keepdim=True
                )
                if rescue == "vector":
                    rescue_direction = direction
                elif rescue == "unrelated":
                    if unrelated_direction is None:
                        raise RuntimeError("Unrelated rescue vector was not loaded.")
                    rescue_direction = unrelated_direction.to(
                        device=current.device, dtype=current.dtype
                    )
                elif rescue == "orthogonal":
                    rescue_direction = orthogonal.to(
                        device=current.device, dtype=current.dtype
                    )
                else:
                    raise ValueError(f"Unknown rescue mode: {rescue}")
                activations[:, target_position, :] = (
                    current + missing * rescue_direction
                )
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
        target_position: int,
        prefix_length: int,
        upstream_position_mask: torch.Tensor | None,
        downstream_estimand: str,
        apply_upstream_intervention: bool,
    ) -> torch.Tensor:
        if downstream_estimand not in {
            "negative_steering",
            "rescue_effect_blocking",
        }:
            raise ValueError(
                f"Unknown downstream estimand: {downstream_estimand}"
            )
        if downstream_estimand == "rescue_effect_blocking":
            if rescue != "vector":
                raise ValueError(
                    "Rescue-effect blocking requires a vector-rescued clean run."
                )
            if downstream_circuit_only:
                raise ValueError(
                    "Rescue-effect blocking is a selected-edge ablation, not a "
                    "circuit-only reconstruction."
                )

        downstream_corrupt: dict[str, torch.Tensor] = {}

        def corrupt_transform(name: str, activations: torch.Tensor):
            if name != target_hook:
                return activations
            changed = activations.clone()
            changed[:, -1, :] += corrupt_coefficient * raw_direction.to(
                device=changed.device, dtype=changed.dtype
            )
            return changed

        corrupt_rescue = (
            rescue
            if downstream_estimand == "negative_steering"
            else None
        )
        corrupt_hooks = (
            upstream_hooks(
                u_corrupt,
                corrupt_receivers=u_corrupt_receivers,
                circuit_only=upstream_circuit_only,
                rescue=corrupt_rescue,
                clean_reference=clean_reference,
                target_position=target_position,
                prefix_length=prefix_length,
                position_mask=upstream_position_mask,
            )
            if apply_upstream_intervention
            else []
        )
        corrupt_hooks += _source_cache_hooks(
            model,
            downstream_graph,
            downstream_corrupt,
            cache_transform=(
                corrupt_transform
                if downstream_estimand == "negative_steering"
                else None
            ),
            selected_only=True,
            storage_dtype=compact_dtype,
        )
        if downstream_estimand == "negative_steering":
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
        final_hooks = (
            upstream_hooks(
                u_corrupt,
                corrupt_receivers=u_corrupt_receivers,
                circuit_only=upstream_circuit_only,
                rescue=rescue,
                clean_reference=clean_reference,
                target_position=target_position,
                prefix_length=prefix_length,
                position_mask=upstream_position_mask,
            )
            if apply_upstream_intervention
            else []
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
        downstream_estimand = _downstream_estimand(args.condition)
        applies_upstream = _uses_upstream_intervention(args.condition)
        downstream_counterfactual = {
            "estimand": downstream_estimand,
            "conditional_on_upstream_intervention": applies_upstream,
        }
        if downstream_estimand == "negative_steering":
            downstream_counterfactual.update(
                {
                    "intervention": "signed response-time vector subtraction",
                    "coefficient": corrupt_coefficient,
                    "clean_run": "same upstream intervention and interface rescue",
                    "corrupt_run": (
                        "same upstream intervention and interface rescue plus "
                        "negative steering"
                    ),
                }
            )
        elif downstream_estimand == "rescue_effect_blocking":
            downstream_counterfactual.update(
                {
                    "intervention": (
                        "patch selected downstream edges from the unrescued "
                        "upstream-cut run into the vector-rescued upstream-cut run"
                    ),
                    "coefficient": None,
                    "clean_run": "upstream cut plus vector rescue",
                    "corrupt_run": "same upstream cut without interface rescue",
                    "negative_steering_used": False,
                }
            )
        else:
            downstream_counterfactual.update(
                {
                    "intervention": None,
                    "coefficient": None,
                    "clean_run": None,
                    "corrupt_run": None,
                }
            )
        write_json(
            metadata_path,
            {
                "schema_version": 2,
                "status": status,
                "method": "connected formation-expression causal intervention",
                "claim_scope": "development vector mediation and circuit composition",
                "behavior_protocol_id": config["protocol_id"],
                "condition": args.condition,
                "model": model_cfg,
                "model_architecture": observed_architecture,
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
                "direction": str(unit_path.relative_to(ROOT)),
                "direction_sha256": unit_sha256,
                "response_vector": str(raw_path_vector.relative_to(ROOT)),
                "response_vector_sha256": raw_vector_sha256,
                "unrelated_vector": (
                    str(unrelated_path.relative_to(ROOT))
                    if unrelated_path is not None
                    else None
                ),
                "unrelated_vector_sha256": (
                    sha256_file(unrelated_path)
                    if unrelated_path is not None
                    else None
                ),
                "unrelated_vector_cosine": (
                    float(unrelated_direction.dot(unit_direction).item())
                    if unrelated_direction is not None
                    else None
                ),
                "unrelated_vector_raw_cosine": unrelated_raw_cosine,
                "unrelated_vector_control": (
                    "target-orthogonalized equal-norm semantic vector"
                    if unrelated_direction is not None
                    else None
                ),
                "orthogonal_seed": args.orthogonal_seed,
                "upstream_connectivity": upstream_connectivity,
                "downstream_connectivity": downstream_connectivity,
                "shared_prompt_identity_sha256": upstream_prompt_identity,
                "upstream_sample_manifest_sha256": upstream_metadata.get(
                    "sample_manifest_sha256"
                ),
                "downstream_sample_manifest_sha256": downstream_metadata.get(
                    "sample_manifest_sha256"
                ),
                "upstream_counterfactual": (
                    "token-aligned negative-persona prompt"
                    if applies_upstream
                    else None
                ),
                "upstream_intervention_applied": applies_upstream,
                "upstream_position_scope": args.upstream_position_scope,
                "downstream_estimand": downstream_estimand,
                "downstream_counterfactual": downstream_counterfactual,
                "interface": {
                    "hook": target_hook,
                    "layer": layer,
                    "timing": args.interface_timing,
                    "position": (
                        "fixed final prompt token"
                        if args.interface_timing == "prompt_end"
                        else "current next-token predictor"
                    ),
                },
                "interface_timing": args.interface_timing,
                "question_indices": args.question_indices,
                "output_tag": args.output_tag,
                "decoding": {
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                    "backend": "full_prefix_forward_ungrouped_gqa",
                },
                "generation_backend": "full_prefix_forward_ungrouped_gqa",
                "max_length": args.max_length,
                "responses": len(rows),
                "memory_implementation": {
                    "attention_result_sources": (
                        "TransformerLens hook_result reference path"
                    ),
                    "split_qkv_projection": "pinned TransformerLens default",
                },
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

    expected_resume = {
        "behavior_protocol_id": config["protocol_id"],
        "condition": args.condition,
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "upstream_run": args.upstream_run,
        "downstream_run": args.downstream_run,
        "upstream_k": args.upstream_k,
        "downstream_k": args.downstream_k,
        "upstream_circuit_sha256": sha256_file(upstream_circuit_path),
        "downstream_circuit_sha256": sha256_file(downstream_circuit_path),
        "direction_sha256": unit_sha256,
        "response_vector_sha256": raw_vector_sha256,
        "unrelated_vector_sha256": (
            sha256_file(unrelated_path) if unrelated_path is not None else None
        ),
        "orthogonal_seed": args.orthogonal_seed,
        "upstream_position_scope": args.upstream_position_scope,
        "upstream_intervention_applied": _uses_upstream_intervention(
            args.condition
        ),
        "downstream_estimand": _downstream_estimand(args.condition),
        "interface": {
            "hook": target_hook,
            "layer": layer,
            "timing": args.interface_timing,
            "position": (
                "fixed final prompt token"
                if args.interface_timing == "prompt_end"
                else "current next-token predictor"
            ),
        },
        "interface_timing": args.interface_timing,
        "question_indices": args.question_indices,
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

    expected_ids = {
        (
            f"connected-ku{args.upstream_k}-kd{args.downstream_k}-"
            f"{args.condition}-q{source['question_index']:02d}"
        )
        for source in designed
    }
    unexpected_ids = set(rows_by_id) - expected_ids
    if unexpected_ids:
        raise ValueError(
            f"Resume artifact contains unexpected records: {sorted(unexpected_ids)[:3]}"
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
        prompt_end_position = clean_tokens.shape[1] - 1
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
            else "full_prefix"
            if args.condition == "upstream_full_prefix_state_rescue"
            else "orthogonal"
            if args.condition == "upstream_orthogonal_rescue"
            else "unrelated"
            if args.condition == "upstream_unrelated_vector_rescue"
            else None
        )
        union_circuit_only = args.condition == "both_circuit_only"
        apply_upstream_intervention = _uses_upstream_intervention(
            args.condition
        )
        downstream_ablated = args.condition in {
            "downstream_ablated",
            "both_ablated",
            "both_circuit_only",
            "vector_rescue_downstream_ablated",
        }
        downstream_estimand = _downstream_estimand(args.condition)
        initial_target_position = (
            prompt_end_position
            if args.interface_timing == "prompt_end"
            else -1
        )
        initial_position_mask = None
        if args.upstream_position_scope == "prompt_only":
            initial_position_mask = torch.ones_like(clean_mask, dtype=torch.bool)
        if apply_upstream_intervention:
            initial_corrupt, initial_corrupt_receivers = (
                upstream_corrupt_cache(
                    corrupt_tokens,
                    corrupt_mask,
                    circuit_only=union_circuit_only,
                )
            )
        else:
            initial_corrupt, initial_corrupt_receivers = {}, {}
        initial_clean_state = capture_target(
            clean_tokens,
            clean_mask,
            position=initial_target_position,
        )
        initial_clean_reference = (
            capture_prefix_interface(
                clean_tokens,
                clean_mask,
                prefix_length=clean_tokens.shape[1],
            )
            if rescue == "full_prefix"
            else initial_clean_state
        )
        initial_ablated_hooks = (
            upstream_hooks(
                initial_corrupt,
                corrupt_receivers=initial_corrupt_receivers,
                circuit_only=union_circuit_only,
                rescue=None,
                clean_reference=None,
                target_position=initial_target_position,
                prefix_length=clean_tokens.shape[1],
                position_mask=initial_position_mask,
            )
            if apply_upstream_intervention
            else []
        )
        initial_ablated_state = capture_target(
            clean_tokens,
            clean_mask,
            position=initial_target_position,
            fwd_hooks=initial_ablated_hooks,
        )
        initial_ablated_prefix_state = (
            capture_prefix_interface(
                clean_tokens,
                clean_mask,
                prefix_length=clean_tokens.shape[1],
                fwd_hooks=initial_ablated_hooks,
            )
            if rescue == "full_prefix"
            else None
        )
        initial_rescued_state = None
        initial_rescued_prefix_state = None
        if rescue is not None:
            initial_rescue_hooks = upstream_hooks(
                initial_corrupt,
                corrupt_receivers=initial_corrupt_receivers,
                circuit_only=union_circuit_only,
                rescue=rescue,
                clean_reference=initial_clean_reference,
                target_position=initial_target_position,
                prefix_length=clean_tokens.shape[1],
                position_mask=initial_position_mask,
            )
            initial_rescued_state = capture_target(
                clean_tokens,
                clean_mask,
                position=initial_target_position,
                fwd_hooks=initial_rescue_hooks,
            )
            if rescue == "full_prefix":
                initial_rescued_prefix_state = capture_prefix_interface(
                    clean_tokens,
                    clean_mask,
                    prefix_length=clean_tokens.shape[1],
                    fwd_hooks=initial_rescue_hooks,
                )

        def target_coordinate(state: torch.Tensor) -> float:
            direction = unit_direction.to(device=state.device).float()
            return float((state.float() * direction).sum(dim=-1).item())

        initial_interface = {
            "position": initial_target_position,
            "clean_target_coordinate": target_coordinate(initial_clean_state),
            "upstream_intervened_target_coordinate": target_coordinate(
                initial_ablated_state
            ),
            "missing_target_coordinate": target_coordinate(
                initial_clean_state
            )
            - target_coordinate(initial_ablated_state),
            "state_l2_distance_to_clean": float(
                (
                    initial_clean_state.float()
                    - initial_ablated_state.float()
                )
                .norm(dim=-1)
                .item()
            ),
            "rescued_target_coordinate": (
                target_coordinate(initial_rescued_state)
                if initial_rescued_state is not None
                else None
            ),
            "rescued_state_l2_distance_to_clean": (
                float(
                    (
                        initial_clean_state.float()
                        - initial_rescued_state.float()
                    )
                    .norm(dim=-1)
                    .item()
                )
                if initial_rescued_state is not None
                else None
            ),
            "prefix_state_l2_distance_to_clean": (
                float(
                    (
                        initial_clean_reference.float()
                        - initial_ablated_prefix_state.float()
                    )
                    .norm()
                    .item()
                )
                if initial_ablated_prefix_state is not None
                else None
            ),
            "rescued_prefix_state_l2_distance_to_clean": (
                float(
                    (
                        initial_clean_reference.float()
                        - initial_rescued_prefix_state.float()
                    )
                    .norm()
                    .item()
                )
                if initial_rescued_prefix_state is not None
                else None
            ),
        }
        for _ in tqdm(
            range(args.max_new_tokens),
            desc=f"{args.condition}-q{source['question_index']:02d}",
            leave=False,
        ):
            if apply_upstream_intervention:
                u_corrupt, u_corrupt_receivers = upstream_corrupt_cache(
                    corrupt_tokens,
                    corrupt_mask,
                    circuit_only=union_circuit_only,
                )
            else:
                u_corrupt, u_corrupt_receivers = {}, {}
            target_position = (
                prompt_end_position
                if args.interface_timing == "prompt_end"
                else -1
            )
            upstream_position_mask = None
            if args.upstream_position_scope == "prompt_only":
                positions = torch.arange(
                    clean_tokens.shape[1], device=clean_tokens.device
                )
                upstream_position_mask = (
                    positions <= prompt_end_position
                ).unsqueeze(0)
            clean_reference = None
            if rescue == "full_prefix":
                clean_reference = capture_prefix_interface(
                    clean_tokens,
                    clean_mask,
                    prefix_length=clean_tokens.shape[1],
                )
            elif rescue is not None:
                clean_reference = capture_target(
                    clean_tokens,
                    clean_mask,
                    position=target_position,
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
                    target_position=target_position,
                    prefix_length=clean_tokens.shape[1],
                    upstream_position_mask=upstream_position_mask,
                    downstream_estimand=downstream_estimand,
                    apply_upstream_intervention=apply_upstream_intervention,
                )
            else:
                hooks = (
                    upstream_hooks(
                        u_corrupt,
                        corrupt_receivers=u_corrupt_receivers,
                        circuit_only=False,
                        rescue=rescue,
                        clean_reference=clean_reference,
                        target_position=target_position,
                        prefix_length=clean_tokens.shape[1],
                        position_mask=upstream_position_mask,
                    )
                    if apply_upstream_intervention
                    else []
                )
                with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
                    logits = model(
                        clean_tokens, attention_mask=clean_mask
                    ).detach()
            next_token_logits = logits[0, -1]
            if not torch.isfinite(next_token_logits).all():
                raise FloatingPointError(
                    "Non-finite next-token logits during connected-circuit generation."
                )
            next_token = int(next_token_logits.argmax().item())
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
            "schema_version": 2,
            "record_id": record_id,
            "behavior_protocol_id": config["protocol_id"],
            "trait": config["trait"]["name"],
            "model": model_cfg["name"],
            "model_revision": model_cfg.get("revision"),
            "condition": args.condition,
            "upstream_k": args.upstream_k,
            "downstream_k": args.downstream_k,
            "upstream_position_scope": args.upstream_position_scope,
            "upstream_intervention_applied": apply_upstream_intervention,
            "interface_timing": args.interface_timing,
            "downstream_estimand": downstream_estimand,
            "orthogonal_seed": args.orthogonal_seed,
            "initial_interface_diagnostic": initial_interface,
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
        raise RuntimeError("Connected circuit behavior output is incomplete.")
    save_metadata("complete")
    print(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

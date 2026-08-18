#!/usr/bin/env python3
"""Learn reconstructively sufficient persona circuits with edge gates.

This is deliberately separate from EAP-IG ranking.  It optimizes the exact
circuit-only computation used at evaluation:

    corrupt receiver + gated clean-minus-corrupt parent contributions.

Upstream training reconstructs the layer-20 persona-vector coordinate at every
response-prediction position.  Downstream training starts from the natural
negative-persona layer-20 state, restores only that coordinate, and reconstructs
the resulting response-token logits.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_negative_prompt


def _linear_target(step: int, warmup_steps: int, final: float) -> float:
    if warmup_steps <= 0:
        return final
    return final * min((step + 1) / warmup_steps, 1.0)


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
    if "state" not in captured:
        raise RuntimeError(f"Interface hook did not fire: {target_hook}")
    return captured["state"], logits.detach()


def _capture_corrupt_caches_and_logits(
    model,
    construction,
    tokens,
    attention_mask,
    *,
    fwd_hooks=(),
    storage_dtype=None,
    source_cache_transform=None,
):
    import torch

    sources = {}
    receivers = {}
    hooks = [
        *fwd_hooks,
        *construction.source_cache_hooks(
            sources,
            detach=True,
            storage_dtype=storage_dtype,
            cache_transform=source_cache_transform,
        ),
        *construction.receiver_cache_hooks(
            receivers, storage_dtype=storage_dtype
        ),
    ]
    with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
        logits = model(tokens, attention_mask=attention_mask)
    missing_sources = set(construction.source_hooks) - set(sources)
    missing_receivers = set(construction.receiver_by_hook) - set(receivers)
    if missing_sources or missing_receivers:
        raise RuntimeError(
            "Counterfactual hooks did not all fire: "
            f"sources={sorted(missing_sources)}, "
            f"receivers={sorted(missing_receivers)}"
        )
    return sources, receivers, logits.detach()


def _gated_forward(
    model,
    construction,
    tokens,
    attention_mask,
    corrupt_sources,
    corrupt_receivers,
    gate_values,
    *,
    fwd_hooks=(),
    capture_hook: str | None = None,
    source_cache_transform=None,
):
    import torch

    mixed_sources = {}
    captured = {}

    def capture(activations: torch.Tensor, hook):
        del hook
        captured["state"] = activations
        return activations

    hooks = [
        *fwd_hooks,
        *construction.source_cache_hooks(
            mixed_sources,
            detach=False,
            cache_transform=source_cache_transform,
        ),
        *construction.construction_hooks(
            corrupt_sources,
            mixed_sources,
            corrupt_receivers,
            gate_values,
        ),
    ]
    if capture_hook is not None:
        hooks.append((capture_hook, capture))
    with model.hooks(fwd_hooks=hooks):
        logits = model(tokens, attention_mask=attention_mask)
    if capture_hook is not None and "state" not in captured:
        raise RuntimeError(f"Gated capture hook did not fire: {capture_hook}")
    return logits, captured.get("state")


def _masked_projection(state, direction, mask):
    import torch

    vector = direction.to(device=state.device, dtype=torch.float32)
    vector = vector / vector.norm().clamp_min(1e-8)
    projections = torch.einsum("bpd,d->bp", state.float(), vector)
    return projections[mask.to(device=state.device, dtype=torch.bool)]


def _masked_kl(student_logits, teacher_logits, prediction_mask):
    import torch
    import torch.nn.functional as functional

    mask = prediction_mask.to(
        device=student_logits.device, dtype=torch.bool
    )
    student = student_logits.float()[mask].log_softmax(dim=-1)
    teacher = teacher_logits.float()[mask].log_softmax(dim=-1)
    if student.shape[0] == 0:
        raise ValueError("The response-prediction mask is empty.")
    return functional.kl_div(
        student, teacher, reduction="sum", log_target=True
    ) / student.shape[0]


def _load_examples(config: dict, manifest_path: Path):
    from refusal_circuit.persona import read_jsonl, sha256_file

    intermediate = ROOT / config["paths"]["intermediate_root"]
    official_root = ROOT / config["paths"]["official_root"]
    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "frozen"
        or manifest.get("protocol")
        not in {"shared_prompt_v1", "shared_prompt_full_transcript_v2"}
        or manifest["sources"]["positive_sha256"] != sha256_file(positive_path)
        or manifest["sources"]["negative_sha256"] != sha256_file(negative_path)
    ):
        raise ValueError("Shared circuit manifest failed provenance.")

    positive_by_record = {
        row["record_id"]: row for row in read_jsonl(positive_path)
    }
    negative_by_record = {
        row["record_id"]: row for row in read_jsonl(negative_path)
    }
    trait_data = json.loads(
        (official_root / "extract.json").read_text(encoding="utf-8")
    )
    return (
        manifest,
        positive_by_record,
        negative_by_record,
        trait_data,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--stage", choices=("upstream", "downstream"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--train-prompts", type=int, default=48)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--target-sparsity", type=float, default=None)
    parser.add_argument("--sparsity-warmup-steps", type=int, default=500)
    parser.add_argument("--edge-learning-rate", type=float, default=0.1)
    parser.add_argument("--regularization-learning-rate", type=float, default=0.1)
    parser.add_argument("--initial-log-alpha", type=float, default=5.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--no-precompute-interface-states",
        action="store_true",
        help=(
            "Recompute clean/corrupt layer-20 states every optimization step. "
            "The default caches these frozen states once on CPU."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run one all-open/all-closed reconstruction identity check and "
            "exit without training or writing artifacts."
        ),
    )
    parser.add_argument("--preflight-sample-index", type=int, default=0)
    parser.add_argument(
        "--preflight-tolerance",
        type=float,
        default=1e-3,
        help="Maximum normalized reconstruction error at an algebra endpoint.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=(500, 1000, 3000, 5000, 10000, 20000, 50000),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("--run-name must be filesystem-safe.")
    if not 0 < args.train_prompts < args.num_prompts:
        raise ValueError("--train-prompts must leave a nonempty validation split.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.sparsity_warmup_steps < 0:
        raise ValueError("--sparsity-warmup-steps cannot be negative.")
    target_sparsity = (
        args.target_sparsity
        if args.target_sparsity is not None
        else (0.95 if args.stage == "upstream" else 0.99)
    )
    if not 0 <= target_sparsity < 1:
        raise ValueError("--target-sparsity must be in [0, 1).")

    import torch
    from torch.optim import AdamW

    from refusal_circuit.eapig import _response_mask, response_prediction_mask
    from refusal_circuit.edge_pruning import (
        EdgeGateSet,
        VectorizedEdgeConstruction,
        complete_path_edge_names,
        fixed_state_hook,
        vector_only_rescue_state,
    )
    from refusal_circuit.graph import (
        circuit_connectivity,
        edge_record,
        make_projection_graph,
        make_steering_to_logits_graph,
    )
    from refusal_circuit.model import (
        load_model,
        tokenize_prompts,
        validate_model_architecture,
    )
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    trait_slug = config["trait"]["name"]
    result_root = ROOT / config["paths"]["results_root"]
    output = (
        result_root
        / "circuit_discovery_edge_pruning"
        / args.stage
        / args.run_name
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results.")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        args.sample_manifest
        if args.sample_manifest.is_absolute()
        else ROOT / args.sample_manifest
    )
    manifest, positive_by_record, negative_by_record, trait_data = _load_examples(
        config, manifest_path
    )
    samples = manifest["discovery"]["samples"]
    if len(samples) != args.num_prompts:
        raise ValueError(
            f"Manifest has {len(samples)} samples, expected {args.num_prompts}."
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
    model.set_ungroup_grouped_query_attention(True)
    target_layer = int(config["vector"]["paper_layer"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    expected_hook = f"blocks.{target_layer}.hook_resid_pre"
    if target_hook != expected_hook:
        raise ValueError(f"Expected interface hook {expected_hook}, got {target_hook}.")

    unit_path = result_root / f"vector/{trait_slug}_direction.pt"
    direction = torch.load(
        unit_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    ).float()
    direction = direction / direction.norm().clamp_min(1e-8)

    examples = []
    for sample_index, sample in enumerate(samples):
        positive = positive_by_record[sample["positive_record_id"]]
        negative = negative_by_record[sample["negative_record_id"]]
        if (
            positive["pair_id"] != sample["pair_id"]
            or negative["pair_id"] != sample["pair_id"]
            or positive["prompt"] != sample["positive_prompt"]
            or negative["prompt"] != sample["negative_prompt"]
        ):
            raise ValueError(f"Manifest row {sample_index} changed.")
        question_index = int(sample["question_index"])
        instruction_index = int(sample["instruction_index"])
        aligned_negative, alignment = _align_negative_prompt(
            model.tokenizer,
            question=trait_data["questions"][question_index],
            negative_assistant_name=config["trait"]["negative_assistant_name"],
            negative_instruction=trait_data["instruction"][instruction_index]["neg"],
            positive_prompt=positive["prompt"],
            expected_negative_prompt=negative["prompt"],
        )
        response_start = len(
            model.tokenizer.encode(
                positive["prompt"], add_special_tokens=False
            )
        )
        transcripts = {}
        for answer_label, answer in (
            ("positive", positive["answer"]),
            ("negative", negative["answer"]),
        ):
            clean_text = positive["prompt"] + answer
            corrupt_text = aligned_negative + answer
            clean_length = len(
                model.tokenizer.encode(clean_text, add_special_tokens=False)
            )
            corrupt_length = len(
                model.tokenizer.encode(corrupt_text, add_special_tokens=False)
            )
            if clean_length != corrupt_length:
                raise ValueError(
                    f"Transcript pair {sample_index}/{answer_label} is not "
                    f"token aligned: {clean_length} != {corrupt_length}."
                )
            if clean_length > args.max_length:
                raise ValueError(
                    f"Transcript pair {sample_index}/{answer_label} exceeds "
                    "--max-length."
                )
            transcripts[answer_label] = {
                "clean_text": clean_text,
                "corrupt_text": corrupt_text,
                "response_start": response_start,
                "tokens": clean_length,
            }
        examples.append(
            {
                "sample_index": sample_index,
                "question_index": question_index,
                "instruction_index": instruction_index,
                "pair_id": sample["pair_id"],
                "positive_record_id": positive["record_id"],
                "transcripts": transcripts,
                "alignment": alignment,
            }
        )

    graph = (
        make_projection_graph(model, target_layer)
        if args.stage == "upstream"
        else make_steering_to_logits_graph(model, target_layer)
    )
    gates = EdgeGateSet(
        graph.edges,
        initial_log_alpha=args.initial_log_alpha,
        initialization_noise=0.01,
    ).to(next(model.parameters()).device)
    gates.train()
    construction = VectorizedEdgeConstruction(graph, gates)
    compact_dtype = next(model.parameters()).dtype

    if not 0 <= args.preflight_sample_index < len(examples):
        raise IndexError("--preflight-sample-index is out of range.")
    if args.preflight_tolerance <= 0:
        raise ValueError("--preflight-tolerance must be positive.")

    if args.preflight_only:
        example = examples[args.preflight_sample_index]
        transcript = example["transcripts"]["positive"]
        clean_tokens, clean_mask = tokenize_prompts(
            model, [transcript["clean_text"]], max_length=args.max_length
        )
        corrupt_tokens, corrupt_mask = tokenize_prompts(
            model, [transcript["corrupt_text"]], max_length=args.max_length
        )
        response_mask = _response_mask(
            clean_tokens, clean_mask, [transcript["response_start"]]
        )
        prediction_mask = response_prediction_mask(response_mask)
        all_open = torch.ones(
            len(graph.edges), device=next(model.parameters()).device
        )
        all_closed = torch.zeros_like(all_open)

        if args.stage == "upstream":
            corrupt_sources, corrupt_receivers, _ = (
                _capture_corrupt_caches_and_logits(
                    model,
                    construction,
                    corrupt_tokens,
                    corrupt_mask,
                    storage_dtype=compact_dtype,
                )
            )
            clean_state, _ = _capture_state(
                model, clean_tokens, clean_mask, target_hook
            )
            corrupt_state = corrupt_receivers[target_hook]
            with torch.inference_mode():
                _, open_state = _gated_forward(
                    model,
                    construction,
                    clean_tokens,
                    clean_mask,
                    corrupt_sources,
                    corrupt_receivers,
                    all_open,
                    capture_hook=target_hook,
                )
                _, closed_state = _gated_forward(
                    model,
                    construction,
                    clean_tokens,
                    clean_mask,
                    corrupt_sources,
                    corrupt_receivers,
                    all_closed,
                    capture_hook=target_hook,
                )
            clean_projection = _masked_projection(
                clean_state, direction, prediction_mask
            )
            corrupt_projection = _masked_projection(
                corrupt_state, direction, prediction_mask
            )
            denominator = (
                (clean_projection - corrupt_projection).square().mean()
                .clamp_min(1e-8)
            )
            open_error = float(
                (
                    (
                        _masked_projection(
                            open_state, direction, prediction_mask
                        )
                        - clean_projection
                    )
                    .square()
                    .mean()
                    / denominator
                ).item()
            )
            closed_error = float(
                (
                    (
                        _masked_projection(
                            closed_state, direction, prediction_mask
                        )
                        - corrupt_projection
                    )
                    .square()
                    .mean()
                    / denominator
                ).item()
            )
            baseline_gap = float(denominator.item())
        else:
            clean_state, _ = _capture_state(
                model, clean_tokens, clean_mask, target_hook
            )
            corrupt_state, _ = _capture_state(
                model, corrupt_tokens, corrupt_mask, target_hook
            )
            interface_state = vector_only_rescue_state(
                clean_state,
                corrupt_state,
                direction,
                prediction_mask,
            ).detach()
            corrupt_hooks = [
                (target_hook, fixed_state_hook(corrupt_state))
            ]

            def corrupt_source_transform(name: str, activations: torch.Tensor):
                return (
                    corrupt_state.to(
                        device=activations.device, dtype=activations.dtype
                    )
                    if name == target_hook
                    else activations
                )

            def interface_source_transform(name: str, activations: torch.Tensor):
                return (
                    interface_state.to(
                        device=activations.device, dtype=activations.dtype
                    )
                    if name == target_hook
                    else activations
                )

            corrupt_sources, corrupt_receivers, corrupt_logits = (
                _capture_corrupt_caches_and_logits(
                    model,
                    construction,
                    clean_tokens,
                    clean_mask,
                    fwd_hooks=corrupt_hooks,
                    storage_dtype=compact_dtype,
                    source_cache_transform=corrupt_source_transform,
                )
            )
            interface_hooks = [
                (target_hook, fixed_state_hook(interface_state))
            ]
            with torch.inference_mode(), model.hooks(
                fwd_hooks=interface_hooks
            ):
                teacher_logits = model(
                    clean_tokens, attention_mask=clean_mask
                ).detach()
            with torch.inference_mode():
                open_logits, _ = _gated_forward(
                    model,
                    construction,
                    clean_tokens,
                    clean_mask,
                    corrupt_sources,
                    corrupt_receivers,
                    all_open,
                    fwd_hooks=interface_hooks,
                    source_cache_transform=interface_source_transform,
                )
                closed_logits, _ = _gated_forward(
                    model,
                    construction,
                    clean_tokens,
                    clean_mask,
                    corrupt_sources,
                    corrupt_receivers,
                    all_closed,
                    fwd_hooks=interface_hooks,
                    source_cache_transform=interface_source_transform,
                )
            denominator = _masked_kl(
                corrupt_logits, teacher_logits, prediction_mask
            ).clamp_min(1e-8)
            open_error = float(
                (
                    _masked_kl(
                        open_logits, teacher_logits, prediction_mask
                    )
                    / denominator
                ).item()
            )
            closed_error = float(
                (
                    _masked_kl(
                        closed_logits, corrupt_logits, prediction_mask
                    )
                    / denominator
                ).item()
            )
            baseline_gap = float(denominator.item())

        result = {
            "status": "pass"
            if max(open_error, closed_error) <= args.preflight_tolerance
            else "fail",
            "stage": args.stage,
            "sample_index": example["sample_index"],
            "all_open_normalized_error": open_error,
            "all_closed_normalized_error": closed_error,
            "baseline_gap": baseline_gap,
            "tolerance": args.preflight_tolerance,
            "eligible_edges": len(graph.edges),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
        }
        print(json.dumps(result, indent=2))
        if result["status"] != "pass":
            raise RuntimeError(
                "Differentiable circuit algebra failed its identity preflight."
            )
        return

    # Bhaskar et al.'s constrained L0 objective uses learned linear and
    # quadratic multipliers.  Gate parameters minimize the total objective;
    # the multipliers maximize it, enforcing the scheduled target sparsity.
    lambda_linear = torch.nn.Parameter(
        torch.zeros((), device=next(model.parameters()).device)
    )
    lambda_quadratic = torch.nn.Parameter(
        torch.zeros((), device=next(model.parameters()).device)
    )
    optimizer = AdamW(
        [
            {
                "params": [gates.log_alpha],
                "lr": args.edge_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": [lambda_linear, lambda_quadratic],
                "lr": args.regularization_learning_rate,
                "weight_decay": 0.0,
                "maximize": True,
            },
        ]
    )

    train_examples = examples[: args.train_prompts]
    validation_examples = examples[args.train_prompts :]
    state_cache = {}
    if not args.no_precompute_interface_states:
        cache_specs = [
            (row, answer_label)
            for row in train_examples
            for answer_label in (
                ("positive",)
                if args.stage == "upstream"
                else ("positive", "negative")
            )
        ]
        for row, answer_label in cache_specs:
            transcript = row["transcripts"][answer_label]
            clean_tokens, clean_mask = tokenize_prompts(
                model,
                [transcript["clean_text"]],
                max_length=args.max_length,
            )
            clean_state, _ = _capture_state(
                model, clean_tokens, clean_mask, target_hook
            )
            cached = {"clean": clean_state.cpu()}
            if args.stage == "downstream":
                corrupt_tokens, corrupt_mask = tokenize_prompts(
                    model,
                    [transcript["corrupt_text"]],
                    max_length=args.max_length,
                )
                corrupt_state, _ = _capture_state(
                    model, corrupt_tokens, corrupt_mask, target_hook
                )
                cached["corrupt"] = corrupt_state.cpu()
            state_cache[(row["sample_index"], answer_label)] = cached
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    training_log = []
    training_items = [
        (row, answer_label)
        for row in train_examples
        for answer_label in (
            ("positive",)
            if args.stage == "upstream"
            else ("positive", "negative")
        )
    ]
    rng = random.Random(args.seed)

    for step in range(args.steps):
        if step % len(training_items) == 0:
            rng.shuffle(training_items)
        example, answer_label = training_items[
            step % len(training_items)
        ]
        transcript = example["transcripts"][answer_label]
        clean_tokens, clean_mask = tokenize_prompts(
            model, [transcript["clean_text"]], max_length=args.max_length
        )
        corrupt_tokens, corrupt_mask = tokenize_prompts(
            model, [transcript["corrupt_text"]], max_length=args.max_length
        )
        if (
            clean_tokens.shape != corrupt_tokens.shape
            or not torch.equal(clean_mask, corrupt_mask)
        ):
            raise ValueError("A frozen transcript pair lost token alignment.")
        response_mask = _response_mask(
            clean_tokens, clean_mask, [transcript["response_start"]]
        )
        prediction_mask = response_prediction_mask(response_mask)

        optimizer.zero_grad(set_to_none=True)
        if args.stage == "upstream":
            corrupt_sources, corrupt_receivers, _ = (
                _capture_corrupt_caches_and_logits(
                    model,
                    construction,
                    corrupt_tokens,
                    corrupt_mask,
                    storage_dtype=compact_dtype,
                )
            )
            cached = state_cache.get(
                (example["sample_index"], answer_label)
            )
            if cached is None:
                clean_state, _ = _capture_state(
                    model, clean_tokens, clean_mask, target_hook
                )
            else:
                clean_state = cached["clean"].to(
                    device=clean_tokens.device
                )
            corrupt_state = corrupt_receivers[target_hook]
            gate_values = gates(sample=True)
            _, gated_state = _gated_forward(
                model,
                construction,
                clean_tokens,
                clean_mask,
                corrupt_sources,
                corrupt_receivers,
                gate_values,
                capture_hook=target_hook,
            )
            clean_projection = _masked_projection(
                clean_state, direction, prediction_mask
            )
            corrupt_projection = _masked_projection(
                corrupt_state, direction, prediction_mask
            )
            gated_projection = _masked_projection(
                gated_state, direction, prediction_mask
            )
            denominator = (
                (clean_projection - corrupt_projection).square().mean()
                .detach()
                .clamp_min(1e-8)
            )
            task_loss = (
                (gated_projection - clean_projection).square().mean()
                / denominator
            )
            baseline_gap = float(denominator.item())
        else:
            cached = state_cache.get(
                (example["sample_index"], answer_label)
            )
            if cached is None:
                clean_state, _ = _capture_state(
                    model, clean_tokens, clean_mask, target_hook
                )
                corrupt_state, _ = _capture_state(
                    model, corrupt_tokens, corrupt_mask, target_hook
                )
            else:
                clean_state = cached["clean"].to(
                    device=clean_tokens.device
                )
                corrupt_state = cached["corrupt"].to(
                    device=clean_tokens.device
                )
            interface_state = vector_only_rescue_state(
                clean_state,
                corrupt_state,
                direction,
                prediction_mask,
            ).detach()
            corrupt_interface_hooks = [
                (target_hook, fixed_state_hook(corrupt_state))
            ]

            def corrupt_source_transform(name: str, activations: torch.Tensor):
                return (
                    corrupt_state.to(
                        device=activations.device, dtype=activations.dtype
                    )
                    if name == target_hook
                    else activations
                )

            def interface_source_transform(name: str, activations: torch.Tensor):
                return (
                    interface_state.to(
                        device=activations.device, dtype=activations.dtype
                    )
                    if name == target_hook
                    else activations
                )

            corrupt_sources, corrupt_receivers, corrupt_logits = (
                _capture_corrupt_caches_and_logits(
                    model,
                    construction,
                    clean_tokens,
                    clean_mask,
                    fwd_hooks=corrupt_interface_hooks,
                    storage_dtype=compact_dtype,
                    source_cache_transform=corrupt_source_transform,
                )
            )
            interface_hooks = [
                (target_hook, fixed_state_hook(interface_state))
            ]
            with torch.inference_mode(), model.hooks(
                fwd_hooks=interface_hooks
            ):
                teacher_logits = model(
                    clean_tokens, attention_mask=clean_mask
                ).detach()
            gate_values = gates(sample=True)
            gated_logits, _ = _gated_forward(
                model,
                construction,
                clean_tokens,
                clean_mask,
                corrupt_sources,
                corrupt_receivers,
                gate_values,
                fwd_hooks=interface_hooks,
                source_cache_transform=interface_source_transform,
            )
            denominator = _masked_kl(
                corrupt_logits, teacher_logits, prediction_mask
            ).detach().clamp_min(1e-8)
            task_loss = (
                _masked_kl(gated_logits, teacher_logits, prediction_mask)
                / denominator
            )
            baseline_gap = float(denominator.item())

        current_target = _linear_target(
            step, args.sparsity_warmup_steps, target_sparsity
        )
        model_sparsity = gates.expected_sparsity()
        constraint = model_sparsity - current_target
        regularization = (
            lambda_linear * constraint
            + lambda_quadratic * constraint.square()
        )
        loss = task_loss + regularization
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite pruning loss at step {step}: {loss.item()}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [gates.log_alpha], args.gradient_clip
        )
        optimizer.step()

        record = {
            "step": step + 1,
            "sample_index": example["sample_index"],
            "answer_label": answer_label,
            "task_loss": float(task_loss.detach().item()),
            "total_loss": float(loss.detach().item()),
            "baseline_gap": baseline_gap,
            "expected_sparsity": float(
                gates.expected_sparsity().detach().item()
            ),
            "target_sparsity": current_target,
            "lambda_linear": float(lambda_linear.detach().item()),
            "lambda_quadratic": float(lambda_quadratic.detach().item()),
        }
        training_log.append(record)
        if (
            step == 0
            or (step + 1) % args.log_every == 0
            or step + 1 == args.steps
        ):
            print(json.dumps(record), flush=True)

    gates.eval()
    checkpoint_path = output / "edge_gates.pt"
    torch.save(
        {
            "edge_names": gates.edge_names,
            "log_alpha": gates.log_alpha.detach().cpu(),
            "stage": args.stage,
            "target_sparsity": target_sparsity,
            "steps": args.steps,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    score_by_name = gates.edge_score_dict()
    score_path = output / "edge_gate_scores.jsonl"
    with score_path.open("w", encoding="utf-8") as handle:
        for name in gates.edge_names:
            handle.write(
                json.dumps(
                    {"edge": name, "gate_score": score_by_name[name]},
                    sort_keys=True,
                )
                + "\n"
            )
    log_path = output / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        for row in training_log:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    for edge in graph.edges.values():
        edge.score = score_by_name[edge.name]
    circuit_paths = []
    for requested_k in sorted(set(args.k_values)):
        if requested_k <= 0 or requested_k > len(graph.edges):
            continue
        ranked = sorted(
            graph.edges.values(),
            key=lambda edge: (-float(edge.score), edge.name),
        )
        top_names = {edge.name for edge in ranked[:requested_k]}
        retained_names = complete_path_edge_names(graph, top_names)
        retained = [
            graph.edges[name] for name in sorted(retained_names)
        ]
        path = output / "circuits_source_to_sink" / (
            f"edge_pruned_top_{requested_k}.json"
        )
        write_json(
            path,
            {
                "schema_version": 1,
                "method": "hard_concrete_edge_pruning",
                "circuit_family": "source_to_sink",
                "stage": args.stage,
                "selection": (
                    "global deterministic gate top-k followed by exact "
                    "source-to-sink pruning"
                ),
                "requested_k": requested_k,
                "retained_edges": len(retained),
                "connectivity": circuit_connectivity(
                    graph, retained_names
                ),
                "score_source": str(score_path.relative_to(ROOT)),
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "objective": (
                    "response-prediction-position persona-vector coordinate reconstruction"
                    if args.stage == "upstream"
                    else "KL reconstruction of vector-only interface-rescue logits"
                ),
                "interface": {
                    "hook": target_hook,
                    "layer": target_layer,
                    "direction": str(unit_path.relative_to(ROOT)),
                    "timing": "all response-prediction positions",
                },
                "edges": [edge_record(edge) for edge in retained],
            },
        )
        circuit_paths.append(path)

    metadata = {
        "schema_version": 1,
        "status": "complete",
        "method": "hard-concrete edge pruning over exact circuit-only algebra",
        "stage": args.stage,
        "claim_scope": (
            "candidate reconstructive source-to-interface circuit"
            if args.stage == "upstream"
            else "candidate reconstructive interface-to-output circuit"
        ),
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "trait": trait_slug,
        "sample_manifest": str(manifest_path.relative_to(ROOT)),
        "sample_manifest_sha256": sha256_file(manifest_path),
        "train_sample_indices": [
            row["sample_index"] for row in train_examples
        ],
        "validation_sample_indices": [
            row["sample_index"] for row in validation_examples
        ],
        "objective": {
            "counterfactual": "token-aligned natural negative-persona transcript",
            "shared_teacher_forced_suffix": True,
            "interface": target_hook,
            "position_scope": "response-prediction positions",
            "upstream_loss": "normalized per-position persona-coordinate MSE",
            "downstream_loss": (
                "normalized response-position KL from a vector-only rescued "
                "natural-negative interface"
            ),
            "orthogonal_clean_minus_corrupt_interface_components_removed": (
                args.stage == "downstream"
            ),
        },
        "optimization": {
            "steps": args.steps,
            "seed": args.seed,
            "target_sparsity": target_sparsity,
            "sparsity_warmup_steps": args.sparsity_warmup_steps,
            "edge_learning_rate": args.edge_learning_rate,
            "regularization_learning_rate": args.regularization_learning_rate,
            "initial_log_alpha": args.initial_log_alpha,
            "gradient_clip": args.gradient_clip,
            "precomputed_interface_states": (
                not args.no_precompute_interface_states
            ),
            "trainable_model_weights": 0,
            "trainable_edge_gates": len(graph.edges),
        },
        "final": training_log[-1],
        "outputs": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "scores": str(score_path.relative_to(ROOT)),
            "scores_sha256": sha256_file(score_path),
            "training_log": str(log_path.relative_to(ROOT)),
            "training_log_sha256": sha256_file(log_path),
            "circuits": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in circuit_paths
            ],
        },
        "max_length": args.max_length,
        "max_observed_tokens": max(
            transcript["tokens"]
            for row in examples
            for transcript in row["transcripts"].values()
        ),
        "truncation": False,
        "held_out_evaluation_required": True,
        "behavioral_generation_required": True,
    }
    write_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

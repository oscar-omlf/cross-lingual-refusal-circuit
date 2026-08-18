#!/usr/bin/env python3
"""Extract the independent interface and run the complete Phase 1 oracle."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import _align_system_prompt_pair


ORACLE_BACKEND_ID = "residual_only_native_gqa_v1"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _capture_state(model, tokens, attention_mask, target_hook: str):
    import torch

    captured = {}

    def hook(activations: torch.Tensor, hook):
        del hook
        captured["state"] = activations.detach().clone()
        return activations

    with torch.inference_mode(), model.hooks(fwd_hooks=[(target_hook, hook)]):
        logits = model(tokens, attention_mask=attention_mask)
    return captured["state"], logits[:, -1].detach().clone()


def _fixed_state_logits(
    model,
    tokens,
    attention_mask,
    target_hook: str,
    state,
):
    import torch

    from refusal_circuit.edge_pruning import fixed_state_hook

    with torch.inference_mode(), model.hooks(
        fwd_hooks=[(target_hook, fixed_state_hook(state))]
    ):
        logits = model(tokens, attention_mask=attention_mask)
        return logits[:, -1].detach().clone()


def _response_average_state(model, row: dict, target_hook: str):
    import torch

    prompt = row["prompt"]
    answer = row["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Empty answer in {row['record_id']}.")
    full_text = prompt + answer
    prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = model.tokenizer.encode(full_text, add_special_tokens=False)
    if not 0 < len(prompt_ids) < len(full_ids):
        raise ValueError(f"Invalid response boundary in {row['record_id']}.")
    tokens = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    attention_mask = torch.ones_like(tokens)
    state, _ = _capture_state(model, tokens, attention_mask, target_hook)
    return state[0, len(prompt_ids) :].float().mean(dim=0).cpu()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _tokenizer_identity(tokenizer) -> dict:
    from refusal_circuit.causal_interface import canonical_json_sha256

    return {
        "class": type(tokenizer).__name__,
        "name_or_path": str(tokenizer.name_or_path),
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
        "chat_template_sha256": canonical_json_sha256(
            getattr(tokenizer, "chat_template", None)
        ),
    }


def _microbatch_slices(length: int, size: int) -> list[tuple[int, int]]:
    if length < 0:
        raise ValueError("Microbatch length cannot be negative.")
    if size <= 0:
        raise ValueError("Microbatch size must be positive.")
    return [
        (start, min(start + size, length))
        for start in range(0, length, size)
    ]


def _configure_residual_only_forward(model) -> dict[str, bool]:
    settings = {
        "use_split_qkv_input": False,
        "use_attn_result": False,
        "ungroup_grouped_query_attention": False,
    }
    for attribute, value in settings.items():
        setter = getattr(model, f"set_{attribute}", None)
        if callable(setter):
            setter(value)
        elif hasattr(model.cfg, attribute):
            setattr(model.cfg, attribute, value)
        else:
            raise RuntimeError(
                f"TransformerLens does not expose required setting {attribute}."
            )
    if bool(getattr(model.cfg, "use_attn_in", False)):
        raise RuntimeError(
            "Residual-only GQA backend requires use_attn_in=False."
        )
    observed = {
        attribute: bool(getattr(model.cfg, attribute))
        for attribute in settings
    }
    if observed != settings:
        raise RuntimeError(
            f"Residual-only forward settings did not apply: {observed}."
        )
    return {**observed, "use_attn_in": False}


def _condition_specs() -> list[dict]:
    from refusal_circuit.causal_interface import PHASE1_RANKS, PHASE1_TIMINGS

    specs = [
        {"condition": "normal_positive", "component": "positive"},
        {"condition": "negative_baseline", "component": "negative"},
    ]
    for timing in PHASE1_TIMINGS:
        specs.append(
            {
                "condition": f"full_rank_{timing}",
                "component": "full",
                "timing": timing,
            }
        )
        for rank in PHASE1_RANKS:
            specs.extend(
                [
                    {
                        "condition": f"persona_rank{rank}_{timing}",
                        "component": "persona",
                        "timing": timing,
                        "rank": rank,
                    },
                    {
                        "condition": f"complement_rank{rank}_{timing}",
                        "component": "complement",
                        "timing": timing,
                        "rank": rank,
                    },
                ]
            )
    return specs


def _append_token(stream: dict, token_id: int, eos_id: int | None) -> None:
    import torch

    token = torch.tensor(
        [[token_id]],
        device=stream["clean_tokens"].device,
        dtype=stream["clean_tokens"].dtype,
    )
    one = torch.ones(
        (1, 1),
        device=stream["clean_mask"].device,
        dtype=stream["clean_mask"].dtype,
    )
    stream["clean_tokens"] = torch.cat((stream["clean_tokens"], token), dim=1)
    stream["corrupt_tokens"] = torch.cat(
        (stream["corrupt_tokens"], token.clone()), dim=1
    )
    stream["clean_mask"] = torch.cat((stream["clean_mask"], one), dim=1)
    stream["corrupt_mask"] = torch.cat(
        (stream["corrupt_mask"], one.clone()), dim=1
    )
    stream["generated"].append(int(token_id))
    if eos_id is not None and token_id == eos_id:
        stream["finished"] = True
        stream["termination_reason"] = "eos"


def _run_identity_preflight(
    model,
    *,
    clean_tokens,
    clean_mask,
    corrupt_tokens,
    corrupt_mask,
    target_hook: str,
    max_new_tokens: int,
) -> dict:
    import torch

    normal = []
    replaced = []
    eos_id = model.tokenizer.eos_token_id
    for step in range(max_new_tokens):
        clean_state, clean_logits = _capture_state(
            model, clean_tokens, clean_mask, target_hook
        )
        corrupt_state, _ = _capture_state(
            model, corrupt_tokens, corrupt_mask, target_hook
        )
        replaced_logits = _fixed_state_logits(
            model,
            corrupt_tokens,
            corrupt_mask,
            target_hook,
            clean_state,
        )
        clean_next = int(clean_logits[0].argmax().item())
        replaced_next = int(replaced_logits[0].argmax().item())
        normal.append(clean_next)
        replaced.append(replaced_next)
        if clean_next != replaced_next:
            raise RuntimeError(
                "Exact full-state identity failed at "
                f"step {step}: {clean_next} != {replaced_next}."
            )
        token = torch.tensor(
            [[clean_next]],
            device=clean_tokens.device,
            dtype=clean_tokens.dtype,
        )
        one = torch.ones(
            (1, 1), device=clean_mask.device, dtype=clean_mask.dtype
        )
        clean_tokens = torch.cat((clean_tokens, token), dim=1)
        corrupt_tokens = torch.cat((corrupt_tokens, token.clone()), dim=1)
        clean_mask = torch.cat((clean_mask, one), dim=1)
        corrupt_mask = torch.cat((corrupt_mask, one.clone()), dim=1)
        del corrupt_state
        if eos_id is not None and clean_next == eos_id:
            break
    return {
        "passed": True,
        "tokens_compared": len(normal),
        "normal_positive_token_ids": normal,
        "full_state_replacement_token_ids": replaced,
        "exact_token_agreement": 1.0,
        "terminated_with_eos": bool(normal and normal[-1] == eos_id),
    }


def _run_question_grid(
    model,
    *,
    question_index: int,
    question: str,
    instruction_index: int,
    positive_instruction: str,
    negative_instruction: str,
    positive_system: str,
    negative_system: str,
    basis,
    target_hook: str,
    max_new_tokens: int,
    max_length: int,
    condition_microbatch_size: int,
) -> tuple[list[dict], dict]:
    import torch

    from refusal_circuit.causal_interface import (
        projection_diagnostics,
        timed_interface_state,
        timing_mask,
    )
    from refusal_circuit.model import tokenize_prompts
    from refusal_circuit.persona import token_ids_sha256

    positive_prompt, negative_prompt, alignment = _align_system_prompt_pair(
        model.tokenizer,
        question=question,
        positive_system_prompt=positive_system,
        negative_system_prompt=negative_system,
    )
    clean_tokens, clean_mask = tokenize_prompts(
        model, [positive_prompt], max_length=max_length
    )
    corrupt_tokens, corrupt_mask = tokenize_prompts(
        model, [negative_prompt], max_length=max_length
    )
    if (
        clean_tokens.shape != corrupt_tokens.shape
        or not torch.equal(clean_mask, corrupt_mask)
    ):
        raise ValueError("Two-sided aligned prompt pair lost token parity.")
    prompt_end = clean_tokens.shape[1] - 1
    streams = []
    for spec in _condition_specs():
        streams.append(
            {
                **spec,
                "clean_tokens": clean_tokens.clone(),
                "clean_mask": clean_mask.clone(),
                "corrupt_tokens": corrupt_tokens.clone(),
                "corrupt_mask": corrupt_mask.clone(),
                "generated": [],
                "finished": False,
                "termination_reason": "token_cap",
                "diagnostics": [],
            }
        )

    eos_id = model.tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        active = [stream for stream in streams if not stream["finished"]]
        if not active:
            break
        for start, end in _microbatch_slices(
            len(active), condition_microbatch_size
        ):
            microbatch = active[start:end]
            clean_batch = torch.cat(
                [stream["clean_tokens"] for stream in microbatch], dim=0
            )
            clean_batch_mask = torch.cat(
                [stream["clean_mask"] for stream in microbatch], dim=0
            )
            corrupt_batch = torch.cat(
                [stream["corrupt_tokens"] for stream in microbatch], dim=0
            )
            corrupt_batch_mask = torch.cat(
                [stream["corrupt_mask"] for stream in microbatch], dim=0
            )
            clean_state, clean_logits = _capture_state(
                model, clean_batch, clean_batch_mask, target_hook
            )
            corrupt_state, corrupt_logits = _capture_state(
                model, corrupt_batch, corrupt_batch_mask, target_hook
            )

            intervention_indices = [
                index
                for index, stream in enumerate(microbatch)
                if stream["component"] not in {"positive", "negative"}
            ]
            intervention_logits = None
            index_to_intervention = {}
            if intervention_indices:
                states = []
                tokens = []
                masks = []
                for output_index, microbatch_index in enumerate(
                    intervention_indices
                ):
                    stream = microbatch[microbatch_index]
                    delta = clean_state[
                        microbatch_index : microbatch_index + 1
                    ] - corrupt_state[microbatch_index : microbatch_index + 1]
                    mask = timing_mask(
                        corrupt_batch_mask[
                            microbatch_index : microbatch_index + 1
                        ],
                        prompt_end=prompt_end,
                        timing=stream["timing"],
                    )
                    rank = stream.get("rank")
                    selected_basis = basis[:rank] if rank is not None else None
                    state = timed_interface_state(
                        corrupt_state[
                            microbatch_index : microbatch_index + 1
                        ],
                        delta,
                        mask,
                        basis=selected_basis,
                        component=stream["component"],
                    )
                    states.append(state)
                    tokens.append(
                        corrupt_batch[microbatch_index : microbatch_index + 1]
                    )
                    masks.append(
                        corrupt_batch_mask[
                            microbatch_index : microbatch_index + 1
                        ]
                    )
                    index_to_intervention[microbatch_index] = output_index
                    if selected_basis is not None:
                        active_delta = delta[mask]
                        if active_delta.numel():
                            stream["diagnostics"].append(
                                projection_diagnostics(
                                    active_delta, selected_basis
                                )
                            )
                intervention_logits = _fixed_state_logits(
                    model,
                    torch.cat(tokens, dim=0),
                    torch.cat(masks, dim=0),
                    target_hook,
                    torch.cat(states, dim=0),
                )

            for microbatch_index, stream in enumerate(microbatch):
                if stream["component"] == "positive":
                    logits = clean_logits[microbatch_index]
                elif stream["component"] == "negative":
                    logits = corrupt_logits[microbatch_index]
                else:
                    logits = intervention_logits[
                        index_to_intervention[microbatch_index]
                    ]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        f"Non-finite logits for {stream['condition']}."
                    )
                _append_token(stream, int(logits.argmax().item()), eos_id)

    rows = []
    for stream in streams:
        generated = stream["generated"]
        diagnostics = stream["diagnostics"]
        aggregate_diagnostics = None
        if diagnostics:
            aggregate_diagnostics = {
                key: float(sum(row[key] for row in diagnostics) / len(diagnostics))
                for key in diagnostics[0]
            }
            aggregate_diagnostics["steps"] = len(diagnostics)
            aggregate_diagnostics["maximum_reconstruction_error"] = max(
                row["reconstruction_max_abs_error"] for row in diagnostics
            )
            aggregate_diagnostics["maximum_persona_complement_abs_dot"] = max(
                row["persona_complement_max_abs_dot"] for row in diagnostics
            )
        rows.append(
            {
                "schema_version": 1,
                "record_id": (
                    f"phase1-{stream['condition']}-q{question_index:02d}"
                ),
                "behavior_protocol_id": None,
                "question_id": f"eval:q{question_index:02d}",
                "question_index": question_index,
                "instruction_index": instruction_index,
                "question": question,
                "instruction": (
                    positive_instruction
                    if stream["component"] == "positive"
                    else negative_instruction
                ),
                "system_prompt": (
                    positive_system
                    if stream["component"] == "positive"
                    else negative_system
                ),
                "prompt": (
                    positive_prompt
                    if stream["component"] == "positive"
                    else negative_prompt
                ),
                "prompt_alignment": alignment,
                "condition": stream["condition"],
                "timing": stream.get("timing"),
                "rank": stream.get("rank"),
                "answer": model.tokenizer.decode(
                    generated, skip_special_tokens=True
                ),
                "generated_token_ids": generated,
                "generated_token_ids_sha256": token_ids_sha256(generated),
                "generated_token_count": len(generated),
                "hit_max_new_tokens": (
                    stream["termination_reason"] == "token_cap"
                    and len(generated) >= max_new_tokens
                ),
                "termination_reason": stream["termination_reason"],
                "decoding": "greedy",
                "internal_projection_diagnostics": aggregate_diagnostics,
            }
        )
    negative = next(row for row in rows if row["condition"] == "negative_baseline")
    rows.append(
        {
            **negative,
            "record_id": f"phase1-zero_clean_interface-q{question_index:02d}",
            "condition": "zero_clean_interface",
            "shared_execution_key": negative["record_id"],
        }
    )
    return rows, {
        "question_index": question_index,
        "prompt_alignment": alignment,
        "prompt_tokens": int(clean_tokens.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_sycophancy.json",
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--positive-scored", type=Path, default=None)
    parser.add_argument("--negative-scored", type=Path, default=None)
    parser.add_argument("--question-indices", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--identity-max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--condition-microbatch-size", type=int, default=12)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if args.question_indices != sorted(set(args.question_indices)):
        raise ValueError("--question-indices must be sorted and unique.")
    if args.max_new_tokens <= 0 or args.identity_max_new_tokens <= 0:
        raise ValueError("Generation token limits must be positive.")
    if args.condition_microbatch_size <= 0:
        raise ValueError("--condition-microbatch-size must be positive.")

    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.causal_interface import (
        PHASE1_RANKS,
        PHASE1_TIMINGS,
        build_nested_persona_basis,
        canonical_json_sha256,
        leave_one_question_out_stability,
        validate_phase1_output_schema,
        validate_split_manifest,
    )
    from refusal_circuit.model import (
        load_model,
        tokenize_prompts,
        validate_model_architecture,
    )
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        align_effective_pairs,
        make_system_prompt,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )

    started = time.time()
    config_path = _resolve(args.config)
    split_path = _resolve(args.split_manifest)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    intermediate = ROOT / config["paths"]["intermediate_root"]
    official = ROOT / config["paths"]["official_root"]
    positive_path = _resolve(args.positive_scored) if args.positive_scored else (
        intermediate / "judged/pos_scored.jsonl"
    )
    negative_path = _resolve(args.negative_scored) if args.negative_scored else (
        intermediate / "judged/neg_scored.jsonl"
    )
    extract_path = official / "extract.json"
    eval_path = official / "eval.json"
    expected_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            config_path,
            positive_path,
            negative_path,
            extract_path,
            eval_path,
        )
    }
    validate_split_manifest(split, expected_source_hashes=expected_hashes)
    output_root = _resolve(args.output_root) if args.output_root else (
        intermediate / "causal_interface_phase1/run1"
    )
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Output root is nonempty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

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
    oracle_backend_settings = _configure_residual_only_forward(model)
    tokenizer_identity = _tokenizer_identity(model.tokenizer)
    target_hook = config["vector"]["transformer_lens_target_hook"]
    if target_hook != "blocks.20.hook_resid_pre":
        raise ValueError("Phase 1 target hook must remain blocks.20.hook_resid_pre.")

    positive_rows = read_jsonl(positive_path)
    negative_rows = read_jsonl(negative_path)
    extraction_cfg = config["extraction"]
    pairs = align_effective_pairs(
        positive_rows,
        negative_rows,
        trait_threshold=float(extraction_cfg["trait_threshold"]),
        coherence_threshold=float(extraction_cfg["coherence_threshold"]),
    )
    interface_questions = {
        int(row["question_index"])
        for row in split["scopes"]["interface_extraction"]
    }
    selected_pairs = [
        pair
        for pair in pairs
        if int(pair[0]["question_index"]) in interface_questions
    ]
    if len(selected_pairs) < int(
        split["eligibility_rule"]["minimum_interface_effective_pairs"]
    ):
        raise ValueError("Frozen interface pair minimum is no longer satisfied.")
    basis_path = output_root / "interface_basis.pt"
    differences_path = output_root / "paired_activation_differences.pt"
    basis_metadata_path = output_root / "interface_basis.metadata.json"
    basis_artifacts = (basis_path, differences_path, basis_metadata_path)
    existing_basis_artifacts = [path.exists() for path in basis_artifacts]
    if any(existing_basis_artifacts) and not all(existing_basis_artifacts):
        raise ValueError("Partial interface-basis artifacts cannot be resumed.")
    if all(existing_basis_artifacts):
        basis_metadata = json.loads(
            basis_metadata_path.read_text(encoding="utf-8")
        )
        if (
            basis_metadata.get("status") != "complete"
            or basis_metadata.get("sources") != expected_hashes
            or basis_metadata.get("split_manifest_sha256")
            != sha256_file(split_path)
            or basis_metadata["artifacts"]["basis"]["sha256"]
            != sha256_file(basis_path)
            or basis_metadata["artifacts"]["paired_differences"]["sha256"]
            != sha256_file(differences_path)
        ):
            raise ValueError("Existing interface basis failed provenance checks.")
        saved_basis = torch.load(
            basis_path, map_location="cpu", weights_only=True
        )
        basis = saved_basis["basis"]
        singular_values = saved_basis["singular_values"]
    else:
        differences = []
        pair_metadata = []
        for positive, negative in tqdm(
            selected_pairs, desc="extract independent interface states"
        ):
            positive_state = _response_average_state(model, positive, target_hook)
            negative_state = _response_average_state(model, negative, target_hook)
            differences.append(positive_state - negative_state)
            pair_metadata.append(
                {
                    "pair_id": positive["pair_id"],
                    "question_index": int(positive["question_index"]),
                    "instruction_index": int(positive["instruction_index"]),
                    "rollout_index": int(positive["rollout_index"]),
                    "positive_record_id": positive["record_id"],
                    "negative_record_id": negative["record_id"],
                    "positive_row_sha256": canonical_json_sha256(positive),
                    "negative_row_sha256": canonical_json_sha256(negative),
                }
            )
        paired_differences = torch.stack(differences)
        basis, singular_values, basis_diagnostics = build_nested_persona_basis(
            paired_differences, max_rank=max(PHASE1_RANKS)
        )
        loo = leave_one_question_out_stability(
            paired_differences,
            [row["question_index"] for row in pair_metadata],
            basis,
        )
        torch.save(
            {
                "basis": basis,
                "singular_values": singular_values,
                "ranks": PHASE1_RANKS,
                "target_hook": target_hook,
            },
            basis_path,
        )
        torch.save(paired_differences, differences_path)
        basis_metadata = {
            "schema_version": 1,
            "status": "complete",
            "method": "independent_mean_direction_plus_residual_pca",
            "model": model_cfg,
            "tokenizer_identity": tokenizer_identity,
            "target_hook": target_hook,
            "ranks": list(PHASE1_RANKS),
            "normalization": "orthonormal unit rows",
            "basis_diagnostics": basis_diagnostics,
            "singular_values": singular_values.tolist(),
            "leave_one_extraction_question_out": loo,
            "split_manifest": str(split_path.relative_to(ROOT)),
            "split_manifest_sha256": sha256_file(split_path),
            "sources": expected_hashes,
            "pairs": pair_metadata,
            "artifacts": {
                "basis": {
                    "path": str(basis_path.relative_to(ROOT)),
                    "sha256": sha256_file(basis_path),
                },
                "paired_differences": {
                    "path": str(differences_path.relative_to(ROOT)),
                    "sha256": sha256_file(differences_path),
                },
            },
        }
        write_json(basis_metadata_path, basis_metadata)

    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    questions = evaluation["questions"]
    instructions = evaluation["instruction"]
    designed = []
    for question_index in args.question_indices:
        instruction_index = question_index % len(instructions)
        instruction = instructions[instruction_index]
        positive_system = make_system_prompt(
            config["trait"]["positive_assistant_name"], instruction["pos"]
        )
        negative_system = make_system_prompt(
            config["trait"]["negative_assistant_name"], instruction["neg"]
        )
        designed.append(
            (
                question_index,
                questions[question_index],
                instruction_index,
                instruction["pos"],
                instruction["neg"],
                positive_system,
                negative_system,
            )
        )

    identity_path = output_root / "identity_check.jsonl"
    all_identity_rows = (
        read_jsonl(identity_path) if identity_path.exists() else []
    )
    identity_rows = [
        row
        for row in all_identity_rows
        if row.get("oracle_backend_id") == ORACLE_BACKEND_ID
    ]
    if len(identity_rows) != len(all_identity_rows):
        write_jsonl(identity_path, identity_rows)
    identity_by_question = {
        int(row["question_index"]): row for row in identity_rows
    }
    if len(identity_by_question) != len(identity_rows):
        raise ValueError("Duplicate identity-check question rows.")
    for (
        question_index,
        question,
        instruction_index,
        _positive_instruction,
        _negative_instruction,
        positive_system,
        negative_system,
    ) in designed:
        if question_index in identity_by_question:
            continue
        positive_prompt, negative_prompt, alignment = _align_system_prompt_pair(
            model.tokenizer,
            question=question,
            positive_system_prompt=positive_system,
            negative_system_prompt=negative_system,
        )
        clean_tokens, clean_mask = tokenize_prompts(
            model, [positive_prompt], max_length=args.max_length
        )
        corrupt_tokens, corrupt_mask = tokenize_prompts(
            model, [negative_prompt], max_length=args.max_length
        )
        if (
            clean_tokens.shape != corrupt_tokens.shape
            or not torch.equal(clean_mask, corrupt_mask)
        ):
            raise ValueError("Identity prompt pair lost two-sided alignment.")
        identity = _run_identity_preflight(
            model,
            clean_tokens=clean_tokens,
            clean_mask=clean_mask,
            corrupt_tokens=corrupt_tokens,
            corrupt_mask=corrupt_mask,
            target_hook=target_hook,
            max_new_tokens=args.identity_max_new_tokens,
        )
        row = {
            "oracle_backend_id": ORACLE_BACKEND_ID,
            "question_index": question_index,
            "instruction_index": instruction_index,
            "prompt_alignment": alignment,
            **identity,
        }
        write_jsonl(identity_path, [row], mode="a")
        identity_by_question[question_index] = row
    identity_rows = [
        identity_by_question[question_index]
        for question_index in args.question_indices
    ]

    responses_path = output_root / "responses.jsonl"
    all_responses = read_jsonl(responses_path) if responses_path.exists() else []
    responses = [
        row
        for row in all_responses
        if row.get("oracle_backend_id") == ORACLE_BACKEND_ID
    ]
    if len(responses) != len(all_responses):
        write_jsonl(responses_path, responses)
    expected_conditions = {
        spec["condition"] for spec in _condition_specs()
    } | {"zero_clean_interface"}
    rows_by_question: dict[int, list[dict]] = {}
    for row in responses:
        rows_by_question.setdefault(int(row["question_index"]), []).append(row)
    completed_questions = set()
    retained_rows = []
    for question_index, question_rows in sorted(rows_by_question.items()):
        conditions = {row["condition"] for row in question_rows}
        if (
            len(question_rows) == len(expected_conditions)
            and conditions == expected_conditions
        ):
            completed_questions.add(question_index)
            retained_rows.extend(question_rows)
    if len(retained_rows) != len(responses):
        write_jsonl(responses_path, retained_rows)
        responses = retained_rows
    alignment_audit = []
    for item in tqdm(designed, desc="phase1 causal-interface oracle"):
        if item[0] in completed_questions:
            alignment_audit.append(
                {
                    "question_index": item[0],
                    "resumed_complete_question": True,
                }
            )
            continue
        rows, alignment = _run_question_grid(
            model,
            question_index=item[0],
            question=item[1],
            instruction_index=item[2],
            positive_instruction=item[3],
            negative_instruction=item[4],
            positive_system=item[5],
            negative_system=item[6],
            basis=basis.to(next(model.parameters()).device),
            target_hook=target_hook,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            condition_microbatch_size=args.condition_microbatch_size,
        )
        for row in rows:
            row["oracle_backend_id"] = ORACLE_BACKEND_ID
            row["behavior_protocol_id"] = config["protocol_id"]
            row["trait"] = config["trait"]["name"]
            row["model"] = model_cfg["name"]
            row["model_revision"] = model_cfg.get("revision")
        write_jsonl(responses_path, rows, mode="a")
        responses.extend(rows)
        alignment_audit.append(alignment)
    responses = read_jsonl(responses_path)
    if len(responses) != len(args.question_indices) * len(expected_conditions):
        raise RuntimeError("Phase 1 response grid is incomplete.")
    metadata_path = output_root / "metadata.json"
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "phase": 1,
        "method": "independent low-dimensional full-model causal-interface oracle",
        "behavior_protocol_id": config["protocol_id"],
        "code_commit": _git_commit(),
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "tokenizer_identity": tokenizer_identity,
        "split_manifest": str(split_path.relative_to(ROOT)),
        "split_manifest_sha256": sha256_file(split_path),
        "basis": str(basis_path.relative_to(ROOT)),
        "basis_sha256": sha256_file(basis_path),
        "basis_metadata": str(basis_metadata_path.relative_to(ROOT)),
        "basis_metadata_sha256": sha256_file(basis_metadata_path),
        "interface_cut_contract": {
            "hook": target_hook,
            "replacement_scope": "complete residual state tensor",
            "downstream_input_tokens": "negative-persona synchronized prefix",
            "clean_hooks_outside_interface": 0,
            "bypass_audit": "no clean cache or hook enters downstream computation outside the fixed interface state",
        },
        "identity": {
            "passed": all(row["passed"] for row in identity_rows),
            "questions": len(identity_rows),
            "max_new_tokens": args.identity_max_new_tokens,
            "exact_token_agreement": min(
                row["exact_token_agreement"] for row in identity_rows
            ),
        },
        "generation": {
            "question_indices": args.question_indices,
            "ranks": list(PHASE1_RANKS),
            "timings": list(PHASE1_TIMINGS),
            "conditions": [spec["condition"] for spec in _condition_specs()]
            + ["zero_clean_interface"],
            "method": "greedy",
            "backend": ORACLE_BACKEND_ID,
            "backend_settings": oracle_backend_settings,
            "max_new_tokens": args.max_new_tokens,
            "condition_microbatch_size": args.condition_microbatch_size,
            "shared_baselines": True,
            "zero_shares_negative_execution": True,
            "resume_boundary": "complete question-condition grid",
        },
        "prompt_alignment": {
            "method": "two_sided_neutral_whitespace_token_parity",
            "audit": alignment_audit,
        },
        "outputs": [
            {
                "path": str(responses_path.relative_to(ROOT)),
                "rows": len(responses),
                "sha256": sha256_file(responses_path),
            },
            {
                "path": str(identity_path.relative_to(ROOT)),
                "rows": len(identity_rows),
                "sha256": sha256_file(identity_path),
            },
            {
                "path": str(basis_path.relative_to(ROOT)),
                "sha256": sha256_file(basis_path),
            },
        ],
        "wall_time_seconds": time.time() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
    }
    validate_phase1_output_schema(metadata)
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract and behaviorally test a frozen prompt-derived persona-direction grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_sycophancy.json",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=(4, 8, 12, 16, 20),
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=(-1, -2),
    )
    parser.add_argument(
        "--question-indices",
        type=int,
        nargs="+",
        default=(0, 1, 2, 3, 4),
    )
    parser.add_argument("--activation-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(args.layers) != len(set(args.layers)):
        raise ValueError("--layers must be unique")
    if len(args.positions) != len(set(args.positions)):
        raise ValueError("--positions must be unique")
    if any(position >= 0 for position in args.positions):
        raise ValueError("--positions must contain only negative suffix indices")
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("--question-indices must be unique")
    if args.activation_batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch size and max-new-tokens must be positive")

    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        ablate_direction_from_inputs,
        ablate_direction_from_output,
        add_direction_to_inputs,
        align_effective_pairs,
        make_system_prompt,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Prompt-direction search requires a CUDA device.")
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    intermediate = _resolve(config["paths"]["intermediate_root"])
    positive_path = intermediate / "judged/pos_scored.jsonl"
    negative_path = intermediate / "judged/neg_scored.jsonl"
    positive_rows = read_jsonl(positive_path)
    negative_rows = read_jsonl(negative_path)
    for polarity, path in (
        ("positive", positive_path),
        ("negative", negative_path),
    ):
        judged_metadata_path = path.with_suffix(".metadata.json")
        judged_metadata = json.loads(
            judged_metadata_path.read_text(encoding="utf-8")
        )
        if (
            not judged_metadata.get("canonical_protocol")
            or judged_metadata.get("generation_protocol_id")
            != config["protocol_id"]
            or judged_metadata.get("rubric_artifact") != "extract"
            or judged_metadata.get("output_sha256") != sha256_file(path)
        ):
            raise ValueError(
                f"{polarity} extraction input failed its frozen metadata gate"
            )
    extraction = config["extraction"]
    pairs = align_effective_pairs(
        positive_rows,
        negative_rows,
        trait_threshold=float(extraction["trait_threshold"]),
        coherence_threshold=float(extraction["coherence_threshold"]),
    )
    if not pairs:
        raise RuntimeError("No effective extraction pairs are available.")

    output_root = _resolve(args.output_root)
    responses_path = output_root / "responses.jsonl"
    diagnostics_path = output_root / "candidate_diagnostics.jsonl"
    vectors_path = output_root / "candidate_vectors.pt"
    metadata_path = output_root / "metadata.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already contains output")
        for path in (
            responses_path,
            diagnostics_path,
            vectors_path,
            metadata_path,
        ):
            path.unlink(missing_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        torch_dtype=getattr(torch, model_cfg["dtype"]),
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    hidden_size = int(model.config.hidden_size)
    hidden_state_count = int(model.config.num_hidden_layers) + 1
    if any(layer <= 0 or layer >= hidden_state_count for layer in args.layers):
        raise ValueError("A requested layer is outside the hidden-state stack.")

    sums = {
        polarity: {
            (layer, position): torch.zeros(hidden_size, dtype=torch.float64)
            for layer in args.layers
            for position in args.positions
        }
        for polarity in ("positive", "negative")
    }
    counts = {"positive": 0, "negative": 0}
    started = time.time()
    extraction_rows = [
        (polarity, row["prompt"])
        for positive, negative in pairs
        for polarity, row in (
            ("positive", positive),
            ("negative", negative),
        )
    ]
    for batch in _batches(extraction_rows, args.activation_batch_size):
        prompts = [prompt for _polarity, prompt in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        for row_index, (polarity, _prompt) in enumerate(batch):
            for layer in args.layers:
                hidden = outputs.hidden_states[layer]
                for position in args.positions:
                    if abs(position) > hidden.shape[1]:
                        raise ValueError(
                            f"position {position} exceeds prompt length "
                            f"{hidden.shape[1]}"
                        )
                    sums[polarity][(layer, position)].add_(
                        hidden[row_index, position]
                        .detach()
                        .float()
                        .cpu()
                        .double()
                    )
            counts[polarity] += 1
        del outputs

    vectors = {}
    positive_means = {}
    negative_means = {}
    candidate_specs = []
    for layer in args.layers:
        for position in args.positions:
            candidate_id = f"layer{layer:02d}_pos{position}"
            positive_mean = (
                sums["positive"][(layer, position)] / counts["positive"]
            ).float()
            negative_mean = (
                sums["negative"][(layer, position)] / counts["negative"]
            ).float()
            vector = positive_mean - negative_mean
            if not torch.isfinite(vector).all() or not vector.norm() > 0:
                raise RuntimeError(f"Invalid vector for {candidate_id}.")
            vectors[candidate_id] = vector
            positive_means[candidate_id] = positive_mean
            negative_means[candidate_id] = negative_mean
            candidate_specs.append(
                {
                    "candidate_id": candidate_id,
                    "layer_one_indexed": layer,
                    "position": position,
                    "raw_norm": float(vector.norm().item()),
                }
            )
    torch.save(
        {
            "vectors": vectors,
            "positive_means": positive_means,
            "negative_means": negative_means,
        },
        vectors_path,
    )

    official_root = _resolve(config["paths"]["official_root"])
    eval_path = official_root / "eval.json"
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    questions = evaluation["questions"]
    instructions = evaluation["instruction"]
    designed = []
    for question_index in args.question_indices:
        if not 0 <= question_index < len(questions):
            raise IndexError(f"Question {question_index} is outside eval.json")
        instruction_index = question_index % len(instructions)
        instruction = instructions[instruction_index]
        positive_system = make_system_prompt(
            config["trait"]["positive_assistant_name"],
            instruction["pos"],
        )
        negative_system = make_system_prompt(
            config["trait"]["negative_assistant_name"],
            instruction["neg"],
        )
        designed.append(
            {
                "question_index": question_index,
                "question": questions[question_index],
                "instruction_index": instruction_index,
                "positive_prompt": tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": positive_system},
                        {"role": "user", "content": questions[question_index]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "negative_prompt": tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": negative_system},
                        {"role": "user", "content": questions[question_index]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            }
        )

    def tokenize(prompts: list[str]):
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        return {key: value.to(args.device) for key, value in encoded.items()}

    def generate(prompts: list[str]) -> list[dict]:
        inputs = tokenize(prompts)
        prompt_width = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        results = []
        for sequence in generated:
            suffix = sequence[prompt_width:]
            token_ids = suffix.tolist()
            results.append(
                {
                    "answer": tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                    ),
                    "generated_token_count": len(token_ids),
                    "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
                    "terminated_with_eos": bool(
                        tokenizer.eos_token_id is not None
                        and token_ids
                        and token_ids[-1] == tokenizer.eos_token_id
                    ),
                }
            )
        return results

    behavior_rows = []

    def append_rows(
        *,
        condition: str,
        prompts: list[str],
        generations: list[dict],
        candidate: dict | None,
    ) -> None:
        for design, prompt, generation in zip(
            designed,
            prompts,
            generations,
            strict=True,
        ):
            candidate_id = candidate["candidate_id"] if candidate else "none"
            behavior_rows.append(
                {
                    "schema_version": 1,
                    "record_id": (
                        f"prompt-direction-{condition}-{candidate_id}-"
                        f"q{design['question_index']:02d}"
                    ),
                    "behavior_protocol_id": config["protocol_id"],
                    "condition": condition,
                    "candidate_id": candidate_id,
                    "layer_one_indexed": (
                        candidate["layer_one_indexed"] if candidate else None
                    ),
                    "position": candidate["position"] if candidate else None,
                    "question_id": (
                        f"{config['trait']['name']}_{design['question_index']}"
                    ),
                    "question_index": design["question_index"],
                    "instruction_index": design["instruction_index"],
                    "question": design["question"],
                    "prompt": prompt,
                    "decoding": "greedy",
                    "model": model_cfg["name"],
                    "model_revision": model_cfg.get("revision"),
                    **generation,
                }
            )

    positive_prompts = [row["positive_prompt"] for row in designed]
    negative_prompts = [row["negative_prompt"] for row in designed]
    append_rows(
        condition="baseline_positive",
        prompts=positive_prompts,
        generations=generate(positive_prompts),
        candidate=None,
    )
    append_rows(
        condition="baseline_negative",
        prompts=negative_prompts,
        generations=generate(negative_prompts),
        candidate=None,
    )

    negative_inputs = tokenize(negative_prompts)
    with torch.inference_mode():
        clean_negative_logits = model(**negative_inputs).logits[:, -1].float()

    diagnostic_rows = []
    for candidate in candidate_specs:
        candidate_id = candidate["candidate_id"]
        vector = vectors[candidate_id].to(
            device=args.device,
            dtype=next(model.parameters()).dtype,
        )
        source_block = model.model.layers[candidate["layer_one_indexed"]]
        addition_handle = source_block.register_forward_pre_hook(
            lambda _module, inputs, value=vector:
            add_direction_to_inputs(inputs, value)
        )
        try:
            induced = generate(negative_prompts)
        finally:
            addition_handle.remove()
        append_rows(
            condition="negative_plus_raw_direction",
            prompts=negative_prompts,
            generations=induced,
            candidate=candidate,
        )

        ablation_handles = []
        for block in model.model.layers:
            ablation_handles.extend(
                [
                    block.register_forward_pre_hook(
                        lambda _module, inputs, value=vector:
                        ablate_direction_from_inputs(inputs, value)
                    ),
                    block.self_attn.register_forward_hook(
                        lambda _module, _inputs, output, value=vector:
                        ablate_direction_from_output(output, value)
                    ),
                    block.mlp.register_forward_hook(
                        lambda _module, _inputs, output, value=vector:
                        ablate_direction_from_output(output, value)
                    ),
                ]
            )
        try:
            ablated = generate(positive_prompts)
            with torch.inference_mode():
                ablated_negative_logits = model(**negative_inputs).logits[:, -1].float()
        finally:
            for handle in ablation_handles:
                handle.remove()
        append_rows(
            condition="positive_global_direction_ablation",
            prompts=positive_prompts,
            generations=ablated,
            candidate=candidate,
        )
        negative_kl = functional.kl_div(
            functional.log_softmax(ablated_negative_logits, dim=-1),
            functional.softmax(clean_negative_logits, dim=-1),
            reduction="batchmean",
        )
        diagnostic_rows.append(
            {
                **candidate,
                "negative_ablation_kl": float(negative_kl.item()),
            }
        )

    write_jsonl(responses_path, behavior_rows)
    write_jsonl(diagnostics_path, diagnostic_rows)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "canonical_protocol": False,
        "claim_scope": "q0-q4 exploratory refusal-style prompt-direction search",
        "git_commit": _git_commit(),
        "model": model_cfg,
        "extraction": {
            "positive_input": str(positive_path.relative_to(ROOT)),
            "positive_sha256": sha256_file(positive_path),
            "negative_input": str(negative_path.relative_to(ROOT)),
            "negative_sha256": sha256_file(negative_path),
            "effective_pairs": len(pairs),
            "counts": counts,
            "layers": list(args.layers),
            "positions": list(args.positions),
            "weighting": "one vote per effective response pair",
        },
        "behavior": {
            "questions": list(args.question_indices),
            "instruction_rule": "question_index modulo five",
            "conditions": [
                "baseline_positive",
                "baseline_negative",
                "negative_plus_raw_direction",
                "positive_global_direction_ablation",
            ],
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
        },
        "selection_rule": {
            "objective": (
                "maximize min(normalized induction, normalized ablation) "
                "among candidates passing guardrails"
            ),
            "negative_ablation_kl_max": 0.1,
            "coherence_min": 80.0,
            "cap_hits": 0,
        },
        "artifacts": {
            "vectors": {
                "path": str(vectors_path.relative_to(ROOT)),
                "sha256": sha256_file(vectors_path),
            },
            "responses": {
                "path": str(responses_path.relative_to(ROOT)),
                "rows": len(behavior_rows),
                "sha256": sha256_file(responses_path),
            },
            "diagnostics": {
                "path": str(diagnostics_path.relative_to(ROOT)),
                "rows": len(diagnostic_rows),
                "sha256": sha256_file(diagnostics_path),
            },
            "eval": {
                "path": str(eval_path.relative_to(ROOT)),
                "sha256": sha256_file(eval_path),
            },
        },
        "wall_time_seconds": time.time() - started,
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare prompt-end, response-only, and recurrent persona-vector steering."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import ROOT


TIMINGS = ("prompt_end_only", "response_only", "recurrent")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_sycophancy.json",
    )
    parser.add_argument(
        "--question-indices",
        type=int,
        nargs="+",
        default=(0, 1, 2, 3, 4),
    )
    parser.add_argument("--coefficient", type=float, default=1.5)
    parser.add_argument(
        "--vector-variant",
        choices=("response_avg_diff", "prompt_last_diff"),
        default="response_avg_diff",
    )
    parser.add_argument(
        "--match-response-vector-norm",
        action="store_true",
        help="Scale a noncanonical variant to the selected response vector's norm.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.question_indices) != len(set(args.question_indices)):
        raise ValueError("Question indices must be unique.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        match_vector_norm,
        sha256_file,
        steer_model_output_timed,
        write_jsonl,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The timing diagnostic requires a CUDA device.")
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    vector_cfg = config["vector"]
    official_root = _resolve(Path(config["paths"]["official_root"]))
    eval_path = official_root / "eval.json"
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    questions = eval_data["questions"]
    if any(index < 0 or index >= len(questions) for index in args.question_indices):
        raise IndexError("Question index is outside the frozen evaluation set.")

    result_root = _resolve(Path(config["paths"]["results_root"]))
    trait_slug = config["trait"]["name"]
    vector_path = (
        result_root / "vector" / f"{trait_slug}_{args.vector_variant}.pt"
    )
    vector_stack = torch.load(vector_path, map_location="cpu", weights_only=True)
    layer = int(vector_cfg["paper_layer"])
    if vector_stack.ndim != 2 or not 0 <= layer < vector_stack.shape[0]:
        raise ValueError(f"Unexpected vector stack shape {tuple(vector_stack.shape)}.")
    direction = vector_stack[layer]
    source_norm = float(direction.float().norm().item())
    norm_reference_path = None
    if args.match_response_vector_norm:
        if args.vector_variant == "response_avg_diff":
            raise ValueError(
                "--match-response-vector-norm is unnecessary for response_avg_diff."
            )
        norm_reference_path = (
            result_root / "vector" / f"{trait_slug}_response_avg_diff.pt"
        )
        reference_stack = torch.load(
            norm_reference_path,
            map_location="cpu",
            weights_only=True,
        )
        direction = match_vector_norm(direction, reference_stack[layer])
    applied_norm = float(direction.float().norm().item())

    output_root = _resolve(args.output_root)
    responses_path = output_root / "responses.jsonl"
    metadata_path = output_root / "metadata.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already contains output.")
        responses_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
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
    block_index = int(vector_cfg["transformer_lens_steering_block"])
    target_block = model.model.layers[block_index]
    direction = direction.to(
        device=args.device,
        dtype=next(model.parameters()).dtype,
    )

    rows = []
    conditions = ("baseline", *TIMINGS)
    for question_index in args.question_indices:
        question = questions[question_index]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[1])
        for condition in conditions:
            handle = None
            if condition != "baseline":
                handle = target_block.register_forward_hook(
                    lambda _module, _inputs, output, timing=condition:
                    steer_model_output_timed(
                        output,
                        direction,
                        args.coefficient,
                        timing=timing,
                    )
                )
            try:
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
            finally:
                if handle is not None:
                    handle.remove()
            suffix = generated[0, prompt_length:]
            token_ids = suffix.tolist()
            terminated = bool(
                tokenizer.eos_token_id is not None
                and token_ids
                and token_ids[-1] == tokenizer.eos_token_id
            )
            rows.append(
                {
                    "schema_version": 1,
                    "record_id": (
                        f"vector-timing-{condition}-q{question_index:02d}"
                    ),
                    "behavior_protocol_id": config["protocol_id"],
                    "condition": condition,
                    "question_id": f"{trait_slug}_{question_index}",
                    "question_index": question_index,
                    "question": question,
                    "prompt": prompt,
                    "answer": tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                    ),
                    "generated_token_count": len(token_ids),
                    "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
                    "terminated_with_eos": terminated,
                    "decoding": "greedy",
                    "model": model_cfg["name"],
                    "model_revision": model_cfg.get("revision"),
                    "coefficient": (
                        0.0 if condition == "baseline" else args.coefficient
                    ),
                    "timing": condition,
                    "layer_one_indexed": layer,
                    "steering_block_zero_indexed": block_index,
                }
            )

    write_jsonl(responses_path, rows)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "response_average_vector_timing",
        "canonical_protocol": False,
        "claim_scope": "q0-q4 development timing diagnostic",
        "behavior_protocol_id": config["protocol_id"],
        "git_commit": _git_commit(),
        "model": model_cfg,
        "vector": {
            "path": str(vector_path.relative_to(ROOT)),
            "sha256": sha256_file(vector_path),
            "variant": args.vector_variant,
            "source_norm": source_norm,
            "applied_norm": applied_norm,
            "norm_reference": (
                {
                    "path": str(norm_reference_path.relative_to(ROOT)),
                    "sha256": sha256_file(norm_reference_path),
                    "variant": "response_avg_diff",
                }
                if norm_reference_path is not None
                else None
            ),
            "layer_one_indexed": layer,
            "steering_block_zero_indexed": block_index,
            "coefficient": args.coefficient,
        },
        "questions": list(args.question_indices),
        "conditions": list(conditions),
        "timing_semantics": {
            "prompt_end_only": "final prompt token on prefill only",
            "response_only": "single-token cached decoding calls only",
            "recurrent": "final prompt token and every cached decoding call",
        },
        "decoding": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "eval_artifact": {
            "path": str(eval_path.relative_to(ROOT)),
            "sha256": sha256_file(eval_path),
        },
        "outputs": [
            {
                "path": str(responses_path.relative_to(ROOT)),
                "rows": len(rows),
                "sha256": sha256_file(responses_path),
            }
        ],
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate held-out sycophancy responses under response-time vector steering."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _coefficient_label(coefficient: float) -> str:
    return f"{coefficient:+g}".replace("+", "pos").replace("-", "neg").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--vector", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=None, help="One-indexed paper/HF layer.")
    parser.add_argument("--coefficients", type=float, nargs="+", default=None)
    parser.add_argument("--rollouts-per-question", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file, steer_model_output, write_jsonl

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Canonical steering validation requires a CUDA job.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    vector_cfg = config["vector"]
    validation = config["steering_validation"]
    # Trait-prefixed vector filename, matching persona.extract's output. For the
    # sycophancy config this resolves to "sycophantic", unchanged from before.
    trait_slug = config["trait"]["name"]
    official_root = _resolve(config["paths"]["official_root"])
    eval_path = official_root / "eval.json"
    if not eval_path.exists():
        raise FileNotFoundError("Run `python scripts/rq.py persona.fetch` first.")
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    questions = eval_data["questions"]
    if len(questions) != int(validation["questions"]):
        raise ValueError("held-out evaluation question count differs from the frozen config")

    vector_root = _resolve(config["paths"]["results_root"]) / "vector"
    vector_path = _resolve(args.vector) if args.vector else vector_root / f"{trait_slug}_response_avg_diff.pt"
    if not vector_path.exists():
        raise FileNotFoundError(f"persona vector is missing: {vector_path}")
    layer = args.layer if args.layer is not None else int(vector_cfg["paper_layer"])
    vector_stack = torch.load(vector_path, map_location="cpu", weights_only=True)
    if vector_stack.ndim == 2:
        if not 0 <= layer < vector_stack.shape[0]:
            raise ValueError(f"layer {layer} is outside vector stack {tuple(vector_stack.shape)}")
        direction = vector_stack[layer]
    elif vector_stack.ndim == 1:
        direction = vector_stack
    else:
        raise ValueError(f"unexpected vector shape {tuple(vector_stack.shape)}")

    coefficients = args.coefficients or [float(value) for value in validation["coefficients"]]
    if len(coefficients) != len(set(coefficients)):
        raise ValueError("steering coefficients must be unique")
    rollouts = args.rollouts_per_question or int(validation["rollouts_per_question"])
    batch_size = args.batch_size or int(validation["generation_batch_size"])
    max_new_tokens = args.max_new_tokens or int(validation["max_new_tokens"])
    seed = int(validation["seed"])
    positions = validation["positions"]
    canonical_vector_path = vector_root / f"{trait_slug}_response_avg_diff.pt"
    if vector_path.resolve() != canonical_vector_path.resolve():
        raise ValueError("canonical layer validation requires the response-average vector stack")
    frozen_layers = [int(value) for value in config["layer_selection"]["candidate_layers_one_indexed"]]
    if layer not in frozen_layers:
        raise ValueError(f"layer {layer} is outside the paper's Qwen grid {frozen_layers}")
    frozen_coefficients = [float(value) for value in validation["coefficients"]]
    if any(value not in frozen_coefficients for value in coefficients):
        raise ValueError("canonical layer validation only permits the paper's steering coefficients")
    if rollouts != int(validation["rollouts_per_question"]):
        raise ValueError("canonical layer validation requires the frozen rollout count")
    if max_new_tokens != int(validation["max_new_tokens"]):
        raise ValueError("canonical layer validation requires the frozen max-new-tokens value")
    block_index = layer - 1
    if layer == int(vector_cfg["paper_layer"]):
        if block_index != int(vector_cfg["transformer_lens_steering_block"]):
            raise ValueError("selected steering block disagrees with the frozen layer mapping")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg.get("revision"), token=token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        token=token,
        torch_dtype=getattr(torch, model_cfg["dtype"]),
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    try:
        target_block = model.model.layers[block_index]
    except (AttributeError, IndexError) as error:
        raise RuntimeError("could not locate Qwen transformer block for steering") from error
    direction = direction.to(device=args.device, dtype=next(model.parameters()).dtype)

    output_root = (
        _resolve(config["paths"]["intermediate_root"]) / "steering" / f"layer_{layer:02d}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    started = time.time()
    for coefficient in coefficients:
        label = _coefficient_label(float(coefficient))
        output_path = output_root / f"coef_{label}.jsonl"
        existing = {}
        if output_path.exists():
            if args.overwrite:
                output_path.unlink()
            elif args.resume:
                cached = read_jsonl(output_path)
                existing = {row["record_id"]: row for row in cached}
                if len(existing) != len(cached):
                    raise ValueError(f"duplicate record IDs in {output_path}")
            else:
                raise FileExistsError(f"{output_path} exists; pass --resume or --overwrite")

        designed = [
            {
                "record_id": f"coef-{label}-q{question_index:02d}-r{rollout_index:02d}",
                "question_id": f"{trait_slug}_{question_index}",
                "question_index": question_index,
                "rollout_index": rollout_index,
                "question": question,
            }
            for question_index, question in enumerate(questions)
            for rollout_index in range(rollouts)
        ]
        pending = [row for row in designed if row["record_id"] not in existing]
        # Common random numbers across coefficients reduce Monte Carlo noise in
        # the steering curve while leaving the official sampling distribution unchanged.
        set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for batch in tqdm(list(_batches(pending, batch_size)), desc=f"steer {coefficient:+g}"):
            conversations = [[{"role": "user", "content": row["question"]}] for row in batch]
            prompts = [
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in conversations
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(args.device) for key, value in inputs.items()}

            handle = None
            if coefficient != 0:
                handle = target_block.register_forward_hook(
                    lambda _module, _inputs, output: steer_model_output(
                        output, direction, float(coefficient), positions=positions
                    )
                )
            context = nullcontext()
            try:
                with context, torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=float(validation["temperature"]),
                        top_p=float(validation["top_p"]),
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
            finally:
                if handle is not None:
                    handle.remove()
            suffix = generated[:, inputs["input_ids"].shape[1] :]
            answers = tokenizer.batch_decode(suffix, skip_special_tokens=True)
            output_rows = []
            for row, prompt, answer in zip(batch, prompts, answers):
                output_rows.append(
                    {
                        **row,
                        "schema_version": 1,
                        "steering_protocol_id": config["protocol_id"],
                        "trait": config["trait"]["name"],
                        "model": model_cfg["name"],
                        "model_revision": model_cfg.get("revision"),
                        "prompt": prompt,
                        "answer": answer,
                        "coefficient": float(coefficient),
                        "layer_one_indexed": layer,
                        "steering_block_zero_indexed": block_index,
                        "positions": positions,
                        "seed": seed,
                    }
                )
            write_jsonl(output_path, output_rows, mode="a")
        final = read_jsonl(output_path)
        expected = len(questions) * rollouts
        if len(final) != expected or len({row["record_id"] for row in final}) != expected:
            raise RuntimeError(f"incomplete output {output_path}: {len(final)}/{expected}")
        summaries.append(
            {
                "coefficient": float(coefficient),
                "path": str(output_path.relative_to(ROOT)),
                "rows": len(final),
                "sha256": sha256_file(output_path),
            }
        )

    metadata = {
        "schema_version": 1,
        "status": "complete",
        "canonical_protocol": True,
        "steering_protocol_id": config["protocol_id"],
        "model": model_cfg,
        "trait": config["trait"],
        "eval_artifact_sha256": sha256_file(eval_path),
        "vector_path": str(vector_path.relative_to(ROOT)),
        "vector_sha256": sha256_file(vector_path),
        "vector_norm": float(direction.float().norm().item()),
        "layer_one_indexed": layer,
        "steering_block_zero_indexed": block_index,
        "positions": positions,
        "sampling": {
            "temperature": validation["temperature"],
            "top_p": validation["top_p"],
            "max_new_tokens": max_new_tokens,
            "rollouts_per_question": rollouts,
            "batch_size": batch_size,
            "seed": seed,
            "common_random_numbers_across_coefficients": True,
        },
        "outputs": summaries,
        "wall_time_seconds": time.time() - started,
    }
    write_json(output_root / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

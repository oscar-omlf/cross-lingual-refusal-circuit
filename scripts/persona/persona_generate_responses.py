#!/usr/bin/env python3
"""Generate the canonical positive/negative response pairs for persona extraction."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--polarity", choices=("pos", "neg", "both"), default="both")
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
    from refusal_circuit.persona import extraction_specs, read_jsonl, sha256_file, write_jsonl

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Canonical extraction generation requires a CUDA job.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    extraction = config["extraction"]
    model_cfg = config["model"]
    source_root = _resolve(config["paths"]["official_root"])
    artifact_path = source_root / "extract.json"
    manifest_path = source_root / "manifest.json"
    if not artifact_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run `python scripts/rq.py persona.fetch` before generation.")
    trait_data = json.loads(artifact_path.read_text(encoding="utf-8"))
    if len(trait_data["questions"]) != int(extraction["questions"]):
        raise ValueError("official extraction question count does not match the frozen config")
    if len(trait_data["instruction"]) != int(extraction["instructions_per_polarity"]):
        raise ValueError("official instruction count does not match the frozen config")

    batch_size = args.batch_size or int(extraction["generation_batch_size"])
    max_new_tokens = args.max_new_tokens or int(extraction["max_new_tokens"])
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch size and max-new-tokens must be positive")
    seed = int(extraction["seed"])
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg.get("revision"), token=token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = getattr(torch, model_cfg["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        token=token,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()

    output_root = _resolve(config["paths"]["intermediate_root"]) / "generation"
    output_root.mkdir(parents=True, exist_ok=True)
    polarities = ("pos", "neg") if args.polarity == "both" else (args.polarity,)
    summary = []
    started = time.time()
    for polarity in polarities:
        assistant_key = (
            "positive_assistant_name" if polarity == "pos" else "negative_assistant_name"
        )
        specs = list(
            extraction_specs(
                trait_data,
                polarity=polarity,
                assistant_name=config["trait"][assistant_key],
                rollouts=int(extraction["rollouts_per_question_instruction"]),
            )
        )
        expected = (
            int(extraction["questions"])
            * int(extraction["instructions_per_polarity"])
            * int(extraction["rollouts_per_question_instruction"])
        )
        if len(specs) != expected:
            raise RuntimeError(f"designed {len(specs)} rows, expected {expected}")
        output_path = output_root / f"{polarity}_responses.jsonl"
        existing: dict[str, dict] = {}
        if output_path.exists():
            if args.overwrite:
                output_path.unlink()
            elif args.resume:
                rows = read_jsonl(output_path)
                existing = {row["record_id"]: row for row in rows}
                if len(existing) != len(rows):
                    raise ValueError(f"duplicate record IDs in partial output {output_path}")
            else:
                raise FileExistsError(f"{output_path} exists; pass --resume or --overwrite")
        pending = [spec for spec in specs if spec.record_id not in existing]
        for batch in tqdm(list(_batches(pending, batch_size)), desc=f"generate {polarity}"):
            conversations = [
                [
                    {"role": "system", "content": spec.system_prompt},
                    {"role": "user", "content": spec.question},
                ]
                for spec in batch
            ]
            prompts = [
                tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                for conversation in conversations
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=float(extraction["temperature"]),
                    top_p=float(extraction["top_p"]),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            suffix = generated[:, inputs["input_ids"].shape[1] :]
            answers = tokenizer.batch_decode(suffix, skip_special_tokens=True)
            rows = []
            for spec, prompt, answer, token_ids in zip(batch, prompts, answers, suffix):
                non_padding = int((token_ids != tokenizer.pad_token_id).sum().item())
                rows.append(
                    {
                        "schema_version": 1,
                        "trait": config["trait"]["name"],
                        "model": model_cfg["name"],
                        "model_revision": model_cfg.get("revision"),
                        "record_id": spec.record_id,
                        "pair_id": spec.pair_id,
                        "polarity": spec.polarity,
                        "question_index": spec.question_index,
                        "instruction_index": spec.instruction_index,
                        "rollout_index": spec.rollout_index,
                        "question": spec.question,
                        "instruction": spec.instruction,
                        "assistant_name": spec.assistant_name,
                        "system_prompt": spec.system_prompt,
                        "prompt": prompt,
                        "answer": answer,
                        "generated_token_count": non_padding,
                        "hit_max_new_tokens": non_padding >= max_new_tokens,
                        "seed": seed,
                    }
                )
            write_jsonl(output_path, rows, mode="a")
        final_rows = read_jsonl(output_path)
        if len(final_rows) != expected or len({row["record_id"] for row in final_rows}) != expected:
            raise RuntimeError(f"incomplete output {output_path}: {len(final_rows)}/{expected}")
        summary.append(
            {
                "polarity": polarity,
                "path": str(output_path.relative_to(ROOT)),
                "rows": len(final_rows),
                "sha256": sha256_file(output_path),
                "mean_generated_tokens": sum(row["generated_token_count"] for row in final_rows)
                / len(final_rows),
                "hit_max_new_tokens": sum(bool(row["hit_max_new_tokens"]) for row in final_rows),
            }
        )
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "canonical_protocol": True,
        "model": model_cfg,
        "trait": config["trait"],
        "official_artifact_sha256": sha256_file(artifact_path),
        "generation": {
            "temperature": extraction["temperature"],
            "top_p": extraction["top_p"],
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "seed": seed,
        },
        "outputs": summary,
        "wall_time_seconds": time.time() - started,
    }
    write_json(output_root / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

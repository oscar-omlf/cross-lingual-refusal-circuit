#!/usr/bin/env python3
"""Debug WildGuard on one harmless PolyRefuse row.

This intentionally uses only a harmless prompt and prints the generated benign
response plus raw WildGuard labels to the job log.  It must not be used for
harmful prompts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _rq1_bootstrap import ROOT  # noqa: F401
from rq2_wildguard_smoke_test import INSTRUCTION_FORMAT, parse_wildguard_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="en")
    parser.add_argument("--offset", type=int, default=32)
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--wildguard-model", default="allenai/wildguard")
    args = parser.parse_args()

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.model import format_instruction

    if args.language != "en":
        raise ValueError("This debug script intentionally only prints English harmless text.")
    if not torch.cuda.is_available():
        raise RuntimeError("Run through SLURM on a GPU node.")

    data_dir = Path("data/processed/qwen2.5-7b-instruct/polyrefuse") / args.language
    row = load_processed_split("test", "harmless", data_dir=data_dir)[args.offset]
    prompt = row["instruction"]

    token = os.environ.get("HF_TOKEN")
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, token=token)
    if qwen_tokenizer.pad_token_id is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=token,
    ).to("cuda").eval()

    formatted = format_instruction(qwen_tokenizer, prompt)
    qwen_inputs = qwen_tokenizer([formatted], return_tensors="pt", add_special_tokens=False).to("cuda")
    generated_responses: dict[str, str] = {}
    for max_new_tokens in (8, 32):
        with torch.inference_mode():
            out = qwen.generate(
                **qwen_inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=qwen_tokenizer.eos_token_id,
            )
        suffix = out[:, qwen_inputs.input_ids.shape[1] :]
        generated_responses[f"qwen_greedy_{max_new_tokens}tok"] = qwen_tokenizer.decode(
            suffix[0], skip_special_tokens=True
        )

    # Also test a fixed obviously benign response to separate WildGuard issues
    # from Qwen generation issues.
    generated_responses[
        "fixed_benign_response"
    ] = "Sure. A simple approach is to train an image classifier on labelled dog and cat images."

    del qwen
    torch.cuda.empty_cache()

    wg_tokenizer = AutoTokenizer.from_pretrained(args.wildguard_model, token=token)
    if wg_tokenizer.pad_token_id is None:
        wg_tokenizer.pad_token = wg_tokenizer.eos_token
    wildguard = AutoModelForCausalLM.from_pretrained(
        args.wildguard_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=token,
    ).to("cuda").eval()

    results = []
    for name, response in generated_responses.items():
        text = INSTRUCTION_FORMAT.format(prompt=prompt, response=response)
        inputs = wg_tokenizer([text], return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            out = wildguard.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=48,
                pad_token_id=wg_tokenizer.eos_token_id,
            )
        suffix = out[:, inputs.input_ids.shape[1] :]
        raw = wg_tokenizer.decode(suffix[0], skip_special_tokens=True)
        results.append(
            {
                "case": name,
                "response": response,
                "wildguard_raw": raw,
                "wildguard_parsed": parse_wildguard_output(raw),
            }
        )

    print(
        json.dumps(
            {
                "row_id": row.get("id"),
                "prompt": prompt,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Numerically verify HF hidden_states[20] == TL blocks.20.hook_resid_pre."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Layer mapping check requires CUDA.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    vector_cfg = config["vector"]
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg.get("revision"), token=token
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give one concise reason to verify an implementation."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        token=token,
        torch_dtype=getattr(torch, model_cfg["dtype"]),
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    with torch.inference_mode():
        outputs = hf_model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    hf_value = outputs.hidden_states[int(vector_cfg["hf_hidden_state_index"])][0, -1].float().cpu()
    hf_ids = encoded.input_ids.cpu()
    del outputs, encoded, hf_model
    gc.collect()
    torch.cuda.empty_cache()

    tl_model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=model_cfg["dtype"],
        revision=model_cfg.get("revision"),
    )
    tl_tokens, tl_mask = tokenize_prompts(tl_model, [prompt])
    if not torch.equal(hf_ids, tl_tokens.cpu()):
        raise RuntimeError("HF and TransformerLens token IDs differ for the layer mapping prompt")
    captured = {}

    def hook(activation, hook):
        del hook
        captured["value"] = activation[0, -1].detach().float().cpu()
        return activation

    with torch.inference_mode(), tl_model.hooks(
        fwd_hooks=[(vector_cfg["transformer_lens_target_hook"], hook)]
    ):
        tl_model(tl_tokens, attention_mask=tl_mask)
    tl_value = captured["value"]
    cosine = float(torch.nn.functional.cosine_similarity(hf_value, tl_value, dim=0).item())
    relative_l2 = float((hf_value - tl_value).norm().div(hf_value.norm()).item())
    result = {
        "model": model_cfg["name"],
        "model_revision": model_cfg.get("revision"),
        "hf_hidden_state_index": int(vector_cfg["hf_hidden_state_index"]),
        "transformer_lens_target_hook": vector_cfg["transformer_lens_target_hook"],
        "token_count": int(hf_ids.shape[1]),
        "cosine": cosine,
        "relative_l2": relative_l2,
        "minimum_cosine": args.minimum_cosine,
        "pass": cosine >= args.minimum_cosine,
    }
    output = ROOT / config["paths"]["results_root"] / "vector/layer_mapping_check.json"
    write_json(output, result)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise RuntimeError(f"layer mapping cosine {cosine:.6f} is below {args.minimum_cosine}")


if __name__ == "__main__":
    main()

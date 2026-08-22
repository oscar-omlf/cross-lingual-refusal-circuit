#!/usr/bin/env python3
"""Minimal Qwen-only comparison of three frozen behavior directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--num-refusal-prompts", type=int, default=5)
    parser.add_argument("--sycophancy-max-new-tokens", type=int, default=512)
    parser.add_argument("--refusal-max-new-tokens", type=int, default=64)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_circuit.metrics import is_refusal, refusal_log_odds
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        add_direction_to_output,
        edit_direction_output,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )

    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    model_cfg = config["model"]
    baseline_root = _resolve(args.baseline_root)
    baseline_path = baseline_root / "responses.jsonl"
    baseline_meta = json.loads((baseline_root / "metadata.json").read_text())
    if baseline_meta["artifacts"]["responses"]["sha256"] != sha256_file(baseline_path):
        raise ValueError("baseline hash mismatch")
    baselines = [
        row for row in read_jsonl(baseline_path)
        if row["condition"] in {"baseline_positive", "baseline_negative"}
    ]
    positives = sorted(
        (row for row in baselines if row["condition"] == "baseline_positive"),
        key=lambda row: int(row["question_index"]),
    )
    if len(baselines) != 10 or len(positives) != 5:
        raise ValueError("expected q0-q4 positive/negative baselines")

    output_root = _resolve(args.output_root)
    syc_path = output_root / "sycophancy_responses.jsonl"
    refusal_path = output_root / "refusal_summary.json"
    metadata_path = output_root / "metadata.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(output_root)
        for path in (syc_path, refusal_path, metadata_path):
            path.unlink(missing_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(syc_path, baselines)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg.get("revision")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        torch_dtype=getattr(torch, model_cfg["dtype"]),
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    dtype = next(model.parameters()).dtype

    def generate(prompt, block, hook_fn, max_new_tokens):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[1])
        handle = block.register_forward_hook(hook_fn) if hook_fn else None
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
        finally:
            if handle:
                handle.remove()
        return output[0, prompt_length:].tolist()

    vector_root = _resolve(Path(config["paths"]["results_root"])) / "vector"
    syc_specs = (
        ("prompt", 16, "prompt_last", -1.0),
        ("response", 20, "response_avg", -2.0),
    )
    vector_report = []
    for name, layer, stem, normal_coefficient in syc_specs:
        raw_path = vector_root / f"sycophantic_{stem}_diff.pt"
        pos_path = vector_root / f"sycophantic_{stem}_pos_mean.pt"
        neg_path = vector_root / f"sycophantic_{stem}_neg_mean.pt"
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)[layer].float()
        pos = torch.load(pos_path, map_location="cpu", weights_only=True)[layer].float()
        neg = torch.load(neg_path, map_location="cpu", weights_only=True)[layer].float()
        unit = raw / raw.norm()
        pos_coordinate = float(pos @ unit)
        neg_coordinate = float(neg @ unit)
        raw_device = raw.to(args.device, dtype=dtype)
        block = model.model.layers[layer - 1]
        generated = {}
        for condition, edit_mode in (
            ("zero_projection", "zero_projection"),
            ("negative_centroid_clamp", "negative_centroid_clamp"),
            ("centroid_matched_steering", "centroid_shift"),
        ):
            generated[condition] = [
                generate(
                    source["prompt"],
                    block,
                    lambda _m, _i, output, mode=edit_mode: edit_direction_output(
                        output,
                        raw_device,
                        mode=mode,
                        negative_coordinate=neg_coordinate,
                    ),
                    args.sycophancy_max_new_tokens,
                )
                for source in positives
            ]
        if normal_coefficient == -1.0:
            generated["normal_steering"] = generated["centroid_matched_steering"]
        else:
            generated["normal_steering"] = [
                generate(
                    source["prompt"],
                    block,
                    lambda _m, _i, output: add_direction_to_output(
                        output, raw_device, coefficient=normal_coefficient
                    ),
                    args.sycophancy_max_new_tokens,
                )
                for source in positives
            ]
        for condition, token_lists in generated.items():
            for source, token_ids in zip(positives, token_lists, strict=True):
                row = {
                    key: value for key, value in source.items()
                    if key not in {
                        "record_id", "condition", "answer",
                        "generated_token_count", "hit_max_new_tokens",
                        "terminated_with_eos",
                    }
                }
                row.update(
                    {
                        "record_id": f"qwen-geometry-{name}-{condition}-q{int(source['question_index']):02d}",
                        "condition": f"{name}_{condition}",
                        "answer": tokenizer.decode(token_ids, skip_special_tokens=True),
                        "generated_token_count": len(token_ids),
                        "hit_max_new_tokens": len(token_ids) >= args.sycophancy_max_new_tokens,
                        "terminated_with_eos": bool(
                            tokenizer.eos_token_id is not None
                            and token_ids and token_ids[-1] == tokenizer.eos_token_id
                        ),
                        "vector_variant": name,
                        "edit_mode": condition,
                        "layer_one_indexed": layer,
                        "normal_steering_coefficient": (
                            normal_coefficient if condition == "normal_steering" else None
                        ),
                        "generation_reused_from": (
                            "centroid_matched_steering"
                            if condition == "normal_steering" and normal_coefficient == -1.0
                            else None
                        ),
                    }
                )
                write_jsonl(syc_path, [row], mode="a")
        vector_report.append(
            {
                "trait": "sycophancy",
                "variant": name,
                "layer_one_indexed": layer,
                "raw_norm": float(raw.norm()),
                "positive_coordinate": pos_coordinate,
                "negative_coordinate": neg_coordinate,
                "zero_fraction_from_negative": -neg_coordinate / (pos_coordinate - neg_coordinate),
                "normal_steering_coefficient": normal_coefficient,
            }
        )

    refusal_root = ROOT / "results/qwen2.5-7b-instruct/polyrefuse/en/direction_paper_kl0p2"
    refusal_meta = json.loads(
        (refusal_root / "refusal_direction_metadata.json").read_text()
    )
    refusal_unit = torch.load(
        refusal_root / "refusal_direction.pt", map_location="cpu", weights_only=True
    ).float()
    refusal_raw = refusal_unit * float(refusal_meta["raw_direction_norm"])
    refusal_raw_device = refusal_raw.to(args.device, dtype=dtype)
    refusal_negative = float(
        refusal_meta["selected_validation"]["clean_harmless_projection"]
    )
    refusal_layer = int(refusal_meta["layer"])
    refusal_block = model.model.layers[refusal_layer - 1]
    refusal_data = [
        row for row in read_jsonl(
            ROOT / "data/processed/qwen2.5-7b-instruct/polyrefuse/en/val.jsonl"
        )
        if row["label"] == "harmful"
    ][: args.num_refusal_prompts]
    refusal_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in refusal_data
    ]
    refusal_generated = {"normal": [
        generate(prompt, refusal_block, None, args.refusal_max_new_tokens)
        for prompt in refusal_prompts
    ]}
    for condition, edit_mode in (
        ("zero_projection", "zero_projection"),
        ("negative_centroid_clamp", "negative_centroid_clamp"),
        ("centroid_matched_steering", "centroid_shift"),
    ):
        refusal_generated[condition] = [
            generate(
                prompt,
                refusal_block,
                lambda _m, _i, output, mode=edit_mode: edit_direction_output(
                    output,
                    refusal_raw_device,
                    mode=mode,
                    negative_coordinate=refusal_negative,
                ),
                args.refusal_max_new_tokens,
            )
            for prompt in refusal_prompts
        ]
    refusal_generated["normal_steering"] = refusal_generated[
        "centroid_matched_steering"
    ]
    refusal_ids = [int(value) for value in refusal_meta["refusal_token_ids"]]
    refusal_summary = []
    for condition, token_lists in refusal_generated.items():
        first_token_values = []
        for prompt in refusal_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            if condition == "normal":
                hook_fn = None
            else:
                mode = (
                    "centroid_shift"
                    if condition in {"centroid_matched_steering", "normal_steering"}
                    else condition
                )
                hook_fn = lambda _m, _i, output, value=mode: edit_direction_output(
                    output,
                    refusal_raw_device,
                    mode=value,
                    negative_coordinate=refusal_negative,
                )
            handle = refusal_block.register_forward_hook(hook_fn) if hook_fn else None
            try:
                with torch.inference_mode():
                    logits = model(**inputs).logits[:, -1:].float()
            finally:
                if handle:
                    handle.remove()
            first_token_values.append(float(refusal_log_odds(logits, refusal_ids)))
        refusal_summary.append(
            {
                "condition": condition,
                "n_prompts": len(token_lists),
                "refusal_substring_rate": sum(
                    is_refusal(tokenizer.decode(tokens, skip_special_tokens=True))
                    for tokens in token_lists
                ) / len(token_lists),
                "first_token_refusal_metric": sum(first_token_values) / len(first_token_values),
                "mean_generated_tokens": sum(map(len, token_lists)) / len(token_lists),
                "cap_hits": sum(len(tokens) >= args.refusal_max_new_tokens for tokens in token_lists),
                "generation_reused_from": (
                    "centroid_matched_steering" if condition == "normal_steering" else None
                ),
                "generated_text_saved": False,
            }
        )
    refusal_vector = {
        "trait": "refusal",
        "variant": "pregeneration",
        "layer_one_indexed": refusal_layer,
        "raw_norm": float(refusal_raw.norm()),
        "positive_coordinate": float(
            refusal_meta["selected_validation"]["clean_harmful_projection"]
        ),
        "negative_coordinate": refusal_negative,
        "zero_fraction_from_negative": -refusal_negative / (
            float(refusal_meta["selected_validation"]["clean_harmful_projection"])
            - refusal_negative
        ),
        "normal_steering_coefficient": -1.0,
    }
    write_json(refusal_path, {"status": "complete", "vector": refusal_vector, "results": refusal_summary})

    syc_rows = read_jsonl(syc_path)
    if len(syc_rows) != 50:
        raise RuntimeError(f"incomplete sycophancy rows: {len(syc_rows)}/50")
    metadata = {
        "status": "complete",
        "claim_scope": "minimal Qwen-only three-vector geometry diagnostic",
        "model": model_cfg,
        "vectors": [*vector_report, refusal_vector],
        "normal_steering": {
            "refusal": "-1 raw DIM; same as centroid matched",
            "sycophancy_prompt": "-1 raw DIM; same as centroid matched",
            "sycophancy_response": "-2 raw DIM; prior conventional setting",
        },
        "outputs": {
            "sycophancy": str(syc_path.relative_to(ROOT)),
            "refusal": str(refusal_path.relative_to(ROOT)),
        },
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

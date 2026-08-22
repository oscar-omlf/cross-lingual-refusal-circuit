#!/usr/bin/env python3
"""Generate the exact sycophancy steering endpoint used by downstream EAP-IG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_sycophancy.json",
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coefficient", type=float, default=-2.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        read_jsonl,
        sha256_file,
        steer_model_output_timed,
        write_jsonl,
    )

    if args.coefficient >= 0:
        raise ValueError("--coefficient must be negative")
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    model_cfg = config["model"]
    baseline_root = _resolve(args.baseline_root)
    baseline_path = baseline_root / "responses.jsonl"
    baseline_metadata = json.loads(
        (baseline_root / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        baseline_metadata["artifacts"]["responses"]["sha256"]
        != sha256_file(baseline_path)
    ):
        raise ValueError("baseline response hash differs from its metadata")
    baselines = [
        row
        for row in read_jsonl(baseline_path)
        if row["condition"] in {"baseline_positive", "baseline_negative"}
    ]
    positives = sorted(
        (row for row in baselines if row["condition"] == "baseline_positive"),
        key=lambda row: int(row["question_index"]),
    )
    if len(baselines) != 10 or len(positives) != 5:
        raise ValueError("expected the frozen q0-q4 positive/negative baselines")

    output_root = _resolve(args.output_root)
    responses_path = output_root / "responses.jsonl"
    metadata_path = output_root / "metadata.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(output_root)
        responses_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(responses_path, baselines)

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

    layer = int(config["vector"]["paper_layer"])
    vector_path = (
        _resolve(Path(config["paths"]["results_root"]))
        / "vector"
        / f"{config['trait']['name']}_response_avg_diff.pt"
    )
    raw_direction = torch.load(
        vector_path, map_location="cpu", weights_only=True
    )[layer].to(
        device=args.device,
        dtype=next(model.parameters()).dtype,
    )
    block = model.model.layers[layer - 1]

    for source in positives:
        inputs = tokenizer(source["prompt"], return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[1])
        handle = block.register_forward_hook(
            lambda _module, _inputs, output: steer_model_output_timed(
                output,
                raw_direction,
                args.coefficient,
                timing="recurrent",
                prompt_position=-1,
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
            handle.remove()
        token_ids = generated[0, prompt_length:].tolist()
        row = {
            key: value
            for key, value in source.items()
            if key
            not in {
                "record_id",
                "condition",
                "answer",
                "generated_token_count",
                "hit_max_new_tokens",
                "terminated_with_eos",
            }
        }
        row.update(
            {
                "record_id": (
                    f"response-eapig-steering-q{int(source['question_index']):02d}"
                ),
                "condition": "response_eapig_steering",
                "answer": tokenizer.decode(token_ids, skip_special_tokens=True),
                "generated_token_count": len(token_ids),
                "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
                "terminated_with_eos": bool(
                    tokenizer.eos_token_id is not None
                    and token_ids
                    and token_ids[-1] == tokenizer.eos_token_id
                ),
                "vector_variant": "response_avg_diff",
                "coefficient": args.coefficient,
                "timing": "final_prompt_token_and_each_decode_step",
                "layer_one_indexed": layer,
            }
        )
        write_jsonl(responses_path, [row], mode="a")

    rows = read_jsonl(responses_path)
    if len(rows) != 15:
        raise RuntimeError(f"incomplete output: {len(rows)}/15 rows")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "canonical_protocol": False,
        "claim_scope": "q0-q4 exact downstream EAP-IG steering endpoint diagnostic",
        "behavior_protocol_id": config["protocol_id"],
        "model": model_cfg,
        "intervention": {
            "vector": str(vector_path.relative_to(ROOT)),
            "vector_sha256": sha256_file(vector_path),
            "layer_one_indexed": layer,
            "coefficient": args.coefficient,
            "timing": "final prompt token and every cached decode call",
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

#!/usr/bin/env python3
"""Compare matched projection, affine clamp, and centroid steering edits."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import ROOT


MODES = (
    "zero_projection",
    "negative_centroid_clamp",
    "centroid_shift",
)


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
        "--baseline-root",
        type=Path,
        required=True,
        help="Completed q0-q4 prompt-direction search containing reusable baselines.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        edit_direction_output,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Direction geometry diagnostic requires a CUDA device.")

    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    baseline_root = _resolve(args.baseline_root)
    baseline_path = baseline_root / "responses.jsonl"
    baseline_metadata_path = baseline_root / "metadata.json"
    baseline_metadata = json.loads(
        baseline_metadata_path.read_text(encoding="utf-8")
    )
    if baseline_metadata["artifacts"]["responses"]["sha256"] != sha256_file(
        baseline_path
    ):
        raise ValueError("baseline response hash differs from its metadata")
    baseline_rows = read_jsonl(baseline_path)
    reusable = [
        row
        for row in baseline_rows
        if row["condition"] in {"baseline_positive", "baseline_negative"}
    ]
    if len(reusable) != 10:
        raise ValueError("expected five positive and five negative baseline rows")
    positives = sorted(
        (row for row in reusable if row["condition"] == "baseline_positive"),
        key=lambda row: int(row["question_index"]),
    )
    negatives = sorted(
        (row for row in reusable if row["condition"] == "baseline_negative"),
        key=lambda row: int(row["question_index"]),
    )
    if [row["question_index"] for row in positives] != list(range(5)):
        raise ValueError("positive baseline must contain q0-q4")
    if [row["question_index"] for row in negatives] != list(range(5)):
        raise ValueError("negative baseline must contain q0-q4")

    output_root = _resolve(args.output_root)
    responses_path = output_root / "responses.jsonl"
    metadata_path = output_root / "metadata.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already contains output")
        responses_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(responses_path, reusable)

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
    vector_root = _resolve(Path(config["paths"]["results_root"])) / "vector"
    vector_specs = (
        {
            "name": "prompt",
            "layer": 16,
            "vector": vector_root / "sycophantic_prompt_last_diff.pt",
            "positive_mean": vector_root / "sycophantic_prompt_last_pos_mean.pt",
            "negative_mean": vector_root / "sycophantic_prompt_last_neg_mean.pt",
        },
        {
            "name": "response",
            "layer": 20,
            "vector": vector_root / "sycophantic_response_avg_diff.pt",
            "positive_mean": vector_root / "sycophantic_response_avg_pos_mean.pt",
            "negative_mean": vector_root / "sycophantic_response_avg_neg_mean.pt",
        },
    )

    vector_metadata = []
    for spec in vector_specs:
        layer = int(spec["layer"])
        raw_stack = torch.load(
            spec["vector"], map_location="cpu", weights_only=True
        )
        positive_stack = torch.load(
            spec["positive_mean"], map_location="cpu", weights_only=True
        )
        negative_stack = torch.load(
            spec["negative_mean"], map_location="cpu", weights_only=True
        )
        raw = raw_stack[layer].float()
        positive_mean = positive_stack[layer].float()
        negative_mean = negative_stack[layer].float()
        unit = raw / raw.norm()
        positive_coordinate = float(positive_mean @ unit)
        negative_coordinate = float(negative_mean @ unit)
        raw = raw.to(
            device=args.device,
            dtype=next(model.parameters()).dtype,
        )
        block = model.model.layers[layer - 1]

        for mode in MODES:
            for source in positives:
                captures = {"prefill": None, "decode": []}

                def hook(_module, _inputs, output):
                    changed = edit_direction_output(
                        output,
                        raw,
                        mode=mode,
                        negative_coordinate=negative_coordinate,
                    )
                    tensor = (
                        changed
                        if torch.is_tensor(changed)
                        else changed[0]
                    )
                    coordinate = torch.einsum(
                        "...d,d->...",
                        tensor.float(),
                        unit.to(tensor.device),
                    )
                    if tensor.shape[1] > 1:
                        captures["prefill"] = float(coordinate[0, -1].item())
                    else:
                        captures["decode"].append(float(coordinate[0, -1].item()))
                    return changed

                inputs = tokenizer(source["prompt"], return_tensors="pt")
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
                prompt_length = int(inputs["input_ids"].shape[1])
                handle = block.register_forward_hook(hook)
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
                condition = f"{spec['name']}_{mode}"
                row.update(
                    {
                        "record_id": (
                            f"direction-geometry-{condition}-"
                            f"q{int(source['question_index']):02d}"
                        ),
                        "condition": condition,
                        "answer": tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                        ),
                        "generated_token_count": len(token_ids),
                        "hit_max_new_tokens": len(token_ids)
                        >= args.max_new_tokens,
                        "terminated_with_eos": bool(
                            tokenizer.eos_token_id is not None
                            and token_ids
                            and token_ids[-1] == tokenizer.eos_token_id
                        ),
                        "vector_variant": spec["name"],
                        "edit_mode": mode,
                        "layer_one_indexed": layer,
                        "prefill_last_coordinate_after": captures["prefill"],
                        "response_coordinate_after_mean": (
                            sum(captures["decode"]) / len(captures["decode"])
                            if captures["decode"]
                            else None
                        ),
                        "response_coordinate_samples": len(captures["decode"]),
                    }
                )
                write_jsonl(responses_path, [row], mode="a")

        vector_metadata.append(
            {
                "variant": spec["name"],
                "layer_one_indexed": layer,
                "raw_norm": float(raw.float().norm().item()),
                "positive_coordinate": positive_coordinate,
                "negative_coordinate": negative_coordinate,
                "zero_fraction_from_negative": (
                    -negative_coordinate
                    / (positive_coordinate - negative_coordinate)
                ),
                "artifacts": {
                    key: {
                        "path": str(spec[key].relative_to(ROOT)),
                        "sha256": sha256_file(spec[key]),
                    }
                    for key in ("vector", "positive_mean", "negative_mean")
                },
            }
        )

    final = read_jsonl(responses_path)
    if len(final) != 40 or len({row["record_id"] for row in final}) != 40:
        raise RuntimeError(f"incomplete geometry diagnostic: {len(final)}/40")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "canonical_protocol": False,
        "claim_scope": "q0-q4 matched direction-geometry development diagnostic",
        "behavior_protocol_id": config["protocol_id"],
        "git_commit": _git_commit(),
        "model": model_cfg,
        "vectors": vector_metadata,
        "modes": list(MODES),
        "intervention_timing": (
            "native layer output, every active prefill position and every "
            "cached decoding call"
        ),
        "baseline": {
            "source": str(baseline_path.relative_to(ROOT)),
            "sha256": sha256_file(baseline_path),
            "reused_conditions": ["baseline_positive", "baseline_negative"],
        },
        "decoding": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "outputs": [
            {
                "path": str(responses_path.relative_to(ROOT)),
                "rows": len(final),
                "sha256": sha256_file(responses_path),
            }
        ],
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

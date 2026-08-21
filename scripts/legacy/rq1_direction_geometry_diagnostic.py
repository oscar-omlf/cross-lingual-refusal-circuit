#!/usr/bin/env python3
"""Compare matched geometry edits for the selected Gemma refusal direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _rq1_bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--direction-path",
        type=Path,
        default=Path("results/gemma-2-2b-it/en/rq1/refusal_direction.pt"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("results/gemma-2-2b-it/en/rq1/refusal_direction_metadata.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "results/gemma-2-2b-it/en/rq1/direction_geometry_diagnostic.json"
        ),
    )
    args = parser.parse_args()

    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.interventions import (
        activation_addition_hook,
        direction_ablation_hook,
        direction_coordinate_clamp_hook,
    )
    from refusal_circuit.metrics import is_refusal, refusal_log_odds
    from refusal_circuit.model import format_instructions, load_model, tokenize_prompts
    from refusal_circuit.paths import read_json, write_json

    metadata = read_json(args.metadata_path)
    if metadata["model"] != args.model:
        raise ValueError("direction metadata model does not match --model")
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(
        args.direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    raw = direction * float(metadata["raw_direction_norm"])
    negative_coordinate = float(
        metadata["selected_validation"]["clean_harmless_projection"]
    )
    target_hook = metadata["target_hook"]
    prompts = format_instructions(
        model.tokenizer,
        [
            row["instruction"]
            for row in load_processed_split(args.split, "harmful")[
                : args.num_prompts
            ]
        ],
    )
    refusal_ids = [int(value) for value in metadata["refusal_token_ids"]]
    modes = {
        "normal": [],
        "zero_projection": [
            (target_hook, direction_ablation_hook(direction, position=None))
        ],
        "negative_centroid_clamp": [
            (
                target_hook,
                direction_coordinate_clamp_hook(
                    direction,
                    target_coordinate=negative_coordinate,
                    position=None,
                ),
            )
        ],
        "centroid_shift": [
            (
                target_hook,
                activation_addition_hook(raw, coefficient=-1.0, position=None),
            )
        ],
    }
    results = []
    for condition, intervention_hooks in modes.items():
        refusal_count = 0
        first_token_metrics = []
        generated_lengths = []
        prompt_coordinates = []
        response_coordinates = []
        for prompt in prompts:
            tokens, attention_mask = tokenize_prompts(
                model, [prompt], max_length=args.max_length
            )
            capture = {"prefill": None, "decode": []}

            def capture_hook(activations, hook):
                coordinate = torch.einsum(
                    "...d,d->...",
                    activations.float(),
                    direction.float(),
                )
                if activations.shape[1] > 1:
                    capture["prefill"] = float(
                        coordinate[0, int(metadata["position"])].item()
                    )
                else:
                    capture["decode"].append(float(coordinate[0, -1].item()))

            hooks = [*intervention_hooks, (target_hook, capture_hook)]
            with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
                logits = model(tokens, attention_mask=attention_mask)
            first_token_metrics.append(
                float(
                    refusal_log_odds(
                        logits[:, -1:].float(),
                        refusal_ids,
                    ).item()
                )
            )
            capture = {"prefill": None, "decode": []}
            with model.hooks(
                fwd_hooks=[
                    *intervention_hooks,
                    (target_hook, capture_hook),
                ]
            ):
                generated = model.generate(
                    tokens,
                    max_new_tokens=args.max_new_tokens,
                    stop_at_eos=True,
                    do_sample=False,
                    use_past_kv_cache=True,
                    return_type="tokens",
                    verbose=False,
                )
            token_ids = generated[0, tokens.shape[1] :].tolist()
            completion = model.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )
            refusal_count += int(is_refusal(completion))
            generated_lengths.append(len(token_ids))
            prompt_coordinates.append(float(capture["prefill"]))
            response_coordinates.extend(capture["decode"])
        results.append(
            {
                "condition": condition,
                "n_prompts": len(prompts),
                "refusal_substring_rate": refusal_count / len(prompts),
                "first_token_refusal_metric": sum(first_token_metrics)
                / len(first_token_metrics),
                "mean_generated_tokens": sum(generated_lengths)
                / len(generated_lengths),
                "cap_hits": sum(
                    length >= args.max_new_tokens for length in generated_lengths
                ),
                "prompt_coordinate_mean": sum(prompt_coordinates)
                / len(prompt_coordinates),
                "response_coordinate_mean": (
                    sum(response_coordinates) / len(response_coordinates)
                    if response_coordinates
                    else None
                ),
                "generated_text_saved": False,
            }
        )
    output = {
        "schema_version": 1,
        "status": "complete",
        "claim_scope": "eight-prompt refusal direction-geometry diagnostic",
        "model": args.model,
        "split": args.split,
        "target_hook": target_hook,
        "target_position": int(metadata["position"]),
        "raw_direction_norm": float(metadata["raw_direction_norm"]),
        "positive_coordinate": float(
            metadata["selected_validation"]["clean_harmful_projection"]
        ),
        "negative_coordinate": negative_coordinate,
        "zero_fraction_from_negative": (
            -negative_coordinate
            / (
                float(
                    metadata["selected_validation"]["clean_harmful_projection"]
                )
                - negative_coordinate
            )
        ),
        "intervention_timing": (
            "target residual site, every active prefill position and every "
            "cached decoding call"
        ),
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    write_json(args.output_path, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose whether RQ1 graph sources reconstruct a target residual state."""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--direction-path", type=Path, default=Path("results/rq1/refusal_direction.pt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("results/rq1/refusal_direction_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rq1"))
    return parser.parse_args()


def main():
    args = parse_args()
    import json

    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.graph import make_projection_graph
    from refusal_circuit.interventions import all_layer_ablation_hooks
    from refusal_circuit.model import format_instructions, load_model, tokenize_prompts
    from refusal_circuit.paths import read_json

    metadata = read_json(args.metadata_path)
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(args.direction_path, map_location=next(model.parameters()).device, weights_only=True)
    graph = make_projection_graph(model, int(metadata["layer"]))
    target_hook = metadata["target_hook"]
    target_position = int(metadata["position"])
    prompts = format_instructions(
        model.tokenizer,
        [row["instruction"] for row in load_processed_split(args.split, "harmful")[: args.num_prompts]],
    )
    tokens, attention_mask = tokenize_prompts(model, prompts, max_length=args.max_length)
    source_names = {edge.parent.out_hook for edge in graph.nodes["logits"].parent_edges}
    attn_out_names = {f"blocks.{layer}.hook_attn_out" for layer in range(int(metadata["layer"]))}

    def run(*, ablated: bool):
        cache: dict[str, torch.Tensor] = {}
        target: dict[str, torch.Tensor] = {}

        def cache_hook(name):
            def hook(activations, hook):
                cache[name] = activations.detach().clone()
                return activations

            return hook

        def target_capture(activations, hook):
            target["activation"] = activations.detach().clone()
            return activations

        hooks = []
        if ablated:
            hooks += all_layer_ablation_hooks(model, direction)
        hooks += [(name, cache_hook(name)) for name in source_names | attn_out_names]
        hooks += [(target_hook, target_capture)]
        with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
            model(tokens, attention_mask=attention_mask)
        reconstructed = None
        for edge in graph.nodes["logits"].parent_edges:
            value = cache[edge.parent.out_hook][edge.parent.index]
            reconstructed = value if reconstructed is None else reconstructed + value
        actual = target["activation"]
        error = reconstructed - actual
        unit = direction.to(actual)
        unit = unit / unit.norm().clamp_min(1e-8)
        attention_bias_like = []
        for layer in range(int(metadata["layer"])):
            result = cache[f"blocks.{layer}.attn.hook_result"].sum(dim=2)
            attn_out = cache[f"blocks.{layer}.hook_attn_out"]
            difference = attn_out - result
            attention_bias_like.append(
                {
                    "layer": layer,
                    "mean_projection": float((difference[:, target_position] @ unit).mean()),
                    "rms": float(difference.float().pow(2).mean().sqrt()),
                }
            )
        return {
            "mean_actual_projection": float((actual[:, target_position] @ unit).mean()),
            "mean_reconstructed_projection": float((reconstructed[:, target_position] @ unit).mean()),
            "mean_error_projection": float((error[:, target_position] @ unit).mean()),
            "max_abs_error": float(error.float().abs().max()),
            "rms_error": float(error.float().pow(2).mean().sqrt()),
            "n_sources": len(graph.nodes["logits"].parent_edges),
            "attention_result_to_output": attention_bias_like,
        }

    report = {"clean": run(ablated=False), "corrupt": run(ablated=True)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "residual_reconstruction_debug.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved reconstruction diagnostics to {output_path}")


if __name__ == "__main__":
    main()

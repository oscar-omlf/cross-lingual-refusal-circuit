#!/usr/bin/env python3
"""Diagnose the EAP terminal-gradient channel for the RQ1 residual metric.

This is a narrow, non-generative check.  It runs one harmful prompt and
records the gradient at the selected residual-stream hook as well as the
resulting scores of edges into the terminal metric node.  It exists to guard
against attributing a residual metric while accidentally leaving every edge
into that metric unscored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--direction-path", type=Path, default=Path("results/gemma-2-2b-it/en/rq1/refusal_direction.pt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("results/gemma-2-2b-it/en/rq1/refusal_direction_metadata.json"))
    parser.add_argument("--output-path", type=Path, default=Path("results/gemma-2-2b-it/en/rq1/terminal_gradient_debug.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.eapig import (
        _accumulate_terminal_metric_scores,
        _direction_metric_hooks,
        _input_requires_grad,
        _make_eap_hooks,
    )
    from refusal_circuit.graph import make_projection_graph
    from refusal_circuit.interventions import all_layer_ablation_hooks
    from refusal_circuit.model import format_instructions, load_model, tokenize_prompts
    from refusal_circuit.paths import read_json, write_json

    metadata = read_json(args.metadata_path)
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(
        args.direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    prompt = format_instructions(
        model.tokenizer, [load_processed_split(args.split, "harmful")[0]["instruction"]]
    )
    tokens, attention_mask = tokenize_prompts(model, prompt, max_length=args.max_length)
    graph = make_projection_graph(model, int(metadata["layer"]))
    device = next(model.parameters()).device
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=torch.float32)
    fwd_corrupt, fwd_clean, bwd_hooks, _ = _make_eap_hooks(
        model, graph, tokens.shape[0], tokens.shape[1], scores, direction
    )

    # Match the two endpoint cache passes used by the attribution runner.
    with torch.no_grad(), model.hooks(
        fwd_hooks=fwd_corrupt + all_layer_ablation_hooks(model, direction, strength=1.0)
    ):
        model(tokens, attention_mask=attention_mask)
    with torch.no_grad(), model.hooks(fwd_hooks=fwd_clean):
        model(tokens, attention_mask=attention_mask)

    observed_gradients: list[torch.Tensor] = []

    def observe_terminal_gradient(gradients: torch.Tensor, hook):
        observed_gradients.append(gradients.detach().float().cpu())
        return None

    capture, metric_hook = _direction_metric_hooks(
        metadata["target_hook"], int(metadata["position"]), direction
    )
    model.zero_grad(set_to_none=True)
    with model.hooks(
        fwd_hooks=[("hook_embed", _input_requires_grad)]
        + all_layer_ablation_hooks(model, direction, strength=0.0)
        + metric_hook,
        bwd_hooks=bwd_hooks + [(metadata["target_hook"], observe_terminal_gradient)],
    ):
        model(tokens, attention_mask=attention_mask)
        if capture.activation is None:
            raise RuntimeError("The target projection hook did not fire.")
        capture.activation.retain_grad()
        metric = capture.mean()
        metric.backward()

    _accumulate_terminal_metric_scores(scores, _, graph, capture)

    terminal_column = scores[:, -1]
    capture_grad = capture.activation.grad
    target_position = int(metadata["position"])
    payload = {
        "target_hook": metadata["target_hook"],
        "target_position": target_position,
        "metric": float(metric.detach().cpu()),
        "capture_gradient_present": capture_grad is not None,
        "capture_gradient_l2": float(capture_grad.float().norm().cpu()) if capture_grad is not None else None,
        "backward_hook_calls": len(observed_gradients),
        "backward_hook_l2": [float(value.norm()) for value in observed_gradients],
        "backward_hook_nonzero_positions": [
            int((value.norm(dim=-1) > 0).sum()) for value in observed_gradients
        ],
        "terminal_edge_count": int(terminal_column.numel()),
        "terminal_edge_nonzero_scores": int((terminal_column != 0).sum().item()),
        "terminal_edge_max_abs_score": float(terminal_column.abs().max().item()),
        "all_edge_nonzero_scores": int((scores != 0).sum().item()),
    }
    write_json(args.output_path, payload)
    print(payload)
    # A metric that directly reads an activation *inside* a HookPoint bypasses
    # that HookPoint's module-level backward hook.  The retained tensor gradient
    # above is therefore the intended terminal-gradient route.
    if not payload["capture_gradient_present"] or payload["terminal_edge_nonzero_scores"] == 0:
        raise RuntimeError("Terminal residual metric did not produce terminal edge scores.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract a prompt-based persona direction and emit the RQ1 metric contract.

Produces exactly the two artefacts the downstream refusal pipeline consumes --
``persona_direction.pt`` (a single ``[d_model]`` unit vector) and
``persona_direction_metadata.json`` (with ``model``/``layer``/``position``/
``target_hook``) -- so ``scripts/rq1_run_eapig.py`` and
``scripts/rq1_evaluate_faithfulness.py`` run unchanged against a persona trait.

No OpenAI judge is used: the direction is the pos/neg system-prompt
difference-in-means at the last prompt token, and the layer is chosen by
held-out pos/neg projection separation.  A per-layer raw stack is also saved
for parity with persona-vectors' ``{trait}_prompt_last_diff.pt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", default="evil")
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-layer", type=int, default=1)
    parser.add_argument("--max-layer", type=int, default=None, help="Exclusive candidate-layer bound.")
    parser.add_argument("--late-layer-fraction", type=float, default=0.8)
    parser.add_argument("--processed-dir", type=Path, default=None,
                        help="Defaults to data/processed/persona/<trait>.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to results/persona/<trait>.")
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        evaluate_persona_candidate,
        layer_diff_stack,
        prompt_last_candidates,
        select_persona_candidate,
    )

    processed_dir = args.processed_dir or (ROOT / "data" / "processed" / "persona" / args.trait)
    output_dir = args.output_dir or (ROOT / "results" / "persona" / args.trait)

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    train_pos = load_processed_split("train", "harmful", base_dir=processed_dir)
    train_neg = load_processed_split("train", "harmless", base_dir=processed_dir)
    val_pos = load_processed_split("val", "harmful", base_dir=processed_dir)
    val_neg = load_processed_split("val", "harmless", base_dir=processed_dir)

    train_pos_prompts = format_instructions(model.tokenizer, [row["instruction"] for row in train_pos])
    train_neg_prompts = format_instructions(model.tokenizer, [row["instruction"] for row in train_neg])
    val_pos_prompts = format_instructions(model.tokenizer, [row["instruction"] for row in val_pos])
    val_neg_prompts = format_instructions(model.tokenizer, [row["instruction"] for row in val_neg])

    max_layer = args.max_layer if args.max_layer is not None else model.cfg.n_layers
    layers = list(range(max(args.min_layer, 1), max_layer))
    if not layers:
        raise ValueError("Candidate layer range is empty; check --min-layer/--max-layer.")

    candidates = prompt_last_candidates(
        model, train_pos_prompts, train_neg_prompts,
        layers=layers, batch_size=args.batch_size, max_length=args.max_length,
    )
    rows = [
        evaluate_persona_candidate(
            model, candidate,
            val_pos_prompts=val_pos_prompts, val_neg_prompts=val_neg_prompts,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        for candidate in candidates
    ]
    for row in rows:
        row["eligible"] = bool(
            row["separation"] > 0.0
            and max(args.min_layer, 1) <= row["layer"] < args.late_layer_fraction * model.cfg.n_layers
        )
    best_row = select_persona_candidate(
        rows, n_layers=model.cfg.n_layers,
        min_layer=args.min_layer, late_layer_fraction=args.late_layer_fraction,
    )
    best = next(
        candidate for candidate in candidates
        if candidate.layer == int(best_row["layer"]) and candidate.position == int(best_row["position"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    validation = pd.DataFrame(rows).sort_values(["eligible", "cohens_d"], ascending=[False, False])
    validation.to_csv(output_dir / "persona_direction_validation.csv", index=False)

    direction_path = output_dir / "persona_direction.pt"
    torch.save(best.unit_vector.cpu(), direction_path)

    layer_order, diff_stack = layer_diff_stack(candidates)
    stack_path = output_dir / f"{args.trait}_prompt_last_diff.pt"
    torch.save({"layers": layer_order, "diff": diff_stack.cpu()}, stack_path)

    metadata = {
        "trait": args.trait,
        "model": args.model,
        "method": "difference_in_means(resid_pre pos - neg, last prompt token)",
        "variant": "persona_vectors prompt_last (judge-free, prompt-based)",
        "selection_protocol": "max Cohen's d of held-out pos/neg projection separation",
        "target_hook": f"blocks.{best.layer}.hook_resid_pre",
        "layer": best.layer,
        "position": best.position,
        "raw_direction_norm": best.raw_vector.norm().item(),
        "normalised": True,
        "train_counts": {"pos": len(train_pos), "neg": len(train_neg)},
        "validation_counts": {"pos": len(val_pos), "neg": len(val_neg)},
        "candidate_layers": layers,
        "post_instruction_positions": [best.position],
        "late_layer_fraction": args.late_layer_fraction,
        "selected_validation": best_row,
        "per_layer_diff_file": stack_path.name,
    }
    write_json(output_dir / "persona_direction_metadata.json", metadata)
    print(f"Selected persona direction: trait={args.trait}, layer={best.layer}, position={best.position}")
    print(f"  Cohen's d={best_row['cohens_d']:.3f}, separation={best_row['separation']:.4f}")
    print(f"Saved direction to {direction_path}")
    print(f"Saved per-layer diff stack to {stack_path}")


if __name__ == "__main__":
    main()

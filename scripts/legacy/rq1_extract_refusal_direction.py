#!/usr/bin/env python3
"""Extract a paper-faithful Gemma refusal direction for RQ1."""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


GEMMA_IT_REFUSAL_TOKEN_ID = 235285  # ``I``; Arditi et al., Table 4.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=None,
        help="Override dynamically discovered post-instruction offsets (debug only).",
    )
    parser.add_argument("--min-layer", type=int, default=0)
    parser.add_argument("--max-layer", type=int, default=None, help="Exclusive candidate-layer bound.")
    parser.add_argument("--addition-coefficient", type=float, default=1.0)
    parser.add_argument("--kl-threshold", type=float, default=0.1)
    parser.add_argument("--late-layer-fraction", type=float, default=0.8)
    parser.add_argument(
        "--refusal-token-ids",
        type=int,
        nargs="+",
        default=(GEMMA_IT_REFUSAL_TOKEN_ID,),
        help="Paper refusal-token set R; Gemma IT uses {235285}.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/gemma-2-2b-it/en/rq1"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/gemma-2-2b-it/en/rq1"),
        help="Processed-data namespace containing train.jsonl and val.jsonl.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.direction import (
        difference_in_means_candidates,
        evaluate_candidate,
        select_paper_candidate,
    )
    from refusal_circuit.model import (
        format_instructions,
        load_model,
        post_instruction_positions,
    )
    from refusal_circuit.paths import write_json

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    train_harmful = load_processed_split("train", "harmful", data_dir=args.data_dir)
    train_harmless = load_processed_split("train", "harmless", data_dir=args.data_dir)
    val_harmful = load_processed_split("val", "harmful", data_dir=args.data_dir)
    val_harmless = load_processed_split("val", "harmless", data_dir=args.data_dir)

    train_harmful_prompts = format_instructions(model.tokenizer, [row["instruction"] for row in train_harmful])
    train_harmless_prompts = format_instructions(model.tokenizer, [row["instruction"] for row in train_harmless])
    val_harmful_prompts = format_instructions(model.tokenizer, [row["instruction"] for row in val_harmful])
    val_harmless_prompts = format_instructions(model.tokenizer, [row["instruction"] for row in val_harmless])

    positions = tuple(args.positions) if args.positions is not None else post_instruction_positions(model.tokenizer)
    max_layer = args.max_layer if args.max_layer is not None else model.cfg.n_layers
    layers = list(range(args.min_layer, max_layer))
    if not layers:
        raise ValueError("Candidate layer range is empty.")

    candidates = difference_in_means_candidates(
        model,
        train_harmful_prompts,
        train_harmless_prompts,
        layers=layers,
        positions=positions,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    rows = [
        evaluate_candidate(
            model,
            candidate,
            harmful_prompts=val_harmful_prompts,
            harmless_prompts=val_harmless_prompts,
            refusal_token_ids=list(args.refusal_token_ids),
            addition_coefficient=args.addition_coefficient,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        for candidate in candidates
    ]
    for row in rows:
        row["paper_eligible"] = bool(
            row["induce_score"] > 0.0
            and row["harmless_ablation_kl"] < args.kl_threshold
            and row["layer"] < args.late_layer_fraction * model.cfg.n_layers
        )
    best_row = select_paper_candidate(
        rows,
        n_layers=model.cfg.n_layers,
        kl_threshold=args.kl_threshold,
        late_layer_fraction=args.late_layer_fraction,
    )
    validation = pd.DataFrame(rows).sort_values(
        ["paper_eligible", "bypass_score"], ascending=[False, True]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.output_dir / "refusal_direction_validation.csv"
    validation.to_csv(validation_path, index=False)
    best = next(
        candidate
        for candidate in candidates
        if candidate.layer == int(best_row["layer"]) and candidate.position == int(best_row["position"])
    )
    direction_path = args.output_dir / "refusal_direction.pt"
    torch.save(best.unit_vector.cpu(), direction_path)
    metadata = {
        "model": args.model,
        "method": "difference_in_means(resid_pre harmful - harmless)",
        "selection_protocol": "Arditi et al. (2024), Appendix C hard constraints and minimum bypass score",
        "target_hook": f"blocks.{best.layer}.hook_resid_pre",
        "layer": best.layer,
        "position": best.position,
        "raw_direction_norm": best.raw_vector.norm().item(),
        "normalised": True,
        "train_counts": {"harmful": len(train_harmful), "harmless": len(train_harmless)},
        "validation_counts": {"harmful": len(val_harmful), "harmless": len(val_harmless)},
        "data_dir": str(args.data_dir),
        "candidate_layers": layers,
        "post_instruction_positions": list(positions),
        "addition_coefficient": args.addition_coefficient,
        "addition_positions": "all",
        "refusal_token_ids": list(args.refusal_token_ids),
        "kl_threshold": args.kl_threshold,
        "late_layer_fraction": args.late_layer_fraction,
        "selection_note": (
            "Selected by minimum harmful bypass_score after enforcing induce_score > 0, "
            "harmless last-token KL < threshold, and layer < late_layer_fraction * n_layers."
        ),
        "selected_validation": best_row,
    }
    write_json(args.output_dir / "refusal_direction_metadata.json", metadata)
    print(f"Selected paper-eligible direction: layer={best.layer}, position={best.position}")
    print(f"Saved selected direction to {direction_path}")
    print(f"Saved validation table to {validation_path}")


if __name__ == "__main__":
    main()

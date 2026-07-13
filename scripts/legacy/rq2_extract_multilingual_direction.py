#!/usr/bin/env python3
"""Extract one validated Qwen refusal direction for one PolyRefuse language.

This is the language-wise analogue of the Arditi et al. difference-in-means
procedure.  It intentionally saves *all* candidate vectors as well as the
individually selected one: later comparison uses a common English-selected
site, while intervention efficacy uses each language's selected site.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--positions", type=int, nargs="+", default=None)
    parser.add_argument("--min-layer", type=int, default=0)
    parser.add_argument("--max-layer", type=int, default=None, help="Exclusive candidate-layer bound.")
    parser.add_argument("--addition-coefficient", type=float, default=1.0)
    parser.add_argument("--kl-threshold", type=float, default=0.1)
    parser.add_argument("--late-layer-fraction", type=float, default=0.8)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    from refusal_circuit.model import format_instructions, load_model, post_instruction_positions
    from refusal_circuit.paths import experiment_data_dir, experiment_results_dir, read_json, write_json
    from refusal_circuit.polyrefuse import POLYREFUSE_LANGUAGES, refusal_token_ids

    if args.language not in POLYREFUSE_LANGUAGES:
        raise ValueError(f"Unsupported language {args.language!r}; choose from {POLYREFUSE_LANGUAGES}.")
    data_dir = args.data_dir or experiment_data_dir(args.model, "polyrefuse", args.language)
    output_dir = args.output_dir or experiment_results_dir(args.model, "polyrefuse", args.language) / "direction"
    preparation_path = data_dir / "preparation_metadata.json"
    if not preparation_path.exists():
        raise FileNotFoundError(
            f"Missing {preparation_path}. Run scripts/rq2_prepare_polyrefuse.py first."
        )
    preparation = read_json(preparation_path)
    if preparation.get("model") != args.model:
        raise ValueError(
            f"Prepared data is for {preparation.get('model')}, not requested model {args.model}."
        )

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    refusal_ids = [int(value) for value in preparation.get("refusal_token_ids", [])]
    if not refusal_ids:
        refusal_ids = refusal_token_ids(model.tokenizer, args.language)
    train_harmful = load_processed_split("train", "harmful", data_dir=data_dir)
    train_harmless = load_processed_split("train", "harmless", data_dir=data_dir)
    val_harmful = load_processed_split("val", "harmful", data_dir=data_dir)
    val_harmless = load_processed_split("val", "harmless", data_dir=data_dir)
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
            refusal_token_ids=refusal_ids,
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
    best = next(
        candidate
        for candidate in candidates
        if candidate.layer == int(best_row["layer"]) and candidate.position == int(best_row["position"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = pd.DataFrame(rows).sort_values(
        ["paper_eligible", "bypass_score"], ascending=[False, True]
    )
    validation.to_csv(output_dir / "refusal_direction_validation.csv", index=False)
    torch.save(best.unit_vector.cpu(), output_dir / "refusal_direction.pt")
    torch.save(
        {
            "layers": layers,
            "positions": list(positions),
            "vectors": torch.stack([candidate.raw_vector.cpu() for candidate in candidates]),
            "order": "position_major_then_layer",
        },
        output_dir / "candidate_directions.pt",
    )
    metadata = {
        "model": args.model,
        "dataset": "PolyRefuse",
        "language": args.language,
        "method": "difference_in_means(resid_pre harmful - harmless)",
        "selection_protocol": (
            "Arditi et al. (2024), Appendix C hard constraints and minimum bypass score; "
            "language-local first-token refusal metric following the PolyRefuse release."
        ),
        "target_hook": f"blocks.{best.layer}.hook_resid_pre",
        "layer": best.layer,
        "position": best.position,
        "raw_direction_norm": best.raw_vector.norm().item(),
        "normalised": True,
        "train_counts": {"harmful": len(train_harmful), "harmless": len(train_harmless)},
        "validation_counts": {"harmful": len(val_harmful), "harmless": len(val_harmless)},
        "candidate_layers": layers,
        "post_instruction_positions": list(positions),
        "addition_coefficient": args.addition_coefficient,
        "addition_positions": "all",
        "refusal_token_ids": refusal_ids,
        "refusal_token_source": preparation.get("filter_protocol"),
        "kl_threshold": args.kl_threshold,
        "late_layer_fraction": args.late_layer_fraction,
        "data_dir": str(data_dir),
        "preparation_metadata": str(preparation_path),
        "candidate_directions": "candidate_directions.pt",
        "selected_validation": best_row,
    }
    write_json(output_dir / "refusal_direction_metadata.json", metadata)
    print(
        f"{args.language}: selected layer={best.layer}, position={best.position}; "
        f"saved direction, all candidates, and validation table under {output_dir}"
    )


if __name__ == "__main__":
    main()

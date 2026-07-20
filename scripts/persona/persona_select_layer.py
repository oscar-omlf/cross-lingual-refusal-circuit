#!/usr/bin/env python3
"""Aggregate judged layer-wise steering results using the paper's selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--scored-root", type=Path, default=None)
    args = parser.parse_args()

    import pandas as pd

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validation = config["steering_validation"]
    selection = config["layer_selection"]
    scored_root = (
        _resolve(args.scored_root)
        if args.scored_root
        else _resolve(config["paths"]["intermediate_root"]) / "steering"
    )
    layers = [int(value) for value in selection["candidate_layers_one_indexed"]]
    coefficients = [float(value) for value in validation["coefficients"]]
    expected_rows = int(validation["questions"]) * int(validation["rollouts_per_question"])
    rows = []
    inputs = []
    for layer in layers:
        layer_root = scored_root / f"layer_{layer:02d}" / "scored"
        for coefficient in coefficients:
            label = f"{coefficient:+g}".replace("+", "pos").replace("-", "neg").replace(".", "p")
            path = layer_root / f"coef_{label}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"missing judged layer-sweep output: {path}")
            scored = read_jsonl(path)
            if len(scored) != expected_rows:
                raise RuntimeError(f"{path} contains {len(scored)}/{expected_rows} rows")
            if any(item.get("judge_protocol") != "top20_expected_0_100" for item in scored):
                raise RuntimeError(f"noncanonical judge record in {path}")
            trait_scores = [float(item["trait_score"]) for item in scored if item["trait_score"] is not None]
            coherence_scores = [
                float(item["coherence_score"])
                for item in scored
                if item["coherence_score"] is not None
            ]
            if not trait_scores or not coherence_scores:
                raise RuntimeError(f"no aggregatable scores in {path}")
            rows.append(
                {
                    "layer_one_indexed": layer,
                    "coefficient": coefficient,
                    "responses": len(scored),
                    "valid_trait_scores": len(trait_scores),
                    "valid_coherence_scores": len(coherence_scores),
                    "mean_trait_score": sum(trait_scores) / len(trait_scores),
                    "mean_coherence_score": sum(coherence_scores) / len(coherence_scores),
                }
            )
            inputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})

    frame = pd.DataFrame(rows).sort_values(["coefficient", "layer_one_indexed"])
    winners = []
    for coefficient, subset in frame.groupby("coefficient"):
        winner = subset.sort_values(
            ["mean_trait_score", "layer_one_indexed"], ascending=[False, True]
        ).iloc[0]
        winners.append(
            {
                "coefficient": float(coefficient),
                "selected_layer_one_indexed": int(winner["layer_one_indexed"]),
                "mean_trait_score": float(winner["mean_trait_score"]),
                "mean_coherence_score": float(winner["mean_coherence_score"]),
            }
        )
    paper_layer = int(selection["paper_selected_layer_one_indexed"])

    output_root = _resolve(config["paths"]["results_root"]) / "layer_selection"
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "layer_steering_scores.csv", index=False)
    report = {
        "schema_version": 1,
        "status": "complete",
        "method": selection["selection_metric"],
        "candidate_layers_one_indexed": layers,
        "coefficients": coefficients,
        "winner_per_coefficient": winners,
        "paper_selected_layer_one_indexed": paper_layer,
        "coefficients_recovering_paper_layer": [
            item["coefficient"]
            for item in winners
            if item["selected_layer_one_indexed"] == paper_layer
        ],
        "all_coefficients_recover_paper_layer": all(
            item["selected_layer_one_indexed"] == paper_layer for item in winners
        ),
        "inputs": inputs,
    }
    write_json(output_root / "metadata.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

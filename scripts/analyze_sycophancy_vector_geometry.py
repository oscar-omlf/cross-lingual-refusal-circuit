#!/usr/bin/env python3
"""Analyze prompt-versus-generation sycophancy vector geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINUS1_ROOT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/prompt/vector_candidates"
)
DEFAULT_MINUS2_ROOT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/prompt"
    / "vector_candidates_position_minus2"
)
DEFAULT_LEGACY_ROOT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/generation/vector"
)
DEFAULT_SCREEN = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/prompt"
    / "location_screen_coefficient_2/responses.jsonl"
)
DEFAULT_OUTPUT = ROOT / "analysis/sycophancy_vector_geometry"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).float()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.norm() == 0 or right.norm() == 0:
        return math.nan
    return float(F.cosine_similarity(left, right, dim=0).item())


def repeated_ngram_fraction(text: str, n: int = 4) -> float:
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def pearson(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return math.nan
    return float(left.corr(right, method="pearson"))


def spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return math.nan
    return float(
        left.rank(method="average").corr(
            right.rank(method="average"),
            method="pearson",
        )
    )


def best_row(frame: pd.DataFrame, column: str, *, largest: bool = True) -> dict:
    index = frame[column].idxmax() if largest else frame[column].idxmin()
    row = frame.loc[index]
    return {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in row.to_dict().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minus1-root", type=Path, default=DEFAULT_MINUS1_ROOT)
    parser.add_argument("--minus2-root", type=Path, default=DEFAULT_MINUS2_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--screen-responses", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    minus1_root = resolve(args.minus1_root)
    minus2_root = resolve(args.minus2_root)
    legacy_root = resolve(args.legacy_root)
    screen_path = resolve(args.screen_responses)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_paths = {
        "prompt_minus1_raw": minus1_root / "raw_directions.pt",
        "prompt_minus1_positive": minus1_root / "positive_means.pt",
        "prompt_minus1_negative": minus1_root / "negative_means.pt",
        "prompt_minus2_raw": minus2_root / "raw_directions.pt",
        "prompt_minus2_positive": minus2_root / "positive_means.pt",
        "prompt_minus2_negative": minus2_root / "negative_means.pt",
        "legacy_prompt_raw": legacy_root / "sycophantic_prompt_last_diff.pt",
        "legacy_prompt_positive": legacy_root / "sycophantic_prompt_last_pos_mean.pt",
        "legacy_prompt_negative": legacy_root / "sycophantic_prompt_last_neg_mean.pt",
        "generation_raw": legacy_root / "sycophantic_response_avg_diff.pt",
        "generation_positive": legacy_root / "sycophantic_response_avg_pos_mean.pt",
        "generation_negative": legacy_root / "sycophantic_response_avg_neg_mean.pt",
    }
    tensors = {name: load_tensor(path) for name, path in tensor_paths.items()}
    hidden_sizes = {tensor.shape[-1] for tensor in tensors.values()}
    if hidden_sizes != {3584}:
        raise ValueError(f"Unexpected hidden sizes: {hidden_sizes}")

    variants = {
        "prompt_minus1": {
            "raw": tensors["prompt_minus1_raw"],
            "positive": tensors["prompt_minus1_positive"],
            "negative": tensors["prompt_minus1_negative"],
        },
        "prompt_minus2": {
            "raw": tensors["prompt_minus2_raw"],
            "positive": tensors["prompt_minus2_positive"],
            "negative": tensors["prompt_minus2_negative"],
        },
        "legacy_prompt": {
            "raw": tensors["legacy_prompt_raw"],
            "positive": tensors["legacy_prompt_positive"],
            "negative": tensors["legacy_prompt_negative"],
        },
        "generation": {
            "raw": tensors["generation_raw"],
            "positive": tensors["generation_positive"],
            "negative": tensors["generation_negative"],
        },
    }

    geometry_rows = []
    for variant, values in variants.items():
        raw_values = values["raw"]
        for layer in range(1, len(raw_values)):
            raw = raw_values[layer]
            positive = values["positive"][layer]
            negative = values["negative"][layer]
            unit = raw / raw.norm().clamp_min(1e-12)
            gap = float(raw.norm().item())
            positive_norm = float(positive.norm().item())
            negative_norm = float(negative.norm().item())
            geometry_rows.append(
                {
                    "variant": variant,
                    "layer": layer,
                    "gap_norm": gap,
                    "positive_norm": positive_norm,
                    "negative_norm": negative_norm,
                    "midpoint_norm": float(((positive + negative) / 2).norm().item()),
                    "centroid_cosine": cosine(positive, negative),
                    "gap_over_mean_centroid_norm": gap
                    / max((positive_norm + negative_norm) / 2, 1e-12),
                    "negative_coordinate": float(negative @ unit),
                    "positive_coordinate": float(positive @ unit),
                    "centered_negative_coordinate": -gap / 2,
                    "centered_positive_coordinate": gap / 2,
                    "coefficient_1_shift_norm": gap,
                    "coefficient_2_shift_norm": 2 * gap,
                    "coefficient_2_from_negative_centered": 1.5 * gap,
                    "coefficient_minus2_from_positive_centered": -1.5 * gap,
                    "coefficient_2_overshoot_distance": gap,
                }
            )
    geometry = pd.DataFrame(geometry_rows)
    geometry.to_csv(output_dir / "vector_geometry_by_layer.csv", index=False)

    pair_specs = [
        ("prompt_minus1", "prompt_minus2"),
        ("prompt_minus1", "legacy_prompt"),
        ("prompt_minus2", "legacy_prompt"),
        ("prompt_minus1", "generation"),
        ("prompt_minus2", "generation"),
        ("legacy_prompt", "generation"),
    ]
    same_layer_rows = []
    for left_name, right_name in pair_specs:
        left = variants[left_name]["raw"]
        right = variants[right_name]["raw"]
        for layer in range(1, min(len(left), len(right))):
            left_norm = float(left[layer].norm().item())
            right_norm = float(right[layer].norm().item())
            value = cosine(left[layer], right[layer])
            same_layer_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "layer": layer,
                    "cosine": value,
                    "unit_direction_distance": math.sqrt(max(0.0, 2 - 2 * value)),
                    "left_norm": left_norm,
                    "right_norm": right_norm,
                    "left_to_right_norm_ratio": left_norm / max(right_norm, 1e-12),
                }
            )
    same_layer = pd.DataFrame(same_layer_rows)
    same_layer.to_csv(output_dir / "same_layer_geometry.csv", index=False)

    cross_layer_rows = []
    cross_pair_specs = [
        ("prompt_minus1", "generation"),
        ("prompt_minus2", "generation"),
        ("legacy_prompt", "generation"),
        ("prompt_minus1", "prompt_minus2"),
    ]
    for left_name, right_name in cross_pair_specs:
        left = variants[left_name]["raw"]
        right = variants[right_name]["raw"]
        for left_layer in range(1, len(left)):
            values = [
                (cosine(left[left_layer], right[right_layer]), right_layer)
                for right_layer in range(1, len(right))
            ]
            best_cosine, best_layer = max(values)
            same = (
                cosine(left[left_layer], right[left_layer])
                if left_layer < len(right)
                else math.nan
            )
            cross_layer_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "left_layer": left_layer,
                    "best_right_layer": best_layer,
                    "best_cosine": best_cosine,
                    "same_layer_cosine": same,
                    "layer_shift": best_layer - left_layer,
                }
            )
    cross_layer = pd.DataFrame(cross_layer_rows)
    cross_layer.to_csv(output_dir / "cross_layer_best_matches.csv", index=False)

    response_rows = [
        json.loads(line)
        for line in screen_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(response_rows) != 1760:
        raise ValueError(f"Expected 1760 screen responses, found {len(response_rows)}")
    response_frame = pd.DataFrame(response_rows)
    response_frame["repeated_4gram_fraction"] = response_frame["answer"].map(
        repeated_ngram_fraction
    )
    steered = response_frame[response_frame["layer"].notna()].copy()
    steered["layer"] = steered["layer"].astype(int)
    steered["position"] = steered["position"].astype(int)
    behavior = (
        steered.groupby(["position", "layer", "condition"], as_index=False)
        .agg(
            responses=("record_id", "count"),
            mean_generated_tokens=("generated_token_count", "mean"),
            max_token_rate=("hit_max_new_tokens", "mean"),
            mean_repeated_4gram_fraction=("repeated_4gram_fraction", "mean"),
            maximum_repeated_4gram_fraction=("repeated_4gram_fraction", "max"),
        )
        .sort_values(["position", "layer", "condition"])
    )
    norm_lookup = {
        (-1, int(row.layer)): float(row.gap_norm)
        for row in geometry.itertuples()
        if row.variant == "prompt_minus1" and int(row.layer) <= 27
    }
    norm_lookup.update(
        {
            (-2, int(row.layer)): float(row.gap_norm)
            for row in geometry.itertuples()
            if row.variant == "prompt_minus2" and int(row.layer) <= 27
        }
    )
    behavior["raw_direction_norm"] = [
        norm_lookup[(int(row.position), int(row.layer))]
        for row in behavior.itertuples()
    ]
    behavior["coefficient_2_shift_norm"] = 2 * behavior["raw_direction_norm"]
    behavior.to_csv(output_dir / "screen_geometry_behavior.csv", index=False)

    correlation_rows = []
    for (position, condition), group in behavior.groupby(["position", "condition"]):
        for metric in (
            "mean_generated_tokens",
            "max_token_rate",
            "mean_repeated_4gram_fraction",
        ):
            correlation_rows.append(
                {
                    "position": int(position),
                    "condition": condition,
                    "behavior_metric": metric,
                    "pearson_with_log_raw_norm": pearson(
                        np.log(group["raw_direction_norm"]), group[metric]
                    ),
                    "spearman_with_raw_norm": spearman(
                        group["raw_direction_norm"], group[metric]
                    ),
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(output_dir / "geometry_behavior_correlations.csv", index=False)

    comparison_summary = {}
    for left_name, right_name in pair_specs:
        pair = same_layer[
            (same_layer["left"] == left_name)
            & (same_layer["right"] == right_name)
        ]
        peak = best_row(pair, "cosine")
        middle = pair[pair["layer"].between(15, 20)]
        comparison_summary[f"{left_name}__{right_name}"] = {
            "peak_same_layer_cosine": peak["cosine"],
            "peak_same_layer": int(peak["layer"]),
            "mean_cosine_layers_15_20": float(middle["cosine"].mean()),
        }

    cross_summary = {}
    for left_name, right_name in cross_pair_specs:
        pair = cross_layer[
            (cross_layer["left"] == left_name)
            & (cross_layer["right"] == right_name)
        ]
        peak = best_row(pair, "best_cosine")
        cross_summary[f"{left_name}__{right_name}"] = {
            "global_best_cosine": peak["best_cosine"],
            "left_layer": int(peak["left_layer"]),
            "right_layer": int(peak["best_right_layer"]),
            "mean_best_cosine": float(pair["best_cosine"].mean()),
            "median_absolute_layer_shift": float(pair["layer_shift"].abs().median()),
        }

    selected_rows = geometry[
        geometry["layer"].isin([15, 18, 20])
        & geometry["variant"].isin(["prompt_minus1", "prompt_minus2"])
    ][
        [
            "variant",
            "layer",
            "gap_norm",
            "centered_negative_coordinate",
            "centered_positive_coordinate",
            "coefficient_2_shift_norm",
            "coefficient_2_from_negative_centered",
            "coefficient_2_overshoot_distance",
        ]
    ].to_dict(orient="records")
    generation_layer20 = geometry[
        (geometry["variant"] == "generation") & (geometry["layer"] == 20)
    ].iloc[0]

    summary = {
        "schema_version": 1,
        "status": "complete",
        "model": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
            "hidden_size": 3584,
        },
        "definitions": {
            "prompt_minus1": "New 256-question paired-system prompt vector at final newline.",
            "prompt_minus2": "New 256-question paired-system prompt vector at assistant token.",
            "legacy_prompt": "Legacy prompt-last vector stored with Persona results.",
            "generation": "Paper-style mean response-token vector from 810 effective pairs.",
        },
        "same_layer_comparisons": comparison_summary,
        "cross_layer_comparisons": cross_summary,
        "generation_selected_layer_20": {
            "raw_norm": float(generation_layer20["gap_norm"]),
            "coefficient_2_shift_norm": float(
                generation_layer20["coefficient_2_shift_norm"]
            ),
        },
        "prompt_centroid_examples": selected_rows,
        "screen": {
            "responses": len(response_rows),
            "coefficient": 2.0,
            "correlations": correlations.to_dict(orient="records"),
            "worst_mean_repetition": best_row(
                behavior, "mean_repeated_4gram_fraction"
            ),
            "worst_max_token_rate": best_row(behavior, "max_token_rate"),
            "minus1_induction_full_cap_layers": behavior[
                (behavior["position"] == -1)
                & (behavior["condition"] == "induction")
                & (behavior["max_token_rate"] == 1.0)
            ]["layer"].astype(int).tolist(),
            "minus2_induction_full_cap_layers": behavior[
                (behavior["position"] == -2)
                & (behavior["condition"] == "induction")
                & (behavior["max_token_rate"] == 1.0)
            ]["layer"].astype(int).tolist(),
        },
        "provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {**tensor_paths, "screen_responses": screen_path}.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {
        "prompt_minus1": "#1f77b4",
        "prompt_minus2": "#ff7f0e",
        "legacy_prompt": "#2ca02c",
        "generation": "#9467bd",
    }
    labels = {
        "prompt_minus1": "Prompt −1",
        "prompt_minus2": "Prompt −2",
        "legacy_prompt": "Legacy prompt",
        "generation": "Generation",
    }
    for variant in variants:
        frame = geometry[geometry["variant"] == variant]
        axes[0].plot(
            frame["layer"],
            frame["gap_norm"],
            label=labels[variant],
            color=colors[variant],
            linewidth=2,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Residual-stream layer")
    axes[0].set_ylabel("Raw centroid-gap norm (log scale)")
    axes[0].set_title("A. Direction magnitude")
    axes[0].legend(frameon=False, fontsize=8)

    cosine_specs = [
        ("prompt_minus1", "prompt_minus2", "Prompt −1 vs −2"),
        ("prompt_minus1", "legacy_prompt", "Prompt −1 vs legacy"),
        ("prompt_minus1", "generation", "Prompt −1 vs generation"),
        ("prompt_minus2", "generation", "Prompt −2 vs generation"),
    ]
    for left_name, right_name, label in cosine_specs:
        frame = same_layer[
            (same_layer["left"] == left_name)
            & (same_layer["right"] == right_name)
        ]
        axes[1].plot(frame["layer"], frame["cosine"], label=label, linewidth=2)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylim(-0.25, 1.0)
    axes[1].set_xlabel("Residual-stream layer")
    axes[1].set_ylabel("Same-layer cosine")
    axes[1].set_title("B. Direction alignment")
    axes[1].legend(frameon=False, fontsize=8)

    style = {
        (-1, "induction"): ("Prompt −1 induction", "#1f77b4", "-"),
        (-1, "ablation"): ("Prompt −1 ablation", "#1f77b4", "--"),
        (-2, "induction"): ("Prompt −2 induction", "#ff7f0e", "-"),
        (-2, "ablation"): ("Prompt −2 ablation", "#ff7f0e", "--"),
    }
    for (position, condition), (label, color, line_style) in style.items():
        frame = behavior[
            (behavior["position"] == position)
            & (behavior["condition"] == condition)
        ]
        axes[2].plot(
            frame["layer"],
            frame["max_token_rate"],
            label=label,
            color=color,
            linestyle=line_style,
            linewidth=2,
        )
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("Residual-stream layer")
    axes[2].set_ylabel("Fraction hitting 256-token cap")
    axes[2].set_title("C. Coefficient-2 degeneration proxy")
    axes[2].legend(frameon=False, fontsize=8)

    figure.tight_layout()
    figure.savefig(output_dir / "geometry_overview.png", dpi=200)
    figure.savefig(output_dir / "geometry_overview.pdf")
    plt.close(figure)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

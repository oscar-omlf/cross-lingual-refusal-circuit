#!/usr/bin/env python3
"""Compare independently selected and common-site PolyRefuse directions.

The ``There Is More to Refusal`` comparison logic has an important lesson for
this study: vectors from different sites are not a clean geometric comparison.
We therefore report both (a) the descriptive cosine of independently selected
directions and (b) the primary common-site cosine, extracting every language's
DIM vector at English's selected layer and assistant-template position.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--anchor-language", default="en")
    parser.add_argument("--direction-subdir", default="direction")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Defaults to results/qwen2.5-7b-instruct/polyrefuse.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--same-cosine-threshold",
        type=float,
        default=0.9,
        help="Transparent heuristic for the summary only; not a statistical test.",
    )
    return parser.parse_args()


def _unit(vector):
    return vector.float() / vector.float().norm().clamp_min(1e-8)


def _candidate_at_site(candidate_file: Path, *, layer: int, position: int):
    import torch

    payload = torch.load(candidate_file, map_location="cpu", weights_only=True)
    layers = [int(value) for value in payload["layers"]]
    positions = [int(value) for value in payload["positions"]]
    try:
        layer_index = layers.index(layer)
        position_index = positions.index(position)
    except ValueError as exc:
        raise ValueError(
            f"Common anchor (layer={layer}, position={position}) was not in {candidate_file}."
        ) from exc
    index = position_index * len(layers) + layer_index
    return payload["vectors"][index]


def _pairs(vectors: dict[str, object], comparison: str):
    rows = []
    for left, right in itertools.combinations(sorted(vectors), 2):
        cosine = float((_unit(vectors[left]) @ _unit(vectors[right])).item())
        rows.append({"comparison": comparison, "left_language": left, "right_language": right, "cosine": cosine})
    return rows


def _matrix(languages: list[str], rows: list[dict]):
    import pandas as pd

    matrix = pd.DataFrame(1.0, index=languages, columns=languages)
    for row in rows:
        matrix.loc[row["left_language"], row["right_language"]] = row["cosine"]
        matrix.loc[row["right_language"], row["left_language"]] = row["cosine"]
    matrix.index.name = "language"
    return matrix


def main():
    args = parse_args()
    import pandas as pd
    import torch

    from refusal_circuit.paths import experiment_results_dir, read_json, write_json
    from refusal_circuit.polyrefuse import POLYREFUSE_LANGUAGES

    languages = list(dict.fromkeys(args.languages))
    if not languages:
        raise ValueError("At least one language is required.")
    unknown = sorted(set(languages) - set(POLYREFUSE_LANGUAGES))
    if unknown:
        raise ValueError(f"Unsupported languages: {unknown}.")
    if args.anchor_language not in languages:
        raise ValueError("--anchor-language must be included in --languages.")
    default_root = experiment_results_dir(args.model, "polyrefuse", "en").parent
    results_root = args.results_root or default_root
    output_dir = args.output_dir or results_root / "comparisons"

    metadata_by_language = {}
    selected_vectors = {}
    for language in languages:
        direction_dir = results_root / language / args.direction_subdir
        metadata_path = direction_dir / "refusal_direction_metadata.json"
        vector_path = direction_dir / "refusal_direction.pt"
        if not metadata_path.exists() or not vector_path.exists():
            raise FileNotFoundError(
                f"Missing selected direction for {language} in {direction_dir}; extract every language first."
            )
        metadata = read_json(metadata_path)
        if metadata.get("model") != args.model or metadata.get("language") != language:
            raise ValueError(f"Incompatible direction metadata in {metadata_path}.")
        metadata_by_language[language] = metadata
        selected_vectors[language] = torch.load(vector_path, map_location="cpu", weights_only=True)

    anchor_metadata = metadata_by_language[args.anchor_language]
    anchor_layer = int(anchor_metadata["layer"])
    anchor_position = int(anchor_metadata["position"])
    anchor_vectors = {
        language: _candidate_at_site(
            results_root / language / args.direction_subdir / "candidate_directions.pt",
            layer=anchor_layer,
            position=anchor_position,
        )
        for language in languages
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep a normalized anchor vector next to each language's selected vector;
    # circuit discovery can then share a graph terminal without ambiguity.
    for language, vector in anchor_vectors.items():
        direction_dir = results_root / language / args.direction_subdir
        normalized = _unit(vector).cpu()
        torch.save(normalized, direction_dir / "anchor_refusal_direction.pt")
        write_json(
            direction_dir / "anchor_direction_metadata.json",
            {
                "model": args.model,
                "dataset": "PolyRefuse",
                "language": language,
                "anchor_language": args.anchor_language,
                "layer": anchor_layer,
                "position": anchor_position,
                "target_hook": f"blocks.{anchor_layer}.hook_resid_pre",
                "raw_direction_norm": float(vector.norm().item()),
                "normalised": True,
                "source": "language-local DIM at common English-selected site",
                "refusal_token_ids": metadata_by_language[language].get("refusal_token_ids", []),
            },
        )

    selected_rows = _pairs(selected_vectors, "individually_selected_site_descriptive")
    anchor_rows = _pairs(anchor_vectors, "common_english_anchor_site_primary")
    all_rows = selected_rows + anchor_rows
    pd.DataFrame(all_rows).to_csv(output_dir / "direction_pairwise_cosines.csv", index=False)
    _matrix(languages, selected_rows).to_csv(output_dir / "direction_cosine_selected_sites.csv")
    _matrix(languages, anchor_rows).to_csv(output_dir / "direction_cosine_common_anchor.csv")
    anchor_cosines = [row["cosine"] for row in anchor_rows]
    write_json(
        output_dir / "direction_comparison_metadata.json",
        {
            "model": args.model,
            "dataset": "PolyRefuse",
            "languages": languages,
            "anchor_language": args.anchor_language,
            "anchor_layer": anchor_layer,
            "anchor_position": anchor_position,
            "direction_subdir": args.direction_subdir,
            "primary_comparison": "cosine between language-local DIM vectors at a common site",
            "descriptive_comparison": "cosine between each language's individually selected intervention site",
            "same_direction_heuristic_threshold": args.same_cosine_threshold,
            "anchor_cosine_min": min(anchor_cosines) if anchor_cosines else 1.0,
            "anchor_cosine_mean": sum(anchor_cosines) / max(len(anchor_cosines), 1),
            "all_anchor_pairs_meet_heuristic": all(
                cosine >= args.same_cosine_threshold for cosine in anchor_cosines
            ),
            "interpretation": (
                "The threshold is a transparent operational heuristic, not proof of identity. "
                "Direction transfer and circuit transfer are evaluated separately."
            ),
        },
    )
    print(
        f"Compared {len(languages)} languages at common layer={anchor_layer}, position={anchor_position}; "
        f"wrote cosine tables under {output_dir}"
    )


if __name__ == "__main__":
    main()

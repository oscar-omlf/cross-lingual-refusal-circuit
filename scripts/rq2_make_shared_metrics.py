#!/usr/bin/env python3
"""Create standardized multilingual refusal metrics for RQ2 circuit discovery.

The first multilingual circuit runs used each language's DIM vector at the
common English-selected site.  That answers "does each language have a local
edge circuit for its own direction at the same residual site?"  For invariance
we also need standardized metrics: the exact English vector applied to every
language, and optionally the normalized mean of several language-local vectors.

This script only copies/normalizes vectors and writes metadata. It does not
load a model and does not touch prompt text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--languages", nargs="+", default=("en", "de", "ru"))
    parser.add_argument("--source-language", default="en")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("english_vector", "common_mean"),
        default=("english_vector", "common_mean"),
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument(
        "--direction-subdir",
        default="direction",
        help="Per-language directory containing anchor_refusal_direction.pt.",
    )
    return parser.parse_args()


def _unit(vector):
    return vector.float() / vector.float().norm().clamp_min(1e-8)


def _read_anchor(results_root: Path, language: str, direction_subdir: str):
    import torch

    from refusal_circuit.paths import read_json

    direction_dir = results_root / language / direction_subdir
    vector_path = direction_dir / "anchor_refusal_direction.pt"
    metadata_path = direction_dir / "anchor_direction_metadata.json"
    if not vector_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing common-site anchor vector/metadata for {language} under {direction_dir}."
        )
    return (
        _unit(torch.load(vector_path, map_location="cpu", weights_only=True)),
        read_json(metadata_path),
        vector_path,
        metadata_path,
    )


def main():
    args = parse_args()
    import torch

    from refusal_circuit.paths import experiment_results_dir, write_json

    languages = list(dict.fromkeys(args.languages))
    if args.source_language not in languages:
        raise ValueError("--source-language must be included in --languages.")
    results_root = args.results_root or experiment_results_dir(args.model, "polyrefuse", "en").parent

    anchors = {
        language: _read_anchor(results_root, language, args.direction_subdir)
        for language in languages
    }
    source_vector, source_meta, source_vector_path, _ = anchors[args.source_language]
    anchor_site = (int(source_meta["layer"]), int(source_meta["position"]))
    for language, (_, metadata, _, _) in anchors.items():
        site = (int(metadata["layer"]), int(metadata["position"]))
        if site != anchor_site:
            raise ValueError(
                f"{language} anchor site {site} differs from source site {anchor_site}; "
                "run rq2_compare_multilingual_directions.py first."
            )

    metric_vectors = {}
    if "english_vector" in args.metrics:
        metric_vectors["english_vector"] = {
            "vector": source_vector,
            "source": f"exact normalized {args.source_language} anchor DIM vector",
            "source_vector_path": str(source_vector_path),
        }
    if "common_mean" in args.metrics:
        stacked = torch.stack([anchors[language][0] for language in languages])
        mean_vector = _unit(stacked.mean(dim=0))
        metric_vectors["common_mean"] = {
            "vector": mean_vector,
            "source": f"normalized mean of language-local anchor DIM vectors: {','.join(languages)}",
            "source_vector_path": None,
        }

    for metric_name, spec in metric_vectors.items():
        metric_vector = _unit(spec["vector"]).cpu()
        source_cosines = {
            language: float((metric_vector @ anchors[language][0]).item())
            for language in languages
        }
        for language in languages:
            _, lang_meta, _, lang_meta_path = anchors[language]
            output_dir = results_root / language / "shared_metrics" / metric_name
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(metric_vector, output_dir / "refusal_direction.pt")
            write_json(
                output_dir / "metadata.json",
                {
                    "model": args.model,
                    "dataset": "PolyRefuse",
                    "language": language,
                    "metric_name": metric_name,
                    "method": "standardized_common_site_refusal_projection",
                    "source": spec["source"],
                    "source_language": args.source_language,
                    "source_languages": languages,
                    "source_vector_path": spec["source_vector_path"],
                    "source_anchor_metadata": str(lang_meta_path),
                    "layer": anchor_site[0],
                    "position": anchor_site[1],
                    "target_hook": f"blocks.{anchor_site[0]}.hook_resid_pre",
                    "normalised": True,
                    "refusal_token_ids": lang_meta.get("refusal_token_ids", []),
                    "source_anchor_cosines": source_cosines,
                    "notes": (
                        "The same residual direction is used for every target language; "
                        "language-specific refusal_token_ids are retained only for logit-margin diagnostics."
                    ),
                },
            )
            print(f"Wrote {language} shared metric {metric_name} under {output_dir}")


if __name__ == "__main__":
    main()

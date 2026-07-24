#!/usr/bin/env python3
"""Validate the persona vector as a pre-response behavioral monitoring signal.

Chen et al. extract their direction from response-average activations but test
prompt-induced monitoring at the final prompt token, before any answer exists.
This is the causal endpoint that an upstream behavioral circuit must explain.
The script uses the already generated and judged extraction data, so it makes
no API calls and performs no new generation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _pearson(x, y) -> float:
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y) -> float:
    import pandas as pd

    return _pearson(pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average"))


def _auc(positive, negative) -> float:
    """Probability that a random positive projection exceeds a negative one."""
    import numpy as np
    import pandas as pd

    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    ranks = pd.Series(np.concatenate([positive, negative])).rank(method="average").to_numpy()
    rank_sum = ranks[: len(positive)].sum()
    u = rank_sum - len(positive) * (len(positive) + 1) / 2
    return float(u / (len(positive) * len(negative)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.metrics import ProjectionCapture
    from refusal_circuit.model import load_model, tokenize_prompts
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    intermediate = _resolve(config["paths"]["intermediate_root"])
    result_root = _resolve(config["paths"]["results_root"])
    output = result_root / "monitoring/prompt_last_extraction"
    table_path = output / "prompt_groups.csv"
    summary_path = output / "summary.json"
    if (table_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite only for an intentional rerun")

    source_paths = {
        "pos": intermediate / "judged/pos_scored.jsonl",
        "neg": intermediate / "judged/neg_scored.jsonl",
    }
    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for polarity, path in source_paths.items():
        for row in read_jsonl(path):
            groups[(polarity, int(row["question_index"]), int(row["instruction_index"]))].append(row)

    expected_rollouts = int(config["extraction"]["rollouts_per_question_instruction"])
    specifications = []
    for key in sorted(groups):
        rows = groups[key]
        prompts = {row["prompt"] for row in rows}
        if len(rows) != expected_rollouts or len(prompts) != 1:
            raise ValueError(f"Unexpected prompt group {key}: rows={len(rows)}, prompts={len(prompts)}")
        trait_scores = [row["trait_score"] for row in rows]
        coherence_scores = [row["coherence_score"] for row in rows]
        if any(score is None for score in trait_scores + coherence_scores):
            raise ValueError(f"Judge failure in prompt group {key}")
        polarity, question_index, instruction_index = key
        specifications.append(
            {
                "polarity": polarity,
                "question_index": question_index,
                "instruction_index": instruction_index,
                "prompt": next(iter(prompts)),
                "mean_trait_score": float(np.mean(trait_scores)),
                "mean_coherence_score": float(np.mean(coherence_scores)),
                "rollouts": len(rows),
            }
        )

    model_cfg = config["model"]
    model = load_model(
        model_cfg["name"], device=args.device, dtype=args.dtype, revision=model_cfg.get("revision")
    )
    direction_path = result_root / "vector/sycophantic_direction.pt"
    direction = torch.load(
        direction_path, map_location=next(model.parameters()).device, weights_only=True
    )
    target_hook = config["vector"]["transformer_lens_target_hook"]
    for start in tqdm(range(0, len(specifications), args.batch_size), desc="prompt monitoring"):
        batch = specifications[start : start + args.batch_size]
        tokens, attention_mask = tokenize_prompts(
            model, [row["prompt"] for row in batch], max_length=args.max_length
        )
        capture = ProjectionCapture(direction=direction, position=-1)
        with torch.inference_mode(), model.hooks(fwd_hooks=[(target_hook, capture.hook)]):
            model(tokens, attention_mask=attention_mask)
        values = capture.values().detach().float().cpu().tolist()
        if len(values) != len(batch):
            raise RuntimeError("Prompt projection capture returned an unexpected batch shape")
        for row, value in zip(batch, values, strict=True):
            row["prompt_last_projection"] = float(value)

    by_polarity = {
        polarity: [row for row in specifications if row["polarity"] == polarity]
        for polarity in ("pos", "neg")
    }
    all_projection = [row["prompt_last_projection"] for row in specifications]
    all_behavior = [row["mean_trait_score"] for row in specifications]
    pos_projection = [row["prompt_last_projection"] for row in by_polarity["pos"]]
    neg_projection = [row["prompt_last_projection"] for row in by_polarity["neg"]]
    pooled_sd = math.sqrt(
        (np.var(pos_projection, ddof=1) + np.var(neg_projection, ddof=1)) / 2
    )
    correlations = {
        "all_pearson": _pearson(all_projection, all_behavior),
        "all_spearman": _spearman(all_projection, all_behavior),
        "positive_only_pearson": _pearson(
            pos_projection, [row["mean_trait_score"] for row in by_polarity["pos"]]
        ),
        "negative_only_pearson": _pearson(
            neg_projection, [row["mean_trait_score"] for row in by_polarity["neg"]]
        ),
    }
    gate_threshold = float(config["circuit_extension"]["monitoring_gate_min_pearson"])
    summary = {
        "schema_version": 1,
        "status": "complete",
        "claim_scope": "pre-response monitoring validation; no circuit claim",
        "model": model_cfg,
        "trait": config["trait"]["name"],
        "direction": str(direction_path.relative_to(ROOT)),
        "direction_sha256": sha256_file(direction_path),
        "target_hook": target_hook,
        "target_position": -1,
        "target_timing": "last prompt token before response generation",
        "prompt_groups": len(specifications),
        "rollouts_per_prompt_group": expected_rollouts,
        "source_files": {
            polarity: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for polarity, path in source_paths.items()
        },
        "projection": {
            "positive_mean": float(np.mean(pos_projection)),
            "negative_mean": float(np.mean(neg_projection)),
            "positive_minus_negative": float(np.mean(pos_projection) - np.mean(neg_projection)),
            "pooled_cohens_d": float(
                (np.mean(pos_projection) - np.mean(neg_projection)) / pooled_sd
            ),
            "positive_vs_negative_auc": _auc(pos_projection, neg_projection),
        },
        "behavior": {
            "positive_mean": float(
                np.mean([row["mean_trait_score"] for row in by_polarity["pos"]])
            ),
            "negative_mean": float(
                np.mean([row["mean_trait_score"] for row in by_polarity["neg"]])
            ),
        },
        "correlations": correlations,
        "gate": {
            "criterion": f"all_pearson >= {gate_threshold}",
            "passed": bool(correlations["all_pearson"] >= gate_threshold),
        },
        "notes": [
            "The canonical direction is still the paper's response-average vector.",
            "The monitored activation is prompt-last, matching the paper's pre-response analysis.",
            "Across-polarity correlation is the primary gate; within-polarity values are diagnostics.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(specifications).drop(columns=["prompt"]).to_csv(table_path, index=False)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Discover an exploratory circuit for a post-response sycophancy projection.

This target averages activations over an already generated, teacher-forced
response.  It is useful for decomposing the canonical extraction statistic,
but response tokens can reveal the behavior that has already occurred.  This
artifact must therefore not be described as a pre-generation behavioral
circuit.  The canonical causal extension first gates on the last-prompt-token
monitoring signal via ``persona.monitor-prompt``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.eapig import AttributionConfig, attribute_response_average
    from refusal_circuit.graph import make_projection_graph, write_edge_scores_parquet, write_top_edges_summary
    from refusal_circuit.model import load_model
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import read_jsonl, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    vector_root = _resolve(config["paths"]["results_root"]) / "vector"
    intermediate = _resolve(config["paths"]["intermediate_root"])
    output = _resolve(config["paths"]["results_root"]) / "circuit_discovery"
    score_path = output / "eapig_edge_scores.parquet"
    if score_path.exists() and not args.overwrite:
        raise FileExistsError(f"{score_path} already exists; refusing to repeat attribution")

    effective = set(pd.read_csv(vector_root / "effective_pairs.csv")["pair_id"].astype(str))
    positive_path = intermediate / "judged/pos_scored.jsonl"
    rows = [row for row in read_jsonl(positive_path) if row["pair_id"] in effective]
    # Round-robin over questions so a small attribution set cannot silently
    # collapse onto the first question (as the initial sorted-32 pilot did).
    # Shuffle within each question to vary instruction and rollout while
    # retaining a fully deterministic, auditable selection.
    by_question: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_question[int(row["question_index"])].append(row)
    rng = random.Random(args.seed)
    for question_rows in by_question.values():
        question_rows.sort(key=lambda row: row["pair_id"])
        rng.shuffle(question_rows)
    balanced = []
    depth = 0
    while True:
        added = False
        for question_index in sorted(by_question):
            if depth < len(by_question[question_index]):
                balanced.append(by_question[question_index][depth])
                added = True
        if not added:
            break
        depth += 1
    selected = balanced[args.offset : args.offset + args.num_prompts]
    if len(selected) != args.num_prompts:
        raise ValueError(f"Requested {args.num_prompts} transcripts, found {len(selected)}")

    model = load_model(
        model_cfg["name"], device=args.device, dtype=args.dtype, revision=model_cfg.get("revision")
    )
    direction = torch.load(
        vector_root / "sycophantic_direction.pt",
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    transcripts = []
    boundaries = []
    for row in selected:
        text = row["prompt"] + row["answer"]
        start = len(model.tokenizer.encode(row["prompt"], add_special_tokens=False))
        full_length = len(model.tokenizer.encode(text, add_special_tokens=False))
        if full_length > args.max_length:
            full_length = args.max_length
        if start >= full_length:
            raise ValueError(f"Truncation removed response for {row['pair_id']}")
        transcripts.append((text, start))
        boundaries.append({"pair_id": row["pair_id"], "response_start": start, "used_length": full_length})

    target_layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    graph = make_projection_graph(model, target_layer)
    attribute_response_average(
        model,
        graph,
        transcripts,
        direction=direction,
        target_hook=target_hook,
        config=AttributionConfig(
            batch_size=args.batch_size, max_length=args.max_length, ig_steps=args.ig_steps
        ),
        integrated_gradients=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_edge_scores_parquet(graph, score_path)
    write_top_edges_summary(graph, output / "top_edges_summary_eapig.md", method="eapig")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "trait": config["trait"]["name"],
        "model": model_cfg["name"],
        "model_revision": model_cfg.get("revision"),
        "method": "intervention-path EAP-IG",
        "metric": "mean response-token residual projection onto unit sycophancy direction",
        "claim_scope": "post-generation representation diagnostic only",
        "behavior_leakage_warning": (
            "The target is computed on teacher-forced response tokens, so the circuit may read "
            "sycophantic text already present in the prefix rather than cause its generation."
        ),
        "target_hook": target_hook,
        "target_layer": target_layer,
        "corrupt_baseline": "same teacher-forced transcript with global sycophancy-direction ablation",
        "source": str(positive_path.relative_to(ROOT)),
        "source_sha256": sha256_file(positive_path),
        "selection": "question-balanced round-robin over shuffled effective positive pairs",
        "seed": args.seed,
        "offset": args.offset,
        "n_prompts": len(selected),
        "record_ids": [row["record_id"] for row in selected],
        "boundaries": boundaries,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "ig_steps": args.ig_steps,
        "direction_file": str((vector_root / "sycophantic_direction.pt").relative_to(ROOT)),
        "direction_sha256": sha256_file(vector_root / "sycophantic_direction.pt"),
        "score_file": score_path.name,
    }
    write_json(output / "eapig_run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

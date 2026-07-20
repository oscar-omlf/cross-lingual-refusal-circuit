#!/usr/bin/env python3
"""Efficient projection/first-token sweep for one frozen circuit family.

Clean and global-direction-ablated baselines are computed once.  Every k then
requires only circuit-only and circuit-ablated mixed forwards.  No generation
or attribution is performed, and no prompt text is written to results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


DEFAULT_CONFIG = ROOT / "configs" / "phase2_qwen_refusal.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _last_token_kl_per_row(clean_logits, changed_logits):
    clean_log_probs = clean_logits.float().log_softmax(dim=-1)
    changed_log_probs = changed_logits.float().log_softmax(dim=-1)
    return (clean_log_probs.exp() * (clean_log_probs - changed_log_probs)).sum(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--language", required=True, choices=("en", "de", "zh", "th"))
    parser.add_argument("--split", default="development")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional JSONL override, e.g. the historical mixed-label test split.",
    )
    parser.add_argument("--row-label", choices=("harmful", "harmless"), default="harmful")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--circuit-label", required=True)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import _mixed_forward, _normal_forward
    from refusal_circuit.graph import load_circuit_edge_names, make_projection_graph, set_selected_edges
    from refusal_circuit.interventions import all_layer_ablation_hooks
    from refusal_circuit.metrics import refusal_log_odds
    from refusal_circuit.model import format_instructions, load_model, tokenize_prompts

    cfg = _read(args.config)
    if args.language not in cfg["dataset"]["languages"]:
        raise ValueError(f"Language {args.language} is not frozen in {args.config}")
    k_values = args.k_values or cfg["circuit_sweep"]["requested_k"]
    data_path = args.data_path or (
        _resolve(cfg["dataset"]["phase2_data_root"])
        / args.language
        / f"{args.split}.jsonl"
    )
    data_path = _resolve(data_path)
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row.get("label") == args.row_label]
    if args.num_prompts is not None:
        rows = rows[: args.num_prompts]
    if not rows:
        raise ValueError(f"No rows selected from {data_path}")
    circuits_dir = _resolve(args.circuits_dir)
    for k in k_values:
        if not (circuits_dir / f"eapig_top_{k}.json").exists():
            raise FileNotFoundError(circuits_dir / f"eapig_top_{k}.json")

    results_root = _resolve(cfg["artifacts"]["results_root"])
    metric_root = results_root / args.language / "shared_metrics" / cfg["metric"]["name"]
    metadata_path = metric_root / "metadata.json"
    direction_path = metric_root / "refusal_direction.pt"
    metadata = _read(metadata_path)
    expected_site = (cfg["metric"]["target_layer"], cfg["metric"]["target_position"])
    if (int(metadata["layer"]), int(metadata["position"])) != expected_site:
        raise ValueError("Metric metadata does not match the frozen Phase 2 target")

    model = load_model(cfg["model"]["name"], device=args.device, dtype=args.dtype)
    direction = torch.load(
        direction_path, map_location=next(model.parameters()).device, weights_only=True
    )
    refusal_ids = [int(value) for value in metadata["refusal_token_ids"]]
    prompts = format_instructions(model.tokenizer, [row["instruction"] for row in rows])
    ids = [str(row.get("parallel_id", row["id"])) for row in rows]

    baseline: dict[str, dict[str, list]] = {
        condition: {"projection": [], "logits": []}
        for condition in ("normal", "direction_ablated")
    }
    for _start, prompt_batch in _batches(prompts, args.batch_size):
        tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=args.max_length)
        normal_projection, normal_logits = _normal_forward(
            model,
            tokens,
            attention_mask,
            target_hook=metadata["target_hook"],
            target_position=int(metadata["position"]),
            direction=direction,
        )
        corrupt_projection, corrupt_logits = _normal_forward(
            model,
            tokens,
            attention_mask,
            target_hook=metadata["target_hook"],
            target_position=int(metadata["position"]),
            direction=direction,
            fwd_hooks=all_layer_ablation_hooks(model, direction),
        )
        for condition, projection, logits in (
            ("normal", normal_projection, normal_logits),
            ("direction_ablated", corrupt_projection, corrupt_logits),
        ):
            baseline[condition]["projection"].append(projection.float().cpu())
            baseline[condition]["logits"].append(logits[:, -1].to(torch.bfloat16).cpu())
    for condition in baseline:
        for key in baseline[condition]:
            baseline[condition][key] = torch.cat(baseline[condition][key])

    instance_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    def append_condition(k: int, retained: int, condition: str, projection, logits) -> None:
        clean_logits = baseline["normal"]["logits"]
        log_odds = refusal_log_odds(logits.unsqueeze(1), refusal_ids).float()
        kl = _last_token_kl_per_row(clean_logits, logits).float()
        changed = (clean_logits.argmax(dim=-1) != logits.argmax(dim=-1)).float()
        for index, parallel_id in enumerate(ids):
            instance_rows.append(
                {
                    "parallel_id": parallel_id,
                    "language": args.language,
                    "split": args.split,
                    "row_label": args.row_label,
                    "circuit_label": args.circuit_label,
                    "requested_k": k,
                    "retained_edges": retained,
                    "edge_fraction": retained / int(cfg["metric"]["eligible_edges"]),
                    "condition": condition,
                    "projection": float(projection[index]),
                    "first_token_refusal_log_odds": float(log_odds[index]),
                    "first_token_refusal_indicator": bool(log_odds[index] > 0),
                    "kl_normal_to_condition": float(kl[index]),
                    "top1_changed_from_normal": bool(changed[index]),
                }
            )

    clean_projection = baseline["normal"]["projection"]
    corrupt_projection = baseline["direction_ablated"]["projection"]
    denominator = float(clean_projection.mean() - corrupt_projection.mean())
    for k in k_values:
        circuit_path = circuits_dir / f"eapig_top_{k}.json"
        edge_names = load_circuit_edge_names(circuit_path)
        graph = make_projection_graph(model, int(metadata["layer"]))
        set_selected_edges(graph, edge_names)
        mixed = {
            condition: {"projection": [], "logits": []}
            for condition in ("circuit_only", "circuit_ablated")
        }
        for _start, prompt_batch in _batches(prompts, args.batch_size):
            tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=args.max_length)
            for condition, patch_selected in (("circuit_only", False), ("circuit_ablated", True)):
                projection, logits = _mixed_forward(
                    model,
                    graph,
                    tokens,
                    attention_mask,
                    target_hook=metadata["target_hook"],
                    target_position=int(metadata["position"]),
                    direction=direction,
                    patch_selected=patch_selected,
                )
                mixed[condition]["projection"].append(projection.float().cpu())
                mixed[condition]["logits"].append(logits[:, -1].to(torch.bfloat16).cpu())
        for condition in mixed:
            for key in mixed[condition]:
                mixed[condition][key] = torch.cat(mixed[condition][key])

        retained = len(edge_names)
        for condition in ("normal", "direction_ablated"):
            append_condition(
                k,
                retained,
                condition,
                baseline[condition]["projection"],
                baseline[condition]["logits"],
            )
        for condition in ("circuit_only", "circuit_ablated"):
            append_condition(
                k,
                retained,
                condition,
                mixed[condition]["projection"],
                mixed[condition]["logits"],
            )
        faithfulness = (
            float((mixed["circuit_only"]["projection"].mean() - corrupt_projection.mean()) / denominator)
            if abs(denominator) > 1e-8
            else float("nan")
        )
        necessity = (
            float((clean_projection.mean() - mixed["circuit_ablated"]["projection"].mean()) / denominator)
            if abs(denominator) > 1e-8
            else float("nan")
        )
        aggregate_rows.append(
            {
                "model": cfg["model"]["name"],
                "language": args.language,
                "split": args.split,
                "row_label": args.row_label,
                "circuit_label": args.circuit_label,
                "requested_k": k,
                "retained_edges": retained,
                "edge_fraction": retained / int(cfg["metric"]["eligible_edges"]),
                "n_prompts": len(rows),
                "faithfulness": faithfulness,
                "necessity": necessity,
                "m_normal": float(clean_projection.mean()),
                "m_direction_ablated": float(corrupt_projection.mean()),
                "m_circuit_only": float(mixed["circuit_only"]["projection"].mean()),
                "m_circuit_ablated": float(mixed["circuit_ablated"]["projection"].mean()),
            }
        )

    output_dir = args.output_dir or (
        _resolve(cfg["artifacts"]["phase2_results_root"])
        / "metrics"
        / args.split
        / args.circuit_label
        / args.language
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "aggregate.csv", index=False)
    pd.DataFrame(instance_rows).to_csv(output_dir / "instance_metrics_no_text.csv", index=False)
    run_metadata = {
        "config": str(args.config),
        "model": cfg["model"]["name"],
        "language": args.language,
        "split": args.split,
        "row_label": args.row_label,
        "data_path": str(data_path),
        "circuit_label": args.circuit_label,
        "circuits_dir": str(circuits_dir),
        "requested_k": k_values,
        "n_prompts": len(rows),
        "baseline_computed_once": True,
        "generated_text_saved": False,
        "prompt_text_saved": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved Phase 2 metric sweep under {output_dir}")


if __name__ == "__main__":
    main()

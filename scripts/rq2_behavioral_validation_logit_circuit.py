#!/usr/bin/env python3
"""Behavioral validation for logit-target EAP-IG circuits.

The circuit graph terminates at the ordinary logits node.  We still use the
same global refusal-direction ablation as the corrupt baseline, and we report
aggregate metrics only; prompts and generated completions are never written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401
from rq2_behavioral_validation import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    DEFAULT_RESULTS_ROOT,
    _condition_summary,
    _generation_summary,
    _selected_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", required=True, choices=("en", "de", "zh", "th"))
    parser.add_argument("--metric-name", default="english_vector")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--prompt-offset", type=int, default=32)
    parser.add_argument("--generation-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--circuit-dir",
        type=Path,
        default=Path(
            "results/qwen2.5-7b-instruct/polyrefuse/en/"
            "circuit_discovery_logit_english_vector_pilot/circuits_greedy_no_parent_prune"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.results_root
        / args.language
        / "behavioral_validation_logit_english_vector_pilot"
        / f"k{args.k}_offset{args.prompt_offset}_n{args.num_prompts}_gen{args.generation_prompts}"
    )


def main() -> None:
    args = parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import evaluate_conditions
    from refusal_circuit.graph import load_circuit_edge_names, make_logit_graph, set_selected_edges
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import read_json, write_json

    circuit_path = args.circuit_dir / f"eapig_top_{args.k}.json"
    if not circuit_path.exists():
        raise FileNotFoundError(f"Missing logit circuit file: {circuit_path}")

    data_dir = args.data_root / args.language
    metric_dir = args.results_root / args.language / "shared_metrics" / args.metric_name
    metadata_path = metric_dir / "metadata.json"
    direction_path = metric_dir / "refusal_direction.pt"
    metadata = read_json(metadata_path)
    if metadata["model"] != args.model:
        raise ValueError(f"Metric metadata is for {metadata['model']}, not {args.model}.")
    refusal_ids = [int(value) for value in metadata.get("refusal_token_ids", [])]
    if not refusal_ids:
        raise ValueError(f"Metric metadata has no refusal_token_ids: {metadata_path}")

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(direction_path, map_location=next(model.parameters()).device, weights_only=True)
    graph = make_logit_graph(model)
    edge_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, edge_names)

    metric_rows: list[dict] = []
    generation_rows: list[dict] = []
    condition_names = {
        "clean": "normal",
        "corrupt": "direction_ablated",
        "target_corrupt": "target_direction_ablated",
        "circuit_only": "circuit_only",
        "circuit_ablated": "circuit_ablated",
    }
    for label in ("harmful", "harmless"):
        rows = _selected_rows(
            data_dir,
            args.split,
            label,
            offset=args.prompt_offset,
            n=args.num_prompts,
        )
        prompts = format_instructions(model.tokenizer, [row["instruction"] for row in rows])
        conditions = evaluate_conditions(
            model,
            graph,
            prompts,
            direction=direction,
            target_hook=metadata["target_hook"],
            target_position=int(metadata["position"]),
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        for internal_name, public_name in condition_names.items():
            row = _condition_summary(
                label=label,
                condition_name=public_name,
                result=conditions[internal_name],
                clean_result=conditions["clean"],
                corrupt_result=conditions["corrupt"],
                refusal_ids=refusal_ids,
            )
            row.update(
                {
                    "model": args.model,
                    "dataset": "PolyRefuse",
                    "language": args.language,
                    "metric_name": args.metric_name,
                    "circuit_family": "logit_eapig_pilot",
                    "requested_k": args.k,
                    "retained_edges": len(edge_names),
                    "split": args.split,
                    "prompt_offset": args.prompt_offset,
                    "circuit_path": str(circuit_path),
                    "generated_text_saved": False,
                }
            )
            metric_rows.append(row)

        if args.generation_prompts > 0 and args.max_new_tokens > 0:
            for row in _generation_summary(
                model=model,
                graph=graph,
                prompts=prompts[: args.generation_prompts],
                label=label,
                language=args.language,
                direction=direction,
                target_hook=metadata["target_hook"],
                target_position=int(metadata["position"]),
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
            ):
                row.update(
                    {
                        "model": args.model,
                        "dataset": "PolyRefuse",
                        "language": args.language,
                        "metric_name": args.metric_name,
                        "circuit_family": "logit_eapig_pilot",
                        "requested_k": args.k,
                        "retained_edges": len(edge_names),
                        "split": args.split,
                        "prompt_offset": args.prompt_offset,
                        "circuit_path": str(circuit_path),
                    }
                )
                generation_rows.append(row)

    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_dir / "behavioral_metrics.csv", index=False)
    pd.DataFrame(generation_rows).to_csv(output_dir / "generation_substring_rates.csv", index=False)
    write_json(
        output_dir / "behavioral_validation_metadata.json",
        {
            "model": args.model,
            "dataset": "PolyRefuse",
            "language": args.language,
            "metric_name": args.metric_name,
            "metric": "next-token refusal log-odds circuit; diagnostics also report h_15,-2 projection",
            "circuit_family": "logit_eapig_pilot",
            "requested_k": args.k,
            "retained_edges": len(edge_names),
            "split": args.split,
            "prompt_offset": args.prompt_offset,
            "num_prompts_per_label": args.num_prompts,
            "generation_prompts_per_label": args.generation_prompts,
            "max_new_tokens": args.max_new_tokens,
            "circuit_path": str(circuit_path),
            "direction_path": str(direction_path),
            "metadata_path": str(metadata_path),
            "generated_text_saved": False,
            "notes": (
                "This evaluates circuits discovered against final next-token refusal log-odds. "
                "Generated completions are decoded only in memory for aggregate substring rates."
            ),
        },
    )
    print(f"Saved logit-circuit behavioral validation under {output_dir}")


if __name__ == "__main__":
    main()

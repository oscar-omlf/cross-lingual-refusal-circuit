#!/usr/bin/env python3
"""Cross-lingual behavioral validation for RQ2 circuits.

This script evaluates aggregate behavioral proxies for one circuit on one
language.  It intentionally writes no prompts or generated completions to disk.

The main checks are:

* harmful prompts: does circuit ablation reduce refusal-direction projection and
  first-token refusal log-odds?
* harmless prompts: does circuit intervention preserve the next-token
  distribution, measured by KL(clean || changed)?
* tiny generated-suffix sanity check: do aggregate refusal-substring rates move
  in the expected direction?  Generated text is kept only in memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_ROOT = Path("results/qwen2.5-7b-instruct/polyrefuse")
DEFAULT_DATA_ROOT = Path("data/processed/qwen2.5-7b-instruct/polyrefuse")
CIRCUITS_SUBDIR = "circuits_greedy_no_parent_prune_small_to_large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", required=True, choices=("en", "de", "zh", "th"))
    parser.add_argument(
        "--circuit-family",
        required=True,
        choices=("local", "english_transfer", "strict_shared", "majority", "consensus"),
        help="Which top-k circuit to evaluate.",
    )
    parser.add_argument("--metric-name", default="english_vector")
    parser.add_argument("--source-language", default="en", help="Source language for english_transfer.")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument(
        "--prompt-offset",
        type=int,
        default=32,
        help="Skip this many examples per label before taking --num-prompts.",
    )
    parser.add_argument("--generation-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _circuit_path(args: argparse.Namespace) -> Path:
    base = args.results_root
    metric = args.metric_name
    if args.circuit_family == "local":
        return (
            base
            / args.language
            / f"circuit_discovery_{metric}"
            / CIRCUITS_SUBDIR
            / f"eapig_top_{args.k}.json"
        )
    if args.circuit_family == "english_transfer":
        return (
            base
            / args.source_language
            / f"circuit_discovery_{metric}"
            / CIRCUITS_SUBDIR
            / f"eapig_top_{args.k}.json"
        )
    shared_map = {
        "strict_shared": "intersection_en_de_zh_th",
        "majority": "majority_at_least_3_of_4",
        "consensus": "consensus_mean_scores",
    }
    return (
        base
        / "shared_circuits"
        / f"circuit_discovery_{metric}"
        / shared_map[args.circuit_family]
        / CIRCUITS_SUBDIR
        / f"eapig_top_{args.k}.json"
    )


def _output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.results_root
        / f"behavioral_validation_{args.metric_name}"
        / args.circuit_family
        / args.language
        / f"k{args.k}_offset{args.prompt_offset}_n{args.num_prompts}"
    )


def _selected_rows(data_dir: Path, split: str, label: str, offset: int, n: int) -> list[dict]:
    from refusal_circuit.data import load_processed_split

    rows = load_processed_split(split, label, data_dir=data_dir)
    selected = rows[offset : offset + n]
    if len(selected) < n:
        raise ValueError(
            f"Requested {n} {label} {split} rows from offset {offset}, "
            f"but only found {len(selected)}."
        )
    return selected


def _condition_summary(
    *,
    label: str,
    condition_name: str,
    result,
    clean_result,
    corrupt_result,
    refusal_ids: list[int],
) -> dict:
    import torch

    from refusal_circuit.metrics import mean_last_token_kl, refusal_log_odds

    projection = result.projection.float()
    clean_projection = clean_result.projection.float()
    corrupt_projection = corrupt_result.projection.float()
    denominator = (clean_projection.mean() - corrupt_projection.mean()).item()
    normalized_retention = (
        float("nan")
        if abs(denominator) < 1e-8
        else ((projection.mean() - corrupt_projection.mean()).item() / denominator)
    )
    log_odds = refusal_log_odds(result.logits.unsqueeze(1), refusal_ids).float()
    clean_logits = clean_result.logits.unsqueeze(1)
    current_logits = result.logits.unsqueeze(1)
    top1_changed = (
        clean_result.logits.argmax(dim=-1) != result.logits.argmax(dim=-1)
    ).float().mean().item()
    kl = 0.0 if condition_name == "normal" else mean_last_token_kl(clean_logits, current_logits).item()
    return {
        "label": label,
        "condition": condition_name,
        "n_prompts": int(len(projection)),
        "projection_mean": projection.mean().item(),
        "projection_std": projection.std(unbiased=False).item(),
        "projection_delta_from_normal": (projection.mean() - clean_projection.mean()).item(),
        "projection_retention_vs_direction_ablated": normalized_retention,
        "first_token_refusal_log_odds_mean": log_odds.mean().item(),
        "first_token_refusal_log_odds_std": log_odds.std(unbiased=False).item(),
        "first_token_refusal_log_odds_gt0_rate": (log_odds > 0).float().mean().item(),
        "first_token_kl_clean_to_condition": kl,
        "first_token_top1_changed_rate": top1_changed,
    }


def _generation_summary(
    *,
    model,
    graph,
    prompts: list[str],
    label: str,
    language: str,
    direction,
    target_hook: str,
    target_position: int,
    max_new_tokens: int,
    max_length: int | None,
) -> list[dict]:
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import _greedy_completion
    from refusal_circuit.metrics import REFUSAL_PREFIXES, is_refusal, refusal_prefixes_for_language
    from refusal_circuit.model import tokenize_prompts

    prefixes = tuple(dict.fromkeys(refusal_prefixes_for_language(language) + REFUSAL_PREFIXES))
    conditions = {
        "normal": "normal",
        "direction_ablated": "refusal_direction_ablated",
        "circuit_only": "circuit_only",
        "circuit_ablated": "circuit_ablated",
    }
    counts = {condition: 0 for condition in conditions}
    for prompt in tqdm(prompts, desc=f"Generating {label}", leave=False):
        tokens, _ = tokenize_prompts(model, [prompt], max_length=max_length)
        for condition, generation_condition in conditions.items():
            completion_ids = _greedy_completion(
                model,
                graph,
                tokens,
                direction=direction,
                target_hook=target_hook,
                target_position=target_position,
                condition=generation_condition,
                max_new_tokens=max_new_tokens,
            )
            completion = model.tokenizer.decode(completion_ids, skip_special_tokens=True)
            counts[condition] += int(is_refusal(completion, prefixes=prefixes))
    n_prompts = len(prompts)
    return [
        {
            "label": label,
            "condition": condition,
            "n_generation_prompts": n_prompts,
            "max_new_tokens": max_new_tokens,
            "refusal_substring_rate": counts[condition] / max(n_prompts, 1),
            "prefix_protocol": "language_local_plus_english",
            "n_prefixes": len(prefixes),
            "generated_text_saved": False,
        }
        for condition in conditions
    ]


def main() -> None:
    args = parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import evaluate_conditions
    from refusal_circuit.graph import load_circuit_edge_names, make_projection_graph, set_selected_edges
    from refusal_circuit.model import format_instructions, load_model
    from refusal_circuit.paths import read_json, write_json

    output_dir = _output_dir(args)
    circuit_path = _circuit_path(args)
    if not circuit_path.exists():
        raise FileNotFoundError(f"Missing circuit file: {circuit_path}")

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
    target_layer = int(metadata["layer"])
    graph = make_projection_graph(model, target_layer)
    edge_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, edge_names)

    metric_rows = []
    generation_rows = []
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
        condition_names = {
            "clean": "normal",
            "corrupt": "direction_ablated",
            "target_corrupt": "target_direction_ablated",
            "circuit_only": "circuit_only",
            "circuit_ablated": "circuit_ablated",
        }
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
                    "circuit_family": args.circuit_family,
                    "source_language": args.source_language if args.circuit_family == "english_transfer" else "",
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
            generation_prompts = prompts[: args.generation_prompts]
            for row in _generation_summary(
                model=model,
                graph=graph,
                prompts=generation_prompts,
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
                        "circuit_family": args.circuit_family,
                        "source_language": args.source_language
                        if args.circuit_family == "english_transfer"
                        else "",
                        "requested_k": args.k,
                        "retained_edges": len(edge_names),
                        "split": args.split,
                        "prompt_offset": args.prompt_offset,
                        "circuit_path": str(circuit_path),
                    }
                )
                generation_rows.append(row)

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
            "metric": "h_15,-2 dot normalized English refusal direction",
            "circuit_family": args.circuit_family,
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
                "First-token metrics are primary behavioral proxies. Generated completions are "
                "only decoded in memory for aggregate substring rates and are not persisted."
            ),
        },
    )
    print(f"Saved RQ2 behavioral validation under {output_dir}")


if __name__ == "__main__":
    main()

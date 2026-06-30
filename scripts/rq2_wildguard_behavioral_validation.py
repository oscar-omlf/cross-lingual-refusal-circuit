#!/usr/bin/env python3
"""WildGuard behavioral validation for RQ2 multilingual refusal circuits.

The script generates short deterministic completions under several circuit
conditions, classifies them with WildGuard, and writes aggregate labels only.
It never writes prompts, completions, or raw WildGuard outputs to disk.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401
from rq2_behavioral_validation import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    DEFAULT_RESULTS_ROOT,
    _circuit_path,
    _condition_summary,
    _selected_rows,
)
from rq2_wildguard_smoke_test import INSTRUCTION_FORMAT, parse_wildguard_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--wildguard-model", default="allenai/wildguard")
    parser.add_argument("--language", required=True, choices=("en", "de", "zh", "th"))
    parser.add_argument(
        "--circuit-family",
        required=True,
        choices=("local", "english_transfer", "strict_shared", "majority", "consensus"),
    )
    parser.add_argument("--metric-name", default="english_vector")
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--prompt-offset", type=int, default=32)
    parser.add_argument(
        "--generation-prompts",
        type=int,
        default=8,
        help="Number of prompts per label to generate/classify with WildGuard.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help=(
            "Number of tokens to generate before WildGuard classification. "
            "Must be at least 32; shorter benign fragments caused false "
            "refusal labels in the validation pilot."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--classifier-batch-size", type=int, default=4)
    parser.add_argument("--classifier-max-length", type=int, default=2048)
    parser.add_argument("--classifier-max-new-tokens", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir or (
        args.results_root
        / f"wildguard_behavioral_validation_{args.metric_name}"
        / args.circuit_family
        / args.language
        / f"k{args.k}_offset{args.prompt_offset}_n{args.num_prompts}_gen{args.generation_prompts}"
    )


def _batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _generate_records(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.circuit import _greedy_completion, evaluate_conditions
    from refusal_circuit.graph import load_circuit_edge_names, make_projection_graph, set_selected_edges
    from refusal_circuit.model import format_instructions, load_model, tokenize_prompts
    from refusal_circuit.paths import read_json

    circuit_path = _circuit_path(args)
    if not circuit_path.exists():
        raise FileNotFoundError(f"Missing circuit file: {circuit_path}")
    data_dir = args.data_root / args.language
    metric_dir = args.results_root / args.language / "shared_metrics" / args.metric_name
    metadata_path = metric_dir / "metadata.json"
    direction_path = metric_dir / "refusal_direction.pt"
    metadata = read_json(metadata_path)
    refusal_ids = [int(value) for value in metadata.get("refusal_token_ids", [])]
    if not refusal_ids:
        raise ValueError(f"Metric metadata has no refusal_token_ids: {metadata_path}")

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(direction_path, map_location=next(model.parameters()).device, weights_only=True)
    target_layer = int(metadata["layer"])
    graph = make_projection_graph(model, target_layer)
    edge_names = load_circuit_edge_names(circuit_path)
    set_selected_edges(graph, edge_names)

    metric_rows: list[dict] = []
    wildguard_inputs: list[dict] = []
    condition_names = {
        "clean": "normal",
        "corrupt": "direction_ablated",
        "circuit_only": "circuit_only",
        "circuit_ablated": "circuit_ablated",
    }
    generation_conditions = {
        "normal": "normal",
        "direction_ablated": "refusal_direction_ablated",
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
        raw_instructions = [row["instruction"] for row in rows]
        prompts = format_instructions(model.tokenizer, raw_instructions)
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
                    "circuit_family": args.circuit_family,
                    "requested_k": args.k,
                    "retained_edges": len(edge_names),
                    "split": args.split,
                    "prompt_offset": args.prompt_offset,
                    "wildguard_used": True,
                    "generated_text_saved": False,
                }
            )
            metric_rows.append(row)

        selected_for_generation = list(zip(rows, raw_instructions, prompts))[: args.generation_prompts]
        for local_index, (source_row, raw_instruction, prompt) in enumerate(
            tqdm(selected_for_generation, desc=f"Generating {label}")
        ):
            tokens, _attention_mask = tokenize_prompts(model, [prompt], max_length=args.max_length)
            for condition, generation_condition in generation_conditions.items():
                completion_ids = _greedy_completion(
                    model,
                    graph,
                    tokens,
                    direction=direction,
                    target_hook=metadata["target_hook"],
                    target_position=int(metadata["position"]),
                    condition=generation_condition,
                    max_new_tokens=args.max_new_tokens,
                )
                # Kept in RAM only until WildGuard classification below.
                completion = model.tokenizer.decode(completion_ids, skip_special_tokens=True)
                wildguard_inputs.append(
                    {
                        "label": label,
                        "condition": condition,
                        "local_generation_index": local_index,
                        "row_id": source_row.get("id", ""),
                        "parallel_id": source_row.get("parallel_id", ""),
                        "generated_token_count": len(completion_ids),
                        "hit_max_new_tokens": len(completion_ids) >= args.max_new_tokens,
                        "prompt": raw_instruction,
                        "response": completion,
                    }
                )

    run_metadata = {
        "model": args.model,
        "wildguard_model": args.wildguard_model,
        "dataset": "PolyRefuse",
        "language": args.language,
        "metric_name": args.metric_name,
        "metric": "h_15,-2 dot normalized English refusal direction",
        "circuit_family": args.circuit_family,
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "split": args.split,
        "prompt_offset": args.prompt_offset,
        "num_prompts_per_label_for_first_token_metrics": args.num_prompts,
        "generation_prompts_per_label_for_wildguard": args.generation_prompts,
        "max_new_tokens": args.max_new_tokens,
        "circuit_path": str(circuit_path),
        "direction_path": str(direction_path),
        "metadata_path": str(metadata_path),
        "generated_text_saved": False,
        "raw_wildguard_output_saved": False,
    }

    del model, graph, direction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_rows, wildguard_inputs, run_metadata


def _classify_with_wildguard(args: argparse.Namespace, records: list[dict]) -> list[dict]:
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("WildGuard validation requires CUDA; run it through SLURM.")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.wildguard_model, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.wildguard_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=token,
    ).to("cuda").eval()

    parsed_rows: list[dict] = []
    for batch in tqdm(
        list(_batches(records, args.classifier_batch_size)),
        desc="WildGuard classify",
    ):
        texts = [
            INSTRUCTION_FORMAT.format(prompt=record["prompt"], response=record["response"])
            for record in batch
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.classifier_max_length,
            add_special_tokens=False,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.classifier_max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        suffix = generated[:, inputs.input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(suffix, skip_special_tokens=True)
        for record, text in zip(batch, decoded):
            # Drop prompt/response before anything is saved.
            base = {
                key: value
                for key, value in record.items()
                if key not in {"prompt", "response"}
            }
            try:
                labels = parse_wildguard_output(text)
                base.update(labels)
                base["parse_ok"] = True
            except Exception:
                base.update(
                    {
                        "request_harmful": None,
                        "refusal": None,
                        "response_harmful": None,
                        "parse_ok": False,
                    }
                )
            parsed_rows.append(base)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return parsed_rows


def _aggregate_wildguard(rows: list[dict]) -> list[dict]:
    import pandas as pd

    frame = pd.DataFrame(rows)
    output = []
    for (label, condition), group in frame.groupby(["label", "condition"], dropna=False):
        parsed = group[group["parse_ok"] == True]  # noqa: E712
        row = {
            "label": label,
            "condition": condition,
            "n_generated": int(len(group)),
            "n_parsed": int(len(parsed)),
            "parse_failure_rate": 1.0 - (len(parsed) / max(len(group), 1)),
            "request_harmful_rate": parsed["request_harmful"].mean() if len(parsed) else float("nan"),
            "refusal_rate": parsed["refusal"].mean() if len(parsed) else float("nan"),
            "response_harmful_rate": parsed["response_harmful"].mean() if len(parsed) else float("nan"),
        }
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    import pandas as pd

    from refusal_circuit.paths import write_json

    if args.max_new_tokens < 32:
        raise ValueError(
            "--max-new-tokens must be at least 32 for WildGuard validation. "
            "A harmless pilot with 8-token truncated responses produced a "
            "false refusal label."
        )

    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows, wildguard_inputs, metadata = _generate_records(args)
    classified_rows = _classify_with_wildguard(args, wildguard_inputs)
    aggregate_rows = _aggregate_wildguard(classified_rows)

    pd.DataFrame(metric_rows).to_csv(output_dir / "first_token_and_kl_metrics.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "wildguard_aggregate.csv", index=False)
    # Instance labels contain no prompt/completion text and are useful for
    # confidence intervals or auditing parser failures.
    pd.DataFrame(classified_rows).to_csv(output_dir / "wildguard_instance_labels_no_text.csv", index=False)
    write_json(output_dir / "wildguard_behavioral_validation_metadata.json", metadata)
    print(f"Saved WildGuard behavioral validation under {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate and judge refusal behavior across a nested circuit-size sweep.

The behavior model and WildGuard are loaded sequentially. Prompt text,
completions, and raw judge output exist only in RAM and are removed before
instance-level labels are written. Baselines are invariant to k and are
generated once per job.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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
        yield items[start : start + batch_size]


def _generate_batch(
    model,
    graph,
    prompts: list[str],
    *,
    direction,
    target_hook: str,
    target_position: int,
    condition: str,
    max_new_tokens: int,
    max_length: int,
) -> list[list[int]]:
    """Greedy batch generation without persisting text or assuming fixed caches."""
    import torch

    from refusal_circuit.circuit import _mixed_forward, _normal_forward
    from refusal_circuit.interventions import all_layer_ablation_hooks
    from refusal_circuit.model import tokenize_prompts

    tokens, attention_mask = tokenize_prompts(model, prompts, max_length=max_length)
    generated = [[] for _ in prompts]
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=tokens.device)
    eos_id = model.tokenizer.eos_token_id
    filler_id = eos_id if eos_id is not None else model.tokenizer.pad_token_id
    for _ in range(max_new_tokens):
        if condition == "normal":
            _projection, logits = _normal_forward(
                model,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
            )
        elif condition == "direction_ablated":
            _projection, logits = _normal_forward(
                model,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                fwd_hooks=all_layer_ablation_hooks(model, direction),
            )
        elif condition in {"circuit_only", "circuit_ablated"}:
            _projection, logits = _mixed_forward(
                model,
                graph,
                tokens,
                attention_mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                patch_selected=condition == "circuit_ablated",
            )
        else:  # pragma: no cover
            raise ValueError(condition)
        next_tokens = logits[:, -1].argmax(dim=-1)
        if filler_id is not None:
            next_tokens = torch.where(finished, torch.full_like(next_tokens, filler_id), next_tokens)
        for index, token in enumerate(next_tokens.tolist()):
            if not finished[index]:
                generated[index].append(int(token))
        if eos_id is not None:
            finished |= next_tokens == eos_id
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((len(prompts), 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        if bool(finished.all()):
            break
    return generated


def _classify_wildguard(records: list[dict], args) -> list[dict]:
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rq2_wildguard_smoke_test import INSTRUCTION_FORMAT, parse_wildguard_output

    if not torch.cuda.is_available():
        raise RuntimeError("Behavior judging requires CUDA and must run through SLURM.")
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    judge = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=token,
    ).to("cuda").eval()
    output: list[dict] = []
    for batch in tqdm(list(_batches(records, args.classifier_batch_size)), desc="WildGuard"):
        texts = [
            INSTRUCTION_FORMAT.format(prompt=row["_prompt"], response=row["_response"])
            for row in batch
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
            generated = judge.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.classifier_max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        suffix = generated[:, inputs.input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(suffix, skip_special_tokens=True)
        for source, raw_output in zip(batch, decoded):
            # Only non-text fields cross this boundary into saved results.
            row = {key: value for key, value in source.items() if not key.startswith("_")}
            try:
                row.update(parse_wildguard_output(raw_output))
                row["parse_ok"] = True
            except Exception:
                row.update({"request_harmful": None, "refusal": None, "response_harmful": None})
                row["parse_ok"] = False
            output.append(row)
    del judge
    gc.collect()
    torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--language", required=True, choices=("en", "de", "zh", "th"))
    parser.add_argument("--split", default="development")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional JSONL override for selected-k benign collateral checks.",
    )
    parser.add_argument("--row-label", choices=("harmful", "harmless"), default="harmful")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--circuit-label", required=True)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=None)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--classifier-batch-size", type=int, default=None)
    parser.add_argument("--classifier-max-length", type=int, default=2048)
    parser.add_argument("--classifier-max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument(
        "--control-conditions",
        nargs="+",
        choices=("normal", "direction_ablated"),
        default=None,
        help=(
            "Mask-independent trajectories to generate once. Defaults to normal and "
            "direction_ablated; use --skip-controls to generate neither."
        ),
    )
    parser.add_argument(
        "--circuit-conditions",
        nargs="+",
        choices=("circuit_only", "circuit_ablated"),
        default=None,
        help=(
            "Circuit-dependent trajectories to generate at every requested k. "
            "Defaults to the circuit conditions frozen in the config."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    from refusal_circuit.graph import load_circuit_edge_names, make_projection_graph, set_selected_edges
    from refusal_circuit.metrics import REFUSAL_PREFIXES, is_refusal, refusal_prefixes_for_language
    from refusal_circuit.model import format_instructions, load_model

    cfg = _read(args.config)
    behavior_cfg = cfg["behavior"]
    args.judge_model = args.judge_model or behavior_cfg["judge_model"]
    args.classifier_batch_size = args.classifier_batch_size or behavior_cfg["classifier_batch_size"]
    generation_batch_size = args.generation_batch_size or behavior_cfg["generation_batch_size"]
    max_new_tokens = args.max_new_tokens or behavior_cfg["max_new_tokens"]
    circuit_conditions = args.circuit_conditions or [
        condition
        for condition in behavior_cfg["conditions_in_curve"]
        if condition in {"circuit_only", "circuit_ablated"}
    ]
    if not circuit_conditions:
        raise ValueError("At least one circuit-dependent condition is required.")
    if args.skip_controls and args.control_conditions:
        raise ValueError("--skip-controls and --control-conditions are mutually exclusive.")
    control_conditions = [] if args.skip_controls else (
        args.control_conditions or ["normal", "direction_ablated"]
    )
    if max_new_tokens < 32:
        raise ValueError("At least 32 generated tokens are required for the frozen refusal judge protocol.")
    k_values = args.k_values or cfg["circuit_sweep"]["behavior_requested_k"]
    data_path = args.data_path or (
        _resolve(cfg["dataset"]["phase2_data_root"])
        / args.language
        / f"{args.split}.jsonl"
    )
    data_path = _resolve(data_path)
    source_rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line]
    source_rows = [row for row in source_rows if row.get("label") == args.row_label]
    default_n = (
        cfg["dataset"]["curve_generation_prompts"]
        if args.split == "development"
        else cfg["dataset"]["confirmation_generation_prompts"]
    )
    source_rows = source_rows[: (args.num_prompts or default_n)]
    if not source_rows:
        raise ValueError(f"No prompts selected from {data_path}")
    circuits_dir = _resolve(args.circuits_dir)
    for k in k_values:
        if not (circuits_dir / f"eapig_top_{k}.json").exists():
            raise FileNotFoundError(circuits_dir / f"eapig_top_{k}.json")

    results_root = _resolve(cfg["artifacts"]["results_root"])
    metric_root = results_root / args.language / "shared_metrics" / cfg["metric"]["name"]
    metadata = _read(metric_root / "metadata.json")
    model = load_model(cfg["model"]["name"], device=args.device, dtype=args.dtype)
    direction = torch.load(
        metric_root / "refusal_direction.pt",
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    formatted = format_instructions(model.tokenizer, [row["instruction"] for row in source_rows])
    prefixes = tuple(dict.fromkeys(refusal_prefixes_for_language(args.language) + REFUSAL_PREFIXES))
    in_memory_records: list[dict] = []

    def generate_condition(condition: str, *, k: int, retained: int, graph) -> None:
        for indices in tqdm(
            list(_batches(list(range(len(source_rows))), generation_batch_size)),
            desc=f"{args.circuit_label} {args.language} k={k} {condition}",
        ):
            prompt_batch = [formatted[index] for index in indices]
            token_lists = _generate_batch(
                model,
                graph,
                prompt_batch,
                direction=direction,
                target_hook=metadata["target_hook"],
                target_position=int(metadata["position"]),
                condition=condition,
                max_new_tokens=max_new_tokens,
                max_length=args.max_length,
            )
            completions = model.tokenizer.batch_decode(token_lists, skip_special_tokens=True)
            for index, token_ids, completion in zip(indices, token_lists, completions):
                source = source_rows[index]
                in_memory_records.append(
                    {
                        "parallel_id": str(source["parallel_id"]),
                        "language": args.language,
                        "split": args.split,
                        "row_label": args.row_label,
                        "circuit_label": args.circuit_label,
                        "requested_k": k,
                        "retained_edges": retained,
                        "edge_fraction": (
                            retained / int(cfg["metric"]["eligible_edges"])
                            if retained
                            else 0.0
                        ),
                        "condition": condition,
                        "generated_token_count": len(token_ids),
                        "hit_max_new_tokens": len(token_ids) >= max_new_tokens,
                        "refusal_substring": is_refusal(completion, prefixes=prefixes),
                        "_prompt": source["instruction"],
                        "_response": completion,
                    }
                )

    # The control trajectories do not depend on the mask and are generated once.
    dummy_graph = make_projection_graph(model, int(metadata["layer"]))
    for condition in control_conditions:
        generate_condition(condition, k=0, retained=0, graph=dummy_graph)
    for k in k_values:
        circuit_path = circuits_dir / f"eapig_top_{k}.json"
        edge_names = load_circuit_edge_names(circuit_path)
        graph = make_projection_graph(model, int(metadata["layer"]))
        set_selected_edges(graph, edge_names)
        for condition in circuit_conditions:
            generate_condition(condition, k=k, retained=len(edge_names), graph=graph)

    del model, direction, dummy_graph
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    classified = _classify_wildguard(in_memory_records, args)
    del in_memory_records
    gc.collect()

    frame = pd.DataFrame(classified)
    aggregate_rows = []
    group_columns = ["language", "split", "circuit_label", "requested_k", "condition"]
    for keys, group in frame.groupby(group_columns, dropna=False):
        parsed = group[group["parse_ok"] == True]  # noqa: E712
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "retained_edges": int(group["retained_edges"].iloc[0]),
                "edge_fraction": float(group["edge_fraction"].iloc[0]),
                "n_generated": len(group),
                "n_parsed": len(parsed),
                "parse_failure_rate": 1.0 - len(parsed) / max(len(group), 1),
                "refusal_rate": parsed["refusal"].mean() if len(parsed) else float("nan"),
                "response_harmful_rate": (
                    parsed["response_harmful"].mean() if len(parsed) else float("nan")
                ),
                "refusal_substring_rate": group["refusal_substring"].mean(),
            }
        )
        aggregate_rows.append(row)

    output_dir = args.output_dir or (
        _resolve(cfg["artifacts"]["phase2_results_root"])
        / "behavior"
        / args.split
        / args.circuit_label
        / args.language
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "instance_labels_no_text.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "aggregate.csv", index=False)
    run_metadata = {
        "config": str(args.config),
        "model": cfg["model"]["name"],
        "judge_model": args.judge_model,
        "language": args.language,
        "split": args.split,
        "row_label": args.row_label,
        "data_path": str(data_path),
        "circuit_label": args.circuit_label,
        "circuits_dir": str(circuits_dir),
        "requested_k": k_values,
        "n_prompts": len(source_rows),
        "max_new_tokens": max_new_tokens,
        "generation_batch_size": generation_batch_size,
        "control_conditions": control_conditions,
        "circuit_conditions": circuit_conditions,
        "controls_generated": bool(control_conditions),
        "prompt_text_saved": False,
        "generated_text_saved": False,
        "raw_judge_output_saved": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved Phase 2 behavioral labels under {output_dir}")


if __name__ == "__main__":
    main()

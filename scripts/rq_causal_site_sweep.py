#!/usr/bin/env python3
"""Causal sweep for local refusal-direction ablation sites.

This diagnostic tests whether removing a saved refusal direction at one
``resid_pre`` layer/token position is enough to change next-token refusal
log-odds.  It writes aggregate scalar metrics only: no prompts, completions, or
raw harmful text are persisted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--direction-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--labels", nargs="+", choices=("harmful", "harmless"), default=("harmful", "harmless"))
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=None,
        help="Negative/positive token positions to test. Defaults to metadata post_instruction_positions or -5..-1.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=("all",),
        help="Layer indices to test, or 'all' for every resid_pre layer.",
    )
    return parser.parse_args()


def _parse_layers(raw_layers: list[str], n_layers: int) -> list[int]:
    if len(raw_layers) == 1 and raw_layers[0] == "all":
        return list(range(n_layers))
    layers = [int(layer) for layer in raw_layers]
    bad = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad:
        raise ValueError(f"Layer indices out of range for n_layers={n_layers}: {bad}")
    return layers


def _batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _mean(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _std(values):
    if not values:
        return float("nan")
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _template_position_metadata(tokenizer, positions: list[int]) -> list[dict]:
    from refusal_circuit.model import format_instruction

    marker = "RQ_CAUSAL_SITE_SWEEP_MARKER_9d0b"
    prompt = format_instruction(tokenizer, marker)
    suffix = prompt.rsplit(marker, 1)[1] if marker in prompt else ""
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    records = []
    for position in positions:
        decoded = None
        if position < 0 and -position <= len(suffix_ids):
            decoded = tokenizer.decode([suffix_ids[position]])
        records.append(
            {
                "position": position,
                "decoded_template_token": decoded,
            }
        )
    return records


def _run_logits_and_projection(
    model,
    tokens,
    attention_mask,
    *,
    direction,
    hook_name: str,
    position: int,
    fwd_hooks=None,
):
    from refusal_circuit.metrics import ProjectionCapture

    capture = ProjectionCapture(direction=direction, position=position)
    hooks = list(fwd_hooks or []) + [(hook_name, capture.hook)]
    with model.hooks(fwd_hooks=hooks):
        logits = model(tokens, attention_mask=attention_mask)
    return logits.detach(), capture.values().detach()


def _site_ablation_hook(direction, position: int, projection_before: list, projection_after: list):
    import torch

    def hook(activations: torch.Tensor, hook):
        vector = direction.to(device=activations.device, dtype=torch.float32)
        vector = vector / vector.norm().clamp_min(1e-8)
        index = position if position >= 0 else activations.shape[1] + position
        if not 0 <= index < activations.shape[1]:
            raise IndexError(f"Position {position} is out of range for sequence length {activations.shape[1]}.")
        selected = activations[:, index].float()
        before = torch.einsum("bd,d->b", selected, vector)
        ablated = selected - before.unsqueeze(-1) * vector
        after = torch.einsum("bd,d->b", ablated, vector)
        projection_before.append(before.detach().cpu())
        projection_after.append(after.detach().cpu())
        result = activations.clone()
        result[:, index] = ablated.to(dtype=activations.dtype)
        return result

    return hook


def _summarize_logits(logits, refusal_ids: list[int]):
    import torch

    from refusal_circuit.metrics import refusal_log_odds

    log_odds = refusal_log_odds(logits, refusal_ids).detach().double().cpu()
    top1 = logits[:, -1].argmax(dim=-1).detach().cpu()
    return log_odds, top1


def _kl_clean_to_changed(clean_logits, changed_logits):
    from refusal_circuit.metrics import mean_last_token_kl

    return float(mean_last_token_kl(clean_logits, changed_logits).detach().cpu().item())


def _kl_last_logits(clean_last_logits, changed_last_logits):
    clean_log_probs = clean_last_logits.float().log_softmax(dim=-1)
    changed_log_probs = changed_last_logits.float().log_softmax(dim=-1)
    return float((clean_log_probs.exp() * (clean_log_probs - changed_log_probs)).sum(dim=-1).mean().detach().cpu().item())


def main() -> None:
    args = parse_args()

    import torch

    from refusal_circuit.data import load_processed_split
    from refusal_circuit.interventions import all_layer_ablation_hooks
    from refusal_circuit.model import format_instructions, load_model, tokenize_prompts
    from refusal_circuit.paths import read_json, write_json

    metadata = read_json(args.metadata_path)
    if metadata.get("model") != args.model:
        raise ValueError(f"Direction metadata is for {metadata.get('model')}, not {args.model}.")
    refusal_ids = [int(value) for value in metadata.get("refusal_token_ids", [])]
    if not refusal_ids:
        raise ValueError(f"No refusal_token_ids in {args.metadata_path}")

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    direction = torch.load(
        args.direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    layers = _parse_layers(list(args.layers), model.cfg.n_layers)
    positions = (
        [int(position) for position in args.positions]
        if args.positions is not None
        else [int(position) for position in metadata.get("post_instruction_positions", [-5, -4, -3, -2, -1])]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    baseline_out: list[dict] = []

    with torch.inference_mode():
        for label in args.labels:
            source_rows = load_processed_split(args.split, label, data_dir=args.data_dir)
            selected_rows = source_rows[args.prompt_offset : args.prompt_offset + args.num_prompts]
            if len(selected_rows) < args.num_prompts:
                raise ValueError(
                    f"Requested {args.num_prompts} {label} rows from offset {args.prompt_offset}, "
                    f"but only found {len(selected_rows)}."
                )
            prompts = format_instructions(model.tokenizer, [row["instruction"] for row in selected_rows])

            clean_log_odds_all = []
            global_log_odds_all = []
            clean_top1_all = []
            global_top1_all = []
            clean_projection_selected_site = []
            global_projection_selected_site = []
            global_kl_values = []
            batch_payloads = []

            # Baselines at the originally selected direction site.
            baseline_hook = metadata["target_hook"]
            baseline_position = int(metadata["position"])
            for prompt_batch in _batches(prompts, args.batch_size):
                tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=args.max_length)
                clean_logits, clean_projection = _run_logits_and_projection(
                    model,
                    tokens,
                    attention_mask,
                    direction=direction,
                    hook_name=baseline_hook,
                    position=baseline_position,
                )
                global_logits, global_projection = _run_logits_and_projection(
                    model,
                    tokens,
                    attention_mask,
                    direction=direction,
                    hook_name=baseline_hook,
                    position=baseline_position,
                    fwd_hooks=all_layer_ablation_hooks(model, direction),
                )
                clean_log_odds, clean_top1 = _summarize_logits(clean_logits, refusal_ids)
                global_log_odds, global_top1 = _summarize_logits(global_logits, refusal_ids)
                clean_log_odds_all.extend(clean_log_odds.tolist())
                global_log_odds_all.extend(global_log_odds.tolist())
                clean_top1_all.extend(clean_top1.tolist())
                global_top1_all.extend(global_top1.tolist())
                clean_projection_selected_site.extend(clean_projection.double().cpu().tolist())
                global_projection_selected_site.extend(global_projection.double().cpu().tolist())
                global_kl_values.append(_kl_clean_to_changed(clean_logits, global_logits))
                batch_payloads.append(
                    {
                        "prompts": prompt_batch,
                        "clean_last_logits": clean_logits[:, -1].detach().float().cpu(),
                    }
                )

            clean_mean = _mean(clean_log_odds_all)
            global_mean = _mean(global_log_odds_all)
            denominator = clean_mean - global_mean
            baseline_out.append(
                {
                    "label": label,
                    "n_prompts": len(prompts),
                    "clean_log_odds_mean": clean_mean,
                    "clean_log_odds_std": _std(clean_log_odds_all),
                    "clean_log_odds_gt0_rate": _mean([float(value > 0.0) for value in clean_log_odds_all]),
                    "global_ablation_log_odds_mean": global_mean,
                    "global_ablation_log_odds_std": _std(global_log_odds_all),
                    "global_ablation_log_odds_gt0_rate": _mean([float(value > 0.0) for value in global_log_odds_all]),
                    "global_ablation_kl_clean_to_changed": _mean(global_kl_values),
                    "global_ablation_top1_changed_rate": _mean(
                        [float(a != b) for a, b in zip(clean_top1_all, global_top1_all)]
                    ),
                    "clean_projection_at_metadata_site_mean": _mean(clean_projection_selected_site),
                    "global_ablation_projection_at_metadata_site_mean": _mean(global_projection_selected_site),
                    "metadata_layer": int(metadata["layer"]),
                    "metadata_position": baseline_position,
                    "metadata_target_hook": baseline_hook,
                }
            )

            for layer in layers:
                hook_name = f"blocks.{layer}.hook_resid_pre"
                for position in positions:
                    site_log_odds_all = []
                    site_top1_all = []
                    site_projection_before = []
                    site_projection_after = []
                    site_kl_values = []

                    for batch_payload in batch_payloads:
                        prompt_batch = batch_payload["prompts"]
                        tokens, attention_mask = tokenize_prompts(model, prompt_batch, max_length=args.max_length)
                        projection_before_batches = []
                        projection_after_batches = []
                        site_logits = model.run_with_hooks(
                            tokens,
                            attention_mask=attention_mask,
                            fwd_hooks=[
                                (
                                    hook_name,
                                    _site_ablation_hook(
                                        direction,
                                        position,
                                        projection_before_batches,
                                        projection_after_batches,
                                    ),
                                )
                            ],
                        )
                        site_log_odds, site_top1 = _summarize_logits(site_logits, refusal_ids)
                        site_log_odds_all.extend(site_log_odds.tolist())
                        site_top1_all.extend(site_top1.tolist())
                        site_projection_before.extend(
                            torch.cat(projection_before_batches).double().tolist()
                            if projection_before_batches
                            else []
                        )
                        site_projection_after.extend(
                            torch.cat(projection_after_batches).double().tolist()
                            if projection_after_batches
                            else []
                        )
                        site_kl_values.append(
                            _kl_last_logits(
                                batch_payload["clean_last_logits"].to(site_logits.device),
                                site_logits[:, -1],
                            )
                        )

                    site_mean = _mean(site_log_odds_all)
                    drop = clean_mean - site_mean
                    rows_out.append(
                        {
                            "label": label,
                            "layer": layer,
                            "position": position,
                            "hook": hook_name,
                            "n_prompts": len(prompts),
                            "clean_log_odds_mean": clean_mean,
                            "global_ablation_log_odds_mean": global_mean,
                            "site_ablation_log_odds_mean": site_mean,
                            "site_ablation_log_odds_std": _std(site_log_odds_all),
                            "site_ablation_log_odds_gt0_rate": _mean(
                                [float(value > 0.0) for value in site_log_odds_all]
                            ),
                            "site_drop_from_clean": drop,
                            "site_fraction_of_global_drop": drop / denominator if abs(denominator) > 1e-8 else float("nan"),
                            "site_kl_clean_to_changed": _mean(site_kl_values),
                            "site_top1_changed_rate": _mean(
                                [float(a != b) for a, b in zip(clean_top1_all, site_top1_all)]
                            ),
                            "site_projection_before_mean": _mean(site_projection_before),
                            "site_projection_after_mean": _mean(site_projection_after),
                            "site_projection_drop": _mean(site_projection_before) - _mean(site_projection_after),
                            "metadata_layer": int(metadata["layer"]),
                            "metadata_position": baseline_position,
                            "is_metadata_site": layer == int(metadata["layer"]) and position == baseline_position,
                        }
                    )

    sweep_path = args.output_dir / "causal_site_sweep.csv"
    baseline_path = args.output_dir / "causal_site_baselines.csv"
    if rows_out:
        with sweep_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
    if baseline_out:
        with baseline_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(baseline_out[0].keys()))
            writer.writeheader()
            writer.writerows(baseline_out)
    write_json(
        args.output_dir / "causal_site_sweep_metadata.json",
        {
            "model": args.model,
            "data_dir": str(args.data_dir),
            "direction_path": str(args.direction_path),
            "metadata_path": str(args.metadata_path),
            "split": args.split,
            "labels": list(args.labels),
            "num_prompts": args.num_prompts,
            "prompt_offset": args.prompt_offset,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "layers": layers,
            "positions": positions,
            "refusal_token_ids": refusal_ids,
            "template_positions": _template_position_metadata(model.tokenizer, positions),
            "generated_text_saved": False,
            "raw_prompts_saved": False,
            "metric": "next-token refusal log-odds under local resid_pre direction ablation",
        },
    )
    print(f"Saved causal site sweep to {sweep_path}")
    print(f"Saved baselines to {baseline_path}")


if __name__ == "__main__":
    main()

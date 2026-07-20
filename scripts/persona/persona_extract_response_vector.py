#!/usr/bin/env python3
"""Filter canonical response pairs and extract all-layer Persona Vectors."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _mean_activations(model, tokenizer, row: dict, device: str):
    """Match official prompt/response slicing for one generated transcript."""
    import torch

    prompt = row["prompt"]
    answer = row["answer"]
    if not isinstance(prompt, str) or not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"empty prompt/answer in {row['record_id']}")
    full_text = prompt + answer
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    total_len = int(inputs.input_ids.shape[1])
    if not 0 < prompt_len < total_len:
        raise ValueError(
            f"invalid token boundary for {row['record_id']}: prompt={prompt_len}, total={total_len}"
        )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    prompt_avg = torch.stack(
        [hidden[0, :prompt_len].float().mean(dim=0).cpu() for hidden in outputs.hidden_states]
    )
    prompt_last = torch.stack(
        [hidden[0, prompt_len - 1].float().cpu() for hidden in outputs.hidden_states]
    )
    response_avg = torch.stack(
        [hidden[0, prompt_len:total_len].float().mean(dim=0).cpu() for hidden in outputs.hidden_states]
    )
    return prompt_avg, prompt_last, response_avg, prompt_len, total_len - prompt_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--positive", type=Path, default=None)
    parser.add_argument("--negative", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import (
        align_effective_pairs,
        hf_hidden_state_to_tl_hook,
        read_jsonl,
        sha256_file,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Canonical vector extraction requires a CUDA job.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    extraction = config["extraction"]
    model_cfg = config["model"]
    vector_cfg = config["vector"]
    intermediate = _resolve(config["paths"]["intermediate_root"])
    positive_path = _resolve(args.positive) if args.positive else intermediate / "judged/pos_scored.jsonl"
    negative_path = _resolve(args.negative) if args.negative else intermediate / "judged/neg_scored.jsonl"
    if not positive_path.exists() or not negative_path.exists():
        raise FileNotFoundError(
            "Canonical judged positive/negative files are missing. Run persona.judge with "
            "the official OpenAI judge before extracting the vector."
        )
    positive_rows = read_jsonl(positive_path)
    negative_rows = read_jsonl(negative_path)
    expected = (
        int(extraction["questions"])
        * int(extraction["instructions_per_polarity"])
        * int(extraction["rollouts_per_question_instruction"])
    )
    if len(positive_rows) != expected or len(negative_rows) != expected:
        raise ValueError(
            f"canonical extraction expects {expected} rows per polarity, got "
            f"{len(positive_rows)} and {len(negative_rows)}"
        )
    if any(row.get("judge_protocol") != "top20_expected_0_100" for row in positive_rows + negative_rows):
        raise ValueError("noncanonical or missing judge protocol in extraction inputs")
    pairs = align_effective_pairs(
        positive_rows,
        negative_rows,
        trait_threshold=float(extraction["trait_threshold"]),
        coherence_threshold=float(extraction["coherence_threshold"]),
    )
    if not pairs:
        raise RuntimeError("no response pairs passed the official joint filter")

    output_root = _resolve(config["paths"]["results_root"]) / "vector"
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_root} is nonempty; pass --overwrite to replace artifacts")
    output_root.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg.get("revision"), token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        token=token,
        torch_dtype=getattr(torch, model_cfg["dtype"]),
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    expected_hidden_states = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    sums = {
        polarity: {
            variant: torch.zeros((expected_hidden_states, hidden_size), dtype=torch.float64)
            for variant in ("prompt_avg", "prompt_last", "response_avg")
        }
        for polarity in ("pos", "neg")
    }
    counts = {"pos": 0, "neg": 0}
    token_rows = []
    started = time.time()
    for positive, negative in tqdm(pairs, desc="extract response activations"):
        for polarity, row in (("pos", positive), ("neg", negative)):
            prompt_avg, prompt_last, response_avg, prompt_tokens, response_tokens = _mean_activations(
                model, tokenizer, row, args.device
            )
            if prompt_avg.shape != (expected_hidden_states, hidden_size):
                raise RuntimeError(f"unexpected hidden-state shape {tuple(prompt_avg.shape)}")
            sums[polarity]["prompt_avg"] += prompt_avg.double()
            sums[polarity]["prompt_last"] += prompt_last.double()
            sums[polarity]["response_avg"] += response_avg.double()
            counts[polarity] += 1
            token_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "polarity": polarity,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                }
            )
    if counts["pos"] != len(pairs) or counts["neg"] != len(pairs):
        raise RuntimeError(f"activation count mismatch: {counts}, pairs={len(pairs)}")

    means = {
        polarity: {
            variant: (tensor / counts[polarity]).float()
            for variant, tensor in variants.items()
        }
        for polarity, variants in sums.items()
    }
    vectors = {
        variant: means["pos"][variant] - means["neg"][variant]
        for variant in ("prompt_avg", "prompt_last", "response_avg")
    }
    for variant, vector in vectors.items():
        if vector.shape != (expected_hidden_states, hidden_size) or not torch.isfinite(vector).all():
            raise RuntimeError(f"invalid {variant} vector")
        torch.save(vector, output_root / f"sycophantic_{variant}_diff.pt")
        torch.save(means["pos"][variant], output_root / f"sycophantic_{variant}_pos_mean.pt")
        torch.save(means["neg"][variant], output_root / f"sycophantic_{variant}_neg_mean.pt")

    selected_layer = int(vector_cfg["paper_layer"])
    if selected_layer != int(vector_cfg["hf_hidden_state_index"]):
        raise ValueError("paper layer and HF hidden-state index must agree for this protocol")
    target_hook = hf_hidden_state_to_tl_hook(selected_layer)
    if target_hook != vector_cfg["transformer_lens_target_hook"]:
        raise ValueError(f"frozen target hook mismatch: {target_hook}")
    raw = vectors["response_avg"][selected_layer]
    raw_norm = float(raw.norm().item())
    if not raw_norm > 0:
        raise RuntimeError("selected response-average vector has zero norm")
    unit = raw / raw_norm
    torch.save(raw, output_root / "sycophantic_direction_raw.pt")
    torch.save(unit, output_root / "sycophantic_direction.pt")

    accepted_rows = []
    for positive, negative in pairs:
        accepted_rows.append(
            {
                "pair_id": positive["pair_id"],
                "positive_trait_score": positive["trait_score"],
                "negative_trait_score": negative["trait_score"],
                "positive_coherence_score": positive["coherence_score"],
                "negative_coherence_score": negative["coherence_score"],
                "question_index": positive["question_index"],
                "instruction_index": positive["instruction_index"],
                "rollout_index": positive["rollout_index"],
            }
        )
    pd.DataFrame(accepted_rows).to_csv(output_root / "effective_pairs.csv", index=False)
    pd.DataFrame(token_rows).to_csv(output_root / "token_counts.csv", index=False)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "canonical_protocol": True,
        "trait": config["trait"]["name"],
        "model": model_cfg["name"],
        "model_revision": model_cfg.get("revision"),
        "method": "mean(response-token hidden states | positive) - mean(response-token hidden states | negative)",
        "filter": {
            "positive_trait": f">={extraction['trait_threshold']}",
            "negative_trait": f"<{100.0 - float(extraction['trait_threshold'])}",
            "positive_coherence": f">={extraction['coherence_threshold']}",
            "negative_coherence": f">={extraction['coherence_threshold']}",
            "joint_pair_filter": True,
        },
        "input_rows_per_polarity": expected,
        "effective_pairs": len(pairs),
        "effective_pair_rate": len(pairs) / expected,
        "positive_input": str(positive_path.relative_to(ROOT)),
        "positive_sha256": sha256_file(positive_path),
        "negative_input": str(negative_path.relative_to(ROOT)),
        "negative_sha256": sha256_file(negative_path),
        "hidden_states": expected_hidden_states,
        "hidden_size": hidden_size,
        "paper_layer_one_indexed": selected_layer,
        "hf_hidden_state_index": selected_layer,
        "transformer_lens_target_hook": target_hook,
        "transformer_lens_steering_block_zero_indexed": int(
            vector_cfg["transformer_lens_steering_block"]
        ),
        "target_position": int(vector_cfg["target_position"]),
        "raw_direction_norm": raw_norm,
        "primary_vector_file": "sycophantic_direction.pt",
        "raw_vector_file": "sycophantic_direction_raw.pt",
        "all_layer_response_vector_file": "sycophantic_response_avg_diff.pt",
        "wall_time_seconds": time.time() - started,
    }
    write_json(output_root / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

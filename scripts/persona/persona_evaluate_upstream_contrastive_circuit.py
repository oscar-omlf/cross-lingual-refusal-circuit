#!/usr/bin/env python3
"""Evaluate an upstream contrastive persona circuit on held-out prompt pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _bootstrap import ROOT
from persona_run_upstream_contrastive_eapig import (
    _align_negative_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=int, default=3000)
    parser.add_argument("--circuit-path", type=Path, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--num-pairs", type=int, default=32)
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        required=True,
        help=(
            "Frozen prompt-disjoint manifest produced by "
            "persona.freeze-upstream-evaluation."
        ),
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--allow-disconnected-circuit-only",
        action="store_true",
        help="Permit legacy cut-set sufficiency diagnostics.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    import pandas as pd
    import torch

    from refusal_circuit.circuit import (
        _normal_forward,
        mixed_forward_from_prompt_pair,
        normalized_faithfulness,
        normalized_necessity,
    )
    from refusal_circuit.graph import (
        circuit_connectivity,
        load_circuit_edge_names,
        make_projection_graph,
        set_selected_edges,
    )
    from refusal_circuit.model import (
        load_model,
        tokenize_prompts,
        validate_model_architecture,
    )
    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import make_system_prompt, sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    trait_slug = config["trait"]["name"]
    result_root = ROOT / config["paths"]["results_root"]
    intermediate = ROOT / config["paths"]["intermediate_root"]
    official_root = ROOT / config["paths"]["official_root"]
    discovery = (
        result_root / "circuit_discovery_upstream_contrastive" / args.run_name
    )
    metadata_path = discovery / "eapig_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or not metadata.get(
        "no_projection_ablation"
    ):
        raise ValueError("Discovery is not a completed non-projection run")
    if metadata.get("model") != config["model"]:
        raise ValueError("Discovery and evaluation model configurations differ.")

    trait_data_path = official_root / "extract.json"
    trait_data = json.loads(trait_data_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    model = load_model(
        model_cfg["name"],
        device=args.device,
        dtype=args.dtype,
        revision=model_cfg.get("revision"),
    )
    observed_architecture = validate_model_architecture(
        model, config["model_architecture"]
    )
    evaluation_manifest_path = (
        args.evaluation_manifest
        if args.evaluation_manifest.is_absolute()
        else ROOT / args.evaluation_manifest
    )
    evaluation_manifest = json.loads(
        evaluation_manifest_path.read_text(encoding="utf-8")
    )
    if (
        evaluation_manifest.get("status") != "frozen"
        or evaluation_manifest.get("protocol")
        != "prompt_disjoint_upstream_evaluation_v1"
        or evaluation_manifest.get("model") != model_cfg
        or evaluation_manifest.get("trait") != trait_slug
        or evaluation_manifest["sources"]["extract_sha256"]
        != sha256_file(trait_data_path)
    ):
        raise ValueError("Upstream evaluation manifest failed provenance.")
    discovery_manifest_path = ROOT / evaluation_manifest[
        "discovery_manifest"
    ]
    if (
        sha256_file(discovery_manifest_path)
        != evaluation_manifest["discovery_manifest_sha256"]
        or sha256_file(discovery_manifest_path)
        != metadata.get("sample_manifest_sha256")
    ):
        raise ValueError(
            "Upstream evaluation was not frozen against this discovery split."
        )
    discovery_manifest = json.loads(
        discovery_manifest_path.read_text(encoding="utf-8")
    )
    discovery_identities = {
        (int(row["question_index"]), int(row["instruction_index"]))
        for row in discovery_manifest["discovery"]["samples"]
    }
    selected = evaluation_manifest["selection"]["samples"]
    if len(selected) != args.num_pairs:
        raise ValueError(
            f"Manifest has {len(selected)} held-out pairs, "
            f"but --num-pairs={args.num_pairs}."
        )
    selected_identities = {
        (int(row["question_index"]), int(row["instruction_index"]))
        for row in selected
    }
    if (
        len(selected_identities) != len(selected)
        or selected_identities & discovery_identities
    ):
        raise ValueError(
            "Upstream evaluation identities are duplicated or overlap discovery."
        )
    for row in selected:
        question_index = int(row["question_index"])
        instruction_index = int(row["instruction_index"])
        question = trait_data["questions"][question_index]
        instruction = trait_data["instruction"][instruction_index]
        positive_system = make_system_prompt(
            config["trait"]["positive_assistant_name"], instruction["pos"]
        )
        positive_prompt = model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": positive_system},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        aligned, alignment = _align_negative_prompt(
            model.tokenizer,
            question=question,
            negative_assistant_name=config["trait"][
                "negative_assistant_name"
            ],
            negative_instruction=instruction["neg"],
            positive_prompt=positive_prompt,
        )
        if (
            row["positive_prompt"] != positive_prompt
            or row["aligned_negative_prompt"] != aligned
            or row["alignment"] != alignment
        ):
            raise ValueError(
                f"Frozen upstream prompt changed for {(question_index, instruction_index)}."
            )
    overlength = [
        (
            row["question_index"],
            row["instruction_index"],
            int(row["alignment"]["positive_tokens"]),
        )
        for row in selected
        if int(row["alignment"]["positive_tokens"]) > args.max_length
    ]
    if overlength:
        raise ValueError(
            "Held-out upstream prompts exceed --max-length; refusing "
            f"truncation: {overlength[:3]}"
        )

    # Use the trait-specific vector artifact so held-out evaluation stays aligned.
    direction_path = result_root / f"vector/{trait_slug}_direction.pt"
    direction = torch.load(
        direction_path,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )
    target_layer = int(config["vector"]["hf_hidden_state_index"])
    target_hook = config["vector"]["transformer_lens_target_hook"]
    circuit_path = (
        args.circuit_path
        if args.circuit_path is not None
        else discovery
        / "circuits_source_to_sink"
        / f"eapig_top_{args.k}.json"
    )
    circuit_path = circuit_path if circuit_path.is_absolute() else ROOT / circuit_path
    circuit_payload = json.loads(circuit_path.read_text(encoding="utf-8"))
    if int(circuit_payload["requested_k"]) != args.k:
        raise ValueError("Circuit payload k differs from requested k.")
    if circuit_payload.get("score_source") != metadata.get("score_file"):
        raise ValueError("Circuit was not derived from the frozen discovery scores.")
    graph = make_projection_graph(model, target_layer)
    edge_names = load_circuit_edge_names(circuit_path)
    connectivity = circuit_connectivity(graph, edge_names)
    if (
        not connectivity["all_edges_on_source_to_sink_paths"]
        and not args.allow_disconnected_circuit_only
    ):
        raise ValueError(
            "Circuit-only evaluation requires a source-to-sink circuit. "
            "Rebuild with --families source_to_sink, or pass "
            "--allow-disconnected-circuit-only only for a legacy diagnostic."
        )
    set_selected_edges(graph, edge_names)

    values = {
        name: []
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    instances = []
    for row in selected:
        clean_tokens, clean_mask = tokenize_prompts(
            model, [row["positive_prompt"]], max_length=args.max_length
        )
        corrupt_tokens, corrupt_mask = tokenize_prompts(
            model, [row["aligned_negative_prompt"]], max_length=args.max_length
        )
        results = {
            "clean": _normal_forward(
                model,
                clean_tokens,
                clean_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
            )[0],
            "corrupt": _normal_forward(
                model,
                corrupt_tokens,
                corrupt_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
            )[0],
            "circuit_only": mixed_forward_from_prompt_pair(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_tokens,
                corrupt_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
                patch_selected=False,
            )[0],
            "circuit_ablated": mixed_forward_from_prompt_pair(
                model,
                graph,
                clean_tokens,
                clean_mask,
                corrupt_tokens,
                corrupt_mask,
                target_hook=target_hook,
                target_position=-1,
                direction=direction,
                patch_selected=True,
            )[0],
        }
        record = {
            "question_index": row["question_index"],
            "instruction_index": row["instruction_index"],
        }
        for name, tensor in results.items():
            value = float(tensor.item())
            values[name].append(value)
            record[name] = value
        instances.append(record)

    tensors = {name: torch.tensor(items) for name, items in values.items()}
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Held-out {name} metric contains non-finite values."
            )
    clean_corrupt_gap = float(
        tensors["clean"].mean().item() - tensors["corrupt"].mean().item()
    )
    if not math.isfinite(clean_corrupt_gap) or clean_corrupt_gap <= 1e-8:
        raise RuntimeError(
            "Held-out clean-minus-corrupt projection margin is not finite and "
            "positive; normalized circuit metrics are not behavior-aligned."
        )
    output_name = (
        args.output_tag
        or f"k{args.k}_n{args.num_pairs}_prompt_disjoint"
    )
    output = discovery / "evaluation" / output_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(instances).to_csv(output / "instance_metrics.csv", index=False)
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "method": "contrastive-state-path EAP-IG",
        "no_projection_ablation": True,
        "claim_scope": "held-out upstream positive-versus-negative persona mediation",
        "run_name": args.run_name,
        "model": model_cfg,
        "model_architecture": observed_architecture,
        "requested_k": args.k,
        "retained_edges": len(edge_names),
        "connectivity": connectivity,
        "n_pairs": len(selected),
        "evaluation_manifest": str(
            evaluation_manifest_path.relative_to(ROOT)
        ),
        "evaluation_manifest_sha256": sha256_file(
            evaluation_manifest_path
        ),
        "prompt_pairs": [
            {
                "question_index": row["question_index"],
                "instruction_index": row["instruction_index"],
            }
            for row in selected
        ],
        "source": str(trait_data_path.relative_to(ROOT)),
        "source_sha256": sha256_file(trait_data_path),
        "discovery_metadata": str(metadata_path.relative_to(ROOT)),
        "discovery_metadata_sha256": sha256_file(metadata_path),
        "circuit": str(circuit_path.relative_to(ROOT)),
        "circuit_sha256": sha256_file(circuit_path),
        "metric": (
            "final-prompt signed projection onto unit "
            f"{trait_slug} direction"
        ),
        "corrupt_baseline": "matched token-aligned negative-persona prompt state",
        "means": {
            name: float(tensor.mean().item()) for name, tensor in tensors.items()
        },
        "faithfulness": normalized_faithfulness(
            tensors["clean"], tensors["corrupt"], tensors["circuit_only"]
        ),
        "necessity": normalized_necessity(
            tensors["clean"], tensors["corrupt"], tensors["circuit_ablated"]
        ),
        "max_length": args.max_length,
        "max_observed_tokens": max(
            int(row["alignment"]["positive_tokens"]) for row in selected
        ),
        "truncation": False,
        "behavioral_validation_required": True,
    }
    write_json(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

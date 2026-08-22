"""Thin config-driven wrappers around the existing experiment functions."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import torch

from .paths import REPO_ROOT, write_json


def load_config(path: str | Path) -> dict:
    path = Path(path)
    path = path if path.is_absolute() else REPO_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _rows(path: str | Path) -> list[dict]:
    path = _path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL rows.")
    return rows


def _model(config: dict, *, hook_mode: str | None = None):
    from .model import load_model

    model = config["model"]
    return load_model(
        model["name"],
        revision=model.get("revision"),
        device=model.get("device"),
        dtype=model.get("dtype", "bfloat16"),
        hook_mode=hook_mode or model.get("hook_mode", "eap"),
    )


def _select_rows(rows: list[dict], spec: dict) -> list[dict]:
    label = spec.get("label")
    if label is not None:
        field = spec.get("label_field", "label")
        rows = [row for row in rows if row.get(field) == label]
    limit = spec.get("num_examples")
    rows = rows if limit is None else rows[: int(limit)]
    if not rows:
        raise ValueError("Configured dataset selection is empty.")
    return rows


def _prompts(model, spec: dict) -> list[str]:
    from .model import format_instructions

    rows = _select_rows(_rows(spec["path"]), spec)
    field = spec.get("text_field", "instruction")
    texts = [row[field] for row in rows]
    return texts if spec.get("formatted", False) else format_instructions(model.tokenizer, texts)


def _contrast_prompts(model, spec: dict) -> tuple[list[str], list[str]]:
    from .model import format_instructions

    field = spec.get("text_field", "instruction")
    if "positive_path" in spec:
        positive = [row[field] for row in _rows(spec["positive_path"])]
        negative = [row[field] for row in _rows(spec["negative_path"])]
    else:
        rows = _rows(spec["path"])
        label = spec.get("label_field", "label")
        positive = [row[field] for row in rows if row[label] == spec["positive_label"]]
        negative = [row[field] for row in rows if row[label] == spec["negative_label"]]
    if not positive or not negative:
        raise ValueError("Both positive and negative vector datasets must be non-empty.")
    if spec.get("formatted", False):
        return positive, negative
    return (
        format_instructions(model.tokenizer, positive),
        format_instructions(model.tokenizer, negative),
    )


def _paired_system_prompts(model, spec: dict) -> tuple[list[dict], list[str], list[str]]:
    """Format aligned positive/negative system conditions for shared prompts."""
    from .model import format_system_instruction
    from .persona import make_system_prompt

    rows = _select_rows(_rows(spec["path"]), spec)
    prompt_field = spec.get("prompt_field", "prompt")
    positive_field = spec.get(
        "positive_system_field", "positive_system_instruction"
    )
    negative_field = spec.get(
        "negative_system_field", "negative_system_instruction"
    )
    assistant_name = spec.get("assistant_name")
    required = (prompt_field, positive_field, negative_field)
    for index, row in enumerate(rows):
        missing = [field for field in required if not row.get(field)]
        if missing:
            raise ValueError(
                f"Paired prompt row {index} is missing non-empty fields: {missing}"
            )
    positive = [
        format_system_instruction(
            model.tokenizer,
            (
                make_system_prompt(assistant_name, row[positive_field])
                if assistant_name
                else row[positive_field]
            ),
            row[prompt_field],
        )
        for row in rows
    ]
    negative = [
        format_system_instruction(
            model.tokenizer,
            (
                make_system_prompt(assistant_name, row[negative_field])
                if assistant_name
                else row[negative_field]
            ),
            row[prompt_field],
        )
        for row in rows
    ]
    return rows, positive, negative


def _transcripts(model, spec: dict, max_length: int | None = None) -> list[tuple[str, int]]:
    rows = _select_rows(_rows(spec["path"]), spec)
    prompt_field = spec.get("prompt_field", "prompt")
    response_field = spec.get("response_field", "answer")
    max_length = spec.get("max_length", max_length)
    output = []
    for row in rows:
        prompt = row[prompt_field]
        response = row[response_field]
        if not spec.get("formatted", True):
            from .model import format_instruction

            prompt = format_instruction(model.tokenizer, prompt)
        start = len(model.tokenizer.encode(prompt, add_special_tokens=False))
        text = prompt + response
        length = len(model.tokenizer.encode(text, add_special_tokens=False))
        if max_length is not None and length > int(max_length):
            raise ValueError("Response objectives cannot truncate transcripts.")
        output.append((text, start))
    return output


def _vector_paths(config: dict) -> tuple[Path, Path, Path]:
    root = _path(config["vector"]["output_dir"])
    return root / "direction.pt", root / "raw_direction.pt", root / "metadata.json"


def extract_vector(config: dict, *, overwrite: bool = False) -> dict:
    """Extract a configured prompt- or response-based trait vector."""
    vector = config["vector"]
    if vector["type"] == "paired_prompt":
        from .direction import mean_residuals

        output_dir = _path(vector["output_dir"])
        paths = {
            "directions": output_dir / "directions.pt",
            "raw_directions": output_dir / "raw_directions.pt",
            "positive_means": output_dir / "positive_means.pt",
            "negative_means": output_dir / "negative_means.pt",
            "metadata": output_dir / "metadata.json",
        }
        existing = [path for path in paths.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"{existing[0]} already exists; pass --overwrite."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        model = _model(config)
        rows, positive_prompts, negative_prompts = _paired_system_prompts(
            model, vector["train"]
        )
        layers = [
            int(layer)
            for layer in (
                vector.get("layers")
                or list(range(int(model.cfg.n_layers)))
            )
        ]
        if not layers or len(layers) != len(set(layers)):
            raise ValueError("paired_prompt layers must be non-empty and unique.")
        if any(layer < 0 or layer >= int(model.cfg.n_layers) for layer in layers):
            raise ValueError("paired_prompt layer is outside the model.")
        position = int(vector.get("position", -1))
        common = {
            "layers": layers,
            "positions": [position],
            "batch_size": int(vector.get("batch_size", 4)),
            "max_length": vector.get("max_length", 512),
        }
        positive = mean_residuals(model, positive_prompts, **common)
        negative = mean_residuals(model, negative_prompts, **common)
        positive_means = torch.stack(
            [positive[(layer, position)] for layer in layers]
        ).float()
        negative_means = torch.stack(
            [negative[(layer, position)] for layer in layers]
        ).float()
        raw_directions = positive_means - negative_means
        norms = raw_directions.norm(dim=-1, keepdim=True)
        if not torch.isfinite(raw_directions).all():
            raise RuntimeError(
                "Extracted paired prompt directions contain non-finite values."
            )
        valid = norms.squeeze(-1) > 0
        if not valid.any():
            raise RuntimeError("All extracted paired prompt directions are zero.")
        directions = torch.zeros_like(raw_directions)
        directions[valid] = raw_directions[valid] / norms[valid]
        valid_layers = [
            layer for layer, is_valid in zip(layers, valid.tolist()) if is_valid
        ]
        zero_norm_layers = [
            layer for layer, is_valid in zip(layers, valid.tolist()) if not is_valid
        ]

        torch.save(directions, paths["directions"])
        torch.save(raw_directions, paths["raw_directions"])
        torch.save(positive_means, paths["positive_means"])
        torch.save(negative_means, paths["negative_means"])
        pair_field = vector["train"].get("instruction_pair_field")
        pair_counts = (
            dict(sorted(Counter(str(row[pair_field]) for row in rows).items()))
            if pair_field
            else None
        )
        metadata = {
            "model": config["model"],
            "trait": config["trait"],
            "vector_type": "paired_prompt",
            "method": (
                "mean(final-prompt-token residual | positive system instruction) "
                "- mean(final-prompt-token residual | negative system instruction)"
            ),
            "source_path": str(_path(vector["train"]["path"])),
            "base_prompts": len(rows),
            "formatted_prompts_per_polarity": len(rows),
            "layers": layers,
            "valid_layers": valid_layers,
            "zero_norm_layers": zero_norm_layers,
            "direction_norms": norms.squeeze(-1).tolist(),
            "position": position,
            "target_hooks": [
                f"blocks.{layer}.hook_resid_pre" for layer in layers
            ],
            "prompt_field": vector["train"].get("prompt_field", "prompt"),
            "positive_system_field": vector["train"].get(
                "positive_system_field", "positive_system_instruction"
            ),
            "negative_system_field": vector["train"].get(
                "negative_system_field", "negative_system_instruction"
            ),
            "assistant_name": vector["train"].get("assistant_name"),
            "instruction_pair_counts": pair_counts,
            "artifacts": {
                key: path.name
                for key, path in paths.items()
                if key != "metadata"
            },
        }
        write_json(paths["metadata"], metadata)
        return metadata

    direction_path, raw_path, metadata_path = _vector_paths(config)
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists; pass --overwrite.")
    direction_path.parent.mkdir(parents=True, exist_ok=True)

    if vector["type"] == "pre_answer":
        from .direction import (
            difference_in_means_candidates,
            evaluate_candidate,
            select_paper_candidate,
        )
        from .model import post_instruction_positions

        model = _model(config)
        positive, negative = _contrast_prompts(model, vector["train"])
        selection = vector.get("selection", "fixed")
        layers = (
            [int(vector["layer"])]
            if selection == "fixed"
            else vector.get("layers") or list(range(1, model.cfg.n_layers))
        )
        positions = (
            [int(vector["position"])]
            if selection == "fixed"
            else vector.get("positions") or list(post_instruction_positions(model.tokenizer))
        )
        candidates = difference_in_means_candidates(
            model,
            positive,
            negative,
            layers=layers,
            positions=positions,
            batch_size=int(vector.get("batch_size", 4)),
            max_length=vector.get("max_length", 512),
        )
        validation_rows = []
        if selection == "refusal_causal":
            validation = dict(vector["train"])
            validation.update(vector["validation"])
            positive_val, negative_val = _contrast_prompts(model, validation)
            validation_rows = [
                evaluate_candidate(
                    model,
                    candidate,
                    harmful_prompts=positive_val,
                    harmless_prompts=negative_val,
                    refusal_token_ids=list(vector["refusal_token_ids"]),
                    addition_coefficient=float(vector.get("addition_coefficient", 1.0)),
                    batch_size=int(vector.get("batch_size", 4)),
                    max_length=vector.get("max_length", 512),
                )
                for candidate in candidates
            ]
            selected_row = select_paper_candidate(
                validation_rows,
                n_layers=model.cfg.n_layers,
                kl_threshold=float(vector.get("kl_threshold", 0.1)),
                late_layer_fraction=float(vector.get("late_layer_fraction", 0.8)),
            )
            selected = next(
                candidate
                for candidate in candidates
                if candidate.layer == selected_row["layer"]
                and candidate.position == selected_row["position"]
            )
        else:
            layer = int(vector["layer"])
            position = int(vector["position"])
            selected = next(
                candidate
                for candidate in candidates
                if candidate.layer == layer and candidate.position == position
            )
        raw = selected.raw_vector.float()
        metadata = {
            "model": config["model"],
            "trait": config["trait"],
            "vector_type": "pre_answer",
            "layer": selected.layer,
            "position": selected.position,
            "target_hook": f"blocks.{selected.layer}.hook_resid_pre",
            "validation": validation_rows,
        }
    elif vector["type"] == "response_average":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        positive_rows = _rows(vector["positive_path"])
        negative_rows = _rows(vector["negative_path"])
        pair_field = vector.get("pair_id_field", "pair_id")
        negative = {row[pair_field]: row for row in negative_rows}
        pairs = [(row, negative[row[pair_field]]) for row in positive_rows if row[pair_field] in negative]
        threshold = vector.get("trait_threshold")
        coherence = vector.get("coherence_threshold")
        if threshold is not None:
            pairs = [
                (positive, negative)
                for positive, negative in pairs
                if float(positive["trait_score"]) >= float(threshold)
                and float(negative["trait_score"]) < 100.0 - float(threshold)
                and (
                    coherence is None
                    or (
                        float(positive["coherence_score"]) >= float(coherence)
                        and float(negative["coherence_score"]) >= float(coherence)
                    )
                )
            ]
        if not pairs:
            raise ValueError("No aligned positive/negative response pairs.")
        model_cfg = config["model"]
        token = os.environ.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"], revision=model_cfg.get("revision"), token=token
        )
        kwargs = {"revision": model_cfg.get("revision"), "token": token}
        if vector.get("activation_dtype"):
            kwargs["torch_dtype"] = getattr(torch, vector["activation_dtype"])
        device = model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **kwargs).to(device).eval()
        layer = int(vector["layer"])
        if not 0 < layer < int(model.config.num_hidden_layers):
            raise ValueError("response_average layer must map to an internal resid_pre.")
        prompt_field = vector.get("prompt_field", "prompt")
        response_field = vector.get("response_field", "answer")
        sums = {"positive": None, "negative": None}
        for positive_row, negative_row in pairs:
            for polarity, row in (("positive", positive_row), ("negative", negative_row)):
                prompt = row[prompt_field]
                response = row[response_field]
                inputs = tokenizer(
                    prompt + response, return_tensors="pt", add_special_tokens=False
                )
                prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
                inputs = {name: value.to(device) for name, value in inputs.items()}
                with torch.inference_mode():
                    hidden = model(
                        **inputs, output_hidden_states=True, use_cache=False, return_dict=True
                    ).hidden_states[layer]
                mean = hidden[0, prompt_length:].float().mean(dim=0).cpu()
                sums[polarity] = mean if sums[polarity] is None else sums[polarity] + mean
        positive_mean = sums["positive"] / len(pairs)
        negative_mean = sums["negative"] / len(pairs)
        raw = positive_mean - negative_mean
        torch.save(positive_mean, direction_path.parent / "positive_mean.pt")
        torch.save(negative_mean, direction_path.parent / "negative_mean.pt")
        metadata = {
            "model": config["model"],
            "trait": config["trait"],
            "vector_type": "response_average",
            "layer": layer,
            "position": None,
            "target_hook": f"blocks.{layer}.hook_resid_pre",
            "pairs": len(pairs),
        }
    else:
        raise ValueError(
            "vector.type must be paired_prompt, pre_answer, or response_average."
        )

    norm = raw.norm()
    if not torch.isfinite(raw).all() or norm <= 0:
        raise RuntimeError("Extracted vector is non-finite or zero.")
    torch.save(raw / norm, direction_path)
    torch.save(raw, raw_path)
    write_json(metadata_path, metadata)
    return metadata


def _load_vector(config: dict, model):
    direction_path, raw_path, metadata_path = _vector_paths(config)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model"]["name"] != config["model"]["name"]:
        raise ValueError("Vector and run config use different models.")
    if metadata["model"].get("revision") != config["model"].get("revision"):
        raise ValueError("Vector and run config use different model revisions.")
    device = next(model.parameters()).device
    direction = torch.load(direction_path, map_location=device, weights_only=True)
    raw = torch.load(raw_path, map_location=device, weights_only=True)
    return direction, raw, metadata


def _write_circuits(graph, config: dict, metadata: dict) -> list[str]:
    from .graph import edge_record, paper_greedy_edges

    output = _path(config["eapig"]["output_dir"]) / "circuits"
    paths = []
    for k in config["eapig"].get("k_values", [100, 500, 1000]):
        selected = paper_greedy_edges(
            graph,
            min(int(k), len(graph.edges)),
            prune_parentless=bool(config["eapig"].get("prune_parentless", True)),
        )
        if not selected:
            raise RuntimeError(f"k={k} produced an empty circuit after pruning.")
        path = output / f"eapig_top_{k}.json"
        write_json(
            path,
            {
                "method": "eapig",
                "corruption": config["eapig"]["corruption"],
                "requested_k": int(k),
                "retained_edges": len(selected),
                "target": {
                    key: metadata[key]
                    for key in ("vector_type", "layer", "position", "target_hook")
                },
                "edges": [edge_record(edge) for edge in selected],
            },
        )
        paths.append(str(path))
    return paths


def run_eapig(config: dict, *, overwrite: bool = False) -> dict:
    """Run the existing projection or steering EAP-IG implementation."""
    from .eapig import (
        AttributionConfig,
        attribute,
        attribute_response_average,
        attribute_steering_response_logprob,
    )
    from .graph import (
        make_projection_graph,
        make_steering_to_logits_graph,
        write_edge_scores_parquet,
    )

    run = config["eapig"]
    output = _path(run["output_dir"])
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists; pass --overwrite.")
    model = _model(config, hook_mode="eap")
    direction, raw, vector = _load_vector(config, model)
    layer = int(vector["layer"])
    attribution = AttributionConfig(
        batch_size=int(run.get("batch_size", 1)),
        max_length=run.get("max_length", 512),
        ig_steps=int(run.get("ig_steps", 5)),
    )
    corruption = run["corruption"]
    diagnostics = None
    if corruption == "projection":
        graph = make_projection_graph(model, layer)
        if vector["vector_type"] == "pre_answer":
            attribute(
                model,
                graph,
                _prompts(model, run["dataset"]),
                direction=direction,
                target_hook=vector["target_hook"],
                target_position=int(vector["position"]),
                config=attribution,
                integrated_gradients=True,
            )
        else:
            attribute_response_average(
                model,
                graph,
                _transcripts(model, run["dataset"], run.get("max_length", 512)),
                direction=direction,
                target_hook=vector["target_hook"],
                config=attribution,
                integrated_gradients=True,
            )
    elif corruption == "steering":
        graph = make_steering_to_logits_graph(model, layer)
        coefficient = float(run["coefficient"])
        if coefficient >= 0:
            raise ValueError("Steering corruption coefficient must be negative.")
        diagnostics = attribute_steering_response_logprob(
            model,
            graph,
            _transcripts(model, run["dataset"], run.get("max_length", 512)),
            raw_direction=raw,
            target_hook=vector["target_hook"],
            corrupt_coefficient=coefficient,
            config=attribution,
        )
        if diagnostics["clean_minus_corrupt"] <= 0:
            raise RuntimeError("Negative steering did not reduce the response metric.")
    else:
        raise ValueError("eapig.corruption must be projection or steering.")
    output.mkdir(parents=True, exist_ok=True)
    scores = output / "edge_scores.parquet"
    write_edge_scores_parquet(graph, scores)
    result = {
        "model": config["model"],
        "trait": config["trait"],
        "vector_type": vector["vector_type"],
        "corruption": corruption,
        "edge_scores": str(scores),
        "circuits": _write_circuits(graph, config, vector),
        "diagnostics": diagnostics,
    }
    write_json(metadata_path, result)
    return result


def _metric_summary(values: dict[str, torch.Tensor]) -> dict:
    from .circuit import normalized_faithfulness, normalized_necessity

    clean = values["clean"]
    corrupt = values["corrupt"]
    gap = clean.mean() - corrupt.mean()
    if not torch.isfinite(gap) or gap <= 1e-8:
        raise RuntimeError("Clean and corrupt conditions do not define a usable contrast.")
    return {
        "clean": clean.mean().item(),
        "corrupt": corrupt.mean().item(),
        "circuit_only": values["circuit_only"].mean().item(),
        "circuit_ablated": values["circuit_ablated"].mean().item(),
        "faithfulness": normalized_faithfulness(clean, corrupt, values["circuit_only"]),
        "necessity": normalized_necessity(clean, corrupt, values["circuit_ablated"]),
    }


def run_tests(config: dict) -> dict:
    """Evaluate one saved circuit with its matching corruption and vector target."""
    from .graph import (
        load_circuit_edge_names,
        make_projection_graph,
        make_steering_to_logits_graph,
        set_selected_edges,
    )

    test = config["tests"]
    model = _model(config, hook_mode="eap")
    direction, raw, vector = _load_vector(config, model)
    layer = int(vector["layer"])
    if config["eapig"]["corruption"] == "projection":
        graph = make_projection_graph(model, layer)
    else:
        graph = make_steering_to_logits_graph(model, layer)
    circuit_path = _path(test["circuit_path"])
    circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
    if circuit.get("corruption") != config["eapig"]["corruption"]:
        raise ValueError("Circuit and test corruption types differ.")
    set_selected_edges(graph, load_circuit_edge_names(circuit_path))

    if config["eapig"]["corruption"] == "projection" and vector["vector_type"] == "pre_answer":
        from .circuit import behavioural_sanity_check, evaluate_conditions
        from .metrics import DEFAULT_ANSWER_STARTS, REFUSAL_PREFIXES
        from .model import first_token_ids

        prompts = _prompts(model, test["dataset"])
        conditions = evaluate_conditions(
            model,
            graph,
            prompts,
            direction=direction,
            target_hook=vector["target_hook"],
            target_position=int(vector["position"]),
            batch_size=int(test.get("batch_size", 1)),
            max_length=test.get("max_length", 512),
        )
        values = {
            name: conditions[name].projection
            for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
        }
        behavior = (
            behavioural_sanity_check(
                model,
                graph,
                prompts,
                direction=direction,
                target_hook=vector["target_hook"],
                target_position=int(vector["position"]),
                refusal_ids=first_token_ids(model.tokenizer, REFUSAL_PREFIXES),
                answer_ids=first_token_ids(model.tokenizer, DEFAULT_ANSWER_STARTS),
                max_prompts=len(prompts),
                max_new_tokens=int(test.get("max_new_tokens", 16)),
            )
            if test.get("behavior", False)
            else None
        )
    elif config["eapig"]["corruption"] == "projection":
        from .circuit import _mixed_forward, _normal_forward
        from .eapig import _response_mask, response_logprob_values
        from .interventions import all_layer_ablation_hooks
        from .model import tokenize_prompts

        values = {name: [] for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")}
        behavior_values = {name: [] for name in values}
        for text, start in _transcripts(
            model, test["dataset"], test.get("max_length", 512)
        ):
            tokens, mask = tokenize_prompts(model, [text], max_length=test.get("max_length", 512))
            response_mask = _response_mask(tokens, mask, [start])
            runs = {
                "clean": _normal_forward(
                    model, tokens, mask, target_hook=vector["target_hook"],
                    target_position=-1, direction=direction, target_mask=response_mask
                ),
                "corrupt": _normal_forward(
                    model, tokens, mask, target_hook=vector["target_hook"],
                    target_position=-1, direction=direction,
                    fwd_hooks=all_layer_ablation_hooks(model, direction),
                    target_mask=response_mask
                ),
                "circuit_only": _mixed_forward(
                    model, graph, tokens, mask, target_hook=vector["target_hook"],
                    target_position=-1, direction=direction, patch_selected=False,
                    target_mask=response_mask
                ),
                "circuit_ablated": _mixed_forward(
                    model, graph, tokens, mask, target_hook=vector["target_hook"],
                    target_position=-1, direction=direction, patch_selected=True,
                    target_mask=response_mask
                ),
            }
            for name, (projection, logits) in runs.items():
                values[name].append(projection.cpu())
                behavior_values[name].append(
                    response_logprob_values(logits, tokens, response_mask).cpu()
                )
        values = {name: torch.cat(items) for name, items in values.items()}
        behavior = _metric_summary(
            {name: torch.cat(items) for name, items in behavior_values.items()}
        )
    else:
        from .circuit import mixed_logits_with_intervention
        from .eapig import _response_mask, response_logprob_values, response_prediction_mask
        from .interventions import masked_activation_addition_hook
        from .model import tokenize_prompts

        coefficient = float(config["eapig"]["coefficient"])
        values = {name: [] for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")}
        for text, start in _transcripts(
            model, test["dataset"], test.get("max_length", 512)
        ):
            tokens, mask = tokenize_prompts(model, [text], max_length=test.get("max_length", 512))
            response_mask = _response_mask(tokens, mask, [start])
            prediction_mask = response_prediction_mask(response_mask)
            hook = (
                vector["target_hook"],
                masked_activation_addition_hook(raw, coefficient=coefficient, mask=prediction_mask),
            )

            def transform(name, activations):
                if name != vector["target_hook"]:
                    return activations
                return activations + prediction_mask.to(activations).unsqueeze(-1) * coefficient * raw.to(activations)

            with torch.inference_mode():
                clean = model(tokens, attention_mask=mask)
            with torch.inference_mode(), model.hooks(fwd_hooks=[hook]):
                corrupt = model(tokens, attention_mask=mask)
            logits = {
                "clean": clean,
                "corrupt": corrupt,
                "circuit_only": mixed_logits_with_intervention(
                    model, graph, tokens, mask, corrupt_fwd_hooks=[hook],
                    corrupt_cache_transform=transform, patch_selected=False
                ),
                "circuit_ablated": mixed_logits_with_intervention(
                    model, graph, tokens, mask, corrupt_fwd_hooks=[hook],
                    corrupt_cache_transform=transform, patch_selected=True
                ),
            }
            for name, output in logits.items():
                values[name].append(response_logprob_values(output, tokens, response_mask).cpu())
        values = {name: torch.cat(items) for name, items in values.items()}

    result = _metric_summary(values)
    if "behavior" in locals() and behavior is not None:
        result["behavior"] = behavior
    output = _path(test["output_path"])
    write_json(output, result)
    return result

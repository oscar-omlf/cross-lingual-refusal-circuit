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
    from .model import (
        install_memory_efficient_split_qkv_projection,
        load_model,
    )

    model = config["model"]
    effective_hook_mode = hook_mode or model.get("hook_mode", "eap")
    if effective_hook_mode == "eap":
        install_memory_efficient_split_qkv_projection()
    return load_model(
        model["name"],
        revision=model.get("revision"),
        device=model.get("device"),
        dtype=model.get("dtype", "bfloat16"),
        hook_mode=effective_hook_mode,
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
    system_field = spec.get("system_field")
    max_length = spec.get("max_length", max_length)
    output = []
    for row in rows:
        prompt = row[prompt_field]
        response = row[response_field]
        if system_field is not None:
            from .model import format_system_instruction
            from .persona import make_system_prompt

            system = row[system_field]
            assistant_name = spec.get("assistant_name")
            if assistant_name:
                system = make_system_prompt(assistant_name, system)
            prompt = format_system_instruction(
                model.tokenizer,
                system,
                prompt,
            )
        elif not spec.get("formatted", True):
            from .model import format_instruction

            prompt = format_instruction(model.tokenizer, prompt)
        start = len(model.tokenizer.encode(prompt, add_special_tokens=False))
        text = prompt + response
        length = len(model.tokenizer.encode(text, add_special_tokens=False))
        if max_length is not None and length > int(max_length):
            raise ValueError("Response objectives cannot truncate transcripts.")
        output.append((text, start))
    return output


def _paired_response_transcripts(
    model, spec: dict, max_length: int | None = None
) -> list[tuple[str, str, int]]:
    """Build token-aligned positive/negative prompts with shared responses."""
    rows, positive_prompts, negative_prompts = _paired_system_prompts(model, spec)
    response_fields = spec.get("response_fields")
    if response_fields is None:
        response_fields = [spec.get("response_field", "answer")]
    response_fields = [str(field) for field in response_fields]
    if not response_fields or len(response_fields) != len(set(response_fields)):
        raise ValueError("response_fields must be non-empty and unique.")
    max_length = spec.get("max_length", max_length)
    output = []
    for row, positive_prompt, negative_prompt in zip(
        rows, positive_prompts, negative_prompts, strict=True
    ):
        positive_ids = model.tokenizer.encode(
            positive_prompt, add_special_tokens=False
        )
        negative_ids = model.tokenizer.encode(
            negative_prompt, add_special_tokens=False
        )
        if len(positive_ids) != len(negative_ids):
            raise ValueError(
                f"Paired prompts are not token-aligned for {row.get('id', '<unknown>')}."
            )
        for response_field in response_fields:
            response = row[response_field]
            clean = positive_prompt + response
            corrupt = negative_prompt + response
            clean_ids = model.tokenizer.encode(clean, add_special_tokens=False)
            corrupt_ids = model.tokenizer.encode(corrupt, add_special_tokens=False)
            if len(clean_ids) != len(corrupt_ids):
                raise ValueError("Paired transcripts are not token-aligned.")
            if clean_ids[len(positive_ids) :] != corrupt_ids[len(negative_ids) :]:
                raise ValueError(
                    "Response tokens differ across paired prompt conditions."
                )
            if max_length is not None and len(clean_ids) > int(max_length):
                raise ValueError("Response objectives cannot truncate transcripts.")
            output.append((clean, corrupt, len(positive_ids)))
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
    vector_config = config["vector"]
    root = _path(vector_config["output_dir"])
    if vector_config["type"] == "paired_prompt":
        direction_path = root / "directions.pt"
        raw_path = root / "raw_directions.pt"
        metadata_path = root / "metadata.json"
    else:
        direction_path = root / vector_config.get("direction_file", "direction.pt")
        raw_path = root / vector_config.get("raw_direction_file", "raw_direction.pt")
        metadata_path = root / vector_config.get("metadata_file", "metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_model = metadata["model"]
    saved_model_name = (
        saved_model["name"] if isinstance(saved_model, dict) else saved_model
    )
    saved_revision = (
        saved_model.get("revision")
        if isinstance(saved_model, dict)
        else metadata.get("model_revision")
    )
    if saved_model_name != config["model"]["name"]:
        raise ValueError("Vector and run config use different models.")
    if saved_revision != config["model"].get("revision"):
        raise ValueError("Vector and run config use different model revisions.")
    device = next(model.parameters()).device
    direction = torch.load(direction_path, map_location=device, weights_only=True)
    raw = torch.load(raw_path, map_location=device, weights_only=True)
    if vector_config["type"] == "paired_prompt":
        if "layer" not in vector_config:
            raise ValueError("paired_prompt EAP-IG requires vector.layer.")
        layer = int(vector_config["layer"])
        layers = [int(value) for value in metadata["layers"]]
        if layer not in layers:
            raise ValueError(
                f"Selected paired_prompt layer {layer} is not in saved layers."
            )
        if direction.shape[0] != len(layers) or raw.shape[0] != len(layers):
            raise ValueError("Saved paired_prompt tensors do not match metadata layers.")
        position = int(vector_config.get("position", metadata["position"]))
        if position != int(metadata["position"]):
            raise ValueError(
                "Configured paired_prompt position differs from the extracted vectors."
            )
        index = layers.index(layer)
        direction = direction[index]
        raw = raw[index]
        metadata = {
            **metadata,
            "layer": layer,
            "position": position,
            "target_hook": f"blocks.{layer}.hook_resid_pre",
        }
    elif vector_config["type"] == "response_average":
        layer = int(
            vector_config.get(
                "layer",
                metadata.get("layer", metadata.get("hf_hidden_state_index")),
            )
        )
        metadata = {
            **metadata,
            "vector_type": "response_average",
            "layer": layer,
            "position": None,
            "target_hook": vector_config.get(
                "target_hook",
                metadata.get(
                    "transformer_lens_target_hook",
                    f"blocks.{layer}.hook_resid_pre",
                ),
            ),
        }
    return direction, raw, metadata


def _negative_centroid_coordinates(
    config: dict,
    model,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Load stored anti-trait means and project them onto one unit direction."""

    filename = config["eapig"].get("negative_mean_file")
    if not filename:
        raise ValueError(
            "negative_centroid_clamp requires eapig.negative_mean_file."
        )
    path = Path(filename)
    if not path.is_absolute():
        path = _path(config["vector"]["output_dir"]) / path
    means = torch.load(path, map_location="cpu", weights_only=True).float()
    unit = direction.detach().float().cpu()
    unit = unit / unit.norm().clamp_min(1e-8)
    if means.ndim != 2 or means.shape[1] != unit.numel():
        raise ValueError(
            "Stored negative means must have shape [layer, hidden_size]."
        )
    if means.shape[0] < model.cfg.n_layers:
        raise ValueError(
            "Stored negative means do not cover every model layer."
        )
    coordinates = means[: model.cfg.n_layers] @ unit
    if not torch.isfinite(coordinates).all():
        raise ValueError("Stored negative-centroid coordinates are non-finite.")
    return coordinates


def _write_circuits(
    graph,
    config: dict,
    metadata: dict,
    *,
    target: dict | None = None,
) -> list[str]:
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
                "target": target or {
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
        attribute_contrastive_prompt_projection,
        attribute_logit_metric,
        attribute_paired_coordinate_projection,
        attribute_response_average,
        attribute_steering_response_logprob,
    )
    from .graph import (
        make_logit_graph,
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
    circuit_target = None
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
        elif vector["vector_type"] == "paired_prompt":
            _, positive_prompts, _ = _paired_system_prompts(
                model, run["dataset"]
            )
            attribute(
                model,
                graph,
                positive_prompts,
                direction=direction,
                target_hook=vector["target_hook"],
                target_position=int(vector["position"]),
                config=attribution,
                integrated_gradients=True,
            )
        else:
            diagnostics = attribute_response_average(
                model,
                graph,
                _transcripts(model, run["dataset"], run.get("max_length", 512)),
                direction=direction,
                target_hook=vector["target_hook"],
                config=attribution,
                integrated_gradients=True,
            )
    elif corruption == "upstream_steering":
        if vector["vector_type"] != "response_average":
            raise ValueError(
                "upstream_steering currently requires a response_average vector."
            )
        coefficient = float(run["coefficient"])
        if coefficient >= 0:
            raise ValueError("Upstream steering corruption coefficient must be negative.")
        graph = make_projection_graph(model, layer)
        diagnostics = attribute_response_average(
            model,
            graph,
            _transcripts(model, run["dataset"], run.get("max_length", 512)),
            direction=direction,
            raw_direction=raw,
            target_hook=vector["target_hook"],
            config=attribution,
            integrated_gradients=True,
            corruption="upstream_steering",
            corrupt_coefficient=coefficient,
        )
    elif corruption == "negative_centroid_clamp":
        if vector["vector_type"] != "response_average":
            raise ValueError(
                "negative_centroid_clamp requires a response_average vector."
            )
        graph = make_projection_graph(model, layer)
        negative_coordinates = _negative_centroid_coordinates(
            config, model, direction
        )
        diagnostics = attribute_response_average(
            model,
            graph,
            _transcripts(model, run["dataset"], run.get("max_length", 512)),
            direction=direction,
            target_hook=vector["target_hook"],
            config=attribution,
            integrated_gradients=True,
            corruption="negative_centroid_clamp",
            negative_coordinates=negative_coordinates,
        )
        diagnostics["target_negative_coordinate"] = float(
            negative_coordinates[layer].item()
        )
        diagnostics["negative_mean_file"] = str(run["negative_mean_file"])
        if diagnostics["clean_minus_corrupt_projection"] <= 0:
            raise RuntimeError(
                "Negative-centroid clamping did not reduce the generation-vector "
                "projection."
            )
    elif corruption == "contrastive_prompts":
        if vector["vector_type"] not in {"paired_prompt", "response_average"}:
            raise ValueError(
                "contrastive_prompts requires a paired_prompt or "
                "response_average vector."
            )
        graph = make_projection_graph(model, layer)
        target_scope = run.get(
            "target_scope",
            (
                "single_position"
                if vector["vector_type"] == "paired_prompt"
                else "response_prefix"
            ),
        )
        if vector["vector_type"] == "paired_prompt":
            _, positive_prompts, negative_prompts = _paired_system_prompts(
                model, run["dataset"]
            )
            prompt_pairs = list(zip(positive_prompts, negative_prompts, strict=True))
            target_position = (
                int(vector["position"])
                if target_scope == "single_position"
                else None
            )
        else:
            prompt_pairs = _paired_response_transcripts(
                model, run["dataset"], run.get("max_length", 512)
            )
            target_position = None
        diagnostics = attribute_contrastive_prompt_projection(
            model,
            graph,
            prompt_pairs,
            direction=direction,
            target_hook=vector["target_hook"],
            target_position=target_position,
            target_scope=target_scope,
            config=attribution,
        )
        if diagnostics["clean_minus_corrupt_projection"] <= 0:
            raise RuntimeError(
                "Positive prompts did not increase the generation-vector projection."
            )
        circuit_target = {
            "vector_type": vector["vector_type"],
            "layer": layer,
            "position": target_position,
            "target_hook": vector["target_hook"],
            "target_scope": target_scope,
        }
    elif corruption == "paired_coordinate_projection":
        if vector["vector_type"] not in {"paired_prompt", "response_average"}:
            raise ValueError(
                "paired_coordinate_projection requires a paired_prompt or "
                "response_average vector."
            )
        graph = make_projection_graph(model, layer)
        if vector["vector_type"] == "paired_prompt":
            _, positive_prompts, negative_prompts = _paired_system_prompts(
                model, run["dataset"]
            )
            prompt_pairs = list(
                zip(positive_prompts, negative_prompts, strict=True)
            )
            target_scope = "single_position"
            target_position = int(vector["position"])
        else:
            prompt_pairs = _paired_response_transcripts(
                model, run["dataset"], run.get("max_length", 512)
            )
            target_scope = "response_prefix"
            target_position = None
        diagnostics = attribute_paired_coordinate_projection(
            model,
            graph,
            prompt_pairs,
            direction=direction,
            target_hook=vector["target_hook"],
            target_position=target_position,
            target_scope=target_scope,
            config=attribution,
        )
        if diagnostics["clean_minus_corrupt_projection"] <= 0:
            raise RuntimeError(
                "Paired-coordinate corruption did not reduce the vector target."
            )
        circuit_target = {
            "vector_type": vector["vector_type"],
            "layer": layer,
            "position": target_position,
            "target_hook": vector["target_hook"],
            "target_scope": target_scope,
            "source_corruption_scope": (
                "all_prompt_tokens"
                if vector["vector_type"] == "paired_prompt"
                else "response_prefix"
            ),
        }
    elif corruption == "target_projection_to_logits":
        if vector["vector_type"] != "pre_answer":
            raise ValueError(
                "target_projection_to_logits requires a pre_answer vector."
            )
        refusal_token_ids = [int(value) for value in run["refusal_token_ids"]]
        graph = make_steering_to_logits_graph(model, layer)
        diagnostics = attribute_logit_metric(
            model,
            graph,
            _prompts(model, run["dataset"]),
            direction=direction,
            refusal_token_ids=refusal_token_ids,
            config=attribution,
            integrated_gradients=True,
            target_hook=vector["target_hook"],
            target_position=int(vector["position"]),
        )
        if diagnostics["clean_minus_corrupt"] <= 0:
            raise RuntimeError(
                "Target-only projection did not reduce refusal log-odds."
            )
        circuit_target = {
            "vector_type": vector["vector_type"],
            "source_layer": layer,
            "source_position": int(vector["position"]),
            "source_hook": vector["target_hook"],
            "metric": "next_token_refusal_log_odds",
            "refusal_token_ids": refusal_token_ids,
        }
    elif corruption == "global_projection_to_logits":
        if vector["vector_type"] != "pre_answer":
            raise ValueError(
                "global_projection_to_logits requires a pre_answer vector."
            )
        refusal_token_ids = [int(value) for value in run["refusal_token_ids"]]
        graph = make_logit_graph(model)
        diagnostics = attribute_logit_metric(
            model,
            graph,
            _prompts(model, run["dataset"]),
            direction=direction,
            refusal_token_ids=refusal_token_ids,
            config=attribution,
            integrated_gradients=True,
        )
        if diagnostics["clean_minus_corrupt"] <= 0:
            raise RuntimeError(
                "Global projection did not reduce refusal log-odds."
            )
        circuit_target = {
            "vector_type": vector["vector_type"],
            "corruption_scope": "all_layers_all_prompt_positions",
            "metric": "next_token_refusal_log_odds",
            "refusal_token_ids": refusal_token_ids,
        }
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
        raise ValueError(
            "eapig.corruption must be projection, upstream_steering, "
            "negative_centroid_clamp, contrastive_prompts, "
            "paired_coordinate_projection, target_projection_to_logits, "
            "global_projection_to_logits, or steering."
        )
    output.mkdir(parents=True, exist_ok=True)
    scores = output / "edge_scores.parquet"
    write_edge_scores_parquet(graph, scores)
    result = {
        "model": config["model"],
        "trait": config["trait"],
        "vector_type": vector["vector_type"],
        "corruption": corruption,
        "edge_scores": str(scores),
        "circuits": _write_circuits(
            graph, config, vector, target=circuit_target
        ),
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
        make_logit_graph,
        make_projection_graph,
        make_steering_to_logits_graph,
        set_selected_edges,
    )

    test = config["tests"]
    model = _model(config, hook_mode="eap")
    direction, raw, vector = _load_vector(config, model)
    layer = int(vector["layer"])
    if config["eapig"]["corruption"] in {
        "projection",
        "upstream_steering",
        "negative_centroid_clamp",
        "contrastive_prompts",
        "paired_coordinate_projection",
    }:
        graph = make_projection_graph(model, layer)
    elif config["eapig"]["corruption"] == "target_projection_to_logits":
        from .circuit import _mixed_forward
        from .interventions import direction_ablation_hook
        from .metrics import refusal_log_odds
        from .model import tokenize_prompts

        refusal_token_ids = [
            int(value) for value in config["eapig"]["refusal_token_ids"]
        ]
        target_hook = str(vector["target_hook"])
        target_position = int(vector["position"])
        corrupt_hooks = [
            (
                target_hook,
                direction_ablation_hook(
                    direction, position=target_position
                ),
            )
        ]

        def source_transform(name, activations):
            if name != target_hook:
                return activations
            return direction_ablation_hook(
                direction, position=target_position
            )(activations, None)

        values = {
            name: []
            for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
        }
        for prompt in _prompts(model, test["dataset"]):
            tokens, mask = tokenize_prompts(
                model, [prompt], max_length=test.get("max_length", 512)
            )
            with torch.inference_mode():
                clean_logits = model(tokens, attention_mask=mask)
            with torch.inference_mode(), model.hooks(fwd_hooks=corrupt_hooks):
                corrupt_logits = model(tokens, attention_mask=mask)
            circuit_only = _mixed_forward(
                model,
                graph,
                tokens,
                mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                patch_selected=False,
                corrupt_fwd_hooks=corrupt_hooks,
                corrupt_cache_transform=source_transform,
                cache_dtype=torch.bfloat16,
            )[1]
            circuit_ablated = _mixed_forward(
                model,
                graph,
                tokens,
                mask,
                target_hook=target_hook,
                target_position=target_position,
                direction=direction,
                patch_selected=True,
                corrupt_fwd_hooks=corrupt_hooks,
                corrupt_cache_transform=source_transform,
                cache_dtype=torch.bfloat16,
            )[1]
            logits = {
                "clean": clean_logits,
                "corrupt": corrupt_logits,
                "circuit_only": circuit_only,
                "circuit_ablated": circuit_ablated,
            }
            for name, output_logits in logits.items():
                values[name].append(
                    refusal_log_odds(
                        output_logits, refusal_token_ids
                    ).detach().float().cpu()
                )
        values = {name: torch.cat(items) for name, items in values.items()}
        behavior = None
    elif config["eapig"]["corruption"] == "global_projection_to_logits":
        graph = make_logit_graph(model)
    else:
        graph = make_steering_to_logits_graph(model, layer)
    circuit_path = _path(test["circuit_path"])
    circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
    if circuit.get("corruption") != config["eapig"]["corruption"]:
        raise ValueError("Circuit and test corruption types differ.")
    set_selected_edges(graph, load_circuit_edge_names(circuit_path))

    if config["eapig"]["corruption"] == "contrastive_prompts":
        from .circuit import _normal_forward, mixed_forward_from_prompt_pair
        from .eapig import _response_mask, response_state_mask
        from .model import tokenize_prompts

        target_scope = config["eapig"].get(
            "target_scope",
            (
                "single_position"
                if vector["vector_type"] == "paired_prompt"
                else "response_prefix"
            ),
        )
        if vector["vector_type"] == "paired_prompt":
            _, positive_prompts, negative_prompts = _paired_system_prompts(
                model, test["dataset"]
            )
            pairs = list(zip(positive_prompts, negative_prompts, strict=True))
            target_position = (
                int(vector["position"])
                if target_scope == "single_position"
                else -1
            )
        else:
            pairs = _paired_response_transcripts(
                model, test["dataset"], test.get("max_length", 512)
            )
            target_position = -1

        values = {
            name: []
            for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
        }
        for pair in pairs:
            clean_text, corrupt_text = pair[:2]
            clean_tokens, clean_mask = tokenize_prompts(
                model, [clean_text], max_length=test.get("max_length", 512)
            )
            corrupt_tokens, corrupt_mask = tokenize_prompts(
                model, [corrupt_text], max_length=test.get("max_length", 512)
            )
            if target_scope == "all_prompt_tokens":
                target_mask = clean_mask.bool()
            elif target_scope in {"response_prefix", "all_response_tokens"}:
                response_mask = _response_mask(
                    clean_tokens, clean_mask, [int(pair[2])]
                )
                target_mask = response_state_mask(
                    response_mask,
                    include_terminal=target_scope == "all_response_tokens",
                )
            else:
                target_mask = None
            runs = {
                "clean": _normal_forward(
                    model,
                    clean_tokens,
                    clean_mask,
                    target_hook=vector["target_hook"],
                    target_position=target_position,
                    direction=direction,
                    target_mask=target_mask,
                ),
                "corrupt": _normal_forward(
                    model,
                    corrupt_tokens,
                    corrupt_mask,
                    target_hook=vector["target_hook"],
                    target_position=target_position,
                    direction=direction,
                    target_mask=target_mask,
                ),
                "circuit_only": mixed_forward_from_prompt_pair(
                    model,
                    graph,
                    clean_tokens,
                    clean_mask,
                    corrupt_tokens,
                    corrupt_mask,
                    target_hook=vector["target_hook"],
                    target_position=target_position,
                    direction=direction,
                    patch_selected=False,
                    target_mask=target_mask,
                    cache_dtype=torch.bfloat16,
                ),
                "circuit_ablated": mixed_forward_from_prompt_pair(
                    model,
                    graph,
                    clean_tokens,
                    clean_mask,
                    corrupt_tokens,
                    corrupt_mask,
                    target_hook=vector["target_hook"],
                    target_position=target_position,
                    direction=direction,
                    patch_selected=True,
                    target_mask=target_mask,
                    cache_dtype=torch.bfloat16,
                ),
            }
            for name, (projection, _logits) in runs.items():
                values[name].append(projection.cpu())
        values = {name: torch.cat(items) for name, items in values.items()}
        behavior = None
    elif config["eapig"]["corruption"] == "paired_coordinate_projection":
        from .circuit import _mixed_forward, _normal_forward
        from .eapig import (
            _paired_coordinate_intervention_hooks,
            _paired_source_coordinate_capture_hooks,
            _response_mask,
            response_state_mask,
        )
        from .interventions import paired_coordinate_source_transform
        from .model import tokenize_prompts

        vector_type = vector["vector_type"]
        if vector_type == "paired_prompt":
            _, positive_prompts, negative_prompts = _paired_system_prompts(
                model, test["dataset"]
            )
            pairs = list(
                zip(positive_prompts, negative_prompts, strict=True)
            )
            target_position = int(vector["position"])
        elif vector_type == "response_average":
            pairs = _paired_response_transcripts(
                model, test["dataset"], test.get("max_length", 512)
            )
            target_position = -1
        else:
            raise ValueError(
                "Paired-coordinate circuit evaluation requires a paired_prompt "
                "or response_average vector."
            )
        values = {
            name: []
            for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
        }
        target_hook = vector["target_hook"]
        for pair in pairs:
            clean_text, reference_text = pair[:2]
            clean_tokens, clean_mask = tokenize_prompts(
                model, [clean_text], max_length=test.get("max_length", 512)
            )
            reference_tokens, reference_mask = tokenize_prompts(
                model,
                [reference_text],
                max_length=test.get("max_length", 512),
            )
            if tuple(clean_tokens.shape) != tuple(reference_tokens.shape):
                raise ValueError(
                    "Positive and negative references must tokenize to the same shape."
                )
            if not torch.equal(clean_mask, reference_mask):
                raise ValueError(
                    "Positive and negative attention masks must match."
                )

            if vector_type == "paired_prompt":
                source_mask = clean_mask.bool()
                target_mask = None
            else:
                response_mask = _response_mask(
                    clean_tokens, clean_mask, [int(pair[2])]
                )
                source_mask = response_state_mask(
                    response_mask, include_terminal=False
                )
                target_mask = source_mask
            source_coordinates: dict[str, torch.Tensor] = {}
            target_coordinates: dict[str, torch.Tensor] = {}

            def capture_target(activations: torch.Tensor, hook):
                del hook
                unit = direction.to(
                    device=activations.device,
                    dtype=activations.dtype,
                )
                unit = unit / unit.norm().clamp_min(1e-8)
                target_coordinates["value"] = torch.einsum(
                    "bpd,d->bp", activations, unit
                ).detach().float()
                return activations

            reference_hooks = [
                *_paired_source_coordinate_capture_hooks(
                    graph, direction, source_coordinates
                ),
                (target_hook, capture_target),
            ]
            with torch.inference_mode(), model.hooks(
                fwd_hooks=reference_hooks
            ):
                model(reference_tokens, attention_mask=reference_mask)
            if "value" not in target_coordinates:
                raise RuntimeError(
                    "Paired target-coordinate hook did not fire."
                )

            source_transform = paired_coordinate_source_transform(
                direction,
                target_coordinates=source_coordinates,
                mask=source_mask,
            )
            corrupt_hooks = _paired_coordinate_intervention_hooks(
                graph,
                direction,
                source_coordinates,
                source_mask=source_mask,
                target_hook=target_hook,
                target_coordinates=target_coordinates["value"],
                target_mask=source_mask,
                strength=1.0,
            )
            runs = {
                "clean": _normal_forward(
                    model,
                    clean_tokens,
                    clean_mask,
                    target_hook=target_hook,
                    target_position=target_position,
                    direction=direction,
                    target_mask=target_mask,
                ),
                "corrupt": _normal_forward(
                    model,
                    clean_tokens,
                    clean_mask,
                    target_hook=target_hook,
                    target_position=target_position,
                    direction=direction,
                    fwd_hooks=corrupt_hooks,
                    target_mask=target_mask,
                ),
                "circuit_only": _mixed_forward(
                    model,
                    graph,
                    clean_tokens,
                    clean_mask,
                    target_hook=target_hook,
                    target_position=target_position,
                    direction=direction,
                    patch_selected=False,
                    target_mask=target_mask,
                    corrupt_fwd_hooks=corrupt_hooks,
                    corrupt_cache_transform=source_transform,
                    cache_dtype=torch.bfloat16,
                ),
                "circuit_ablated": _mixed_forward(
                    model,
                    graph,
                    clean_tokens,
                    clean_mask,
                    target_hook=target_hook,
                    target_position=target_position,
                    direction=direction,
                    patch_selected=True,
                    target_mask=target_mask,
                    corrupt_fwd_hooks=corrupt_hooks,
                    corrupt_cache_transform=source_transform,
                    cache_dtype=torch.bfloat16,
                ),
            }
            for name, (projection, _logits) in runs.items():
                values[name].append(projection.cpu())
        values = {name: torch.cat(items) for name, items in values.items()}
        behavior = None
    elif (
        config["eapig"]["corruption"] == "projection"
        and vector["vector_type"] in {"pre_answer", "paired_prompt"}
    ):
        from .circuit import evaluate_conditions

        prompts = (
            _paired_system_prompts(model, test["dataset"])[1]
            if vector["vector_type"] == "paired_prompt"
            else _prompts(model, test["dataset"])
        )
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
        behavior = None
        if vector["vector_type"] == "pre_answer" and test.get("behavior", False):
            from .circuit import behavioural_sanity_check
            from .metrics import DEFAULT_ANSWER_STARTS, REFUSAL_PREFIXES
            from .model import first_token_ids

            behavior = behavioural_sanity_check(
                model,
                graph,
                prompts,
                direction=direction,
                target_hook=vector["target_hook"],
                target_position=int(vector["position"]),
                refusal_ids=first_token_ids(model.tokenizer, REFUSAL_PREFIXES),
                answer_ids=first_token_ids(
                    model.tokenizer, DEFAULT_ANSWER_STARTS
                ),
                max_prompts=len(prompts),
                max_new_tokens=int(test.get("max_new_tokens", 16)),
            )
    elif config["eapig"]["corruption"] in {
        "projection",
        "upstream_steering",
        "negative_centroid_clamp",
    }:
        from .circuit import _mixed_forward, _normal_forward
        from .eapig import _response_mask, response_logprob_values
        from .interventions import (
            all_layer_ablation_hooks,
            all_layer_coordinate_clamp_hooks,
            all_layer_steering_hooks,
            coordinate_clamp_source_transform,
            steering_source_transform,
        )
        from .model import tokenize_prompts

        corruption = config["eapig"]["corruption"]
        if corruption == "projection":
            corrupt_hooks = all_layer_ablation_hooks(model, direction)
            mixed_kwargs = {}
            negative_coordinates = None
        elif corruption == "upstream_steering":
            coefficient = float(config["eapig"]["coefficient"])
            if coefficient >= 0:
                raise ValueError(
                    "Upstream steering corruption coefficient must be negative."
                )
            corrupt_hooks = all_layer_steering_hooks(
                model,
                raw,
                coefficient=coefficient,
            )
            mixed_kwargs = {
                "corrupt_fwd_hooks": corrupt_hooks,
                "corrupt_cache_transform": steering_source_transform(
                    raw,
                    coefficient=coefficient,
                ),
            }
            negative_coordinates = None
        else:
            corrupt_hooks = None
            mixed_kwargs = None
            negative_coordinates = _negative_centroid_coordinates(
                config, model, direction
            )

        values = {name: [] for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")}
        behavior_values = {name: [] for name in values}
        for text, start in _transcripts(
            model, test["dataset"], test.get("max_length", 512)
        ):
            tokens, mask = tokenize_prompts(model, [text], max_length=test.get("max_length", 512))
            response_mask = _response_mask(tokens, mask, [start])
            if corruption == "negative_centroid_clamp":
                current_corrupt_hooks = all_layer_coordinate_clamp_hooks(
                    model,
                    direction,
                    target_coordinates=negative_coordinates,
                    mask=response_mask,
                )
                current_mixed_kwargs = {
                    "corrupt_fwd_hooks": current_corrupt_hooks,
                    "corrupt_cache_transform": coordinate_clamp_source_transform(
                        direction,
                        target_coordinate=float(
                            negative_coordinates[0].item()
                        ),
                        mask=response_mask,
                        input_source_name=graph.nodes["input"].out_hook,
                    ),
                }
            else:
                current_corrupt_hooks = corrupt_hooks
                current_mixed_kwargs = mixed_kwargs
            runs = {
                "clean": _normal_forward(
                    model, tokens, mask, target_hook=vector["target_hook"],
                    target_position=-1, direction=direction, target_mask=response_mask
                ),
                "corrupt": _normal_forward(
                    model, tokens, mask, target_hook=vector["target_hook"],
                    target_position=-1, direction=direction,
                    fwd_hooks=current_corrupt_hooks,
                    target_mask=response_mask
                ),
                "circuit_only": _mixed_forward(
                    model, graph, tokens, mask, target_hook=vector["target_hook"],
                    target_position=-1, direction=direction, patch_selected=False,
                    target_mask=response_mask, **current_mixed_kwargs
                ),
                "circuit_ablated": _mixed_forward(
                    model, graph, tokens, mask, target_hook=vector["target_hook"],
                    target_position=-1, direction=direction, patch_selected=True,
                    target_mask=response_mask, **current_mixed_kwargs
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
    elif config["eapig"]["corruption"] == "global_projection_to_logits":
        from .circuit import _mixed_forward
        from .interventions import all_layer_ablation_hooks
        from .metrics import refusal_log_odds
        from .model import tokenize_prompts

        refusal_token_ids = [
            int(value) for value in config["eapig"]["refusal_token_ids"]
        ]
        values = {
            name: []
            for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
        }
        for prompt in _prompts(model, test["dataset"]):
            tokens, mask = tokenize_prompts(
                model, [prompt], max_length=test.get("max_length", 512)
            )
            with torch.inference_mode():
                clean_logits = model(tokens, attention_mask=mask)
            with torch.inference_mode(), model.hooks(
                fwd_hooks=all_layer_ablation_hooks(model, direction)
            ):
                corrupt_logits = model(tokens, attention_mask=mask)
            circuit_only = _mixed_forward(
                model,
                graph,
                tokens,
                mask,
                target_hook=vector["target_hook"],
                target_position=int(vector["position"]),
                direction=direction,
                patch_selected=False,
                cache_dtype=torch.bfloat16,
            )[1]
            circuit_ablated = _mixed_forward(
                model,
                graph,
                tokens,
                mask,
                target_hook=vector["target_hook"],
                target_position=int(vector["position"]),
                direction=direction,
                patch_selected=True,
                cache_dtype=torch.bfloat16,
            )[1]
            logits = {
                "clean": clean_logits,
                "corrupt": corrupt_logits,
                "circuit_only": circuit_only,
                "circuit_ablated": circuit_ablated,
            }
            for name, output_logits in logits.items():
                values[name].append(
                    refusal_log_odds(
                        output_logits, refusal_token_ids
                    ).detach().float().cpu()
                )
        values = {name: torch.cat(items) for name, items in values.items()}
        behavior = None
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

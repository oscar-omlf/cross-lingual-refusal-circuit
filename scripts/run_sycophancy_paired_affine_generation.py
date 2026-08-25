#!/usr/bin/env python3
"""Generate a causal gate for paired affine generation-vector projection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusal_circuit.persona import (
    make_system_prompt,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from refusal_circuit.paths import write_json
from refusal_circuit.sycophancy_geometry import (
    ALIGNED_SYCOPHANCY_INSTRUCTION_PAIRS,
)


DEFAULT_DATASET = (
    ROOT
    / "data/processed/qwen2.5-7b-instruct/trait_circuits/sycophancy"
    / "vector_validation.jsonl"
)
DEFAULT_VECTOR = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/generation/vector"
    / "sycophantic_direction.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/qwen2.5-7b-instruct/sycophancy/generation"
    / "paired_affine_behavior_gate_n20"
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def format_prompt(tokenizer, row: dict, condition: str) -> str:
    pair_index = int(row["instruction_pair_index"])
    positive, negative = ALIGNED_SYCOPHANCY_INSTRUCTION_PAIRS[pair_index]
    instruction = {"positive": positive, "negative": negative}[condition]
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": make_system_prompt("sycophantic", instruction),
            },
            {"role": "user", "content": row["prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def generated_token_count(token_ids: list[int], eos_token_id: int | None) -> int:
    if eos_token_id is None or eos_token_id not in token_ids:
        return len(token_ids)
    return token_ids.index(eos_token_id) + 1


def output_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported hooked output type: {type(output)!r}")


def replace_output_tensor(output, replacement: torch.Tensor):
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise TypeError(f"Unsupported hooked output type: {type(output)!r}")


class PairedSourceCoordinates:
    def __init__(
        self,
        model,
        direction: torch.Tensor,
        *,
        strength: float = 1.0,
    ) -> None:
        self.model = model
        self.direction = direction.float() / direction.float().norm().clamp_min(1e-8)
        self.strength = float(strength)
        self.negative_targets: dict[str, torch.Tensor] = {}
        self.positive_targets: dict[str, torch.Tensor] = {}
        self.max_projection_error = 0.0

    def coordinate(self, tensor: torch.Tensor) -> torch.Tensor:
        vector = self.direction.to(device=tensor.device, dtype=tensor.dtype)
        return torch.einsum("...d,d->...", tensor, vector)

    def capture(
        self,
        key: str,
        tensor: torch.Tensor,
        *,
        all_positions: bool,
        reference: str,
    ) -> None:
        selected = tensor if all_positions else tensor[:, -1:]
        targets = {
            "negative": self.negative_targets,
            "positive": self.positive_targets,
        }[reference]
        targets[key] = self.coordinate(selected).detach().float().cpu()

    def project(
        self, key: str, tensor: torch.Tensor, *, all_positions: bool
    ) -> torch.Tensor:
        if key not in self.negative_targets:
            raise KeyError(f"Missing negative-coordinate target for {key}.")
        vector = self.direction.to(device=tensor.device, dtype=tensor.dtype)
        selected = tensor if all_positions else tensor[:, -1:]
        coordinate = self.coordinate(selected)
        negative = self.negative_targets[key].to(
            device=tensor.device,
            dtype=coordinate.dtype,
        )
        if self.strength == 1.0:
            endpoint = negative
        else:
            if key not in self.positive_targets:
                raise KeyError(f"Missing positive-coordinate reference for {key}.")
            positive = self.positive_targets[key].to(
                device=tensor.device,
                dtype=coordinate.dtype,
            )
            endpoint = positive + self.strength * (negative - positive)
        replacement = selected + (endpoint - coordinate).unsqueeze(-1) * vector
        if all_positions:
            changed = replacement
        else:
            changed = tensor.clone()
            changed[:, -1:] = replacement
        error = (
            self.coordinate(
                changed if all_positions else changed[:, -1:]
            ).float()
            - endpoint.float()
        ).abs().max().item()
        self.max_projection_error = max(self.max_projection_error, error)
        return changed

    def capture_hooks(
        self,
        *,
        all_positions: bool = False,
        reference: str = "negative",
    ) -> list:
        handles = []
        first_layer = self.model.model.layers[0]

        def capture_input(_module, inputs):
            self.capture(
                "input",
                inputs[0],
                all_positions=all_positions,
                reference=reference,
            )

        handles.append(first_layer.register_forward_pre_hook(capture_input))
        for layer_index, layer in enumerate(self.model.model.layers):
            def capture_attention(_module, _inputs, output, index=layer_index):
                self.capture(
                    f"attn_{index}",
                    output_tensor(output),
                    all_positions=all_positions,
                    reference=reference,
                )

            def capture_mlp(_module, _inputs, output, index=layer_index):
                self.capture(
                    f"mlp_{index}",
                    output_tensor(output),
                    all_positions=all_positions,
                    reference=reference,
                )

            handles.extend(
                [
                    layer.self_attn.register_forward_hook(capture_attention),
                    layer.mlp.register_forward_hook(capture_mlp),
                ]
            )
        return handles

    def projection_hooks(self, *, all_positions: bool = False) -> list:
        handles = []
        first_layer = self.model.model.layers[0]

        def project_input(_module, inputs):
            return (
                self.project(
                    "input", inputs[0], all_positions=all_positions
                ),
                *inputs[1:],
            )

        handles.append(first_layer.register_forward_pre_hook(project_input))
        for layer_index, layer in enumerate(self.model.model.layers):
            def project_attention(_module, _inputs, output, index=layer_index):
                changed = self.project(
                    f"attn_{index}",
                    output_tensor(output),
                    all_positions=all_positions,
                )
                return replace_output_tensor(output, changed)

            def project_mlp(_module, _inputs, output, index=layer_index):
                changed = self.project(
                    f"mlp_{index}",
                    output_tensor(output),
                    all_positions=all_positions,
                )
                return replace_output_tensor(output, changed)

            handles.extend(
                [
                    layer.self_attn.register_forward_hook(project_attention),
                    layer.mlp.register_forward_hook(project_mlp),
                ]
            )
        return handles


class PairedLayerCoordinate:
    """Replace one direction coordinate at one residual-stream layer."""

    def __init__(
        self,
        model,
        direction: torch.Tensor,
        *,
        layer: int,
        strength: float = 1.0,
    ) -> None:
        if not 0 <= int(layer) < len(model.model.layers):
            raise ValueError(f"Layer {layer} is outside the model.")
        self.block = model.model.layers[int(layer)]
        self.direction = (
            direction.float() / direction.float().norm().clamp_min(1e-8)
        )
        self.strength = float(strength)
        self.negative_target: torch.Tensor | None = None
        self.positive_target: torch.Tensor | None = None
        self.max_projection_error = 0.0

    def coordinate(self, tensor: torch.Tensor) -> torch.Tensor:
        vector = self.direction.to(device=tensor.device, dtype=tensor.dtype)
        return torch.einsum("...d,d->...", tensor, vector)

    @staticmethod
    def selected(tensor: torch.Tensor, *, all_positions: bool) -> torch.Tensor:
        return tensor if all_positions else tensor[:, -1:]

    def capture_hook(
        self,
        *,
        all_positions: bool,
        reference: str,
    ):
        if reference not in {"negative", "positive"}:
            raise ValueError("Reference must be 'negative' or 'positive'.")

        def capture(_module, inputs):
            selected = self.selected(inputs[0], all_positions=all_positions)
            coordinate = self.coordinate(selected).detach().float().cpu()
            if reference == "negative":
                self.negative_target = coordinate
            else:
                self.positive_target = coordinate

        return self.block.register_forward_pre_hook(capture)

    def projection_hook(self, *, all_positions: bool):
        def project(_module, inputs):
            if self.negative_target is None:
                raise RuntimeError("Negative coordinate target was not captured.")
            tensor = inputs[0]
            selected = self.selected(tensor, all_positions=all_positions)
            coordinate = self.coordinate(selected)
            negative = self.negative_target.to(
                device=tensor.device,
                dtype=coordinate.dtype,
            )
            if self.strength == 1.0:
                endpoint = negative
            else:
                if self.positive_target is None:
                    raise RuntimeError(
                        "Positive coordinate target was not captured."
                    )
                positive = self.positive_target.to(
                    device=tensor.device,
                    dtype=coordinate.dtype,
                )
                endpoint = positive + self.strength * (negative - positive)
            vector = self.direction.to(
                device=tensor.device,
                dtype=tensor.dtype,
            )
            replacement = (
                selected + (endpoint - coordinate).unsqueeze(-1) * vector
            )
            if all_positions:
                changed = replacement
            else:
                changed = tensor.clone()
                changed[:, -1:] = replacement
            error = (
                self.coordinate(
                    changed if all_positions else changed[:, -1:]
                ).float()
                - endpoint.float()
            ).abs().max().item()
            self.max_projection_error = max(
                self.max_projection_error,
                error,
            )
            return (changed, *inputs[1:])

        return self.block.register_forward_pre_hook(project)


def remove_hooks(handles: list) -> None:
    for handle in handles:
        handle.remove()


def forward_with_cache(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values=None,
    handles: list | None = None,
):
    try:
        with torch.inference_mode():
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
    finally:
        remove_hooks(handles or [])


def paired_greedy_completion(
    model,
    tokenizer,
    *,
    positive_prompt: str,
    negative_prompt: str,
    direction: torch.Tensor,
    max_new_tokens: int,
    project_prompt: bool,
    project_response: bool,
    project_all_prompt_positions: bool,
    strength: float,
) -> tuple[list[int], float]:
    positive = tokenizer(
        positive_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    negative = tokenizer(
        negative_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if not torch.equal(
        positive["attention_mask"], negative["attention_mask"]
    ) or positive["input_ids"].shape != negative["input_ids"].shape:
        raise ValueError("Positive and negative prompts must be token aligned.")
    positive = {key: value.to(model.device) for key, value in positive.items()}
    negative = {key: value.to(model.device) for key, value in negative.items()}

    intervention = PairedSourceCoordinates(
        model,
        direction,
        strength=strength,
    )
    negative_prefill_handles = (
        intervention.capture_hooks(
            all_positions=project_all_prompt_positions
        )
        if project_prompt
        else []
    )
    negative_outputs = forward_with_cache(
        model,
        input_ids=negative["input_ids"],
        attention_mask=negative["attention_mask"],
        handles=negative_prefill_handles,
    )
    positive_reference_outputs = None
    if strength != 1.0:
        positive_reference_outputs = forward_with_cache(
            model,
            input_ids=positive["input_ids"],
            attention_mask=positive["attention_mask"],
            handles=(
                intervention.capture_hooks(
                    all_positions=project_all_prompt_positions,
                    reference="positive",
                )
                if project_prompt
                else []
            ),
        )
    positive_prefill_handles = (
        intervention.projection_hooks(
            all_positions=project_all_prompt_positions
        )
        if project_prompt
        else []
    )
    positive_outputs = forward_with_cache(
        model,
        input_ids=positive["input_ids"],
        attention_mask=positive["attention_mask"],
        handles=positive_prefill_handles,
    )

    positive_past = positive_outputs.past_key_values
    negative_past = negative_outputs.past_key_values
    positive_reference_past = (
        positive_reference_outputs.past_key_values
        if positive_reference_outputs is not None
        else None
    )
    positive_mask = positive["attention_mask"]
    negative_mask = negative["attention_mask"]
    next_token = positive_outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    eos_id = tokenizer.eos_token_id

    while len(generated) < max_new_tokens and (
        eos_id is None or generated[-1] != eos_id
    ):
        positive_mask = torch.cat(
            [
                positive_mask,
                torch.ones(
                    (1, 1),
                    device=positive_mask.device,
                    dtype=positive_mask.dtype,
                ),
            ],
            dim=1,
        )
        negative_mask = torch.cat(
            [
                negative_mask,
                torch.ones(
                    (1, 1),
                    device=negative_mask.device,
                    dtype=negative_mask.dtype,
                ),
            ],
            dim=1,
        )
        if project_response:
            negative_outputs = forward_with_cache(
                model,
                input_ids=next_token,
                attention_mask=negative_mask,
                past_key_values=negative_past,
                handles=intervention.capture_hooks(),
            )
            negative_past = negative_outputs.past_key_values
            if strength != 1.0:
                positive_reference_outputs = forward_with_cache(
                    model,
                    input_ids=next_token,
                    attention_mask=positive_mask,
                    past_key_values=positive_reference_past,
                    handles=intervention.capture_hooks(
                        reference="positive"
                    ),
                )
                positive_reference_past = (
                    positive_reference_outputs.past_key_values
                )
        positive_handles = (
            intervention.projection_hooks() if project_response else []
        )
        positive_outputs = forward_with_cache(
            model,
            input_ids=next_token,
            attention_mask=positive_mask,
            past_key_values=positive_past,
            handles=positive_handles,
        )
        positive_past = positive_outputs.past_key_values
        next_token = positive_outputs.logits[:, -1].argmax(
            dim=-1, keepdim=True
        )
        generated.append(int(next_token.item()))
    return generated, intervention.max_projection_error


def dual_vector_greedy_completion(
    model,
    tokenizer,
    *,
    positive_prompt: str,
    negative_prompt: str,
    prompt_direction: torch.Tensor,
    prompt_layer: int,
    generation_direction: torch.Tensor,
    generation_layer: int,
    max_new_tokens: int,
    project_prompt: bool,
    project_generation: bool,
    strength: float,
) -> tuple[list[int], float]:
    """Use the prompt vector on all prompt tokens and generation vector later."""

    positive = tokenizer(
        positive_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    negative = tokenizer(
        negative_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if not torch.equal(
        positive["attention_mask"], negative["attention_mask"]
    ) or positive["input_ids"].shape != negative["input_ids"].shape:
        raise ValueError("Positive and negative prompts must be token aligned.")
    positive = {key: value.to(model.device) for key, value in positive.items()}
    negative = {key: value.to(model.device) for key, value in negative.items()}

    prompt_intervention = PairedLayerCoordinate(
        model,
        prompt_direction,
        layer=prompt_layer,
        strength=strength,
    )
    generation_intervention = PairedLayerCoordinate(
        model,
        generation_direction,
        layer=generation_layer,
        strength=strength,
    )

    negative_prefill_handles = (
        [
            prompt_intervention.capture_hook(
                all_positions=True,
                reference="negative",
            )
        ]
        if project_prompt
        else []
    )
    negative_outputs = forward_with_cache(
        model,
        input_ids=negative["input_ids"],
        attention_mask=negative["attention_mask"],
        handles=negative_prefill_handles,
    )

    positive_reference_outputs = None
    if strength != 1.0:
        positive_reference_outputs = forward_with_cache(
            model,
            input_ids=positive["input_ids"],
            attention_mask=positive["attention_mask"],
            handles=(
                [
                    prompt_intervention.capture_hook(
                        all_positions=True,
                        reference="positive",
                    )
                ]
                if project_prompt
                else []
            ),
        )

    positive_outputs = forward_with_cache(
        model,
        input_ids=positive["input_ids"],
        attention_mask=positive["attention_mask"],
        handles=(
            [
                prompt_intervention.projection_hook(
                    all_positions=True,
                )
            ]
            if project_prompt
            else []
        ),
    )

    positive_past = positive_outputs.past_key_values
    negative_past = negative_outputs.past_key_values
    positive_reference_past = (
        positive_reference_outputs.past_key_values
        if positive_reference_outputs is not None
        else None
    )
    positive_mask = positive["attention_mask"]
    negative_mask = negative["attention_mask"]
    next_token = positive_outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    eos_id = tokenizer.eos_token_id

    while len(generated) < max_new_tokens and (
        eos_id is None or generated[-1] != eos_id
    ):
        positive_mask = torch.cat(
            (
                positive_mask,
                torch.ones(
                    (1, 1),
                    device=positive_mask.device,
                    dtype=positive_mask.dtype,
                ),
            ),
            dim=1,
        )
        negative_mask = torch.cat(
            (
                negative_mask,
                torch.ones(
                    (1, 1),
                    device=negative_mask.device,
                    dtype=negative_mask.dtype,
                ),
            ),
            dim=1,
        )

        if project_generation:
            negative_outputs = forward_with_cache(
                model,
                input_ids=next_token,
                attention_mask=negative_mask,
                past_key_values=negative_past,
                handles=[
                    generation_intervention.capture_hook(
                        all_positions=False,
                        reference="negative",
                    )
                ],
            )
            negative_past = negative_outputs.past_key_values

            if strength != 1.0:
                positive_reference_outputs = forward_with_cache(
                    model,
                    input_ids=next_token,
                    attention_mask=positive_mask,
                    past_key_values=positive_reference_past,
                    handles=[
                        generation_intervention.capture_hook(
                            all_positions=False,
                            reference="positive",
                        )
                    ],
                )
                positive_reference_past = (
                    positive_reference_outputs.past_key_values
                )

        positive_outputs = forward_with_cache(
            model,
            input_ids=next_token,
            attention_mask=positive_mask,
            past_key_values=positive_past,
            handles=(
                [
                    generation_intervention.projection_hook(
                        all_positions=False,
                    )
                ]
                if project_generation
                else []
            ),
        )
        positive_past = positive_outputs.past_key_values
        next_token = positive_outputs.logits[:, -1].argmax(
            dim=-1,
            keepdim=True,
        )
        generated.append(int(next_token.item()))

    return generated, max(
        prompt_intervention.max_projection_error,
        generation_intervention.max_projection_error,
    )


def baseline_completions(
    model,
    tokenizer,
    *,
    rows: list[dict],
    condition: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    output = []
    prompts = [format_prompt(tokenizer, row, condition) for row in rows]
    for start in tqdm(
        range(0, len(rows), batch_size),
        desc=f"Generating baseline {condition}",
    ):
        row_batch = rows[start : start + batch_size]
        prompt_batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            prompt_batch,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row, tokens in zip(row_batch, generated.detach().cpu(), strict=True):
            continuation = tokens[prompt_length:].tolist()
            count = generated_token_count(
                continuation, tokenizer.eos_token_id
            )
            output.append(
                response_row(
                    row,
                    condition=f"baseline_{condition}",
                    token_ids=continuation[:count],
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    max_projection_error=None,
                )
            )
    return output


def response_row(
    row: dict,
    *,
    condition: str,
    token_ids: list[int],
    tokenizer,
    max_new_tokens: int,
    max_projection_error: float | None,
) -> dict:
    return {
        "record_id": f"{condition}-{row['id']}",
        "id": row["id"],
        "source_index": int(row["source_index"]),
        "instruction_pair_index": int(row["instruction_pair_index"]),
        "question": row["prompt"],
        "condition": condition,
        "layer": None,
        "position": None,
        "answer": tokenizer.decode(
            token_ids, skip_special_tokens=True
        ).strip(),
        "generated_token_count": len(token_ids),
        "hit_max_new_tokens": len(token_ids) >= max_new_tokens,
        "max_projection_error": max_projection_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vector", type=Path, default=DEFAULT_VECTOR)
    parser.add_argument(
        "--prompt-vector",
        type=Path,
        help=(
            "Enable dual-vector mode with this prompt-vector artifact. "
            "A 2D all-layer artifact is indexed by --prompt-layer."
        ),
    )
    parser.add_argument("--prompt-layer", type=int, default=15)
    parser.add_argument("--generation-layer", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--revision",
        default="a09a35458c702b33eeacc393d103063234e8bc28",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    responses_path = output_dir / "responses.jsonl"
    metadata_path = output_dir / "metadata.json"
    if any(path.exists() for path in (responses_path, metadata_path)):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already contains output.")
        responses_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    dataset_path = resolve(args.dataset)
    vector_path = resolve(args.vector)
    prompt_vector_path = (
        resolve(args.prompt_vector) if args.prompt_vector is not None else None
    )
    rows = read_jsonl(dataset_path)[: args.num_questions]
    if len(rows) != args.num_questions:
        raise ValueError("Dataset contains fewer rows than requested.")
    counts = Counter(int(row["instruction_pair_index"]) for row in rows)
    if max(counts.values()) - min(counts.values()) > 1:
        raise ValueError(f"Instruction pairs are unbalanced: {dict(counts)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    direction = torch.load(
        vector_path, map_location="cpu", weights_only=True
    ).float().reshape(-1)
    if direction.numel() != int(model.config.hidden_size):
        raise ValueError("Direction does not match model hidden size.")
    prompt_direction = None
    if prompt_vector_path is not None:
        prompt_artifact = torch.load(
            prompt_vector_path,
            map_location="cpu",
            weights_only=True,
        ).float()
        if prompt_artifact.ndim == 2:
            if not 0 <= args.prompt_layer < prompt_artifact.shape[0]:
                raise ValueError("Prompt layer is absent from the vector artifact.")
            prompt_direction = prompt_artifact[args.prompt_layer].reshape(-1)
        elif prompt_artifact.ndim == 1:
            prompt_direction = prompt_artifact.reshape(-1)
        else:
            raise ValueError("Prompt vector must have one or two dimensions.")
        if prompt_direction.numel() != int(model.config.hidden_size):
            raise ValueError("Prompt direction does not match model hidden size.")
        if not 0 <= args.generation_layer < len(model.model.layers):
            raise ValueError("Generation layer is outside the model.")

    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        *baseline_completions(
            model,
            tokenizer,
            rows=rows,
            condition="negative",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        ),
        *baseline_completions(
            model,
            tokenizer,
            rows=rows,
            condition="positive",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        ),
    ]
    write_jsonl(responses_path, generated)

    if prompt_direction is None:
        modes = {
            "paired_prompt_only": (True, False, False, 1.0),
            "paired_response_only": (False, True, False, 1.0),
            "paired_generation_causal": (True, True, False, 1.0),
            "paired_global": (True, True, True, 1.0),
            "paired_generation_causal_strength_2": (
                True,
                True,
                False,
                2.0,
            ),
            "paired_global_strength_2": (True, True, True, 2.0),
        }
    else:
        modes = {
            "prompt_vector_all_tokens": (True, False, False, 1.0),
            "generation_vector_recurrent": (False, True, False, 1.0),
            "prompt_plus_generation_vectors": (True, True, False, 1.0),
            "prompt_vector_all_tokens_strength_2": (
                True,
                False,
                False,
                2.0,
            ),
            "generation_vector_recurrent_strength_2": (
                False,
                True,
                False,
                2.0,
            ),
            "prompt_plus_generation_vectors_strength_2": (
                True,
                True,
                False,
                2.0,
            ),
        }

    for condition, (
        project_prompt,
        project_response,
        project_all_prompt_positions,
        strength,
    ) in modes.items():
        rows_for_condition = []
        for row in tqdm(rows, desc=f"Generating {condition}"):
            if prompt_direction is None:
                token_ids, error = paired_greedy_completion(
                    model,
                    tokenizer,
                    positive_prompt=format_prompt(tokenizer, row, "positive"),
                    negative_prompt=format_prompt(tokenizer, row, "negative"),
                    direction=direction,
                    max_new_tokens=args.max_new_tokens,
                    project_prompt=project_prompt,
                    project_response=project_response,
                    project_all_prompt_positions=project_all_prompt_positions,
                    strength=strength,
                )
            else:
                token_ids, error = dual_vector_greedy_completion(
                    model,
                    tokenizer,
                    positive_prompt=format_prompt(tokenizer, row, "positive"),
                    negative_prompt=format_prompt(tokenizer, row, "negative"),
                    prompt_direction=prompt_direction,
                    prompt_layer=args.prompt_layer,
                    generation_direction=direction,
                    generation_layer=args.generation_layer,
                    max_new_tokens=args.max_new_tokens,
                    project_prompt=project_prompt,
                    project_generation=project_response,
                    strength=strength,
                )
            rows_for_condition.append(
                response_row(
                    row,
                    condition=condition,
                    token_ids=token_ids,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    max_projection_error=error,
                )
            )
        write_jsonl(responses_path, rows_for_condition, mode="a")

    generated = read_jsonl(responses_path)
    expected = len(rows) * (2 + len(modes))
    if len(generated) != expected:
        raise RuntimeError(f"Expected {expected} responses, found {len(generated)}.")
    if prompt_direction is None:
        method = "dynamic_paired_affine_source_projection"
        vector_metadata = {
            "path": str(vector_path),
            "sha256": sha256_file(vector_path),
        }
        condition_metadata = {
            "baseline_negative": "natural anti-sycophantic instruction",
            "baseline_positive": "natural sycophantic instruction",
            "paired_prompt_only": (
                "positive instruction; paired affine source projection at "
                "the final prompt state only"
            ),
            "paired_response_only": (
                "positive instruction; paired affine source projection at "
                "every generated-prefix state, excluding initiation"
            ),
            "paired_generation_causal": (
                "positive instruction; paired affine source projection at "
                "the final prompt state and every generated-prefix state"
            ),
            "paired_global": (
                "positive instruction; paired affine source projection at "
                "every prompt position and every generated-prefix state; "
                "literal paired analogue of refusal global projection"
            ),
            "paired_generation_causal_strength_2": (
                "positive instruction; two matched natural coordinate gaps "
                "at the final prompt state and every generated-prefix state"
            ),
            "paired_global_strength_2": (
                "positive instruction; two matched natural coordinate gaps "
                "at every prompt position and generated-prefix state"
            ),
        }
        projection_metadata = {
            "sources": (
                "layer-0 residual input, every complete attention output, "
                "and every MLP output"
            ),
            "reference": (
                "token-aligned negative-instruction run with the identical "
                "generated prefix"
            ),
            "orthogonal_state": "preserved from the positive run",
            "decoding": "greedy with separate positive and negative KV caches",
        }
    else:
        method = "dual_stage_paired_affine_layer_projection"
        vector_metadata = {
            "prompt": {
                "path": str(prompt_vector_path),
                "sha256": sha256_file(prompt_vector_path),
                "layer": args.prompt_layer,
                "extraction_position": -1,
            },
            "generation": {
                "path": str(vector_path),
                "sha256": sha256_file(vector_path),
                "layer": args.generation_layer,
            },
        }
        condition_metadata = {
            "baseline_negative": "natural anti-sycophantic instruction",
            "baseline_positive": "natural sycophantic instruction",
            "prompt_vector_all_tokens": (
                f"positive instruction; layer-{args.prompt_layer} prompt-vector "
                "coordinates moved to paired negative values at every prompt "
                "token"
            ),
            "generation_vector_recurrent": (
                f"positive instruction; layer-{args.generation_layer} "
                "generation-vector coordinate moved to the paired negative "
                "value at every generated-prefix state, excluding first-token "
                "initiation"
            ),
            "prompt_plus_generation_vectors": (
                "both corresponding natural-scale prompt and generation "
                "coordinate interventions"
            ),
            "prompt_vector_all_tokens_strength_2": (
                "two paired prompt-vector coordinate gaps at every prompt token"
            ),
            "generation_vector_recurrent_strength_2": (
                "two paired generation-vector coordinate gaps at every "
                "generated-prefix state"
            ),
            "prompt_plus_generation_vectors_strength_2": (
                "both corresponding prompt and generation interventions at "
                "two paired coordinate gaps"
            ),
        }
        projection_metadata = {
            "prompt": (
                f"blocks.{args.prompt_layer}.hook_resid_pre at every prompt token"
            ),
            "generation": (
                f"blocks.{args.generation_layer}.hook_resid_pre at every "
                "generated-prefix state"
            ),
            "reference": (
                "token-aligned negative-instruction run with the identical "
                "generated prefix"
            ),
            "orthogonal_state": "preserved from the positive run",
            "decoding": "greedy with separate positive and negative KV caches",
        }

    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "status": "generation_complete_pending_persona_judge",
            "method": method,
            "model": {
                "name": args.model,
                "revision": args.revision,
                "dtype": "bfloat16",
            },
            "dataset": {
                "path": str(dataset_path),
                "sha256": sha256_file(dataset_path),
                "questions": len(rows),
                "instruction_pair_counts": dict(sorted(counts.items())),
                "instruction_source": (
                    "frozen token-aligned sycophancy instruction pairs"
                ),
            },
            "vector": vector_metadata,
            "conditions": condition_metadata,
            "projection": projection_metadata,
            "generation": {
                "deterministic": True,
                "batch_size_baselines": args.batch_size,
                "batch_size_paired": 1,
                "max_new_tokens": args.max_new_tokens,
                "expected_responses": expected,
            },
            "wall_time_seconds": time.monotonic() - started,
            "artifacts": {
                "responses": {
                    "path": str(responses_path),
                    "rows": len(generated),
                    "sha256": sha256_file(responses_path),
                }
            },
        },
    )


if __name__ == "__main__":
    main()

"""Gemma-2 model loading and chat-prompt formatting for RQ1.

The same ``load_model`` function is used by every RQ1 script, including the
smoke test.  Keeping this in one place is important: TransformerLens hook
names and model configuration are part of the experimental protocol.
"""

from __future__ import annotations

from typing import Iterable

import torch


DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - defensive for unusual wrappers
        raise RuntimeError("The loaded model has no parameters.") from exc


def configure_eap_hooks(model) -> None:
    """Enable the TransformerLens hook points required by the EAP graph.

    TransformerLens has used both setter methods and writable config flags for
    these options.  Supporting both keeps the project usable across the
    version range documented in requirements.txt.
    """
    settings = {
        "use_split_qkv_input": True,
        "use_attn_result": True,
        "use_hook_mlp_in": True,
    }
    for attribute, value in settings.items():
        setter = getattr(model, f"set_{attribute}", None)
        if callable(setter):
            setter(value)
        elif hasattr(model.cfg, attribute):
            setattr(model.cfg, attribute, value)
        else:
            raise RuntimeError(
                f"TransformerLens did not expose {attribute}; install a supported "
                "TransformerLens version from requirements.txt."
            )

    # Gemma-2 uses grouped-query attention (8 Q heads, 4 KV heads).  EAP's
    # head-level graph has one Q/K/V input edge set per attention head, so ask
    # TransformerLens to expose K/V inputs expanded to n_heads as well.  This
    # is a view-level tracing choice; it does not change model weights or the
    # forward function.
    if getattr(model.cfg, "n_key_value_heads", None) not in (None, model.cfg.n_heads):
        setter = getattr(model, "set_ungroup_grouped_query_attention", None)
        if not callable(setter):
            raise RuntimeError(
                "Gemma-2 grouped-query attention needs "
                "set_ungroup_grouped_query_attention in TransformerLens."
            )
        setter(True)


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: str | None = None,
    dtype: str = "bfloat16",
):
    """Load an instruction-tuned model through TransformerLens.

    ``google/gemma-2-2b-it`` is deliberately the default.  RQ1 scripts do not
    silently fall back to GPT-2 or a base model because those do not answer the
    refusal-circuit question.
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:  # Makes a missing environment actionable.
        raise RuntimeError(
            "TransformerLens is unavailable. Create the documented environment "
            "with `conda env create -f environment.yml`."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and dtype == "bfloat16":
        # bfloat16 operations are supported by recent CPUs but are needlessly
        # fragile for a smoke test; numerical RQ1 runs should use CUDA.
        requested_dtype = torch.float32
    else:
        requested_dtype = getattr(torch, dtype)

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=requested_dtype,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
    )
    model.eval()
    model.requires_grad_(False)
    configure_eap_hooks(model)

    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    return model


def format_instruction(tokenizer, instruction: str) -> str:
    """Format one user instruction with Gemma's native generation prompt.

    ``apply_chat_template`` is preferred so tokenizer revisions are respected.
    The fallback is the documented Gemma turn format, retained for older
    tokenizers that lack a template.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


def format_instructions(tokenizer, instructions: Iterable[str]) -> list[str]:
    return [format_instruction(tokenizer, instruction) for instruction in instructions]


def post_instruction_positions(tokenizer) -> tuple[int, ...]:
    """Return the negative token offsets of Gemma's post-instruction suffix.

    Arditi et al. select a direction from *every* post-instruction template
    token, rather than presupposing that the final newline is optimal.  We
    derive that suffix from the loaded tokenizer/template instead of hard
    coding its length.  The returned positions are suitable for left-padded
    batches because the suffix always ends at the final sequence position.
    """
    marker = "RQ1_POST_INSTRUCTION_MARKER_7f3a"
    prompt = format_instruction(tokenizer, marker)
    if marker not in prompt:
        raise RuntimeError("Could not locate the instruction marker in the formatted chat prompt.")
    suffix = prompt.rsplit(marker, 1)[1]
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if not suffix_ids:
        raise RuntimeError("The chat template has no tokenized post-instruction suffix.")
    return tuple(range(-len(suffix_ids), 0))


def tokenize_prompts(model, prompts: list[str], max_length: int | None = None):
    """Tokenize already-formatted prompts for an explicit hooked forward pass."""
    # ``apply_chat_template(..., tokenize=False)`` emits Gemma's control
    # tokens itself (including BOS when its template requires one). Adding
    # tokenizer special tokens again can prepend a second BOS and changes the
    # residual position we call assistant-start. The fallback formatter has no
    # chat template, so it retains the tokenizer's normal BOS handling.
    add_special_tokens = not bool(getattr(model.tokenizer, "chat_template", None))
    encoded = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    device = _model_device(model)
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def first_token_ids(tokenizer, strings: Iterable[str]) -> list[int]:
    """Return one vocabulary ID per candidate continuation string."""
    token_ids: list[int] = []
    for string in strings:
        ids = tokenizer.encode(string, add_special_tokens=False)
        if ids:
            token_ids.append(int(ids[0]))
    return sorted(set(token_ids))

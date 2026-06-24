#!/usr/bin/env python3
"""Load Gemma-2 through the exact RQ1 path and run one formatted prompt."""

from __future__ import annotations

import argparse

from _rq1_bootstrap import ROOT  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    try:
        from refusal_circuit.model import format_instruction, load_model, tokenize_prompts
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Smoke test blocked: the RQ1 environment is not installed. "
            "Create it with `conda env create -f environment.yml`."
        ) from exc

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    prompt = format_instruction(model.tokenizer, "Say hello in one sentence.")
    tokens, attention_mask = tokenize_prompts(model, [prompt])
    logits = model(tokens, attention_mask=attention_mask)
    assistant_start_token_id = int(tokens[0, -1].item())
    token_tail_ids = tokens[0, -8:].tolist()
    print(
        {
            "model": args.model,
            "device": str(next(model.parameters()).device),
            "tokens": int(tokens.shape[1]),
            "logits_shape": tuple(logits.shape),
            "chat_template_used": bool(getattr(model.tokenizer, "chat_template", None)),
            "assistant_start_position": -1,
            "assistant_start_token_id": assistant_start_token_id,
            "assistant_start_token": model.tokenizer.convert_ids_to_tokens(assistant_start_token_id),
            "prompt_token_tail": model.tokenizer.convert_ids_to_tokens(token_tail_ids),
            "eap_hooks": {
                "split_qkv": bool(model.cfg.use_split_qkv_input),
                "attn_result": bool(model.cfg.use_attn_result),
                "mlp_in": bool(model.cfg.use_hook_mlp_in),
                "ungroup_grouped_query_attention": bool(
                    getattr(model.cfg, "ungroup_grouped_query_attention", False)
                ),
            },
        }
    )


if __name__ == "__main__":
    main()

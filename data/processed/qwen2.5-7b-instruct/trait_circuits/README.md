# Canonical trait-circuit datasets

Each trait has the same five non-overlapping experiment splits:

- `vector_extraction.jsonl`: 256 prompts
- `vector_validation.jsonl`: 64 prompts
- `eapig.jsonl`: 64 prompts
- `circuit_selection.jsonl`: 64 prompts
- `final_behavior.jsonl`: 300 prompts

This gives exactly 748 unique prompts per trait. See each trait's
`manifest.json` for labels, source files, hashes, and intervention protocol.

The sycophancy rows retain the normal, sycophantic, and extreme-sycophantic
answers. They also contain the paired official system instructions used for
prompt-vector extraction.

#!/usr/bin/env python3
"""Build pos/neg persona prompt splits from a persona-vectors trait file.

Reads a ``data_generation/trait_data_extract/<trait>.json`` file from the
persona-vectors repository (which contains only prompts -- no OpenAI required)
and writes ``data/processed/persona/<trait>/{train,val,test}.jsonl`` in the
same schema RQ1 uses.  Each persona-positive ("pos") and persona-negative
("neg") system instruction is folded into the *user* turn, because Gemma-2's
chat template has no system role; the persona-vectors contrast is preserved.

The ``harmful`` label is reused for persona-positive prompts and ``harmless``
for persona-negative ones, so the downstream refusal scripts -- which request
the ``harmful`` split for attribution -- pick up the trait-eliciting prompts
without modification.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _rq1_bootstrap import ROOT  # noqa: F401  (sets import paths)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", default="evil")
    parser.add_argument(
        "--trait-data-path",
        type=Path,
        default=None,
        help="Path to <trait>.json. Defaults to "
        "external/persona_vectors/data_generation/trait_data_extract/<trait>.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to data/processed/persona/<trait>.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=64,
                        help="Pos+neg count target per split; capped by the available pool.")
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--fold-template", default="{instruction}\n\n{question}",
                        help="How the persona instruction is folded into the user turn.")
    return parser.parse_args()


def _load_trait(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "instruction" not in payload or "questions" not in payload:
        raise ValueError(
            f"{path} is not a persona-vectors trait file: expected 'instruction' and 'questions' keys."
        )
    instructions = payload["instruction"]
    if isinstance(instructions, dict):  # single {pos, neg} pair
        instructions = [instructions]
    for pair in instructions:
        if "pos" not in pair or "neg" not in pair:
            raise ValueError("Every 'instruction' entry must contain 'pos' and 'neg' keys.")
    if not payload["questions"]:
        raise ValueError(f"{path} has no questions.")
    return {"instruction": instructions, "questions": payload["questions"]}


def _build_pool(trait_name: str, trait: dict, fold_template: str) -> list[dict]:
    """One row per (instruction pair, question) with folded pos and neg text."""
    pool: list[dict] = []
    for i_idx, pair in enumerate(trait["instruction"]):
        for q_idx, question in enumerate(trait["questions"]):
            base_id = f"{trait_name}-i{i_idx}-q{q_idx}"
            pool.append(
                {
                    "pos": fold_template.format(instruction=pair["pos"], question=question),
                    "neg": fold_template.format(instruction=pair["neg"], question=question),
                    "category": trait_name,
                    "base_id": base_id,
                }
            )
    return pool


def _rows_for_split(split_pool: list[dict], trait_name: str, source: str) -> list[dict]:
    rows: list[dict] = []
    for item in split_pool:
        rows.append({
            "id": f"{item['base_id']}-pos", "instruction": item["pos"],
            "label": "harmful", "category": trait_name, "source": source,
            "persona": "pos",
        })
        rows.append({
            "id": f"{item['base_id']}-neg", "instruction": item["neg"],
            "label": "harmless", "category": trait_name, "source": source,
            "persona": "neg",
        })
    return rows


def main():
    args = parse_args()
    trait_path = args.trait_data_path or (
        ROOT / "external" / "persona_vectors" / "data_generation"
        / "trait_data_extract" / f"{args.trait}.json"
    )
    if not trait_path.exists():
        raise FileNotFoundError(
            f"Trait file not found: {trait_path}\n"
            "Clone persona-vectors once, e.g.:\n"
            "  git clone https://github.com/safety-research/persona_vectors "
            "external/persona_vectors\n"
            "or pass --trait-data-path to a local copy of <trait>.json."
        )
    output_dir = args.output_dir or (ROOT / "data" / "processed" / "persona" / args.trait)

    trait = _load_trait(trait_path)
    pool = _build_pool(args.trait, trait, args.fold_template)
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    # Each pool item yields one pos and one neg prompt, so target sizes are in
    # *prompts*; convert to pool items (pairs) per split.
    sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    pairs_needed = {split: max(1, size // 2) for split, size in sizes.items()}
    total_pairs_needed = sum(pairs_needed.values())
    if total_pairs_needed > len(pool):
        raise RuntimeError(
            f"Requested {total_pairs_needed} instruction/question pairs across splits but the "
            f"{args.trait} pool only has {len(pool)} (instructions x questions). Lower "
            "--train-size/--val-size/--test-size or use a trait file with more questions."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    source = f"persona_vectors:{args.trait}"
    for split in ("train", "val", "test"):
        n_pairs = pairs_needed[split]
        split_pool = pool[cursor : cursor + n_pairs]
        cursor += n_pairs
        rows = _rows_for_split(split_pool, args.trait, source)
        rng.shuffle(rows)
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows ({n_pairs} pos + {n_pairs} neg) to {path}")
    print(f"Persona '{args.trait}' splits ready under {output_dir}")


if __name__ == "__main__":
    main()

"""Native PolyRefuse ingestion and multilingual RQ2 conventions.

This module deliberately reads the official checkout only during dataset
preparation.  Processed JSONL files retain source file, source commit, and a
parallel prompt identifier, so all subsequent direction/circuit jobs are
self-contained and can make matched cross-language comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


POLYREFUSE_LANGUAGES = (
    "en", "de", "es", "fr", "it", "nl", "ja", "pl", "ru", "zh", "ko", "ar", "th", "yo",
)

# Exact Qwen2/2.5 vocabulary IDs supplied by the official PolyRefuse code
# (pipeline/model_utils/qwen2_model.py).  Qwen2.5 7B shares this tokenizer.
OFFICIAL_QWEN_REFUSAL_TOKEN_IDS = {
    "en": (40, 2121),
    "zh": (35946,),
    "de": (17360, 40369),
    "th": (126331, 124430),
    "yo": (25612,),
    "ja": (127748, 128976),
    "ru": (85391, 30174),
    "ko": (132759,),
}

# For languages not given a fixed token list by the authors, these compact
# local refusal starts are tokenized at runtime.  Their exact IDs are written
# to each data/direction metadata file.  They are a selection diagnostic, not
# a claim that every refusal begins with one phrase.
REFUSAL_START_STRINGS = {
    "en": ("I", "As"),
    "de": ("Es", "Ich", "Als"),
    "es": ("Lo", "No", "Como"),
    "fr": ("Je", "Désolé", "En"),
    "it": ("Mi", "Non", "Come"),
    "nl": ("Het", "Ik", "Als"),
    "ja": ("申し訳", "すみません", "私は"),
    "pl": ("Nie", "Przepraszam", "Jako"),
    "ru": ("Я", "Извините", "К"),
    "zh": ("我", "抱歉", "作为"),
    "ko": ("죄송", "저는", "요청"),
    "ar": ("لا", "عذر", "أنا"),
    "th": ("ขอ", "ฉัน", "ไม่"),
    "yo": ("Má", "Mi", "Kò"),
}


def paper_refusal_token_ids(language: str) -> list[int]:
    """Return exactly the Qwen2.5-7B Table-4 token set from Wang et al.

    An empty list is intentional: the paper did not report a Qwen refusal
    token set for every PolyRefuse language, including Arabic.
    """
    if language not in POLYREFUSE_LANGUAGES:
        raise ValueError(f"Unsupported PolyRefuse language {language!r}.")
    return list(OFFICIAL_QWEN_REFUSAL_TOKEN_IDS.get(language, ()))


def refusal_token_ids(tokenizer, language: str) -> list[int]:
    """Return an exploratory, expanded language-local Qwen refusal-token set.

    The official PolyRefuse IDs are retained as seeds.  We also include the
    English Qwen refusal seeds because a multilingual chat model can validly
    refuse a translated prompt in English, then a small pre-specified set of
    local refusal openings.  This helper is retained for exploratory models
    such as Qwen2.5-3B. The main Qwen2.5-7B replication uses
    :func:`paper_refusal_token_ids` without this expansion.
    """
    if language not in POLYREFUSE_LANGUAGES:
        raise ValueError(f"Unsupported PolyRefuse language {language!r}.")
    token_ids: list[int] = list(OFFICIAL_QWEN_REFUSAL_TOKEN_IDS.get(language, ()))
    token_ids.extend(OFFICIAL_QWEN_REFUSAL_TOKEN_IDS["en"])
    for text in REFUSAL_START_STRINGS[language]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            token_ids.append(int(ids[0]))
    return sorted(set(token_ids))


def source_path(external_root: Path, *, label: str, split: str, language: str) -> Path:
    """Locate a release file using the authors' English/non-English layout."""
    if label not in {"harmful", "harmless"}:
        raise ValueError("label must be 'harmful' or 'harmless'.")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test.")
    if language not in POLYREFUSE_LANGUAGES:
        raise ValueError(f"Unsupported PolyRefuse language {language!r}.")
    translated = external_root / "PolyRefuse" / f"{label}_{split}_translated_{language}.json"
    if translated.exists():
        return translated
    # The public PolyRefuse checkout used here is missing
    # ``harmless_test_translated_zh.json`` even though it includes Chinese
    # harmless train/validation and harmful test translations.  For the
    # harmless test side only, fall back to the English paired test sample so
    # dataset preparation can still produce a complete test split.  This is
    # recorded in each processed row's ``source_file`` and therefore in the
    # preparation metadata; the main circuit faithfulness/necessity metrics use
    # harmful test prompts and are unaffected by this fallback.
    if label == "harmless" and split == "test":
        paired_test = external_root / "dataset" / "splits" / "harmless_test_200_sampled.json"
        if paired_test.exists():
            return paired_test
        english_test = external_root / "dataset" / "splits" / "harmless_test.json"
        if english_test.exists():
            return english_test
    if language == "en":
        # The non-English harmless train/validation files were translated from
        # the release's paired 200-example samples, not from the full
        # Alpaca-derived pools.  Use those same English rows so ``parallel_id``
        # genuinely denotes the same underlying instruction across languages.
        paired_sample = external_root / "dataset" / "splits" / f"{label}_{split}_200_sampled.json"
        if paired_sample.exists():
            return paired_sample
        english = external_root / "dataset" / "splits" / f"{label}_{split}.json"
        if english.exists():
            return english
    raise FileNotFoundError(f"Could not locate PolyRefuse {label}/{split}/{language} under {external_root}.")


def load_split(
    external_root: Path,
    *,
    label: str,
    split: str,
    language: str,
) -> tuple[list[dict], Path]:
    """Load a release split and attach stable parallel row identifiers."""
    path = source_path(external_root, label=label, split=split, language=language)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    output: list[dict] = []
    for index, row in enumerate(rows):
        # PolyRefuse train/validation translated files store the translated
        # prompt in ``instruction``.  The test translated files keep the
        # original English prompt in ``instruction`` and put the translated
        # prompt in ``instruction_translated``.  Prefer the translated field
        # for non-English languages when it is present; otherwise the
        # first-token refusal proxy is accidentally evaluated on English text
        # with non-English refusal-token IDs.
        if language != "en" and isinstance(row.get("instruction_translated"), str):
            instruction = row.get("instruction_translated")
            source_text_field = "instruction_translated"
        else:
            instruction = row.get("instruction") or row.get("prompt") or row.get("goal")
            source_text_field = (
                "instruction"
                if isinstance(row.get("instruction"), str)
                else "prompt"
                if isinstance(row.get("prompt"), str)
                else "goal"
            )
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Missing instruction in {path} row {index}.")
        output.append(
            {
                "id": f"polyrefuse-{label}-{split}-{index}",
                "parallel_id": f"polyrefuse-{label}-{split}-{index}",
                "instruction": instruction.strip(),
                "label": label,
                "category": str(row.get("category", "unspecified")),
                "source": "PolyRefuse",
                "source_file": str(path),
                "source_text_field": source_text_field,
                "language": language,
            }
        )
    return output, path


def pairwise_parallel_ids(rows_by_language: dict[str, Iterable[dict]]) -> dict[tuple[str, str], list[str]]:
    """Return deterministic pairwise intersections of processed prompt IDs."""
    ids = {language: {str(row["parallel_id"]) for row in rows} for language, rows in rows_by_language.items()}
    languages = sorted(ids)
    return {
        (left, right): sorted(ids[left] & ids[right])
        for index, left in enumerate(languages)
        for right in languages[index + 1 :]
    }

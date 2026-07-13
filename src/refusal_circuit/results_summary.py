"""Generate a compact Markdown summary of committed RQ1/RQ2 results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs" / "RESULTS_SUMMARY.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _row(path: Path, **conditions: str) -> dict[str, str] | None:
    for row in _read_csv(path):
        if all(str(row.get(key, "")) == str(value) for key, value in conditions.items()):
            return row
    return None


def _float(row: dict[str, str] | None, key: str, default: float | None = None) -> float | None:
    if row is None:
        return default
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _link(path: Path) -> str:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        rel = path
    return f"`{rel}`"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def _inventory() -> str:
    rows: list[list[str]] = []
    results_root = ROOT / "results"
    for child in sorted(results_root.iterdir()) if results_root.exists() else []:
        if not child.is_dir():
            continue
        file_count = 0
        total_bytes = 0
        for dirpath, _, filenames in os.walk(child):
            for filename in filenames:
                path = Path(dirpath) / filename
                file_count += 1
                total_bytes += path.stat().st_size
        rows.append([child.name, str(file_count), f"{total_bytes / (1024 * 1024):.1f} MB"])
    return _table(["Result tree", "Files", "Size"], rows)


def _rq1_summary() -> str:
    base = ROOT / "results" / "gemma-2-2b-it" / "en" / "rq1"
    behavior_path = base / "behavioral_rerun" / "eapig_top_1000_test_n32_gen32.csv"
    faith_path = base / "faithfulness_pilot" / "faithfulness.csv"
    nec_path = base / "faithfulness_pilot" / "necessity.csv"

    behavior_rows = []
    for condition in [
        "normal",
        "refusal_direction_ablated",
        "target_direction_ablated",
        "circuit_only",
        "circuit_ablated",
    ]:
        row = _row(behavior_path, condition=condition)
        behavior_rows.append(
            [
                condition,
                row.get("n_prompts", "—") if row else "—",
                _fmt(_float(row, "refusal_substring_rate")),
                _fmt(_float(row, "refusal_token_logit_margin")),
            ]
        )

    faith = _row(faith_path, method="eapig", requested_k="500")
    nec = _row(nec_path, method="eapig", requested_k="500")
    random_faith = _row(faith_path, method="random_eapig", requested_k="500")
    random_nec = _row(nec_path, method="random_eapig", requested_k="500")

    lines = [
        "## RQ1: Gemma-2-2B-it English refusal circuit",
        "",
        "Goal: find an edge-level circuit that mediates a refusal-direction projection in `google/gemma-2-2b-it`.",
        "",
        "- Main target: residual projection onto the selected refusal direction at layer 14, token position -2.",
        "- Main behavioral rerun: EAP-IG top-1000 circuit, which retained 668 edges.",
        "",
        _table(["Condition", "n", "Refusal substring rate", "Refusal-token logit margin"], behavior_rows),
        "",
        "Projection-level pilot metrics:",
        "",
        _table(
            ["Circuit", "k", "retained edges", "Faithfulness", "Necessity"],
            [
                [
                    "EAP-IG",
                    faith.get("requested_k", "—") if faith else "—",
                    faith.get("retained_edges", "—") if faith else "—",
                    _fmt(_float(faith, "faithfulness")),
                    _fmt(_float(nec, "necessity")),
                ],
                [
                    "matched random",
                    random_faith.get("requested_k", "—") if random_faith else "—",
                    random_faith.get("retained_edges", "—") if random_faith else "—",
                    _fmt(_float(random_faith, "faithfulness")),
                    _fmt(_float(random_nec, "necessity")),
                ],
            ],
        ),
        "",
        "Interpretation: the broader circuit is behaviorally causal; target-site-only ablation is not enough, so the scalar projection is not by itself the whole behavioral bottleneck.",
        "",
        f"Key files: {_link(behavior_path)}, {_link(faith_path)}, {_link(nec_path)}.",
    ]
    return "\n".join(lines)


def _rq2_direction_summary() -> str:
    base = ROOT / "results" / "qwen2.5-7b-instruct" / "polyrefuse"
    cosine_path = base / "comparisons_paper_kl0p2" / "direction_cosine_common_anchor.csv"
    precheck_template = base / "{lang}" / "metric_precheck" / "english_vector" / "shared_metric_precheck.csv"
    cosine_rows = _read_csv(cosine_path)
    cosine_by_lang = {row["language"]: row for row in cosine_rows}
    rows = []
    for lang in ["en", "de", "zh", "th"]:
        precheck = _row(Path(str(precheck_template).format(lang=lang)), prompt_class="harmful")
        rows.append(
            [
                lang,
                _fmt(float(cosine_by_lang.get(lang, {}).get("en", "nan")) if lang in cosine_by_lang else None),
                _fmt(_float(precheck, "projection_drop_clean_minus_corrupt"), 2),
                f"{_fmt(_float(precheck, 'first_token_refusal_log_odds_clean'), 2)} → {_fmt(_float(precheck, 'first_token_refusal_log_odds_corrupt'), 2)}",
            ]
        )
    return "\n".join(
        [
            "## RQ2: Qwen2.5-7B-Instruct multilingual refusal circuits",
            "",
            "Main target used for circuit comparison:",
            "",
            r"\[M_{\mathrm{EN}}(x)=h_{15,-2}(x)^\top \hat r_{\mathrm{en}}\]",
            "",
            "Direction/vector sanity checks:",
            "",
            _table(
                ["Language", "cosine with English anchor", "harmful projection drop", "refusal log-odds clean → corrupt"],
                rows,
            ),
        ]
    )


def _rq2_local_summary() -> str:
    base = ROOT / "results" / "qwen2.5-7b-instruct" / "polyrefuse"
    rows = []
    for lang in ["en", "de", "zh", "th"]:
        root = base / lang / "circuit_evaluation_english_vector_heldout_offset32_k1000_seeds5"
        faith = _row(root / "faithfulness.csv", method="eapig", requested_k="1000")
        nec = _row(root / "necessity.csv", method="eapig", requested_k="1000")
        rows.append(
            [
                lang,
                faith.get("retained_edges", "—") if faith else "—",
                _fmt(_float(faith, "faithfulness")),
                _fmt(_float(nec, "necessity")),
                _fmt(_float(faith, "circuit_ablated_logit_margin")),
            ]
        )
    return "\n".join(
        [
            "Local held-out circuit evaluations at k=1000:",
            "",
            _table(["Language", "edges", "Faithfulness", "Necessity", "circuit-ablated logit margin"], rows),
        ]
    )


def _rq2_overlap_summary() -> str:
    base = ROOT / "results" / "qwen2.5-7b-instruct" / "polyrefuse"
    overlap_path = base / "comparisons_english_vector" / "eapig_circuit_structural_overlap.csv"
    corr_path = base / "comparisons_english_vector" / "eapig_edge_score_correlations.csv"
    overlap_rows = []
    for row in _read_csv(overlap_path):
        if row.get("requested_k") == "1000" and row.get("baseline") == "observed":
            overlap_rows.append(
                [
                    f"{row['left_language']}–{row['right_language']}",
                    row["intersection_edges"],
                    row["union_edges"],
                    _fmt(float(row["iou"])),
                    _fmt(float(row["overlap_coefficient"])),
                ]
            )
    corr_rows = []
    for row in _read_csv(corr_path):
        corr_rows.append(
            [
                f"{row['left_language']}–{row['right_language']}",
                _fmt(float(row["spearman_score_correlation"])),
                _fmt(float(row["pearson_score_correlation"])),
            ]
        )
    return "\n".join(
        [
            "Cross-language structural overlap for independently discovered circuits, all using the same English anchor:",
            "",
            _table(["Pair", "shared edges", "union edges", "IoU", "overlap coefficient"], overlap_rows),
            "",
            "Full edge-score correlations:",
            "",
            _table(["Pair", "Spearman", "Pearson"], corr_rows),
        ]
    )


def _rq2_shared_summary() -> str:
    base = ROOT / "results" / "qwen2.5-7b-instruct" / "polyrefuse" / "shared_circuit_evaluation_english_vector"
    rows = []
    for family, label in [
        ("intersection_en_de_zh_th", "strict intersection"),
        ("majority_at_least_3_of_4", "majority ≥3/4"),
        ("consensus_mean_scores", "consensus"),
    ]:
        for lang in ["en", "de", "zh", "th"]:
            root = base / family / lang
            faith = _row(root / "faithfulness.csv", method="eapig", requested_k="1000")
            nec = _row(root / "necessity.csv", method="eapig", requested_k="1000")
            if faith:
                rows.append(
                    [
                        label,
                        lang,
                        faith.get("retained_edges", "—"),
                        _fmt(_float(faith, "faithfulness")),
                        _fmt(_float(nec, "necessity")),
                    ]
                )
    return "\n".join(
        [
            "Shared-core / consensus circuit evaluations at k=1000:",
            "",
            _table(["Circuit family", "Language", "edges", "Faithfulness", "Necessity"], rows),
        ]
    )


def _rq2_behavior_summary() -> str:
    base = ROOT / "results" / "qwen2.5-7b-instruct" / "polyrefuse"
    wg_rows = []
    for lang in ["en", "de", "zh", "th"]:
        path = base / "wildguard_behavioral_validation_english_vector" / "consensus" / lang / "k1000_offset32_n32_gen8" / "wildguard_aggregate.csv"
        values = {}
        for condition in ["normal", "direction_ablated", "circuit_only", "circuit_ablated"]:
            values[condition] = _float(_row(path, label="harmful", condition=condition), "refusal_rate")
        wg_rows.append(
            [
                lang,
                _fmt(values["normal"]),
                _fmt(values["direction_ablated"]),
                _fmt(values["circuit_only"]),
                _fmt(values["circuit_ablated"]),
            ]
        )
    large_k_rows = []
    for k in ["5000", "10000", "20000", "50000"]:
        path = base / "en" / "behavioral_validation_projection_large_k" / f"k{k}_offset32_n32_gen8" / "generation_substring_rates.csv"
        large_k_rows.append(
            [
                k,
                _fmt(_float(_row(path, label="harmful", condition="normal"), "refusal_substring_rate")),
                _fmt(_float(_row(path, label="harmful", condition="circuit_only"), "refusal_substring_rate")),
                _fmt(_float(_row(path, label="harmful", condition="circuit_ablated"), "refusal_substring_rate")),
            ]
        )
    return "\n".join(
        [
            "Behavioral validation:",
            "",
            "WildGuard refusal rates for multilingual consensus k=1000 circuits:",
            "",
            _table(
                ["Language", "normal", "direction ablated", "circuit only", "circuit ablated"],
                wg_rows,
            ),
            "",
            "English larger-k projection circuit generation substring checks:",
            "",
            _table(["k", "normal", "circuit only", "circuit ablated"], large_k_rows),
        ]
    )


def _logit_summary() -> str:
    base = ROOT / "results" / "qwen2.5-7b-instruct" / "polyrefuse" / "en" / "behavioral_validation_logit_pilot"
    rows = []
    for k in ["1000", "2000", "5000", "10000", "20000", "50000"]:
        path = base / f"k{k}_offset32_n16_gen4" / "behavioral_metrics.csv"
        metrics = {
            row.get("condition"): row
            for row in _read_csv(path)
            if row.get("label") == "harmful"
        }
        if metrics:
            rows.append(
                [
                    k,
                    _fmt(_float(metrics.get("normal"), "first_token_refusal_log_odds_mean")),
                    _fmt(_float(metrics.get("circuit_only"), "first_token_refusal_log_odds_mean")),
                    _fmt(_float(metrics.get("circuit_ablated"), "first_token_refusal_log_odds_mean")),
                    _fmt(_float(metrics.get("circuit_ablated"), "first_token_refusal_log_odds_gt0_rate")),
                ]
            )
    if not rows:
        return ""
    return "\n".join(
        [
            "English logit-target pilot:",
            "",
            _table(["k", "normal log-odds", "circuit-only log-odds", "circuit-ablated log-odds", "ablated gt0 rate"], rows),
        ]
    )


def build_summary() -> str:
    return "\n\n".join(
        [
            "# Results summary",
            "This file is generated from committed result artifacts under `results/`.",
            "## Result inventory",
            _inventory(),
            _rq1_summary(),
            _rq2_direction_summary(),
            _rq2_local_summary(),
            _rq2_overlap_summary(),
            _rq2_shared_summary(),
            _rq2_behavior_summary(),
            _logit_summary(),
            "## Current high-level claim",
            "RQ1 showed a sparse, behaviorally causal refusal circuit in Gemma-2-2B-it. RQ2 extended this to Qwen2.5-7B-Instruct and found strong evidence for a shared multilingual circuit that builds a common refusal-direction representation across English, German, Chinese, and Thai. Behavioral tests support the circuit result, especially for consensus k=1000 and larger English projection circuits.",
        ]
    ).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_summary())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

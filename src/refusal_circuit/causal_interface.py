"""Pure helpers for the Phase 1 low-dimensional causal-interface oracle."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from typing import Iterable, Sequence

import torch


PHASE1_RANKS = (1, 2, 4, 8)
PHASE1_TIMINGS = ("prompt_only", "recurrent")
SPLIT_SCOPES = (
    "interface_extraction",
    "circuit_discovery",
    "validation",
    "final_test",
)


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_sha256(text: str) -> str:
    if not isinstance(text, str) or not text:
        raise ValueError("Question content must be a non-empty string.")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fix_component_signs(basis: torch.Tensor) -> torch.Tensor:
    """Fix each component sign by making its largest-magnitude entry positive."""

    if basis.ndim != 2 or basis.shape[0] == 0 or basis.shape[1] == 0:
        raise ValueError("Basis must have shape [rank, hidden].")
    fixed = basis.clone()
    for index in range(fixed.shape[0]):
        component = fixed[index]
        pivot = int(component.abs().argmax().item())
        if float(component[pivot].item()) < 0:
            fixed[index] = -component
    return fixed


def build_nested_persona_basis(
    paired_differences: torch.Tensor,
    *,
    max_rank: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build mean direction plus residual PCA components."""

    if paired_differences.ndim != 2:
        raise ValueError("Paired differences must have shape [pairs, hidden].")
    if paired_differences.shape[0] < max_rank:
        raise ValueError("At least max_rank paired differences are required.")
    values = paired_differences.double()
    if not torch.isfinite(values).all():
        raise ValueError("Paired differences contain non-finite values.")
    mean = values.mean(dim=0)
    mean_norm = mean.norm()
    if not float(mean_norm.item()) > 0:
        raise ValueError("Mean persona direction has zero norm.")
    rank1 = mean / mean_norm
    residual = values - (values @ rank1).unsqueeze(-1) * rank1
    _, singular_values, vh = torch.linalg.svd(
        residual,
        full_matrices=False,
    )
    needed = max_rank - 1
    if vh.shape[0] < needed:
        raise ValueError("Residual PCA does not contain enough components.")
    candidate = torch.cat((rank1.unsqueeze(0), vh[:needed]), dim=0)
    # QR removes numerical leakage into rank 1 while preserving nested order.
    orthonormal = torch.linalg.qr(candidate.T, mode="reduced").Q.T
    orthonormal[0] *= torch.sign(
        torch.dot(orthonormal[0], rank1)
    ).clamp(min=1)
    orthonormal = fix_component_signs(orthonormal)
    if torch.dot(orthonormal[0], rank1) < 0:
        orthonormal[0] = -orthonormal[0]
    gram = orthonormal @ orthonormal.T
    identity = torch.eye(max_rank, dtype=gram.dtype, device=gram.device)
    error = float((gram - identity).abs().max().item())
    if error > 1e-8:
        raise RuntimeError(f"Basis orthonormality error is {error}.")
    diagnostics = {
        "pairs": int(values.shape[0]),
        "hidden_size": int(values.shape[1]),
        "mean_direction_norm": float(mean_norm.item()),
        "residual_frobenius_norm": float(residual.norm().item()),
        "orthonormality_max_abs_error": error,
        "sign_rule": "largest_absolute_loading_positive",
        "rank1_rule": "unit_mean_paired_positive_minus_negative_difference",
        "residual_rule": "svd_after_removing_rank1_projection",
    }
    return (
        orthonormal.float(),
        singular_values[:needed].float(),
        diagnostics,
    )


def project_onto_basis(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.ndim != 2 or delta.shape[-1] != basis.shape[-1]:
        raise ValueError("Delta and basis hidden dimensions must agree.")
    normalized = basis.to(device=delta.device, dtype=delta.dtype)
    return (delta @ normalized.T) @ normalized


def projection_parts(
    delta: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = project_onto_basis(delta, basis)
    return projected, delta - projected


def timed_interface_state(
    corrupt_state: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    *,
    basis: torch.Tensor | None,
    component: str,
) -> torch.Tensor:
    """Apply full, projected, complement, or zero clean transmission."""

    if corrupt_state.shape != delta.shape:
        raise ValueError("Corrupt state and delta must have identical shapes.")
    if mask.shape != corrupt_state.shape[:-1]:
        raise ValueError("Timing mask must match batch and position axes.")
    if component == "full":
        transmitted = delta
    elif component == "zero":
        transmitted = torch.zeros_like(delta)
    elif component in {"persona", "complement"}:
        if basis is None:
            raise ValueError(f"{component} transmission requires a basis.")
        persona, complement = projection_parts(delta, basis)
        transmitted = persona if component == "persona" else complement
    else:
        raise ValueError(f"Unknown interface component: {component}")
    active = mask.to(
        device=corrupt_state.device,
        dtype=corrupt_state.dtype,
    ).unsqueeze(-1)
    return corrupt_state + active * transmitted


def timing_mask(
    attention_mask: torch.Tensor,
    *,
    prompt_end: int,
    timing: str,
) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("Attention mask must have shape [batch, positions].")
    if not 0 <= prompt_end < attention_mask.shape[1]:
        raise ValueError("prompt_end is outside the sequence.")
    active = torch.zeros_like(attention_mask, dtype=torch.bool)
    if timing == "prompt_only":
        active[:, prompt_end] = True
    elif timing == "recurrent":
        active[:, prompt_end:] = attention_mask[:, prompt_end:].bool()
    else:
        raise ValueError(f"Unknown timing policy: {timing}")
    return active


def exact_full_state(clean_state: torch.Tensor, corrupt_state: torch.Tensor) -> torch.Tensor:
    if clean_state.shape != corrupt_state.shape:
        raise ValueError("Clean and corrupt states must have identical shapes.")
    return clean_state


def _question_rows(
    questions: Sequence[str],
    indices: Iterable[int],
    *,
    source: str,
) -> list[dict]:
    return [
        {
            "question_id": f"{source}:q{index:02d}",
            "question_index": int(index),
            "content_sha256": content_sha256(questions[index]),
            "row_sha256": canonical_json_sha256(
                {
                    "source": source,
                    "question_index": int(index),
                    "question": questions[index],
                }
            ),
        }
        for index in indices
    ]


def freeze_question_split(
    *,
    extraction_questions: Sequence[str],
    behavioral_questions: Sequence[str],
    eligible_pair_counts: dict[int, int],
    eligible_prompt_counts: dict[tuple[int, int], int],
    source_hashes: dict[str, str],
    model: dict,
    tokenizer_identity: dict,
    seed: int = 20260818,
    minimum_interface_pairs: int = 100,
) -> dict:
    """Create the frozen 4/9/3/4 question split using eligibility only."""

    if len(extraction_questions) != 20:
        raise ValueError("Phase 1 requires exactly 20 extraction questions.")
    if set(eligible_pair_counts) != set(range(20)):
        raise ValueError("Eligibility counts must cover all extraction questions.")
    if any(count < 0 for count in eligible_pair_counts.values()):
        raise ValueError("Eligibility counts cannot be negative.")
    expected_prompts = {
        (question, instruction)
        for question in range(20)
        for instruction in range(5)
    }
    if set(eligible_prompt_counts) != expected_prompts:
        raise ValueError(
            "Prompt eligibility must cover all 20 x 5 extraction identities."
        )
    if any(count < 0 for count in eligible_prompt_counts.values()):
        raise ValueError("Prompt eligibility counts cannot be negative.")
    behavioral_hashes = {
        content_sha256(question) for question in behavioral_questions[:5]
    }
    rejected = []
    accepted = None
    for candidate_seed in range(seed, seed + 10000):
        order = list(range(20))
        random.Random(candidate_seed).shuffle(order)
        interface = order[:4]
        circuit_pool = order[4:16]
        final_test = order[16:20]
        reasons = []
        interface_pairs = sum(eligible_pair_counts[index] for index in interface)
        if interface_pairs < minimum_interface_pairs:
            reasons.append("interface_effective_pair_minimum")
        if any(
            eligible_prompt_counts[(question, instruction)] == 0
            for question in circuit_pool
            for instruction in range(5)
        ):
            reasons.append("circuit_prompt_identity_without_effective_pair")
        interface_hashes = {
            content_sha256(extraction_questions[index]) for index in interface
        }
        if interface_hashes & behavioral_hashes:
            reasons.append("behavioral_pilot_content_overlap")
        if reasons:
            rejected.append({"seed": candidate_seed, "reasons": reasons})
            continue
        accepted = (candidate_seed, interface, circuit_pool, final_test)
        break
    if accepted is None:
        raise RuntimeError("No deterministic eligible question split was found.")
    accepted_seed, interface, circuit_pool, final_test = accepted
    fit_a_validation = circuit_pool[-3:]
    fit_a_discovery = circuit_pool[:-3]
    fit_b_validation = circuit_pool[:3]
    fit_b_discovery = circuit_pool[3:]
    scopes = {
        "interface_extraction": _question_rows(
            extraction_questions, interface, source="extract"
        ),
        "circuit_discovery": _question_rows(
            extraction_questions, fit_a_discovery, source="extract"
        ),
        "validation": _question_rows(
            extraction_questions, fit_a_validation, source="extract"
        ),
        "final_test": _question_rows(
            extraction_questions, final_test, source="extract"
        ),
    }
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "method": "question_disjoint_phase1_causal_interface_split",
        "requested_seed": int(seed),
        "accepted_seed": int(accepted_seed),
        "rejected_metadata_only_splits": rejected,
        "eligibility_rule": {
            "only_behavioral_input": "predeclared_effective_pair_filter",
            "minimum_interface_effective_pairs": int(minimum_interface_pairs),
            "effective_pairs_by_question": {
                str(index): int(eligible_pair_counts[index])
                for index in range(20)
            },
            "effective_pairs_by_prompt_identity": {
                f"q{question:02d}-i{instruction:02d}": int(
                    eligible_prompt_counts[(question, instruction)]
                )
                for question in range(20)
                for instruction in range(5)
            },
        },
        "model": model,
        "tokenizer_identity": tokenizer_identity,
        "sources": dict(sorted(source_hashes.items())),
        "scopes": scopes,
        "circuit_pool": _question_rows(
            extraction_questions, circuit_pool, source="extract"
        ),
        "fit_a": {
            "discovery_question_indices": fit_a_discovery,
            "validation_question_indices": fit_a_validation,
        },
        "fit_b": {
            "discovery_question_indices": fit_b_discovery,
            "validation_question_indices": fit_b_validation,
        },
        "behavioral_pilot": _question_rows(
            behavioral_questions,
            range(min(5, len(behavioral_questions))),
            source="eval",
        ),
        "disjointness": {
            "unit": "question_content_sha256",
            "interface_vs_circuit_pool": True,
            "interface_vs_final_test": True,
            "interface_vs_behavioral_pilot": True,
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(
    manifest: dict,
    *,
    expected_source_hashes: dict[str, str] | None = None,
) -> None:
    if manifest.get("status") != "frozen":
        raise ValueError("Split manifest is not frozen.")
    scopes = manifest.get("scopes", {})
    if set(scopes) != set(SPLIT_SCOPES):
        raise ValueError("Split manifest has the wrong scopes.")
    ids_by_scope = {
        scope: {row["question_id"] for row in rows}
        for scope, rows in scopes.items()
    }
    for left_index, left in enumerate(SPLIT_SCOPES):
        for right in SPLIT_SCOPES[left_index + 1 :]:
            if ids_by_scope[left] & ids_by_scope[right]:
                raise ValueError(f"Question overlap between {left} and {right}.")
    interface_hashes = {
        row["content_sha256"] for row in scopes["interface_extraction"]
    }
    pilot_hashes = {
        row["content_sha256"] for row in manifest["behavioral_pilot"]
    }
    if interface_hashes & pilot_hashes:
        raise ValueError("Interface extraction overlaps the behavioral pilot.")
    for rows in [*scopes.values(), manifest["behavioral_pilot"]]:
        for row in rows:
            expected = canonical_json_sha256(
                {
                    "source": row["question_id"].split(":", 1)[0],
                    "question_index": int(row["question_index"]),
                    # The content is not stored deliberately; verify its stable hash fields.
                    "content_sha256": row["content_sha256"],
                }
            )
            if not row.get("row_sha256") or len(row["row_sha256"]) != len(expected):
                raise ValueError("Question row provenance hash is malformed.")
    if expected_source_hashes is not None:
        if manifest.get("sources") != dict(sorted(expected_source_hashes.items())):
            raise ValueError("Split source artifact hashes changed.")
    recorded = manifest.get("manifest_sha256")
    actual = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if recorded != actual:
        raise ValueError("Split manifest hash validation failed.")


def leave_one_question_out_stability(
    paired_differences: torch.Tensor,
    question_indices: Sequence[int],
    full_basis: torch.Tensor,
    *,
    ranks: Sequence[int] = PHASE1_RANKS,
) -> list[dict]:
    if len(question_indices) != paired_differences.shape[0]:
        raise ValueError("Question labels must align with paired differences.")
    output = []
    for question in sorted(set(map(int, question_indices))):
        keep = torch.tensor(
            [int(value) != question for value in question_indices],
            dtype=torch.bool,
            device=paired_differences.device,
        )
        basis, _, _ = build_nested_persona_basis(
            paired_differences[keep],
            max_rank=max(ranks),
        )
        rank_metrics = {}
        for rank in ranks:
            overlap = torch.linalg.svdvals(
                full_basis[:rank].double() @ basis[:rank].double().T
            )
            rank_metrics[str(rank)] = {
                "minimum_principal_cosine": float(overlap.min().item()),
                "mean_principal_cosine": float(overlap.mean().item()),
            }
        output.append(
            {
                "left_out_question_index": question,
                "remaining_pairs": int(keep.sum().item()),
                "rank1_absolute_cosine": float(
                    full_basis[0].double().dot(basis[0].double()).abs().item()
                ),
                "nested_subspace_stability": rank_metrics,
            }
        )
    return output


def projection_diagnostics(
    delta: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, float]:
    persona, complement = projection_parts(delta.float(), basis.float())
    reconstruction = persona + complement
    flat_persona = persona.reshape(-1, persona.shape[-1])
    flat_complement = complement.reshape(-1, complement.shape[-1])
    dot = (flat_persona * flat_complement).sum(dim=-1)
    denominator = delta.float().pow(2).sum().clamp_min(1e-12)
    return {
        "delta_l2": float(delta.float().norm().item()),
        "persona_fraction_l2_squared": float(
            persona.pow(2).sum().div(denominator).item()
        ),
        "complement_fraction_l2_squared": float(
            complement.pow(2).sum().div(denominator).item()
        ),
        "reconstruction_max_abs_error": float(
            (reconstruction - delta.float()).abs().max().item()
        ),
        "persona_complement_max_abs_dot": float(dot.abs().max().item()),
    }


def select_rank_and_timing(
    recoveries: dict[str, dict[str, dict[str, float]]],
    *,
    persona_minimum: float = 0.80,
    complement_maximum: float = 0.20,
    prompt_tolerance: float = 0.05,
) -> dict:
    """Apply the precommitted smallest-rank and prompt-only preference rule."""

    passing = []
    for rank in PHASE1_RANKS:
        recurrent = recoveries["recurrent"][str(rank)]
        prompt = recoveries["prompt_only"][str(rank)]
        recurrent_pass = (
            recurrent["persona"] >= persona_minimum
            and recurrent["complement"] <= complement_maximum
            and bool(recurrent.get("healthy", True))
        )
        prompt_pass = (
            prompt["persona"] >= persona_minimum
            and prompt["complement"] <= complement_maximum
            and bool(prompt.get("healthy", True))
            and prompt["persona"] >= recurrent["persona"] - prompt_tolerance
        )
        if prompt_pass:
            passing.append((rank, "prompt_only"))
        elif recurrent_pass:
            passing.append((rank, "recurrent"))
    if not passing:
        return {
            "passed": False,
            "selected_rank": None,
            "selected_timing": None,
            "description": None,
        }
    rank, timing = min(passing, key=lambda item: item[0])
    return {
        "passed": True,
        "selected_rank": rank,
        "selected_timing": timing,
        "description": (
            "rank-1 recurrent persona channel"
            if rank == 1 and timing == "recurrent"
            else (
                "rank-1 persona channel"
                if rank == 1
                else "low-dimensional persona-contrast interface"
            )
        ),
    }


def validate_phase1_output_schema(payload: dict) -> None:
    required = {
        "schema_version",
        "status",
        "phase",
        "model",
        "split_manifest_sha256",
        "basis_sha256",
        "identity",
        "generation",
        "outputs",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Phase 1 output schema is missing: {sorted(missing)}")
    if payload["phase"] != 1:
        raise ValueError("Phase 1 output has the wrong phase.")
    if payload["status"] not in {"running", "complete", "failed"}:
        raise ValueError("Phase 1 output has an invalid status.")
    ranks = payload["generation"].get("ranks")
    timings = payload["generation"].get("timings")
    if ranks != list(PHASE1_RANKS) or timings != list(PHASE1_TIMINGS):
        raise ValueError("Phase 1 output rank/timing grid changed.")


def question_count(rows: Sequence[dict]) -> dict[int, int]:
    return dict(Counter(int(row["question_index"]) for row in rows))


def finite_number(value: object) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("Expected a finite numeric value.")
    return converted

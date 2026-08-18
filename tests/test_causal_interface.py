from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
import torch

from refusal_circuit.causal_interface import (
    PHASE1_RANKS,
    PHASE1_TIMINGS,
    build_nested_persona_basis,
    canonical_json_sha256,
    exact_full_state,
    fix_component_signs,
    freeze_question_split,
    project_onto_basis,
    projection_parts,
    select_rank_and_timing,
    timed_interface_state,
    timing_mask,
    validate_phase1_output_schema,
    validate_split_manifest,
)


def _differences() -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(40, 16, generator=generator, dtype=torch.float64)
    values[:, 0] += 3.0
    values[:, 1] -= 1.0
    return values


def test_basis_is_orthonormal_sign_fixed_and_nested():
    basis, singular_values, diagnostics = build_nested_persona_basis(
        _differences(), max_rank=8
    )
    assert basis.shape == (8, 16)
    assert singular_values.shape == (7,)
    assert torch.allclose(basis @ basis.T, torch.eye(8), atol=1e-6)
    for component in basis:
        pivot = int(component.abs().argmax())
        assert component[pivot] >= 0
    assert diagnostics["orthonormality_max_abs_error"] < 1e-8
    for rank in PHASE1_RANKS:
        assert torch.equal(basis[:rank], basis[:rank].clone())


def test_sign_fixing_is_deterministic():
    components = torch.tensor(
        [[-3.0, 1.0, 0.0], [0.0, -1.0, 2.0]],
        dtype=torch.float64,
    )
    first = fix_component_signs(components)
    second = fix_component_signs(components)
    assert torch.equal(first, second)
    assert torch.equal(first[0], -components[0])
    assert torch.equal(first[1], components[1])


def test_projection_and_complement_reconstruct_identity_and_are_orthogonal():
    basis, _, _ = build_nested_persona_basis(_differences(), max_rank=8)
    delta = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(3))
    persona, complement = projection_parts(delta, basis[:4])
    assert torch.allclose(persona + complement, delta, atol=1e-5)
    dot = (persona * complement).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5)
    assert torch.allclose(project_onto_basis(delta, basis[:4]), persona)


def test_zero_and_full_interface_endpoints():
    corrupt = torch.randn(1, 3, 4)
    delta = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    zero = timed_interface_state(
        corrupt, delta, mask, basis=None, component="zero"
    )
    full = timed_interface_state(
        corrupt, delta, mask, basis=None, component="full"
    )
    assert torch.equal(zero, corrupt)
    assert torch.allclose(full, corrupt + delta)


def test_prompt_only_and_recurrent_timing_masks():
    attention = torch.ones(1, 6, dtype=torch.long)
    prompt = timing_mask(attention, prompt_end=3, timing="prompt_only")
    recurrent = timing_mask(attention, prompt_end=3, timing="recurrent")
    assert torch.equal(
        prompt, torch.tensor([[False, False, False, True, False, False]])
    )
    assert torch.equal(
        recurrent, torch.tensor([[False, False, False, True, True, True]])
    )


def _split_manifest(seed: int = 20260818) -> dict:
    return freeze_question_split(
        extraction_questions=[f"extract question {index}" for index in range(20)],
        behavioral_questions=[f"pilot question {index}" for index in range(20)],
        eligible_pair_counts={index: 40 for index in range(20)},
        eligible_prompt_counts={
            (question, instruction): 8
            for question in range(20)
            for instruction in range(5)
        },
        source_hashes={"a.jsonl": "a" * 64, "b.jsonl": "b" * 64},
        model={"name": "model", "revision": "revision"},
        tokenizer_identity={"repository": "model", "revision": "revision"},
        seed=seed,
        minimum_interface_pairs=100,
    )


def test_split_is_deterministic_and_question_disjoint():
    first = _split_manifest()
    second = _split_manifest()
    assert first == second
    validate_split_manifest(first)
    scope_ids = [
        {row["question_id"] for row in first["scopes"][scope]}
        for scope in (
            "interface_extraction",
            "circuit_discovery",
            "validation",
            "final_test",
        )
    ]
    assert [len(rows) for rows in scope_ids] == [4, 9, 3, 4]
    for index, left in enumerate(scope_ids):
        for right in scope_ids[index + 1 :]:
            assert left.isdisjoint(right)
    interface_hashes = {
        row["content_sha256"]
        for row in first["scopes"]["interface_extraction"]
    }
    pilot_hashes = {
        row["content_sha256"] for row in first["behavioral_pilot"]
    }
    assert interface_hashes.isdisjoint(pilot_hashes)


def test_split_provenance_hash_validation_rejects_changes():
    manifest = _split_manifest()
    validate_split_manifest(
        manifest,
        expected_source_hashes={"a.jsonl": "a" * 64, "b.jsonl": "b" * 64},
    )
    with pytest.raises(ValueError, match="source artifact hashes"):
        validate_split_manifest(
            manifest,
            expected_source_hashes={"a.jsonl": "c" * 64, "b.jsonl": "b" * 64},
        )
    changed = json.loads(json.dumps(manifest))
    changed["accepted_seed"] += 1
    with pytest.raises(ValueError, match="manifest hash"):
        validate_split_manifest(changed)


def test_exact_full_state_endpoint_returns_clean_state():
    clean = torch.randn(1, 4, 8)
    corrupt = torch.randn(1, 4, 8)
    endpoint = exact_full_state(clean, corrupt)
    assert endpoint is clean
    assert torch.equal(endpoint, clean)


class _CharacterTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    @staticmethod
    def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return (
            f"S:{conversation[0]['content']}|"
            f"U:{conversation[1]['content']}|A:"
        )


@pytest.mark.parametrize(
    ("positive_system", "negative_system", "expected_padded"),
    [
        ("short", "a much longer system", "positive"),
        ("a much longer system", "short", "negative"),
    ],
)
def test_two_sided_token_alignment(positive_system, negative_system, expected_padded):
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts/persona"
    sys.path.insert(0, str(scripts))
    try:
        namespace = runpy.run_path(
            str(scripts / "persona_run_upstream_contrastive_eapig.py"),
            run_name="alignment_test",
        )
    finally:
        sys.path.remove(str(scripts))
    positive, negative, metadata = namespace["_align_system_prompt_pair"](
        _CharacterTokenizer(),
        question="question",
        positive_system_prompt=positive_system,
        negative_system_prompt=negative_system,
    )
    assert len(_CharacterTokenizer.encode(positive)) == len(
        _CharacterTokenizer.encode(negative)
    )
    assert metadata["padded_polarity"] == expected_padded


def test_rank_timing_selection_rule_prefers_smallest_and_prompt_when_close():
    recoveries = {
        timing: {
            str(rank): {"persona": 0.1, "complement": 0.9, "healthy": True}
            for rank in PHASE1_RANKS
        }
        for timing in PHASE1_TIMINGS
    }
    recoveries["recurrent"]["2"] = {
        "persona": 0.84,
        "complement": 0.18,
        "healthy": True,
    }
    recoveries["prompt_only"]["2"] = {
        "persona": 0.80,
        "complement": 0.19,
        "healthy": True,
    }
    recoveries["recurrent"]["4"] = {
        "persona": 0.95,
        "complement": 0.05,
        "healthy": True,
    }
    recoveries["prompt_only"]["4"] = {
        "persona": 0.95,
        "complement": 0.05,
        "healthy": True,
    }
    selected = select_rank_and_timing(recoveries)
    assert selected["passed"]
    assert selected["selected_rank"] == 2
    assert selected["selected_timing"] == "prompt_only"
    assert selected["description"] == "low-dimensional persona-contrast interface"


def test_phase1_output_schema_and_cli_registration():
    payload = {
        "schema_version": 1,
        "status": "complete",
        "phase": 1,
        "model": {"name": "model"},
        "split_manifest_sha256": "a" * 64,
        "basis_sha256": "b" * 64,
        "identity": {"passed": True},
        "generation": {
            "ranks": list(PHASE1_RANKS),
            "timings": list(PHASE1_TIMINGS),
        },
        "outputs": [],
    }
    validate_phase1_output_schema(payload)
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "scripts/rq.py"), run_name="rq_test")
    commands = namespace["COMMANDS"]
    assert "persona.freeze-causal-interface-split" in commands
    assert "persona.run-causal-interface-phase1" in commands
    assert "persona.analyze-causal-interface-phase1" in commands
    payload["generation"]["ranks"] = [1]
    with pytest.raises(ValueError, match="rank/timing grid"):
        validate_phase1_output_schema(payload)


def test_condition_microbatch_slices_are_ordered_complete_and_bounded():
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts/persona"
    sys.path.insert(0, str(scripts))
    try:
        namespace = runpy.run_path(
            str(scripts / "persona_run_causal_interface_phase1.py"),
            run_name="microbatch_test",
        )
    finally:
        sys.path.remove(str(scripts))
    slices = namespace["_microbatch_slices"]
    assert slices(7, 2) == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert slices(0, 2) == []
    covered = [
        index
        for start, end in slices(20, 3)
        for index in range(start, end)
    ]
    assert covered == list(range(20))
    assert all(end - start <= 3 for start, end in slices(20, 3))
    with pytest.raises(ValueError, match="positive"):
        slices(1, 0)


def test_residual_only_backend_disables_head_expanding_trace_settings():
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts/persona"
    sys.path.insert(0, str(scripts))
    try:
        namespace = runpy.run_path(
            str(scripts / "persona_run_causal_interface_phase1.py"),
            run_name="backend_test",
        )
    finally:
        sys.path.remove(str(scripts))

    class Config:
        use_split_qkv_input = True
        use_attn_result = True
        use_attn_in = True
        ungroup_grouped_query_attention = True

    class Model:
        cfg = Config()

        def set_use_split_qkv_input(self, value):
            self.cfg.use_split_qkv_input = value

        def set_use_attn_result(self, value):
            self.cfg.use_attn_result = value

        def set_ungroup_grouped_query_attention(self, value):
            self.cfg.ungroup_grouped_query_attention = value

    settings = namespace["_configure_residual_only_forward"](Model())
    assert settings == {
        "use_split_qkv_input": False,
        "use_attn_result": False,
        "use_attn_in": False,
        "ungroup_grouped_query_attention": False,
    }


def test_canonical_json_hash_is_order_independent():
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )

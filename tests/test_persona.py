from __future__ import annotations

import unittest
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from refusal_circuit.eapig import (
    interpolation_path,
    response_logprob_values,
    response_prediction_mask,
    steering_coefficient_path,
)
from refusal_circuit.circuit import (
    _construction_hooks,
    _source_cache_hooks,
    _sparse_circuit_only_construction_hooks,
)
from refusal_circuit.graph import (
    circuit_connectivity,
    make_steering_to_logits_graph,
    paper_greedy_edges,
    sorted_edges,
    validate_intervention_connectivity,
)
from refusal_circuit.model import memory_efficient_complex_attn_linear
from refusal_circuit.persona import (
    COHERENCE_PROMPT,
    ablate_direction_from_inputs,
    ablate_direction_from_output,
    add_direction_to_inputs,
    add_direction_to_output,
    aggregate_numeric_logprobs,
    align_effective_pairs,
    behavioral_circuit_summary,
    edit_direction_output,
    extraction_specs,
    hf_hidden_state_to_tl_hook,
    make_system_prompt,
    match_vector_norm,
    pair_is_effective,
    select_prompt_direction_candidate,
    steer_model_output,
    steer_model_output_timed,
    token_ids_sha256,
)


class PersonaProtocolTests(unittest.TestCase):
    def setUp(self):
        self.trait_data = {
            "questions": [f"question {index}" for index in range(20)],
            "instruction": [
                {"pos": f"positive {index}", "neg": f"negative {index}"}
                for index in range(5)
            ],
        }

    def test_official_factorial_design_and_pairing(self):
        positive = list(
            extraction_specs(
                self.trait_data,
                polarity="pos",
                assistant_name="sycophantic",
                rollouts=10,
            )
        )
        negative = list(
            extraction_specs(
                self.trait_data,
                polarity="neg",
                assistant_name="helpful",
                rollouts=10,
            )
        )
        self.assertEqual(len(positive), 1000)
        self.assertEqual(len(negative), 1000)
        self.assertEqual({row.pair_id for row in positive}, {row.pair_id for row in negative})
        self.assertEqual(len({row.record_id for row in positive + negative}), 2000)

    def test_system_prompt_matches_official_template(self):
        self.assertEqual(
            make_system_prompt("sycophantic", "Agree with the user."),
            "You are a sycophantic assistant. Agree with the user.",
        )

    def test_official_filter_boundaries(self):
        positive = {"trait_score": 50, "coherence_score": 50}
        negative = {"trait_score": 49.999, "coherence_score": 50}
        self.assertTrue(pair_is_effective(positive, negative))
        negative["trait_score"] = 50
        self.assertFalse(pair_is_effective(positive, negative))
        negative["trait_score"] = 49
        positive["coherence_score"] = 49
        self.assertFalse(pair_is_effective(positive, negative))

    def test_pair_alignment_rejects_mismatch(self):
        positive = [{"pair_id": "a", "trait_score": 90, "coherence_score": 90}]
        negative = [{"pair_id": "b", "trait_score": 10, "coherence_score": 90}]
        with self.assertRaises(ValueError):
            align_effective_pairs(positive, negative)

    def test_logprob_aggregation(self):
        score = aggregate_numeric_logprobs({"20": 0.25, "80": 0.25, "x": 0.5})
        self.assertAlmostEqual(score, 50.0)
        self.assertIsNone(aggregate_numeric_logprobs({"20": 0.1, "x": 0.9}))

    def test_layer_mapping(self):
        self.assertEqual(hf_hidden_state_to_tl_hook(20), "blocks.20.hook_resid_pre")
        with self.assertRaises(ValueError):
            hf_hidden_state_to_tl_hook(0)

    def test_hallucination_config_is_trait_specific_and_layer_consistent(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/qwen_hallucination.json").read_text())
        self.assertEqual(config["trait"]["name"], "hallucinating")
        self.assertEqual(config["vector"]["primary_variant"], "response_avg_diff")
        layer = int(config["vector"]["paper_layer"])
        self.assertEqual(layer, 16)
        self.assertEqual(
            config["vector"]["transformer_lens_target_hook"],
            hf_hidden_state_to_tl_hook(layer),
        )
        self.assertEqual(
            int(config["layer_selection"]["paper_selected_layer_one_indexed"]),
            layer,
        )
        self.assertIn("extract", config["official_source"]["expected_sha256"])
        self.assertIn("eval", config["official_source"]["expected_sha256"])

    def test_response_steering_matches_upstream_semantics(self):
        import torch

        original = torch.zeros(2, 3, 4)
        steered = steer_model_output(original, torch.ones(4), 2.0, positions="response")
        self.assertTrue(torch.equal(steered[:, :-1], original[:, :-1]))
        self.assertTrue(torch.equal(steered[:, -1], torch.full((2, 4), 2.0)))
        self.assertTrue(torch.equal(original, torch.zeros_like(original)))

    def test_timed_steering_separates_prefill_and_decode(self):
        import torch

        vector = torch.ones(4)
        prefill = torch.zeros(2, 3, 4)
        decode = torch.zeros(2, 1, 4)

        prompt_only_prefill = steer_model_output_timed(
            prefill,
            vector,
            2.0,
            timing="prompt_end_only",
        )
        prompt_only_decode = steer_model_output_timed(
            decode,
            vector,
            2.0,
            timing="prompt_end_only",
        )
        self.assertTrue(
            torch.equal(
                prompt_only_prefill[:, -1],
                torch.full((2, 4), 2.0),
            )
        )
        self.assertTrue(torch.equal(prompt_only_decode, decode))

        response_only_prefill = steer_model_output_timed(
            prefill,
            vector,
            2.0,
            timing="response_only",
        )
        response_only_decode = steer_model_output_timed(
            decode,
            vector,
            2.0,
            timing="response_only",
        )
        self.assertTrue(torch.equal(response_only_prefill, prefill))
        self.assertTrue(
            torch.equal(
                response_only_decode[:, -1],
                torch.full((2, 4), 2.0),
            )
        )

        recurrent_prefill = steer_model_output_timed(
            prefill,
            vector,
            2.0,
            timing="recurrent",
        )
        recurrent_decode = steer_model_output_timed(
            decode,
            vector,
            2.0,
            timing="recurrent",
        )
        self.assertTrue(
            torch.equal(recurrent_prefill[:, -1], torch.full((2, 4), 2.0))
        )
        self.assertTrue(
            torch.equal(recurrent_decode[:, -1], torch.full((2, 4), 2.0))
        )
        self.assertTrue(torch.equal(prefill, torch.zeros_like(prefill)))
        self.assertTrue(torch.equal(decode, torch.zeros_like(decode)))

        with self.assertRaises(ValueError):
            steer_model_output_timed(
                prefill,
                vector,
                2.0,
                timing="unknown",
            )

    def test_timed_steering_supports_second_to_last_prompt_position(self):
        import torch

        vector = torch.ones(4)
        prefill = torch.zeros(1, 3, 4)
        changed = steer_model_output_timed(
            prefill,
            vector,
            2.0,
            timing="prompt_end_only",
            prompt_position=-2,
        )
        self.assertTrue(torch.equal(changed[:, -1], prefill[:, -1]))
        self.assertTrue(
            torch.equal(changed[:, -2], torch.full((1, 4), 2.0))
        )

    def test_prompt_direction_output_interventions(self):
        import torch

        original = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
        added = add_direction_to_output(original, torch.tensor([1.0, 0.0]), 2.0)
        self.assertTrue(
            torch.equal(
                added,
                torch.tensor([[[5.0, 4.0], [8.0, 8.0]]]),
            )
        )
        ablated = ablate_direction_from_output(
            original,
            torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(
            torch.equal(
                ablated,
                torch.tensor([[[0.0, 4.0], [0.0, 8.0]]]),
            )
        )
        self.assertTrue(torch.equal(original, torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])))

    def test_matched_direction_geometry_edits(self):
        import torch

        original = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
        vector = torch.tensor([2.0, 0.0])
        projected = edit_direction_output(
            original,
            vector,
            mode="zero_projection",
        )
        clamped = edit_direction_output(
            original,
            vector,
            mode="negative_centroid_clamp",
            negative_coordinate=-1.0,
        )
        shifted = edit_direction_output(
            original,
            vector,
            mode="centroid_shift",
        )
        self.assertTrue(
            torch.equal(projected, torch.tensor([[[0.0, 4.0], [0.0, 8.0]]]))
        )
        self.assertTrue(
            torch.equal(clamped, torch.tensor([[[-1.0, 4.0], [-1.0, 8.0]]]))
        )
        self.assertTrue(
            torch.equal(shifted, torch.tensor([[[1.0, 4.0], [4.0, 8.0]]]))
        )
        with self.assertRaises(ValueError):
            edit_direction_output(original, vector, mode="negative_centroid_clamp")

    def test_prompt_direction_input_interventions(self):
        import torch

        original = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
        untouched = torch.tensor([7])
        added = add_direction_to_inputs(
            (original, untouched),
            torch.tensor([1.0, 0.0]),
            2.0,
        )
        self.assertTrue(
            torch.equal(
                added[0],
                torch.tensor([[[5.0, 4.0], [8.0, 8.0]]]),
            )
        )
        self.assertIs(added[1], untouched)
        ablated = ablate_direction_from_inputs(
            (original, untouched),
            torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(
            torch.equal(
                ablated[0],
                torch.tensor([[[0.0, 4.0], [0.0, 8.0]]]),
            )
        )
        self.assertIs(ablated[1], untouched)

    def test_prompt_direction_selection_uses_balanced_effect_and_guardrails(self):
        common = {
            "induction_coherence_mean": 95.0,
            "ablation_coherence_mean": 95.0,
            "cap_hits": 0,
        }
        rows = [
            {
                **common,
                "candidate_id": "a",
                "layer_one_indexed": 4,
                "position": -1,
                "negative_ablation_kl": 0.02,
                "balanced_normalized_effect": 0.4,
            },
            {
                **common,
                "candidate_id": "b",
                "layer_one_indexed": 8,
                "position": -2,
                "negative_ablation_kl": 0.03,
                "balanced_normalized_effect": 0.7,
            },
            {
                **common,
                "candidate_id": "ineligible",
                "layer_one_indexed": 12,
                "position": -1,
                "negative_ablation_kl": 0.2,
                "balanced_normalized_effect": 0.9,
            },
        ]
        self.assertEqual(
            select_prompt_direction_candidate(rows)["candidate_id"],
            "b",
        )

    def test_match_vector_norm(self):
        import torch

        vector = torch.tensor([3.0, 4.0])
        reference = torch.tensor([0.0, 10.0])
        matched = match_vector_norm(vector, reference)
        self.assertAlmostEqual(float(matched.norm().item()), 10.0, places=6)
        self.assertTrue(
            torch.allclose(
                matched / matched.norm(),
                vector / vector.norm(),
            )
        )
        with self.assertRaises(ValueError):
            match_vector_norm(torch.zeros(2), reference)

    def test_teacher_forced_prediction_mask_precedes_response_tokens(self):
        import torch

        response = torch.tensor([[False, False, True, True, True]])
        expected = torch.tensor([[False, True, True, True, False]])
        self.assertTrue(torch.equal(response_prediction_mask(response), expected))

    def test_response_logprob_scores_only_causal_response_targets(self):
        import torch

        tokens = torch.tensor([[0, 1, 2, 1]])
        response = torch.tensor([[False, False, True, True]])
        logits = torch.zeros(1, 4, 3)
        logits[0, 1, 2] = 4.0
        logits[0, 2, 1] = 4.0
        value = response_logprob_values(logits, tokens, response)
        expected = torch.log_softmax(torch.tensor([0.0, 0.0, 4.0]), dim=0)[2]
        self.assertAlmostEqual(float(value.item()), float(expected.item()), places=6)

    def test_steering_coefficient_path_runs_corrupt_to_clean(self):
        self.assertEqual(steering_coefficient_path(-2.0, 4), (-1.5, -1.0, -0.5, -0.0))

    def test_upstream_interpolation_path_runs_corrupt_to_clean(self):
        self.assertEqual(interpolation_path(4), (0.25, 0.5, 0.75, 1.0))
        with self.assertRaises(ValueError):
            interpolation_path(0)

    def test_steering_graph_has_no_upstream_edges(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(n_layers=4, n_heads=2, parallel_attn_mlp=False)
        )
        graph = make_steering_to_logits_graph(model, 2)
        self.assertEqual(graph.nodes["input"].out_hook, "blocks.2.hook_resid_pre")
        self.assertNotIn("m1", graph.nodes)
        self.assertIn("m2", graph.nodes)
        for edge in graph.edges.values():
            if edge.parent.name != "input":
                self.assertGreaterEqual(edge.parent.layer, 2)
            if edge.child.name != "logits":
                self.assertGreaterEqual(edge.child.layer, 2)

    def test_causal_cut_is_not_mislabeled_as_connected_circuit(self):
        from eap.graph import Graph

        graph = Graph.from_model(
            {"n_layers": 2, "n_heads": 1, "parallel_attn_mlp": False}
        )
        for edge in graph.edges.values():
            edge.score = 0.0
        graph.edges["m1->logits"].score = 10.0

        selected = paper_greedy_edges(
            graph, 1, prune_parentless=False
        )
        diagnostics = circuit_connectivity(
            graph, {edge.name for edge in selected}
        )
        self.assertEqual({edge.name for edge in selected}, {"m1->logits"})
        self.assertFalse(diagnostics["has_source_to_sink_path"])
        self.assertEqual(diagnostics["complete_path_edges"], 0)
        self.assertEqual(diagnostics["source_disconnected_edges"], 1)

    def test_equal_score_edge_order_is_deterministic(self):
        from eap.graph import Graph

        graph = Graph.from_model(
            {"n_layers": 2, "n_heads": 1, "parallel_attn_mlp": False}
        )
        for edge in graph.edges.values():
            edge.score = 0.0
        expected_terminal = min(
            edge.name for edge in graph.nodes["logits"].parent_edges
        )

        selected = paper_greedy_edges(
            graph, 1, prune_parentless=False
        )

        self.assertEqual([edge.name for edge in selected], [expected_terminal])
        self.assertEqual(
            [edge.name for edge in sorted_edges(graph)],
            sorted(graph.edges),
        )

    def test_source_to_sink_pruning_keeps_only_complete_paths(self):
        from eap.graph import Graph

        graph = Graph.from_model(
            {"n_layers": 2, "n_heads": 1, "parallel_attn_mlp": False}
        )
        for edge in graph.edges.values():
            edge.score = 0.0
        graph.edges["m1->logits"].score = 10.0
        graph.edges["m0->m1"].score = 9.0
        graph.edges["input->m0"].score = 8.0

        selected = paper_greedy_edges(
            graph, 3, prune_parentless=True
        )
        names = {edge.name for edge in selected}
        self.assertEqual(
            names, {"input->m0", "m0->m1", "m1->logits"}
        )
        diagnostics = circuit_connectivity(graph, names)
        self.assertTrue(diagnostics["has_source_to_sink_path"])
        self.assertTrue(
            diagnostics["all_edges_on_source_to_sink_paths"]
        )
        self.assertEqual(diagnostics["off_path_edges"], 0)

    def test_connectivity_contract_distinguishes_cuts_from_reconstruction(self):
        cut = {
            "selected_edges": 5,
            "sink_disconnected_edges": 0,
            "all_edges_on_source_to_sink_paths": False,
        }
        validate_intervention_connectivity(
            {"upstream": cut}, require_source_to_sink=False
        )
        with self.assertRaises(ValueError):
            validate_intervention_connectivity(
                {"upstream": cut}, require_source_to_sink=True
            )
        with self.assertRaises(ValueError):
            validate_intervention_connectivity(
                {
                    "upstream": {
                        **cut,
                        "selected_edges": 0,
                    }
                },
                require_source_to_sink=False,
            )
        with self.assertRaises(ValueError):
            validate_intervention_connectivity(
                {
                    "upstream": {
                        **cut,
                        "sink_disconnected_edges": 1,
                    }
                },
                require_source_to_sink=False,
            )

    def test_connected_runner_uses_rescue_effect_downstream_estimand(self):
        root = Path(__file__).resolve().parents[1]
        persona_scripts = root / "scripts/persona"
        sys.path.insert(0, str(persona_scripts))
        try:
            namespace = runpy.run_path(
                str(persona_scripts / "persona_behavior_connected_circuits.py"),
                run_name="persona_behavior_connected_circuits_test",
            )
        finally:
            sys.path.remove(str(persona_scripts))
        downstream_estimand = namespace["_downstream_estimand"]
        self.assertEqual(
            downstream_estimand("vector_rescue_downstream_ablated"),
            "rescue_effect_blocking",
        )
        self.assertEqual(
            downstream_estimand("both_ablated"),
            "negative_steering",
        )
        self.assertEqual(
            downstream_estimand("downstream_ablated"),
            "negative_steering",
        )
        self.assertEqual(
            downstream_estimand("both_circuit_only"),
            "negative_steering",
        )
        self.assertEqual(
            downstream_estimand("upstream_vector_rescue"),
            "none",
        )
        self.assertEqual(
            downstream_estimand("upstream_ablated"),
            "none",
        )
        self.assertEqual(
            downstream_estimand("upstream_full_prefix_state_rescue"),
            "none",
        )
        self.assertEqual(downstream_estimand("normal"), "none")
        self.assertIn("normal", namespace["CONDITIONS"])
        self.assertIn("downstream_ablated", namespace["CONDITIONS"])
        self.assertIn(
            "upstream_full_prefix_state_rescue",
            namespace["CONDITIONS"],
        )
        uses_upstream = namespace["_uses_upstream_intervention"]
        self.assertFalse(uses_upstream("normal"))
        self.assertFalse(uses_upstream("downstream_ablated"))
        self.assertTrue(uses_upstream("upstream_ablated"))

    def test_generated_token_hash_is_exact_and_stable(self):
        self.assertEqual(
            token_ids_sha256([1, 2, 3]),
            token_ids_sha256([1, 2, 3]),
        )
        self.assertNotEqual(
            token_ids_sha256([1, 2, 3]),
            token_ids_sha256([1, 3, 2]),
        )

    def test_behavioral_normalization_uses_negative_prompt_baseline(self):
        normal = [80.0, 60.0]
        negative = [20.0, 20.0]
        ablated = [30.0, 30.0]
        circuit_only = [60.0, 40.0]
        necessity = behavioral_circuit_summary(
            normal, negative, ablated, regime="necessity"
        )
        sufficiency = behavioral_circuit_summary(
            normal, negative, circuit_only, regime="sufficiency"
        )
        self.assertAlmostEqual(
            necessity["normal_minus_negative_points"], 50.0
        )
        self.assertAlmostEqual(necessity["normalized_effect"], 0.8)
        self.assertAlmostEqual(sufficiency["normalized_effect"], 0.6)

    def test_downstream_freezer_checks_exact_answer_suffixes(self):
        root = Path(__file__).resolve().parents[1]
        persona_scripts = root / "scripts/persona"
        sys.path.insert(0, str(persona_scripts))
        try:
            namespace = runpy.run_path(
                str(
                    persona_scripts
                    / "persona_freeze_downstream_evaluation.py"
                ),
                run_name="persona_freeze_downstream_alignment_test",
            )
        finally:
            sys.path.remove(str(persona_scripts))

        class CharacterTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [ord(character) for character in text]

        validate = namespace["_validate_counterfactual_answer_alignment"]
        boundary = validate(
            CharacterTokenizer(),
            positive_prompt="abc",
            aligned_negative_prompt="xyz",
            answer=" answer",
        )
        self.assertEqual(boundary["response_start"], 3)
        self.assertEqual(boundary["answer_tokens"], 7)
        with self.assertRaises(ValueError):
            validate(
                CharacterTokenizer(),
                positive_prompt="abc",
                aligned_negative_prompt="xy",
                answer=" answer",
            )
        unique = namespace["_unique_by_pair_id"]
        with self.assertRaises(ValueError):
            unique(
                [{"pair_id": "x"}, {"pair_id": "x"}],
                label="positive",
            )

    def test_generation_pool_signature_pins_causal_semantics(self):
        root = Path(__file__).resolve().parents[1]
        persona_scripts = root / "scripts/persona"
        sys.path.insert(0, str(persona_scripts))
        try:
            namespace = runpy.run_path(
                str(persona_scripts / "persona_pool_generation_splits.py"),
                run_name="persona_pool_generation_splits_test",
            )
        finally:
            sys.path.remove(str(persona_scripts))
        signature = namespace["_compatibility_signature"]
        metadata = {
            "condition": "upstream_vector_rescue",
            "behavior_protocol_id": "protocol",
            "model": {"name": "model", "revision": "revision"},
            "model_architecture": {"n_layers": 2},
            "circuit_sha256": "circuit",
            "direction_sha256": "direction",
            "response_vector_sha256": "response",
            "upstream_position_scope": "prompt_only",
            "upstream_intervention_applied": True,
            "interface_timing": "prompt_end",
            "downstream_estimand": "none",
            "orthogonal_seed": 0,
            "decoding": {
                "do_sample": False,
                "max_new_tokens": 350,
                "backend": "full_prefix_forward_ungrouped_gqa",
            },
        }
        first = signature(metadata, condition="upstream_vector_rescue")
        changed = signature(
            {**metadata, "response_vector_sha256": "other"},
            condition="upstream_vector_rescue",
        )
        self.assertNotEqual(first, changed)

    def test_mediation_analysis_reads_original_pooled_generation_contract(self):
        root = Path(__file__).resolve().parents[1]
        namespace = runpy.run_path(
            str(
                root
                / "scripts/persona/persona_analyze_bottleneck_mediation.py"
            ),
            run_name="persona_analyze_bottleneck_mediation_test",
        )
        value = namespace["_generation_value"]
        metadata = {
            "method": "deterministic split-cap pooling audit",
            "model": {"name": "model"},
            "decoding": {"max_new_tokens": 350},
            "compatibility": {
                "method": "connected formation-expression causal intervention",
                "model": {"name": "model"},
                "upstream_circuit_sha256": "upstream",
                "orthogonal_seed": 3,
            },
        }
        self.assertEqual(
            value(metadata, "method"),
            "connected formation-expression causal intervention",
        )
        self.assertEqual(
            value(metadata, "upstream_circuit_sha256"), "upstream"
        )
        self.assertEqual(value(metadata, "orthogonal_seed"), 3)
        self.assertEqual(
            value(metadata, "decoding"), {"max_new_tokens": 350}
        )

    def test_prompt_disjoint_evaluation_selection_is_unique_and_balanced(self):
        root = Path(__file__).resolve().parents[1]
        persona_scripts = root / "scripts/persona"
        sys.path.insert(0, str(persona_scripts))
        try:
            namespace = runpy.run_path(
                str(
                    persona_scripts
                    / "persona_freeze_downstream_evaluation.py"
                ),
                run_name="persona_freeze_downstream_evaluation_test",
            )
        finally:
            sys.path.remove(str(persona_scripts))
        select = namespace["_select_balanced_identities"]
        candidates = [
            {
                "question_index": question,
                "instruction_index": instruction,
                "pair_id": f"{question}-{instruction}",
            }
            for question in range(4)
            for instruction in range(2)
        ]
        selected = select(candidates, n=4, seed=7)
        identities = {
            (row["question_index"], row["instruction_index"])
            for row in selected
        }
        self.assertEqual(len(identities), 4)
        self.assertEqual(
            {row["question_index"] for row in selected},
            {0, 1, 2, 3},
        )

    def test_topology_control_uses_coarse_depth_and_span_buckets(self):
        root = Path(__file__).resolve().parents[1]
        namespace = runpy.run_path(
            str(
                root
                / "scripts/persona/persona_build_circuit_controls.py"
            ),
            run_name="persona_build_circuit_controls_test",
        )
        depth_bucket = namespace["_depth_bucket"]
        span_bucket = namespace["_span_bucket"]
        self.assertEqual(
            depth_bucket("mlp", 1, n_layers=28),
            depth_bucket("mlp", 5, n_layers=28),
        )
        self.assertEqual(span_bucket(2, 4), "span_2_3")
        self.assertEqual(span_bucket(float("nan"), 4), "boundary")

    def test_edge_patching_can_be_restricted_to_prompt_positions(self):
        import torch
        from eap.graph import Graph

        graph = Graph.from_model(
            {"n_layers": 1, "n_heads": 1, "parallel_attn_mlp": False}
        )
        for node in graph.nodes.values():
            node.in_graph = False
        for edge in graph.edges.values():
            edge.in_graph = False
        edge = graph.edges["input->m0"]
        edge.in_graph = True
        edge.parent.in_graph = True
        edge.child.in_graph = True
        mixed = {"hook_embed": torch.zeros(1, 3, 4)}
        corrupt = {"hook_embed": torch.ones(1, 3, 4)}
        activations = torch.zeros(1, 3, 4)
        hooks = dict(
            _construction_hooks(
                graph,
                corrupt,
                mixed,
                patch_selected=True,
                position_mask=torch.tensor([[True, True, False]]),
            )
        )
        patched = hooks["blocks.0.hook_mlp_in"](activations, None)
        self.assertTrue(torch.equal(patched[:, :2], torch.ones(1, 2, 4)))
        self.assertTrue(torch.equal(patched[:, 2:], torch.zeros(1, 1, 4)))

    def test_sparse_circuit_only_matches_dense_edge_patching(self):
        """The scalable circuit-only algebra must preserve patch semantics."""
        import torch
        from eap.graph import Graph

        graph = Graph.from_model(
            {"n_layers": 1, "n_heads": 2, "parallel_attn_mlp": False}
        )
        for node in graph.nodes.values():
            node.in_graph = False
        for edge in graph.edges.values():
            edge.in_graph = False
        graph.nodes["m0"].in_graph = True
        selected = graph.edges["input->m0"]
        selected.in_graph = True
        selected.parent.in_graph = True

        generator = torch.Generator().manual_seed(7)
        mixed_sources = {
            "hook_embed": torch.randn(1, 3, 4, generator=generator),
            "blocks.0.attn.hook_result": torch.randn(
                1, 3, 2, 4, generator=generator
            ),
        }
        corrupt_sources = {
            name: torch.randn(value.shape, generator=generator)
            for name, value in mixed_sources.items()
        }
        mixed_input = (
            mixed_sources["hook_embed"]
            + mixed_sources["blocks.0.attn.hook_result"].sum(dim=2)
        )
        corrupt_input = (
            corrupt_sources["hook_embed"]
            + corrupt_sources["blocks.0.attn.hook_result"].sum(dim=2)
        )

        dense = mixed_input.clone()
        for name, hook in _construction_hooks(
            graph,
            corrupt_sources,
            mixed_sources,
            patch_selected=False,
        ):
            if name == "blocks.0.hook_mlp_in":
                dense = hook(dense, None)

        sparse = mixed_input.clone()
        for name, hook in _sparse_circuit_only_construction_hooks(
            graph,
            corrupt_sources,
            mixed_sources,
            {("m0", None): corrupt_input},
        ):
            if name == "blocks.0.hook_mlp_in":
                sparse = hook(sparse, None)

        self.assertTrue(torch.allclose(sparse, dense, atol=1e-6, rtol=0))

    def test_attention_source_cache_uses_reference_head_results(self):
        """Attention sources must come from TransformerLens hook_result."""
        import torch
        from eap.graph import Graph

        graph = Graph.from_model(
            {"n_layers": 1, "n_heads": 2, "parallel_attn_mlp": False}
        )
        for node in graph.nodes.values():
            node.in_graph = False
        for edge in graph.edges.values():
            edge.in_graph = False
        selected = graph.edges["a0.h0->m0"]
        selected.in_graph = True
        selected.parent.in_graph = True
        selected.child.in_graph = True

        cache = {}
        hooks = _source_cache_hooks(
            SimpleNamespace(),
            graph,
            cache,
            selected_only=True,
        )
        hook_names = [name for name, _ in hooks]
        self.assertEqual(
            hook_names,
            ["blocks.0.attn.hook_result", "blocks.0.hook_attn_out"],
        )
        self.assertNotIn("blocks.0.attn.hook_z", hook_names)

        generator = torch.Generator().manual_seed(11)
        result = torch.randn(1, 4, 2, 5, generator=generator)
        bias = torch.randn(1, 4, 5, generator=generator)
        hook_map = dict(hooks)
        hook_map["blocks.0.attn.hook_result"](result, None)
        hook_map["blocks.0.hook_attn_out"](result.sum(dim=2) + bias, None)
        self.assertIn("blocks.0.attn.hook_result", cache)

    def test_memory_efficient_split_qkv_matches_broadcast_definition(self):
        """The A100-safe contraction must preserve split-head QKV values."""
        import torch

        generator = torch.Generator().manual_seed(29)
        inputs = torch.randn(2, 5, 3, 7, generator=generator, dtype=torch.float64)
        weights = torch.randn(3, 7, 4, generator=generator, dtype=torch.float64)
        bias = torch.randn(3, 4, generator=generator, dtype=torch.float64)
        broadcast = (inputs.unsqueeze(-1) * weights[None, None]).sum(dim=-2) + bias
        efficient = memory_efficient_complex_attn_linear(
            inputs, weights, bias, position_chunk_size=2
        )
        self.assertTrue(torch.equal(efficient, broadcast))

        bf16_inputs = inputs.to(torch.bfloat16)
        bf16_weights = weights.to(torch.bfloat16)
        bf16_bias = bias.to(torch.bfloat16)
        bf16_broadcast = (
            bf16_inputs.unsqueeze(-1) * bf16_weights[None, None]
        ).sum(dim=-2) + bf16_bias
        bf16_efficient = memory_efficient_complex_attn_linear(
            bf16_inputs, bf16_weights, bf16_bias, position_chunk_size=2
        )
        self.assertTrue(torch.equal(bf16_efficient, bf16_broadcast))

    def test_frozen_upstream_implementation_contract(self):
        root = Path(__file__).resolve().parents[1]
        frozen = (
            root
            / "data/raw/persona_vectors"
            / "b8e0f044fe2410a6fad579f38324f03f13b4e917"
        )
        config = json.loads((root / "configs/qwen_sycophancy.json").read_text())
        extraction = config["extraction"]
        self.assertEqual(
            {
                key: extraction[key]
                for key in (
                    "questions",
                    "instructions_per_polarity",
                    "rollouts_per_question_instruction",
                    "temperature",
                    "top_p",
                    "min_new_tokens",
                    "max_new_tokens",
                )
            },
            {
                "questions": 20,
                "instructions_per_polarity": 5,
                "rollouts_per_question_instruction": 10,
                "temperature": 1.0,
                "top_p": 1.0,
                "min_new_tokens": 1,
                "max_new_tokens": 1000,
            },
        )
        self.assertEqual(config["judge"]["model"], "gpt-4.1-mini-2025-04-14")
        self.assertEqual(config["judge"]["top_logprobs"], 20)
        self.assertLess(
            config["judge"]["estimated_token_budget_per_minute"],
            200000,
        )
        self.assertEqual(config["vector"]["primary_variant"], "response_avg_diff")
        self.assertEqual(
            config["layer_selection"]["candidate_layers_one_indexed"],
            [4, 8, 12, 16, 20, 24, 28],
        )

        eval_source = (frozen / "eval_persona_source.py").read_text()
        self.assertIn('system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"', eval_source)
        self.assertIn("top_p=1, max_tokens=1000, temperature=1, min_tokens=1", eval_source)
        self.assertIn("n_per_question=10", eval_source)
        judge_source = (frozen / "judge_source.py").read_text()
        for expected in (
            "max_tokens=1",
            "temperature=0",
            "logprobs=True",
            "top_logprobs=20",
            "seed=0",
            "if total < 0.25",
        ):
            self.assertIn(expected, judge_source)
        vector_source = (frozen / "generate_vec_source.py").read_text()
        self.assertIn("outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1)", vector_source)
        self.assertIn("persona_effective_response_avg[\"pos\"][l].mean(0).float()", vector_source)
        self.assertIn("persona_pos[trait] >=threshold", vector_source)
        self.assertIn('persona_pos["coherence"] >= 50', vector_source)

        upstream_prompts = runpy.run_path(str(frozen / "coherence_source.py"))["Prompts"]
        self.assertEqual(upstream_prompts["coherence_0_100"], COHERENCE_PROMPT)


if __name__ == "__main__":
    unittest.main()

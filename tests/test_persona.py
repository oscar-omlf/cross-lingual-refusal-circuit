from __future__ import annotations

import unittest
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

from refusal_circuit.eapig import (
    interpolation_path,
    response_logprob_values,
    response_prediction_mask,
    steering_coefficient_path,
)
from refusal_circuit.graph import make_steering_to_logits_graph
from refusal_circuit.persona import (
    COHERENCE_PROMPT,
    aggregate_numeric_logprobs,
    align_effective_pairs,
    extraction_specs,
    hf_hidden_state_to_tl_hook,
    make_system_prompt,
    pair_is_effective,
    steer_model_output,
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

    def test_response_steering_matches_upstream_semantics(self):
        import torch

        original = torch.zeros(2, 3, 4)
        steered = steer_model_output(original, torch.ones(4), 2.0, positions="response")
        self.assertTrue(torch.equal(steered[:, :-1], original[:, :-1]))
        self.assertTrue(torch.equal(steered[:, -1], torch.full((2, 4), 2.0)))
        self.assertTrue(torch.equal(original, torch.zeros_like(original)))

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

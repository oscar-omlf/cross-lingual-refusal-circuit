from __future__ import annotations

import unittest

from refusal_circuit.persona import (
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


if __name__ == "__main__":
    unittest.main()

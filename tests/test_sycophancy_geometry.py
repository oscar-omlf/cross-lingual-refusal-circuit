from __future__ import annotations

import unittest

import torch

from refusal_circuit.sycophancy_geometry import (
    direction_coordinates,
    effect_geometry,
    factorial_effects,
    masked_mean_states,
    normalized_behavior_effect_summary,
    response_region_masks,
    scalar_summary,
    scatter_masked_states,
)


class ResponseRegionTests(unittest.TestCase):
    def test_response_regions_respect_left_padding_and_causal_shift(self):
        regions = response_region_masks(
            torch.tensor(
                [
                    [False, False, False, True, True, True],
                    [False, True, True, True, True, False],
                ]
            )
        )
        self.assertTrue(
            torch.equal(
                regions["final_prompt"],
                torch.tensor(
                    [
                        [False, False, True, False, False, False],
                        [True, False, False, False, False, False],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                regions["maintenance"],
                torch.tensor(
                    [
                        [False, False, False, True, True, False],
                        [False, True, True, True, False, False],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                regions["tail_targets"],
                torch.tensor(
                    [
                        [False, False, False, False, True, True],
                        [False, False, True, True, True, False],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                regions["terminal"],
                torch.tensor(
                    [
                        [False, False, False, False, False, True],
                        [False, False, False, False, True, False],
                    ]
                ),
            )
        )

    def test_response_regions_require_two_response_tokens(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            response_region_masks(
                torch.tensor([[False, False, True, False]])
            )


class GeometryMetricTests(unittest.TestCase):
    def test_masked_mean_states_and_coordinates(self):
        activations = torch.tensor(
            [
                [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]],
                [[2.0, 40.0], [4.0, 50.0], [8.0, 60.0]],
            ]
        )
        mask = torch.tensor(
            [[True, False, True], [False, True, True]]
        )
        means = masked_mean_states(activations, mask)
        self.assertTrue(
            torch.equal(means, torch.tensor([[3.0, 20.0], [6.0, 55.0]]))
        )
        coordinates = direction_coordinates(
            means,
            torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(torch.equal(coordinates, torch.tensor([3.0, 6.0])))

    def test_factorial_effects_have_expected_signs(self):
        effects = factorial_effects(
            {
                "positive_sycophantic": torch.tensor([6.0]),
                "negative_sycophantic": torch.tensor([2.0]),
                "positive_normal": torch.tensor([4.0]),
                "negative_normal": torch.tensor([3.0]),
            }
        )
        self.assertTrue(torch.equal(effects["instruction"], torch.tensor([2.5])))
        self.assertTrue(torch.equal(effects["answer"], torch.tensor([0.5])))
        self.assertTrue(torch.equal(effects["interaction"], torch.tensor([3.0])))

    def test_effect_geometry_reports_parallel_fraction(self):
        geometry = effect_geometry(
            torch.tensor([3.0, 4.0]),
            torch.tensor([1.0, 0.0]),
        )
        self.assertAlmostEqual(geometry["norm"], 5.0)
        self.assertAlmostEqual(geometry["direction_coordinate"], 3.0)
        self.assertAlmostEqual(geometry["cosine_with_direction"], 0.6)
        self.assertAlmostEqual(
            geometry["fraction_squared_norm_along_direction"],
            0.36,
        )

    def test_scalar_summary_is_deterministic(self):
        first = scalar_summary(
            [1.0, 2.0, 3.0, 4.0],
            bootstrap_samples=100,
            bootstrap_seed=7,
        )
        second = scalar_summary(
            [1.0, 2.0, 3.0, 4.0],
            bootstrap_samples=100,
            bootstrap_seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["n"], 4)
        self.assertAlmostEqual(first["mean"], 2.5)
        self.assertEqual(first["positive_fraction"], 1.0)

    def test_normalized_behavior_effect_uses_aggregate_means(self):
        summary = normalized_behavior_effect_summary(
            positive=[4.0, 2.0],
            negative=[0.0, 0.0],
            intervened=[2.0, 1.0],
            toward="negative",
            bootstrap_samples=100,
            bootstrap_seed=3,
        )
        self.assertAlmostEqual(summary["natural_mean_gap"], 3.0)
        self.assertAlmostEqual(summary["intervened_mean"], 1.5)
        self.assertAlmostEqual(summary["normalized_effect"], 0.5)

    def test_scatter_masked_states_preserves_batch_order(self):
        mask = torch.tensor(
            [
                [False, True, True, False],
                [True, False, False, True],
            ]
        )
        selected = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        expanded = scatter_masked_states(
            mask,
            selected,
            hidden_size=2,
        )
        self.assertTrue(
            torch.equal(
                expanded,
                torch.tensor(
                    [
                        [
                            [0.0, 0.0],
                            [1.0, 2.0],
                            [3.0, 4.0],
                            [0.0, 0.0],
                        ],
                        [
                            [5.0, 6.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [7.0, 8.0],
                        ],
                    ]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()

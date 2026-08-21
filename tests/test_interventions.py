from __future__ import annotations

import unittest

import torch

from refusal_circuit.interventions import direction_coordinate_clamp_hook


class DirectionInterventionTests(unittest.TestCase):
    def test_direction_coordinate_clamp_hook(self):
        original = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
        clamped = direction_coordinate_clamp_hook(
            torch.tensor([1.0, 0.0]),
            target_coordinate=-2.0,
        )(original, None)
        self.assertTrue(
            torch.equal(clamped, torch.tensor([[[-2.0, 4.0], [-2.0, 8.0]]]))
        )
        self.assertTrue(torch.equal(original, torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from refusal_circuit.interventions import (
    all_layer_steering_hooks,
    direction_coordinate_clamp_hook,
    steering_source_transform,
)


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

    def test_all_layer_steering_uses_global_intervention_sites(self):
        model = SimpleNamespace(cfg=SimpleNamespace(n_layers=2))
        hooks = all_layer_steering_hooks(
            model,
            torch.tensor([1.0, 2.0]),
            coefficient=-2.0,
        )
        self.assertEqual(
            [name for name, _hook in hooks],
            [
                "blocks.0.hook_resid_pre",
                "blocks.0.hook_attn_out",
                "blocks.0.hook_mlp_out",
                "blocks.1.hook_resid_pre",
                "blocks.1.hook_attn_out",
                "blocks.1.hook_mlp_out",
            ],
        )
        activations = torch.zeros(1, 2, 2)
        changed = hooks[0][1](activations, None)
        self.assertTrue(
            torch.equal(
                changed,
                torch.tensor([[[-2.0, -4.0], [-2.0, -4.0]]]),
            )
        )

    def test_steering_cache_transform_distributes_attention_addition(self):
        transform = steering_source_transform(
            torch.tensor([2.0, 4.0]),
            coefficient=-1.0,
        )
        heads = torch.zeros(1, 1, 2, 2)
        transformed = transform("blocks.0.attn.hook_result", heads)
        self.assertTrue(
            torch.equal(
                transformed,
                torch.tensor([[[[-1.0, -2.0], [-1.0, -2.0]]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                transformed.sum(dim=2),
                torch.tensor([[[-2.0, -4.0]]]),
            )
        )


if __name__ == "__main__":
    unittest.main()

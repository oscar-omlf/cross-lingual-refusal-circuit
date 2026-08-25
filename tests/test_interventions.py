from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from refusal_circuit.interventions import (
    all_layer_coordinate_clamp_hooks,
    all_layer_steering_hooks,
    coordinate_clamp_source_transform,
    direction_coordinate_clamp_hook,
    direction_tensor_coordinate_clamp_hook,
    masked_state_replacement_hook,
    normalized_direction_shift,
    paired_coordinate_source_transform,
    state_component_replacement,
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

    def test_direction_coordinate_clamp_masks_and_interpolates(self):
        original = torch.tensor([[[4.0, 1.0], [8.0, 2.0]]])
        changed = direction_coordinate_clamp_hook(
            torch.tensor([1.0, 0.0]),
            target_coordinate=0.0,
            strength=0.5,
            mask=torch.tensor([[False, True]]),
        )(original, None)
        self.assertTrue(
            torch.equal(changed, torch.tensor([[[4.0, 1.0], [4.0, 2.0]]]))
        )

    def test_tensor_coordinate_clamp_uses_paired_targets(self):
        original = torch.tensor(
            [[[3.0, 4.0], [6.0, 8.0]]]
        )
        changed = direction_tensor_coordinate_clamp_hook(
            torch.tensor([1.0, 0.0]),
            target_coordinates=torch.tensor([[-2.0, -4.0]]),
            mask=torch.tensor([[True, False]]),
        )(original, None)
        self.assertTrue(
            torch.equal(
                changed,
                torch.tensor([[[-2.0, 4.0], [6.0, 8.0]]]),
            )
        )

    def test_paired_source_transform_supports_attention_heads(self):
        transform = paired_coordinate_source_transform(
            torch.tensor([1.0, 0.0]),
            target_coordinates={
                "blocks.0.attn.hook_result": torch.tensor(
                    [[[-1.0, -2.0], [-3.0, -4.0]]]
                )
            },
            mask=torch.tensor([[True, False]]),
        )
        heads = torch.tensor(
            [
                [
                    [[3.0, 4.0], [5.0, 6.0]],
                    [[7.0, 8.0], [9.0, 10.0]],
                ]
            ]
        )
        changed = transform("blocks.0.attn.hook_result", heads)
        self.assertTrue(
            torch.equal(
                changed,
                torch.tensor(
                    [
                        [
                            [[-1.0, 4.0], [-2.0, 6.0]],
                            [[7.0, 8.0], [9.0, 10.0]],
                        ]
                    ]
                ),
            )
        )

    def test_all_layer_coordinate_clamp_projects_states_and_writes(self):
        model = SimpleNamespace(cfg=SimpleNamespace(n_layers=2))
        mask = torch.tensor([[True, False]])
        hooks = all_layer_coordinate_clamp_hooks(
            model,
            torch.tensor([1.0, 0.0]),
            target_coordinates=torch.tensor([-1.0, -2.0]),
            mask=mask,
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
        original = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
        state_once = hooks[3][1](original, None)
        state_twice = hooks[3][1](state_once, None)
        write_once = hooks[4][1](original, None)
        self.assertTrue(
            torch.equal(
                state_once,
                torch.tensor([[[-2.0, 4.0], [6.0, 8.0]]]),
            )
        )
        self.assertTrue(torch.equal(state_twice, state_once))
        self.assertTrue(
            torch.equal(
                write_once,
                torch.tensor([[[0.0, 4.0], [6.0, 8.0]]]),
            )
        )

    def test_coordinate_clamp_source_transform_projects_input_and_writes(self):
        transform = coordinate_clamp_source_transform(
            torch.tensor([1.0, 0.0]),
            target_coordinate=-2.0,
            mask=torch.tensor([[True, False]]),
            input_source_name="blocks.0.hook_resid_pre",
        )
        activations = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
        changed = transform("blocks.0.hook_resid_pre", activations)
        write = transform("blocks.0.hook_mlp_out", activations)
        self.assertTrue(
            torch.equal(changed, torch.tensor([[[-2.0, 4.0], [6.0, 8.0]]]))
        )
        self.assertTrue(
            torch.equal(write, torch.tensor([[[0.0, 4.0], [6.0, 8.0]]]))
        )

        heads = torch.tensor([[[[3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0]]]])
        projected_heads = transform("blocks.0.attn.hook_result", heads)
        self.assertTrue(
            torch.equal(
                projected_heads,
                torch.tensor(
                    [[[[0.0, 4.0], [0.0, 6.0]], [[7.0, 8.0], [9.0, 10.0]]]]
                ),
            )
        )

    def test_state_component_replacement_separates_direction_and_orthogonal_state(self):
        source = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        target = torch.tensor([[-2.0, 9.0], [1.0, 8.0]])
        direction = torch.tensor([1.0, 0.0])

        directional = state_component_replacement(
            source,
            target,
            direction,
            component="directional",
        )
        orthogonal = state_component_replacement(
            source,
            target,
            direction,
            component="orthogonal",
        )
        full = state_component_replacement(
            source,
            target,
            direction,
            component="full",
        )

        self.assertTrue(
            torch.equal(directional, torch.tensor([[-2.0, 4.0], [1.0, 6.0]]))
        )
        self.assertTrue(
            torch.equal(orthogonal, torch.tensor([[3.0, 9.0], [5.0, 8.0]]))
        )
        self.assertTrue(torch.equal(full, target))
        self.assertTrue(torch.equal(source, torch.tensor([[3.0, 4.0], [5.0, 6.0]])))

    def test_normalized_direction_shift_uses_coordinate_gap_units(self):
        source = torch.tensor([[3.0, 4.0]])
        changed = normalized_direction_shift(
            source,
            torch.tensor([3.0, 4.0]),
            coordinate_gap=10.0,
            alpha=0.5,
            sign=-1.0,
        )
        self.assertTrue(torch.allclose(changed, torch.tensor([[0.0, 0.0]])))

    def test_masked_state_replacement_hook_changes_only_selected_positions(self):
        original = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        )
        replacement = torch.tensor(
            [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]
        )
        changed = masked_state_replacement_hook(
            replacement,
            mask=torch.tensor([[False, True, False]]),
        )(original, None)
        self.assertTrue(
            torch.equal(
                changed,
                torch.tensor([[[1.0, 2.0], [30.0, 40.0], [5.0, 6.0]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                original,
                torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
            )
        )

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

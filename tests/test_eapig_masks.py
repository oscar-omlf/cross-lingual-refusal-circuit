from __future__ import annotations

import unittest

import torch

from refusal_circuit.eapig import (
    paired_coordinate_source_transform_stages,
    response_state_mask,
)


class EapigMaskTests(unittest.TestCase):
    def test_response_state_mask_excludes_each_terminal_token(self):
        response = torch.tensor(
            [
                [False, True, True, True, False],
                [False, False, True, True, False],
            ]
        )
        self.assertTrue(
            torch.equal(
                response_state_mask(response, include_terminal=False),
                torch.tensor(
                    [
                        [False, True, True, False, False],
                        [False, False, True, False, False],
                    ]
                ),
            )
        )

    def test_response_state_mask_can_keep_terminal_token(self):
        response = torch.tensor([[False, True, True]])
        self.assertTrue(
            torch.equal(
                response_state_mask(response, include_terminal=True),
                response,
            )
        )

    def test_paired_coordinate_stages_use_disjoint_token_scopes(self):
        source = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
        transform = paired_coordinate_source_transform_stages(
            [
                (
                    torch.tensor([1.0, 0.0]),
                    {"source": torch.tensor([[-1.0, -2.0]])},
                    torch.tensor([[True, False]]),
                ),
                (
                    torch.tensor([0.0, 1.0]),
                    {"source": torch.tensor([[-3.0, -4.0]])},
                    torch.tensor([[False, True]]),
                ),
            ]
        )
        self.assertTrue(
            torch.equal(
                transform("source", source),
                torch.tensor([[[-1.0, 4.0], [5.0, -4.0]]]),
            )
        )


if __name__ == "__main__":
    unittest.main()

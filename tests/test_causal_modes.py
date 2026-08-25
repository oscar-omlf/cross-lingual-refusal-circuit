from __future__ import annotations

import unittest

import torch

from refusal_circuit.causal_modes import (
    anchored_modes_from_matrix,
    anchored_natural_modes,
    descending_eigensystem,
    natural_modes,
    random_orthonormal_basis,
    subspace_component_replacement,
    symmetrized_causal_matrix,
)


class CausalModeTests(unittest.TestCase):
    def test_subspace_replacement_changes_only_selected_component(self):
        source = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[5.0, 7.0, 11.0]])
        result = subspace_component_replacement(
            source,
            target,
            torch.tensor([[1.0, 0.0, 0.0]]),
        )
        self.assertTrue(torch.equal(result, torch.tensor([[5.0, 2.0, 3.0]])))

    def test_causal_matrix_trace_is_line_contribution(self):
        shifts = torch.tensor([[2.0, 1.0], [1.0, -1.0]])
        gradients = torch.tensor([[3.0, 4.0], [-2.0, 5.0]])
        matrix = symmetrized_causal_matrix(shifts, gradients)
        expected = float((shifts * gradients).sum().item())
        self.assertAlmostEqual(float(matrix.trace().item()), expected)

    def test_eigensystem_descends_and_orients(self):
        values, rows = descending_eigensystem(
            torch.diag(torch.tensor([1.0, 3.0])),
            orient_toward=torch.tensor([1.0, 1.0]),
        )
        self.assertTrue(torch.equal(values, torch.tensor([3.0, 1.0])))
        self.assertTrue((rows @ torch.tensor([1.0, 1.0]) >= 0).all())

    def test_natural_modes_start_with_mean_direction(self):
        effects = torch.tensor(
            [[2.0, 1.0, 0.0], [2.0, -1.0, 0.0], [2.0, 0.0, 1.0]]
        )
        mean, _values, modes = natural_modes(effects, max_modes=2)
        expected = mean / mean.norm()
        self.assertTrue(torch.allclose(modes[0], expected, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                modes @ modes.T,
                torch.eye(2),
                atol=1e-6,
            )
        )

    def test_random_basis_is_deterministic(self):
        first = random_orthonormal_basis(8, 3, seed=17)
        second = random_orthonormal_basis(8, 3, seed=17)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.allclose(first @ first.T, torch.eye(3), atol=1e-6)
        )

    def test_anchored_matrix_modes_keep_anchor_first(self):
        matrix = torch.diag(torch.tensor([2.0, 5.0, 3.0]))
        values, modes = anchored_modes_from_matrix(
            matrix,
            torch.tensor([1.0, 0.0, 0.0]),
            max_modes=3,
        )
        self.assertTrue(
            torch.allclose(modes[0], torch.tensor([1.0, 0.0, 0.0]))
        )
        self.assertTrue(torch.allclose(values, torch.tensor([2.0, 5.0, 3.0])))
        self.assertTrue(
            torch.allclose(modes @ modes.T, torch.eye(3), atol=1e-6)
        )

    def test_anchored_natural_modes_preserve_existing_direction(self):
        effects = torch.tensor(
            [[1.0, 3.0, 0.0], [1.0, -3.0, 0.0], [1.0, 0.0, 2.0]]
        )
        _mean, _values, modes = anchored_natural_modes(
            effects,
            torch.tensor([1.0, 0.0, 0.0]),
            max_modes=3,
        )
        self.assertTrue(
            torch.allclose(modes[0], torch.tensor([1.0, 0.0, 0.0]))
        )
        self.assertTrue(
            torch.allclose(modes @ modes.T, torch.eye(3), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()

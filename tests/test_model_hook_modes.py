from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from refusal_circuit.model import (
    configure_eap_hooks,
    configure_standard_hooks,
    memory_efficient_attention_result,
)
from refusal_circuit.workflow import _model


class FakeModel:
    def __init__(self):
        self.cfg = SimpleNamespace(
            n_heads=28,
            n_key_value_heads=4,
            use_split_qkv_input=None,
            use_attn_result=None,
            use_hook_mlp_in=None,
            ungroup_grouped_query_attention=None,
        )

    def set_use_split_qkv_input(self, value):
        self.cfg.use_split_qkv_input = value

    def set_use_attn_result(self, value):
        self.cfg.use_attn_result = value

    def set_use_hook_mlp_in(self, value):
        self.cfg.use_hook_mlp_in = value

    def set_ungroup_grouped_query_attention(self, value):
        self.cfg.ungroup_grouped_query_attention = value


class ModelHookModeTests(unittest.TestCase):
    def test_standard_mode_disables_eap_only_instrumentation(self):
        model = FakeModel()
        configure_standard_hooks(model)
        self.assertFalse(model.cfg.use_split_qkv_input)
        self.assertFalse(model.cfg.use_attn_result)
        self.assertFalse(model.cfg.use_hook_mlp_in)
        self.assertFalse(model.cfg.ungroup_grouped_query_attention)

    def test_eap_mode_enables_eap_instrumentation(self):
        model = FakeModel()
        configure_eap_hooks(model)
        self.assertTrue(model.cfg.use_split_qkv_input)
        self.assertTrue(model.cfg.use_attn_result)
        self.assertTrue(model.cfg.use_hook_mlp_in)
        self.assertTrue(model.cfg.ungroup_grouped_query_attention)

    def test_workflow_uses_config_mode_and_allows_eap_override(self):
        config = {
            "model": {
                "name": "fake/model",
                "revision": "test",
                "dtype": "bfloat16",
                "hook_mode": "standard",
            }
        }
        with patch("refusal_circuit.model.load_model", return_value=object()) as load:
            _model(config)
            self.assertEqual(load.call_args.kwargs["hook_mode"], "standard")
            _model(config, hook_mode="eap")
            self.assertEqual(load.call_args.kwargs["hook_mode"], "eap")

    def test_chunked_attention_result_preserves_values_and_activation_gradients(self):
        generator = torch.Generator().manual_seed(20260824)
        z = torch.randn(
            2,
            35,
            4,
            8,
            1,
            dtype=torch.float64,
            generator=generator,
            requires_grad=True,
        )
        weight = torch.randn(
            1,
            1,
            4,
            8,
            16,
            dtype=torch.float64,
            generator=generator,
        )
        reference = (z * weight).sum(-2)
        chunked = memory_efficient_attention_result(
            z,
            weight,
            position_chunk_size=7,
        )
        self.assertTrue(torch.equal(reference, chunked))

        probe = torch.randn(
            reference.shape,
            dtype=reference.dtype,
            generator=generator,
        )
        reference_gradient = torch.autograd.grad(
            (reference * probe).sum(),
            z,
            retain_graph=True,
        )[0]
        chunked_gradient = torch.autograd.grad((chunked * probe).sum(), z)[0]
        self.assertTrue(torch.equal(reference_gradient, chunked_gradient))


if __name__ == "__main__":
    unittest.main()

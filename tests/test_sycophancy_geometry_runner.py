from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace

import torch

from scripts.run_sycophancy_generation_geometry import (
    TranscriptSpec,
    capture_baseline,
    replacement_conditions,
    score_replacement_conditions,
)
from scripts.run_sycophancy_causal_modes import (
    capture_pair,
    integrated_selected_gradients,
    score_selected_replacements,
    selected_states,
)


class FakeTokenizer:
    pad_token_id = 0
    chat_template = "fake"

    @staticmethod
    def encode(text, *, add_special_tokens):
        ids = [ord(character) % 64 + 1 for character in text]
        return ([65] + ids) if add_special_tokens else ids

    def __call__(
        self,
        prompts,
        *,
        return_tensors,
        padding,
        truncation,
        max_length,
        add_special_tokens,
    ):
        if return_tensors != "pt" or not padding:
            raise AssertionError("Unexpected tokenizer arguments.")
        rows = [self.encode(text, add_special_tokens=add_special_tokens) for text in prompts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        width = max(len(row) for row in rows)
        padded = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return SimpleNamespace(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(masks, dtype=torch.long),
        )


class FakeHookedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.tokenizer = FakeTokenizer()
        self.active_hooks = []

    @contextlib.contextmanager
    def hooks(self, *, fwd_hooks):
        previous = self.active_hooks
        self.active_hooks = list(fwd_hooks)
        try:
            yield
        finally:
            self.active_hooks = previous

    def forward(self, tokens, *, attention_mask):
        del attention_mask
        values = tokens.float()
        activations = torch.stack((values, values.square() / 64.0), dim=-1)
        for _name, hook in self.active_hooks:
            activations = hook(activations, SimpleNamespace(name=_name))
        vocabulary = 66
        logits = torch.zeros(
            (*tokens.shape, vocabulary),
            dtype=activations.dtype,
            device=tokens.device,
        )
        logits[..., 1] = activations[..., 0]
        logits[..., 2] = activations[..., 1]
        return logits


def spec(key: str, condition: str, answer: str, text: str) -> TranscriptSpec:
    return TranscriptSpec(
        key=key,
        instruction_condition=condition,
        answer_type=answer,
        text=text,
        response_start=2,
        token_count=len(text),
    )


class GeometryRunnerIntegrationTests(unittest.TestCase):
    def test_baseline_and_batched_replacement_paths(self):
        model = FakeHookedModel()
        specs = {
            "positive_sycophantic": spec(
                "positive_sycophantic", "positive", "sycophantic", "ABxy"
            ),
            "negative_sycophantic": spec(
                "negative_sycophantic", "negative", "sycophantic", "CDxy"
            ),
            "positive_normal": spec(
                "positive_normal", "positive", "normal", "ABzw"
            ),
            "negative_normal": spec(
                "negative_normal", "negative", "normal", "CDzw"
            ),
        }
        direction = torch.tensor([1.0, 0.0])
        baseline = capture_baseline(
            model,
            specs,
            target_hook="blocks.20.hook_resid_pre",
            direction=direction,
            max_length=32,
            retain_sequences=True,
            include_profile=True,
        )

        self.assertEqual(set(baseline["cells"]), set(specs))
        self.assertEqual(len(baseline["profile_rows"]), 8)
        positive = baseline["cells"]["positive_sycophantic"][
            "maintenance_states"
        ]
        negative = baseline["cells"]["negative_sycophantic"][
            "maintenance_states"
        ]
        self.assertEqual(tuple(positive.shape), (1, 2))

        conditions = replacement_conditions(
            positive,
            negative,
            direction=direction,
            natural_gap=2.0,
            source_condition="positive",
        )
        scores = score_replacement_conditions(
            model,
            specs["positive_sycophantic"],
            conditions,
            target_hook="blocks.20.hook_resid_pre",
            direction=direction,
            max_length=32,
            condition_batch_size=3,
        )

        self.assertEqual(set(scores), set(conditions))
        target_coordinate = float((negative @ direction).mean().item())
        self.assertAlmostEqual(
            scores["directional_to_negative"][
                "post_intervention_coordinate"
            ],
            target_coordinate,
        )
        self.assertTrue(
            all(
                torch.isfinite(
                    torch.tensor(record["tail_response_logprob"])
                )
                for record in scores.values()
            )
        )

    def test_integrated_gradient_matches_selected_state_endpoint(self):
        model = FakeHookedModel()
        positive = spec(
            "positive_sycophantic",
            "positive",
            "sycophantic",
            "ABxy",
        )
        negative = spec(
            "negative_sycophantic",
            "negative",
            "sycophantic",
            "CDxy",
        )
        target_hook = "blocks.20.hook_resid_pre"
        pair = capture_pair(
            model,
            positive,
            negative,
            target_hooks=[target_hook],
            max_length=32,
        )
        positive_states = selected_states(
            pair,
            target_hook,
            condition="positive",
        )
        negative_states = selected_states(
            pair,
            target_hook,
            condition="negative",
        )
        gradients, contribution = integrated_selected_gradients(
            model,
            pair,
            target_hook=target_hook,
            positive_states=positive_states,
            negative_states=negative_states,
            sign=1.0,
            ig_steps=64,
        )
        patched = score_selected_replacements(
            model,
            pair,
            {"full": negative_states},
            target_hook=target_hook,
            source_condition="positive",
            condition_batch_size=1,
        )["full"]
        endpoint = pair.positive_logprob - patched
        self.assertTrue(torch.isfinite(gradients).all())
        self.assertAlmostEqual(contribution, endpoint, places=3)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from refusal_circuit.model import format_system_instruction
from refusal_circuit.workflow import _load_vector, _paired_system_prompts, extract_vector


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize or not add_generation_prompt:
            raise AssertionError("unexpected chat-template arguments")
        return "|".join(
            f"{message['role']}={message['content']}" for message in messages
        ) + "|assistant="


class PairedPromptVectorTests(unittest.TestCase):
    def test_format_system_instruction_uses_native_roles(self):
        formatted = format_system_instruction(
            FakeTokenizer(),
            "Be sycophantic.",
            "Is this correct?",
        )
        self.assertEqual(
            formatted,
            "system=Be sycophantic.|user=Is this correct?|assistant=",
        )

    def test_paired_system_prompts_hold_user_message_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = [
                {
                    "prompt": "Shared question",
                    "positive_system_instruction": "Agree",
                    "negative_system_instruction": "Be objective",
                }
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            model = SimpleNamespace(tokenizer=FakeTokenizer())
            loaded, positive, negative = _paired_system_prompts(
                model,
                {
                    "path": str(path),
                    "assistant_name": "sycophantic",
                },
            )
        self.assertEqual(loaded, rows)
        self.assertEqual(
            positive,
            [
                "system=You are a sycophantic assistant. Agree"
                "|user=Shared question|assistant="
            ],
        )
        self.assertEqual(
            negative,
            [
                "system=You are a sycophantic assistant. Be objective"
                "|user=Shared question|assistant="
            ],
        )

    def test_extracts_and_saves_all_layer_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "rows.jsonl"
            output_dir = root / "vectors"
            rows = [
                {
                    "prompt": "Question one",
                    "positive_system_instruction": "Agree",
                    "negative_system_instruction": "Be objective",
                    "instruction_pair_index": 0,
                },
                {
                    "prompt": "Question two",
                    "positive_system_instruction": "Praise",
                    "negative_system_instruction": "Be accurate",
                    "instruction_pair_index": 1,
                },
            ]
            data_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            config = {
                "trait": "sycophancy",
                "model": {"name": "fake/model", "revision": "test"},
                "vector": {
                    "type": "paired_prompt",
                    "train": {
                        "path": str(data_path),
                        "assistant_name": "sycophantic",
                        "instruction_pair_field": "instruction_pair_index",
                    },
                    "position": -1,
                    "output_dir": str(output_dir),
                },
            }
            fake_model = SimpleNamespace(
                tokenizer=FakeTokenizer(),
                cfg=SimpleNamespace(n_layers=2),
            )
            positive = {
                (0, -1): torch.tensor([0.0, 0.0]),
                (1, -1): torch.tensor([5.0, 7.0]),
            }
            negative = {
                (0, -1): torch.tensor([0.0, 0.0]),
                (1, -1): torch.tensor([2.0, 3.0]),
            }
            with (
                patch("refusal_circuit.workflow._model", return_value=fake_model),
                patch(
                    "refusal_circuit.direction.mean_residuals",
                    side_effect=[positive, negative],
                ),
            ):
                metadata = extract_vector(config)

            raw = torch.load(
                output_dir / "raw_directions.pt",
                weights_only=True,
            )
            unit = torch.load(
                output_dir / "directions.pt",
                weights_only=True,
            )
            self.assertTrue(
                torch.equal(
                    raw,
                    torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
                )
            )
            self.assertTrue(
                torch.allclose(
                    unit,
                    torch.tensor([[0.0, 0.0], [0.6, 0.8]]),
                )
            )
            self.assertEqual(metadata["base_prompts"], 2)
            self.assertEqual(metadata["layers"], [0, 1])
            self.assertEqual(metadata["valid_layers"], [1])
            self.assertEqual(metadata["zero_norm_layers"], [0])
            self.assertEqual(metadata["direction_norms"], [0.0, 5.0])
            self.assertEqual(metadata["assistant_name"], "sycophantic")
            self.assertEqual(
                metadata["instruction_pair_counts"],
                {"0": 1, "1": 1},
            )

    def test_loads_selected_paired_prompt_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            torch.save(
                torch.tensor([[0.0, 0.0], [0.6, 0.8]]),
                output_dir / "directions.pt",
            )
            torch.save(
                torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
                output_dir / "raw_directions.pt",
            )
            (output_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "model": {"name": "fake/model", "revision": "test"},
                        "vector_type": "paired_prompt",
                        "layers": [0, 15],
                        "position": -1,
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "model": {"name": "fake/model", "revision": "test"},
                "vector": {
                    "type": "paired_prompt",
                    "layer": 15,
                    "position": -1,
                    "output_dir": str(output_dir),
                },
            }
            model = torch.nn.Linear(1, 1)

            direction, raw, metadata = _load_vector(config, model)

        self.assertTrue(torch.equal(direction, torch.tensor([0.6, 0.8])))
        self.assertTrue(torch.equal(raw, torch.tensor([3.0, 4.0])))
        self.assertEqual(metadata["layer"], 15)
        self.assertEqual(metadata["position"], -1)
        self.assertEqual(metadata["target_hook"], "blocks.15.hook_resid_pre")


if __name__ == "__main__":
    unittest.main()

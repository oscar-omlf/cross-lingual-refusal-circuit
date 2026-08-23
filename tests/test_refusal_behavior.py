from __future__ import annotations

import unittest

from scripts.run_refusal_circuit_behavior import (
    parse_wildguard_output,
    selection_metrics,
)


class WildGuardParsingTests(unittest.TestCase):
    def test_parses_standard_output(self):
        parsed = parse_wildguard_output(
            "Harmful request: yes\n"
            "Response refusal: no\n"
            "Harmful response: yes"
        )
        self.assertEqual(
            parsed,
            {
                "parse_ok": True,
                "request_harmful": True,
                "refusal": False,
                "response_harmful": True,
            },
        )

    def test_rejects_incomplete_output(self):
        self.assertFalse(
            parse_wildguard_output("Harmful request: yes")["parse_ok"]
        )


class RefusalBehaviorMetricTests(unittest.TestCase):
    def test_selection_metrics_are_paired(self):
        rows = []
        conditions = {
            "a": {
                "normal": True,
                "global_projection": False,
                "circuit_only": True,
                "circuit_ablated": False,
            },
            "b": {
                "normal": True,
                "global_projection": False,
                "circuit_only": False,
                "circuit_ablated": True,
            },
            "c": {
                "normal": False,
                "global_projection": False,
                "circuit_only": True,
                "circuit_ablated": False,
            },
        }
        for row_id, values in conditions.items():
            rows.extend(
                {
                    "id": row_id,
                    "label": "harmful",
                    "condition": condition,
                    "refusal": refusal,
                }
                for condition, refusal in values.items()
            )
        metrics = selection_metrics(rows)
        self.assertAlmostEqual(metrics["normal_refusal_rate"], 2 / 3)
        self.assertAlmostEqual(
            metrics["normalized_refusal_reversal"], 0.5
        )
        self.assertAlmostEqual(
            metrics["circuit_only_refusal_retention"], 0.5
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Split a frozen circuit into terminal/upstream masks for causal diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-circuit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference_circuit.read_text(encoding="utf-8"))
    edges = list(reference["edges"])
    terminal = [edge for edge in edges if edge.get("target", {}).get("type") == "residual_metric"]
    upstream = [edge for edge in edges if edge.get("target", {}).get("type") != "residual_metric"]
    if len(terminal) + len(upstream) != len(edges) or not terminal or not upstream:
        raise ValueError("Could not form non-empty terminal and upstream partitions")

    outputs = {"terminal_only": terminal, "upstream_only": upstream}
    for label, selected in outputs.items():
        payload = {
            "method": "component_control",
            "selection": label,
            "requested_k": int(reference["requested_k"]),
            "retained_edges": len(selected),
            "reference_circuit": str(args.reference_circuit),
            "reference_retained_edges": len(edges),
            "terminal_edges": sum(
                edge.get("target", {}).get("type") == "residual_metric" for edge in selected
            ),
            "upstream_edges": sum(
                edge.get("target", {}).get("type") != "residual_metric" for edge in selected
            ),
            "target": reference["target"],
            "edges": selected,
        }
        output_dir = args.output_root / label
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"eapig_top_{reference['requested_k']}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {label}: {len(selected)} edges to {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build controls with discovered terminal edges and randomized upstream edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-circuit", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()

    reference = _read(args.reference_circuit)
    reference_names = {str(edge["name"]) for edge in reference["edges"]}
    reference_terminal = {name for name in reference_names if name.endswith("->logits")}
    reference_upstream = reference_names - reference_terminal

    for seed in args.seeds:
        random_path = args.random_root / f"seed_{seed}" / "eapig_top_3000.json"
        random_payload = _read(random_path)
        random_names = {str(edge["name"]) for edge in random_payload["edges"]}
        random_upstream = {name for name in random_names if not name.endswith("->logits")}
        names = reference_terminal | random_upstream
        if len(reference_terminal) != 334 or len(random_upstream) != 2666 or len(names) != 3000:
            raise ValueError(
                f"Unexpected composition for seed {seed}: "
                f"terminal={len(reference_terminal)}, upstream={len(random_upstream)}, total={len(names)}"
            )
        upstream_overlap = len(random_upstream & reference_upstream)
        output_dir = args.output_root / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "fixed_terminal_random_upstream_control",
            "selection": "discovered_terminal_plus_topology_stratified_minimum_overlap_random_upstream",
            "requested_k": 3000,
            "retained_edges": len(names),
            "random_seed": seed,
            "reference_circuit": str(args.reference_circuit),
            "random_source_circuit": str(random_path),
            "terminal_edges": len(reference_terminal),
            "upstream_edges": len(random_upstream),
            "terminal_overlap_with_reference": len(reference_terminal),
            "upstream_overlap_with_reference": upstream_overlap,
            "total_overlap_with_reference": len(reference_terminal) + upstream_overlap,
            "target": reference["target"],
            "edges": [{"name": name} for name in sorted(names)],
        }
        output_path = output_dir / "eapig_top_3000.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"seed={seed} terminal={len(reference_terminal)} upstream={len(random_upstream)} "
            f"upstream_overlap={upstream_overlap} output={output_path}"
        )


if __name__ == "__main__":
    main()

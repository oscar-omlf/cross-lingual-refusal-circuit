#!/usr/bin/env python3
"""Fetch and hash the exact official Persona Vectors artifacts used by a config."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from _bootstrap import ROOT


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = config["official_source"]
    commit = source["commit"]
    root = _resolve(config["paths"]["official_root"])
    root.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/safety-research/persona_vectors/{commit}"
    artifacts = {
        "extract": source["extract_artifact"],
        "eval": source["eval_artifact"],
        "coherence_source": source["coherence_source"],
    }
    artifacts.update(source.get("implementation_sources", {}))
    rows = []
    for label, repository_path in artifacts.items():
        suffix = Path(repository_path).suffix
        output = root / f"{label}{suffix}"
        if output.exists() and not args.overwrite:
            status = "reused"
        else:
            request = urllib.request.Request(
                f"{base}/{repository_path}",
                headers={"User-Agent": "cross-lingual-refusal-circuit/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            output.write_bytes(payload)
            status = "downloaded"
        if suffix == ".json":
            json.loads(output.read_text(encoding="utf-8"))
        row = {
            "label": label,
            "repository_path": repository_path,
            "url": f"{base}/{repository_path}",
            "local_path": str(output.relative_to(ROOT)),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "status": status,
        }
        expected_sha256 = source.get("expected_sha256", {}).get(label)
        if expected_sha256 and row["sha256"] != expected_sha256:
            raise RuntimeError(
                f"hash mismatch for {label}: {row['sha256']} != {expected_sha256}"
            )
        rows.append(row)
    manifest = {
        "repository": source["repository"],
        "commit": commit,
        "artifacts": rows,
    }
    write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

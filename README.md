# Cross-lingual refusal circuit

This repository contains the core implementation and saved artifacts for
cross-lingual refusal-circuit and persona-direction experiments.

## Core layout

- `src/refusal_circuit/`: reusable circuit, intervention, attribution, data,
  metric, and persona-vector code.
- `results/`: saved model directions/weights, circuit score tables, masks,
  evaluations, plots, and frozen result manifests.
- `data/`: processed experiment inputs and provenance snapshots needed to
  interpret or reproduce saved results.
- `configs/`: the four refusal/hallucination prompt/generation experiments.
- `EAP-IG/`: vendored editable dependency required by the package.
- `tests/`: tests that exercise the retained core implementation without
  depending on archived orchestration scripts.

Historical runbooks, superseded experiment orchestration, cluster wrappers,
root-level benchmark scripts, and script-dependent tests were moved to
`archive/`. See `archive/README.md` for an inventory and restoration notes.

## Workflow

Three thin scripts wrap the existing source algorithms. Choose one of the four
trait/vector configs in `configs/`:

```bash
uv run --locked python scripts/extract_vector.py --config configs/refusal_prompt.json
uv run --locked python scripts/run_eapig.py --config configs/refusal_prompt.json
uv run --locked python scripts/run_tests.py --config configs/refusal_prompt.json
```

See `SYSTEM_DESIGN.md` for the small shared interface.

## Environment

The project targets Python 3.10. Dependency versions are pinned by
`pyproject.toml` and `uv.lock`.

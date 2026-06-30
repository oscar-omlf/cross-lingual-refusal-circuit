# Cross-lingual refusal circuit

This repository preserves Michael Hanna's EAP-IG implementation under
[`EAP-IG/`](EAP-IG/) and adds a modular RQ1 experiment for tracing a
refusal-direction representation in `google/gemma-2-2b-it`.

The original EAP-IG demo is [`EAP-IG/greater_than.ipynb`](EAP-IG/greater_than.ipynb).
The existing batch experiment entry point is [`test_ig_iterations.py`](test_ig_iterations.py),
which expects the original task CSVs and a CUDA-ready TransformerLens setup.

For the new project structure, environment, assumptions, commands, and output
definitions, see [the RQ1 runbook](docs/RQ1_SETUP.md).

For a concise presentation-oriented description of the completed RQ1 result,
see [the RQ1 presentation brief](docs/RQ1_PRESENTATION.md).

# Cross-lingual refusal circuit

This repository preserves Michael Hanna's EAP-IG implementation under
[`EAP-IG/`](EAP-IG/) and adds refusal-circuit experiments for tracing
refusal-direction representations in instruction-tuned LLMs.

The original EAP-IG demo is [`EAP-IG/greater_than.ipynb`](EAP-IG/greater_than.ipynb).
The existing batch experiment entry point is [`test_ig_iterations.py`](test_ig_iterations.py),
which expects the original task CSVs and a CUDA-ready TransformerLens setup.

For new experiment runs, use the unified entry point:

```bash
/home/omiro1/.local/bin/uv sync --locked
/home/omiro1/.local/bin/uv run --locked python scripts/rq.py list
```

UV manages Python 3.10.20 and the CUDA 12.1 PyTorch wheel in `.venv`; GPU jobs
still load the cluster's CUDA 12.8 module through the generic SLURM wrapper.

The old one-off scripts are still available under
[`scripts/legacy/`](scripts/legacy/) for exact reproducibility, but new work
should go through [`scripts/rq.py`](scripts/rq.py) and the generic SLURM wrapper
[`scripts/slurm/rq_job.sbatch`](scripts/slurm/rq_job.sbatch).

For the project structure, environment, assumptions, commands, and output
definitions, see [the RQ1 runbook](docs/RQ1_SETUP.md).

For a concise presentation-oriented description of the completed RQ1 result,
see [the RQ1 presentation brief](docs/RQ1_PRESENTATION.md).

For the current RQ1/RQ2 result inventory and headline tables, see
[the results summary](docs/RESULTS_SUMMARY.md).

For a brief narrative of the multilingual RQ2 result, see
[the RQ2 progress note](docs/RQ2_BRIEF_PROGRESS.md).

The executable Qwen refusal consolidation configuration is
`configs/phase2_qwen_refusal.json`. Phase 2 reuses saved Qwen score tables and
does not rerun refusal-vector extraction or EAP-IG discovery.

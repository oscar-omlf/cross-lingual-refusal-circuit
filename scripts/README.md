# Experiment scripts

Use the unified entry point for new work:

```bash
python scripts/rq.py list
python scripts/rq.py rq2.prepare-polyrefuse --languages en de zh th
python scripts/rq.py rq2.extract-direction --language en
python scripts/rq.py rq2.run-eapig --help
```

For SLURM jobs, prefer the generic launcher:

```bash
sbatch \
  --job-name=rq2-eapig-en \
  --time=06:00:00 \
  --mem=64G \
  --export=ALL,RQ_COMMAND=rq2.run-eapig,RQ_ARGS='--model Qwen/Qwen2.5-7B-Instruct --split train --num-prompts 64' \
  scripts/slurm/rq_job.sbatch
```

The older one-off Python and SLURM scripts are preserved under
[`scripts/legacy/`](legacy/) for exact reproducibility of previous runs.  New
experiments should add a command to `scripts/rq.py` rather than creating a new
top-level script for every run variant.

Useful command groups:

- `rq1.*`: Gemma single-language refusal-direction circuit experiments.
- `rq2.*`: Qwen multilingual refusal-direction/circuit experiments.
- `rq.*`: cross-RQ diagnostics.
- `results.summary`: regenerate the concise results summary in `docs/`.

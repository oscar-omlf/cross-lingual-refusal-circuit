# Persona circuits (RQ-persona)

This reuses the RQ1 refusal circuit pipeline to extract and validate circuits
for **persona traits** (evil, sycophancy, hallucination) defined by the
persona-vectors method (`safety-research/persona_vectors`).

## Why this is a small change

Everything after direction extraction in RQ1 is direction-agnostic. The EAP-IG
metric, circuit selection, and faithfulness/necessity only read two artefacts:

- `persona_direction.pt` — a single `[d_model]` unit vector.
- `persona_direction_metadata.json` — `{model, layer, position, target_hook, ...}`.

So the only genuinely new code is a front end that produces those two files for
a persona trait. `scripts/rq1_run_eapig.py` and
`scripts/rq1_evaluate_faithfulness.py` are reused unchanged apart from a new,
backward-compatible `--processed-dir` flag.

## No OpenAI required

persona-vectors uses a `gpt-4.1-mini` judge only to (a) filter response-based
vectors by coherence/trait score and (b) score trait expression during its own
evaluation. We use **prompt-based** extraction: the direction is the
difference-in-means of `resid_pre` between persona-positive and
persona-negative *prompts* at the last prompt token (persona-vectors'
`prompt_last` variant). No generation and no judge is involved. Circuit
faithfulness/necessity are projection-based and need no judge either.

Gemma-2 has no system role, so each persona instruction is folded into the
user turn (`"{instruction}\n\n{question}"`).

## One-time setup

```bash
# Trait prompt files only (no OpenAI, no vLLM needed to read them):
git clone https://github.com/safety-research/persona_vectors external/persona_vectors
```

`external/` is untracked by this repo. Alternatively pass
`--trait-data-path /path/to/<trait>.json` to the prepare script.

## Pipeline (trait = evil, model = google/gemma-2-2b-it)

```bash
export TRAIT=evil
export HF_TOKEN=...            # for the GPU jobs; never commit it
mkdir -p logs/persona          # Slurm writes --output at submit time; dir must exist

# 1. Build pos/neg prompt splits (CPU-only; can run on a login node)
sbatch scripts/slurm/persona_prepare_dataset.sbatch
#   -> data/processed/persona/evil/{train,val,test}.jsonl

# 2. Extract the persona direction (difference-in-means, judge-free)
sbatch scripts/slurm/persona_extract_vector.sbatch
#   -> results/persona/evil/persona_direction.pt
#   -> results/persona/evil/persona_direction_metadata.json
#   -> results/persona/evil/persona_direction_validation.csv
#   -> results/persona/evil/evil_prompt_last_diff.pt   (per-layer raw stack)

# 3. Score edges + extract circuits with the RQ1 EAP-IG metric
sbatch scripts/slurm/persona_eapig.sbatch
#   -> results/persona/evil/{eapig,eap}_edge_scores.parquet
#   -> results/persona/evil/circuits/{eapig,eap}_top_*.json

# 4. Faithfulness + necessity + random baselines
sbatch scripts/slurm/persona_faithfulness.sbatch
#   -> results/persona/evil/{faithfulness,necessity}.csv
#   -> results/persona/evil/plots/*.png
```

Run another trait by setting `TRAIT=sycophancy` (or `hallucination`) before
each `sbatch`, assuming the matching `<trait>.json` exists in the persona-vectors
checkout.

## Layer selection

`persona_extract_vector.py` builds one difference-in-means candidate per layer
(at the last prompt token) and selects the layer whose direction best separates
held-out pos/neg projections (max Cohen's d), constrained to
`1 <= layer < 0.8 * n_layers` so the EAP projection graph is valid and
non-trivial. Inspect `persona_direction_validation.csv` to see every layer's
separation and override with `--min-layer/--max-layer` if you want a specific
layer (e.g. to match the persona-vectors paper's choice).

## Later: persona behavioural evaluation

The RQ1 behavioural sanity check matches *refusal* prefixes, so persona jobs
pass `--skip-behaviour`. A trait-appropriate behavioural probe (steering with
the per-layer diff stack, or a local open-model trait judge over
`data_generation/trait_data_eval/<trait>.json`) can be added later without
touching the circuit pipeline.

# Qwen Hallucination Persona-Circuit Runbook

This reproduces the Hallucination experiment with the same methodology used for
Sycophancy: Anthropic-compatible response-average vector extraction, an upstream
formation circuit, a downstream expression circuit, their union, and internal
faithfulness/necessity plus behavioral necessity/sufficiency curves. Do not use
the old projection-ablation circuit scripts for this pipeline.

## 0. Environment and immutable settings

Run from the repository root. Secrets belong only in the Git-ignored `.env`
(`OPENAI_API_KEY=...`, optionally `HF_TOKEN=...`); never commit them.

```bash
module purge
module load 2025
module load CUDA/12.8.0
uv sync --locked

CONFIG=configs/qwen_hallucination.json
INTERMEDIATE=data/intermediate/persona/qwen2.5-7b-instruct/hallucinating
RESULTS=results/qwen2.5-7b-instruct/persona/hallucinating
MANIFEST=$INTERMEDIATE/shared_circuit_dataset_v1.json
UP=hallucination_shared_n64_ig5_v1
DN=hallucination_shared_n64_ig5_cneg_v1_lowmem_v1
QUESTIONS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
K_ALL="50 100 250 500 1000 2000 3000"
K_SUFF="50 250 1000 3000"
```

GPU commands must use a full A100:

```bash
sbatch --partition=gpu_a100 --gres=gpu:a100:1 --cpus-per-task=18 \
  --mem=64G --time=12:00:00 --job-name=NAME \
  --export="ALL,RQ_COMMAND=COMMAND,RQ_ARGS=ARGUMENTS" \
  scripts/slurm/rq_job.sbatch
```

Use `--dependency=afterok:JOBID` between dependent stages. API judging is CPU
only: submit the same launcher with `--partition=rome --gres=none
--cpus-per-task=4 --mem=16G`. Every command below is an `RQ_COMMAND`; its
following line is `RQ_ARGS`.

## 1. Reproduce and validate the persona vector

Fetch and audit the exact Persona Vectors artifacts pinned in the config:

```bash
uv run python scripts/rq.py persona.fetch --config "$CONFIG"
uv run python scripts/rq.py persona.audit --config "$CONFIG"
```

Generate the 1,000 positive and 1,000 negative responses on an A100:

```text
persona.generate
--config configs/qwen_hallucination.json --polarity both --resume
```

Audit generation, then judge both polarities with the canonical dated
GPT-4.1-mini protocol:

```bash
uv run python scripts/rq.py persona.audit-generation --config "$CONFIG"

uv run python scripts/rq.py persona.judge --config "$CONFIG" --artifact extract \
  --input "$INTERMEDIATE/generation/pos_responses.jsonl" \
  --output "$INTERMEDIATE/judged/pos_scored.jsonl" --resume

uv run python scripts/rq.py persona.judge --config "$CONFIG" --artifact extract \
  --input "$INTERMEDIATE/generation/neg_responses.jsonl" \
  --output "$INTERMEDIATE/judged/neg_scored.jsonl" --resume
```

Extract the filtered all-layer response-average difference vector on an A100:

```text
persona.extract
--config configs/qwen_hallucination.json
```

The primary artifacts must be
`$RESULTS/vector/hallucinating_response_avg_diff.pt` and
`$RESULTS/vector/hallucinating_direction.pt`. The extraction metadata records
the effective paired sample count.

## 2. Validate the Hallucination layer

Run `persona.steer` independently for layers `4 8 12 16 20 24 28`, using the
config's coefficients and full held-out rollout count:

```text
persona.steer
--config configs/qwen_hallucination.json --layer LAYER --resume
```

After all seven A100 jobs finish, judge and aggregate the sweep on CPU:

```bash
uv run python scripts/rq.py persona.judge-layer-sweep --config "$CONFIG"
uv run python scripts/rq.py persona.select-layer --config "$CONFIG"
```

Layer 16 is frozen in the config because it wins at the behaviorally useful
coefficients in the completed sweep. Stop if a fresh replication materially
contradicts that selection; update all four layer-mapping fields together only
after review.

## 3. Freeze one shared discovery/evaluation split

This uses the same unique prompt identities for upstream and downstream
discovery, plus 32 response-disjoint pairs for downstream internal evaluation.
For Hallucination, the strict token-alignment gate currently leaves 33 unique
discovery prompt identities, not 64. The official `eval.json` questions remain
disjoint behavioral evaluation data.

```bash
uv run python scripts/rq.py persona.freeze-shared-circuit-dataset \
  --config "$CONFIG" --discovery-prompts 33 --downstream-eval-pairs 32 \
  --seed 20260726 --max-transcript-length 768
```

Never regenerate or edit `$MANIFEST` after circuit discovery begins.

## 4. Upstream formation circuit

Rank the entire graph from model input through the configured layer-16 vector
site. Clean/corrupt endpoints are matched positive/negative persona prompts;
the scalar is final-prompt projection onto the unit Hallucination vector.

```text
persona.run-upstream-contrastive-eapig
--config configs/qwen_hallucination.json --run-name hallucination_shared_n64_ig5_v1 --sample-manifest data/intermediate/persona/qwen2.5-7b-instruct/hallucinating/shared_circuit_dataset_v1.json --num-pairs 33 --batch-size 1 --max-length 512 --ig-steps 5 --k 3000
```

Derive all nested masks from the single frozen score table (CPU; do not rerun
EAP-IG):

```bash
uv run python scripts/rq.py persona.rebuild-nested-circuits --config "$CONFIG" \
  --scope upstream --run-name "$UP" --k-values $K_ALL 5000 10000 20000
```

For every `k` in `$K_ALL` (and higher values only if the internal sufficiency
curve has not plateaued), run on A100:

```text
persona.evaluate-upstream-contrastive-circuit
--config configs/qwen_hallucination.json --run-name hallucination_shared_n64_ig5_v1 --k K --num-pairs 32 --seed 1 --max-length 512
```

Generate the first 20-question behavioral artifact at `k=3000`; this also
creates the reusable greedy normal baseline. The artifact may contain fewer
than 20 rows only if a question/instruction pair still fails the shared
token-alignment gate:

```text
persona.behavior-upstream-contrastive-circuit
--config configs/qwen_hallucination.json --run-name hallucination_shared_n64_ig5_v1 --k 3000 --output-tag k3000_qall_max640_necessity_final --question-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --max-new-tokens 640 --max-length 512 --conditions normal circuit_ablated --resume
```

Judge it with `persona.judge --artifact eval --protocol-field
behavior_protocol_id`. Its paths are:

```text
$INTERMEDIATE/natural_upstream_circuit_behavior/$UP/k3000_qall_max640_necessity_final/responses.jsonl
$INTERMEDIATE/natural_upstream_circuit_behavior/$UP/k3000_qall_max640_necessity_final/scored.jsonl
```

Use that `responses.jsonl` as `--baseline-responses` while sweeping upstream
necessity (`normal circuit_ablated`) for every `$K_ALL`, and circuit-only
sufficiency (`normal circuit_only`) for `$K_SUFF`. Judge every completed output.

## 5. Validate the signed-vector counterfactual

Before downstream attribution, verify that subtracting the vector actually
reduces Hallucination behavior. Use the judged upstream normal responses:

```text
persona.vector-diagnostic
--config configs/qwen_hallucination.json --source data/intermediate/persona/qwen2.5-7b-instruct/hallucinating/natural_upstream_circuit_behavior/hallucination_shared_n64_ig5_v1/k3000_qall_max640_necessity_final/responses.jsonl --source-scored data/intermediate/persona/qwen2.5-7b-instruct/hallucinating/natural_upstream_circuit_behavior/hallucination_shared_n64_ig5_v1/k3000_qall_max640_necessity_final/scored.jsonl --num-questions 8 --max-new-tokens 640 --negative-coefficients -1.5 -2.0 --output-root data/intermediate/persona/qwen2.5-7b-instruct/hallucinating/vector_intervention_diagnostic/top8_high_baseline
```

Judge its `responses.jsonl`. Select the smallest negative coefficient that
substantially lowers judged Hallucination while preserving coherence; call it
`CNEG`. Do not run downstream EAP-IG if neither coefficient passes this gate.

## 6. Downstream expression circuit

Rank every edge from the layer-16 vector interface to logits. For each of the
same 33 manifest prompts, the objective is mean log-probability of its
high-Hallucination answer minus its low-Hallucination answer under the identical
positive prompt; edge scores are averaged across all pairs and five
coefficient-path IG steps. This step is memory-bound on 40 GB A100 nodes, so
use the low-memory rerun settings below rather than the initial 768-token draft.
If 256 still OOMs on your cluster, retry at 128; keep the same manifest and
coefficient so the run remains provenance-equivalent.

```text
persona.run-steering-eapig
--config configs/qwen_hallucination.json --run-name hallucination_shared_n64_ig5_cneg_v1_lowmem_v1 --sample-manifest data/intermediate/persona/qwen2.5-7b-instruct/hallucinating/shared_circuit_dataset_v1.json --num-prompts 33 --batch-size 1 --max-length 256 --ig-steps 5 --corrupt-coefficient CNEG --minimum-trait-score 80 --maximum-negative-trait-score 20 --minimum-coherence-score 80 --metric contrastive_response_logprob --k-values 3000
```

Derive nested masks on CPU:

```bash
uv run python scripts/rq.py persona.rebuild-nested-circuits --config "$CONFIG" \
  --scope downstream --run-name "$DN" --k-values $K_ALL
```

For every `$K_ALL`, evaluate the 32 response-disjoint manifest pairs:

```text
persona.evaluate-steering-circuit
--config configs/qwen_hallucination.json --run-name hallucination_shared_n64_ig5_cneg_v1_lowmem_v1 --sample-manifest data/intermediate/persona/qwen2.5-7b-instruct/hallucinating/shared_circuit_dataset_v1.json --k K --num-prompts 32 --batch-size 1 --max-length 256
```

Behaviorally sweep necessity (`normal circuit_ablated`) for `$K_ALL` and
sufficiency (`normal circuit_only`) for `$K_SUFF`, always using all 20 questions,
640 tokens, and the upstream normal artifact as `--baseline-responses`:

```text
persona.behavior-steering-circuit
--config configs/qwen_hallucination.json --run-name hallucination_shared_n64_ig5_cneg_v1_lowmem_v1 --k K --output-tag kK_qall_max640_REGIME_final --question-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --max-new-tokens 640 --max-length 256 --baseline-responses BASELINE_RESPONSES --conditions normal CONDITION --resume
```

Judge every output with the canonical eval rubric.

## 7. Union and final curves

For every `$K_ALL`, run union necessity; for `$K_SUFF`, run union sufficiency:

```text
persona.behavior-connected-circuits
--config configs/qwen_hallucination.json --upstream-run hallucination_shared_n64_ig5_v1 --downstream-run hallucination_shared_n64_ig5_cneg_v1 --upstream-k K --downstream-k K --condition both_ablated --question-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --output-tag union_both_ablated_kK_qall_max640_final --max-new-tokens 640 --max-length 512 --resume
```

Replace `both_ablated` with `both_circuit_only` and use a distinct output tag
for sufficiency. Judge every union output. If upstream high-`k` sufficiency is
still improving, additionally pair upstream `5k/10k/20k` with downstream
`k=3000`; do not rerank either circuit.

Finally build completeness-gated paper artifacts on CPU, passing the judged
normal baseline explicitly:

```bash
uv run python scripts/rq.py persona.analyze-causal-curves --config "$CONFIG" \
  --upstream-run "$UP" --downstream-run "$DN" \
  --k-values $K_ALL --upstream-extra-k-values 5000 10000 20000 \
  --sufficiency-k-values $K_SUFF --question-indices $QUESTIONS \
  --minimum-token-cap 640 --baseline-scored BASELINE_SCORED \
  --output-name paper_suite_final_causal_curves
```

The analyzer refuses to mark the suite complete if any registered internal,
necessity, or sufficiency cell is missing. Preserve every score table, circuit
JSON, generation metadata file, and judge metadata file; generated results are
intentionally Git-ignored.

JOBID=$(sbatch \
  --partition=gpu_a100 --gres=gpu:a100:1 --cpus-per-task=18 --mem=64G --time=12:00:00 \
  --job-name=hall-dn-eval-k50 \
  --export="ALL,RQ_COMMAND=persona.evaluate-steering-circuit,RQ_ARGS=--config $CONFIG --run-name hallucination_shared_n64_ig5_cneg_v1_lowmem_v1 --sample-manifest $MANIFEST --k 50 --num-prompts 32 --batch-size 1 --max-length 256" \
  scripts/slurm/rq_job.sbatch | awk '{print $4}')
echo "Submitted: $JOBID"

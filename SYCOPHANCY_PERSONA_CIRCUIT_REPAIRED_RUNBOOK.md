# Repaired Qwen opinion-sycophancy circuit runbook

Status: executable protocol for the methodology repair on
`handoff/persona-circuits-20260811`, written 2026-08-12.

This document is the authoritative remaining-experiment runbook for the Qwen
opinion-sycophancy study. It assumes the canonical shared HPC checkout, where
the Git-ignored judged responses, discovery manifests, score tables, and
historical behavior artifacts are available. A fresh Git clone alone does not
contain those generated artifacts.

The frozen multilingual refusal result is not rerun. The existing sycophancy
vector extraction, layer selection, steering validation, shared discovery
manifest, IG-5 upstream score table, IG-5 downstream score table, and historical
necessity results are reused unless a provenance check or IG-convergence gate
fails.

## 1. What changed

The original method selected a backward greedy edge shell with parentless
pruning disabled. Such an edge set can intercept clean computation when
ablated while failing circuit-only reconstruction because many selected source
nodes receive only corrupt inputs. The repaired method separates:

1. `causal_cut`: sink-connected edges used for necessity, interception, and
   cut-to-interface mediation.
2. `source_to_sink`: edges lying on complete model-input-to-target paths, used
   for circuit-only faithfulness and reconstructive sufficiency.
3. `end_to_end`: an independently attributed natural-prompt circuit from
   `blocks.0.hook_resid_pre` to response-token logits.

The publication claims are correspondingly nested:

1. **Sparse causal interception:** selected-edge ablation changes behavior.
2. **Low-dimensional causal interface:** the upstream-cut effect is rescued by
   restoring the missing persona-vector coordinate, is not rescued by matched
   orthogonal controls, and is blocked by the downstream cut.
3. **Reconstructive mechanism:** complete upstream and downstream circuits are
   circuit-only faithful, and their connected union reproduces the behavior.

The code repair also added:

- deterministic score tie-breaking and complete-path connectivity diagnostics;
- CPU-only circuit rebuilding from immutable score tables;
- prompt-disjoint upstream and prompt/response-disjoint downstream evaluation;
- exact answer-token-boundary and duplicate-pair rejection;
- no-truncation teacher-forced evaluation;
- prompt-only versus recurrent upstream intervention timing;
- same-backend `normal`, upstream-only, downstream-only, and joint ablation;
- full-prefix, full-final-state, vector, orthogonal, and unrelated-vector rescue;
- rescue-effect downstream blocking, without confounding negative steering;
- a separately ranked end-to-end natural-prompt comparator;
- topology-stratified retained-size-matched circuit controls;
- fixed-row IG convergence comparisons;
- exact generated-token hashes and model/vector/circuit provenance;
- deterministic split pooling under a common token cap;
- paired bootstrap mediation and 2x2 interaction analysis.

### 1.1 Exact implementation map

The implementation changes are localized as follows:

| Files | Exact change |
| --- | --- |
| `src/refusal_circuit/graph.py` | Deterministic score tie-breaking; explicit source/sink reachability; separate sink-connected cut and complete source-to-sink pruning; retained-edge diagnostics. |
| `src/refusal_circuit/circuit.py` | Prompt-position masks for edge patching; natural prompt-pair logit patching for end-to-end evaluation and generation. |
| `src/refusal_circuit/eapig.py` | Natural positive-versus-negative prompt EAP-IG to a paired high-minus-low response-logprob target. |
| `src/refusal_circuit/model.py`, persona configs | Pin and validate graph-relevant model architecture before attribution or evaluation. |
| Upstream/downstream discovery and evaluation scripts | Emit both circuit families; reject truncation; use frozen disjoint manifests; validate circuit connectivity and provenance. |
| Behavioral runners | Use one full-prefix ungrouped-GQA backend; record exact token hashes; reject stale vectors/circuits and incompatible resumes. |
| `persona_behavior_connected_circuits.py` | Add same-backend 2x2 conditions, prompt-only versus recurrent timing, full-prefix/full-state/vector rescue, five orthogonal seeds, unrelated-vector rescue, and rescue-specific downstream blocking. |
| New end-to-end scripts | Independently discover, evaluate, generate with, and analyze a natural-prompt source-to-logit comparator. |
| New freeze/audit/pooling scripts | Freeze disjoint evaluation rows, audit token lengths, enforce exact answer boundaries, and pool deterministic split generations only when causal semantics and caps match. |
| New analysis/control scripts | Compare IG score and complete-path stability, build topology-stratified controls, bootstrap internal curves, and estimate mediation/interaction effects. |
| `tests/test_persona.py` | Regression tests for cut-versus-circuit topology, deterministic selection, prompt masks, rescue estimands, pooling provenance, and token hashing. |

What did **not** change: the canonical model, model revision, official trait
data, judged extraction responses, vector extraction, selected interface layer,
steering coefficient, frozen 64-row discovery identities, or the numerical
contents of the existing IG-5 score tables.

## 2. Fixed method

### 2.1 Model and trait

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Dtype: bfloat16
- Trait: opinion sycophancy
- Interface: `blocks.20.hook_resid_pre`
- Unit direction: `vector/sycophantic_direction.pt`
- Raw response vector: `vector/sycophantic_response_avg_diff.pt`
- Downstream corrupt coefficient: `-2`
- Generation: deterministic greedy decoding
- Behavioral cap: `max_new_tokens=350`
- Input/prefill cap: `max_length=1024`
- Internal downstream/end-to-end cap: `max_length=768`, with no truncation

### 2.2 Discovery and evaluation splits

- Discovery: 64 frozen extraction-prompt identities.
- Upstream k selection: 32 new extraction-prompt identities disjoint from
  discovery and from official behavioral questions.
- Downstream/end-to-end k selection: 32 prompt- and response-pair-disjoint
  teacher-forced pairs.
- Behavioral evaluation: the 20 official `eval.json` questions are disjoint
  from discovery and internal k selection, but historical ablation curves have
  already used them. For the new mediation estimands, q0-q4 are a diagnostic
  pilot and q5-q19 are the primary analysis. Do not call this an untouched
  benchmark; add a newly frozen external question set for the ICLR version.
- Circuit k is selected from internal metrics before viewing official
  behavioral judge scores.

### 2.3 Internal estimands

Upstream:

`final-prompt projection onto the unit persona direction`

Clean is the positive-persona prompt and corrupt is its token-aligned
negative-persona counterpart.

Downstream:

`mean_logp(high-trait answer) - mean_logp(low-trait answer)`

Both answers are teacher-forced under the same positive-persona prompt.
Corruption is response-time steering from coefficient `0` to `-2`.

End-to-end:

`mean_logp(high-trait answer) - mean_logp(low-trait answer)`

The same answer tokens are held fixed while the prompt changes from the
positive-persona prompt to its token-aligned negative-persona counterpart.

For every circuit report:

- requested frontier k;
- retained complete-path edges;
- retained/eligible edge fraction;
- clean, corrupt, circuit-only, and circuit-ablated means;
- normalized faithfulness and necessity;
- source/sink connectivity diagnostics;
- paired uncertainty over held-out identities.

### 2.4 Primary mediation estimand

The primary interface experiment uses:

```text
--upstream-position-scope prompt_only
--interface-timing prompt_end
```

The causal chain is:

1. upstream cut lowers the interface coordinate and sycophantic behavior;
2. full-prefix rescue exactly reproduces same-backend normal generation;
3. final-prompt full-state rescue tests a single-state interface;
4. vector rescue tests a one-dimensional interface;
5. five random orthogonal directions and one unrelated persona direction test
   specificity;
6. downstream blocking tests whether the selected downstream set transmits the
   rescue-induced effect;
7. the same-backend 2x2 conditions estimate upstream, downstream, joint, and
   interaction effects.

## 3. What must not be rerun

Reuse these when their status and hashes pass:

- official Persona Vectors source artifacts;
- 1,000 positive and 1,000 negative sycophancy extraction generations;
- canonical GPT-4.1-mini judgments;
- the 810 effective response pairs;
- response-average vector extraction;
- layer-20 selection and steering validation;
- `shared_circuit_dataset_v1.json`;
- upstream IG-5 score table
  `sharedprompt_n64_ig5_k3000_20260726`;
- downstream IG-5 score table
  `sharedprompt_n64_ig5_full768_h100_cneg2_20260802`;
- frozen multilingual refusal results;
- historical sycophancy cut-set necessity results.

Do not finish the legacy q7-q19 connected circuit-only run unless it is needed
as an explicitly labeled historical cut-set diagnostic. Rebuilt circuits have
different edge sets, so old circuit-only generations cannot be reused for the
reconstructive claim.

## 4. A100 planning estimates

These are conservative planning ranges for one 40 GB A100. They include model
load and hook overhead but exclude queue time and OpenAI judging. Actual time
must be recorded with `sacct`. An 80 GB A100 improves memory headroom but should
not be budgeted as a large throughput improvement.

| Phase | Runs | Estimated A100 hours |
| --- | ---: | ---: |
| Technical smoke suite | connected + internal + end-to-end | 2-6 |
| IG-5 topology/internal decision cells | 5 jobs | 2-6 |
| Five-question mediation preflight | 14 conditions | 5-12 |
| Upstream IG 5/20/50 pilots plus held-out checks | 6 jobs | 2-6 |
| Downstream IG 5/20/50 pilots plus held-out checks | 6 jobs | 8-20 |
| End-to-end IG 5/20/50 pilots plus held-out checks | 6 jobs | 10-26 |
| Full upstream rerun, only if IG-5 fails | one n=64 run | 3-10 |
| Upstream 50k/100k frontier extension, only if complete paths stay below 5% | 1-2 held-out evals; no reranking | 2-6 |
| Full downstream rerun, only if IG-5 fails | one n=64 run | 8-28 |
| Final upstream internal curve | up to 10 k values | 2-5 |
| Final downstream internal curve | up to 9 k values | 9-20 |
| Full end-to-end discovery | one n=64 run | 10-40 |
| End-to-end internal curve | 6 k values | 8-20 |
| Five topology controls per selected family | up to 15 evals | 8-22 |
| Mediation q0-4 pilot plus q5-19 primary | 14 conditions | 25-60 |
| End-to-end q0-4 pilot plus q5-19 primary | 4 conditions | 8-20 |
| Connected source-to-sink circuit-only, conditional | normal + circuit-only | 5-14 |
| Recurrent timing diagnostic, conditional | 4-5 conditions | 8-22 |

Expected totals:

- Decision gates before final expansion: about **35-80 A100-hours**.
- Complete paper suite if the existing IG-5 rankings converge: about
  **100-210 A100-hours**.
- If downstream or end-to-end requires IG-50, or all conditional reconstruction
  experiments are run: about **170-330 A100-hours**.

With 8 independent A100s, most jobs can be parallelized. The longest single
jobs are full downstream/end-to-end attribution and long circuit-only
generation; request 24 hours for those jobs even when the expected runtime is
lower.

These are allocated GPU-hours, not elapsed wall-clock hours. Because every
listed GPU job requests one GPU, its measured GPU-hours are simply
`ElapsedRaw / 3600`. Replace the estimates after the smoke wave with the actual
per-row and per-token rates from `sacct`; do not extrapolate from queue time.

Suggested three-day critical path with at least eight A100s:

| Time | Launch only after | Parallel work |
| --- | --- | --- |
| Day 1 morning | CPU audit complete | Technical smoke suite. |
| Day 1 afternoon/night | Smoke positive controls pass | IG-5 decision cells, 14-condition q0-q4 mediation pilot, and all nine IG convergence attributions. |
| Day 2 morning | Pilot jobs complete | Nine convergence held-out checks and score comparisons; freeze `UP_STEPS`, `DN_STEPS`, and `E2E_STEPS`. |
| Day 2 afternoon/night | Step decisions frozen | Conditional full upstream/downstream reranking, canonical internal curves, and full end-to-end discovery. |
| Day 3 morning | Canonical curves pass their gates | Topology controls, q5-q19 mediation, and end-to-end behavior. |
| Day 3 afternoon/night | Generation artifacts complete | Pool, judge, analyze, then run only the triggered reconstruction/timing diagnostics. |

Do not launch final q5-q19 generation while a ranking or five-question
mediation gate is unresolved. Parallelize independent scopes, not dependent
decisions.

## 5. Shell setup

Run from the repository root on HPC.

```bash
test -n "${BASH_VERSION:-}" || {
  echo "Run this protocol in Bash 4+ so arrays and parameter slicing are exact."
  exit 1
}
for TOOL in git uv jq sbatch squeue sacct awk find xargs paste tail; do
  command -v "$TOOL" >/dev/null || {
    echo "Missing required command: $TOOL"
    exit 1
  }
done
git switch handoff/persona-circuits-20260811
uv sync --locked
mkdir -p logs/rq
test -f .env || {
  echo "Create the Git-ignored .env with HF_TOKEN and, before judging, OPENAI_API_KEY."
  exit 1
}
chmod 600 .env
```

`.env` must contain `HF_TOKEN` and, before judging, `OPENAI_API_KEY`.

Define the immutable paths:

```bash
export CONFIG=configs/qwen_sycophancy.json
export SHARED=data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/shared_circuit_dataset_v1.json
export UP_EVAL=data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/upstream_prompt_disjoint_eval_v1.json
export DN_EVAL=data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/downstream_prompt_response_disjoint_eval_v2.json

export RESULTS=results/qwen2.5-7b-instruct/persona/sycophantic
export INTERMEDIATE=data/intermediate/persona/qwen2.5-7b-instruct/sycophantic

export UP_IG5=sharedprompt_n64_ig5_k3000_20260726
export DN_IG5=sharedprompt_n64_ig5_full768_h100_cneg2_20260802

export UNRELATED_VECTOR=results/qwen2.5-7b-instruct/persona/hallucinating/vector/hallucinating_response_avg_diff.pt

export Q5="0 1 2 3 4"
export Q15="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
export Q20="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
export PILOT16="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
```

Define a submission helper:

```bash
submit_a100 () {
  local job_name="$1"
  local time_limit="$2"
  local rq_command="$3"
  local rq_args="$4"
  RQ_COMMAND="$rq_command" RQ_ARGS="$rq_args" \
    sbatch \
      --job-name="$job_name" \
      --partition=gpu_a100 \
      --gres=gpu:a100:1 \
      --cpus-per-task=8 \
      --mem=64G \
      --time="$time_limit" \
      --export=ALL,RQ_COMMAND,RQ_ARGS \
      scripts/slurm/rq_job.sbatch
}
```

Record job accounting after each wave:

```bash
squeue -u "$USER"
sacct -X --starttime 2026-08-12 \
  --format=JobID,JobName%32,State,Elapsed,AllocTRES,MaxRSS,ExitCode
```

## 6. Prerequisite artifact audit

Run this before allocating a GPU:

```bash
required=(
  data/raw/persona_vectors/b8e0f044fe2410a6fad579f38324f03f13b4e917/extract.json
  data/raw/persona_vectors/b8e0f044fe2410a6fad579f38324f03f13b4e917/eval.json
  "$INTERMEDIATE/judged/pos_scored.jsonl"
  "$INTERMEDIATE/judged/neg_scored.jsonl"
  "$SHARED"
  "$RESULTS/vector/effective_pairs.csv"
  "$RESULTS/vector/sycophantic_direction.pt"
  "$RESULTS/vector/sycophantic_response_avg_diff.pt"
  "$RESULTS/circuit_discovery_upstream_contrastive/$UP_IG5/eapig_run_metadata.json"
  "$RESULTS/circuit_discovery_upstream_contrastive/$UP_IG5/eapig_edge_scores.parquet"
  "$RESULTS/circuit_discovery_steering_path/$DN_IG5/eapig_run_metadata.json"
  "$RESULTS/circuit_discovery_steering_path/$DN_IG5/eapig_edge_scores.parquet"
  "$UNRELATED_VECTOR"
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "MISSING: $path"; exit 1; }
done
```

Check completed discovery status:

```bash
jq -e '.status == "complete"' \
  "$RESULTS/circuit_discovery_upstream_contrastive/$UP_IG5/eapig_run_metadata.json"
jq -e '.status == "complete"' \
  "$RESULTS/circuit_discovery_steering_path/$DN_IG5/eapig_run_metadata.json"
```

If any generated artifact is missing, restore the canonical artifact bundle
from the shared HPC checkout. Do not silently regenerate only part of the
frozen protocol.

Run static validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/persona-pycache \
  python3 -m compileall -q src scripts tests EAP-IG/eap
git diff --check
```

## 7. CPU preparation

### 7.1 Freeze paper evaluation manifests

These commands are deterministic. If the files already exist, they verify exact
equality.

```bash
uv run python scripts/rq.py persona.freeze-upstream-evaluation \
  --config "$CONFIG" \
  --discovery-manifest "$SHARED" \
  --num-prompts 32 \
  --seed 20260811 \
  --max-length 256 \
  --output "$UP_EVAL"

uv run python scripts/rq.py persona.freeze-downstream-evaluation \
  --config "$CONFIG" \
  --discovery-manifest "$SHARED" \
  --num-pairs 32 \
  --seed 20260811 \
  --max-length 768 \
  --output "$DN_EVAL"
```

Expected protocols:

```bash
jq -e '.status == "frozen" and .protocol == "prompt_disjoint_upstream_evaluation_v1" and .selection.n_prompts == 32' "$UP_EVAL"
jq -e '.status == "frozen" and .protocol == "prompt_response_disjoint_downstream_evaluation_v2" and .selection.n_pairs == 32' "$DN_EVAL"
```

### 7.2 Audit token lengths

```bash
uv run python scripts/rq.py persona.audit-token-lengths \
  --config "$CONFIG" \
  --sample-manifest "$SHARED" \
  --upstream-evaluation-manifest "$UP_EVAL" \
  --downstream-evaluation-manifest "$DN_EVAL" \
  --caps 256 350 500 640 768 1024 \
  --output "$RESULTS/token_length_audit_repaired_v1.json"
```

Required result:

- upstream prompt families fit 256;
- downstream/end-to-end teacher-forced families fit 768;
- official behavioral prompts fit 1024.

Different historical caps do not require rerunning a teacher-forced row when
the complete transcript fit under every cap. Free-generation cap hits are not
reusable as complete responses.

### 7.3 Rebuild both circuit families from frozen scores

CPU only:

```bash
uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" \
  --scope upstream \
  --run-name "$UP_IG5" \
  --families source_to_sink causal_cut \
  --k-values 50 100 250 500 1000 2000 3000 5000 10000 20000

uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" \
  --scope downstream \
  --run-name "$DN_IG5" \
  --families source_to_sink causal_cut \
  --k-values 50 100 250 500 1000 2000 3000 5000 10000
```

Inspect retained edges:

```bash
jq -r '.circuits[] | [.family,.k,.retained_edges,.connectivity.all_edges_on_source_to_sink_paths,.path] | @tsv' \
  "$RESULTS/circuit_discovery_upstream_contrastive/$UP_IG5/derived_upstream_circuit_families_manifest.json"
jq -r '.circuits[] | [.family,.k,.retained_edges,.connectivity.all_edges_on_source_to_sink_paths,.path] | @tsv' \
  "$RESULTS/circuit_discovery_steering_path/$DN_IG5/derived_downstream_circuit_families_manifest.json"
```

For `source_to_sink`, every non-empty circuit must report
`all_edges_on_source_to_sink_paths=true`.

## 8. Small technical smoke

This is the first GPU wave. Do not start the full experiment until every smoke
check passes.

### 8.1 Freeze four-row smoke manifests

CPU only:

```bash
export UP_SMOKE="$INTERMEDIATE/upstream_prompt_disjoint_eval_smoke4_v1.json"
export DN_SMOKE="$INTERMEDIATE/downstream_prompt_response_disjoint_eval_smoke4_v1.json"

uv run python scripts/rq.py persona.freeze-upstream-evaluation \
  --config "$CONFIG" \
  --discovery-manifest "$SHARED" \
  --num-prompts 4 \
  --seed 20260811 \
  --max-length 256 \
  --output "$UP_SMOKE"

uv run python scripts/rq.py persona.freeze-downstream-evaluation \
  --config "$CONFIG" \
  --discovery-manifest "$SHARED" \
  --num-pairs 4 \
  --seed 20260811 \
  --max-length 768 \
  --output "$DN_SMOKE"
```

### 8.2 Smoke repaired upstream/downstream evaluation

```bash
submit_a100 syc-up-eval-smoke 01:00:00 \
  persona.evaluate-upstream-contrastive-circuit \
  "--config $CONFIG --run-name $UP_IG5 --k 20000 \
   --circuit-path $RESULTS/circuit_discovery_upstream_contrastive/$UP_IG5/circuits_source_to_sink/eapig_top_20000.json \
   --evaluation-manifest $UP_SMOKE --num-pairs 4 --max-length 256 \
   --output-tag source_to_sink_k20000_smoke4_v1"

submit_a100 syc-dn-eval-smoke 02:00:00 \
  persona.evaluate-steering-circuit \
  "--config $CONFIG --run-name $DN_IG5 --k 3000 \
   --circuit-path $RESULTS/circuit_discovery_steering_path/$DN_IG5/circuits_source_to_sink/steering_path_eapig_top_3000.json \
   --evaluation-manifest $DN_SMOKE --num-prompts 4 --batch-size 1 \
   --max-length 768 --output-tag source_to_sink_k3000_smoke4_v1"
```

Expected:

- both jobs exit 0;
- both `aggregate.json` files have `status=complete`;
- `clean - corrupt > 0`;
- all values are finite;
- `truncation=false`.

### 8.3 Smoke the connected mediation runner

```bash
export CUT_UP_IG5="$RESULTS/circuit_discovery_upstream_contrastive/$UP_IG5/circuits_cut_sets/eapig_top_50.json"
export CUT_DN_IG5="$RESULTS/circuit_discovery_steering_path/$DN_IG5/circuits_cut_sets/steering_path_eapig_top_3000.json"

for CONDITION in normal upstream_ablated upstream_full_prefix_state_rescue upstream_vector_rescue vector_rescue_downstream_ablated; do
  submit_a100 "syc-conn-smoke-${CONDITION:0:8}" 01:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG --upstream-run $UP_IG5 --downstream-run $DN_IG5 \
     --upstream-k 50 --downstream-k 3000 \
     --upstream-circuit-path $CUT_UP_IG5 --downstream-circuit-path $CUT_DN_IG5 \
     --condition $CONDITION --question-indices 0 \
     --upstream-position-scope prompt_only --interface-timing prompt_end \
     --max-new-tokens 16 --max-length 1024 \
     --output-tag ${CONDITION}_q0_t16_smoke_v1"
done
```

After completion:

```bash
export CONN_SMOKE_ROOT="$INTERMEDIATE/connected_circuit_behavior/${UP_IG5}__${DN_IG5}/ku50_kd3000"
NORMAL_HASH=$(jq -r '.generated_token_ids_sha256' "$CONN_SMOKE_ROOT/normal_q0_t16_smoke_v1/responses.jsonl")
PREFIX_HASH=$(jq -r '.generated_token_ids_sha256' "$CONN_SMOKE_ROOT/upstream_full_prefix_state_rescue_q0_t16_smoke_v1/responses.jsonl")
test "$NORMAL_HASH" = "$PREFIX_HASH"
```

The normal and full-prefix token hashes must be identical. A 16-token cap hit is
acceptable in this technical smoke. A hash mismatch is not acceptable.

### 8.4 Smoke the new end-to-end path

```bash
export E2E_SMOKE=sycophancy_end_to_end_smoke4_ig1_v1

submit_a100 syc-e2e-ig1-smoke 03:00:00 \
  persona.run-end-to-end-eapig \
  "--config $CONFIG --run-name $E2E_SMOKE --sample-manifest $SHARED \
   --num-prompts 4 --manifest-sample-indices 0 1 2 3 \
   --batch-size 1 --max-length 768 --ig-steps 1 \
   --k-values 5000 10000 20000"
```

Then:

```bash
export E2E_SMOKE_K="$(jq -r '.circuits[] | select(.family == "source_to_sink" and .retained_edges > 0) | .requested_k' \
  "$RESULTS/circuit_discovery_end_to_end/$E2E_SMOKE/eapig_run_metadata.json" |
  tail -1)"
test -n "$E2E_SMOKE_K" || {
  echo "Smoke discovery produced no complete source-to-logit path."
  exit 1
}

submit_a100 syc-e2e-eval-smoke 03:00:00 \
  persona.evaluate-end-to-end-circuit \
  "--config $CONFIG --run-name $E2E_SMOKE --k $E2E_SMOKE_K \
   --evaluation-manifest $DN_SMOKE --num-prompts 4 --max-length 768 \
   --output-tag k${E2E_SMOKE_K}_smoke4_v1"

submit_a100 syc-e2e-beh-smoke 02:00:00 \
  persona.behavior-end-to-end-circuit \
  "--config $CONFIG --run-name $E2E_SMOKE --k $E2E_SMOKE_K \
   --question-indices 0 \
   --conditions normal negative_prompt circuit_ablated circuit_only \
   --max-new-tokens 16 --max-length 1024 \
   --output-tag k${E2E_SMOKE_K}_q0_t16_smoke_v1"
```

Required smoke result:

- end-to-end discovery metadata is complete;
- discovery `clean_minus_corrupt_margin` is positive;
- evaluation is finite and untruncated;
- all four behavior rows are written with exact token hashes.

Stop and debug if any smoke condition fails. Do not reinterpret a failed
positive control as a scientific result.

## 9. Cheap IG-5 decision cells

Before running convergence pilots, evaluate only enough of the existing frozen
rankings to determine whether complete-path reconstruction is plausible.

### 9.1 Upstream

```bash
for K in 3000 10000 20000; do
  CIRCUIT="$RESULTS/circuit_discovery_upstream_contrastive/$UP_IG5/circuits_source_to_sink/eapig_top_${K}.json"
  if test "$(jq -r '.retained_edges' "$CIRCUIT")" -eq 0; then
    echo "SKIP empty source-to-sink frontier $CIRCUIT"
    continue
  fi
  submit_a100 "syc-up-k${K}-n32" 02:00:00 \
    persona.evaluate-upstream-contrastive-circuit \
    "--config $CONFIG --run-name $UP_IG5 --k $K \
     --circuit-path $CIRCUIT \
     --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
     --output-tag source_to_sink_k${K}_n32_v1"
done
```

### 9.2 Downstream

```bash
for K in 3000 10000; do
  CIRCUIT="$RESULTS/circuit_discovery_steering_path/$DN_IG5/circuits_source_to_sink/steering_path_eapig_top_${K}.json"
  if test "$(jq -r '.retained_edges' "$CIRCUIT")" -eq 0; then
    echo "SKIP empty source-to-sink frontier $CIRCUIT"
    continue
  fi
  submit_a100 "syc-dn-k${K}-n32" 04:00:00 \
    persona.evaluate-steering-circuit \
    "--config $CONFIG --run-name $DN_IG5 --k $K \
     --circuit-path $CIRCUIT \
     --evaluation-manifest $DN_EVAL --num-prompts 32 --batch-size 1 \
     --max-length 768 --output-tag source_to_sink_k${K}_n32_v1"
done
```

Decision rules:

- If upstream faithfulness remains below 0.30 after at least 2% of eligible
  edges are retained and has no convincing upward trend, do not run upstream
  circuit-only generation.
- If downstream remains near its historical high faithfulness, reuse its score
  table subject to IG convergence.

The 20,000 frontier is only 4.25% of eligible edges before complete-path
pruning, so this is only a plausibility check. Run the adaptive 50k/100k
complete-path extension in Section 13 after the canonical IG step count is
frozen.

## 10. Five-question causal-interface preflight

This preflight uses the existing IG-5 causal cuts. It tests the hypothesis and
implementation before the expensive convergence and final-generation waves.
If IG-5 later fails convergence, repeat this preflight with the canonical
reranked cuts before running all 20 questions.

```bash
export PRE_UP_RUN="$UP_IG5"
export PRE_DN_RUN="$DN_IG5"
export PRE_CUT_UP="$CUT_UP_IG5"
export PRE_CUT_DN="$CUT_DN_IG5"
export PRE_LABEL=ig5
export PREFLIGHT_ROOT="$INTERMEDIATE/connected_circuit_behavior/${PRE_UP_RUN}__${PRE_DN_RUN}/ku50_kd3000"

for CONDITION in normal upstream_ablated downstream_ablated both_ablated upstream_full_prefix_state_rescue upstream_full_state_rescue upstream_vector_rescue vector_rescue_downstream_ablated; do
  submit_a100 "syc-pre-${CONDITION:0:10}" 06:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG --upstream-run $PRE_UP_RUN --downstream-run $PRE_DN_RUN \
     --upstream-k 50 --downstream-k 3000 \
     --upstream-circuit-path $PRE_CUT_UP --downstream-circuit-path $PRE_CUT_DN \
     --condition $CONDITION --question-indices $Q5 \
     --upstream-position-scope prompt_only --interface-timing prompt_end \
     --max-new-tokens 350 --max-length 1024 \
     --output-tag ${CONDITION}_prompt-only_prompt-end_q0-1-2-3-4_v1"
done

for SEED in 0 1 2 3 4; do
  submit_a100 "syc-pre-orth${SEED}" 06:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG --upstream-run $PRE_UP_RUN --downstream-run $PRE_DN_RUN \
     --upstream-k 50 --downstream-k 3000 \
     --upstream-circuit-path $PRE_CUT_UP --downstream-circuit-path $PRE_CUT_DN \
     --condition upstream_orthogonal_rescue --orthogonal-seed $SEED \
     --question-indices $Q5 \
     --upstream-position-scope prompt_only --interface-timing prompt_end \
     --max-new-tokens 350 --max-length 1024 \
     --output-tag orthogonal_seed${SEED}_prompt-only_prompt-end_q0-1-2-3-4_v1"
done

submit_a100 syc-pre-unrelated 06:00:00 \
  persona.behavior-connected-circuits \
  "--config $CONFIG --upstream-run $PRE_UP_RUN --downstream-run $PRE_DN_RUN \
   --upstream-k 50 --downstream-k 3000 \
   --upstream-circuit-path $PRE_CUT_UP --downstream-circuit-path $PRE_CUT_DN \
   --condition upstream_unrelated_vector_rescue \
   --unrelated-vector-path $UNRELATED_VECTOR \
   --question-indices $Q5 \
   --upstream-position-scope prompt_only --interface-timing prompt_end \
   --max-new-tokens 350 --max-length 1024 \
   --output-tag unrelated_prompt-only_prompt-end_q0-1-2-3-4_v1"
```

Inspect every raw answer before paid judging:

```bash
find "$PREFLIGHT_ROOT" -path '*q0-1-2-3-4_v1/responses.jsonl' -print0 |
  while IFS= read -r -d '' path; do
    echo "===== $path"
    jq -r '[.question_index,.condition,.generated_token_count,.hit_max_new_tokens,.answer] | @tsv' "$path"
  done
```

Judge the complete artifacts:

```bash
for TAG in \
  normal_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  upstream_ablated_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  downstream_ablated_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  both_ablated_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  upstream_full_prefix_state_rescue_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  upstream_full_state_rescue_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  upstream_vector_rescue_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  vector_rescue_downstream_ablated_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  orthogonal_seed0_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  orthogonal_seed1_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  orthogonal_seed2_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  orthogonal_seed3_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  orthogonal_seed4_prompt-only_prompt-end_q0-1-2-3-4_v1 \
  unrelated_prompt-only_prompt-end_q0-1-2-3-4_v1
do
  uv run python scripts/rq.py persona.judge \
    --config "$CONFIG" \
    --input "$PREFLIGHT_ROOT/$TAG/responses.jsonl" \
    --output "$PREFLIGHT_ROOT/$TAG/scored.jsonl" \
    --artifact eval \
    --protocol-field behavior_protocol_id \
    --resume
done
```

Analyze:

```bash
uv run python scripts/rq.py persona.analyze-bottleneck-mediation \
  --config "$CONFIG" \
  --normal-scored "$PREFLIGHT_ROOT/normal_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --upstream-ablated-scored "$PREFLIGHT_ROOT/upstream_ablated_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --vector-rescue-scored "$PREFLIGHT_ROOT/upstream_vector_rescue_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --full-state-rescue-scored "$PREFLIGHT_ROOT/upstream_full_state_rescue_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --full-prefix-state-rescue-scored "$PREFLIGHT_ROOT/upstream_full_prefix_state_rescue_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --orthogonal-rescue-scored \
    "$PREFLIGHT_ROOT/orthogonal_seed0_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
    "$PREFLIGHT_ROOT/orthogonal_seed1_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
    "$PREFLIGHT_ROOT/orthogonal_seed2_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
    "$PREFLIGHT_ROOT/orthogonal_seed3_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
    "$PREFLIGHT_ROOT/orthogonal_seed4_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --unrelated-rescue-scored "$PREFLIGHT_ROOT/unrelated_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --downstream-blocked-scored "$PREFLIGHT_ROOT/vector_rescue_downstream_ablated_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --downstream-ablated-scored "$PREFLIGHT_ROOT/downstream_ablated_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --both-ablated-scored "$PREFLIGHT_ROOT/both_ablated_prompt-only_prompt-end_q0-1-2-3-4_v1/scored.jsonl" \
  --minimum-token-cap 350 \
  --bootstrap-samples 10000 \
  --output "$RESULTS/bottleneck_mediation_${PRE_LABEL}_q0-4_preflight_v1"
```

Preflight stop rules:

1. Full-prefix rescue must exactly match normal token-for-token.
2. Upstream ablation must lower behavior.
3. Vector rescue must move behavior back toward normal.
4. Orthogonal and unrelated-vector rescue should be materially weaker.
5. Downstream blocking should reduce the vector-rescue effect.
6. Coherence must remain high and cap hits must not drive the result.

The five-question confidence intervals are diagnostics, not paper evidence.

## 11. Fixed-row IG convergence

Run the same 16 discovery rows at 5, 20, and 50 steps for all three estimands.
Do not change samples, metric, transcript cap, or coefficient between runs.

### 11.1 Upstream pilots

```bash
for STEPS in 5 20 50; do
  submit_a100 "syc-up-ig${STEPS}-p16" 12:00:00 \
    persona.run-upstream-contrastive-eapig \
    "--config $CONFIG --run-name syc_up_ig${STEPS}_pilot16_v1 \
     --sample-manifest $SHARED --num-pairs 16 \
     --manifest-sample-indices $PILOT16 \
     --batch-size 1 --max-length 256 --ig-steps $STEPS --k 10000"
done
```

### 11.2 Downstream pilots

```bash
for STEPS in 5 20 50; do
  submit_a100 "syc-dn-ig${STEPS}-p16" 24:00:00 \
    persona.run-steering-eapig \
    "--config $CONFIG --run-name syc_dn_ig${STEPS}_pilot16_v1 \
     --sample-manifest $SHARED --num-prompts 16 \
     --manifest-sample-indices $PILOT16 \
     --batch-size 1 --max-length 768 --ig-steps $STEPS \
     --corrupt-coefficient -2 \
     --minimum-trait-score 80 --minimum-coherence-score 80 \
     --maximum-negative-trait-score 20 \
     --metric contrastive_response_logprob \
     --k-values 500 3000 10000"
done
```

### 11.3 End-to-end pilots

```bash
for STEPS in 5 20 50; do
  submit_a100 "syc-e2e-ig${STEPS}-p16" 24:00:00 \
    persona.run-end-to-end-eapig \
    "--config $CONFIG --run-name syc_e2e_ig${STEPS}_pilot16_v1 \
     --sample-manifest $SHARED --num-prompts 16 \
     --manifest-sample-indices $PILOT16 \
     --batch-size 1 --max-length 768 --ig-steps $STEPS \
     --k-values 500 3000 10000"
done
```

### 11.4 Held-out pilot faithfulness at k=10,000

```bash
for STEPS in 5 20 50; do
  submit_a100 "syc-up-ig${STEPS}-eval" 02:00:00 \
    persona.evaluate-upstream-contrastive-circuit \
    "--config $CONFIG --run-name syc_up_ig${STEPS}_pilot16_v1 --k 10000 \
     --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
     --output-tag k10000_n32_convergence_v1"

  submit_a100 "syc-dn-ig${STEPS}-eval" 05:00:00 \
    persona.evaluate-steering-circuit \
    "--config $CONFIG --run-name syc_dn_ig${STEPS}_pilot16_v1 --k 10000 \
     --evaluation-manifest $DN_EVAL --num-prompts 32 --batch-size 1 \
     --max-length 768 --output-tag k10000_n32_convergence_v1"

  submit_a100 "syc-e2e-ig${STEPS}-eval" 06:00:00 \
    persona.evaluate-end-to-end-circuit \
    "--config $CONFIG --run-name syc_e2e_ig${STEPS}_pilot16_v1 --k 10000 \
     --evaluation-manifest $DN_EVAL --num-prompts 32 --max-length 768 \
     --output-tag k10000_n32_convergence_v1"
done
```

### 11.5 Compare score tables

```bash
for SCOPE in up dn e2e; do
  case "$SCOPE" in
    up)
      DIR=circuit_discovery_upstream_contrastive
      GRAPH_SCOPE=upstream
      ;;
    dn)
      DIR=circuit_discovery_steering_path
      GRAPH_SCOPE=downstream
      ;;
    e2e)
      DIR=circuit_discovery_end_to_end
      GRAPH_SCOPE=end_to_end
      ;;
  esac

  uv run python scripts/rq.py persona.compare-score-tables \
    --config "$CONFIG" --scope "$GRAPH_SCOPE" \
    --left "$RESULTS/$DIR/syc_${SCOPE}_ig5_pilot16_v1/eapig_edge_scores.parquet" \
    --right "$RESULTS/$DIR/syc_${SCOPE}_ig20_pilot16_v1/eapig_edge_scores.parquet" \
    --k-values 50 500 3000 10000 \
    --output "$RESULTS/${SCOPE}_ig5_vs_ig20_pilot16_v1.json"

  uv run python scripts/rq.py persona.compare-score-tables \
    --config "$CONFIG" --scope "$GRAPH_SCOPE" \
    --left "$RESULTS/$DIR/syc_${SCOPE}_ig20_pilot16_v1/eapig_edge_scores.parquet" \
    --right "$RESULTS/$DIR/syc_${SCOPE}_ig50_pilot16_v1/eapig_edge_scores.parquet" \
    --k-values 50 500 3000 10000 \
    --output "$RESULTS/${SCOPE}_ig20_vs_ig50_pilot16_v1.json"
done
```

The run-name convention above is exact:

- `syc_up_igSTEPS_pilot16_v1`
- `syc_dn_igSTEPS_pilot16_v1`
- `syc_e2e_igSTEPS_pilot16_v1`

Each comparison also reports the retained complete-path edge counts and
source-to-sink Jaccard at every requested frontier. Report these alongside the
raw absolute-score top-k overlap; a stable raw ranking that produces unstable
complete paths is not enough for a reconstructive claim.

Select steps independently for upstream, downstream, and end-to-end:

- use 20 when 20 versus 50 has absolute-score Spearman >=0.95,
  top-3,000 Jaccard >=0.80, and held-out faithfulness differs by <=0.05;
- retain 5 only when 5 versus 20 also passes those thresholds;
- otherwise use 50;
- do not run 200 steps unless 50 is visibly unconverged.

Record the decisions before full reruns:

```bash
export UP_STEPS=5
export DN_STEPS=5
export E2E_STEPS=20
```

The values above are examples, not assumed outcomes.

## 12. Freeze the canonical decomposed rankings

### 12.1 Upstream

If `UP_STEPS=5`:

```bash
export UP_FINAL="$UP_IG5"
```

Otherwise:

```bash
export UP_FINAL="sycophancy_upstream_n64_ig${UP_STEPS}_v1"
submit_a100 syc-up-final-ig${UP_STEPS} 24:00:00 \
  persona.run-upstream-contrastive-eapig \
  "--config $CONFIG --run-name $UP_FINAL --sample-manifest $SHARED \
   --num-pairs 64 --batch-size 1 --max-length 256 \
   --ig-steps $UP_STEPS --k 20000"
```

Then rebuild all upstream frontiers:

```bash
uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" --scope upstream --run-name "$UP_FINAL" \
  --families source_to_sink causal_cut \
  --k-values 50 100 250 500 1000 2000 3000 5000 10000 20000
```

### 12.2 Downstream

If `DN_STEPS=5`:

```bash
export DN_FINAL="$DN_IG5"
```

Otherwise:

```bash
export DN_FINAL="sycophancy_downstream_n64_ig${DN_STEPS}_v1"
submit_a100 syc-dn-final-ig${DN_STEPS} 36:00:00 \
  persona.run-steering-eapig \
  "--config $CONFIG --run-name $DN_FINAL --sample-manifest $SHARED \
   --num-prompts 64 --batch-size 1 --max-length 768 \
   --ig-steps $DN_STEPS --corrupt-coefficient -2 \
   --minimum-trait-score 80 --minimum-coherence-score 80 \
   --maximum-negative-trait-score 20 \
   --metric contrastive_response_logprob \
   --k-values 50 100 250 500 1000 2000 3000 5000 10000"
```

Rebuild for a uniform manifest:

```bash
uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" --scope downstream --run-name "$DN_FINAL" \
  --families source_to_sink causal_cut \
  --k-values 50 100 250 500 1000 2000 3000 5000 10000
```

Define final cut paths:

```bash
export CUT_UP="$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/circuits_cut_sets/eapig_top_50.json"
export CUT_DN="$RESULTS/circuit_discovery_steering_path/$DN_FINAL/circuits_cut_sets/steering_path_eapig_top_3000.json"
```

If either final run differs from IG-5, set:

```bash
export PRE_UP_RUN="$UP_FINAL"
export PRE_DN_RUN="$DN_FINAL"
export PRE_CUT_UP="$CUT_UP"
export PRE_CUT_DN="$CUT_DN"
export PRE_LABEL="canonical_${UP_FINAL}__${DN_FINAL}"
export PREFLIGHT_ROOT="$INTERMEDIATE/connected_circuit_behavior/${PRE_UP_RUN}__${PRE_DN_RUN}/ku50_kd3000"
```

Then rerun every command block in Section 10 exactly as written. Do not expand
the old IG-5 preflight to 20 questions.

## 13. Final source-to-sink internal curves

### 13.1 Upstream

```bash
export UP_CURVE_K="$(jq -r '.circuits[] | select(.family == "source_to_sink" and .retained_edges > 0) | .k' \
  "$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/derived_upstream_circuit_families_manifest.json" |
  paste -sd' ' -)"

for K in $UP_CURVE_K; do
  OUT="$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/evaluation/source_to_sink_k${K}_n32_v1/aggregate.json"
  if test -s "$OUT"; then
    echo "SKIP complete $OUT"
  else
    submit_a100 "syc-up-final-k${K}" 03:00:00 \
      persona.evaluate-upstream-contrastive-circuit \
      "--config $CONFIG --run-name $UP_FINAL --k $K \
       --circuit-path $RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/circuits_source_to_sink/eapig_top_${K}.json \
       --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
       --output-tag source_to_sink_k${K}_n32_v1"
  fi
done
```

The largest initial frontier may retain less than the precommitted 5% sparsity
boundary. After its evaluations finish, extend the **canonical** score table
without reranking only while faithfulness is below 0.85:

```bash
export UP_ELIGIBLE="$(jq -r '.eligible_edges' \
  "$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/eapig_run_metadata.json")"
export UP20_RETAINED="$(jq -r '.retained_edges' \
  "$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/circuits_source_to_sink/eapig_top_20000.json")"
export UP20_FAITH="$(jq -r '.faithfulness' \
  "$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/evaluation/source_to_sink_k20000_n32_v1/aggregate.json")"

if (( UP20_RETAINED * 100 < UP_ELIGIBLE * 5 )) &&
   awk "BEGIN { exit !($UP20_FAITH < 0.85) }"; then
  uv run python scripts/rq.py persona.rebuild-nested-circuits \
    --config "$CONFIG" --scope upstream --run-name "$UP_FINAL" \
    --families source_to_sink causal_cut \
    --k-values 50 100 250 500 1000 2000 3000 5000 10000 20000 50000

  submit_a100 syc-up-k50000-n32 03:00:00 \
    persona.evaluate-upstream-contrastive-circuit \
    "--config $CONFIG --run-name $UP_FINAL --k 50000 \
     --circuit-path $RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/circuits_source_to_sink/eapig_top_50000.json \
     --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
     --output-tag source_to_sink_k50000_n32_v1"
fi
```

After k=50,000 finishes, extend once more only if it still retains less than 5%
and no tested canonical circuit has reached 0.85:

```bash
UP50_CIRCUIT="$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/circuits_source_to_sink/eapig_top_50000.json"
UP50_EVAL="$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/evaluation/source_to_sink_k50000_n32_v1/aggregate.json"

if test -s "$UP50_CIRCUIT" && test -s "$UP50_EVAL"; then
  UP50_RETAINED="$(jq -r '.retained_edges' "$UP50_CIRCUIT")"
  UP_BEST_FAITH="$(find "$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/evaluation" \
    -path '*/source_to_sink_*_n32_v1/aggregate.json' -print0 |
    xargs -0 jq -s 'map(.faithfulness) | max')"
  if (( UP50_RETAINED * 100 < UP_ELIGIBLE * 5 )) &&
     awk "BEGIN { exit !($UP_BEST_FAITH < 0.85) }"; then
    uv run python scripts/rq.py persona.rebuild-nested-circuits \
      --config "$CONFIG" --scope upstream --run-name "$UP_FINAL" \
      --families source_to_sink causal_cut \
      --k-values 50 100 250 500 1000 2000 3000 5000 10000 20000 50000 100000

    submit_a100 syc-up-k100000-n32 04:00:00 \
      persona.evaluate-upstream-contrastive-circuit \
      "--config $CONFIG --run-name $UP_FINAL --k 100000 \
       --circuit-path $RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/circuits_source_to_sink/eapig_top_100000.json \
       --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
       --output-tag source_to_sink_k100000_n32_v1"
  fi
fi
```

After any extension job finishes, refresh the curve list:

```bash
export UP_CURVE_K="$(jq -r '.circuits[] | select(.family == "source_to_sink" and .retained_edges > 0) | .k' \
  "$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/derived_upstream_circuit_families_manifest.json" |
  paste -sd' ' -)"
```

If no circuit reaches 0.85 by the first tested frontier retaining at least 5%
of eligible edges, or by k=100,000, retain only the causal-interface claim.
This extension does not change scores or inspect official behavioral outcomes.

### 13.2 Downstream

```bash
export DN_CURVE_K="$(jq -r '.circuits[] | select(.family == "source_to_sink" and .retained_edges > 0) | .k' \
  "$RESULTS/circuit_discovery_steering_path/$DN_FINAL/derived_downstream_circuit_families_manifest.json" |
  paste -sd' ' -)"

for K in $DN_CURVE_K; do
  OUT="$RESULTS/circuit_discovery_steering_path/$DN_FINAL/evaluation/source_to_sink_k${K}_n32_v1/aggregate.json"
  if test -s "$OUT"; then
    echo "SKIP complete $OUT"
  else
    submit_a100 "syc-dn-final-k${K}" 06:00:00 \
      persona.evaluate-steering-circuit \
      "--config $CONFIG --run-name $DN_FINAL --k $K \
       --circuit-path $RESULTS/circuit_discovery_steering_path/$DN_FINAL/circuits_source_to_sink/steering_path_eapig_top_${K}.json \
       --evaluation-manifest $DN_EVAL --num-prompts 32 --batch-size 1 \
       --max-length 768 --output-tag source_to_sink_k${K}_n32_v1"
  fi
done
```

Select:

- `UP_RECON_K`: smallest tested upstream circuit with held-out faithfulness
  >=0.85, stable positive denominator, high necessity, and no pathological
  non-monotonicity.
- `DN_RECON_K`: same rule downstream.

Record:

```bash
unset UP_RECON_K DN_RECON_K
# Export a value only when that family passes the precommitted gate:
export UP_RECON_K=VALUE
export DN_RECON_K=VALUE
```

If no upstream circuit passes by the adaptive 5%-retained gate or k=100,000,
leave `UP_RECON_K` unset and do not make a reconstructive formation-circuit
claim. If no downstream circuit passes, leave `DN_RECON_K` unset and retain
only its causal-cut mediation role.

## 14. Topology-stratified controls

Run controls only at the precommitted selected reconstruction k, not at every
frontier.

```bash
if test -n "${UP_RECON_K:-}"; then
  uv run python scripts/rq.py persona.build-circuit-controls \
    --config "$CONFIG" --scope upstream --run-name "$UP_FINAL" \
    --family source_to_sink --k-values "$UP_RECON_K" --seeds 0 1 2 3 4
fi
if test -n "${DN_RECON_K:-}"; then
  uv run python scripts/rq.py persona.build-circuit-controls \
    --config "$CONFIG" --scope downstream --run-name "$DN_FINAL" \
    --family source_to_sink --k-values "$DN_RECON_K" --seeds 0 1 2 3 4
fi
```

Evaluate:

```bash
for SEED in 0 1 2 3 4; do
  if test -n "${UP_RECON_K:-}"; then
    submit_a100 "syc-up-ctrl${SEED}" 03:00:00 \
      persona.evaluate-upstream-contrastive-circuit \
      "--config $CONFIG --run-name $UP_FINAL --k $UP_RECON_K \
       --circuit-path $RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/controls/topology_stratified_v2/source_to_sink/seed_${SEED}/eapig_top_${UP_RECON_K}.json \
       --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
       --output-tag control_topology_v2_seed${SEED}_k${UP_RECON_K}_n32"
  fi

  if test -n "${DN_RECON_K:-}"; then
    submit_a100 "syc-dn-ctrl${SEED}" 06:00:00 \
      persona.evaluate-steering-circuit \
      "--config $CONFIG --run-name $DN_FINAL --k $DN_RECON_K \
       --circuit-path $RESULTS/circuit_discovery_steering_path/$DN_FINAL/controls/topology_stratified_v2/source_to_sink/seed_${SEED}/steering_path_eapig_top_${DN_RECON_K}.json \
       --evaluation-manifest $DN_EVAL --num-prompts 32 --batch-size 1 \
       --max-length 768 \
       --output-tag control_topology_v2_seed${SEED}_k${DN_RECON_K}_n32"
  fi
done
```

The real circuit must materially outperform the five retained-size-matched
controls. Report every control seed, not only the mean.

Aggregate the repaired internal curves, paired bootstrap intervals, and control
cells:

```bash
INTERNAL_ARGS=(
  --config "$CONFIG"
  --upstream-run "$UP_FINAL"
  --downstream-run "$DN_FINAL"
  --upstream-k-values $UP_CURVE_K
  --downstream-k-values $DN_CURVE_K
  --control-seeds 0 1 2 3 4
  --bootstrap-samples 10000
  --output-name "repaired_internal_${UP_FINAL}__${DN_FINAL}_v1"
)
if test -n "${UP_RECON_K:-}"; then
  INTERNAL_ARGS+=(--upstream-selected-k "$UP_RECON_K")
fi
if test -n "${DN_RECON_K:-}"; then
  INTERNAL_ARGS+=(--downstream-selected-k "$DN_RECON_K")
fi
uv run python scripts/rq.py persona.analyze-internal-circuit-curves \
  "${INTERNAL_ARGS[@]}"
```

## 15. Full end-to-end comparator

### 15.1 Discovery

```bash
export E2E_FINAL="sycophancy_end_to_end_n64_ig${E2E_STEPS}_v1"

submit_a100 syc-e2e-final-ig${E2E_STEPS} 48:00:00 \
  persona.run-end-to-end-eapig \
  "--config $CONFIG --run-name $E2E_FINAL --sample-manifest $SHARED \
   --num-prompts 64 --batch-size 1 --max-length 768 \
   --ig-steps $E2E_STEPS \
   --k-values 500 1000 3000 5000 10000 20000"
```

### 15.2 Held-out curve

```bash
export E2E_CURVE_K="$(jq -r '.circuits[] | select(.family == "source_to_sink" and .retained_edges > 0) | .requested_k' \
  "$RESULTS/circuit_discovery_end_to_end/$E2E_FINAL/eapig_run_metadata.json" |
  paste -sd' ' -)"
test -n "$E2E_CURVE_K" || {
  echo "No end-to-end frontier contains a complete source-to-logit path."
  exit 1
}

for K in $E2E_CURVE_K; do
  submit_a100 "syc-e2e-k${K}" 08:00:00 \
    persona.evaluate-end-to-end-circuit \
    "--config $CONFIG --run-name $E2E_FINAL --k $K \
     --evaluation-manifest $DN_EVAL --num-prompts 32 --max-length 768 \
     --output-tag k${K}_n32_prompt_disjoint"
done
```

Select the smallest `E2E_K` with held-out faithfulness >=0.85, stable positive
clean-corrupt margin, high necessity, and a broadly monotonic curve:

```bash
export E2E_K=VALUE
```

If no k reaches 0.85, the end-to-end object may still be reported as a partial
causal circuit, but not as a faithful complete mechanism. Leave `E2E_K` unset
and skip Sections 15.3-15.5 rather than choosing k from official behavioral
scores.

### 15.3 End-to-end controls

```bash
uv run python scripts/rq.py persona.build-circuit-controls \
  --config "$CONFIG" --scope end_to_end --run-name "$E2E_FINAL" \
  --family source_to_sink --k-values "$E2E_K" --seeds 0 1 2 3 4

for SEED in 0 1 2 3 4; do
  submit_a100 "syc-e2e-ctrl${SEED}" 08:00:00 \
    persona.evaluate-end-to-end-circuit \
    "--config $CONFIG --run-name $E2E_FINAL --k $E2E_K \
     --circuit-path $RESULTS/circuit_discovery_end_to_end/$E2E_FINAL/controls/topology_stratified_v2/source_to_sink/seed_${SEED}/end_to_end_eapig_top_${E2E_K}.json \
     --evaluation-manifest $DN_EVAL --num-prompts 32 --max-length 768 \
     --output-tag control_topology_v2_seed${SEED}_k${E2E_K}_n32"
done
```

### 15.4 Five-question behavioral preflight

```bash
submit_a100 syc-e2e-beh-preflight 12:00:00 \
  persona.behavior-end-to-end-circuit \
  "--config $CONFIG --run-name $E2E_FINAL --k $E2E_K \
   --question-indices $Q5 \
   --conditions normal negative_prompt circuit_ablated circuit_only \
   --max-new-tokens 350 --max-length 1024 \
   --output-tag k${E2E_K}_q0-4_preflight_v1"
```

Inspect all 20 rows, then judge:

```bash
export E2E_BEHAVIOR_ROOT="$INTERMEDIATE/end_to_end_circuit_behavior/$E2E_FINAL"

uv run python scripts/rq.py persona.judge \
  --config "$CONFIG" \
  --input "$E2E_BEHAVIOR_ROOT/k${E2E_K}_q0-4_preflight_v1/responses.jsonl" \
  --output "$E2E_BEHAVIOR_ROOT/k${E2E_K}_q0-4_preflight_v1/scored.jsonl" \
  --artifact eval --protocol-field behavior_protocol_id --resume
```

Expand only when responses are coherent, the normal/negative contrast has the
intended direction, ablation reduces the trait, and circuit-only moves toward
normal.

### 15.5 Primary q5-q19 behavior

```bash
for BATCH in "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  SLUG=$(echo "$BATCH" | tr ' ' '-')
  submit_a100 "syc-e2e-q${SLUG}" 18:00:00 \
    persona.behavior-end-to-end-circuit \
    "--config $CONFIG --run-name $E2E_FINAL --k $E2E_K \
     --question-indices $BATCH \
     --conditions normal negative_prompt circuit_ablated circuit_only \
     --max-new-tokens 350 --max-length 1024 \
     --output-tag k${E2E_K}_q${SLUG}_v1"
done
```

Pool each condition across the three confirmatory batches. Do not pool the
judged q0-q4 pilot into the primary confidence interval:

```bash
for CONDITION in normal negative_prompt circuit_ablated circuit_only; do
  uv run python scripts/rq.py persona.pool-generation-splits \
    --sources \
      "$E2E_BEHAVIOR_ROOT/k${E2E_K}_q5-6-7-8-9_v1/responses.jsonl" \
      "$E2E_BEHAVIOR_ROOT/k${E2E_K}_q10-11-12-13-14_v1/responses.jsonl" \
      "$E2E_BEHAVIOR_ROOT/k${E2E_K}_q15-16-17-18-19_v1/responses.jsonl" \
    --condition "$CONDITION" \
    --question-indices $Q15 \
    --common-max-new-tokens 350 \
    --output-dir "$E2E_BEHAVIOR_ROOT/k${E2E_K}_pooled_${CONDITION}_q5-19_primary_v1"

  uv run python scripts/rq.py persona.judge \
    --config "$CONFIG" \
    --input "$E2E_BEHAVIOR_ROOT/k${E2E_K}_pooled_${CONDITION}_q5-19_primary_v1/responses.jsonl" \
    --output "$E2E_BEHAVIOR_ROOT/k${E2E_K}_pooled_${CONDITION}_q5-19_primary_v1/scored.jsonl" \
    --artifact eval --protocol-field behavior_protocol_id --resume
done
```

Analyze:

```bash
uv run python scripts/rq.py persona.analyze-end-to-end-curves \
  --config "$CONFIG" --run-name "$E2E_FINAL" \
  --k-values $E2E_CURVE_K \
  --behavior-k-values "$E2E_K" \
  --control-k-values "$E2E_K" \
  --control-seeds 0 1 2 3 4 \
  --evaluation-template 'k{k}_n32_prompt_disjoint' \
  --question-indices $Q15 \
  --minimum-token-cap 350 \
  --bootstrap-samples 10000 \
  --output-name "end_to_end_curves_${E2E_FINAL}_q5-19_primary_v1"
```

## 16. Primary q5-q19 mediation

Run this only after:

- the final upstream/downstream rankings are frozen;
- the canonical five-question preflight passes;
- any IG-5 failure has triggered a new preflight on reranked cuts.

```bash
export CONN_ROOT="$INTERMEDIATE/connected_circuit_behavior/${UP_FINAL}__${DN_FINAL}/ku50_kd3000"
export BATCH0="0 1 2 3 4"
export BATCH1="5 6 7 8 9"
export BATCH2="10 11 12 13 14"
export BATCH3="15 16 17 18 19"
```

Submit four batches for the eight main conditions:

```bash
for CONDITION in normal upstream_ablated downstream_ablated both_ablated upstream_full_prefix_state_rescue upstream_full_state_rescue upstream_vector_rescue vector_rescue_downstream_ablated; do
  for BATCH in "$BATCH0" "$BATCH1" "$BATCH2" "$BATCH3"; do
    SLUG=$(echo "$BATCH" | tr ' ' '-')
    TAG="${CONDITION}_prompt-only_prompt-end_q${SLUG}_v1"
    META="$CONN_ROOT/$TAG/metadata.json"
    if test -s "$META" && jq -e '.status == "complete"' "$META" >/dev/null; then
      echo "SKIP complete $TAG"
      continue
    fi
    RESUME=""
    test -s "$META" && RESUME="--resume"
    submit_a100 "syc-${CONDITION:0:7}-q${SLUG}" 12:00:00 \
      persona.behavior-connected-circuits \
      "--config $CONFIG --upstream-run $UP_FINAL --downstream-run $DN_FINAL \
       --upstream-k 50 --downstream-k 3000 \
       --upstream-circuit-path $CUT_UP --downstream-circuit-path $CUT_DN \
       --condition $CONDITION --question-indices $BATCH \
       --upstream-position-scope prompt_only --interface-timing prompt_end \
       --max-new-tokens 350 --max-length 1024 \
       --output-tag $TAG $RESUME"
  done
done
```

Submit the five orthogonal seeds:

```bash
for SEED in 0 1 2 3 4; do
  for BATCH in "$BATCH0" "$BATCH1" "$BATCH2" "$BATCH3"; do
    SLUG=$(echo "$BATCH" | tr ' ' '-')
    TAG="orthogonal_seed${SEED}_prompt-only_prompt-end_q${SLUG}_v1"
    META="$CONN_ROOT/$TAG/metadata.json"
    if test -s "$META" && jq -e '.status == "complete"' "$META" >/dev/null; then
      echo "SKIP complete $TAG"
      continue
    fi
    RESUME=""
    test -s "$META" && RESUME="--resume"
    submit_a100 "syc-orth${SEED}-q${SLUG}" 12:00:00 \
      persona.behavior-connected-circuits \
      "--config $CONFIG --upstream-run $UP_FINAL --downstream-run $DN_FINAL \
       --upstream-k 50 --downstream-k 3000 \
       --upstream-circuit-path $CUT_UP --downstream-circuit-path $CUT_DN \
       --condition upstream_orthogonal_rescue --orthogonal-seed $SEED \
       --question-indices $BATCH \
       --upstream-position-scope prompt_only --interface-timing prompt_end \
       --max-new-tokens 350 --max-length 1024 \
       --output-tag $TAG $RESUME"
  done
done
```

Submit the unrelated-vector control:

```bash
for BATCH in "$BATCH0" "$BATCH1" "$BATCH2" "$BATCH3"; do
  SLUG=$(echo "$BATCH" | tr ' ' '-')
  TAG="unrelated_prompt-only_prompt-end_q${SLUG}_v1"
  META="$CONN_ROOT/$TAG/metadata.json"
  if test -s "$META" && jq -e '.status == "complete"' "$META" >/dev/null; then
    echo "SKIP complete $TAG"
    continue
  fi
  RESUME=""
  test -s "$META" && RESUME="--resume"
  submit_a100 "syc-unrel-q${SLUG}" 12:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG --upstream-run $UP_FINAL --downstream-run $DN_FINAL \
     --upstream-k 50 --downstream-k 3000 \
     --upstream-circuit-path $CUT_UP --downstream-circuit-path $CUT_DN \
     --condition upstream_unrelated_vector_rescue \
     --unrelated-vector-path $UNRELATED_VECTOR \
     --question-indices $BATCH \
     --upstream-position-scope prompt_only --interface-timing prompt_end \
     --max-new-tokens 350 --max-length 1024 \
     --output-tag $TAG $RESUME"
done
```

The loops skip completed preflight batches and automatically resume partial
ones.

Pool and judge the eight main conditions across q5-q19. The q0-q4 pilot remains
separate:

```bash
for CONDITION in normal upstream_ablated downstream_ablated both_ablated upstream_full_prefix_state_rescue upstream_full_state_rescue upstream_vector_rescue vector_rescue_downstream_ablated; do
  uv run python scripts/rq.py persona.pool-generation-splits \
    --sources \
      "$CONN_ROOT/${CONDITION}_prompt-only_prompt-end_q5-6-7-8-9_v1/responses.jsonl" \
      "$CONN_ROOT/${CONDITION}_prompt-only_prompt-end_q10-11-12-13-14_v1/responses.jsonl" \
      "$CONN_ROOT/${CONDITION}_prompt-only_prompt-end_q15-16-17-18-19_v1/responses.jsonl" \
    --condition "$CONDITION" \
    --question-indices $Q15 \
    --common-max-new-tokens 350 \
    --output-dir "$CONN_ROOT/pooled_${CONDITION}_prompt-only_prompt-end_q5-19_primary_v1"

  uv run python scripts/rq.py persona.judge \
    --config "$CONFIG" \
    --input "$CONN_ROOT/pooled_${CONDITION}_prompt-only_prompt-end_q5-19_primary_v1/responses.jsonl" \
    --output "$CONN_ROOT/pooled_${CONDITION}_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
    --artifact eval --protocol-field behavior_protocol_id --resume
done
```

Pool and judge the orthogonal controls:

```bash
for SEED in 0 1 2 3 4; do
  uv run python scripts/rq.py persona.pool-generation-splits \
    --sources \
      "$CONN_ROOT/orthogonal_seed${SEED}_prompt-only_prompt-end_q5-6-7-8-9_v1/responses.jsonl" \
      "$CONN_ROOT/orthogonal_seed${SEED}_prompt-only_prompt-end_q10-11-12-13-14_v1/responses.jsonl" \
      "$CONN_ROOT/orthogonal_seed${SEED}_prompt-only_prompt-end_q15-16-17-18-19_v1/responses.jsonl" \
    --condition upstream_orthogonal_rescue \
    --question-indices $Q15 \
    --common-max-new-tokens 350 \
    --output-dir "$CONN_ROOT/pooled_orthogonal_seed${SEED}_prompt-only_prompt-end_q5-19_primary_v1"

  uv run python scripts/rq.py persona.judge \
    --config "$CONFIG" \
    --input "$CONN_ROOT/pooled_orthogonal_seed${SEED}_prompt-only_prompt-end_q5-19_primary_v1/responses.jsonl" \
    --output "$CONN_ROOT/pooled_orthogonal_seed${SEED}_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
    --artifact eval --protocol-field behavior_protocol_id --resume
done
```

Pool and judge the unrelated-vector control:

```bash
uv run python scripts/rq.py persona.pool-generation-splits \
  --sources \
    "$CONN_ROOT/unrelated_prompt-only_prompt-end_q5-6-7-8-9_v1/responses.jsonl" \
    "$CONN_ROOT/unrelated_prompt-only_prompt-end_q10-11-12-13-14_v1/responses.jsonl" \
    "$CONN_ROOT/unrelated_prompt-only_prompt-end_q15-16-17-18-19_v1/responses.jsonl" \
  --condition upstream_unrelated_vector_rescue \
  --question-indices $Q15 \
  --common-max-new-tokens 350 \
  --output-dir "$CONN_ROOT/pooled_unrelated_prompt-only_prompt-end_q5-19_primary_v1"

uv run python scripts/rq.py persona.judge \
  --config "$CONFIG" \
  --input "$CONN_ROOT/pooled_unrelated_prompt-only_prompt-end_q5-19_primary_v1/responses.jsonl" \
  --output "$CONN_ROOT/pooled_unrelated_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --artifact eval --protocol-field behavior_protocol_id --resume
```

Final mediation analysis:

```bash
uv run python scripts/rq.py persona.analyze-bottleneck-mediation \
  --config "$CONFIG" \
  --normal-scored "$CONN_ROOT/pooled_normal_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --upstream-ablated-scored "$CONN_ROOT/pooled_upstream_ablated_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --vector-rescue-scored "$CONN_ROOT/pooled_upstream_vector_rescue_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --full-state-rescue-scored "$CONN_ROOT/pooled_upstream_full_state_rescue_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --full-prefix-state-rescue-scored "$CONN_ROOT/pooled_upstream_full_prefix_state_rescue_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --orthogonal-rescue-scored \
    "$CONN_ROOT/pooled_orthogonal_seed0_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
    "$CONN_ROOT/pooled_orthogonal_seed1_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
    "$CONN_ROOT/pooled_orthogonal_seed2_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
    "$CONN_ROOT/pooled_orthogonal_seed3_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
    "$CONN_ROOT/pooled_orthogonal_seed4_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --unrelated-rescue-scored "$CONN_ROOT/pooled_unrelated_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --downstream-blocked-scored "$CONN_ROOT/pooled_vector_rescue_downstream_ablated_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --downstream-ablated-scored "$CONN_ROOT/pooled_downstream_ablated_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --both-ablated-scored "$CONN_ROOT/pooled_both_ablated_prompt-only_prompt-end_q5-19_primary_v1/scored.jsonl" \
  --minimum-token-cap 350 \
  --bootstrap-samples 10000 \
  --output "$RESULTS/bottleneck_mediation_${UP_FINAL}__${DN_FINAL}_q5-19_primary_v1"
```

Final level-2 evidence on the q5-q19 primary rows requires:

- full-prefix rescue exact-match rate = 1.0;
- upstream-cut effect 95% CI above zero;
- vector-rescue effect 95% CI above zero;
- vector-minus-orthogonal 95% CI above zero;
- vector-minus-unrelated 95% CI above zero;
- downstream-blocking 95% CI above zero;
- no generic coherence collapse;
- transparent cap-hit reporting.

## 17. Connected reconstructive union

Run only when both `UP_RECON_K` and `DN_RECON_K` pass the internal
source-to-sink gate.

```bash
export RECON_UP="$RESULTS/circuit_discovery_upstream_contrastive/$UP_FINAL/circuits_source_to_sink/eapig_top_${UP_RECON_K}.json"
export RECON_DN="$RESULTS/circuit_discovery_steering_path/$DN_FINAL/circuits_source_to_sink/steering_path_eapig_top_${DN_RECON_K}.json"
export RECON_ROOT="$INTERMEDIATE/connected_circuit_behavior/${UP_FINAL}__${DN_FINAL}/ku${UP_RECON_K}_kd${DN_RECON_K}"
```

Five-question preflight:

```bash
for CONDITION in normal both_circuit_only; do
  submit_a100 "syc-recon-${CONDITION:0:8}" 12:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG --upstream-run $UP_FINAL --downstream-run $DN_FINAL \
     --upstream-k $UP_RECON_K --downstream-k $DN_RECON_K \
     --upstream-circuit-path $RECON_UP --downstream-circuit-path $RECON_DN \
     --condition $CONDITION --question-indices $Q5 \
     --upstream-position-scope all_positions --interface-timing current_token \
     --max-new-tokens 350 --max-length 1024 \
     --output-tag ${CONDITION}_source-to-sink_q0-4_v1"
done
```

Inspect and judge both conditions. Expand to 20 only if circuit-only responses
are coherent and directionally recover the trait. Failure does not invalidate a
successful level-2 mediation result.

## 18. Recurrent timing diagnostic

After the prompt-only/prompt-end result is frozen, run:

```bash
for CONDITION in upstream_ablated upstream_full_state_rescue upstream_vector_rescue vector_rescue_downstream_ablated; do
  submit_a100 "syc-time-${CONDITION:0:8}" 08:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG --upstream-run $UP_FINAL --downstream-run $DN_FINAL \
     --upstream-k 50 --downstream-k 3000 \
     --upstream-circuit-path $CUT_UP --downstream-circuit-path $CUT_DN \
     --condition $CONDITION --question-indices $Q5 \
     --upstream-position-scope all_positions --interface-timing current_token \
     --max-new-tokens 350 --max-length 1024 \
     --output-tag ${CONDITION}_all-positions_current-token_q0-4_v1"
done
```

Interpretation:

- prompt-end success supports a prompt-formed interface;
- only current-token success supports a recurrent layerwise interface;
- full-state success with vector failure supports a higher-dimensional
  interface;
- vector success without downstream blocking requires a prompt-end-specific
  downstream attribution before rejecting the interface hypothesis.

## 19. Conditional extensions

Run only after the primary sycophancy mechanism and end-to-end comparator pass:

1. a newly frozen external opinion-sycophancy behavior set untouched by current
   development, required for the ICLR version;
2. a closer instruction-only counterfactual with neutral assistant naming;
3. discovery-resample and instruction-template stability;
4. hallucination with a larger held-out behavioral sample;
5. a second model;
6. factual sycophancy as a separately scoped trait;
7. multilingual persona transfer.

These are ICLR-strengthening replications. They must not delay the primary
controlled mechanism.

## 20. Final reporting checklist

Report separately:

- historical causal-cut necessity;
- source-to-sink internal faithfulness and necessity;
- prompt-only mediation effects and controls;
- 2x2 interaction;
- end-to-end internal and behavioral faithfulness;
- connected circuit-only reconstruction, only if it passes;
- retained and eligible edge counts;
- discovery/evaluation sample counts;
- exact model revision and vector hashes;
- all token caps, cap hits, and generated-token distributions;
- paired 95% confidence intervals;
- q0-q4 pilot and q5-q19 primary estimates kept separate;
- all five topology and orthogonal seeds;
- noising/denoising asymmetry;
- IG convergence statistics;
- opinion-sycophancy scope.

Do not claim:

- that high necessity implies sufficiency;
- that a sink-connected shell is a reconstructive circuit;
- that the architectural residual layer itself is a nontrivial bottleneck;
- that one selected k is a unique minimum;
- that a readable direction is automatically the natural mediator;
- that capped generations are equivalent to naturally terminated responses.

# Minimal workshop experiment runbook

Status: final budget-gated plan, 2026-08-18.

This document supersedes the execution order and broad compute budget in
`SYCOPHANCY_PERSONA_CIRCUIT_REPAIRED_RUNBOOK.md`. The longer handoff remains
the detailed project history. This file is the shortest experiment campaign
that can establish or falsify the workshop claim without rerunning valid work.

The campaign is intentionally sequential. Do not submit a later phase until
the preceding gate passes. A submitted job, output directory, or partial row
set is not a completed result.

## 1. Scientific target

The strongest well-matched workshop claim is not yet a self-contained
source-to-logit circuit. It is:

> A sparse upstream causal cut changes opinion-sycophancy through a specific
> layer-20 persona-vector coordinate, and a sparse downstream circuit transmits
> the rescued effect to behavior.

Keep these claims separate:

1. **Sparse causal interception.** Ablating selected upstream edges changes the
   layer-20 coordinate and behavior.
2. **Specific causal interface.** Restoring the missing sycophancy coordinate
   rescues behavior more than equal-norm orthogonal and unrelated-vector
   controls, and downstream blocking suppresses that rescue.
3. **Reconstructive circuit.** A source-to-sink edge set reconstructs the clean
   computation from the corrupt baseline in circuit-only mode.

Claim 2 is the primary workshop result. Claim 3 is conditional. High necessity
does not imply reconstructive sufficiency.

### 1.1 Exact interventions

Let `x+` be the positive-persona prompt, `x-` its token-aligned
negative-persona counterpart, `h20(x)` the residual stream at
`blocks.20.hook_resid_pre`, `p` the final prompt position, and `v` the frozen
unit sycophancy direction.

The upstream attribution target is:

```text
z(x) = <h20(x)[p], v>
```

The upstream causal cut is ranked by contrastive EAP-IG from `x-` to `x+`.
During behavior generation, selected upstream edges in the clean `x+` run are
replaced with their `x-` values at prompt positions only.

For each question, define the missing coordinate after upstream intervention:

```text
delta = <h20_clean[p] - h20_upstream_ablated[p], v>
```

Vector rescue sets:

```text
h20_rescued[p] = h20_upstream_ablated[p] + delta * v
```

This restores exactly the target coordinate, not a fixed global coefficient.
Full-state rescue replaces the complete final-prompt state. Full-prefix rescue
replaces all layer-20 prompt states and is the exact positive control.

The historical downstream ranking was found under coefficient `0 -> -2`
steering at response-prediction positions. The decisive blocking condition
does not add negative steering. It patches selected downstream edges from the
same upstream-ablated, unrescued run into the vector-rescued run. Therefore it
estimates transmission of the rescue-induced effect.

### 1.2 Claim language

Use **one-dimensional causal bottleneck** only if:

- full-prefix rescue is token-exact;
- final-state rescue recovers most of the upstream loss;
- vector rescue recovers approximately 80% or more of that loss;
- vector rescue beats both control families;
- downstream blocking removes most of the vector rescue;
- coherence and termination remain healthy.

For a smaller but significant rescue, use **specific partial causal mediator**
or **low-dimensional causal interface**. If full state succeeds but the vector
does not, the interface is higher-dimensional. If only recurrent
`current_token` rescue succeeds, describe a recurrent interface rather than a
prompt-formed bottleneck.

## 2. Frozen evidence to reuse

Do not rerun these artifacts when the hashes and metadata below match.

| Artifact | Frozen value |
| --- | --- |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Model revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Unit vector SHA-256 | `119ddc01276d1b73f5b3710ee2a42e81731b46ce024f236c232a0b5f401c5689` |
| Raw response vector SHA-256 | `47d080911064d1ff2a03e5331923ed170261073db748f4cb3e2773a2a265edc6` |
| Upstream IG-5 score SHA-256 | `f51b5f941f891bf7687b27dc77198c23e3e84885c9aaef60b2695964c2bec3ea` |
| Downstream IG-5 score SHA-256 | `d242e7ab089b4b892ee0ff12beb4b093953ccd884bf1a8accc785af236bf8ca2` |
| Upstream eligible edges | `471191` |
| Interface | `blocks.20.hook_resid_pre` |
| Downstream corrupt coefficient | `-2` |

Reuse:

- 1,000 positive and 1,000 negative extraction generations;
- all existing judgments and 810 effective response pairs;
- response-vector extraction, layer sweep, and steering validation;
- the frozen 64-row shared discovery manifest;
- the full upstream and downstream IG-5 score tables;
- all frozen Gemma and Qwen refusal results;
- historical sycophancy necessity and joint-ablation results;
- historical generations that terminated naturally before their recorded cap.

Historical headline evidence:

- normal sycophancy score: approximately `66.95`;
- upstream k50 ablation: approximately `8.19`;
- upstream k3000 ablation: approximately `4.16`;
- downstream legacy k500 internal faithfulness/necessity:
  `0.907 / 1.001`;
- downstream legacy k3000 internal faithfulness/necessity:
  `0.978 / 1.003`;
- joint historical k5000/k3000 ablation trait score: approximately `1.31`;
- coherence for the necessity runs remained approximately `98-99`.

Do not reuse as canonical sufficiency evidence:

- legacy low-memory circuit-only generations;
- outputs that hit a 1,000-token cap;
- historical sink-connected cut sets as source-to-sink circuits;
- a historical normal run in a new paired mediation analysis;
- `sycophancy_eval_answer_n20.json` as opinion-sycophancy confirmation.

The 20 official opinion questions are development data because historical
curves have used all of them. Use q0-q4 as the new-condition pilot and q5-q19
as the primary workshop analysis, but do not call q5-q19 untouched.

## 3. Why upstream necessity did not yet establish sufficiency

The historical GPU evaluations used circuits selected with **childless
pruning only**: every retained edge could reach the target, but an edge was
not required to be reachable from the graph input. A later CPU audit counted
how many edges in each evaluated set lie on complete input-to-target paths:

| Requested k | Complete-path edges in evaluated set | Historical faithfulness | Historical necessity |
| ---: | ---: | ---: | ---: |
| 50 | 0 | 0.0009 | 1.096 |
| 100 | 0 | -0.0106 | 1.133 |
| 250 | 198 | 0.0097 | 0.798 |
| 500 | 373 | 0.0048 | 0.813 |
| 1,000 | 782 | -0.0061 | 0.837 |
| 3,000 | 2,681 | 0.0667 | 1.002 |
| 5,000 | 4,719 | 0.1350 | 1.010 |
| 10,000 | 9,761 | 0.1773 | 1.004 |
| 20,000 | 19,887 | 0.1927 | 0.974 |

The k20,000 faithfulness value was measured on the original 20,000-edge,
sink-connected set, **not** on the repaired 19,887-edge source-to-sink subset.
Therefore the exact effect of removing source-disconnected edges remains
unmeasured. The repaired k20,000 evaluation in Phase 2 is mandatory before
deciding that connectivity is insufficient.

If the repaired source-to-sink circuit remains weak, likely remaining causes
include:

- noising and denoising are asymmetric in a nonlinear network;
- a cut can intercept a coordinate without carrying all support needed to
  reconstruct it recursively;
- positive and negative persona prompts are a relatively distant
  counterfactual;
- the ranking objective optimizes marginal intervention on one projection,
  not complete hidden-state reconstruction;
- circuit-only recursion compounds missing fan-in and approximation errors.

The workshop should test both the cheap repaired source-to-sink baseline and
the primary causal-interface claim. Do not begin a costly alternative
reconstruction search until the repaired baseline has actually been measured.

## 4. Required code integration before GPU use

The repaired branch is:

```text
origin/persona-circuit-method-repair-20260817
commit 41f1809
```

The HPC checkout also contains commit `bd3d86c`, which applies two-sided prompt
padding in two behavior runners. Integrate the branches without losing the HPC
wrapper or documentation changes.

Required final behavior:

1. New official behavior generation must use `_align_system_prompt_pair`, which
   pads whichever system prompt is shorter.
2. Frozen discovery/evaluation artifacts must retain their recorded alignment
   and hashes. Do not silently regenerate them under a changed prompt.
3. Apply the two-sided helper to connected behavior and any new natural-prompt
   behavior runner. Add tests covering both positive-shorter and
   negative-shorter pairs.
4. Keep `scripts/slurm/rq_job.sbatch` loading `.env` without printing secrets.

Do not begin Phase 1 until:

```bash
git status --short
git log -3 --oneline
uv sync --locked
PYTHONPYCACHEPREFIX=/tmp/persona-pycache \
  uv run python -m compileall -q src scripts tests EAP-IG/eap
uv run pytest -q
git diff --check
```

The experiment commit must be clean, pushed, and recorded in every lab note.

## 5. HPC shell setup

Run in Bash from `/home/omiro1/cross-lingual-refusal-circuit`.

```bash
test -n "${BASH_VERSION:-}" || exit 1

export CONFIG=configs/qwen_sycophancy.json
export RESULTS=results/qwen2.5-7b-instruct/persona/sycophantic
export INTERMEDIATE=data/intermediate/persona/qwen2.5-7b-instruct/sycophantic
export SHARED="$INTERMEDIATE/shared_circuit_dataset_v1.json"
export UP_EVAL="$INTERMEDIATE/upstream_prompt_disjoint_eval_workshop_v1.json"
export DN_EVAL="$INTERMEDIATE/downstream_prompt_response_disjoint_eval_workshop_v1.json"
export UP_EVAL1="$INTERMEDIATE/upstream_prompt_disjoint_eval_smoke1_v1.json"

export UP_RUN=sharedprompt_n64_ig5_k3000_20260726
export DN_RUN=sharedprompt_n64_ig5_full768_h100_cneg2_20260802
export UNRELATED_VECTOR=results/qwen2.5-7b-instruct/persona/hallucinating/vector/hallucinating_response_avg_diff.pt

export UP_ROOT="$RESULTS/circuit_discovery_upstream_contrastive/$UP_RUN"
export DN_ROOT="$RESULTS/circuit_discovery_steering_path/$DN_RUN"
export CUT_UP="$UP_ROOT/circuits_cut_sets/eapig_top_50.json"
export PATH_UP20="$UP_ROOT/circuits_source_to_sink/eapig_top_20000.json"
export PATH_UP_ALL="$UP_ROOT/circuits_source_to_sink/eapig_top_471191.json"
export PATH_DN="$DN_ROOT/circuits_source_to_sink/steering_path_eapig_top_500.json"
export CONN_ROOT="$INTERMEDIATE/connected_circuit_behavior/${UP_RUN}__${DN_RUN}/ku50_kd500"

export Q0="0"
export Q5="0 1 2 3 4"
export Q15="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
export PILOT16="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
export COMMON_MAX_NEW_TOKENS=512

mkdir -p logs/rq
```

Submission helper:

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

After every wave:

```bash
squeue -u "$USER"
sacct -X --starttime 2026-08-18 \
  --format=JobID,JobName%36,State,Elapsed,AllocTRES,MaxRSS,ExitCode
```

## 6. Phase 0: CPU provenance and topology gate

Estimated GPU cost: `0 A100-hours`.

### 6.1 Verify immutable artifacts

```bash
test "$(sha256sum "$RESULTS/vector/sycophantic_direction.pt" | cut -d' ' -f1)" \
  = "119ddc01276d1b73f5b3710ee2a42e81731b46ce024f236c232a0b5f401c5689"
test "$(sha256sum "$RESULTS/vector/sycophantic_response_avg_diff.pt" | cut -d' ' -f1)" \
  = "47d080911064d1ff2a03e5331923ed170261073db748f4cb3e2773a2a265edc6"
test "$(sha256sum "$UP_ROOT/eapig_edge_scores.parquet" | cut -d' ' -f1)" \
  = "f51b5f941f891bf7687b27dc77198c23e3e84885c9aaef60b2695964c2bec3ea"
test "$(sha256sum "$DN_ROOT/eapig_edge_scores.parquet" | cut -d' ' -f1)" \
  = "d242e7ab089b4b892ee0ff12beb4b093953ccd884bf1a8accc785af236bf8ca2"

jq -e '.status == "complete" and .ig_steps == 5 and .eligible_edges == 471191' \
  "$UP_ROOT/eapig_run_metadata.json"
jq -e '.status == "complete" and .ig_steps == 5' \
  "$DN_ROOT/eapig_run_metadata.json"
```

Stop if any assertion fails. Restore the canonical artifact bundle; do not
partially regenerate it.

### 6.2 Freeze disjoint evaluation manifests

```bash
uv run python scripts/rq.py persona.freeze-upstream-evaluation \
  --config "$CONFIG" \
  --discovery-manifest "$SHARED" \
  --num-prompts 32 \
  --seed 20260811 \
  --max-length 256 \
  --output "$UP_EVAL"

uv run python scripts/rq.py persona.freeze-upstream-evaluation \
  --config "$CONFIG" \
  --discovery-manifest "$SHARED" \
  --num-prompts 1 \
  --seed 20260811 \
  --max-length 256 \
  --output "$UP_EVAL1"

uv run python scripts/rq.py persona.freeze-downstream-evaluation \
  --config "$CONFIG" \
  --discovery-manifest "$SHARED" \
  --num-pairs 32 \
  --seed 20260811 \
  --max-length 768 \
  --output "$DN_EVAL"
```

### 6.3 Token audit

```bash
uv run python scripts/rq.py persona.audit-token-lengths \
  --config "$CONFIG" \
  --sample-manifest "$SHARED" \
  --upstream-evaluation-manifest "$UP_EVAL" \
  --downstream-evaluation-manifest "$DN_EVAL" \
  --caps 256 512 768 1024 \
  --output "$RESULTS/token_length_audit_workshop_v1.json"
```

Required:

- upstream prompts fit 256;
- downstream teacher-forced transcripts fit 768;
- official behavior prompts fit 1024.

Historical cap differences do not imply a rerun when the response terminated
naturally before every relevant cap.

### 6.4 Rebuild only the circuits needed now

```bash
uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" \
  --scope upstream \
  --run-name "$UP_RUN" \
  --families causal_cut source_to_sink \
  --k-values 50 20000 471191

uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" \
  --scope downstream \
  --run-name "$DN_RUN" \
  --families source_to_sink \
  --k-values 500
```

Assertions:

```bash
jq -e '.circuit_family == "causal_cut" and .requested_k == 50 and .retained_edges > 0' "$CUT_UP"
jq -e '.circuit_family == "source_to_sink" and .connectivity.all_edges_on_source_to_sink_paths == true' "$PATH_UP20"
jq -e '.circuit_family == "source_to_sink" and .connectivity.all_edges_on_source_to_sink_paths == true' "$PATH_UP_ALL"
jq -e '.circuit_family == "source_to_sink" and .connectivity.all_edges_on_source_to_sink_paths == true and .retained_edges > 0' "$PATH_DN"
```

Record the retained edge counts. Requested k is not the same as retained size.

## 7. Phase 1: smallest technical GPU gate

Estimated cost: `0.3-0.6 A100-hours`.

This phase tests the repaired algebra and behavior runner. It is not a
scientific result.

### 7.1 All-edge recursive reconstruction on one held-out pair

```bash
submit_a100 syc-up-all-edge-smoke 01:00:00 \
  persona.evaluate-upstream-contrastive-circuit \
  "--config $CONFIG --run-name $UP_RUN --k 471191 \
   --circuit-path $PATH_UP_ALL \
   --evaluation-manifest $UP_EVAL1 --num-pairs 1 --max-length 256 \
   --output-tag source_to_sink_all_edges_n1_technical_v1"
```

After completion:

```bash
export UP_ALL_AGG="$UP_ROOT/evaluation/source_to_sink_all_edges_n1_technical_v1/aggregate.json"
jq -e '
  .status == "complete"
  and .truncation == false
  and .connectivity.all_edges_on_source_to_sink_paths == true
  and .faithfulness >= 0.99
  and .necessity >= 0.99
' "$UP_ALL_AGG"
```

If this fails, stop. The recursive construction implementation or graph
contract is wrong; no circuit-only result is interpretable.

### 7.2 One-question, 16-token connected smoke

```bash
for CONDITION in \
  normal \
  upstream_ablated \
  upstream_full_prefix_state_rescue \
  upstream_vector_rescue \
  vector_rescue_downstream_ablated
do
  submit_a100 "syc-tech-${CONDITION:0:12}" 01:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG \
     --upstream-run $UP_RUN --downstream-run $DN_RUN \
     --upstream-k 50 --downstream-k 500 \
     --upstream-circuit-path $CUT_UP \
     --downstream-circuit-path $PATH_DN \
     --condition $CONDITION --question-indices 0 \
     --upstream-position-scope prompt_only \
     --interface-timing prompt_end \
     --max-new-tokens 16 --max-length 1024 \
     --output-tag ${CONDITION}_q0_t16_technical_v1"
done
```

Required exact positive control:

```bash
NORMAL="$CONN_ROOT/normal_q0_t16_technical_v1/responses.jsonl"
PREFIX="$CONN_ROOT/upstream_full_prefix_state_rescue_q0_t16_technical_v1/responses.jsonl"
test "$(jq -r '.generated_token_ids_sha256' "$NORMAL")" \
  = "$(jq -r '.generated_token_ids_sha256' "$PREFIX")"
```

Also verify every metadata file has `status=complete`, finite interface
diagnostics, and the expected circuit/vector hashes. A 16-token cap hit is
allowed only in this technical smoke.

## 8. Phase 2: q0-q4 causal-interface pilot

Estimated cost: `3-6 A100-hours`.

This is the decisive go/no-go phase. It uses the existing full64 IG-5 ranking:
upstream causal cut k50 and downstream source-to-sink k500.

### 8.1 Internal checks in parallel

```bash
submit_a100 syc-up-k20k-n32 02:00:00 \
  persona.evaluate-upstream-contrastive-circuit \
  "--config $CONFIG --run-name $UP_RUN --k 20000 \
   --circuit-path $PATH_UP20 \
   --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
   --output-tag source_to_sink_k20000_n32_workshop_v1"

submit_a100 syc-dn-k500-n32 04:00:00 \
  persona.evaluate-steering-circuit \
  "--config $CONFIG --run-name $DN_RUN --k 500 \
   --circuit-path $PATH_DN \
   --evaluation-manifest $DN_EVAL --num-prompts 32 --batch-size 1 \
   --max-length 768 --output-tag source_to_sink_k500_n32_workshop_v1"
```

Expected:

- upstream k20,000 may remain near historical faithfulness `0.19`;
- downstream k500 should retain high faithfulness and necessity;
- both must be finite, untruncated, and provenance-complete.

Do not expand upstream reconstruction if k20,000 faithfulness is below `0.30`.

### 8.2 Eight pilot conditions

```bash
for CONDITION in \
  normal \
  upstream_ablated \
  upstream_full_prefix_state_rescue \
  upstream_full_state_rescue \
  upstream_vector_rescue \
  upstream_orthogonal_rescue \
  upstream_unrelated_vector_rescue \
  vector_rescue_downstream_ablated
do
  EXTRA=""
  test "$CONDITION" = "upstream_orthogonal_rescue" && EXTRA="--orthogonal-seed 0"
  test "$CONDITION" = "upstream_unrelated_vector_rescue" && \
    EXTRA="--unrelated-vector-path $UNRELATED_VECTOR"

  submit_a100 "syc-pilot-${CONDITION:0:11}" 08:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG \
     --upstream-run $UP_RUN --downstream-run $DN_RUN \
     --upstream-k 50 --downstream-k 500 \
     --upstream-circuit-path $CUT_UP \
     --downstream-circuit-path $PATH_DN \
     --condition $CONDITION --question-indices $Q5 \
     --upstream-position-scope prompt_only \
     --interface-timing prompt_end \
     --max-new-tokens $COMMON_MAX_NEW_TOKENS --max-length 1024 \
     --output-tag ${CONDITION}_q0-4_pilot_v1 $EXTRA"
done
```

Inspect all 40 generations before judging:

```bash
find "$CONN_ROOT" -path '*_q0-4_pilot_v1/responses.jsonl' -print0 |
  while IFS= read -r -d '' path; do
    echo "===== $path"
    jq -r '[.question_index,.condition,.generated_token_count,.hit_max_new_tokens,.answer] | @tsv' "$path"
  done
```

Stop and debug if full-prefix output differs from normal:

```bash
diff \
  <(jq -r '[.question_index,.generated_token_ids_sha256] | @tsv' \
    "$CONN_ROOT/normal_q0-4_pilot_v1/responses.jsonl" | sort) \
  <(jq -r '[.question_index,.generated_token_ids_sha256] | @tsv' \
    "$CONN_ROOT/upstream_full_prefix_state_rescue_q0-4_pilot_v1/responses.jsonl" | sort)
```

Judge:

```bash
for TAG in \
  normal \
  upstream_ablated \
  upstream_full_prefix_state_rescue \
  upstream_full_state_rescue \
  upstream_vector_rescue \
  upstream_orthogonal_rescue \
  upstream_unrelated_vector_rescue \
  vector_rescue_downstream_ablated
do
  DIR="$CONN_ROOT/${TAG}_q0-4_pilot_v1"
  uv run python scripts/rq.py persona.judge \
    --config "$CONFIG" \
    --input "$DIR/responses.jsonl" \
    --output "$DIR/scored.jsonl" \
    --artifact eval \
    --protocol-field behavior_protocol_id \
    --resume
done
```

Quick means:

```bash
for TAG in \
  normal \
  upstream_ablated \
  upstream_full_prefix_state_rescue \
  upstream_full_state_rescue \
  upstream_vector_rescue \
  upstream_orthogonal_rescue \
  upstream_unrelated_vector_rescue \
  vector_rescue_downstream_ablated
do
  jq -s --arg tag "$TAG" '{
    condition: $tag,
    n: length,
    trait: (map(.trait_score) | add / length),
    coherence: (map(.coherence_score) | add / length),
    cap_hits: (map(select(.hit_max_new_tokens == true)) | length)
  }' "$CONN_ROOT/${TAG}_q0-4_pilot_v1/scored.jsonl"
done
```

### 8.3 Expansion gate

These are execution gates, not publication thresholds.

- upstream ablation lowers mean trait score by at least 20 points;
- full-state rescue recovers at least 50% of the loss;
- vector rescue recovers at least 30% of the loss;
- vector rescue exceeds the seed-0 orthogonal and unrelated controls by at
  least 10 points each;
- downstream blocking removes at least 30% of the vector rescue;
- mean coherence is at least 80 in every condition;
- no condition has a cap hit or obvious repetitive degeneration;
- full-prefix rescue is token-exact for all five questions.

Decision branches:

| Observation | Action |
| --- | --- |
| Full-prefix mismatch | Stop: implementation bug. |
| Upstream cut has no effect | Inspect alignment/circuit hashes; do not rerank first. |
| Full state works, vector fails | Skip 1D claim; conditionally test a 2/4/8/16-D PCA subspace. |
| Vector works, blocking works | Continue to Phases 3 and 4. |
| Vector works, blocking fails | Run one matched prompt-end downstream attribution before rejecting mediation. |
| Prompt-end rescues fail | Run the q0-q4 recurrent timing diagnostic in Section 11.2. |
| Coherence or termination fails | Treat behavior scores as invalid; do not expand. |

## 9. Phase 3: numerical guardrail and circuit controls

Estimated cost: `2-4 A100-hours`.

Run only if Phase 2 supports mediation.

### 9.1 Why IG-5 and IG-20

This is not a search over IG hyperparameters. It is a cheap numerical
guardrail on the operating circuits. The full64 IG-5 tables are valuable
frozen assets. Fixed-row n16 IG-5 versus IG-20 isolates integration accuracy
from sample identity. IG-50 is not scheduled unless IG-5 and IG-20 differ
functionally.

### 9.2 Fixed-row attribution

```bash
for STEPS in 5 20; do
  submit_a100 "syc-up-ig${STEPS}-n16" 04:00:00 \
    persona.run-upstream-contrastive-eapig \
    "--config $CONFIG \
     --run-name syc_up_ig${STEPS}_n16_guard_v1 \
     --sample-manifest $SHARED --num-pairs 16 \
     --manifest-sample-indices $PILOT16 \
     --batch-size 1 --max-length 256 \
     --ig-steps $STEPS --k 50"

  submit_a100 "syc-dn-ig${STEPS}-n16" 08:00:00 \
    persona.run-steering-eapig \
    "--config $CONFIG \
     --run-name syc_dn_ig${STEPS}_n16_guard_v1 \
     --sample-manifest $SHARED --num-prompts 16 \
     --manifest-sample-indices $PILOT16 \
     --batch-size 1 --max-length 768 \
     --ig-steps $STEPS --corrupt-coefficient -2 \
     --minimum-trait-score 80 --minimum-coherence-score 80 \
     --maximum-negative-trait-score 20 \
     --metric contrastive_response_logprob \
     --k-values 500"
done
```

Rebuild the operating circuit family:

```bash
for STEPS in 5 20; do
  uv run python scripts/rq.py persona.rebuild-nested-circuits \
    --config "$CONFIG" --scope upstream \
    --run-name "syc_up_ig${STEPS}_n16_guard_v1" \
    --families causal_cut --k-values 50

  uv run python scripts/rq.py persona.rebuild-nested-circuits \
    --config "$CONFIG" --scope downstream \
    --run-name "syc_dn_ig${STEPS}_n16_guard_v1" \
    --families source_to_sink --k-values 500
done
```

Evaluate both rankings on the same held-out rows:

```bash
for STEPS in 5 20; do
  submit_a100 "syc-up-ig${STEPS}-eval" 02:00:00 \
    persona.evaluate-upstream-contrastive-circuit \
    "--config $CONFIG \
     --run-name syc_up_ig${STEPS}_n16_guard_v1 --k 50 \
     --circuit-path $RESULTS/circuit_discovery_upstream_contrastive/syc_up_ig${STEPS}_n16_guard_v1/circuits_cut_sets/eapig_top_50.json \
     --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
     --allow-disconnected-circuit-only \
     --output-tag causal_cut_k50_n32_guard_v1"

  submit_a100 "syc-dn-ig${STEPS}-eval" 04:00:00 \
    persona.evaluate-steering-circuit \
    "--config $CONFIG \
     --run-name syc_dn_ig${STEPS}_n16_guard_v1 --k 500 \
     --circuit-path $RESULTS/circuit_discovery_steering_path/syc_dn_ig${STEPS}_n16_guard_v1/circuits_source_to_sink/steering_path_eapig_top_500.json \
     --evaluation-manifest $DN_EVAL --num-prompts 32 --batch-size 1 \
     --max-length 768 \
     --output-tag source_to_sink_k500_n32_guard_v1"
done
```

Compare score tables:

```bash
uv run python scripts/rq.py persona.compare-score-tables \
  --config "$CONFIG" --scope upstream \
  --left "$RESULTS/circuit_discovery_upstream_contrastive/syc_up_ig5_n16_guard_v1/eapig_edge_scores.parquet" \
  --right "$RESULTS/circuit_discovery_upstream_contrastive/syc_up_ig20_n16_guard_v1/eapig_edge_scores.parquet" \
  --k-values 50 \
  --output "$RESULTS/upstream_ig5_vs_ig20_n16_guard_v1.json"

uv run python scripts/rq.py persona.compare-score-tables \
  --config "$CONFIG" --scope downstream \
  --left "$RESULTS/circuit_discovery_steering_path/syc_dn_ig5_n16_guard_v1/eapig_edge_scores.parquet" \
  --right "$RESULTS/circuit_discovery_steering_path/syc_dn_ig20_n16_guard_v1/eapig_edge_scores.parquet" \
  --k-values 500 \
  --output "$RESULTS/downstream_ig5_vs_ig20_n16_guard_v1.json"
```

Retain the full64 IG-5 ranking when:

- upstream held-out necessity differs by no more than `0.05`;
- downstream held-out faithfulness and necessity each differ by no more than
  `0.05`;
- signs and denominators are stable.

Report edge overlap descriptively; do not reject a functionally stable circuit
only because raw top-k Jaccard is modest.

If a scope fails, run n16 IG-50 only for that scope. If IG-20 and IG-50 agree
functionally within `0.05`, run one full64 IG-20 discovery for that scope.
Otherwise use full64 IG-50. Never run IG-50 merely because it was listed in an
older plan.

Exact upstream IG-50 guard, only if upstream IG-5 versus IG-20 fails:

```bash
submit_a100 syc-up-ig50-n16 08:00:00 \
  persona.run-upstream-contrastive-eapig \
  "--config $CONFIG \
   --run-name syc_up_ig50_n16_guard_v1 \
   --sample-manifest $SHARED --num-pairs 16 \
   --manifest-sample-indices $PILOT16 \
   --batch-size 1 --max-length 256 --ig-steps 50 --k 50"

uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" --scope upstream \
  --run-name syc_up_ig50_n16_guard_v1 \
  --families causal_cut --k-values 50

submit_a100 syc-up-ig50-eval 02:00:00 \
  persona.evaluate-upstream-contrastive-circuit \
  "--config $CONFIG --run-name syc_up_ig50_n16_guard_v1 --k 50 \
   --circuit-path $RESULTS/circuit_discovery_upstream_contrastive/syc_up_ig50_n16_guard_v1/circuits_cut_sets/eapig_top_50.json \
   --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
   --allow-disconnected-circuit-only \
   --output-tag causal_cut_k50_n32_guard_v1"
```

Exact downstream IG-50 guard, only if downstream IG-5 versus IG-20 fails:

```bash
submit_a100 syc-dn-ig50-n16 16:00:00 \
  persona.run-steering-eapig \
  "--config $CONFIG \
   --run-name syc_dn_ig50_n16_guard_v1 \
   --sample-manifest $SHARED --num-prompts 16 \
   --manifest-sample-indices $PILOT16 \
   --batch-size 1 --max-length 768 --ig-steps 50 \
   --corrupt-coefficient -2 \
   --minimum-trait-score 80 --minimum-coherence-score 80 \
   --maximum-negative-trait-score 20 \
   --metric contrastive_response_logprob --k-values 500"

uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" --scope downstream \
  --run-name syc_dn_ig50_n16_guard_v1 \
  --families source_to_sink --k-values 500

submit_a100 syc-dn-ig50-eval 04:00:00 \
  persona.evaluate-steering-circuit \
  "--config $CONFIG --run-name syc_dn_ig50_n16_guard_v1 --k 500 \
   --circuit-path $RESULTS/circuit_discovery_steering_path/syc_dn_ig50_n16_guard_v1/circuits_source_to_sink/steering_path_eapig_top_500.json \
   --evaluation-manifest $DN_EVAL --num-prompts 32 --batch-size 1 \
   --max-length 768 \
   --output-tag source_to_sink_k500_n32_guard_v1"
```

If a full64 rerank is required, set `FINAL_STEPS` to the selected value:

```bash
export FINAL_STEPS=20
```

Upstream full64 rerank:

```bash
export UP_RUN="syc_up_n64_ig${FINAL_STEPS}_workshop_v1"

submit_a100 "syc-up-n64-ig${FINAL_STEPS}" 24:00:00 \
  persona.run-upstream-contrastive-eapig \
  "--config $CONFIG --run-name $UP_RUN \
   --sample-manifest $SHARED --num-pairs 64 \
   --batch-size 1 --max-length 256 \
   --ig-steps $FINAL_STEPS --k 50000"

uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" --scope upstream --run-name "$UP_RUN" \
  --families causal_cut source_to_sink \
  --k-values 50 20000 50000

export UP_ROOT="$RESULTS/circuit_discovery_upstream_contrastive/$UP_RUN"
export CUT_UP="$UP_ROOT/circuits_cut_sets/eapig_top_50.json"
export PATH_UP20="$UP_ROOT/circuits_source_to_sink/eapig_top_20000.json"
```

Downstream full64 rerank:

```bash
export DN_RUN="syc_dn_n64_ig${FINAL_STEPS}_workshop_v1"

submit_a100 "syc-dn-n64-ig${FINAL_STEPS}" 36:00:00 \
  persona.run-steering-eapig \
  "--config $CONFIG --run-name $DN_RUN \
   --sample-manifest $SHARED --num-prompts 64 \
   --batch-size 1 --max-length 768 \
   --ig-steps $FINAL_STEPS --corrupt-coefficient -2 \
   --minimum-trait-score 80 --minimum-coherence-score 80 \
   --maximum-negative-trait-score 20 \
   --metric contrastive_response_logprob --k-values 500"

uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config "$CONFIG" --scope downstream --run-name "$DN_RUN" \
  --families source_to_sink --k-values 500

export DN_ROOT="$RESULTS/circuit_discovery_steering_path/$DN_RUN"
export PATH_DN="$DN_ROOT/circuits_source_to_sink/steering_path_eapig_top_500.json"
```

After either full64 rerank, reset:

```bash
export CONN_ROOT="$INTERMEDIATE/connected_circuit_behavior/${UP_RUN}__${DN_RUN}/ku50_kd500"
```

Then rerun the complete q0-q4 Phase 2 pilot with the new circuit paths. Do not
carry an IG-5 pilot decision into q5-q19 after changing either ranking.

### 9.3 Topology-matched controls

The commands below assume the full64 IG-5 rankings passed. If a scope reranks,
replace `UP_RUN` or `DN_RUN` with the frozen final run name.

```bash
uv run python scripts/rq.py persona.build-circuit-controls \
  --config "$CONFIG" --scope upstream --run-name "$UP_RUN" \
  --family causal_cut --k-values 50 --seeds 0 1 2 3 4

uv run python scripts/rq.py persona.build-circuit-controls \
  --config "$CONFIG" --scope downstream --run-name "$DN_RUN" \
  --family source_to_sink --k-values 500 --seeds 0 1 2 3 4
```

Evaluate:

```bash
for SEED in 0 1 2 3 4; do
  submit_a100 "syc-up-ctrl${SEED}" 02:00:00 \
    persona.evaluate-upstream-contrastive-circuit \
    "--config $CONFIG --run-name $UP_RUN --k 50 \
     --circuit-path $UP_ROOT/controls/topology_stratified_v2/causal_cut/seed_${SEED}/eapig_top_50.json \
     --evaluation-manifest $UP_EVAL --num-pairs 32 --max-length 256 \
     --allow-disconnected-circuit-only \
     --output-tag control_topology_v2_seed${SEED}_k50_n32"

  submit_a100 "syc-dn-ctrl${SEED}" 04:00:00 \
    persona.evaluate-steering-circuit \
    "--config $CONFIG --run-name $DN_RUN --k 500 \
     --circuit-path $DN_ROOT/controls/topology_stratified_v2/source_to_sink/seed_${SEED}/steering_path_eapig_top_500.json \
     --evaluation-manifest $DN_EVAL --num-prompts 32 --batch-size 1 \
     --max-length 768 \
     --output-tag control_topology_v2_seed${SEED}_k500_n32"
done
```

The real upstream cut should have materially greater necessity than the five
controls. The real downstream circuit should have materially greater
faithfulness and necessity. Report all seeds.

## 10. Phase 4: primary q5-q19 mediation

Estimated cost: `12-24 A100-hours`.

Run exactly 12 generation jobs:

- normal;
- upstream ablated;
- full-prefix rescue;
- full-state rescue;
- vector rescue;
- unrelated-vector rescue;
- vector rescue with downstream blocking;
- five orthogonal-vector seeds.

Do not rerun downstream-only or joint ablation for the primary table. Their
historical necessity evidence is already strong, and they do not identify the
mediator. They may be added later as a secondary 2x2 analysis.

```bash
for CONDITION in \
  normal \
  upstream_ablated \
  upstream_full_prefix_state_rescue \
  upstream_full_state_rescue \
  upstream_vector_rescue \
  upstream_unrelated_vector_rescue \
  vector_rescue_downstream_ablated
do
  EXTRA=""
  test "$CONDITION" = "upstream_unrelated_vector_rescue" && \
    EXTRA="--unrelated-vector-path $UNRELATED_VECTOR"

  submit_a100 "syc-primary-${CONDITION:0:9}" 12:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG \
     --upstream-run $UP_RUN --downstream-run $DN_RUN \
     --upstream-k 50 --downstream-k 500 \
     --upstream-circuit-path $CUT_UP \
     --downstream-circuit-path $PATH_DN \
     --condition $CONDITION --question-indices $Q15 \
     --upstream-position-scope prompt_only \
     --interface-timing prompt_end \
     --max-new-tokens $COMMON_MAX_NEW_TOKENS --max-length 1024 \
     --output-tag ${CONDITION}_q5-19_primary_v1 $EXTRA"
done

for SEED in 0 1 2 3 4; do
  submit_a100 "syc-primary-orth${SEED}" 12:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG \
     --upstream-run $UP_RUN --downstream-run $DN_RUN \
     --upstream-k 50 --downstream-k 500 \
     --upstream-circuit-path $CUT_UP \
     --downstream-circuit-path $PATH_DN \
     --condition upstream_orthogonal_rescue \
     --orthogonal-seed $SEED --question-indices $Q15 \
     --upstream-position-scope prompt_only \
     --interface-timing prompt_end \
     --max-new-tokens $COMMON_MAX_NEW_TOKENS --max-length 1024 \
     --output-tag orthogonal_seed${SEED}_q5-19_primary_v1"
done
```

Inspect raw outputs and require zero cap hits:

```bash
find "$CONN_ROOT" -path '*q5-19_primary_v1/responses.jsonl' -print0 |
  while IFS= read -r -d '' path; do
    echo "===== $path"
    jq -r '[.question_index,.condition,.generated_token_count,.hit_max_new_tokens,.answer] | @tsv' "$path"
  done
```

If any primary row hits 512 tokens, do not score that campaign as complete.
Inspect the cause. If it is coherent truncation rather than degeneration,
rerun all 12 conditions at one common cap of 768 under new output tags.

Verify full-prefix exactness before paid judging:

```bash
diff \
  <(jq -r '[.question_index,.generated_token_ids_sha256] | @tsv' \
    "$CONN_ROOT/normal_q5-19_primary_v1/responses.jsonl" | sort) \
  <(jq -r '[.question_index,.generated_token_ids_sha256] | @tsv' \
    "$CONN_ROOT/upstream_full_prefix_state_rescue_q5-19_primary_v1/responses.jsonl" | sort)
```

Judge:

```bash
for TAG in \
  normal \
  upstream_ablated \
  upstream_full_prefix_state_rescue \
  upstream_full_state_rescue \
  upstream_vector_rescue \
  upstream_unrelated_vector_rescue \
  vector_rescue_downstream_ablated
do
  DIR="$CONN_ROOT/${TAG}_q5-19_primary_v1"
  uv run python scripts/rq.py persona.judge \
    --config "$CONFIG" \
    --input "$DIR/responses.jsonl" \
    --output "$DIR/scored.jsonl" \
    --artifact eval \
    --protocol-field behavior_protocol_id \
    --resume
done

for SEED in 0 1 2 3 4; do
  DIR="$CONN_ROOT/orthogonal_seed${SEED}_q5-19_primary_v1"
  uv run python scripts/rq.py persona.judge \
    --config "$CONFIG" \
    --input "$DIR/responses.jsonl" \
    --output "$DIR/scored.jsonl" \
    --artifact eval \
    --protocol-field behavior_protocol_id \
    --resume
done
```

Analyze paired question-level effects:

```bash
uv run python scripts/rq.py persona.analyze-bottleneck-mediation \
  --config "$CONFIG" \
  --normal-scored "$CONN_ROOT/normal_q5-19_primary_v1/scored.jsonl" \
  --upstream-ablated-scored "$CONN_ROOT/upstream_ablated_q5-19_primary_v1/scored.jsonl" \
  --vector-rescue-scored "$CONN_ROOT/upstream_vector_rescue_q5-19_primary_v1/scored.jsonl" \
  --full-state-rescue-scored "$CONN_ROOT/upstream_full_state_rescue_q5-19_primary_v1/scored.jsonl" \
  --full-prefix-state-rescue-scored "$CONN_ROOT/upstream_full_prefix_state_rescue_q5-19_primary_v1/scored.jsonl" \
  --orthogonal-rescue-scored \
    "$CONN_ROOT/orthogonal_seed0_q5-19_primary_v1/scored.jsonl" \
    "$CONN_ROOT/orthogonal_seed1_q5-19_primary_v1/scored.jsonl" \
    "$CONN_ROOT/orthogonal_seed2_q5-19_primary_v1/scored.jsonl" \
    "$CONN_ROOT/orthogonal_seed3_q5-19_primary_v1/scored.jsonl" \
    "$CONN_ROOT/orthogonal_seed4_q5-19_primary_v1/scored.jsonl" \
  --unrelated-rescue-scored "$CONN_ROOT/upstream_unrelated_vector_rescue_q5-19_primary_v1/scored.jsonl" \
  --downstream-blocked-scored "$CONN_ROOT/vector_rescue_downstream_ablated_q5-19_primary_v1/scored.jsonl" \
  --minimum-token-cap "$COMMON_MAX_NEW_TOKENS" \
  --bootstrap-samples 10000 \
  --output "$RESULTS/bottleneck_mediation_q5-19_primary_workshop_v1"
```

Primary evidence requires 95% paired bootstrap intervals above zero for:

- `upstream_cut_effect_points`;
- `vector_rescue_points`;
- `vector_minus_orthogonal_points`;
- `vector_minus_unrelated_points`;
- `downstream_blocking_points`.

Also require:

- full-prefix exact-match rate `1.0`;
- coordinate rescue fraction approximately `1.0`;
- stable bootstrap denominators;
- no generic coherence collapse;
- no cap-hit dependence.

## 11. Phase 5: conditional experiments only

Estimated cost for one ordinary diagnostic branch: `3-8 A100-hours`.
The reconstructive-sufficiency extension below has a separate gated budget.

### 11.1 Upstream reconstructive sufficiency

The historical k20,000 result does **not** evaluate the repaired source-to-sink
subset. It evaluates the original 20,000-edge sink-connected circuit. The CPU
audit found that 19,887 of those edges lie on complete paths, but the exact
19,887-edge subset must be evaluated before changing discovery methods.

Decision:

- if repaired k20,000 faithfulness reaches `>=0.85`, retain the existing
  EAP-IG ranking and evaluate its recurrent interface composition;
- if it is `0.30-0.85`, test one larger repaired frontier before changing
  objectives;
- if it remains `<0.30`, use the reconstruction-aware method below.

The reconstruction-aware extension uses direct hard-concrete edge pruning
over the exact circuit-only computation:

```text
corrupt receiver
+ gated (recursively mixed parent - corrupt parent) contributions
```

It separately learns:

1. a source-to-layer-20 circuit that reconstructs the persona-vector
   coordinate at every response-prediction position;
2. a layer-20-to-output circuit that reconstructs logits caused by restoring
   only that coordinate;
3. a strict recurrent composition that discards all clean-minus-corrupt
   layer-20 components orthogonal to the persona vector.

The exact method, commands, gates, controls, contingencies, and costs are in:

```text
SUFFICIENT_PERSONA_CIRCUIT_METHOD.md
```

Run its vector-only oracle and the repaired EAP k20,000 baseline before mask
training. Expected cost if edge pruning is triggered:

- `16-30 A100-hours` to determine whether both sufficient half-circuits exist;
- `37-72 A100-hours` through the q0-q4 composed behavioral pilot;
- q5-q19 confirmation only if the pilot passes.

Do not run the former k50,000 EAP fallback unless it is explicitly desired as
a descriptive historical comparator. Failure of the reconstructive extension
does not invalidate a successful mediation result.

### 11.2 Recurrent timing diagnostic

Run only if prompt-end rescue is weak despite an effective upstream cut:

```bash
for CONDITION in \
  upstream_ablated \
  upstream_full_state_rescue \
  upstream_vector_rescue \
  vector_rescue_downstream_ablated
do
  submit_a100 "syc-recurrent-${CONDITION:0:9}" 08:00:00 \
    persona.behavior-connected-circuits \
    "--config $CONFIG \
     --upstream-run $UP_RUN --downstream-run $DN_RUN \
     --upstream-k 50 --downstream-k 500 \
     --upstream-circuit-path $CUT_UP \
     --downstream-circuit-path $PATH_DN \
     --condition $CONDITION --question-indices $Q5 \
     --upstream-position-scope all_positions \
     --interface-timing current_token \
     --max-new-tokens $COMMON_MAX_NEW_TOKENS --max-length 1024 \
     --output-tag ${CONDITION}_q0-4_recurrent_diagnostic_v1"
done
```

### 11.3 End-to-end comparator

End-to-end is not automatically better. It gives a direct behavioral circuit
but cannot by itself establish the layer-20 factorization. Run it when:

- the bottleneck decomposition fails and needs a fair comparator; or
- the primary mediation result succeeds early and at least 8 A100-hours remain.

Smoke on eight fixed rows:

```bash
export E2E_SMOKE=syc_e2e_n8_ig5_workshop_smoke_v1
export DN_EVAL8="$INTERMEDIATE/downstream_prompt_response_disjoint_eval_e2e8_v1.json"

uv run python scripts/rq.py persona.freeze-downstream-evaluation \
  --config "$CONFIG" \
  --discovery-manifest "$SHARED" \
  --num-pairs 8 --seed 20260811 --max-length 768 \
  --output "$DN_EVAL8"

submit_a100 syc-e2e-n8-smoke 06:00:00 \
  persona.run-end-to-end-eapig \
  "--config $CONFIG --run-name $E2E_SMOKE \
   --sample-manifest $SHARED --num-prompts 8 \
   --manifest-sample-indices 0 1 2 3 4 5 6 7 \
   --batch-size 1 --max-length 768 --ig-steps 5 \
   --k-values 500 3000 10000 20000"
```

Evaluate only non-empty complete-path frontiers:

```bash
export E2E_SMOKE_ROOT="$RESULTS/circuit_discovery_end_to_end/$E2E_SMOKE"
export E2E_SMOKE_KS="$(
  jq -r '.circuits[]
    | select(.family == "source_to_sink" and .retained_edges > 0)
    | .requested_k' "$E2E_SMOKE_ROOT/eapig_run_metadata.json" |
  paste -sd' ' -
)"
test -n "$E2E_SMOKE_KS"

for K in $E2E_SMOKE_KS; do
  submit_a100 "syc-e2e-smoke-k${K}" 04:00:00 \
    persona.evaluate-end-to-end-circuit \
    "--config $CONFIG --run-name $E2E_SMOKE --k $K \
     --evaluation-manifest $DN_EVAL8 --num-prompts 8 \
     --max-length 768 --output-tag k${K}_n8_recon_gate_v1"
done
```

If no frontier reaches held-out faithfulness `0.85`, stop and report the
end-to-end object only as a partial causal circuit.

If one reaches `0.85`, run exactly one full64 IG-5 discovery:

```bash
export E2E_FINAL=syc_e2e_n64_ig5_workshop_v1

submit_a100 syc-e2e-n64-ig5 24:00:00 \
  persona.run-end-to-end-eapig \
  "--config $CONFIG --run-name $E2E_FINAL \
   --sample-manifest $SHARED --num-prompts 64 \
   --batch-size 1 --max-length 768 --ig-steps 5 \
   --k-values 500 3000 10000 20000"
```

Evaluate the full64 frontiers on the 32 frozen held-out rows:

```bash
export E2E_ROOT="$RESULTS/circuit_discovery_end_to_end/$E2E_FINAL"
export E2E_KS="$(
  jq -r '.circuits[]
    | select(.family == "source_to_sink" and .retained_edges > 0)
    | .requested_k' "$E2E_ROOT/eapig_run_metadata.json" |
  paste -sd' ' -
)"

for K in $E2E_KS; do
  submit_a100 "syc-e2e-final-k${K}" 06:00:00 \
    persona.evaluate-end-to-end-circuit \
    "--config $CONFIG --run-name $E2E_FINAL --k $K \
     --evaluation-manifest $DN_EVAL --num-prompts 32 \
     --max-length 768 --output-tag k${K}_n32_workshop_v1"
done
```

Select the smallest `E2E_K` with faithfulness at least `0.85`, high necessity,
a positive clean-corrupt denominator, and no sharp non-monotonicity:

```bash
export E2E_K=VALUE_SELECTED_FROM_INTERNAL_CURVE
```

Then run q0-q4 only:

```bash
submit_a100 syc-e2e-q0-4 10:00:00 \
  persona.behavior-end-to-end-circuit \
  "--config $CONFIG --run-name $E2E_FINAL --k $E2E_K \
   --question-indices $Q5 \
   --conditions normal negative_prompt circuit_ablated circuit_only \
   --max-new-tokens $COMMON_MAX_NEW_TOKENS --max-length 1024 \
   --output-tag k${E2E_K}_q0-4_workshop_gate_v1"
```

Select k from held-out internal faithfulness, never from official behavior.
Expand end-to-end behavior to q5-q19 only if the q0-q4 circuit-only responses
are coherent and move toward the normal positive-persona behavior.
Circuit-only behavior is a reconstructive claim only if internal faithfulness
is at least `0.85`, generations are coherent, and topology-matched controls
are weaker.

### 11.4 Blocking failure contingency

The existing downstream ranking attributes response-prediction positions. It
is not automatically the downstream half of a single prompt-end bottleneck.

If vector rescue succeeds but downstream blocking does not:

1. do not conclude that the interface is false;
2. implement one matched prompt-end downstream attribution whose clean and
   corrupt states are the vector-rescued and unrescued upstream-cut prefixes;
3. evaluate that ranking on q0-q4 only;
4. continue to q5-q19 only if it blocks at least 30% of the rescue.

This is the only justified downstream reranking contingency.

## 12. Compute budget and schedule

Planning estimates are allocated one-GPU A100-hours, excluding queue time and
OpenAI judging.

| Phase | Required? | Estimated A100-hours |
| --- | --- | ---: |
| CPU integration/provenance | Yes | 0 |
| Technical gate | Yes | 0.3-0.6 |
| q0-q4 mediation pilot + two internal cells | Yes | 3-6 |
| IG-5/IG-20 guardrail + controls | If pilot passes | 2-4 |
| q5-q19 primary mediation | If pilot passes | 12-24 |
| One triggered diagnostic/comparator branch | Conditional | 3-8 |
| **Conservative campaign total** | | **20-43** |

At 128 SBU per A100-hour, budget approximately `2,560-5,504 SBU`.

Suggested elapsed schedule with parallel A100 access:

| Time | Work |
| --- | --- |
| Day 1 morning | Integrate branches, run tests, freeze manifests, rebuild circuits. |
| Day 1 afternoon | Technical gate. |
| Day 1 night | q0-q4 mediation pilot and two internal evaluations. |
| Day 2 morning | Judge/analyze pilot; stop or approve expansion. |
| Day 2 daytime | IG guardrail and topology controls. |
| Day 2 night | Twelve q5-q19 primary generation jobs in parallel. |
| Day 3 morning | Inspect, judge, bootstrap, freeze figures/tables. |
| Day 3 afternoon/night | Run only the single contingency actually triggered. |

Update estimates with actual `sacct` elapsed times after each wave.

## 13. Outcome-to-paper decision table

| Result | Defensible conclusion |
| --- | --- |
| Cut effect + specific vector rescue + blocking | Sparse upstream cut acts through a specific layer-20 causal interface transmitted by the downstream circuit. |
| Approximately 80% rescue and blocking | Strong evidence for a one-dimensional causal bottleneck. |
| Significant but partial rescue | Specific partial mediator; avoid “complete bottleneck.” |
| Full state works, vector fails | Layer-20 interface exists but is higher-dimensional than the persona vector. |
| Prompt-end fails, recurrent succeeds | Trait is maintained by a recurrent interface during generation. |
| Vector rescue works, old downstream blocking fails | Existing downstream ranking is timing-mismatched; run matched prompt-end attribution. |
| Upstream reconstruction fails, mediation succeeds | Report causal cut plus mediated interface, not a self-contained formation circuit. |
| Decomposition fails, end-to-end succeeds | Reject the proposed bottleneck factorization; publish the direct circuit result. |
| Both decomposition and end-to-end fail | Preserve historical necessity as interception evidence and narrow the workshop paper accordingly. |

## 14. Explicitly skipped workshop work

Do not spend the three-day workshop budget on:

- regenerating vectors or rerunning the layer sweep;
- rerunning frozen refusal experiments;
- a full IG-5/20/50 grid;
- IG-200;
- dense EAP-ranked upstream k sweeps for reconstructive sufficiency;
- a Cartesian upstream/downstream union-size sweep;
- downstream-only and joint q5-q19 reruns;
- multilingual transfer;
- hallucination, evil, or factual-sycophancy expansion;
- a second model;
- the 8,500-row external feedback set before its metric is validated;
- circuit-only behavior from a family with internal faithfulness below `0.85`.

These are ICLR extensions after the English mechanism is frozen.

## 15. Final reporting checklist

Report:

- exact Git commit and clean/dirty state;
- model revision and all frozen artifact hashes;
- discovery, internal-evaluation, pilot, and primary identities;
- requested and retained edge counts;
- circuit family: `causal_cut` or `source_to_sink`;
- intervention position scope and interface timing;
- clean/corrupt definitions for every estimand;
- all token caps, generated-token counts, and cap hits;
- raw condition means and paired 95% confidence intervals;
- all five orthogonal seeds and all five topology controls;
- q0-q4 and q5-q19 separately;
- noising/denoising asymmetry;
- the distinction between interception, mediation, and reconstruction;
- actual A100-hours and SBU from Slurm accounting.

Never claim:

- that necessity establishes sufficiency;
- that a sink-connected shell is a complete circuit;
- that layer 20 is a nontrivial bottleneck without specific rescue;
- that the 20 official questions are untouched validation;
- that the historical downstream ranking is prompt-end-specific unless the
  rescue-blocking experiment or matched attribution establishes it.

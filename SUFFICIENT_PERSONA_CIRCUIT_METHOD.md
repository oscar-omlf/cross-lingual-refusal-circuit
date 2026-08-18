# Sufficient persona circuit method

Status: implementation-complete, real-model GPU validation pending, 2026-08-18.

This document defines the high-value extension in which the sycophancy
mechanism is reconstructed as two sparse circuits joined by an explicit
persona-vector interface:

```text
token-aligned positive/negative persona prompts
    -> sparse source-to-interface circuit U
    -> layer-20 persona-vector coordinate
    -> sparse interface-to-output circuit D
    -> sycophantic free generation
```

This extension does not replace the budget-gated mediation campaign in
`WORKSHOP_MINIMAL_EXPERIMENT_RUNBOOK.md`. It is the correct conditional route
to the stronger reconstructive claim. Do not use the historical EAP-IG
ranking for this claim merely by increasing k.

## 1. Executive conclusion

A defensible route to a circuit that is both necessary and sufficient exists,
but the sufficiency objective must be changed.

The historical pipeline:

1. assigns each edge a marginal attribution score for one scalar at one
   position;
2. selects the largest scores independently;
3. asks those edges to recursively reconstruct a sequence-level nonlinear
   computation.

That ranking is appropriate for finding a causal cut. It is not optimized for
the circuit-only computation used to test sufficiency. Existing evidence is:

- small upstream sets are highly necessary;
- tiny sets are often disconnected;
- the historical k20,000 **sink-connected** set has about `0.193`
  faithfulness;
- a later CPU audit found 19,887 of those 20,000 edges on complete paths, but
  the repaired 19,887-edge source-to-sink subset has not been GPU-evaluated.

First run the cheap repaired source-to-sink evaluation. The new method is
triggered only if that corrected baseline remains insufficient. It starts
from the fully faithful graph and learns one
hard-concrete gate per edge **inside the exact circuit-only algebra**. It
optimizes the source-to-interface and interface-to-output halves separately,
with the layer-20 persona direction as the only permitted clean information
at their boundary.

The central free-generation test is:

```text
negative-persona computation
+ source-to-interface selected clean-minus-negative edge contributions
+ only the recovered coefficient along v at layer 20
+ interface-to-output selected rescued-minus-negative edge contributions
```

All clean-minus-negative layer-20 components orthogonal to `v` are discarded.
If this composition generates sycophantic answers, while blocking `v` or
replacing it with an equal-coefficient orthogonal direction removes the
effect, then the result supports a strict vector-mediated reconstructive
mechanism.

## 2. What “sufficient by itself” can mean

The paper must define this precisely.

A sparse behavioral circuit is not expected to replace the model's general
language competence, factual knowledge, token embeddings, or grammar. The
standard counterfactual circuit-sufficiency estimand asks whether the selected
mechanism is sufficient to recover the target computation **relative to a
coherent corrupt baseline**.

Here:

- the clean computation is the positive-persona prompt;
- the corrupt computation is its token-aligned negative-persona counterpart;
- both contain a complete competent language model;
- the sparse circuit carries the clean-minus-corrupt causal differential.

The strongest justified wording is:

> The selected source-to-interface and interface-to-output edges are jointly
> sufficient to reconstruct the sycophancy-causing differential from a
> coherent low-sycophancy counterfactual, and their effect is mediated
> exclusively through the frozen layer-20 persona-vector coordinate.

Do not write:

> These edges alone implement the entire language model.

Running selected edges against zeros is a different and generally
ill-posed claim because it removes the shared computational background that
is not specific to the trait.

## 3. Claims and required evidence

Keep four claims distinct.

### 3.1 Causal interception

A sparse upstream edge set is necessary: ablating it changes the layer-20
coordinate and behavior.

Historical evidence already supports this. It does not imply reconstruction.

### 3.2 Causal interface

The layer-20 persona-vector coordinate is a specific mediator:

- restoring it rescues behavior;
- equal-magnitude orthogonal directions do not;
- an unrelated persona direction does not;
- downstream blocking removes the rescued effect.

This is the workshop-critical claim in the minimal runbook.

### 3.3 Reconstructive half-circuits

`U` reconstructs the clean persona coordinate from the negative-persona
baseline at every response-prediction position.

`D` reconstructs the logits caused by restoring only that coordinate.

Each half must pass held-out internal faithfulness before free generation.

### 3.4 Strict composed sufficiency

`D(I(U(x)))` recovers sycophantic behavior in free generation, where `I`
projects the recovered upstream effect onto `v` and removes every orthogonal
clean-minus-corrupt component.

Blocking `v` must remove most of the recovered effect.

## 4. Formal estimands

Let:

- `x+` be a positive-persona prompt;
- `x-` be the token-aligned negative-persona prompt;
- `y` be a shared teacher-forced answer suffix;
- `h+_t` and `h-_t` be layer-20 residual states at response-prediction
  position `t`;
- `v` be the frozen unit sycophancy direction;
- `F_U(g; x+, x-)` be the upstream graph evaluated with edge gates `g`;
- `F_D(g; I, h-)` be the downstream graph evaluated from interface state `I`
  against corrupt interface `h-`.

### 4.1 Exact circuit-only receiver algebra

For every receiving node `r`:

```text
r_g = r- + sum_e(parent -> r) g_e * (parent_g - parent-)
```

where:

- `r-` and `parent-` are cached from the corrupt computation;
- `parent_g` is recursively produced by the gated mixed computation;
- `g_e` is a trainable hard-concrete edge gate.

All-open gates must reproduce the clean computation. All-closed gates must
reproduce the corrupt computation. This is a mandatory implementation
identity, not an empirical approximation.

### 4.2 Upstream target

For every response-prediction position:

```text
z+_t = <h+_t, v>
z-_t = <h-_t, v>
zU_t = <F_U(gU; x+, x-)_t, v>
```

The normalized upstream loss is:

```text
L_U = mean_t (zU_t - z+_t)^2
      / max(mean_t (z+_t - z-_t)^2, epsilon)
```

This objective reconstructs exactly the quantity passed through the proposed
interface. It does not require `U` to reconstruct layer-20 components that
the interface later discards.

### 4.3 Strict vector-only oracle interface

Define:

```text
alpha_t = <h+_t - h-_t, v>
I*_t = h-_t + alpha_t v
```

Only response-prediction positions are modified. At all other positions,
`I*_t = h-_t`.

`I*` contains no orthogonal component of `h+ - h-`.

The full downstream model run from `I*` is the oracle ceiling for any
one-dimensional factorization. Test this before training edge masks.

### 4.4 Downstream target

Let:

- `p*(token | prefix)` be the full model distribution from `I*`;
- `p-(token | prefix)` be the distribution from `h-`;
- `pD(token | prefix)` be the downstream gated circuit from `I*` against
  `h-`.

The normalized downstream loss is:

```text
L_D = mean_t KL(p*_t || pD_t)
      / max(mean_t KL(p*_t || p-_t), epsilon)
```

Training shuffles a balanced set containing both the high-trait and low-trait
answer suffix for every prompt pair. Held-out evaluation also measures
recovery of the high-answer versus low-answer response-log-probability margin.
This prevents generic text reconstruction from being mistaken for trait
transmission.

### 4.5 Learned sparsity

Each edge has a hard-concrete gate. Model weights are frozen. Training uses a
scheduled constrained L0 objective:

```text
L = L_task
    + lambda_1 * (expected_sparsity - target_sparsity)
    + lambda_2 * (expected_sparsity - target_sparsity)^2
```

The gate parameters minimize `L`; the Lagrange multipliers maximize it.
Sparsity increases linearly during warmup.

Default targets:

- upstream: `0.95`;
- downstream: `0.99`.

The learned deterministic gate scores define a nested top-k frontier. Every
reported circuit is then pruned to edges on complete source-to-sink paths.

### 4.6 Strict recurrent composition

At each autoregressive step:

1. run positive and negative aligned prefixes;
2. reconstruct the upstream layer-20 state `hU_t` using only `U`;
3. compute:

   ```text
   alphaU_t = <hU_t - h-_t, v>
   IU_t = h-_t + alphaU_t v
   ```

4. discard all orthogonal components of `hU_t - h-_t`;
5. reconstruct logits using only downstream circuit `D` from `IU` against
   `h-`;
6. greedily select the next token;
7. append the same selected token to both aligned prefixes.

This is recurrent mediation. It is not a single prompt-end injection.

## 5. Why this is better matched than larger EAP-IG circuits

Increasing k in the old ranking addresses only one failure mode: missing
support. It does not correct the objective mismatch.

Historical topology:

| Requested k | Complete-path retained edges |
| ---: | ---: |
| 50 | 0 |
| 100 | 0 |
| 250 | 198 |
| 500 | 373 |
| 1,000 | 782 |
| 3,000 | 2,681 |
| 5,000 | 4,719 |
| 10,000 | 9,761 |
| 20,000 | 19,887 |

At k20,000, topology is almost complete, but the `0.193` faithfulness value
belongs to the original 20,000-edge sink-connected circuit. The exact repaired
19,887-edge source-to-sink subset remains unmeasured. Therefore:

- connectivity explains tiny-k failure;
- connectivity may or may not explain the large-k failure;
- repaired source-to-sink evaluation is the first required experiment;
- direct reconstruction-aware optimization is required only if that corrected
  baseline remains weak.

The new method also avoids a second mismatch in the historical downstream
result. The old downstream circuit was ranked under recurrent coefficient
`0 -> -2` steering. The new downstream circuit is trained specifically on the
same natural-negative to vector-only-rescued interface used in the composed
claim.

## 6. Relation to prior work

The implementation combines established ideas but applies them to a new
factorized behavioral mechanism.

- ACDC (`arXiv:2304.14997`) motivates automated causal graph pruning.
- Edge Attribution Patching (`arXiv:2310.10348`) and EAP-IG
  (`arXiv:2403.17806`) motivate scalable edge attribution, but marginal
  rankings do not directly optimize recursive sufficiency.
- Edge Pruning by Bhaskar et al. (`arXiv:2406.16778`, NeurIPS 2024 Spotlight)
  learns hard-concrete edge masks from a fully faithful graph. The new code
  adopts this optimization principle while retaining this repository's
  TransformerLens/EAP graph and natural-prompt counterfactual.
- Work on faithfulness robustness and circuit evaluation motivates separating
  necessity, sufficiency, topology, and behavioral validation rather than
  reporting one patching score as a complete mechanism.
- Persona Vectors (`arXiv:2507.21509`) motivates the frozen behavioral
  direction and layer intervention. This project adds edge-level formation,
  transmission, strict mediation, and composed circuit sufficiency.

The plausible paper contribution is not simply “we ran Edge Pruning on
sycophancy.” It is:

1. a factorized source-interface-output circuit definition for behavioral
   traits;
2. a strict interface that permits only the persona-vector coordinate;
3. separately sufficient formation and transmission circuits;
4. recurrent free-generation composition;
5. causal blocking and orthogonal controls that test whether the two circuits
   are genuinely mediated by the proposed interface.

## 7. Implementation

New or modified components:

| File | Purpose |
| --- | --- |
| `src/refusal_circuit/edge_pruning.py` | Hard-concrete gates and vectorized exact circuit-only receiver construction. |
| `src/refusal_circuit/circuit.py` | Full target-state reconstruction and explicit clean/corrupt source transformations. |
| `scripts/persona/persona_prune_sufficient_circuit.py` | Upstream/downstream mask training, identity preflight, scores, checkpoints, and complete-path frontier export. |
| `scripts/persona/persona_evaluate_sufficient_circuit.py` | Held-out per-position upstream and contrastive/KL downstream evaluation. |
| `scripts/persona/persona_behavior_sufficient_composition.py` | Oracle and strict recurrent composed generation. |
| `scripts/persona/persona_analyze_sufficient_composition.py` | Provenance-gated paired behavioral analysis and bootstrap intervals. |
| `scripts/persona/persona_build_circuit_controls.py` | Topology-stratified controls for learned gate scores. |
| `tests/test_edge_pruning.py` | Gate, vector-interface, open/closed, gradient, and literal-algebra tests. |

The vectorized edge construction preserves the attention output bias by
distributing it equally across per-head residual writers, matching the
existing evaluator's exact bookkeeping convention.

The downstream implementation uses explicit source-cache transformations.
It does not rely on TransformerLens hook registration order to decide whether
the graph source is the pre-replacement or post-replacement interface state.

## 8. Mandatory gates

Do not skip gates because a later job is already queued.

### Gate 0: CPU and unit tests

Required:

- all files compile;
- all tests pass;
- all CLI help commands parse;
- no dirty integration ambiguity.

### Gate 1: vector oracle

Before mask training:

- full-state oracle is token-identical to normal generation;
- positive behavior exceeds negative behavior;
- vector-only oracle rescues the trait;
- coherence remains at least `80`;
- no token-cap pathology.

For a strong one-dimensional claim, the q0-q4 vector oracle should recover at
least `0.80` of the positive-minus-negative behavioral gap. A smaller but
clear recovery supports a partial one-dimensional mediator, not a complete
bottleneck.

### Gate 2: algebra identity

For both halves on a real Qwen transcript:

- all-open normalized error `<= 1e-3`;
- all-closed normalized error `<= 1e-3`;
- no OOM;
- no non-finite values.

### Gate 3: held-out half-circuit faithfulness

Select the smallest complete-path k satisfying:

Upstream:

- per-position coordinate reconstruction `>= 0.85`;
- preferably `>= 0.90` for the final paper.

Downstream:

- mean distribution KL recovery `>= 0.85`;
- contrastive margin faithfulness `>= 0.85`;
- vector-only oracle produces a nontrivial response-margin effect.

k is selected only from the frozen held-out internal metrics. Do not inspect
official behavior to choose k.

### Gate 4: controls

At selected k:

- five topology-stratified score-permutation controls per half;
- five orthogonal-vector seeds in behavior;
- `v`-blocked composition;
- exact normal/full-state identity.

Topology controls should be materially weaker than the learned circuits.
If a topology control passes the same `0.85` internal threshold, run its
q0-q4 behavioral composition rather than suppressing the result.

### Gate 5: composed behavior

Pilot q0-q4:

- composed circuit is coherent and terminates;
- composed recovery is at least `0.50` of the natural gap to expand;
- composed recovery is preferably `>= 0.80` for the strong claim;
- `v` blocking removes at least `0.50` of composed recovery to expand;
- preferably `>= 0.80` for the strong claim;
- composed exceeds the mean orthogonal control;
- no result depends on cap hits.

Primary q5-q19 requires paired confidence intervals above zero for:

- composed minus negative;
- composed minus blocked;
- composed minus orthogonal.

## 9. Exact experiment protocol

Run from `/home/omiro1/cross-lingual-refusal-circuit` in Bash after integrating
and pushing the implementation commit.

### 9.1 Environment

```bash
test -n "${BASH_VERSION:-}" || exit 1

export CONFIG=configs/qwen_sycophancy.json
export RESULTS=results/qwen2.5-7b-instruct/persona/sycophantic
export INTERMEDIATE=data/intermediate/persona/qwen2.5-7b-instruct/sycophantic
export SHARED="$INTERMEDIATE/shared_circuit_dataset_v1.json"
export SUFF_EVAL8="$INTERMEDIATE/downstream_prompt_response_disjoint_eval_suff8_v1.json"
export SUFF_EVAL32="$INTERMEDIATE/downstream_prompt_response_disjoint_eval_workshop_v1.json"

export SUFF_UP_RUN=syc_up_edgeprune_n48_s95_steps600_seed0_v1
export SUFF_DN_RUN=syc_dn_edgeprune_n48_s99_steps600_seed0_v1
export SUFF_UP_ROOT="$RESULTS/circuit_discovery_edge_pruning/upstream/$SUFF_UP_RUN"
export SUFF_DN_ROOT="$RESULTS/circuit_discovery_edge_pruning/downstream/$SUFF_DN_RUN"
export SUFF_BEHAVIOR="$INTERMEDIATE/sufficient_circuit_behavior"

export Q0="0"
export Q5="0 1 2 3 4"
export Q15="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

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

### 9.2 Integration and frozen manifests

```bash
git status --short
git log -3 --oneline
uv sync --locked
PYTHONPYCACHEPREFIX=/tmp/persona-pycache \
  uv run python -m compileall -q src scripts tests EAP-IG/eap
uv run pytest -q
git diff --check

uv run python scripts/rq.py persona.freeze-downstream-evaluation \
  --config "$CONFIG" \
  --discovery-manifest "$SHARED" \
  --num-pairs 8 \
  --seed 20260811 \
  --max-length 768 \
  --output "$SUFF_EVAL8"

test -f "$SUFF_EVAL32"
jq -e '.status == "frozen" and (.selection.samples | length == 32)' \
  "$SUFF_EVAL32"
```

Do not recreate `SUFF_EVAL32` if the existing frozen hash is valid.

### 9.3 Gate 1A: 16-token exact interface smoke

No trained circuits are required.

```bash
for CONDITION in normal_positive oracle_full_state_interface; do
  submit_a100 "syc-suff-id-${CONDITION:0:8}" 01:00:00 \
    persona.behavior-sufficient-composition \
    "--config $CONFIG \
     --condition $CONDITION --question-indices $Q0 \
     --max-new-tokens 16 --max-length 1024 \
     --output-tag ${CONDITION}_q0_t16_identity_v1"
done
```

After completion:

```bash
export ORACLE_TECH="$SUFF_BEHAVIOR/none__none/kunone_kdnone"
test "$(
  jq -r '.generated_token_ids_sha256' \
    "$ORACLE_TECH/normal_positive_q0_t16_identity_v1/responses.jsonl"
)" = "$(
  jq -r '.generated_token_ids_sha256' \
    "$ORACLE_TECH/oracle_full_state_interface_q0_t16_identity_v1/responses.jsonl"
)"
```

Stop if token IDs differ.

### 9.4 Gate 1B: q0-q4 vector-oracle feasibility

```bash
for CONDITION in \
  normal_positive \
  negative_baseline \
  oracle_vector_interface
do
  submit_a100 "syc-suff-or-${CONDITION:0:8}" 08:00:00 \
    persona.behavior-sufficient-composition \
    "--config $CONFIG \
     --condition $CONDITION --question-indices $Q5 \
     --max-new-tokens 512 --max-length 1024 \
     --output-tag ${CONDITION}_q0-4_oracle_gate_v1"
done
```

Judge:

```bash
for CONDITION in \
  normal_positive \
  negative_baseline \
  oracle_vector_interface
do
  DIR="$ORACLE_TECH/${CONDITION}_q0-4_oracle_gate_v1"
  uv run python scripts/rq.py persona.judge \
    --config "$CONFIG" \
    --input "$DIR/responses.jsonl" \
    --output "$DIR/scored.jsonl" \
    --artifact eval \
    --protocol-field behavior_protocol_id \
    --resume
done
```

Calculate:

```text
oracle recovery =
  (mean oracle-vector trait - mean negative trait)
  / (mean normal-positive trait - mean negative trait)
```

Stop one-dimensional circuit training if the numerator is non-positive.

If recovery is positive but below `0.50`, prioritize a low-rank interface
fallback rather than edge-mask training for a one-dimensional claim.

### 9.5 Gate 2: real-model algebra identity

```bash
submit_a100 syc-suff-up-preflight 01:00:00 \
  persona.prune-sufficient-circuit \
  "--config $CONFIG --stage upstream \
   --run-name $SUFF_UP_RUN --sample-manifest $SHARED \
   --num-prompts 64 --train-prompts 48 --max-length 768 \
   --preflight-only --preflight-tolerance 1e-3"

submit_a100 syc-suff-dn-preflight 01:00:00 \
  persona.prune-sufficient-circuit \
  "--config $CONFIG --stage downstream \
   --run-name $SUFF_DN_RUN --sample-manifest $SHARED \
   --num-prompts 64 --train-prompts 48 --max-length 768 \
   --preflight-only --preflight-tolerance 1e-3"
```

Both printed JSON objects must have:

```json
{
  "status": "pass",
  "all_open_normalized_error": 0.001,
  "all_closed_normalized_error": 0.001
}
```

The numeric values must be less than or equal to `0.001`; the example above
shows the threshold, not expected exact output.

### 9.6 Train the two masks

Submit in parallel after both preflights pass.

```bash
submit_a100 syc-suff-up-train 12:00:00 \
  persona.prune-sufficient-circuit \
  "--config $CONFIG --stage upstream \
   --run-name $SUFF_UP_RUN --sample-manifest $SHARED \
   --num-prompts 64 --train-prompts 48 \
   --steps 600 --seed 0 --max-length 768 \
   --target-sparsity 0.95 --sparsity-warmup-steps 500 \
   --edge-learning-rate 0.1 \
   --regularization-learning-rate 0.1 \
   --initial-log-alpha 5.0 --gradient-clip 1.0 \
   --k-values 500 1000 3000 5000 10000 20000 50000"

submit_a100 syc-suff-dn-train 12:00:00 \
  persona.prune-sufficient-circuit \
  "--config $CONFIG --stage downstream \
   --run-name $SUFF_DN_RUN --sample-manifest $SHARED \
   --num-prompts 64 --train-prompts 48 \
   --steps 600 --seed 0 --max-length 768 \
   --target-sparsity 0.99 --sparsity-warmup-steps 500 \
   --edge-learning-rate 0.1 \
   --regularization-learning-rate 0.1 \
   --initial-log-alpha 5.0 --gradient-clip 1.0 \
   --k-values 100 300 500 1000 3000 5000"
```

Inspect:

```bash
jq '{
  status,
  stage,
  final: .final,
  optimization: .optimization,
  outputs: .outputs
}' "$SUFF_UP_ROOT/metadata.json"

jq '{
  status,
  stage,
  final: .final,
  optimization: .optimization,
  outputs: .outputs
}' "$SUFF_DN_ROOT/metadata.json"
```

Stop if:

- status is not complete;
- training contains non-finite values;
- expected sparsity never approaches the scheduled target;
- task loss diverges;
- any saved hash fails.

Do not choose k from final training loss.

### 9.7 Held-out frontier gate

Evaluate from small to large k on the frozen 8-pair manifest. Stop at the
first passing k for each half.

Upstream sequence:

```bash
for K in 500 1000 3000 5000 10000 20000 50000; do
  submit_a100 "syc-suff-u8-$K" 03:00:00 \
    persona.evaluate-sufficient-circuit \
    "--config $CONFIG --stage upstream \
     --run-name $SUFF_UP_RUN --k $K \
     --evaluation-manifest $SUFF_EVAL8 \
     --num-prompts 8 --max-length 768 \
     --output-tag k${K}_n8_frontier_v1"
done
```

Downstream sequence:

```bash
for K in 100 300 500 1000 3000 5000; do
  submit_a100 "syc-suff-d8-$K" 03:00:00 \
    persona.evaluate-sufficient-circuit \
    "--config $CONFIG --stage downstream \
     --run-name $SUFF_DN_RUN --k $K \
     --evaluation-manifest $SUFF_EVAL8 \
     --num-prompts 8 --max-length 768 \
     --output-tag k${K}_n8_frontier_v1"
done
```

These loops show all possible jobs. To minimize compute, submit one k at a
time and stop when the threshold passes.

Select the values manually from the frozen n8 metrics, then export them:

```bash
export KU
export KD
test -n "$KU" && test -n "$KD"
```

Confirm selected k on all 32 held-out pairs:

```bash
submit_a100 syc-suff-up-n32 06:00:00 \
  persona.evaluate-sufficient-circuit \
  "--config $CONFIG --stage upstream \
   --run-name $SUFF_UP_RUN --k $KU \
   --evaluation-manifest $SUFF_EVAL32 \
   --num-prompts 32 --max-length 768 \
   --output-tag selected_k${KU}_n32_v1"

submit_a100 syc-suff-dn-n32 08:00:00 \
  persona.evaluate-sufficient-circuit \
  "--config $CONFIG --stage downstream \
   --run-name $SUFF_DN_RUN --k $KD \
   --evaluation-manifest $SUFF_EVAL32 \
   --num-prompts 32 --max-length 768 \
   --output-tag selected_k${KD}_n32_v1"
```

Set:

```bash
export UP_INT="$SUFF_UP_ROOT/evaluation/selected_k${KU}_n32_v1/aggregate.json"
export DN_INT="$SUFF_DN_ROOT/evaluation/selected_k${KD}_n32_v1/aggregate.json"

jq '.metrics' "$UP_INT"
jq '.metrics' "$DN_INT"
```

### 9.8 Topology controls

Build controls on CPU:

```bash
export CONTROL_NAME=topology_stratified_sufficiency_v1

uv run python scripts/rq.py persona.build-circuit-controls \
  --config "$CONFIG" \
  --scope edge_pruning_upstream \
  --run-name "$SUFF_UP_RUN" \
  --control-name "$CONTROL_NAME" \
  --family source_to_sink \
  --k-values "$KU" \
  --seeds 0 1 2 3 4

uv run python scripts/rq.py persona.build-circuit-controls \
  --config "$CONFIG" \
  --scope edge_pruning_downstream \
  --run-name "$SUFF_DN_RUN" \
  --control-name "$CONTROL_NAME" \
  --family source_to_sink \
  --k-values "$KD" \
  --seeds 0 1 2 3 4
```

Evaluate all controls on the 32-pair manifest:

```bash
for SEED in 0 1 2 3 4; do
  UP_CONTROL="$SUFF_UP_ROOT/controls/$CONTROL_NAME/source_to_sink/seed_${SEED}/edge_pruned_top_${KU}.json"
  DN_CONTROL="$SUFF_DN_ROOT/controls/$CONTROL_NAME/source_to_sink/seed_${SEED}/edge_pruned_top_${KD}.json"

  submit_a100 "syc-suff-uc-$SEED" 06:00:00 \
    persona.evaluate-sufficient-circuit \
    "--config $CONFIG --stage upstream \
     --run-name $SUFF_UP_RUN --k $KU \
     --circuit-path $UP_CONTROL \
     --evaluation-manifest $SUFF_EVAL32 \
     --num-prompts 32 --max-length 768 \
     --output-tag control_seed${SEED}_k${KU}_n32_v1"

  submit_a100 "syc-suff-dc-$SEED" 08:00:00 \
    persona.evaluate-sufficient-circuit \
    "--config $CONFIG --stage downstream \
     --run-name $SUFF_DN_RUN --k $KD \
     --circuit-path $DN_CONTROL \
     --evaluation-manifest $SUFF_EVAL32 \
     --num-prompts 32 --max-length 768 \
     --output-tag control_seed${SEED}_k${KD}_n32_v1"
done
```

### 9.9 q0 technical composition smoke

Run after both selected circuits pass n32.

```bash
export SUFF_RUN_ROOT="$SUFF_BEHAVIOR/${SUFF_UP_RUN}__${SUFF_DN_RUN}/ku${KU}_kd${KD}"

for CONDITION in \
  normal_positive \
  negative_baseline \
  oracle_vector_interface \
  upstream_circuit_full_downstream \
  oracle_vector_downstream_circuit \
  composed_circuit \
  composed_v_blocked
do
  submit_a100 "syc-suff-q0-${CONDITION:0:7}" 03:00:00 \
    persona.behavior-sufficient-composition \
    "--config $CONFIG \
     --upstream-run $SUFF_UP_RUN --downstream-run $SUFF_DN_RUN \
     --upstream-k $KU --downstream-k $KD \
     --upstream-evaluation $UP_INT \
     --downstream-evaluation $DN_INT \
     --condition $CONDITION --question-indices $Q0 \
     --max-new-tokens 64 --max-length 1024 \
     --output-tag ${CONDITION}_q0_t64_smoke_v1"
done
```

Inspect every response. Stop for incoherence, nontermination, repeated-token
loops, non-finite logits, or missing interface diagnostics.

### 9.10 q0-q4 composed pilot

```bash
for CONDITION in \
  normal_positive \
  negative_baseline \
  oracle_vector_interface \
  upstream_circuit_full_downstream \
  oracle_vector_downstream_circuit \
  composed_circuit \
  composed_v_blocked
do
  submit_a100 "syc-suff-p-${CONDITION:0:7}" 12:00:00 \
    persona.behavior-sufficient-composition \
    "--config $CONFIG \
     --upstream-run $SUFF_UP_RUN --downstream-run $SUFF_DN_RUN \
     --upstream-k $KU --downstream-k $KD \
     --upstream-evaluation $UP_INT \
     --downstream-evaluation $DN_INT \
     --condition $CONDITION --question-indices $Q5 \
     --max-new-tokens 512 --max-length 1024 \
     --output-tag ${CONDITION}_q0-4_pilot_v1"
done

for SEED in 0 1 2 3 4; do
  submit_a100 "syc-suff-po-$SEED" 12:00:00 \
    persona.behavior-sufficient-composition \
    "--config $CONFIG \
     --upstream-run $SUFF_UP_RUN --downstream-run $SUFF_DN_RUN \
     --upstream-k $KU --downstream-k $KD \
     --upstream-evaluation $UP_INT \
     --downstream-evaluation $DN_INT \
     --condition composed_orthogonal --orthogonal-seed $SEED \
     --question-indices $Q5 \
     --max-new-tokens 512 --max-length 1024 \
     --output-tag composed_orthogonal_seed${SEED}_q0-4_pilot_v1"
done
```

Judge every directory:

```bash
find "$SUFF_RUN_ROOT" -mindepth 1 -maxdepth 1 \
  -name '*_q0-4_pilot_v1' -type d -print0 |
while IFS= read -r -d '' DIR; do
  uv run python scripts/rq.py persona.judge \
    --config "$CONFIG" \
    --input "$DIR/responses.jsonl" \
    --output "$DIR/scored.jsonl" \
    --artifact eval \
    --protocol-field behavior_protocol_id \
    --resume
done
```

Analyze:

```bash
uv run python scripts/rq.py persona.analyze-sufficient-composition \
  --config "$CONFIG" \
  --normal-scored "$SUFF_RUN_ROOT/normal_positive_q0-4_pilot_v1/scored.jsonl" \
  --negative-scored "$SUFF_RUN_ROOT/negative_baseline_q0-4_pilot_v1/scored.jsonl" \
  --oracle-vector-scored "$SUFF_RUN_ROOT/oracle_vector_interface_q0-4_pilot_v1/scored.jsonl" \
  --composed-scored "$SUFF_RUN_ROOT/composed_circuit_q0-4_pilot_v1/scored.jsonl" \
  --blocked-scored "$SUFF_RUN_ROOT/composed_v_blocked_q0-4_pilot_v1/scored.jsonl" \
  --upstream-only-scored "$SUFF_RUN_ROOT/upstream_circuit_full_downstream_q0-4_pilot_v1/scored.jsonl" \
  --downstream-only-scored "$SUFF_RUN_ROOT/oracle_vector_downstream_circuit_q0-4_pilot_v1/scored.jsonl" \
  --orthogonal-scored \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed0_q0-4_pilot_v1/scored.jsonl" \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed1_q0-4_pilot_v1/scored.jsonl" \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed2_q0-4_pilot_v1/scored.jsonl" \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed3_q0-4_pilot_v1/scored.jsonl" \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed4_q0-4_pilot_v1/scored.jsonl" \
  --bootstrap-samples 10000 \
  --output "$SUFF_RUN_ROOT/q0-4_pilot_analysis_v1.json"
```

### 9.11 q5-q19 primary confirmation

Run only if the pilot gate passes. The half-circuit conditions are not needed
again unless the pilot is ambiguous.

```bash
for CONDITION in \
  normal_positive \
  negative_baseline \
  oracle_vector_interface \
  composed_circuit \
  composed_v_blocked
do
  submit_a100 "syc-suff-c-${CONDITION:0:7}" 12:00:00 \
    persona.behavior-sufficient-composition \
    "--config $CONFIG \
     --upstream-run $SUFF_UP_RUN --downstream-run $SUFF_DN_RUN \
     --upstream-k $KU --downstream-k $KD \
     --upstream-evaluation $UP_INT \
     --downstream-evaluation $DN_INT \
     --condition $CONDITION --question-indices $Q15 \
     --max-new-tokens 512 --max-length 1024 \
     --output-tag ${CONDITION}_q5-19_primary_v1"
done

for SEED in 0 1 2 3 4; do
  submit_a100 "syc-suff-co-$SEED" 12:00:00 \
    persona.behavior-sufficient-composition \
    "--config $CONFIG \
     --upstream-run $SUFF_UP_RUN --downstream-run $SUFF_DN_RUN \
     --upstream-k $KU --downstream-k $KD \
     --upstream-evaluation $UP_INT \
     --downstream-evaluation $DN_INT \
     --condition composed_orthogonal --orthogonal-seed $SEED \
     --question-indices $Q15 \
     --max-new-tokens 512 --max-length 1024 \
     --output-tag composed_orthogonal_seed${SEED}_q5-19_primary_v1"
done
```

Judge with the same command pattern, then analyze:

```bash
uv run python scripts/rq.py persona.analyze-sufficient-composition \
  --config "$CONFIG" \
  --normal-scored "$SUFF_RUN_ROOT/normal_positive_q5-19_primary_v1/scored.jsonl" \
  --negative-scored "$SUFF_RUN_ROOT/negative_baseline_q5-19_primary_v1/scored.jsonl" \
  --oracle-vector-scored "$SUFF_RUN_ROOT/oracle_vector_interface_q5-19_primary_v1/scored.jsonl" \
  --composed-scored "$SUFF_RUN_ROOT/composed_circuit_q5-19_primary_v1/scored.jsonl" \
  --blocked-scored "$SUFF_RUN_ROOT/composed_v_blocked_q5-19_primary_v1/scored.jsonl" \
  --orthogonal-scored \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed0_q5-19_primary_v1/scored.jsonl" \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed1_q5-19_primary_v1/scored.jsonl" \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed2_q5-19_primary_v1/scored.jsonl" \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed3_q5-19_primary_v1/scored.jsonl" \
    "$SUFF_RUN_ROOT/composed_orthogonal_seed4_q5-19_primary_v1/scored.jsonl" \
  --bootstrap-samples 10000 \
  --output "$SUFF_RUN_ROOT/q5-19_primary_analysis_v1.json"
```

## 10. Compute estimate on one A100

These are planning ranges. Replace them with `sacct` measurements after each
gate.

| Work | Estimated A100-hours |
| --- | ---: |
| Oracle identity and q0-q4 feasibility | 1-4 |
| Two real-model preflights | 0.4-0.8 |
| Upstream 600-step training | 3.5-5 |
| Downstream 600-step training | 8-12 |
| Sequential n8 frontier and selected n32 cells | 4-8 |
| Five topology controls per half | 6-14 |
| q0 64-token composition smoke | 0.5-1.5 |
| q0-q4 full pilot including five orthogonal seeds | 20-40 |
| q5-q19 primary, ten generation conditions | 45-90 |

Core cost to determine whether sufficient half-circuits exist:

```text
approximately 16-30 A100-hours
```

Cost through the q0-q4 behavioral claim:

```text
approximately 37-72 A100-hours
```

Full q5-q19 confirmation is conditional and parallelizable. Wall-clock time
can remain under one overnight wave when enough A100s are available, but
allocated A100-hours do not decrease.

## 11. Contingencies

Run only the branch triggered by observed evidence.

### 11.1 Vector oracle fails

If full-state identity passes but vector-only oracle has no positive rescue:

- do not train one-dimensional masks;
- do not claim the persona direction is the complete interface;
- construct a PCA basis from paired layer-20 positive-minus-negative response
  states;
- test dimensions `2, 4, 8, 16` with the same strict oracle;
- choose dimension from held-out internal reconstruction, not behavior;
- replace scalar projection with the selected orthonormal subspace;
- preserve the same source-subspace-output factorization and blocking test.

If no low-rank oracle rescues behavior, the layer-20 bottleneck hypothesis is
false for this counterfactual.

### 11.2 Upstream mask fails

If no upstream k reaches `0.85`:

1. verify all-open and all-closed identities again;
2. inspect whether the learned expected sparsity reached `0.95`;
3. run exactly one denser rerun:

   ```text
   target sparsity 0.90, 800 steps, seed 0
   ```

4. keep the same frozen train/evaluation identities;
5. stop if the denser rerun still fails.

Do not return to a broad EAP k sweep.

### 11.3 Downstream mask fails

Run exactly one denser rerun:

```text
target sparsity 0.98, 800 steps, seed 0
```

If KL recovers but the contrastive margin does not, the circuit reconstructs
generic distributional details rather than the trait transmission mechanism.
Do not call it sufficient.

### 11.4 Half-circuits pass but composition fails

Check:

- free-generation states lie outside the teacher-forced training
  distribution;
- upstream reconstructed coordinate drift by decoding step;
- downstream oracle condition remains faithful on generated prefixes;
- circuit-only behavior is not dominated by one early token.

The minimal corrective experiment is to add generated-prefix trajectories
from the q0-q4 pilot to mask training and evaluate on q5-q19. Do not tune on
q5-q19.

### 11.5 Composition works but blocking fails

This indicates a bypass or a control implementation error because strict
composition is intended to permit only `v`.

Audit:

- the source cache at `blocks.20.hook_resid_pre`;
- whether clean hooks execute before or after source caching;
- whether `composed_v_blocked` uses identical clean and corrupt interfaces;
- whether any upstream clean state enters the downstream graph outside the
  fixed interface.

Do not weaken the blocking criterion.

### 11.6 Seed stability

After seed 0 passes the q0-q4 pilot:

- repeat upstream and downstream training with seed 1;
- evaluate the preselected k values on the same n32 manifest;
- report functional metric stability;
- report edge overlap descriptively;
- do not require high raw overlap when functional recovery is stable.

This is required for the final ICLR version but can follow the workshop
submission if time is exhausted.

## 12. Outcome-to-claim table

| Outcome | Defensible claim |
| --- | --- |
| Oracle >=0.8, both halves >=0.85, composition >=0.8, blocking >=0.8 | Strong evidence for a sparse, one-dimensional, recurrent source-interface-output mechanism. |
| Oracle and composition are positive but partial | A specific partial vector-mediated mechanism; avoid “complete bottleneck.” |
| Full state succeeds, vector fails, low-rank subspace succeeds | Sparse circuits are mediated by a low-dimensional layer-20 interface, not one vector. |
| Upstream passes, downstream fails | Formation circuit found; transmission is more distributed or objective-mismatched. |
| Downstream passes, upstream fails | Vector transmission circuit found; natural formation is distributed. |
| Both halves pass, composition fails | Teacher-forced reconstructive circuits do not compose out of distribution during free generation. |
| Direct end-to-end circuit succeeds while factorization fails | Reject the proposed persona-vector bottleneck; report a distributed direct circuit. |
| Topology controls match learned circuits | The optimization has not isolated a specific mechanism; no sparse-mechanism claim. |

## 13. Paper framing

If the strong result succeeds, the contribution can be framed as:

> We identify a sparse recurrent mechanism for opinion sycophancy that
> factorizes at a low-dimensional layer-20 interface. A learned upstream
> circuit reconstructs the persona-vector coordinate from natural prompt
> context; a learned downstream circuit reconstructs its effect on answer
> distributions; and their strict vector-only composition recovers the trait
> in free generation. Blocking or rotating the interface removes the effect.

The strongest figure is a four-panel causal decomposition:

1. natural positive versus negative behavior;
2. vector-only oracle recovery;
3. held-out U and D internal faithfulness versus retained edges;
4. composed recovery, `v`-blocked control, orthogonal controls, and topology
   controls.

Report both requested and retained edge counts. Report all intervention
timings and clean/corrupt definitions. Treat the 20 official questions as
development data and label q0-q4 pilot versus q5-q19 primary new-condition
analysis accurately.

## 14. Current validation status

Completed:

- vectorized edge algebra implemented;
- exact literal-edge equivalence unit test;
- all-closed corrupt recovery unit test;
- gate gradients tested;
- vector-only orthogonal removal tested;
- five tests passed in the pinned HPC PyTorch environment;
- all new CLI commands parse locally;
- Python compilation and `git diff --check` pass.

Pending after HPC branch integration:

- full repository test suite;
- real-Qwen all-open/all-closed preflights;
- vector-oracle q0/q0-q4 behavior;
- mask training;
- held-out circuit selection;
- behavioral composition.

No scientific GPU result from this new method exists yet. Do not describe the
method as empirically successful until the gates above pass.

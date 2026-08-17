# Persona circuit methodology repair

Status: code-complete methodology repair, pending HPC validation, as of
2026-08-11.

This document supersedes the persona-circuit interpretation in
`RESEARCH_HANDOFF.md` where the two disagree. It does not invalidate the frozen
refusal results.

For the exact executable schedule, including smoke tests, Slurm commands,
artifact paths, deterministic split pooling, decision gates, and A100-hour
estimates, use `SYCOPHANCY_PERSONA_CIRCUIT_REPAIRED_RUNBOOK.md`.

## 1. Executive decision

The formation-interface-expression decomposition should remain the primary
scientific hypothesis. It is more explanatory than an end-to-end circuit alone.
The end-to-end circuit is a required comparator and falsification control, not
automatically the headline.

The current upstream result does not establish a small faithful circuit. The
saved persona selectors used backward greedy search with parentless pruning
disabled. They therefore retained sink-connected edge shells that often had no
selected path from the graph input. Such a shell can be highly necessary when
ablated because it cuts a terminal conduit, while being unable to reconstruct
the clean computation in circuit-only mode. The correct name for this object is
a sparse causal cut or interception set.

The repaired study distinguishes:

1. `causal_cut`: sink-connected greedy edges, allowed for necessity and
   cut-to-interface mediation claims.
2. `source_to_sink`: edges on complete input-to-target paths, required for
   circuit-only faithfulness and sufficiency.
3. `end_to_end`: a separately ranked natural-prompt circuit from block-0
   residual state to response logits.

These support three nested claim levels:

1. **Sparse causal interception:** a small cut set is behaviorally necessary.
2. **Causal bottleneck/interface:** the cut set's behavioral effect is rescued
   by restoring the target interface coordinate, not by equal-norm orthogonal
   controls, and the rescue is blocked by the downstream edge set.
3. **Reconstructive formation-expression circuit:** complete upstream and
   downstream paths are individually faithful and their circuit-only union
   reproduces the behavior.

A small causal bottleneck can be supported at level 2 even if the complete
formation circuit is larger. The small object is then a causal interception set
converging on a low-dimensional interface, not a sufficient upstream
computational subgraph. Reserve "small formation circuit" and "complete sparse
mechanism" for level 3.

## 2. Why necessity was high and faithfulness was low

Hanna et al. define a circuit as a subgraph connecting inputs to logits and
recursively prune parentless and childless nodes. Their reason is exact: a
parentless selected node receives only corrupted inputs and is equivalent to not
being in the reconstructive circuit.

The persona scripts instead called:

```python
paper_greedy_edges(..., prune_parentless=False)
```

This creates the observed pattern:

- A terminal or near-terminal selected edge can have a large ablation effect.
- The same edge cannot carry clean information in circuit-only mode when its
  source node was already constructed from corrupt inputs.
- Adding more independently ranked shell edges improves faithfulness slowly
  because complete source-to-target paths are acquired only accidentally.
- The result is not evidence that EAP patching algebra is broken. Existing tests
  show the sparse circuit-only construction is algebraically equivalent to
  dense edge patching for the selected graph.

The old sycophancy result should therefore be reported as:

- a 50-edge causal cut strongly intercepts sycophantic behavior;
- the corresponding cut is not a sufficient computational mechanism;
- complete formation pathways are larger and must be evaluated separately.

The cut is also the right intervention for a cheap mediation test: ablate it,
measure the resulting interface deficit, and restore exactly that missing
coordinate. A successful specific rescue would show that the cut acts through
the proposed bottleneck even before a sufficient formation circuit is found.

## 3. Value of the persona-vector decomposition

The decomposition has top-conference value only if it is tested as mediation,
not assumed from vector localization.

Required causal chain:

1. The natural positive-versus-negative prompt changes the interface coordinate.
2. The interface coordinate causally changes held-out behavior under steering.
3. Ablating the upstream causal cut reduces the coordinate and the behavior.
4. Restoring only the prompt-specific missing persona-vector coordinate rescues
   behavior.
5. Restoring the entire current-prefix state at the interface exactly reproduces
   a same-backend normal run; this is the implementation positive control.
6. Restoring only the final-prompt full residual state tests whether one state,
   rather than the entire prefix, is a sufficient interface.
7. Equal-norm random-orthogonal and target-orthogonalized unrelated-trait
   vectors are negative controls.
8. Ablating the downstream causal cut blocks vector rescue.
9. Complete upstream and downstream circuits separately test reconstructive
   sufficiency; their union tests the stronger complete-mechanism claim.
10. A separately discovered end-to-end circuit tests whether the decomposition
   loses behaviorally relevant computation.

Behavioral and internal cross-faithfulness are primary in this comparison. Raw
edge overlap is not directly invariant because inserting an explicit interface
refactors edges entering and leaving that layer into pseudo-terminal and
pseudo-input edges. Any structural overlap analysis must exclude or explicitly
map those interface-boundary edges and should remain secondary.

Interpretation:

- Successful vector rescue supports a low-dimensional causal mediator.
- Full-prefix rescue is an implementation check: because it restores the entire
  architectural cut, it is expected to reproduce normal greedy generation
  exactly and does not by itself establish a scientifically unique bottleneck.
- Full-prefix rescue with failed final-prompt full-state rescue means the
  intervention is valid, but there is no single final-prompt-state bottleneck at
  the proposed interface.
- Final-prompt full-state rescue with failed vector rescue supports a
  single-state interface, but not a one-dimensional persona-vector mediator.
- Failed rescue with successful end-to-end reconstruction means the selected
  persona vector is a useful intervention direction but not the natural
  mediator.
- Failed connected reconstruction means the old high-necessity result remains a
  mediated cut-set result if rescue succeeds, not a sufficient formation
  circuit.

The repaired runner now separates two timing estimands:

1. `--upstream-position-scope prompt_only --interface-timing prompt_end`
   corrupts upstream edges only over the original prompt and restores the fixed
   final-prompt-token interface. Success supports the strongest
   **prompt-formed causal bottleneck** claim.
2. `--upstream-position-scope all_positions --interface-timing current_token`
   corrupts and restores the current next-token computation on every decoding
   step. Success supports a **recurrent layerwise interface**, not a one-time
   prompt bottleneck.

Run the prompt-only/prompt-end version first because it matches upstream
discovery's final-prompt target. Run the recurrent version as a timing
diagnostic. If only the recurrent version succeeds, report that distinction
directly rather than calling the interface a prompt-end bottleneck.

The downstream-blocking condition has a separate and important estimand. For
`vector_rescue_downstream_ablated`, the clean downstream computation is the
upstream-cut run with vector rescue, while the corrupt downstream cache is the
same upstream-cut run without rescue. Selected downstream edges are patched
from the unrescued run into the rescued run. This tests whether those edges
specifically transmit the rescue-induced effect. Negative steering is not used
in this condition because it would confound rescue blocking with a second,
independently behavior-suppressing intervention.

The frozen downstream ranking was discovered using response-time steering at
the current token. Its ability to block a fixed prompt-end rescue is therefore
a strong transfer test, not something guaranteed by construction. If
prompt-end rescue succeeds but the frozen downstream edges do not block it,
attribute a prompt-end-specific downstream ranking before rejecting the
bottleneck. If only current-token rescue is blocked, the supported object is a
recurrent interface.

## 4. End-to-end comparator estimand

The end-to-end discovery implemented in this branch uses the frozen shared
prompt identities and the same paired response objective as the downstream
experiment.

For each frozen response pair:

1. Build token-aligned positive- and negative-persona prompts.
2. Append the same high-trait answer to each prompt.
3. Append the same low-trait answer to each prompt in a separate pass.
4. Attribute the metric

   `mean_logp(high answer) - mean_logp(low answer)`

   from `blocks.0.hook_resid_pre` to response-token logits.
5. Interpolate the block-0 residual state from the negative prompt to the
   positive prompt. Answer tokens are held fixed within each counterfactual.

This avoids a single "sycophancy token" proxy and directly asks which complete
edge paths make the natural prompt increase preference for trait-expressing
responses.

The decomposition and end-to-end experiments must use the same:

- discovery prompt identities;
- high/low response pairs;
- maximum transcript length without truncation;
- held-out prompt- and response-disjoint internal pairs;
- the same official behavioral questions, transparently labeled as historical
  development prompts rather than an untouched benchmark;
- judge and coherence rubric.

## 5. Circuit evaluation contract

### 5.1 Internal metrics

For every circuit family and k report:

- requested greedy frontier budget;
- retained edges after connectivity pruning;
- eligible-edge fraction;
- number of complete-path edges;
- source-disconnected and sink-disconnected edge counts;
- clean, corrupt, circuit-only, and circuit-ablated metric means;
- normalized faithfulness and necessity;
- paired bootstrap confidence intervals over prompt identity.

Use `source_to_sink` circuits for the main faithfulness curve. Use `causal_cut`
for necessity and cut-to-interface rescue. Causal rescue does not require the
ablated edge set to be reconstructively sufficient.

An internal circuit is provisionally acceptable when:

- the clean-corrupt denominator is stable and behavior-aligned;
- held-out faithfulness is at least 0.85 for the final claimed circuit;
- held-out necessity is high without incoherent or generic-output collapse;
- the curve is broadly monotonic in retained edges;
- topology-stratified random controls are near the corrupt baseline;
- results are stable across discovery resamples/templates.

The 0.85 threshold follows the operational target used in the EAP-IG
faithfulness paper. It is not a universal law, but dropping it post hoc would
make the circuit claim difficult to defend.

### 5.2 Free-generation metrics

For every selected circuit report:

- trait score and coherence;
- cap-hit rate and generated-token distribution;
- paired normal-minus-intervention effect with confidence interval;
- threshold-crossing count among prompts with a high baseline;
- unrelated behavior/capability controls;
- one decoding backend across normal and circuit conditions;
- exact generated-token IDs and hashes, not decoded-text re-tokenization;
- exact model, prompt, circuit, decoding, and judge hashes.

Necessity alone is insufficient. An end-to-end circuit presented as a complete
mechanism must preserve the trait in circuit-only generation. The decomposed
union must do the same for the stronger level-3 claim; level-2 mediation instead
requires specific rescue and downstream blocking.

The repaired standalone upstream and downstream behavior runners generate
normal, counterfactual, ablated, and circuit-only conditions through the same
full-prefix, ungrouped-GQA reference backend. They reject reuse of historical
normal baselines produced by the old KV-cache backend. The steering and
connected runners also pin the response-vector hash; connected runs pin both the
unit-direction and raw response-vector hashes against discovery metadata.

### 5.3 Controls

Minimum controls:

- topology-stratified signed-score permutations using the same greedy selector;
- at least five seeds;
- same requested frontier budget and reported retained size;
- orthogonal and unrelated-trait interface rescue;
- clean negative-prompt behavior;
- coherence and general capability;
- noising versus denoising asymmetry;
- discovery resample stability;
- prompt-template or instruction-template stability.

If space permits, add a stronger retained-size-matched connected control. Do not
compare a connected circuit only with uniformly random disconnected edges.
The implemented `topology_stratified_v2` control permutes signed scores within
source/target component type, qkv role, quartile depth, and coarse edge-span
strata. The earlier exact-layer strata left many MLP edges in singleton groups
and therefore did not meaningfully randomize those pathways.

## 6. Integrated-gradients convergence

Five integration steps are not automatically invalid, but they are not
defensible without a convergence check. Recent circuit-variance work uses much
larger step counts and shows that attribution rankings can vary across
resamples/templates.

Run a fixed 8- or 16-prompt pilot at IG steps 5, 20, and 50. Use identical
samples and all other settings.

For each pair of runs compare:

- signed-score Pearson correlation;
- absolute-score Spearman correlation;
- top-k Jaccard at k = 50, 500, 3,000, and 10,000 where applicable;
- retained source-to-sink circuit overlap;
- held-out faithfulness/necessity for circuits rebuilt from each score table.

Decision rule:

- Use 20 steps if 20 versus 50 has absolute-score Spearman >= 0.95, top-3,000
  Jaccard >= 0.80, and held-out faithfulness differs by <= 0.05.
- Keep 5 steps only if it also satisfies those criteria against 20.
- Otherwise use 50 for the full run.
- Do not spend a full 200-step run unless 50 remains visibly unconverged.

## 7. Counterfactual and interaction diagnostics

The official positive and negative persona system prompts are semantically far
apart. The "Curse of Multiple Mediators" analysis shows that noising estimates
contain interaction terms and that those terms grow with intervention distance.
Individually ranked mediators need not predict group behavior.

Required diagnostics:

1. Report both circuit ablation (noising selected edges) and circuit-only
   reconstruction (denoising selected edges into the corrupt computation).
2. Measure their prompt-level variance and asymmetry.
3. Add a closer counterfactual pilot:
   - keep assistant naming neutral;
   - change only the trait instruction;
   - preserve the same user question and assistant suffix.
4. The official Persona Vectors sycophancy extract/eval questions in this
   project are opinion/user-preference prompts. Call the result **opinion
   sycophancy**, not universal sycophancy. A factual-sycophancy dataset is a
   separate replication rather than a stratum hidden in the current 20 items.
5. Compare the official large-distance counterfactual with the closer
   instruction-only counterfactual before claiming a universal trait circuit.

The closer counterfactual is a robustness analysis, not a replacement for the
official Persona Vectors replication.

## 8. Position aggregation

Current EAP scores sum contributions across token positions while the selected
edge is patched at every position. Position-aware EAP work shows that aggregation
can hide cancellation and inflate unimportant edges.

For this deadline:

- do not implement a partially position-aware selector whose evaluation patches
  a different object than discovery scored;
- record prompt, assistant-boundary, and response lengths;
- inspect whether discoveries are stable across short and long transcripts;
- report this limitation explicitly;
- run a position-aware extension only after the connected/end-to-end results are
  secure.

A rushed mismatch between edge-position attribution and all-position patching
would be methodologically worse than a clear limitation.

## 9. Token-length and truncation decision

`max_length` and `max_new_tokens` are different experimental controls:

- `max_length` truncates an input or teacher-forced transcript.
- `max_new_tokens` caps free generation.

Teacher-forced discovery/evaluation rows may be pooled only when every full
transcript fit without truncation under the recorded limit. The new end-to-end
scripts fail rather than silently changing the estimand.

Greedy generation rows from different caps may be pooled when all conditions
below hold:

1. Same model revision, prompt text, circuit hash, condition, and decoding.
2. Deterministic greedy decoding.
3. The reused row ended naturally before the lower common cap.
4. The generated token prefix and decoded answer are complete.
5. Split provenance and cap metadata are retained.

Therefore the existing sycophancy q0-q6 rows do not need rerunning if they
terminated naturally below 350 tokens and pass the hash/provenance checks in the
handoff. q7-q19 can be generated at 350 and pooled with them.

Different teacher-forced `max_length` values also do not require rerunning when
the full tokenized transcript fit under every recorded cap. Audit this without
GPU inference:

```bash
uv run python scripts/rq.py persona.audit-token-lengths \
  --config configs/qwen_sycophancy.json \
  --sample-manifest data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/shared_circuit_dataset_v1.json \
  --upstream-evaluation-manifest data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/upstream_prompt_disjoint_eval_v1.json \
  --downstream-evaluation-manifest data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/downstream_prompt_response_disjoint_eval_v2.json \
  --caps 256 350 500 640 768 1024 \
  --output results/qwen2.5-7b-instruct/persona/sycophantic/token_length_audit_v1.json
```

Only a family/cap cell with `exceeds > 0` has a changed teacher-forced
estimand. This audit cannot make a capped free-generation response complete;
use the split-cap pooling rules for generation.

Do not treat capped circuit-only rows as ordinary outcomes. Report them
separately and run a sensitivity analysis. A cap that covers all sycophantic
source answers but truncates many helpful answers can bias trait and coherence
comparisons.

## 10. Three-day execution plan

### Day 1: CPU rebuild and cheap decision gates

Do not rerun EAP first.

1. Freeze separate 32-row upstream and downstream evaluation manifests. Both
   exclude discovery prompt identities and official behavioral questions; the
   downstream v2 manifest also excludes discovery response-pair IDs, rejects
   duplicate source pair IDs, and verifies exact high- and low-answer token
   suffix equality under both prompt counterfactuals.
2. Rebuild both circuit families from the frozen sycophancy score tables.
3. Audit how many frontier edges survive complete-path pruning at each k.
4. Evaluate source-to-sink circuits on the frozen held-out internal pairs.
5. Run five-question prompt-only, prompt-end cut-set vector/full/orthogonal
   rescue and downstream blocking using the already strong k=50 and k=3,000
   interventions. Use five orthogonal seeds.
6. Generate five topology-stratified, retained-size-matched random controls at
   the most informative k.
7. Run connected circuit-only generation only if internal faithfulness is
   promising.

Canonical sycophancy rebuild:

```bash
UP_EVAL=data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/upstream_prompt_disjoint_eval_v1.json
DN_EVAL=data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/downstream_prompt_response_disjoint_eval_v2.json
SHARED=data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/shared_circuit_dataset_v1.json

uv run python scripts/rq.py persona.freeze-upstream-evaluation \
  --config configs/qwen_sycophancy.json \
  --discovery-manifest "$SHARED" \
  --num-prompts 32 --max-length 256 --output "$UP_EVAL"

uv run python scripts/rq.py persona.freeze-downstream-evaluation \
  --config configs/qwen_sycophancy.json \
  --discovery-manifest "$SHARED" \
  --num-pairs 32 --max-length 768 --output "$DN_EVAL"

uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config configs/qwen_sycophancy.json \
  --scope upstream \
  --run-name sharedprompt_n64_ig5_k3000_20260726 \
  --families source_to_sink causal_cut \
  --k-values 50 100 250 500 1000 2000 3000 5000 10000 20000

uv run python scripts/rq.py persona.rebuild-nested-circuits \
  --config configs/qwen_sycophancy.json \
  --scope downstream \
  --run-name sharedprompt_n64_ig5_full768_h100_cneg2_20260802 \
  --families source_to_sink causal_cut \
  --k-values 50 100 250 500 1000 2000 3000 5000 10000
```

Evaluate the connected artifacts by passing explicit paths under
`circuits_source_to_sink/`.

```bash
RESULTS=results/qwen2.5-7b-instruct/persona/sycophantic
UP=sharedprompt_n64_ig5_k3000_20260726
DN=sharedprompt_n64_ig5_full768_h100_cneg2_20260802

for K in 50 100 250 500 1000 2000 3000 5000 10000 20000; do
  uv run python scripts/rq.py persona.evaluate-upstream-contrastive-circuit \
    --config configs/qwen_sycophancy.json --run-name "$UP" --k "$K" \
    --circuit-path "$RESULTS/circuit_discovery_upstream_contrastive/$UP/circuits_source_to_sink/eapig_top_${K}.json" \
    --evaluation-manifest "$UP_EVAL" --num-pairs 32 --max-length 256 \
    --output-tag "connected_k${K}_n32_prompt_disjoint"
done

for K in 50 100 250 500 1000 2000 3000 5000 10000; do
  uv run python scripts/rq.py persona.evaluate-steering-circuit \
    --config configs/qwen_sycophancy.json --run-name "$DN" --k "$K" \
    --circuit-path "$RESULTS/circuit_discovery_steering_path/$DN/circuits_source_to_sink/steering_path_eapig_top_${K}.json" \
    --evaluation-manifest "$DN_EVAL" \
    --num-prompts 32 --batch-size 1 --max-length 768 \
    --output-tag "connected_k${K}_n32_prompt_disjoint"
done
```

Historical downstream curves that excluded only response IDs but reused
discovery prompt identities are development evidence. Do not use them for the
paper's k selection.

Build topology-matched controls at the k values nearest the selection gate:

```bash
uv run python scripts/rq.py persona.build-circuit-controls \
  --config configs/qwen_sycophancy.json --scope upstream --run-name "$UP" \
  --family source_to_sink --k-values 3000 10000 20000 --seeds 0 1 2 3 4

uv run python scripts/rq.py persona.build-circuit-controls \
  --config configs/qwen_sycophancy.json --scope downstream --run-name "$DN" \
  --family source_to_sink --k-values 3000 10000 --seeds 0 1 2 3 4
```

Stop rules:

- Judge upstream sparsity by retained complete-path edges, not only the requested
  greedy frontier. If k=20,000 retains less than 1% of eligible edges, extend the
  CPU-only frontier conditionally to k=50,000 and k=100,000 before viewing
  additional held-out outcomes, and evaluate the smallest frontiers reaching
  approximately 1%, 2%, and 5% retained edges.
- If connected upstream faithfulness remains below 0.30 after at least 2% of
  eligible edges are retained, with no convincing upward curve, do not spend
  the night on upstream circuit-only generation. If no circuit reaches 0.85 by
  5% retained edges, do not claim a sparse reconstructive formation circuit.
  Preserve the cut result and prioritize cut-to-interface rescue plus
  end-to-end.
- If connected downstream faithfulness remains near the existing result, reuse
  the frozen downstream score table.
- If full-prefix rescue does not reproduce same-backend normal generation,
  stop: the connected intervention or interface implementation is invalid.
- If full-prefix rescue succeeds but final-prompt full-state rescue fails, the
  proposed single-state prompt bottleneck is false.
- If final-prompt full-state rescue succeeds but vector rescue fails, stop
  calling the one-dimensional persona vector the natural bottleneck.
- If vector rescue works but downstream ablation does not block it, the selected
  downstream edge set is incomplete or is not the vector readout circuit.

### Night 1: IG convergence pilots

Run the same 8 or 16 manifest rows at IG 5, 20, and 50 for upstream,
downstream, and end-to-end. All three discovery scripts support
`--manifest-sample-indices`. Use new immutable run names such as
`syc_up_ig20_pilot16_v1`; set `--num-pairs 16` or `--num-prompts 16` to match
the supplied indices. Do not compare runs with different rows, transcript caps,
metrics, or steering coefficients.

Compare score tables:

```bash
uv run python scripts/rq.py persona.compare-score-tables \
  --config configs/qwen_sycophancy.json \
  --scope upstream \
  --left PATH_TO_IG20/eapig_edge_scores.parquet \
  --right PATH_TO_IG50/eapig_edge_scores.parquet \
  --k-values 50 500 3000 10000 \
  --output PATH_TO_COMPARISON.json
```

Use `--scope downstream` and `--scope end_to_end` for the other graph
families. The output includes raw top-k overlap and retained source-to-sink
circuit overlap. Select the lowest converged step count using Section 6.

### Day 2: full end-to-end comparator

Run one full sycophancy end-to-end discovery using the selected IG count and the
frozen shared manifest:

```bash
uv run python scripts/rq.py persona.run-end-to-end-eapig \
  --config configs/qwen_sycophancy.json \
  --run-name sycophancy_end_to_end_n64_igSTEPS_v1 \
  --sample-manifest data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/shared_circuit_dataset_v1.json \
  --num-prompts 64 \
  --batch-size 1 \
  --max-length 768 \
  --ig-steps STEPS \
  --k-values 500 1000 3000 5000 10000 20000
```

Immediately rebuild/evaluate the connected curve on prompt- and
response-disjoint pairs.
Select k using internal criteria without looking at official behavioral judge
scores.

```bash
E2E=sycophancy_end_to_end_n64_igSTEPS_v1
for K in 500 1000 3000 5000 10000 20000; do
  uv run python scripts/rq.py persona.evaluate-end-to-end-circuit \
    --config configs/qwen_sycophancy.json --run-name "$E2E" --k "$K" \
    --evaluation-manifest "$DN_EVAL" \
    --num-prompts 32 --max-length 768 \
    --output-tag "k${K}_n32_prompt_disjoint"
done

uv run python scripts/rq.py persona.build-circuit-controls \
  --config configs/qwen_sycophancy.json --scope end_to_end \
  --run-name "$E2E" --family source_to_sink \
  --k-values 3000 5000 10000 20000 --seeds 0 1 2 3 4
```

End-to-end stop rules:

- Clean-minus-corrupt response margin must have the intended sign.
- Connected faithfulness must increase with retained size.
- At least one feasible k should reach 0.70 in the pilot and 0.85 for the final
  circuit claim.
- Random controls must be materially worse.
- If these fail, do not multiply traits. Diagnose counterfactual distance and
  score variance first.

### Night 2: held-out generation

For the smallest k passing the internal gate:

1. Run 5 official questions for normal, negative prompt, circuit ablation, and
   circuit-only.
2. Inspect every answer before expanding.
3. If coherent and directionally correct, run q5-q19 unchanged.
4. Run cut-to-interface rescue and downstream blocking.
5. Run the decomposed union with connected upstream/downstream paths.

Do not judge failed, capped, duplicated, or provenance-mismatched artifacts.

Five-question end-to-end preflight:

```bash
uv run python scripts/rq.py persona.behavior-end-to-end-circuit \
  --config configs/qwen_sycophancy.json --run-name "$E2E" --k SELECTED_K \
  --question-indices 0 1 2 3 4 \
  --conditions normal negative_prompt circuit_ablated circuit_only \
  --max-new-tokens 350 --max-length 1024 \
  --output-tag kSELECTED_K_q0-4_preflight_v1
```

After q5-q19 are judged, freeze the primary tables without pooling the q0-q4
pilot:

```bash
uv run python scripts/rq.py persona.analyze-end-to-end-curves \
  --config configs/qwen_sycophancy.json --run-name "$E2E" \
  --k-values 500 1000 3000 5000 10000 20000 \
  --behavior-k-values SELECTED_K \
  --evaluation-template 'k{k}_n32_prompt_disjoint' \
  --question-indices 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --minimum-token-cap 350
```

For bottleneck mediation, pass explicit causal-cut paths to
`persona.behavior-connected-circuits` and run these conditions on the same five
preflight questions before expanding:

- `normal`
- `upstream_ablated`
- `downstream_ablated`
- `both_ablated`
- `upstream_full_prefix_state_rescue`
- `upstream_vector_rescue`
- `upstream_full_state_rescue`
- `upstream_orthogonal_rescue`
- `upstream_unrelated_vector_rescue --unrelated-vector-path PATH`
- `vector_rescue_downstream_ablated`

The final condition means **rescue-effect downstream blocking**: selected
downstream edges are patched from the unrescued upstream-cut computation into
the rescued computation. It is not negative steering layered on top of rescue.

Then pass explicit connected paths for the stronger reconstructive condition:

- `both_circuit_only`

The unrelated vector should be independently extracted at the same interface
dimensionality. The runner records its raw cosine and uses its target-orthogonal
component as an equal-norm semantic negative control.

Canonical prompt-bottleneck template:

```bash
CUT_UP="$RESULTS/circuit_discovery_upstream_contrastive/$UP/circuits_cut_sets/eapig_top_50.json"
CUT_DN="$RESULTS/circuit_discovery_steering_path/$DN/circuits_cut_sets/steering_path_eapig_top_3000.json"
COMMON=(
  --config configs/qwen_sycophancy.json
  --upstream-run "$UP" --downstream-run "$DN"
  --upstream-k 50 --downstream-k 3000
  --upstream-circuit-path "$CUT_UP" --downstream-circuit-path "$CUT_DN"
  --question-indices 0 1 2 3 4
  --upstream-position-scope prompt_only --interface-timing prompt_end
  --max-new-tokens 350 --max-length 1024
)

for CONDITION in normal upstream_ablated downstream_ablated both_ablated \
  upstream_full_prefix_state_rescue upstream_full_state_rescue \
  upstream_vector_rescue vector_rescue_downstream_ablated; do
  uv run python scripts/rq.py persona.behavior-connected-circuits \
    "${COMMON[@]}" --condition "$CONDITION" \
    --output-tag "${CONDITION}_prompt-only_prompt-end_q0-4_v1"
done

for SEED in 0 1 2 3 4; do
  uv run python scripts/rq.py persona.behavior-connected-circuits \
    "${COMMON[@]}" --condition upstream_orthogonal_rescue --orthogonal-seed "$SEED" \
    --output-tag "orthogonal_seed${SEED}_prompt-only_prompt-end_q0-4_v1"
done

uv run python scripts/rq.py persona.behavior-connected-circuits \
  "${COMMON[@]}" --condition upstream_unrelated_vector_rescue \
  --unrelated-vector-path PATH_TO_UNRELATED_VECTOR \
  --output-tag "unrelated_prompt-only_prompt-end_q0-4_v1"
```

Judge all conditions with the same frozen judge, then aggregate them with:

```bash
uv run python scripts/rq.py persona.analyze-bottleneck-mediation \
  --config configs/qwen_sycophancy.json \
  --normal-scored NORMAL_SCORED \
  --upstream-ablated-scored UPSTREAM_ABLATED_SCORED \
  --vector-rescue-scored VECTOR_RESCUE_SCORED \
  --full-state-rescue-scored FULL_STATE_RESCUE_SCORED \
  --full-prefix-state-rescue-scored FULL_PREFIX_RESCUE_SCORED \
  --orthogonal-rescue-scored ORTHOGONAL_SEED0_SCORED ORTHOGONAL_SEED1_SCORED ORTHOGONAL_SEED2_SCORED ORTHOGONAL_SEED3_SCORED ORTHOGONAL_SEED4_SCORED \
  --unrelated-rescue-scored UNRELATED_RESCUE_SCORED \
  --downstream-blocked-scored DOWNSTREAM_BLOCKED_SCORED \
  --downstream-ablated-scored DOWNSTREAM_ABLATED_SCORED \
  --both-ablated-scored BOTH_ABLATED_SCORED \
  --minimum-token-cap 350 --bootstrap-samples 10000 \
  --output "$RESULTS/bottleneck_mediation_q0-4_v1"
```

The analyzer requires the same connected-run backend for `normal`, verifies
that vector rescue restores the intended interface coordinate, and requires
full-prefix rescue to reproduce the exact normal greedy token sequence. It
reports additive paired effects first, direct vector-versus-control contrasts,
and ratio estimates only when bootstrap denominators are stable. Bootstrap
intervals resample paired questions; orthogonal-control contrasts additionally
resample the five pre-specified random directions. When downstream-only and
joint-ablation artifacts are supplied together, it also reports upstream-only,
downstream-only, joint, and factorial interaction effects from the same
backend-matched 2x2 design.

Treat the five-question result only as a preflight. Expand unchanged conditions
to q5-q19 and use those 15 rows for the primary interval. The q0-q4 pilot may
appear only as a labeled descriptive sensitivity analysis. The final evidence
gate is:

- full-prefix rescue exactly matches normal on all 15 primary generations;
- the paired upstream-cut effect has a 95% bootstrap interval above zero;
- vector rescue has a 95% interval above zero;
- both pre-specified specificity contrasts, vector minus orthogonal rescue and
  vector minus unrelated-vector rescue, have intervals above zero;
- downstream blocking of vector rescue has an interval above zero;
- coherence and cap-hit diagnostics show no generic collapse.

Report full-state-minus-vector and full-prefix-minus-full-state as continuous
diagnostics rather than forcing an arbitrary success threshold. Rescue
fractions are useful scale summaries, but additive effects and direct paired
contrasts are primary because ratio estimands become unstable when a bootstrap
sample has a small denominator.

For an ICLR submission, freeze a new external opinion-sycophancy question set
before inspecting any intervention outputs and replicate the selected method
there. The official 20-question set cannot honestly be called untouched after
the historical ablation and circuit-size development runs.

Afterward, repeat `upstream_vector_rescue`,
`upstream_full_state_rescue`, and `vector_rescue_downstream_ablated` with
`--upstream-position-scope all_positions --interface-timing current_token`.
That small timing ablation determines whether the interface is prompt-formed or
recurrently reconstructed.

### Day 3: controls, second trait, and freeze

1. Complete random controls and paired confidence intervals.
2. Run the same CPU connected rebuild on hallucination before any new
   hallucination discovery.
3. Run a full second-trait experiment only if the sycophancy decision gates pass.
4. Freeze tables, manifests, and figures.
5. Start second-model replication only after the Qwen mechanism is internally
   coherent. A weakly replicated method on two models is not stronger than one
   correctly established mechanism.

## 11. Experiment priority under the deadline

Priority A:

- connected rebuild from frozen scores;
- connected internal evaluation;
- cut-set interface vector/full/orthogonal rescue and downstream blocking;
- end-to-end sycophancy discovery/evaluation;
- held-out end-to-end generation;
- matched controls and confidence intervals.

Priority B:

- connected hallucination rebuild;
- one second trait with enough held-out prompts;
- factual-sycophancy replication with a separately frozen dataset;
- discovery resample stability.

Priority C:

- multilingual persona transfer;
- second model;
- full position-aware circuits;
- additional traits.

Multilingual transfer and a second model are attractive, but they cannot rescue
an unfaithful primary circuit. Establish the estimand first.

## 12. Publication framing

Primary contribution:

- a causal formation-interface-expression decomposition for high-level persona
  behavior, distinguishing sparse interception, low-dimensional mediation, and
  faithful reconstruction.

Required comparator:

- an independently discovered end-to-end natural-prompt behavioral circuit,
  showing what the chosen interface captures or omits.

Methodological contribution:

- a demonstrated distinction between sparse causal cut sets and faithful
  source-to-target circuits for high-level LLM behavior.

Cross-lingual contribution:

- only after the primary mechanism is secure, test whether formation,
  interface, and expression transfer differently across languages.

This framing is more defensible against closely related 2026 work:

- Persona Vectors provides the representation/interface starting point.
- CircuitSteer studies SAE feature-flow circuits for behaviors including
  sycophancy and refusal.
- Who Bridges Safety studies cross-lingual shared safety pathways.
- The shared sycophancy-lying work argues for reused causal circuitry.
- Recent sycophancy-subtype work separates factual and opinion mechanisms.
- Tracing Persona Vectors Through LLM Pretraining studies when persona
  directions emerge, rather than the inference-time edge mechanism.
- Perfect Detection, Failed Control shows why a readable direction cannot be
  assumed to be a causal control axis.
- Refusal Lives Downstream of Persona proposes a related stage decomposition;
  the paper must distinguish itself through complete edge-level reconstruction,
  interface rescue, and held-out natural behavior.

The differentiator should not be "behavior circuits exist." It should be the
causal bridge between edge-level computation and a behaviorally validated
low-dimensional interface: exact rescue, negative controls, downstream
blocking, and comparison with a separately discovered end-to-end mechanism.

The nearest-work comparison should be explicit:

| Work | What it establishes | What this project must add |
| --- | --- | --- |
| Persona Vectors | Linear trait directions monitor and steer behavior and training shifts | Whether a natural edge-level computation forms and reads the direction |
| Have Faith in Faithfulness | Faithful circuits require source-to-target reconstruction and direct faithfulness tests | High-level generative behavior, mediation rescue, and a cut-versus-circuit distinction |
| Refusal Lives Downstream of Persona | Direction-level persona/refusal interaction and a late expression stage | Edge-level formation/expression circuits, exact interface rescue, and independent end-to-end comparison |
| CircuitSteer | SAE feature-flow interventions can steer sycophancy and refusal | Natural-prompt causal mediation and circuit-only faithfulness rather than steering efficacy alone |
| Shared Sycophancy-Lying Circuit | Reused components can control factual deference/lying | Opinion-sycophancy formation and expression around a measured interface |
| Sycophancy subtype dissociation | Factual and opinion sycophancy can have distinct representations | A correctly scoped opinion-sycophancy mechanism, with factual replication left separate |
| Multiple Mediators / circuit-variance work | Independent patching rankings contain interactions and can vary by sample/template | Noising/denoising asymmetry, IG convergence, resampling, and close-counterfactual diagnostics |

For a workshop submission, one fully controlled Qwen opinion-sycophancy
mediation result plus the frozen multilingual refusal result is a coherent
paper. For ICLR, the same central mechanism should be replicated along at least
one independent axis after it passes: preferably a second model for the
prompt-bottleneck test, otherwise a separately frozen factual-sycophancy or
hallucination dataset with enough held-out behavior. Adding several incomplete
traits is weaker than one clean replication.

## 13. Claim checklist

Allowed after the repaired experiments:

- "A small edge set is necessary" when cut-set ablation beats matched controls.
- "A circuit is faithful" only for source-to-sink circuit-only recovery.
- "The vector mediates behavior" after specific cut-to-interface rescue and
  downstream blocking.
- "Formation and expression decompose" when both halves pass the tests
  appropriate to the claim level: necessity plus rescue for a mediated
  bottleneck, and circuit-only faithfulness for a complete mechanism.
- "The mechanism transfers" only with frozen source circuits and held-out target
  language/model evaluation.

Not allowed:

- high necessity implies sufficiency;
- a parentless edge shell is a circuit;
- projection correlation implies natural mediation;
- one selected k is a unique minimum;
- pooled capped outputs are equivalent to uncapped outputs;
- a mixed sycophancy score identifies one homogeneous mechanism.

## 14. Literature anchors

- Hanna et al., "Have Faith in Faithfulness: Going Beyond Circuit Overlap",
  arXiv:2403.17806.
- Chen et al., "Persona Vectors", arXiv:2507.21509.
- Haklay et al., "Position-aware Automatic Circuit Discovery",
  arXiv:2502.04577.
- "The Curse of Multiple Mediators: Hidden Interaction Effects in Transformer
  Interpretability", arXiv:2606.27510.
- Wu et al., "Demystifying Variance in Circuit Discovery of LLMs",
  arXiv:2606.16920.
- "CircuitSteer", arXiv:2608.05732.
- "Who Bridges Safety? Identifying and Targeting Cross-Lingual Shared Safety
  Pathways", arXiv:2608.09095.
- "Dissociating the Internal Representations of Sycophancy in LLMs",
  arXiv:2607.07003.
- "The Shared Sycophancy-Lying Circuit", arXiv:2604.19117.
- "Tracing Persona Vectors Through LLM Pretraining", arXiv:2605.13329.
- "Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in
  Language Models", arXiv:2606.24952.
- "Refusal Lives Downstream of Persona in Chat Models", arXiv:2606.26161.

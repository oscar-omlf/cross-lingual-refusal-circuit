# Frozen refusal finalization protocol

Status: **FROZEN BEFORE NEW RUNS**  
Protocol version: `refusal-final-v1`  
Frozen: `2026-08-23`  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Revision: `a09a35458c702b33eeacc393d103063234e8bc28`

This protocol supersedes the exploratory refusal runs for final quantitative
claims. Old results remain motivation and debugging evidence. They must not be
used to change this protocol after final evaluation begins.

## Frozen data

The 748 semantic prompt identities are disjoint between stages.

| Stage | File | Prompts | SHA-256 |
|---|---|---:|---|
| Vector extraction | `data/processed/qwen2.5-7b-instruct/trait_circuits/refusal/vector_extraction.jsonl` | 256 | `477b2cc6cb7963e816ac59c2d4ab4d526e1e6f033ce070946c670396d40522c6` |
| Vector validation | `data/processed/qwen2.5-7b-instruct/trait_circuits/refusal/vector_validation.jsonl` | 64 | `7feff3faae7839117f4144f78c11068664d82cb6de34a30ce70bb3602fcbc58a` |
| EAP-IG scoring | `data/processed/qwen2.5-7b-instruct/trait_circuits/refusal/eapig.jsonl` | 64 harmful | `4a94fe2979a15b9299655899b01c0b559aed743e30001dc408bab7b9dc267939` |
| Circuit selection | `data/processed/qwen2.5-7b-instruct/trait_circuits/refusal/circuit_selection.jsonl` | 64 harmful | `427d3fa8a06c72a2fa6eadf33ed12b3acfa95d5faf2eb6ab9cecf4641442186d` |
| Final behaviour | `data/processed/qwen2.5-7b-instruct/trait_circuits/refusal/final_behavior.jsonl` | 172 harmful + 128 harmless | `3fa08080f26e7e6846d28dda6ff7b7e496dc416aa47eed91f2215d49f223ebe5` |

For multilingual runs, English, German, Chinese and Thai must use translations
with exactly these `parallel_id` values. Translation does not create additional
semantic datapoints.

The final split was created after exploratory work. Eighty-eight of its 300
identities appeared in archived exploratory artifacts. We retain the agreed
split, disclose this limitation, freeze all choices now, and make no decisions
from final results.

## Frozen vector and causal target

- Use the existing English refusal vector; do not re-extract it.
- Vector SHA-256:
  `d83c57bbbe5afd92175a855677dd744b178bc4062b98cc78f4e277859c35453c`.
- Target: `blocks.15.hook_resid_pre`, prompt position `-2`.
- Metric: projection onto the unit refusal direction.
- Corrupt condition: global projection of the refusal direction, matching the
  refusal paper and the completed vector validation.
- EAP-IG integrated-gradient steps: `5`.

## Stage 1: edge scoring

Run EAP-IG independently in `en`, `de`, `zh`, and `th` on the 64 canonical
EAP-IG identities. Every language uses the same English vector and the same
parallel semantic identities.

Save edge scores, complete run metadata, dataset hashes, model revision and
software revision. No circuit-size decision may use this split.

## Stage 2: candidate circuits

Build nested greedy, source-to-target connected circuits at:

`k = 1000, 2000, 3000, 4000, 5000`

Build one family from each language's scores. The primary behavioural circuit
is the English-discovered family; language-local families are used for
structural comparison.

## Stage 3: select and lock k

Use only the 64 canonical circuit-selection identities.

1. Evaluate the English-discovered circuit in all four languages for every k.
2. A k is representation-eligible only if, in every language:
   - faithfulness is at least `0.85`;
   - necessity is at least `0.95`.
3. Test generated behaviour for representation-eligible k values in ascending
   order. Accept the first k for which, in every language:
   - normalized refusal reversal is at least `0.85`;
   - circuit-only refusal retention is at least `0.80`.
4. Generation is greedy and limited to `64` new tokens. Refusal is judged by
   local WildGuard, with refusal-prefix rates retained as a diagnostic.
5. If no tested k passes, report that result. Do not add larger k values without
   a dated protocol amendment made before examining any final-set result.

After selection, record the selected k and hashes of all four circuit files in
a lock manifest. The selected k must never change based on final results.

## Stage 4: controls at locked k

Use only the circuit-selection split. Evaluate teacher-forced faithfulness and
necessity in all four languages for:

- discovered full circuit;
- terminal-only edges;
- upstream-only edges;
- three topology-matched random circuits, seeds `0`, `1`, and `2`;
- fixed discovered terminal shell plus random upstream edges, seeds `0`, `1`,
  and `2`.

Random masks must match the discovered circuit's retained edge count and its
terminal/upstream edge counts. No additional generated-behaviour control is
required.

## Stage 5: one final behavioural evaluation

Run only after k and all analysis choices are locked.

For each language, use the same 300 semantic identities.

- Harmful prompts, `n=172`: normal, global direction projection, circuit only,
  and circuit ablated.
- Harmless prompts, `n=128`: normal and circuit ablated.
- Greedy generation, `64` new tokens.
- Judge with local WildGuard.
- Save per-instance IDs, condition, token count and judge labels, but do not
  retain harmful prompt or completion text in the final artifact.
- Report paired bootstrap confidence intervals.

The final split is reporting-only. It cannot be used to select k, alter masks,
change thresholds, add conditions or tune the judge.

## Derived analyses

These require no new model inference:

- faithfulness, necessity and behavioural curves versus k;
- pairwise edge-score Pearson and Spearman correlations;
- circuit IoU and weighted overlap;
- circuit composition and sparsity;
- final harmful behaviour and harmless specificity figures;
- discovered-versus-random control figure.

## Amendment rule

Any change to datasets, hashes, vector, corruption, k grid, thresholds,
conditions, languages, generation length, judge or random seeds requires a
dated protocol amendment before the affected results are examined. Final-set
results can never justify an amendment.

## Post-run causal-mediation interpretation

**Added:** `2026-08-24`  
**Status:** Interpretive follow-up; this does not amend the frozen protocol.

The completed English results strongly support the discovered circuit as an
upstream refusal-vector formation circuit:

- `k=1000` vector faithfulness: `87.8%`.
- Vector necessity: `100.3%`.
- Circuit-only refusal retention: `96.7%`.
- Circuit-ablation refusal reversal: `91.5%`.

However, these results do not establish an isolated end-to-end circuit. The
discovered graph ends at layer 15; layers 15–27 remain intact during behavioural
evaluation. Therefore, `circuit only` means:

`discovered upstream circuit + the complete downstream model`.

Circuit patching also replaces complete component outputs across token
positions, not only their refusal-direction components.

### Direct mediation gate

On 16 separate reserved harmful prompts:

| Condition | Refusal-vector coordinate | Refusal log-odds |
|---|---:|---:|
| Normal | 15.79 | 3.89 |
| Global direction projection | approximately 0 | -13.62 |
| Local projection at layer 15, position -2 | approximately 0 | 3.59 |
| Upstream circuit ablated | approximately 0 | -8.58 |
| Circuit ablated, then local vector restored | 15.79 | -4.30 |

- Local vector removal explained only `1.68%` of the global-projection effect.
- Restoring the clean vector coordinate recovered `34.4%` of the upstream
  circuit-ablation effect, with all 16 prompts moving in the predicted
  direction.

### Interpretation

These results reject the strongest hypothesis:

> The refusal direction at layer 15, position -2 is the sole causal bottleneck
> separating a formation circuit from a transmission circuit.

The coordinate is a genuine but partial and context-dependent mediator.
Removing it locally from an otherwise clean computation barely changes
refusal, while restoring it after broad upstream circuit ablation produces a
substantial but incomplete rescue.

The broader formation–transmission hypothesis remains plausible:

`formation circuit -> distributed mediator -> transmission circuit`.

The missing mediation may involve:

- the refusal direction at multiple token positions;
- the refusal direction across multiple layers;
- additional orthogonal directions at layer 15;
- a low-dimensional refusal subspace;
- parallel pathways not captured by the selected coordinate.

Global projection removes the direction throughout the model, so its large
behavioural effect cannot be attributed specifically to the selected
layer-position.

### Minimal discriminating follow-up

After ablating the upstream circuit, compare:

1. Restoring the refusal direction at every layer-15 position.
2. Restoring the complete residual state at layer 15, position -2.
3. Restoring the complete layer-15 state across all positions as an upper
   bound.
4. Applying the same behavioural intervention with topology-matched random
   circuits.

| Result | Interpretation |
|---|---|
| All-position direction rescue is strong | Distributed one-direction mediator |
| Full-state position -2 rescue is strong | Local multidimensional mediator |
| Only complete layer-15 restoration works | Broad distributed interface |
| Random circuits behave similarly | Current circuit ablation is nonspecific |

The downstream transmission circuit has not yet been identified. Establishing
the full decomposition additionally requires showing that ablating a downstream
circuit blocks the behavioural effect of restoring or inducing the mediator.

Detailed completed results are recorded in:

- `results/qwen2.5-7b-instruct/refusal/final_v1/en/EXPERIMENT_RECORD.md`
- `results/qwen2.5-7b-instruct/refusal/downstream_v1/en/CAUSAL_MEDIATION_NOTE.md`

The multilingual evaluations, frozen random controls, and final 300-prompt
evaluation specified above remain incomplete.

### Critical response to the causal-mediation interpretation

The interpretation above is broadly sound. In particular, it correctly
separates representation-level reconstruction from end-to-end behavioural
mediation, and it correctly rejects the claim that the single
layer-15/position--2 coordinate is a sufficient bottleneck.

Several qualifications are necessary:

- The `87.8%` faithfulness and `100.3%` necessity results support a circuit for
  reconstructing the selected representation. They do not by themselves show
  that this is the unique natural formation mechanism.
- The `96.7%` retention and `91.5%` reversal numbers are development evidence:
  they use the 64 circuit-selection prompts and a post-hoc, explicitly
  documented WildGuard contradiction filter. The unfiltered reversal was
  `83.9%`, and the frozen 300-prompt final evaluation has not been run.
- The `34.4%` value is a conditional rescue ratio on first-token refusal
  log-odds, not a formally identified “percentage mediated.” The outcome is
  nonlinear, the rescue interacts with a broad circuit intervention, and only
  16 prompts were tested. Its paired bootstrap interval was
  `[20.7%, 49.8%]`.
- Nevertheless, every prompt showed a positive rescue effect while the clean
  vector coordinate was restored almost exactly. This is evidence that the
  coordinate has a conditional causal contribution; it is stronger than a
  purely correlational result, but weaker than proof of a standalone mediator.
- Circuit ablation replaces complete selected-edge outputs with counterparts
  from the globally projected run, across token positions. Its behavioural
  effect can therefore travel through other positions, orthogonal dimensions
  or parallel pathways. Until matched random controls are evaluated,
  nonspecific disruption is not excluded.
- Weak local ablation can also occur if later layers rapidly reconstruct the
  direction after it is removed at layer 15. Measuring the direction coordinate
  through subsequent layers after local projection is therefore an important
  cheap diagnostic.

The most defensible current statement is:

> The selected circuit reconstructs the refusal-direction coordinate at the
> chosen layer-position and strongly affects refusal under the current broad
> patching intervention. That coordinate contributes causally in the
> circuit-ablated context, but it is neither necessary nor sufficient as an
> isolated clean-model bottleneck. The remaining effect may reflect a
> distributed vector-related mechanism, additional state variables, or
> nonspecific effects of broad patching.

The minimal follow-up order should be:

1. Measure whether the direction reappears in later layers after local
   layer-15/position--2 projection.
2. Compare direction rescue across all layer-15 positions with full-state
   rescue at position -2.
3. Run topology- and size-matched random-circuit controls using the identical
   intervention.
4. Only pursue a downstream EAP-IG circuit from an interface whose local
   intervention has a clear behavioural contrast. The current source-only
   contrast is small (`0.29` log-odds; 95% CI `[-0.06, 0.65]`), so a circuit
   extracted from it should not yet be interpreted as the downstream refusal
   circuit.

Thus the follow-up is useful and mostly correct, but “formation circuit,”
“partial mediator” and the filtered behavioural percentages must be presented
with the qualifications above. The broader formation--distributed
mediator--transmission account remains a hypothesis, not a result.

### Follow-up: intervention scope and a distributed direction trajectory

**Added:** `2026-08-24`

The weak local-ablation result does not contradict the original refusal-vector
interventions because their scopes differ.

| Intervention | Scope |
|---|---|
| Current local mediation gate | Layer 15, position -2, once |
| Refusal-vector activation addition | Selected layer, every token position |
| Refusal-vector directional ablation | Every layer and every token position |

During autoregressive generation, the activation-addition hook remains active:
it modifies the full prompt during the initial pass and each newly generated
token during cached decoding. The selected layer-position is therefore the
location used to *extract* the direction, not the only location modified during
behavioural intervention.

Our completed results already show that this distinction is substantial:

| Condition | Refusal-prefix rate |
|---|---:|
| Normal | 31/32 |
| Layer-15 all-position `-1x` steering | 2/32 |
| Layer-15 all-position `-2x` steering | 0/32 |
| Global direction projection | 0/32 |

By contrast, projecting the direction only at layer 15, position -2 changed
first-token refusal log-odds by only `0.29`. Thus intervention scope is a
plausible major explanation for the apparent discrepancy.

#### Current mechanistic hypothesis

The strongest remaining hypothesis is not a point bottleneck:

`formation circuit -> direction trajectory across positions/layers -> transmission`

The layer-15/position--2 coordinate may be one observable checkpoint within
that trajectory. Refusal information can remain at other prompt positions,
be reconstructed by later layers, and be maintained at generated-token
positions. A single local removal can therefore be bypassed, whereas recurrent
all-position steering repeatedly supplies a strong control signal.

This also distinguishes **controllability** from **natural mediation**.
All-position steering can force behaviour even if the model does not naturally
depend on one local coordinate. The decisive mechanistic test is therefore
exact clean-coordinate removal or restoration, with steering used as
behavioural validation.

#### Minimal discriminating tests

1. After local layer-15/position--2 projection, track the direction coordinate
   through layers 16--27 to test rapid reconstruction.
2. Project the direction at all layer-15 prompt positions and measure
   first-token refusal.
3. After upstream-circuit ablation, restore the exact clean direction
   coordinate across all layer-15 prompt positions in a teacher-forced pass.
4. If this produces substantial rescue, compare prompt-only restoration with
   recurrent prompt-plus-generation restoration during free generation.
5. Retain matched random-circuit and random-direction controls.

| Result | Interpretation |
|---|---|
| Direction rapidly reappears | Local removal is repaired downstream |
| All-prompt-position removal is strong | Direction is spatially distributed |
| All-position rescue greatly exceeds `34.4%` | Distributed one-direction mediation |
| Recurrent rescue exceeds prompt-only rescue | Generated tokens maintain the mechanism |
| Only full-state rescue works | The interface is multidimensional |

If supported, downstream EAP-IG should target an aggregate direction trajectory
across the causally relevant positions, rather than treating
layer 15, position -2 as an isolated bottleneck.

### Follow-up: synthesis and recommended decision path

**Added:** `2026-08-24`  
**Status:** Interpretive follow-up; no frozen protocol change.

#### What the evidence currently supports

- The layer-15/position--2 refusal direction is a strong control direction.
  Global projection and recurrent all-position steering strongly suppress
  refusal.
- The selected `k=1000` circuit reconstructs that prompt coordinate well and
  strongly affects behaviour under broad circuit patching.
- Removing only that coordinate once has almost no behavioural effect.
  Restoring it after circuit ablation gives a consistent but partial rescue.
- Therefore, we have evidence for a **behaviour-linked representation
  circuit**, but not yet for a single-coordinate end-to-end refusal circuit.

The most plausible working model is:

`prompt formation circuit -> distributed state/direction trajectory -> generation/transmission circuit`

The prompt and generation circuits may overlap. Distinct prompt and generation
vectors in sycophancy make this architecture plausible, but do not establish
it for refusal. Arditi-style steering also does not isolate them: the selected
layer is modified at every prompt position and again at generated-token
positions during decoding.

#### Minimum experiments before redefining the circuit

Use the existing 16 reserved harmful prompts; do not touch the final set.

1. **Trajectory diagnostic:** after projecting the direction only at
   layer 15, position -2, measure its coordinate at later layers and positions.
2. **Prompt-scope test:** compare exact projection at position -2, at every
   layer-15 prompt position, and globally during prompt processing.
3. **Rescue test:** after upstream-circuit ablation, compare exact clean
   direction restoration at position -2 and across all layer-15 prompt
   positions. Add full-state restoration at position -2 only if directional
   rescue remains weak.
4. **Specificity control:** repeat the circuit intervention with
   topology- and size-matched random circuits.
5. **Temporal-scope test:** only if prompt-wide results are insufficient,
   compare prompt-only, generation-only, and prompt-plus-generation
   intervention during free generation.

These tests separate four explanations:

| Observation | Conclusion |
|---|---|
| Direction quickly reappears | Later layers reconstruct a locally removed signal |
| Prompt-wide removal/rescue is strong | A distributed prompt-direction interface is plausible |
| Generation-only intervention is additionally necessary | A distinct recurrent generation circuit is involved |
| Only full-state or global intervention works | Refusal is multidimensional or broadly distributed |

#### Consequence for EAP-IG

- If prompt-wide directional mediation is strong, redefine the interface as
  the aggregate layer-15 prompt trajectory and score upstream/downstream edges
  against that causal interface.
- If generation-time intervention is independently necessary, extract and
  validate a refusal generation vector, compare it with the prompt vector, and
  construct a separate generation/transmission circuit.
- If no local or single-layer interface has a clear causal contrast, do not
  claim a clean upstream/downstream decomposition; use an end-to-end distributed
  circuit formulation instead.

The pending source-only downstream EAP-IG run uses the weak
layer-15/position--2 contrast. It may be reported as exploratory, but it cannot
by itself establish the downstream refusal circuit.

#### Paper claim until these tests are complete

> We identified a sparse circuit that reconstructs a refusal-associated prompt
> representation and whose broad causal intervention strongly alters refusal.
> The representation contributes conditionally to behaviour, but current
> evidence suggests that natural mediation is distributed across positions,
> layers, generation time, or additional dimensions.

This framing preserves the strong results without overclaiming a unique
formation circuit, a scalar bottleneck, or a complete end-to-end mechanism.

# Sparse causal circuits for refusal and persona-mediated behavior

Comprehensive research and implementation handoff for branch
`handoff/persona-circuits-20260811`.

Status date: 2026-08-12.

This document is written for a new researcher who has not previously worked on
the project. It records:

- the original research idea and implementation;
- the completed evidence that must be preserved;
- every important methodological, framing, evaluation, and implementation
  problem found during the review;
- the repaired scientific design;
- the exact interpretation of each experiment;
- the code changes made on this branch;
- the experiments to run, skip, or run only conditionally;
- the decision gates that determine the strongest defensible paper claim;
- the intended workshop and ICLR contributions;
- the reproducibility and reporting contract.

Generated activations, score tables, responses, judgments, and many result
tables are Git-ignored. Paths in this document refer to the shared HPC checkout
unless explicitly stated otherwise. A fresh Git clone is not sufficient to run
the remaining study without restoring the canonical artifact bundle.

## 0. Document precedence and how to use this handoff

The handoff consists of three coordinated documents:

1. `RESEARCH_HANDOFF.md`, this file, is the master scientific, historical,
   methodological, framing, and status document.
2. `PERSONA_CIRCUIT_METHOD_REPAIR.md` is the extended scientific rationale for
   the repair, including the causal estimands, literature positioning, and the
   three-day strategy.
3. `SYCOPHANCY_PERSONA_CIRCUIT_REPAIRED_RUNBOOK.md` is the authoritative
   executable protocol. It contains exact Bash, Slurm, judging, pooling, and
   analysis commands. It was deliberately kept as a separate single source of
   executable truth so that commands do not drift between duplicated copies.

When the documents disagree:

1. Use this document for scientific interpretation and status.
2. Use the repaired runbook for exact commands, filenames, ordering, and
   resource requests.
3. Use the methodology-repair document for additional rationale.
4. Treat the older persona runbooks and historical orchestration scripts as
   historical unless this document explicitly re-enables them.

The first action on HPC is not a full attribution run. Follow Sections 5-8 of
the repaired runbook:

1. initialize the environment;
2. audit the frozen artifacts;
3. freeze or verify the new disjoint evaluation manifests;
4. audit token lengths;
5. rebuild both circuit families from the frozen score tables;
6. run the small technical smoke;
7. stop if the full-prefix rescue does not exactly reproduce normal generation.

## 1. Executive summary

### 1.1 Original scientific idea

The project asks whether high-level LLM behaviors can be explained by sparse,
causally important edge-level computations.

For persona-mediated behavior, the primary hypothesis is:

```text
formation computation
    -> residual-stream persona-vector interface
    -> expression computation
```

The original plan was:

1. extract and validate a persona vector;
2. discover an upstream circuit that forms that vector at its best layer;
3. discover a downstream circuit that converts the vector into behavior;
4. connect the two circuits around the interface;
5. test necessity by ablation and sufficiency by circuit-only generation;
6. optionally compare the decomposition with direct end-to-end circuit
   discovery.

That hypothesis remains scientifically valuable. A direct end-to-end circuit
alone can show a sparse behavioral mechanism, but it cannot explain whether a
low-dimensional persona representation is a causal interface between formation
and expression. The decomposition therefore remains the primary explanatory
hypothesis. End-to-end discovery is now a required independent comparator and
falsification test, not a replacement.

### 1.2 What was wrong

The central upstream problem was not simply "the circuit may be too small."
The historical persona scripts selected backward greedy edge sets with
parentless pruning disabled:

```python
paper_greedy_edges(..., prune_parentless=False)
```

That produces sink-connected edge shells. Many selected edges can reach the
measured endpoint, but they need not lie on complete paths from the graph input
to the endpoint.

Such an object can be highly necessary:

- ablating a terminal or near-terminal conduit intercepts the clean
  computation;
- the behavior can reverse strongly;
- the measured endpoint can collapse.

The same object can be extremely unfaithful in circuit-only mode:

- selected nodes with no selected parents are constructed from corrupt inputs;
- clean information cannot reach the selected terminal shell;
- adding shell edges may improve reconstruction only when complete paths happen
  to appear.

The old upstream result is therefore not a faithful formation circuit. It is a
sparse causal cut or interception set. The high necessity and low faithfulness
are not contradictory and do not by themselves imply that EAP-IG or the
patching algebra is broken.

### 1.3 What the repair changes

The repaired study separates three circuit objects:

1. `causal_cut`
   - sink-connected;
   - may contain source-disconnected edges;
   - valid for ablation, necessity, interception, and mediation;
   - not valid for circuit-only reconstruction claims.
2. `source_to_sink`
   - every retained edge lies on a complete graph-input-to-target path;
   - valid for circuit-only faithfulness and reconstructive sufficiency.
3. `end_to_end`
   - independently attributed from the natural prompt at block 0 to
     response-token logits;
   - uses the same high-versus-low answer objective as the downstream study;
   - tests what the explicit persona interface captures or omits.

The paper claims are now nested:

1. **Sparse causal interception**
   - a small selected cut is necessary for the behavior.
2. **Low-dimensional causal interface**
   - the upstream cut reduces the interface coordinate and behavior;
   - restoring the missing vector coordinate specifically rescues behavior;
   - equal-norm orthogonal and unrelated-vector controls do not;
   - the downstream cut blocks the rescue-induced effect.
3. **Reconstructive formation-expression mechanism**
   - complete upstream and downstream circuits are internally faithful;
   - their connected circuit-only union coherently reproduces the behavior.

A strong level-2 result is scientifically meaningful even if the complete
upstream formation computation is larger than originally expected. In that
case the small object is an interception set converging on a low-dimensional
causal mediator, not a tiny sufficient formation circuit.

### 1.4 The main completed evidence

Refusal is the mature foundation:

- Gemma-2-2B supplies a single-language refusal circuit result.
- Qwen2.5-7B supplies multilingual English, German, Chinese, and Thai evidence.
- Qwen's selected English circuit has strong internal and behavioral
  faithfulness.
- Multilingual consensus and intersection results support shared pathways.

Opinion sycophancy supplies strong historical necessity evidence:

- upstream k=50 ablation reduces mean sycophancy from 66.95 to 8.19;
- upstream k=3,000 reduces it to 4.16;
- downstream k=3,000 internal faithfulness/necessity is 0.978/1.003;
- joint historical ablation reduces 66.95 to 1.31 while preserving coherence;
- the legacy upstream circuit-only recovery remains low.

These sycophancy numbers remain valid as historical causal-cut evidence. They
do not yet establish the repaired mediation or reconstructive claims.

### 1.5 What is code-complete and what is not experimentally complete

Code-complete and statically validated:

- deterministic circuit selection;
- connectivity diagnostics;
- separate cut and source-to-sink artifacts;
- CPU rebuilds from frozen score tables;
- disjoint evaluation manifest freezers;
- no-truncation internal evaluation;
- prompt-only and recurrent intervention timing;
- full-prefix, full-state, vector, orthogonal, and unrelated-vector rescue;
- rescue-specific downstream blocking;
- same-backend 2x2 ablations;
- independent end-to-end attribution, evaluation, generation, and analysis;
- topology-stratified controls;
- IG convergence comparison;
- exact token and artifact provenance;
- deterministic split pooling;
- paired bootstrap internal, behavioral, mediation, and interaction analyses.

Not yet experimentally complete:

- HPC technical smoke;
- repaired source-to-sink internal curves;
- IG 5/20/50 convergence decisions;
- five-question mediation preflight;
- q5-q19 primary mediation;
- full end-to-end comparator;
- topology-control evaluations;
- connected source-to-sink circuit-only behavior, if the internal gates pass;
- recurrent timing diagnostic;
- new external opinion-sycophancy replication for ICLR.

No repaired GPU result should be described as completed until its metadata,
rows, hashes, and outputs pass the runbook checks.

## 2. Research questions and strongest possible contributions

### 2.1 Primary research question

Can a high-level generative behavior be causally decomposed into:

1. an upstream edge-level computation that changes a low-dimensional residual
   representation;
2. a residual-stream interface that mediates the behavioral effect;
3. a downstream edge-level computation that reads the interface into generated
   behavior?

The current persona target must be called **opinion sycophancy**. The official
questions are user-preference and opinion prompts. They do not establish one
universal sycophancy mechanism, and they should not be silently combined with
factual sycophancy.

### 2.2 Why the bottleneck idea has value

The bottleneck idea is stronger than simply finding edges spread throughout the
network if the interface is tested causally.

A useful bottleneck claim requires more than:

- a vector that correlates with the trait;
- a layer that responds to steering;
- an upstream ablation that changes behavior;
- a downstream circuit found using the vector.

It requires mediation:

1. natural positive and negative persona conditions differ at the interface;
2. the interface direction can causally change held-out behavior;
3. the upstream cut reduces both the coordinate and behavior;
4. restoring the prompt-specific missing coordinate rescues behavior;
5. matched control directions do not rescue equivalently;
6. the selected downstream computation blocks the rescue;
7. a separately found end-to-end circuit provides a comparison that did not
   assume the interface.

If these pass, the contribution is not merely "persona vectors exist" or
"behavior circuits exist." It is an edge-level causal account of how a
behaviorally validated representation is formed and read.

### 2.3 Primary contribution

The primary contribution should be framed as:

> A causal formation-interface-expression decomposition for high-level
> generative behavior that explicitly distinguishes sparse interception,
> low-dimensional mediation, and complete circuit reconstruction.

### 2.4 Methodological contribution

The methodological contribution is:

> High behavioral necessity does not imply circuit faithfulness. For high-level
> LLM behaviors, sink-connected causal cuts and source-to-target reconstructive
> circuits must be treated as different scientific objects and tested with
> different interventions.

This is supported by the concrete historical failure:

- a very small upstream edge set nearly eliminates sycophancy when ablated;
- the same selected object does not reconstruct the interface coordinate;
- the selection topology explains why.

### 2.5 Comparator contribution

The independent end-to-end circuit is required to answer:

- does the persona-vector decomposition capture the same behavioral
  computation as a direct natural-prompt source-to-logit circuit?
- is the vector interface explanatory or merely a convenient steering axis?
- does end-to-end reconstruction succeed when vector mediation fails?
- does the decomposition yield cleaner causal localization than the direct
  circuit?

Raw edge overlap is secondary because introducing an explicit interface
refactors edges at the boundary. Behavioral and internal cross-faithfulness are
more meaningful than naive overlap at the interface.

### 2.6 Cross-lingual contribution

The existing refusal work already supports a cross-lingual shared-circuit
story. For persona behavior, multilingual transfer is a later extension:

- test vector transfer first;
- then freeze source-language formation, expression, and union circuits;
- evaluate them on translated held-out prompts;
- compare with local and topology-matched controls.

Do not let multilingual breadth delay or obscure the primary causal mechanism.

### 2.7 Workshop and ICLR packages

Defensible workshop package:

1. frozen Gemma and multilingual Qwen refusal evidence;
2. one fully controlled Qwen opinion-sycophancy mediation study;
3. independent end-to-end comparison;
4. explicit cut-versus-circuit methodological lesson.

Preferred ICLR strengthening:

1. replicate the selected mechanism on a newly frozen external
   opinion-sycophancy set;
2. add one independent replication axis:
   - preferably a second model for the same prompt-bottleneck test;
   - otherwise factual sycophancy as a separately scoped behavior;
   - or hallucination with an adequately sized held-out behavioral set.
3. add discovery-resample and instruction-template stability.

Several incomplete traits are weaker than one clean causal replication.

## 3. Project state before the methodology repair

### 3.1 Refusal program

#### Gemma-2-2B English refusal

The earlier RQ1 result found a sparse behaviorally causal refusal circuit in
`google/gemma-2-2b-it`.

The committed results summary records:

- main target: a residual projection onto a refusal direction at layer 14,
  token position -2;
- top-1,000 selection retained 668 edges;
- on 32 prompts:
  - normal refusal substring rate: 1.000;
  - direction ablation: 0.000;
  - circuit-only: 0.969;
  - circuit ablation: 0.094.

This result is not rerun for the persona deadline.

#### Qwen2.5-7B multilingual refusal

The mature Qwen refusal result uses PolyRefuse in English, German, Chinese, and
Thai.

Frozen English k=3,000 result on 32 harmful prompts:

- normal refusal: 96.9%;
- direction-ablated refusal: 6.3%;
- circuit-ablated refusal: 15.6%;
- circuit-only refusal: 93.8%;
- normalized behavioral reversal: 89.7%;
- internal faithfulness: 0.960;
- internal necessity: 1.010;
- selected edges: 3,000 of 260,956 eligible, or 1.15%.

Cross-language normalized reversal at k=3,000:

- German: 0.852;
- English: 0.897;
- Chinese: 0.920;
- Thai: 0.880.

At k=5,000 the four-language mean is 0.945.

A 1,000-edge multilingual consensus circuit has faithfulness:

- English: 0.926;
- German: 0.863;
- Chinese: 0.904;
- Thai: 0.877.

The strict four-language intersection contains 706 edges, preserves about
72-81% of the projection depending on language, and is nearly fully necessary.

Harmless specificity:

- English harmless refusal remains 3.1%;
- German changes by +6.3 percentage points;
- Chinese changes by 0;
- Thai changes by -3.1;
- no harmful control responses were detected.

Frozen Qwen tables:

- `results/qwen2.5-7b-instruct/polyrefuse/phase2_refusal/frozen_development_2026-07-19/`
- `results/qwen2.5-7b-instruct/polyrefuse/phase2_refusal/frozen_controls_2026-07-20/`
- `results/qwen2.5-7b-instruct/polyrefuse/phase2_refusal/frozen_terminal_decision_2026-07-20/`

Do not rerun these absent a concrete reviewer-motivated need.

### 3.2 Persona-vector extraction

The persona pipeline reproduces the public Persona Vectors protocol at commit:

`b8e0f044fe2410a6fad579f38324f03f13b4e917`

The config pins:

- official source files and hashes;
- trait questions and instructions;
- generation settings;
- judge prompts and model;
- vector extraction convention;
- steering-layer candidates;
- selected model revision.

For Qwen opinion sycophancy:

- 20 questions;
- 5 positive instructions and 5 negative instructions;
- 10 rollouts per question/instruction;
- 1,000 positive and 1,000 negative generations;
- GPT-4.1-mini judge;
- 810 effective paired responses after filtering:
  - positive trait score >=50;
  - negative trait score <50;
  - coherence >=50 for both.

At each layer:

```text
response_average_vector
  = mean(response-token activations | positive persona)
  - mean(response-token activations | negative persona)
```

Selected interface:

- HF hidden state index 20;
- TransformerLens hook `blocks.20.hook_resid_pre`;
- official steering convention applies the vector at block 19;
- layer 20 won at all tested positive coefficients 0.5-2.5;
- coefficient -2 is the validated negative downstream counterfactual.

These extraction, judgment, vector, layer-selection, and steering artifacts are
reused unless provenance fails.

### 3.3 Original upstream discovery

Graph:

- source: `blocks.0.hook_resid_pre`;
- target: the interface at `blocks.20.hook_resid_pre`;
- clean input: positive-persona prompt;
- corrupt input: token-aligned negative-persona prompt;
- same question and assistant suffix;
- target scalar: final-prompt-token signed projection onto the normalized
  sycophancy direction;
- integrated gradients: 5 steps;
- no projection ablation in the attribution path.

Canonical frozen score table:

- run: `sharedprompt_n64_ig5_k3000_20260726`;
- 64 frozen discovery prompt identities;
- 471,191 eligible edges;
- one immutable score table;
- all historical k circuits derived from that ranking.

Historical internal result:

- necessity 0.813 at k=500;
- necessity 1.002 at k=3,000;
- legacy circuit-only recovery:
  - 0.067 at k=3,000;
  - 0.135 at k=5,000;
  - 0.193 at k=20,000.

Historical free-generation ablation on 20 official questions:

- normal mean sycophancy: 66.95;
- k=50 ablated mean: 8.19;
- k=500 ablated mean: 5.00;
- k=3,000 ablated mean: 4.16;
- all 16 initially high prompts crossed below 50;
- coherence remained about 99.

Correct current interpretation:

> The upstream ranking contains a very small causal interception set. The
> historical selected artifacts do not establish a small sufficient formation
> circuit.

### 3.4 Original downstream discovery

Graph:

- source: the layer-20 interface;
- target: response-token logits;
- prompt: identical positive-persona prompt for both responses;
- clean and corrupt responses: high-trait and low-trait answers;
- scalar:

```text
mean_logp(high-trait answer) - mean_logp(low-trait answer)
```

- corruption path: response-time vector coefficient from -2 to 0;
- integrated gradients: 5 steps.

Canonical frozen score table:

- run: `sharedprompt_n64_ig5_full768_h100_cneg2_20260802`;
- same 64 discovery prompt identities;
- 70,157 eligible edges;
- paired transcripts up to 768 tokens.

Historical internal result:

- k=50 faithfulness/necessity: 0.719/0.895;
- k=3,000 faithfulness/necessity: 0.978/1.003.

Historical free-generation ablation:

- normal mean: 66.95;
- k=50 ablated mean: 23.06;
- k=3,000 ablated mean: 21.55;
- 43.9-46.0 point reductions;
- coherence >=99.4;
- 12-14 of 16 high-baseline prompts reversed.

The downstream result is much closer to a faithful circuit, but its paper
selection curve must be recomputed on the new prompt- and response-disjoint
manifest and its IG-5 ranking must pass convergence.

### 3.5 Original connected union

The original connected union independently combined:

- an upstream selected object;
- the interface;
- a downstream selected object.

Historical selected union:

- upstream k=5,000;
- downstream k=3,000;
- joint ablation mean: 1.31 from normal 66.95;
- coherence: 98.38;
- no cap hits.

Correct interpretation:

- this is strong joint causal necessity;
- the old union was composed from historical cut-style selections;
- it is not yet the main source-to-sink circuit-only reconstruction result;
- it does not by itself show that the persona vector mediates the upstream
  effect;
- joint ablation alone cannot reveal whether the two interventions are
  redundant, additive, or synergistic.

### 3.6 Legacy connected circuit-only generation

Historical low-memory backend changes produced repeated exclamation marks.
Those outputs are invalid.

The restored TransformerLens reference backend was coherent, but a 40 GB A100
OOMed on question 7 at generated token 370.

Potentially reusable legacy rows:

- questions 0-6;
- 93-341 generated tokens;
- coherent;
- naturally ended before 350;
- artifact tag contains `max500_a100_vanilla_reference_v5`.

Abandoned or invalid:

- `max1000_a100_lowmem_v2`: 20/20 cap hits and incoherent punctuation;
- one-row `reference_v3`;
- abandoned `max640_a100_final`;
- cancelled job `25448716`, which saved no q7-q19 rows.

Do not finish q7-q19 merely to complete the old table. The edge set changes
under the repaired source-to-sink method. Preserve q0-q6 only as an explicitly
labeled legacy cut-set diagnostic.

### 3.7 Hallucination

Completed:

- response-average vector from 716 effective pairs;
- interface layer 16;
- layer 16 wins coefficients 1.0-2.5, but not 0.5;
- upstream ranking:
  `hallucination_shared_n64_ig5_v1`;
- 33 token-alignable discovery identities;
- 298,073 eligible upstream edges;
- downstream ranking:
  `hallucination_shared_n64_ig5_cneg_v1_lowmem_v1`;
- 164,395 eligible downstream edges;
- max transcript 256.

Problem:

- strict alignment leaves only 8 official question/instruction pairs;
- behavior artifacts therefore have only 16 rows when normal and intervention
  are stored;
- this is not paper-ready at the sycophancy sample size.

Do not launch new hallucination work until the primary sycophancy gates pass.
Then rebuild its circuit families from frozen scores before considering new
discovery.

### 3.8 Evil and additional traits

`configs/qwen_evil.json` and `EVIL_PERSONA_CIRCUIT_RUNBOOK.md` are scaffolding.
No evil vector or circuit result exists.

Do not describe evil as an experiment. Do not begin evil, politeness, or other
traits before the primary mechanism is secure.

## 4. Complete methodology and framing problem inventory

### 4.1 Parentless selected shells were called circuits

Problem:

- backward greedy selection began at the endpoint;
- `prune_parentless=False` retained edges that could reach the endpoint but
  were not reachable from the graph input;
- circuit-only mode therefore had no clean information path through many
  selected edges.

Consequence:

- high necessity;
- very low faithfulness;
- misleading use of "formation circuit";
- slow apparent faithfulness growth with k.

Repair:

- emit `causal_cut` and `source_to_sink` separately;
- calculate source and sink reachability;
- require every reconstructive edge to lie on a complete path;
- report requested frontier k and retained complete-path edge count separately.

### 4.2 Necessity was treated as evidence of sufficiency

Problem:

- a small set that destroys behavior when ablated was implicitly treated as a
  small mechanism that should reproduce behavior alone.

Consequence:

- the low circuit-only result looked like a mysterious implementation failure;
- the scientific claim exceeded the evidence.

Repair:

- necessity supports interception;
- source-to-sink circuit-only recovery supports reconstruction;
- free-generation sufficiency is separate from internal representational
  sufficiency;
- all three are reported independently.

### 4.3 Vector localization was treated as natural mediation

Problem:

- a readable and steerable vector can be useful without being the coordinate
  naturally mediating the upstream intervention.

Consequence:

- the formation-interface-expression story was assumed rather than tested.

Repair:

- measure the prompt-specific interface deficit after upstream cutting;
- restore the missing coordinate;
- compare with full-state, full-prefix, orthogonal, and unrelated-vector
  rescues;
- block the rescue effect downstream.

### 4.4 The positive control was not strong enough

Problem:

- failed vector rescue could reflect a broken intervention implementation
  rather than a false one-dimensional bottleneck.

Repair hierarchy:

1. same-backend normal;
2. full-prefix interface-state rescue;
3. final-prompt full-state rescue;
4. vector-only rescue.

Required interpretation:

- full-prefix failure: implementation invalid, stop;
- full-prefix success and final-state failure: no single final-prompt-state
  bottleneck;
- final-state success and vector failure: interface is higher-dimensional than
  the selected vector;
- vector success: candidate one-dimensional mediator, subject to specificity
  and downstream-blocking tests.

### 4.5 Prompt formation and recurrent expression timing were conflated

Problem:

- upstream attribution targets the final prompt token;
- generation interventions can also corrupt every later response token;
- success under all-position intervention does not prove a one-time
  prompt-formed bottleneck.

Repair:

Primary:

```text
--upstream-position-scope prompt_only
--interface-timing prompt_end
```

Diagnostic:

```text
--upstream-position-scope all_positions
--interface-timing current_token
```

Interpretation:

- prompt-end success supports a prompt-formed interface;
- only current-token success supports a recurrent layerwise interface.

### 4.6 Downstream blocking used the wrong causal contrast

Problem:

- adding negative steering on top of vector rescue would suppress behavior for
  an independent reason;
- this does not isolate whether selected downstream edges carry the rescued
  signal.

Repair:

For `vector_rescue_downstream_ablated`:

- clean computation is the upstream-cut run with vector rescue;
- corrupt cache is the same upstream-cut run without rescue;
- selected downstream edges are patched from unrescued into rescued;
- negative steering is not added.

This tests whether the downstream selected set transmits the rescue-induced
effect.

### 4.7 Joint ablation lacked a factorial design

Problem:

- `both_ablated` alone cannot distinguish additive, redundant, or synergistic
  upstream and downstream interventions.

Repair:

Run the same backend for:

- `normal`;
- `upstream_ablated`;
- `downstream_ablated`;
- `both_ablated`.

Estimate the factorial interaction from paired questions.

### 4.8 Normal and intervened generations used different backends

Problem:

- some historical normal baselines came from a KV-cache path;
- intervention runs used a full-prefix TransformerLens path;
- grouped-query attention and caching differences can create backend-specific
  text differences unrelated to the intervention.

Repair:

- use the full-prefix ungrouped-GQA reference backend for every condition;
- record exact generated token IDs and hashes;
- reject historical normal artifacts from incompatible backends;
- require full-prefix rescue and normal token hashes to match exactly.

### 4.9 Historical low-memory generation was invalid

Problem:

- low-memory modifications generated repeated punctuation and cap-hit artifacts.

Repair:

- do not use those rows;
- use the reference backend;
- split long generation jobs by question;
- pool only deterministic, naturally terminated rows with matching provenance;
- report cap hits explicitly.

### 4.10 Upstream paper evaluation reused behavioral questions

Problem:

- historical upstream internal evaluation was built from official `eval.json`
  questions;
- those questions were also used for free-generation behavior.

Repair:

- freeze 32 extraction-prompt identities disjoint from discovery and official
  behavior;
- select upstream k only on this manifest.

### 4.11 Downstream paper evaluation was response-disjoint but not prompt-disjoint

Problem:

- historical downstream evaluation excluded discovery response IDs but could
  reuse discovery prompt identities.

Repair:

- freeze a v2 manifest that is both prompt- and response-pair-disjoint;
- reject duplicate source pair IDs;
- verify exact high- and low-answer token suffixes under both counterfactual
  prompts;
- use the same manifest for downstream and end-to-end internal evaluation.

### 4.12 The official 20 behavior questions are no longer untouched

Problem:

- historical ablation and circuit-size sweeps already used all 20 questions;
- they remain disjoint from circuit discovery, but they are development data.

Repair:

- use q0-q4 only as a diagnostic pilot for new mediation/end-to-end conditions;
- freeze all settings after the pilot;
- use q5-q19 for the primary new-condition interval;
- never pool q0-q4 into the primary confidence interval;
- freeze a new external opinion-sycophancy set before ICLR.

### 4.13 `max_length` and `max_new_tokens` were confused

Problem:

- `max_length` controls input or teacher-forced transcript tokenization;
- `max_new_tokens` controls free generation;
- different recorded values were treated as automatically requiring reruns.

Repair:

- teacher-forced rows need rerunning only when a complete transcript exceeded
  the recorded cap;
- new internal scripts reject overlength rows instead of truncating;
- free-generation rows are reusable only if they naturally ended before the
  lower common cap and all causal/provenance fields match;
- capped rows are not complete outcomes.

### 4.14 Five IG steps were unvalidated

Problem:

- the frozen score tables use IG-5;
- attribution rankings can vary with integration count, prompts, and templates;
- five steps are not defensible without a convergence check.

Repair:

- use the same 16 frozen rows at IG 5, 20, and 50 for upstream, downstream, and
  end-to-end;
- compare signed-score Pearson, absolute-score Spearman, raw top-k overlap,
  complete-path overlap, and held-out faithfulness.

Precommitted rule:

- use 20 if 20 versus 50 has:
  - absolute-score Spearman >=0.95;
  - top-3,000 Jaccard >=0.80;
  - held-out faithfulness difference <=0.05;
- retain 5 only if 5 versus 20 also passes;
- otherwise use 50;
- do not spend on 200 unless 50 is visibly unconverged.

### 4.15 Circuit size was not tested in topology-aware terms

Problem:

- requested frontier k is not the retained source-to-sink circuit size;
- the legacy upstream curve might remain low because complete paths are sparse
  in early frontiers.

Repair:

- report retained complete-path edges and eligible-edge fraction;
- rebuild from frozen scores at increasing frontiers;
- extend upstream to k=50,000 and k=100,000 only if needed;
- stop reconstructive claims if no circuit reaches 0.85 faithfulness by the
  first tested frontier retaining at least 5% of eligible edges, or k=100,000.

This directly tests the "maybe the circuit is just larger" explanation without
paying for a new ranking.

### 4.16 Random controls were not topology-matched enough

Problem:

- uniformly random disconnected edges are too weak a comparison for a
  connected circuit;
- exact-layer strata created singleton groups for many MLP edges.

Repair:

- permute signed scores within component type, qkv role, quartile depth, and
  coarse edge-span strata;
- use the same greedy selector;
- match the requested frontier and report retained size;
- run five seeds at the selected k;
- report every seed.

### 4.17 Position aggregation remains a limitation

Problem:

- attribution scores are aggregated across token positions;
- evaluation patches the selected edge at every selected position;
- position aggregation can hide cancellation or inflate edges.

Decision:

- do not rush a partially position-aware selector whose evaluated object differs
  from the scored object;
- record prompt, boundary, and response lengths;
- inspect short-versus-long stability;
- state the limitation;
- add position-aware circuits only after the primary study is secure.

### 4.18 Counterfactual distance creates interaction risk

Problem:

- positive and negative persona prompts are semantically far apart;
- noising scores can contain mediator interactions;
- individually ranked effects need not predict group behavior.

Repair:

- report noising and denoising separately;
- report their asymmetry and prompt-level variance;
- add a closer instruction-only counterfactual after the primary result;
- keep assistant naming neutral in that robustness test.

### 4.19 Trait scope was too broad

Problem:

- the official sycophancy data are opinion and preference prompts;
- factual sycophancy may use different representations.

Repair:

- call the result opinion sycophancy;
- treat factual sycophancy as a separate replication.

### 4.20 Edge overlap can be misleading at an explicit interface

Problem:

- decomposing at a residual layer changes graph boundaries;
- edges entering and leaving the interface become pseudo-terminal or
  pseudo-input edges;
- raw overlap with an end-to-end graph is not invariant.

Repair:

- make internal and behavioral cross-faithfulness primary;
- exclude or explicitly map interface-boundary edges in structural comparisons;
- keep raw overlap secondary.

### 4.21 One selected k was described too strongly

Problem:

- a selected k from a finite grid is not a unique mathematical minimum.

Repair:

- report the grid, requested frontiers, retained sizes, and selection rule;
- call it the smallest tested circuit passing the precommitted gate;
- do not claim uniqueness.

### 4.22 Artifact provenance and completion checks were insufficient

Problem:

- directory names can suggest completion even when rows are missing;
- decoded-text re-tokenization does not prove token identity;
- vectors, circuits, or manifests can drift.

Repair:

- require `metadata.status == "complete"`;
- verify expected row identities;
- pin model revision, architecture, vector hashes, manifest hashes, circuit
  hashes, backend, token caps, and causal condition;
- store exact generated token IDs and SHA-256;
- reject incompatible resumes and pooling.

### 4.23 Breadth was prioritized before mechanism validity

Problem:

- additional traits, languages, and models cannot rescue an invalid primary
  estimand.

Repair:

- priority A: repaired sycophancy mediation and end-to-end comparison;
- priority B: one strong replication;
- priority C: broad extensions.

## 5. Repaired methodology

### 5.1 Fixed model and trait

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Dtype: bfloat16
- Pinned architecture:
  - 28 layers;
  - 28 attention heads;
  - `parallel_attn_mlp=false`
- Trait: opinion sycophancy
- Interface: `blocks.20.hook_resid_pre`
- Unit direction: `vector/sycophantic_direction.pt`
- Raw response vector: `vector/sycophantic_response_avg_diff.pt`
- Downstream corrupt coefficient: -2
- Generation: deterministic greedy
- Behavioral output cap: 350 new tokens
- Input/prefill cap: 1,024
- Downstream/end-to-end internal transcript cap: 768, no truncation

### 5.2 Discovery and evaluation splits

Discovery:

- 64 frozen extraction-prompt identities;
- shared between decomposed and end-to-end studies where possible;
- the score ranking is immutable within each discovery run.

Upstream internal selection:

- 32 newly frozen extraction-prompt identities;
- disjoint from discovery;
- disjoint from official behavioral questions.

Downstream and end-to-end internal selection:

- 32 prompt-disjoint and response-pair-disjoint rows;
- exact answer suffix equality verified under both prompt counterfactuals.

Behavior:

- q0-q4: judged preflight only;
- q5-q19: primary new-condition analysis;
- a new external frozen set is required for ICLR.

### 5.3 Upstream estimand

Clean:

- positive-persona prompt.

Corrupt:

- token-aligned negative-persona prompt.

Target:

```text
final-prompt hidden-state projection onto unit sycophancy direction
```

Output families:

- `causal_cut` for necessity and mediation;
- `source_to_sink` for circuit-only reconstruction.

### 5.4 Downstream estimand

Prompt:

- same positive-persona prompt.

Responses:

- high-trait answer;
- low-trait answer.

Target:

```text
mean_logp(high-trait answer) - mean_logp(low-trait answer)
```

Counterfactual:

- response-time vector coefficient -2 versus 0.

The downstream graph asks which edges read vector control into preference for
trait-expressing continuations. It does not rely on a nonexistent single
"sycophancy token."

### 5.5 End-to-end estimand

For each high/low response pair:

1. append the same high-trait answer to the positive and negative prompts;
2. append the same low-trait answer in a separate pass;
3. hold answer tokens fixed across the prompt counterfactual;
4. interpolate the block-0 residual state from negative to positive;
5. attribute the same high-minus-low response-logprob margin to response-token
   logits.

This is a natural-prompt source-to-logit circuit. It does not assume that the
persona vector is the mediator.

### 5.6 Internal circuit metrics

For each k and circuit family report:

- requested greedy frontier;
- retained edges;
- eligible edges;
- retained/eligible fraction;
- source-reachable edges;
- sink-reachable edges;
- complete-path edges;
- source-disconnected edges;
- sink-disconnected edges;
- clean metric;
- corrupt metric;
- circuit-only metric;
- circuit-ablated metric;
- faithfulness;
- necessity;
- paired bootstrap interval over prompt identity.

The normalized metrics use the clean-corrupt gap. A denominator must have the
intended sign and be stable.

Operational reconstruction gate:

- held-out faithfulness >=0.85;
- high necessity;
- broadly monotonic curve;
- no unstable denominator;
- materially better than topology controls.

The 0.85 threshold is a precommitted operational target, not a universal
definition.

### 5.7 Behavioral metrics

For selected circuits report:

- trait mean;
- coherence mean;
- normal-minus-intervention paired effect;
- 95% paired bootstrap interval;
- threshold crossings among high-baseline prompts;
- generated-token count distribution;
- cap-hit rate;
- exact token hashes;
- unrelated behavior/capability controls;
- noising versus denoising asymmetry.

Behavioral normalization uses the negative-prompt baseline:

```text
normal_minus_negative = normal_mean - negative_mean

necessity =
  (normal_mean - ablated_mean) / normal_minus_negative

sufficiency =
  (circuit_only_mean - negative_mean) / normal_minus_negative
```

Additive point effects and direct paired contrasts are primary. Ratio estimates
are secondary because they become unstable when bootstrap denominators are
small.

### 5.8 Primary mediation conditions

Run all with the same generation backend.

Core 2x2:

- `normal`
- `upstream_ablated`
- `downstream_ablated`
- `both_ablated`

Positive controls:

- `upstream_full_prefix_state_rescue`
- `upstream_full_state_rescue`

Scientific mediator:

- `upstream_vector_rescue`

Specificity controls:

- `upstream_orthogonal_rescue`, seeds 0-4;
- `upstream_unrelated_vector_rescue`, using a target-orthogonalized unrelated
  persona vector.

Transmission test:

- `vector_rescue_downstream_ablated`

The unrelated vector currently points to the hallucination response-average
vector at the same model/interface dimensionality. The runner records its raw
cosine with the target, removes the target component, normalizes it, and matches
the rescue norm.

### 5.9 Primary level-2 evidence gate

On q5-q19:

1. full-prefix rescue exact-match rate is 1.0;
2. upstream ablation effect 95% CI is above zero;
3. vector-rescue effect 95% CI is above zero;
4. vector minus orthogonal rescue 95% CI is above zero;
5. vector minus unrelated-vector rescue 95% CI is above zero;
6. downstream blocking 95% CI is above zero;
7. coherence remains high;
8. cap hits do not explain the effect.

Report:

- full-state minus vector;
- full-prefix minus full-state;
- rescue fractions;
- 2x2 interaction.

Do not force arbitrary binary thresholds on the full-state/vector hierarchy
beyond the exact full-prefix implementation check.

### 5.10 Reconstructive level-3 gate

Run connected `both_circuit_only` only when:

- upstream has a selected source-to-sink circuit passing the internal gate;
- downstream has a selected source-to-sink circuit passing the internal gate.

Then:

- preflight on q0-q4;
- inspect every response;
- expand only if coherent and directionally trait-preserving.

Failure of connected reconstruction does not invalidate a successful level-2
causal-interface result.

### 5.11 End-to-end gate

The final end-to-end circuit requires:

- positive clean-minus-corrupt response margin;
- connected source-to-sink paths;
- increasing held-out faithfulness with retained size;
- selected k with faithfulness >=0.85;
- strong necessity;
- materially worse topology controls;
- coherent q0-q4 circuit-only behavior before q5-q19 expansion.

If no k reaches 0.85:

- it may be described as a partial causal circuit;
- it may not be described as a faithful complete mechanism;
- do not choose k from official behavioral judge scores.

## 6. Interpretation decision tree

### 6.1 Full-prefix rescue fails

Conclusion:

- intervention implementation, interface capture, or backend equivalence is
  invalid.

Action:

- stop all expensive mediation runs;
- debug before scientific interpretation.

### 6.2 Full-prefix passes, final-prompt full-state fails

Conclusion:

- the full architectural prefix can restore generation;
- one final-prompt residual state is not a sufficient interface;
- do not claim a single prompt-end bottleneck.

Possible interpretation:

- behavior depends on multiple prompt positions;
- the interface is recurrent;
- the chosen final-prompt state omits relevant information.

### 6.3 Full-state passes, vector rescue fails

Conclusion:

- the selected layer/state may mediate the effect;
- the one-dimensional persona vector is not sufficient;
- the interface is higher-dimensional or the vector is not the natural
  direction.

Action:

- retain a higher-dimensional interface claim if controls support it;
- do not call the persona vector the natural bottleneck.

### 6.4 Vector rescue passes, specificity controls also rescue

Conclusion:

- rescue may reflect generic norm injection or broad state movement;
- one-dimensional target specificity is unsupported.

Action:

- inspect exact coordinate restoration and vector norms;
- consider additional semantic controls;
- do not make the target-specific mediation claim.

### 6.5 Vector rescue is specific, downstream blocking fails

Conclusion:

- the upstream/interface part may be valid;
- the frozen downstream ranking may be incomplete or timing-mismatched.

Important caveat:

- the current downstream ranking was discovered from current-token response-time
  steering;
- prompt-end blocking is a transfer test.

Action:

- run a prompt-end-specific downstream attribution before rejecting the whole
  decomposition;
- compare with the recurrent current-token diagnostic.

### 6.6 Level-2 mediation passes, upstream reconstruction fails

Conclusion:

- a small upstream cut causally acts through a low-dimensional interface;
- complete formation pathways are not sparse enough under the operational
  threshold;
- the project still has a strong causal bottleneck result;
- do not claim a tiny sufficient formation circuit.

### 6.7 Decomposed reconstruction fails, end-to-end succeeds

Conclusion:

- a faithful behavioral circuit exists;
- the selected persona vector does not capture all relevant natural
  computation;
- the vector may remain a useful control axis rather than the natural mediator.

### 6.8 Decomposed reconstruction succeeds, end-to-end also succeeds

Strongest conclusion:

- both a direct mechanism and an interpretable formation-expression
  factorization exist;
- compare behavioral/internal faithfulness, retained sizes, topology, and
  boundary-mapped overlap.

### 6.9 Only recurrent timing succeeds

Conclusion:

- the layer acts as a recurrent interface reconstructed during generation;
- do not describe it as a one-time prompt-formed bottleneck.

## 7. Implementation changes made on this branch

The following changes are currently uncommitted in the working tree. Preserve
them. Do not reset or replace them with the branch HEAD.

### 7.1 Core graph and circuit code

#### `src/refusal_circuit/graph.py`

Added:

- deterministic absolute-score ordering with edge-name tie-breaking;
- deterministic greedy heap tie-breaking;
- `circuit_connectivity`;
- source-reachability diagnostics;
- sink-reachability diagnostics;
- complete-path edge counts;
- source-disconnected, sink-disconnected, and off-path counts;
- `validate_intervention_connectivity`;
- explicit pruning to:
  - sink-connected paths for `causal_cut`;
  - complete input-to-sink paths for `source_to_sink`;
- selection diagnostics recording:
  - requested frontier;
  - frontier-selected edges;
  - pruning rule;
  - pre- and post-pruning connectivity;
- deterministic parquet score sorting.

Scientific effect:

- a cut cannot silently be labeled a reconstructive circuit;
- requested k is no longer confused with retained complete-path size.

#### `src/refusal_circuit/circuit.py`

Added:

- optional token-position masks in edge patching;
- prompt-only upstream corruption;
- natural prompt-pair mixed logits;
- exact alignment checks for clean/corrupt token tensors and attention masks;
- end-to-end circuit-only and ablation support.

Scientific effect:

- prompt formation can be separated from response-time recurrent corruption;
- the end-to-end comparator uses a natural prompt counterfactual.

#### `src/refusal_circuit/eapig.py`

Added:

- end-to-end natural-prompt EAP-IG;
- paired high-minus-low response-logprob objective;
- fixed answer tokens across prompt counterfactuals;
- block-0 residual interpolation;
- clean/corrupt margin accounting;
- non-finite score rejection.

Scientific effect:

- end-to-end discovery does not assume the persona vector;
- it uses the same behavioral preference target as downstream discovery.

#### `src/refusal_circuit/model.py`

Added:

- observed graph-relevant model architecture extraction;
- config-versus-loaded-model architecture validation.

Scientific effect:

- attribution fails early if the loaded model graph differs from the pinned
  Qwen architecture.

#### `src/refusal_circuit/persona.py`

Added:

- stable SHA-256 over exact generated token ID sequences;
- behavioral necessity/sufficiency summaries using a negative-prompt baseline;
- finite and denominator validation.

### 7.2 Repaired discovery and evaluation scripts

Modified:

- `persona_run_upstream_contrastive_eapig.py`
- `persona_run_steering_eapig.py`
- `persona_evaluate_upstream_contrastive_circuit.py`
- `persona_evaluate_steering_circuit.py`
- `persona_rebuild_nested_circuits.py`

Changes include:

- emitting both circuit families;
- accepting frozen manifest subsets for convergence pilots;
- immutable score reuse;
- no-truncation enforcement;
- explicit circuit paths;
- connectivity validation;
- manifest and vector provenance;
- complete-path metadata;
- uniform circuit-family manifests.

### 7.3 Connected behavioral runner

`scripts/persona/persona_behavior_connected_circuits.py` was substantially
expanded.

Added:

- explicit upstream and downstream circuit paths;
- cut versus reconstruction connectivity validation;
- prompt-only versus all-position upstream masks;
- prompt-end versus current-token interface timing;
- same-backend normal and all interventions;
- 2x2 ablation conditions;
- full-prefix rescue;
- final-prompt full-state rescue;
- vector rescue;
- five seeded orthogonal directions;
- unrelated-vector rescue;
- rescue-specific downstream blocking;
- exact vector and manifest hash validation;
- exact generated token IDs and hashes;
- full intervention metadata;
- resume compatibility checks.

### 7.4 Standalone behavioral runners

Modified:

- `persona_behavior_upstream_contrastive_circuit.py`
- `persona_behavior_steering_circuit.py`

Changes include:

- one full-prefix ungrouped-GQA backend;
- same-backend normal/counterfactual/ablation/circuit-only conditions;
- no prompt truncation;
- source-to-sink validation for circuit-only;
- exact token hashes;
- vector/circuit/model provenance;
- rejection of incompatible historical normal baselines.

### 7.5 New end-to-end scripts

Added:

- `persona_run_end_to_end_eapig.py`
- `persona_evaluate_end_to_end_circuit.py`
- `persona_behavior_end_to_end_circuit.py`
- `persona_analyze_end_to_end_curves.py`

These implement:

- independent natural-prompt attribution;
- source-to-logit circuit construction;
- prompt/response-disjoint internal evaluation;
- held-out normal, negative prompt, ablation, and circuit-only generation;
- topology controls;
- paired internal and behavioral bootstrap analysis.

### 7.6 New split, audit, pooling, and control scripts

Added:

- `persona_freeze_upstream_evaluation.py`
- `persona_freeze_downstream_evaluation.py`
- `persona_audit_token_lengths.py`
- `persona_pool_generation_splits.py`
- `persona_build_circuit_controls.py`
- `persona_compare_score_tables.py`

They enforce:

- prompt-disjoint upstream selection;
- prompt- and response-disjoint downstream selection;
- unique source pair IDs;
- exact answer suffix equality;
- token-cap audits without GPU inference;
- deterministic pooling only under matching causal semantics;
- topology-stratified score permutation;
- raw and complete-path IG stability comparisons.

### 7.7 New analysis scripts

Added:

- `persona_analyze_bottleneck_mediation.py`
- `persona_analyze_internal_circuit_curves.py`

Expanded:

- `persona_analyze_causal_curves.py`

They provide:

- paired bootstrap intervals;
- vector-versus-control contrasts;
- downstream-blocking effects;
- full-prefix exact-match validation;
- optional 2x2 interaction;
- retained-size and topology-control reporting;
- cap and provenance validation.

### 7.8 CLI and configuration

Modified:

- `scripts/rq.py`
- `configs/qwen_sycophancy.json`
- `configs/qwen_hallucination.json`
- `configs/qwen_evil.json`

Added CLI commands for every new freezer, audit, control, end-to-end, pooling,
and analysis script.

Added pinned graph-relevant architecture:

```json
{
  "n_layers": 28,
  "n_heads": 28,
  "parallel_attn_mlp": false
}
```

Historical final-sycophancy commands are relabeled as cut-set history rather
than final authoritative reconstruction.

### 7.9 Tests

`tests/test_persona.py` now includes regressions for:

- cut objects not being mislabeled as connected circuits;
- deterministic equal-score ordering;
- source-to-sink complete-path pruning;
- cut-versus-reconstruction connectivity contracts;
- correct rescue-effect downstream estimand;
- exact generated-token hashing;
- negative-baseline behavioral normalization;
- exact answer suffix checks;
- causal-semantics-aware split pooling;
- pooled-generation mediation provenance;
- unique balanced prompt-disjoint selection;
- coarse topology-control strata;
- prompt-position-restricted edge patching.

### 7.10 What did not change

The repair did not change:

- the canonical Qwen model or revision;
- bfloat16;
- official Persona Vectors source commit;
- source trait JSON;
- 2,000 extraction generations;
- canonical judge and 810 effective pairs;
- response-average vector;
- selected layer 20;
- validated steering result;
- frozen 64 discovery identities;
- numerical contents of the existing upstream IG-5 score table;
- numerical contents of the existing downstream IG-5 score table;
- frozen refusal evidence.

## 8. Reproducibility and artifact contract

### 8.1 Environment

- Python: >=3.10,<3.11
- Dependency source: `pyproject.toml` and `uv.lock`
- Install: `uv sync --locked`
- Entry point: `uv run python scripts/rq.py <command> ...`
- Slurm launcher: `scripts/slurm/rq_job.sbatch`
- Modules:
  - `2025`
  - `CUDA/12.8.0`
- Credentials:
  - `.env`, Git-ignored, mode 600;
  - `HF_TOKEN`;
  - `OPENAI_API_KEY` only before paid judging.

Never put credentials in Git, commands, or Slurm logs.

### 8.2 Canonical paths

Config:

`configs/qwen_sycophancy.json`

Shared discovery manifest:

`data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/shared_circuit_dataset_v1.json`

New upstream evaluation manifest:

`data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/upstream_prompt_disjoint_eval_v1.json`

New downstream/end-to-end evaluation manifest:

`data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/downstream_prompt_response_disjoint_eval_v2.json`

Results root:

`results/qwen2.5-7b-instruct/persona/sycophantic`

Intermediate root:

`data/intermediate/persona/qwen2.5-7b-instruct/sycophantic`

Canonical frozen upstream run:

`sharedprompt_n64_ig5_k3000_20260726`

Canonical frozen downstream run:

`sharedprompt_n64_ig5_full768_h100_cneg2_20260802`

### 8.3 Completion rule

Never infer completion from a directory name.

Require:

- metadata exists;
- `status == "complete"`;
- expected row identities are present exactly once;
- input hashes match;
- model revision matches;
- vector and circuit hashes match;
- values are finite;
- `truncation == false` for teacher-forced rows;
- generation text is coherent;
- cap hits are understood;
- exact token hashes exist;
- judge input hashes match current responses.

### 8.4 Frozen score rule

Within a ranking:

- one score table;
- nested frontiers from that score table;
- no reranking per k;
- deterministic ties;
- preserve immutable metadata.

### 8.5 Generation pooling rule

Rows from different generation splits or caps may be pooled only if:

1. model and revision match;
2. prompt text and question identity match;
3. condition and intervention semantics match;
4. circuit and vector hashes match;
5. decoding is deterministic greedy;
6. reused rows ended naturally before the common lower cap;
7. exact token IDs are complete;
8. split provenance is retained.

Do not pool capped rows as complete outputs.

## 9. What to reuse, rerun, skip, or run conditionally

### 9.1 Reuse without rerunning unless audit fails

- Gemma refusal results.
- Frozen Qwen multilingual refusal results.
- Official Persona Vectors source artifacts.
- 1,000 positive sycophancy extraction generations.
- 1,000 negative sycophancy extraction generations.
- Existing canonical judgments.
- 810 effective response pairs.
- Response-average vector.
- Layer-20 selection.
- Positive steering validation.
- Shared 64-row discovery manifest.
- Existing upstream IG-5 score table.
- Existing downstream IG-5 score table.
- Historical upstream, downstream, and joint cut-set necessity evidence.

### 9.2 Must run

1. artifact and static audit;
2. disjoint evaluation manifest freeze/verification;
3. token-length audit;
4. CPU circuit-family rebuild;
5. technical smoke;
6. cheap source-to-sink decision cells;
7. five-question mediation preflight;
8. IG 5/20/50 convergence for all three estimands;
9. canonical repaired upstream/downstream internal curves;
10. topology controls at selected reconstruction k;
11. full independent end-to-end discovery and held-out curve;
12. q5-q19 primary mediation if preflight passes;
13. q5-q19 end-to-end behavior if its preflight passes;
14. final paired analyses and reporting.

### 9.3 Run only if triggered

- full upstream reranking:
  - only if IG-5 fails convergence;
- full downstream reranking:
  - only if IG-5 fails convergence;
- upstream k=50,000/100,000:
  - only while retained complete-path size is below 5% and faithfulness is
    below 0.85;
- connected source-to-sink circuit-only generation:
  - only when both upstream and downstream pass internal reconstruction;
- recurrent current-token timing:
  - after the prompt-end result is frozen;
- prompt-end-specific downstream attribution:
  - only if vector rescue works but frozen downstream blocking fails;
- hallucination:
  - only after primary sycophancy gates pass;
- second model, factual sycophancy, multilingual persona, closer
  counterfactual:
  - after the primary study is secure.

### 9.4 Skip

- refusal reruns;
- vector extraction reruns;
- layer sweep reruns;
- finishing legacy q7-q19 connected cut-set circuit-only before repaired gates;
- invalid low-memory generations;
- evil experiments;
- additional traits before sycophancy;
- selecting k from official behavioral scores;
- mixing q0-q4 into q5-q19 confidence intervals;
- full IG-200 by default;
- partially position-aware selection under the deadline.

## 10. Exact remaining experiment program

The exact commands are in
`SYCOPHANCY_PERSONA_CIRCUIT_REPAIRED_RUNBOOK.md`. This section is the complete
experiment index and decision dependency graph.

### 10.1 Phase 0: CPU and artifact preparation

Runbook Sections 5-7.

Tasks:

1. switch to the handoff branch;
2. `uv sync --locked`;
3. verify required tools;
4. load `.env` securely;
5. define canonical paths;
6. verify every frozen artifact exists;
7. verify discovery metadata status;
8. compile Python and run `git diff --check`;
9. freeze or verify the 32-row upstream evaluation manifest;
10. freeze or verify the 32-row downstream evaluation manifest;
11. audit caps 256, 350, 500, 640, 768, and 1,024;
12. rebuild `source_to_sink` and `causal_cut` artifacts for all initial k values;
13. inspect retained connectivity.

GPU estimate: 0.

Required output:

- frozen manifests with exact protocols;
- token-length audit JSON;
- derived circuit-family manifests;
- every nonempty source-to-sink artifact has
  `all_edges_on_source_to_sink_paths=true`.

### 10.2 Phase 1: small technical smoke

Runbook Section 8.

Run:

- four-row upstream source-to-sink evaluation;
- four-row downstream source-to-sink evaluation;
- one-question connected conditions:
  - normal;
  - upstream ablated;
  - full-prefix rescue;
  - vector rescue;
  - vector rescue with downstream blocking;
- four-row, IG-1 end-to-end discovery;
- four-row end-to-end evaluation;
- one-question end-to-end behavior.

Estimate: 2-6 A100-hours.

Hard stop:

- normal and full-prefix-rescue exact token hashes must match;
- internal margins must be finite and correctly signed;
- no teacher-forced truncation;
- end-to-end path must produce at least one complete circuit.

### 10.3 Phase 2: cheap IG-5 topology and reconstruction cells

Runbook Section 9.

Run:

- upstream k=3,000, 10,000, 20,000 on 32 new held-out rows;
- downstream k=3,000 and 10,000 on 32 new held-out rows.

Estimate: 2-6 A100-hours.

Purpose:

- determine whether source-to-sink reconstruction is plausible before
  expensive reranking;
- estimate retained complete-path growth;
- verify that downstream remains strong under the corrected split.

Stop upstream circuit-only generation if:

- faithfulness remains below 0.30 after at least 2% retained eligible edges;
- there is no convincing upward trend.

### 10.4 Phase 3: five-question causal-interface preflight

Runbook Section 10.

Run 14 conditions on q0-q4:

- normal;
- upstream ablated;
- downstream ablated;
- both ablated;
- full-prefix rescue;
- full-state rescue;
- vector rescue;
- vector rescue with downstream blocking;
- orthogonal rescue seeds 0-4;
- unrelated-vector rescue.

Estimate: 5-12 A100-hours.

Then:

- inspect every raw answer;
- judge only complete valid artifacts;
- run mediation analysis.

This is a decision pilot, not paper evidence.

Pass direction:

- exact full-prefix equality;
- upstream ablation lowers trait;
- vector rescue moves toward normal;
- controls are weaker;
- downstream blocking reduces rescue;
- coherence is preserved.

If IG-5 later fails convergence, repeat this preflight with the canonical
reranked cuts before q5-q19.

### 10.5 Phase 4: fixed-row IG convergence

Runbook Section 11.

Run the same 16 discovery rows at IG 5, 20, and 50 for:

- upstream;
- downstream;
- end-to-end.

Then evaluate each pilot ranking at k=10,000 on held-out data and compare:

- signed Pearson;
- absolute Spearman;
- top-k Jaccard;
- source-to-sink Jaccard;
- retained edge counts;
- held-out faithfulness.

Estimated A100-hours:

- upstream: 2-6;
- downstream: 8-20;
- end-to-end: 10-26.

Freeze `UP_STEPS`, `DN_STEPS`, and `E2E_STEPS` before full runs.

### 10.6 Phase 5: freeze canonical decomposed rankings

Runbook Section 12.

If a scope retains IG-5:

- reuse its existing full n=64 score table.

If it needs IG-20 or IG-50:

- run one new full n=64 discovery;
- use a new immutable run name;
- rebuild both circuit families.

Conditional estimates:

- upstream full rerun: 3-10 A100-hours;
- downstream full rerun: 8-28 A100-hours.

If either canonical run changes, repeat the q0-q4 mediation preflight on the new
cuts. Do not expand the obsolete IG-5 pilot.

### 10.7 Phase 6: final source-to-sink internal curves

Runbook Section 13.

Upstream:

- evaluate every nonempty complete-path frontier;
- initial frontiers through k=20,000;
- conditionally extend to 50,000 and 100,000.

Downstream:

- evaluate every nonempty complete-path frontier through k=10,000.

Estimated A100-hours:

- upstream curve: 2-5;
- downstream curve: 9-20;
- adaptive upstream extension: 2-6 if triggered.

Select:

- `UP_RECON_K`: smallest tested upstream circuit passing the gate;
- `DN_RECON_K`: smallest tested downstream circuit passing the gate.

Leave a value unset if its family does not pass.

### 10.8 Phase 7: topology-stratified controls

Runbook Section 14.

At each selected reconstructive k:

- build five topology-stratified controls;
- evaluate all five on the same held-out rows;
- aggregate with paired bootstrap intervals.

Do not run controls at every frontier.

Estimated A100-hours: 8-22 across selected families.

### 10.9 Phase 8: full independent end-to-end comparator

Runbook Section 15.

Run:

1. n=64 discovery at frozen `E2E_STEPS`;
2. held-out source-to-sink curve;
3. select smallest passing `E2E_K`;
4. five topology controls;
5. q0-q4 behavior:
   - normal;
   - negative prompt;
   - circuit ablated;
   - circuit only;
6. inspect and judge;
7. expand q5-q19 only if coherent and directionally correct;
8. pool q5-q19 by condition;
9. analyze without q0-q4.

Estimated A100-hours:

- discovery: 10-40;
- internal curve: 8-20;
- behavior: 8-20;
- controls included in the control budget above.

### 10.10 Phase 9: primary q5-q19 mediation

Runbook Section 16.

Run only after:

- canonical rankings are frozen;
- canonical q0-q4 mediation preflight passes;
- any IG-5 reranking has been reflected in a new preflight.

Run q5-q19 for:

- eight main conditions;
- five orthogonal seeds;
- one unrelated-vector control.

Split into five-question batches for memory and resume safety.

Then:

- pool each condition under the 350-token common cap;
- judge each pooled artifact;
- run 10,000 paired bootstrap samples;
- apply the level-2 gate.

Estimated A100-hours for preflight plus primary: 25-60.

### 10.11 Phase 10: connected reconstructive union

Runbook Section 17.

Run only when both reconstruction k values are set.

Preflight q0-q4:

- normal;
- `both_circuit_only`.

Use:

- source-to-sink upstream circuit;
- source-to-sink downstream circuit;
- all-position/current-token construction appropriate for the complete
  autoregressive mechanism.

Estimated A100-hours: 5-14.

Expand only if coherent and trait-preserving.

### 10.12 Phase 11: recurrent timing diagnostic

Runbook Section 18.

After freezing prompt-end results, run q0-q4 with:

- upstream ablated;
- full-state rescue;
- vector rescue;
- vector rescue with downstream blocking;
- all positions/current token.

Estimated A100-hours: 8-22.

Purpose:

- distinguish prompt-formed and recurrent interfaces;
- diagnose downstream timing mismatch.

### 10.13 Phase 12: extensions

Runbook Section 19.

Only after the primary study:

- external opinion-sycophancy set;
- closer instruction-only counterfactual;
- discovery resample/template stability;
- hallucination with adequate behavior;
- second model;
- factual sycophancy;
- multilingual persona transfer.

## 11. Compute and three-day scheduling

All estimates are allocated one-GPU A100-hours, not queue time. Record actual
`sacct` runtimes after every wave.

| Phase | Estimated A100-hours |
| --- | ---: |
| Technical smoke | 2-6 |
| Cheap IG-5 decision cells | 2-6 |
| Five-question mediation preflight | 5-12 |
| Upstream IG convergence | 2-6 |
| Downstream IG convergence | 8-20 |
| End-to-end IG convergence | 10-26 |
| Conditional full upstream rerun | 3-10 |
| Conditional full downstream rerun | 8-28 |
| Final upstream internal curve | 2-5 |
| Final downstream internal curve | 9-20 |
| Adaptive upstream frontier extension | 2-6 |
| Full end-to-end discovery | 10-40 |
| End-to-end internal curve | 8-20 |
| Topology controls | 8-22 |
| Primary mediation generation | 25-60 |
| End-to-end behavior | 8-20 |
| Conditional connected reconstruction | 5-14 |
| Conditional recurrent diagnostic | 8-22 |

Planning totals:

- decision gates: 35-80 A100-hours;
- full suite if current IG-5 rankings converge: 100-210 A100-hours;
- full conditional suite with reranking/extensions: 170-330 A100-hours.

With at least eight independent A100s:

Day 1 morning:

- CPU audit and rebuild;
- technical smoke.

Day 1 afternoon/night, only after smoke:

- cheap IG-5 cells;
- q0-q4 mediation preflight;
- all nine convergence attributions.

Day 2 morning:

- held-out convergence evaluations;
- score comparisons;
- freeze step counts.

Day 2 afternoon/night:

- conditional upstream/downstream reranking;
- canonical internal curves;
- full end-to-end discovery.

Day 3 morning:

- topology controls;
- q5-q19 mediation;
- end-to-end held-out behavior.

Day 3 afternoon/night:

- pool;
- judge;
- analyze;
- run only triggered reconstruction/timing diagnostics.

Do not launch q5-q19 while a ranking or pilot decision remains unresolved.

## 12. Statistical and reporting plan

### 12.1 Unit of resampling

- Internal curves: prompt or response-pair identity.
- Behavior: paired question identity.
- Orthogonal controls: paired questions and the five prespecified directions.

### 12.2 Primary quantities

- additive point effects;
- paired differences;
- 95% bootstrap intervals;
- full-prefix exact-match rate;
- cap-hit rate;
- coherence;
- retained/eligible fraction.

### 12.3 Secondary quantities

- normalized necessity;
- normalized sufficiency;
- rescue fraction;
- edge overlap after boundary mapping;
- raw score correlation.

### 12.4 Pilot separation

- q0-q4 is descriptive and diagnostic;
- q5-q19 is primary for the new conditions;
- do not pool them;
- label both clearly.

### 12.5 Noising and denoising

Report:

- circuit ablation;
- circuit-only reconstruction;
- their prompt-level variance;
- their asymmetry.

Do not infer sufficiency from necessity.

### 12.6 Cap reporting

For every free-generation condition:

- maximum token cap;
- mean, median, percentiles, and maximum generated tokens;
- number and fraction of cap hits;
- sensitivity excluding cap hits where appropriate.

## 13. Publication framing and paper organization

### 13.1 Suggested thesis

> High-level LLM behaviors can admit small causal interception sets that are
> distinct from complete reconstructive circuits. For opinion sycophancy, we
> test whether such an upstream interception set acts through a low-dimensional
> persona-vector interface and a downstream expression computation, with exact
> rescue, specificity controls, downstream blocking, and an independent
> end-to-end comparator.

### 13.2 Suggested contribution list

1. A formation-interface-expression causal design for generative persona
   behavior.
2. A distinction between sink-connected causal cuts and source-to-target
   faithful circuits.
3. Exact interface rescue with full-prefix and full-state controls, orthogonal
   and unrelated-vector controls, and rescue-specific downstream blocking.
4. An independent natural-prompt end-to-end circuit using the same behavioral
   response objective.
5. Frozen refusal evidence showing sparse causal circuits and multilingual
   sharing.

Only include contribution 3 or 4 as an empirical success after its gate passes.

### 13.3 Recommended results sequence

1. Refusal foundation:
   - Gemma;
   - multilingual Qwen;
   - behavior and internal faithfulness.
2. Historical opinion-sycophancy observation:
   - tiny cuts are highly necessary;
   - legacy upstream reconstruction is low.
3. Topological diagnosis:
   - cut versus source-to-sink retained paths.
4. Repaired internal curves:
   - upstream;
   - downstream;
   - controls;
   - IG convergence.
5. Causal interface:
   - ablation;
   - full-prefix;
   - full-state;
   - vector;
   - controls;
   - blocking;
   - 2x2 interaction.
6. Independent end-to-end comparator.
7. Optional connected reconstruction and timing.
8. Limitations and replication.

### 13.4 Claims allowed after each evidence level

After historical cut ablation:

- "A small edge set is necessary for opinion-sycophantic behavior."

After topology controls:

- "The selected cut outperforms topology-matched controls."

After specific rescue and downstream blocking:

- "The upstream cut acts through a low-dimensional causal interface read by the
  selected downstream computation."

After source-to-sink internal faithfulness:

- "The selected circuit faithfully reconstructs the internal estimand."

After coherent circuit-only generation:

- "The selected circuit is behaviorally sufficient under the tested prompts
  and decoding protocol."

After external replication:

- "The mechanism generalizes beyond the development question set."

### 13.5 Claims not allowed

Do not claim:

- high necessity implies sufficiency;
- a parentless edge shell is a complete circuit;
- a readable direction is automatically a natural mediator;
- the architectural residual layer alone is a nontrivial bottleneck;
- one selected k is a unique minimum;
- the current 20 questions are untouched;
- opinion sycophancy is universal sycophancy;
- capped output is equivalent to natural termination;
- raw edge overlap alone establishes mechanistic equivalence;
- unrun repaired experiments are completed results.

## 14. Literature positioning

The bibliography metadata should be verified before submission. The following
are the working anchors used to design the repair.

### Persona Vectors

Chen et al., "Persona Vectors," arXiv:2507.21509.

Provides:

- trait directions;
- monitoring and steering;
- training-shift interpretation.

This project must add:

- the natural edge-level computation that forms and reads the direction;
- direct causal mediation rather than steering alone.

### Have Faith in Faithfulness

Hanna et al., "Have Faith in Faithfulness: Going Beyond Circuit Overlap,"
arXiv:2403.17806.

Provides:

- source-to-target circuit definition;
- direct faithfulness evaluation;
- rationale for pruning parentless and childless nodes.

This project must add:

- high-level free generation;
- persona-vector mediation;
- explicit cut-versus-circuit interpretation.

### Position-aware circuit discovery

Haklay et al., "Position-aware Automatic Circuit Discovery,"
arXiv:2502.04577.

Relevance:

- aggregated edge scores can hide position-specific structure.

Current response:

- document the limitation;
- keep discovery and evaluation objects aligned;
- defer full position-aware extension until the primary result is secure.

### Multiple mediators and circuit variance

Working anchors:

- "The Curse of Multiple Mediators: Hidden Interaction Effects in Transformer
  Interpretability," arXiv:2606.27510.
- Wu et al., "Demystifying Variance in Circuit Discovery of LLMs,"
  arXiv:2606.16920.

Relevance:

- large counterfactuals introduce interaction terms;
- rankings vary across samples and templates.

Current response:

- noising/denoising asymmetry;
- 2x2 interaction;
- IG convergence;
- later closer counterfactual;
- later resample/template stability.

### CircuitSteer

"CircuitSteer," arXiv:2608.05732.

Provides:

- behavior steering through SAE feature-flow circuits, including sycophancy and
  refusal.

This project must add:

- natural-prompt causal mediation;
- circuit-only faithfulness;
- interface rescue and downstream blocking.

### Shared safety and sycophancy circuits

Working anchors:

- "Who Bridges Safety? Identifying and Targeting Cross-Lingual Shared Safety
  Pathways," arXiv:2608.09095.
- "The Shared Sycophancy-Lying Circuit," arXiv:2604.19117.
- "Dissociating the Internal Representations of Sycophancy in LLMs,"
  arXiv:2607.07003.

Relevance:

- shared cross-lingual safety pathways;
- component reuse across related behaviors;
- factual and opinion sycophancy may dissociate.

Current response:

- use refusal for multilingual evidence;
- scope the persona result to opinion sycophancy;
- reserve factual sycophancy as a separate replication.

### Persona-vector emergence and control limitations

Working anchors:

- "Tracing Persona Vectors Through LLM Pretraining," arXiv:2605.13329.
- "Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in
  Language Models," arXiv:2606.24952.

Relevance:

- when persona directions emerge;
- readable directions need not be natural control axes.

Current response:

- study inference-time edge formation/expression;
- require causal rescue and specificity.

### Refusal and persona stage decompositions

"Refusal Lives Downstream of Persona in Chat Models," arXiv:2606.26161.

Provides:

- a related stage-level persona/refusal decomposition.

This project must distinguish itself through:

- edge-level upstream and downstream computations;
- complete-path faithfulness;
- exact interface rescue;
- independent end-to-end comparison;
- held-out natural behavior.

## 15. Risks and honest limitations

### 15.1 Sample size

The q5-q19 primary behavior set has only 15 questions. Use paired intervals and
avoid overstating generality. The external ICLR set is important.

### 15.2 Judge dependence

Trait and coherence scores use a fixed automated judge. Preserve judge version,
prompt hashes, and raw outputs. Manually inspect all pilot responses.

### 15.3 Counterfactual semantics

Positive and negative persona prompts differ substantially. The closer
instruction-only counterfactual is needed as a robustness analysis.

### 15.4 Position aggregation

Current edge scores aggregate positions. The timing and length analyses reduce
but do not remove this limitation.

### 15.5 Sparse threshold

Faithfulness >=0.85 and retained <=5% are operational criteria. Report the full
curve rather than presenting them as natural laws.

### 15.6 Model scope

The repaired persona mechanism is initially tested on one 7B model. Refusal
provides model and language breadth, but ICLR benefits from a direct persona
replication.

### 15.7 Interface uniqueness

Even exact full-prefix rescue does not prove that layer 20 is the only valid
architectural cut. The scientific claim concerns the tested mediator, not a
proof of global uniqueness.

## 16. New-collaborator checklist

Before using a GPU:

- read this document;
- read the repaired runbook;
- inspect `git status`;
- do not discard the uncommitted repair;
- restore Git-ignored canonical artifacts from the shared HPC checkout;
- run the artifact audit;
- verify the two frozen score-table statuses;
- freeze/verify the disjoint manifests;
- run the token audit;
- rebuild both circuit families;
- inspect complete-path counts.

Before expensive jobs:

- pass the technical smoke;
- verify exact full-prefix token equality;
- inspect Slurm logs;
- record `sacct`;
- confirm no truncation.

Before q5-q19:

- freeze IG step decisions;
- freeze canonical run names;
- rerun q0-q4 if a ranking changed;
- inspect all q0-q4 text;
- verify mediation directions.

Before judging:

- inspect raw answers;
- exclude failed, duplicate, capped-as-complete, or provenance-mismatched rows;
- verify judge input hashes;
- use `--resume` only with compatible metadata.

Before writing:

- separate frozen, historical-reinterpreted, and new repaired evidence;
- separate q0-q4 and q5-q19;
- report requested and retained sizes;
- report all five seeds;
- report all cap hits;
- use opinion sycophancy;
- apply the claim checklist.

## 17. Validation already performed

Local static validation completed before this handoff:

- all 67 Bash blocks in the repaired runbook passed `bash -n`;
- every referenced CLI command accepted `--help`;
- `python3 -m compileall` passed;
- `git diff --check` passed.

Local full tests and GPU execution were not performed because the local
environment lacks the complete HPC Python/GPU setup and Git-ignored artifacts.

The next required validation is the HPC technical smoke. Static validation is
not evidence that the scientific interventions work.

## 18. Current repository state

Branch:

`handoff/persona-circuits-20260811`

The working tree contains a large uncommitted repair across:

- core circuit, graph, EAP-IG, model, and persona utilities;
- persona discovery, evaluation, behavior, analysis, and orchestration scripts;
- Qwen persona configs;
- tests;
- this handoff;
- the methodology-repair document;
- the repaired runbook.

Do not:

- run `git reset --hard`;
- restore modified files from HEAD;
- overwrite repaired scripts with older branch versions;
- assume untracked repair scripts are disposable.

Before making additional changes, inspect:

```bash
git status --short --branch
git diff --stat
git diff --check
```

## 19. Final concise decision

The project should not abandon the formation-interface-expression idea.

The correct strategy is:

1. preserve the strong historical cut-set necessity results;
2. stop calling those upstream cut sets faithful formation circuits;
3. test the low-dimensional bottleneck directly through exact controlled
   mediation;
4. separately rebuild complete paths to determine whether a sparse
   reconstructive mechanism exists;
5. run an independent end-to-end circuit as a comparator;
6. make the paper claim match whichever level passes.

This gives three scientifically useful outcomes instead of one brittle
all-or-nothing claim:

- a sparse causal interception result;
- a low-dimensional causal-interface result;
- a complete reconstructive mechanism result.

The immediate operational instruction is to execute
`SYCOPHANCY_PERSONA_CIRCUIT_REPAIRED_RUNBOOK.md` from Section 5 onward and stop
at every stated gate. Do not spend the remaining budget on breadth until the
primary causal estimand is validated.

# Sparse causal circuits for refusal and persona vectors

Authoritative handoff as of 2026-08-11. This records the scientific state,
canonical methods and artifacts, known failure modes, and next decisions.
Generated activations, responses, scores, and most result tables are Git-ignored;
paths below refer to the shared HPC checkout unless stated otherwise.

## 1. Research question and intended contribution

The central hypothesis is that refusal and learned persona traits are mediated
by sparse, causally important edge-level subnetworks rather than being uniformly
distributed across the model. For persona traits we further hypothesize a
formation-expression decomposition around a residual-stream persona-vector
interface:

1. **Localization:** a trait direction is reproducibly localized at a specific
   residual-stream layer and causally controls trait expression under steering.
2. **Sparse causal bottleneck:** a small ranked edge set is necessary for the
   target representation or behavior, and increasing circuit size yields
   interpretable faithfulness/necessity and behavioral curves.
3. **Specificity and generalization:** interventions preserve coherence and
   unrelated behavior, generalize to held-out prompts, and eventually transfer
   across languages and model backbones.
4. **Formation-expression decomposition:** an upstream circuit forms the trait
   coordinate at the interface; a downstream circuit translates that coordinate
   into generated behavior; their connected union tests the complete mechanism.

The direction is an experimentally useful interface, not the claim itself. A
circuit can be behaviorally necessary without being sufficient to recreate the
full vector. Conversely, vector faithfulness alone does not prove control of
free-generation behavior. Internal metrics and behavioral interventions must
therefore be reported separately.

The intended final study is refusal plus several Anthropic Persona Vectors
(sycophancy, hallucination, politeness, and toxicity/evil according to official
trait definitions), first on Qwen2.5-7B-Instruct and then one independent model
no larger than 8B. Multilingual transfer is a primary analysis.

## 2. Reproducibility contract

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16.
- Environment: UV, defined by `pyproject.toml` and `uv.lock`; run `uv sync
  --locked`. SLURM jobs load the 2025 stack and CUDA 12.8 through
  `scripts/slurm/rq_job.sbatch`.
- Credentials belong in Git-ignored `.env`, never `.venv` or a tracked file.
  Use `OPENAI_API_KEY` and `HF_TOKEN`; never put keys in commands or logs.
- Entry point: `uv run python scripts/rq.py <command> ...`. The generic SLURM
  launcher accepts `RQ_COMMAND` and `RQ_ARGS`.
- Canonical datasets, vectors, rankings, circuits, and judged outputs have
  metadata and SHA-256 provenance. Never overwrite a run. Top-k sweeps are
  nested masks derived from one frozen score ranking, not separate rerankings.
- Discovery prompts, response-disjoint internal evaluation pairs, and official
  behavioral questions stay disjoint according to frozen manifests. Generate a
  baseline once and reuse it across the sweep.
- Report retained edges and eligible-edge fractions, sample count, paired
  confidence intervals, coherence, cap hits, and threshold reversals.
- `max_length` truncates only the input prompt; `max_new_tokens` caps output.
  Qwen's context window is 32,768, but connected circuit-only generation on a
  40 GB A100 becomes memory-bound much earlier.

## 3. Method

### Refusal

PolyRefuse covers English, German, Chinese, and Thai. The primary scalar is the
selected residual state's projection onto the normalized English refusal
direction. Refusal EAP-IG uses a clean-to-direction-ablated intervention path and
selects a graph explaining that interface scalar. Faithfulness measures recovery
of the clean-corrupt scalar gap under circuit-only patching; necessity measures
how much circuit ablation reproduces the clean-to-corrupt change. Free generation
then establishes behavioral causality.

Do not interpret the terminal-edge control as showing that upstream edges are
irrelevant. Ablating the terminal shell cuts the only measured conduit into the
endpoint, so terminal-only and full-circuit behavioral ablation can match even
when upstream edges form the signal. Interpret teacher forcing, circuit-only,
multilingual transfer, and specificity jointly.

### Persona-vector extraction

The persona pipeline is scientifically equivalent to Anthropic's public Persona
Vectors code at commit `b8e0f044fe2410a6fad579f38324f03f13b4e917`; model and
generation backend are parameters. Official trait JSON, instructions, questions,
judge prompts, categorical sampling, and layer selection are pinned by hashes.

Sycophancy extraction generated 1,000 positive and 1,000 negative responses: 20
questions x 5 instructions x 10 rollouts per polarity. The canonical judge
retained 810 pairs satisfying positive trait >=50, negative trait <50, and
coherence >=50 on both. At each layer the vector is:

`mean(response-token activations | positive) - mean(response-token activations | negative)`.

The Qwen sycophancy interface is HF hidden state 20, TransformerLens
`blocks.20.hook_resid_pre`; response steering is applied at block 19 under the
official convention. Layer 20 won at every tested positive coefficient 0.5--2.5.
Positive steering strongly raises sycophancy; coefficient -2 is the validated
negative counterfactual used downstream. Projection removal was diagnostically
effective but is not the canonical persona attribution path.

### Upstream formation circuit

The upstream graph contains eligible edges from `blocks.0.hook_resid_pre` through
the interface. Clean/corrupt inputs are positive- and matched negative-persona
system prompts with the same question and assistant suffix. They are strictly
token-aligned. EAP-IG interpolates corrupt graph-input residual state to clean
over five steps. The scalar is final-prompt-token signed projection onto the unit
trait direction. No direction projection ablation is used.

Canonical sycophancy upstream discovery is
`sharedprompt_n64_ig5_k3000_20260726`: 64 frozen prompt identities, 32
response-disjoint internal pairs, 471,191 eligible edges, and one immutable score
table used for all nested k circuits. Its circuit-only metric is representational
sufficiency, not behavioral sufficiency.

### Downstream expression circuit

The downstream graph begins at the interface and ends at logits. High- and
low-trait answers are teacher-forced under the identical positive-persona prompt,
avoiding a nonexistent single "sycophancy token" proxy. The target is mean
log-probability(high-trait answer) minus mean log-probability(low-trait answer).
Coefficient-path EAP-IG follows signed response-time vector removal from -2 to 0
over five steps. It ranks edges mediating vector control of preference for the
two continuations.

Canonical sycophancy downstream discovery is
`sharedprompt_n64_ig5_full768_h100_cneg2_20260802`: the same 64 discovery
identities and 32 internal pairs, paired transcripts up to 768 tokens, 70,157
eligible edges, and one immutable ranking with nested k=50--3000 circuits.

### Connected union

The connected experiment composes independently ranked upstream formation and
downstream expression circuits around the interface. `both_ablated` tests joint
necessity; `both_circuit_only` tests whether the connected sparse union is
sufficient for coherent trait expression. It is not an end-to-end reranking, and
the two k values must be reported separately. The selected sycophancy union is
upstream k=5,000 plus downstream k=3,000; it is not claimed to be a unique
mathematical minimum.

## 4. Frozen Qwen refusal result

Qwen refusal is the most complete section and should remain frozen absent a
reviewer-motivated addition.

- The English 3,000-edge circuit is 1.15% of 260,956 eligible edges. On 32
  harmful prompts: normal refusal 96.9%, direction-ablated 6.3%, circuit-ablated
  15.6%, circuit-only 93.8%; normalized reversal 89.7%, faithfulness 0.960,
  necessity 1.010.
- The frozen English circuit transfers to German, Chinese, and Thai. At k=3,000,
  normalized reversal is 0.852/0.897/0.920/0.880 for
  German/English/Chinese/Thai; at k=5,000 the four-language mean is 0.945.
- A 1,000-edge multilingual consensus circuit has faithfulness
  0.926/0.863/0.904/0.877 and necessity about 1.0 across
  English/German/Chinese/Thai. The strict four-language intersection has only
  706 edges, preserves 72--81% of projection, and is nearly fully necessary.
- Harmless specificity is good: English harmless refusal stays 3.1%; German
  changes +6.3 points, Chinese 0, Thai -3.1 (n=32 each); no harmful responses
  were detected.

Canonical tables:

- `results/qwen2.5-7b-instruct/polyrefuse/phase2_refusal/frozen_development_2026-07-19/`
- `results/qwen2.5-7b-instruct/polyrefuse/phase2_refusal/frozen_controls_2026-07-20/`
- `results/qwen2.5-7b-instruct/polyrefuse/phase2_refusal/frozen_terminal_decision_2026-07-20/`

## 5. Qwen sycophancy result

### Completed evidence

- **Upstream internal, n=32:** necessity is 0.813 at k=500 and 1.002 at k=3,000;
  representational faithfulness is 0.067 at k=3,000, 0.135 at k=5,000, and
  0.193 at k=20,000. A small set is highly necessary but does not reconstruct
  the full vector coordinate.
- **Upstream behavior, n=20:** ablating 50 edges lowers mean sycophancy from
  66.95 to 8.19; k=500 gives 5.00 and k=3,000 gives 4.16. All 16 initially-high
  prompts cross below 50 and coherence remains about 99. This is a sparse
  necessary behavioral bottleneck, not proof that 50 edges are sufficient.
- **Downstream internal, n=32:** faithfulness/necessity rise from 0.719/0.895 at
  k=50 to 0.978/1.003 at k=3,000.
- **Downstream behavior, n=20:** ablation lowers mean sycophancy from 66.95 to
  23.06 at k=50 and 21.55 at k=3,000; reductions are 43.9--46.0 points, coherence
  >=99.4, and 12--14 of 16 high baselines reverse.
- **Selected union necessity, n=20:** upstream k=5,000 plus downstream k=3,000
  lowers mean sycophancy from 66.95 to 1.31, coherence 98.38, with no cap hits.
  The complete judged artifact ends in
  `union_both_ablated_..._max1000_a100_lowmem_v2/`.

Canonical curve inputs are in
`results/qwen2.5-7b-instruct/persona/sycophantic/paper_suite_final_lowmem_v2/`.
Older `paper_suite_rigorous_shared64_eval20` union tables use an earlier
shorter-transcript downstream ranking and are history, not the final selection.

### Exact pending item

Only selected-union circuit-only sufficiency remains unresolved. Earlier
low-memory backend modifications generated repeated exclamation marks; those
outputs are invalid. The restored TransformerLens reference path is coherent,
but a 40 GB A100 OOMed on question 7 at generated token 370.

- Questions 0--6 are coherent, uncapped, and 93--341 tokens in the
  `max500_a100_vanilla_reference_v5` artifact.
- SLURM job `25448716` attempted questions 7--19 with the unchanged reference
  implementation, greedy decoding, input `max_length=1024`, and
  `max_new_tokens=350`, but was cancelled after 25:38 with zero saved rows. No
  sycophancy continuation job was active when this handoff was finalized. Rerun
  only questions 7--19 with the same tag ending in
  `q7-8-...-19_max350_a100_vanilla_reference_v6`.
- Pool the artifacts only because deterministic q0--6 ended before 350 and
  therefore would be identical under the lower cap. Preserve both metadata files
  and disclose split provenance; report cap-hit rate.
- Never use `max1000_a100_lowmem_v2` circuit-only (20/20 cap hits and incoherent
  punctuation), one-row `reference_v3`, or abandoned `max640_a100_final`.
- After completion: verify status/row identities, inspect every response, combine
  q0--6 and q7--19 without regeneration, judge only missing rows with the frozen
  judge, and rebuild the selected-union table/figure. Paid judging requires
  separate authorization.

Length evidence: original sycophantic outputs average 97 tokens (max 235);
helpful outputs average 261, p90 406, p95 446, p99 513, max 574. A 500 cap covers
98.5% of helpful outputs but was not memory-safe. The 350 cap is below the
observed OOM point and covers every source sycophantic output; 20.9% of helpful
source outputs exceed it, so cap hits must be explicit.

## 6. Hallucination and evil status

Hallucination follows the same protocol. Its response-average vector is complete
from 716 effective pairs. The interface is layer 16
(`blocks.16.hook_resid_pre`); layer 16 wins coefficients 1.0--2.5, not 0.5.
Rankings are complete:

- upstream `hallucination_shared_n64_ig5_v1`: 33 token-alignable discovery
  identities and 298,073 eligible edges;
- downstream `hallucination_shared_n64_ig5_cneg_v1_lowmem_v1`: max transcript
  256 and 164,395 eligible edges.

Behavioral upstream/downstream artifacts exist, but strict alignment leaves only
eight official question/instruction pairs (16 rows when normal and intervened are
stored). This is **not yet paper-ready at the sycophancy sample size**. Union and
dependent judge/analysis jobs were active at handoff; inspect `squeue` and
metadata before launching anything. See `HALLUCINATION_PERSONA_CIRCUIT_RUNBOOK.md`.

`configs/qwen_evil.json` and `EVIL_PERSONA_CIRCUIT_RUNBOOK.md` are scaffolding.
No evil vector or circuit result exists; do not describe them as experiments.

## 7. Next work

1. Finish/judge selected sycophancy union circuit-only, inspect cap hits, and
   regenerate its final table/figure. If coherent but low-sycophancy, report a
   strongly necessary but insufficient union; do not tune k on held-out scores.
2. Audit hallucination jobs and the eight-question alignment loss before more
   compute. Correct alignment only if the causal estimand remains unchanged;
   otherwise freeze a larger official held-out rule before evaluation.
3. Apply the finalized formation/expression/union pipeline to remaining official
   traits, freezing manifest, coefficient, interface, and k grid in advance.
4. Add multilingual persona evaluation: validate vector transfer, then test
   frozen source-language upstream/downstream/union circuits across translated
   held-out prompts and k, versus local and matched-random circuits.
5. Replicate the full protocol on one second <=8B instruction model. Add a third
   only after two are complete.
6. For ICLR: matched-random/size controls where absent, paired bootstrap CIs,
   multiple-testing correction, discovery-resample stability/overlap, and a
   secondary end-to-end comparison. Ground claims in held-out generation, not
   internal faithfulness alone.

## 8. Code map and cautions

- `configs/qwen_*.json`: pinned protocols and paths.
- `scripts/persona/persona_extract_response_vector.py`: response-average vector.
- `scripts/persona/persona_freeze_shared_circuit_dataset.py`: shared split.
- `persona_run_upstream_contrastive_eapig.py` / corresponding evaluator:
  formation ranking/evaluation.
- `persona_run_steering_eapig.py` / corresponding evaluator: expression
  ranking/evaluation.
- `scripts/persona/persona_behavior_*`: held-out generation interventions.
- `scripts/persona/persona_behavior_connected_circuits.py`: connected union.
- `src/refusal_circuit/eapig.py`, `graph.py`, `circuit.py`: core primitives.
- `tests/test_persona.py`: protocol/alignment/intervention regressions.

The final-sycophancy orchestration utilities captured on this branch contain
historical artifact tags and are not authoritative for the pending split-cap
run. Prefer Section 5 until they are updated and tested. Never infer completion
from a directory name: require `metadata.status == "complete"`, expected row
identities, current judge input hashes, finite scores, coherent text, and no
unexplained cap hits.

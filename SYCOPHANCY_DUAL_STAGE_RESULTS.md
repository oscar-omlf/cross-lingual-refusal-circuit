# Stage-matched prompt and generation sycophancy vectors

**Status:** Frozen completed experiments  
**Frozen:** 2026-08-24  
**Model:** `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, BF16

## Main result

Sycophancy is strongly affected by two geometrically distinct,
stage-matched coordinates:

- a prompt direction at `blocks.15.hook_resid_pre`;
- a generation direction at `blocks.20.hook_resid_pre`.

Moving either coordinate alone toward its paired anti-sycophantic value had a
partial effect. Moving both by two paired coordinate gaps reduced judged
sycophancy from `81.11` to `35.75`, recovering **92.52%** of the natural
positive-to-negative instruction effect while preserving high coherence
(`90.12`).

This supports complementary prompt-time and generation-time representations.
It does **not** yet prove four distinct circuits or a serial
formation/transmission decomposition.

## Frozen vectors

| Vector | Construction | Data | Selected site |
|---|---|---:|---|
| Prompt | Mean final-prompt-token residual under positive instruction minus matched negative instruction | 256 questions; 512 prompt-only forwards | Layer 15, position −1 |
| Generation | Mean positive response-token residual minus negative response-token residual after trait/coherence filtering | 810 effective pairs from 1,000 pairs | Layer 20 response states |

Prompt selection used 16 held-out `vector_validation` questions, layers 1–27,
and positions −1/−2. Layer 15, position −1 was the best balanced candidate:

| Measure | Result |
|---|---:|
| Projection normalized effect | 83.42% |
| Induction normalized effect | 59.93% |
| Balanced normalized effect | **71.68%** |
| Projection coherence | 96.97 |
| Induction coherence | 94.85 |

The vectors are not interchangeable. Their maximum cross-layer cosine is only
`0.4897` (prompt −1 versus generation, at layer 21).

## Stage-matched free-generation experiment

### Protocol

- Jobs: generation `26005958`; judge `26006086`.
- Data: first 20 `circuit_selection` questions, exactly four from each of five
  instruction pairs.
- Generation: deterministic greedy decoding, 128-token limit.
- Outputs: 8 conditions × 20 questions = 160 answers.
- Judge: Persona Vectors sycophancy and coherence log-probability judges,
  `gpt-4.1-mini-2025-04-14`; 320 API requests.

For stage \(s\), let:

- \(h^+\) be the positive-instruction state;
- \(h^-\) be the paired negative-instruction reference state;
- \(\hat d_s\) be the unit stage-specific direction;
- \(z^+=\hat d_s^\top h^+\) and \(z^-=\hat d_s^\top h^-\);
- \(\alpha\) be the intervention strength.

The positive state was replaced by

\[
h' = h + [z^+ + \alpha(z^- - z^+) - \hat d_s^\top h]\hat d_s.
\]

Thus `1×` reaches the paired negative coordinate and `2×` moves two paired
coordinate gaps. The orthogonal state is preserved.

- Prompt intervention: prompt direction, layer 15, **every prompt token**.
  Each token uses its own token-aligned negative-instruction coordinate.
- Generation intervention: generation direction, layer 20, every cached
  generated-prefix state after first-token initiation. The negative reference
  uses the identical generated prefix.
- Combined intervention: both rules in the same generation.

The prompt vector was extracted at the final prompt token. Applying its
direction at all prompt positions is therefore a deliberate causal
generalization, not an additional extraction result.

### Judged results

For trait score \(S\), normalized reversal is

\[
R=\frac{S_{\mathrm{positive}}-S}
        {S_{\mathrm{positive}}-S_{\mathrm{negative}}}.
\]

| Condition | Sycophancy | Reversal | Coherence | 128-token cap hits |
|---|---:|---:|---:|---:|
| Positive baseline | 81.11 | 0.00% | 87.09 | 17/20 |
| Prompt vector, 1× | 73.92 | 14.67% | 87.03 | 15/20 |
| Generation vector, 1× | 78.19 | 5.95% | 86.84 | 14/20 |
| Both vectors, 1× | 67.93 | 26.88% | 87.66 | 17/20 |
| Prompt vector, 2× | 56.02 | 51.18% | 89.47 | 18/20 |
| Generation vector, 2× | 68.41 | 25.90% | 85.89 | 15/20 |
| **Both vectors, 2×** | **35.75** | **92.52%** | **90.12** | 16/20 |
| Negative baseline | 32.08 | 100.00% | 85.07 | 19/20 |

The combined effect exceeded the sum of the two isolated normalized effects:

| Strength | Prompt + generation separately | Joint result | Excess joint effect |
|---|---:|---:|---:|
| 1× | 20.62% | 26.88% | +6.26 pp |
| 2× | 77.08% | 92.52% | +15.44 pp |

This is suggestive of complementarity or nonlinear interaction. Because the
judge score is nonlinear and bounded, it is not yet evidence of a literal
edge-level circuit interaction.

## Preceding experiments

### 1. Generation-vector geometry and mediation

Job `25984661`; 64 held-out `vector_validation` questions; teacher-forced
stored sycophantic and normal answers.

| Result | Fraction of natural instruction effect |
|---|---:|
| Exact paired generation-direction replacement | 29.73% removed |
| Exact paired orthogonal replacement | 48.51% removed |
| Exact paired full-state replacement | 80.80% removed |
| Reverse direction replacement | 34.49% recovered |

The generation direction therefore carries a genuine but partial natural
effect. Directional and orthogonal effects sum to approximately the local
full-state control; their aggregate non-additivity was only 2.57 percentage
points.

Additional geometry:

| Quantity | Result |
|---|---:|
| Natural positive-minus-negative behavioural margin | 0.3968 |
| Final-prompt instruction coordinate effect | 11.9365 |
| Response-maintenance instruction effect | 5.9516 |
| Response-maintenance answer effect | 10.9855 |
| Maintenance instruction-effect cosine with generation direction | 0.7304 |

Interpretation: the instruction initializes the response-time coordinate, and
generated content increasingly reinforces it.

### 2. Fixed-centroid controls

These 64-question teacher-forced tests established why naive repeated
steering/clamping was unreliable.

| Intervention | Target coordinate | Coordinate reversal | Behavioural reversal |
|---|---:|---:|---:|
| Uncalibrated global fixed clamp, job `26000083` | −19.9379 | 247.1% | 147.0% |
| Calibrated residual clamp, job `26000261` | −11.8842 | approximately 97% | 60.56% |
| Calibrated affine source projection, job `26000703` | layerwise negative centroids | approximately 98% locally | 48.01% |

The first target was too negative and overshot. The calibrated methods reached
natural negative coordinates more faithfully but still did not reproduce the
whole negative-instruction behaviour, showing that one fixed coordinate cannot
replace all stage-specific information.

### 3. Source-coordinate diagnostic

On 64 teacher-forced questions, paired interpolation toward the natural
negative source coordinates produced a smooth dose response:

| Paired-coordinate fraction | Behavioural reversal |
|---:|---:|
| 0.25 | 11.61% |
| 0.50 | 24.00% |
| 0.75 | 36.44% |
| 1.00 | 49.03% |

| Additional source intervention | Behavioural reversal |
|---|---:|
| Fixed source-centroid projection | 68.24% |
| Mean source-coordinate translation | 35.65% |

This motivated using exact paired coordinates rather than an arbitrary global
raw-vector coefficient.

### 4. Earlier same-generation-vector free-generation test

Generation job `26001950`; judge job `26004165`; 20 `vector_validation`
questions. Jobs `26001956` and `26001957` repeated generation on H100 and MIG
hardware for runtime validation.

This experiment used the **generation vector at every source and temporal
stage**, including prompt positions. It did not use the prompt vector.

| Condition | Sycophancy | Reversal | Coherence |
|---|---:|---:|---:|
| Positive baseline | 82.42 | 0.00% | 86.25 |
| Final prompt only, 1× | 81.95 | 0.92% | 89.65 |
| Generation only, 1× | 74.22 | 16.02% | 87.66 |
| Final prompt + generation, 1× | 71.90 | 20.56% | 89.49 |
| All prompt tokens + generation, 1× | 69.36 | 25.51% | 89.16 |
| Final prompt + generation, 2× | 49.68 | 63.98% | 92.57 |
| All prompt tokens + generation, 2× | 55.64 | 52.33% | 88.98 |
| Negative baseline | 31.25 | 100.00% | 85.70 |

Compared with this earlier test, using the corresponding prompt vector at
prompt time and generation vector at generation time produced the strongest
condition: 92.52% reversal rather than 52.33–63.98%. This comparison is
suggestive, not definitive, because the experiments used different 20-question
splits.

## Current interpretation

The minimal hypothesis consistent with all results is:

1. Positive/negative instructions create a prompt-time disposition represented
   partly by the layer-15 prompt direction.
2. During generation, a distinct layer-20 direction represents or maintains
   response-time sycophancy.
3. Either stage can partially sustain the behaviour.
4. Intervening on both stage-matched coordinates nearly reaches natural
   anti-sycophantic behaviour.

A possible circuit decomposition is:

| Candidate object | Proposed role |
|---|---|
| Prompt upstream | Forms the layer-15 prompt coordinate |
| Prompt downstream | Carries prompt-state information into initiation/KV state |
| Generation upstream | Forms or maintains the layer-20 response coordinate |
| Generation downstream | Converts that coordinate into next-token behaviour |

These four objects may overlap heavily and need not be four disjoint circuits.

## What is established

- Prompt and generation vectors are geometrically distinct.
- Their selected layers and causal effects differ.
- Both stage-specific interventions causally reduce judged sycophancy.
- Their combination is stronger than either intervention alone.
- Combined 2× intervention nearly matches the natural negative baseline without
  lowering judged coherence.

## What is not established

- Four distinct or disjoint circuits.
- That the two directions are sufficient mediators of all sycophancy.
- That the observed excess joint effect is mechanistically multiplicative.
- Necessity, circuit faithfulness, or circuit-level rescue/blocking.
- Generalization to the 300-question final behavioural split.
- A clean estimate unaffected by response truncation: 14–19 of 20 answers per
  condition reached the 128-token cap.
- Direct comparability between the earlier and current 20-question experiments,
  because they used different splits.

The current `circuit_selection` pilot must not be reported as final held-out
behavioural evaluation.

## Frozen artifacts

| Experiment | Local artifact |
|---|---|
| Prompt vectors | `results/qwen2.5-7b-instruct/sycophancy/prompt/vector_candidates/` |
| Prompt causal screen | `results/qwen2.5-7b-instruct/sycophancy/prompt/location_screen_induction_1_projection/` |
| Generation vector | `results/qwen2.5-7b-instruct/sycophancy/generation/vector/` |
| Prompt/generation geometry comparison | `analysis/sycophancy_vector_geometry/` |
| Generation mediation geometry | `results/qwen2.5-7b-instruct/sycophancy/generation/geometry_coordinate_n64/` |
| Source-coordinate diagnostic | `results/qwen2.5-7b-instruct/sycophancy/generation/source_projection_diagnostic_n64/` |
| Uncalibrated clamp | `results/qwen2.5-7b-instruct/sycophancy/generation/centroid_clamp_gate_n64/` |
| Calibrated clamp | `results/qwen2.5-7b-instruct/sycophancy/generation/centroid_clamp_gate_calibrated_n64/` |
| Calibrated affine projection | `results/qwen2.5-7b-instruct/sycophancy/generation/centroid_affine_projection_gate_calibrated_n64/` |
| Same-generation-vector generation test | `results/qwen2.5-7b-instruct/sycophancy/generation/paired_affine_behavior_gate_n20/` |
| Stage-matched dual-vector generation test | `results/qwen2.5-7b-instruct/sycophancy/dual_vector_behavior_gate_circuit_selection_n20/` |

Related frozen documents:

- `VECTOR_RESULTS.md`
- `SYCOPHANCY_GENERATION_GEOMETRY_RESULTS.md`
- `SYCOPHANCY_GENERATION_MEDIATION_PLAN.md`
- `SYCOPHANCY_END_TO_END_RESULTS.md`

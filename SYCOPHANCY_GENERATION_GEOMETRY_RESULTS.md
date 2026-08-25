# Sycophancy generation-vector geometry: frozen protocol and results

**Status:** Frozen completed experiment  
**Date:** 2026-08-23  
**HPC job:** `25984661`  
**HPC results:** `/gpfs/home5/omiro1/cross-lingual-refusal-circuit-sycophancy-geometry-20260823/results/qwen2.5-7b-instruct/sycophancy/generation/geometry_coordinate_n64/`

## Question

Does the previously extracted layer-20 generation direction behave as a
natural causal mediator between a sycophantic instruction and sycophantic
answer probabilities?

## Direction tested

The direction was fixed before this experiment:

\[
d_{\mathrm{old}} =
\operatorname{mean}(h_{\mathrm{response}}\mid\mathrm{positive})
-
\operatorname{mean}(h_{\mathrm{response}}\mid\mathrm{negative}).
\]

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Location: `blocks.20.hook_resid_pre`
- Extraction: 810 effective positive/negative generated-response pairs from
  the Persona Vectors protocol.
- Raw norm: 23.7771.
- This direction mixes instruction effects and generated-answer effects.
- The 64 prompts below were not used to define it.

Therefore the experiment tests whether a preselected mean-difference direction
generalizes and is causal. It does **not** test whether this is the optimal,
unique, or most causal possible direction.

## Held-out test data and metric

- Split: `vector_validation`
- Questions: 64
- Stored answers: one sycophantic and one normal answer per question
- Natural transcripts: 256, from the 2 × 2 design below
- No new generation or external judging

| Instruction | Teacher-forced answer |
|---|---|
| Positive | Sycophantic |
| Negative | The same sycophantic answer |
| Positive | Normal |
| Negative | The same normal answer |

The behavioural target was the mean next-token log-probability margin after
the first answer token:

\[
Y=\operatorname{mean}\log P(\text{sycophantic answer})
 -\operatorname{mean}\log P(\text{normal answer}).
\]

## State decomposition and interventions

For a paired positive and negative layer-20 state, let

\[
\Delta h=h_N-h_P,\qquad \hat d=d/\lVert d\rVert .
\]

| Intervention | Replacement state |
|---|---|
| Direction only | \(h_P+(\Delta h^\top\hat d)\hat d\) |
| Orthogonal only | \(h_P+\Delta h-(\Delta h^\top\hat d)\hat d\) |
| Full state | \(h_N\) |

The direction intervention changes one coordinate. The orthogonal intervention
changes the complementary 3,583-dimensional state while preserving the
original coordinate along \(d\). Both use the paired negative-instruction state
with the same answer tokens.

Interventions were applied at response-token states except the final one. The
metric excluded the first answer token so every scored token had an
intervened causal predecessor.

## Natural geometry

### Behaviour

| Condition | Mean margin |
|---|---:|
| Positive instruction | 0.2508 |
| Negative instruction | -0.1460 |
| Natural effect | **0.3968** |
| 95% CI | **[0.3467, 0.4490]** |
| Correct sign | **63/64** |

### Coordinate effects

| Position | Instruction effect | Answer effect | Interaction |
|---|---:|---:|---:|
| Final prompt | **11.9365** | -0.0034 | -0.0159 |
| Response maintenance | **5.9516** | **10.9855** | 0.8253 |
| Entire response | **5.9736** | **10.9942** | 0.7989 |
| Terminal response token | **9.1343** | **9.9997** | -2.1666 |

- The final-prompt instruction effect was positive on 64/64 questions, before
  any answer was present.
- The maintenance instruction effect was positive on 64/64 questions.
- The maintenance instruction-effect vector had cosine 0.7304 with \(d\);
  53.35% of its squared mean-vector norm lay along \(d\).
- The maintenance answer effect was larger than the instruction effect,
  confirming that this generation direction also represents answer
  realization/content.
- Natural coordinate shift and natural behavioural effect correlated
  \(r=0.521\), or \(r=0.326\) after removing instruction-template means.

### Response trajectory

| Response region | Instruction effect | Answer effect |
|---|---:|---:|
| First 20% | 8.2970 | 7.6538 |
| 20–40% | 6.4068 | 10.4727 |
| 40–60% | 5.3040 | 11.4350 |
| 60–80% | 5.1533 | 11.2872 |
| Final 20% | 4.9024 | 13.8305 |

The instruction initializes the coordinate; generated answer tokens
increasingly dominate and reinforce it.

## Causal results

### Positive to negative

| Intervention | Resulting margin | Fraction of natural effect removed | 95% CI |
|---|---:|---:|---:|
| Positive baseline | 0.2508 | 0% | — |
| Half average natural directional shift | 0.2093 | **10.47%** | 9.05–12.08% |
| One average natural directional shift | 0.1643 | **21.80%** | 19.06–25.00% |
| Exact paired directional replacement | 0.1329 | **29.73%** | 26.01–33.47% |
| Exact paired orthogonal replacement | 0.0584 | **48.51%** | 43.01–53.32% |
| Exact paired full-state replacement | -0.0698 | **80.80%** | 77.54–83.68% |
| Negative baseline | -0.1460 | 100% | — |

### Negative to positive

| Intervention | Fraction of natural effect recovered | 95% CI |
|---|---:|---:|
| One average natural directional shift | **26.33%** | 23.31–29.85% |
| Exact paired directional replacement | **34.49%** | 31.15–38.26% |

Additional diagnostics:

- Exact directional replacement moved behaviour correctly on 63/64 prompts.
- Reverse exact replacement moved behaviour correctly on 64/64 prompts.
- Forward and reverse directional effects correlated \(r=0.823\).
- Paired natural coordinate gap and directional causal effect correlated
  \(r=0.604\).
- Directional plus orthogonal effects explained 78.23%; full replacement
  explained 80.80%.
- Their aggregate non-additivity was only 2.57 percentage points.
- Per-prompt summed component effects and full-state effects correlated
  \(r=0.982\).

## Why the effects do not sum to 100%

The component interventions sum to the **full replacement at selected
layer-20 response positions**, not to the entire instruction effect:

\[
29.73\% + 48.51\% \simeq 80.80\%.
\]

The remaining approximately 19.2% can use routes not replaced by this
intervention, including:

- unpatched prompt-position states that later attention can still read;
- first-token/initiation pathways;
- pathways that bypass the selected layer/positions;
- nonlinear interactions and the difference between teacher-forced likelihood
  and free generation.

Thus 80.8% is the local full-state control, not a claim that layer 20 explains
only 80.8% of all sycophancy.

## Supported conclusions

1. The preselected mean-difference direction is naturally moved by the
   instruction on held-out prompts.
2. This movement exists before the answer, so it is not solely a readout of
   sycophantic text.
3. Natural-scale coordinate replacement changes behaviour bidirectionally and
   dose-dependently.
4. The direction is therefore a genuine but partial causal mediator.
5. One scalar coordinate accounts for roughly 30–35% of the measured natural
   effect, which is substantial for one dimension.
6. Other layer-20 information carries a larger combined effect, but it spans
   3,583 dimensions and has not been decomposed.

## Not established

- That \(d_{\mathrm{old}}\) is the best causal direction.
- That it is the first principal component of natural instruction effects.
- That every direction in the orthogonal complement is relevant.
- That the mediator is intrinsically one-dimensional.
- That the reported percentages transfer unchanged to free generation.
- That a formation and transmission circuit compose through \(d\); this
  requires circuit-level rescue and blocking tests.

## Minimal dimensionality experiment

Vanilla PCA of ordinary activations is not appropriate: high variance need not
be trait-related or causal.

Use the 256 `vector_extraction` questions and stored answers:

1. Construct paired natural instruction-effect vectors while holding answer
   tokens fixed:

   \[
   \Delta h_i=\tfrac12[(h_{P,S}-h_{N,S})+(h_{P,N}-h_{N,N})].
   \]

2. Compare three candidate objects:

   - the existing generated-response mean difference \(d_{\mathrm{old}}\);
   - the natural held-answer mean difference
     \(d_{\mathrm{instruction}}=\operatorname{mean}_i\Delta h_i\);
   - an orthonormal residual SVD basis obtained after removing
     \(d_{\mathrm{old}}\) or \(d_{\mathrm{instruction}}\) from every
     \(\Delta h_i\).

3. On the separate 64-question validation split, replace paired coordinates in
   nested subspaces of size \(k=1,2,4,8\), plus a few matched random-direction
   controls.

4. Measure cumulative causal recovery relative to the 80.8% local full-state
   control.

| Possible result | Conclusion |
|---|---|
| \(k=1\) dominates and extra directions add little | Approximately one-dimensional mediator |
| Small \(k\) approaches the full-state control | Low-dimensional mediator subspace |
| PCA/SVD captures natural change but not behaviour | Natural correlates, not causal mediators |
| Many dimensions are required | Distributed layer-20 mechanism |
| \(d_{\mathrm{instruction}}\) beats \(d_{\mathrm{old}}\) | Old vector was diluted by answer-content differences |

If a low-dimensional subspace is found:

- formation attribution should target its natural coordinates;
- transmission attribution should replace the complete paired subspace and
  target the answer-log-probability margin;
- individual direction circuits may be scored separately, but the decisive
  result is the causal effect of the nested subspace;
- the end-to-end circuit remains necessary for bypass pathways.

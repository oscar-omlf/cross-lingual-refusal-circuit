# WiML abstract evidence report

**Status:** 24 August 2026  
**Model:** Qwen2.5-7B-Instruct unless stated otherwise  
**Purpose:** Freeze what we have actually established, distinguish it from hypotheses, and define the smallest defensible paper story.

## Executive conclusion

**Recommended title**

> **Towards Vector-Mediated Circuits for Refusal and Sycophancy in Language Models**

**Paper question**

> When a direction in activation space controls a behavior, does that direction correspond to a faithful and necessary circuit—and is it a natural causal bottleneck for the behavior?

**Main finding**

Activation vectors are useful causal coordinates, but they should not automatically be interpreted as unique scalar bottlenecks. Refusal admits a compact circuit that reconstructs its prompt-level vector, while sycophancy has distinct prompt- and generation-stage coordinates whose joint intervention is much more effective than either alone. Early circuit evidence further suggests that forming a natural trait state may be more distributed than transmitting an already imposed trait direction to output behavior.

This is a **“towards” paper**: it presents a framework, strong case-study evidence, and clear limitations—not a completed universal decomposition theorem.

## Data used

Each trait has **748 unique prompts** with no cross-stage reuse:

| Phase | Refusal | Sycophancy |
|---|---:|---:|
| Vector extraction | 256 | 256 |
| Vector validation/location selection | 64 | 64 |
| EAP-IG edge scoring | 64 | 64 |
| Circuit-size selection | 64 | 64 |
| Final behavior test | 300 | 300 |
| **Total** | **748** | **748** |

- Refusal extraction uses 128 harmful and 128 harmless prompts.
- Sycophancy extraction uses 256 questions under matched sycophantic and anti-sycophantic instructions: 512 prompt-only forward passes.
- The stored Persona Vectors source contains 10,099 unique sycophancy questions; 748 are assigned here and 9,351 remain unused.
- Stored answers and prior model assessments are reused where applicable.

## Experiment ledger

### 1. Refusal vector and interventions

| Experiment | Data | Method | Result | Interpretation |
|---|---:|---|---|---|
| Vector extraction | 128 harmful + 128 harmless | Difference in means; layer 15, `resid_pre`, token −2 | Vector norm 20.09 | Reproduces the Arditi-style prompt-level refusal coordinate |
| Location selection | 32 harmful + 32 harmless validation prompts | Compare causal effect across locations | Layer 15, token −2 selected | Location selected behaviorally, not merely by separability |
| Intervention comparison | 32 harmful + 32 harmless | Baseline, global projection, −1 steering, −2 steering | Refusal prefixes: 31/32, 0/32, 2/32, 0/32; harmless KL: 0, 0.133, 0.373, 1.502 | Projection and −2 steering remove refusal, but projection causes less collateral change |

### 2. Refusal representation circuit

EAP-IG scores edges by how much they reconstruct the refusal-vector projection under global projection corruption.

| Circuit size | Faithfulness | Necessity |
|---:|---:|---:|
| 100 | 53.2% | 89.2% |
| 500 | 78.7% | 99.7% |
| 1,000 | 87.8% | 100.3% |
| 3,000 | 106.2% | 99.6% |

The 1,000-edge circuit is the smallest strong representation circuit.

On 64 development prompts:

| Condition | Refusal rate |
|---|---:|
| Normal model | 93.8% |
| Global vector projection | 6.3% |
| Circuit retained | 95.3% |
| Circuit ablated | 20.3% |

- Raw normalized behavioral reversal: **83.9%**.
- Circuit-only retention: **98.3%**.
- A post-hoc contradiction filter gives 91.5% reversal, but this correction is exploratory and must not replace the raw result.
- These are development results, not the untouched 300-prompt final test.

### 3. Refusal mediation test

This asks whether the selected vector at its nominal bottleneck is itself sufficient and necessary for behavior.

| Condition | Vector projection | Refusal log odds |
|---|---:|---:|
| Normal | 15.79 | 3.89 |
| Global projection | ≈0 | −13.62 |
| Local projection at layer 15, token −2 | ≈0 | 3.59 |
| Circuit ablated | ≈0 | −8.58 |
| Circuit ablated + local coordinate restored | 15.79 | −4.30 |

- Local coordinate removal explains only **1.7%** of the global behavioral effect.
- Restoring the coordinate rescues **34.4%** of the circuit-ablation effect.
- Broad intervention across all prompt positions is behaviorally effective; a one-token intervention is not.

**Supported conclusion:** the circuit strongly represents refusal, but the scalar at one layer/token is not the sole behavioral bottleneck. The behavior may depend on the same direction across positions/layers, orthogonal state, and/or parallel pathways.

### 4. Multilingual refusal

Preliminary English/German/Chinese/Thai experiments target a shared English-derived refusal coordinate.

| Result | Range |
|---|---:|
| Language-specific 1,000-edge faithfulness | 85.6–87.8% |
| English-circuit transfer faithfulness | 77.8–84.4% |
| Pairwise circuit IoU | 0.658–0.764 |
| Full edge-score Pearson correlation | 0.976–0.997 |
| Consensus-circuit faithfulness | 86.3–92.6% |

Matched random circuits have near-zero faithfulness.

**Supported preliminary claim:** refusal-vector computation has substantial cross-lingual functional and structural overlap. This concerns the hidden representation, not yet a complete multilingual behavior-to-logits circuit.

### 5. Prompt-level sycophancy vector

| Experiment | Data | Result |
|---|---:|---|
| Extraction | 256 matched questions; positive and negative instructions | Difference in means at every layer and token −1/−2 |
| Held-out location screen | 16 validation questions | Layer 15, token −1 selected |
| Global projection | Same screen | Trait score 29.23; coherence 96.97 |
| +1 induction | Same screen | Trait score 56.68; coherence 94.85 |

Baseline positive and negative scores are 82.05 and 18.74. Projection therefore removes 83.4% of the positive-to-negative gap; induction moves 59.9% in the opposite direction.

Prompt token −1 and −2 vectors are related but not identical: peak cosine 0.797.

### 6. Generation-level sycophancy vector

- Persona Vectors-style response-token difference in means.
- Approximately 1,000 positive and 1,000 negative judged responses supplied the source pool; filtering leaves **810 effective pairs**.
- Layer 20 was selected through behavioral steering.
- Vector norm: 23.78.
- The location/strength sweep generated and judged **7,000 answers**: seven layers × five coefficients × 200 answers.

| Coefficient at layer 20 | Sycophancy | Coherence |
|---:|---:|---:|
| 0.5 | 8.98 | 99.46 |
| 1.0 | 20.15 | 99.23 |
| 1.5 | 63.24 | 96.78 |
| 2.0 | 91.56 | 85.86 |
| 2.5 | 98.44 | 39.99 |

This confirms causal control but also shows that raw coefficients can leave the natural data manifold.

### 7. Prompt and generation vectors are distinct

| Comparison | Maximum cosine across tested layers |
|---|---:|
| Prompt token −1 vs generation | 0.490 |
| Prompt token −2 vs generation | 0.378 |

This is not explained by a simple layer shift. The vectors encode related but substantially different stages of sycophantic behavior.

### 8. Natural geometry of the generation vector

The clean factorial design holds the question fixed while varying:

1. the instruction: sycophantic or anti-sycophantic;
2. the stored answer: sycophantic or anti-sycophantic.

It uses 64 held-out questions and 256 stored instruction-answer combinations.

| Measurement | Result |
|---|---:|
| Natural answer log-probability margin: positive instruction | 0.251 |
| Natural answer log-probability margin: negative instruction | −0.146 |
| Correct natural sign | 63/64 |
| Generation-coordinate instruction contribution during response | 5.95 |
| Answer-content contribution during response | 10.99 |
| Interaction | 0.83 |
| Maintenance-effect cosine with generation vector | 0.730 |
| Squared maintenance effect along the vector | 53.4% |

**Interpretation:** the instruction initializes the state; generated content then reinforces it. The selected generation direction captures a major, but not complete, part of this maintenance effect.

Natural state replacement decomposes the local causal effect:

| Replacement | Natural effect removed/recovered |
|---|---:|
| Mean direction | 21.8% |
| Exact paired direction | 29.7% |
| Reverse-direction restoration | 34.5% |
| Orthogonal state | 48.5% |
| Direction + orthogonal state | 78.2% |
| Full state | 80.8% |
| Non-additivity | 2.6 percentage points |

**Supported conclusion:** the mean-difference direction is a genuine partial natural mediator. It is neither arbitrary nor sufficient; important information remains in orthogonal state and bypassing pathways.

### 9. Corruption-method diagnostics

These tests were needed because EAP-IG requires a meaningful clean-to-corrupt path.

| Corruption | Coordinate change | Behavioral change toward negative condition | Conclusion |
|---|---:|---:|---|
| Fixed repeated raw clamp to −19.94 | 247.1% | 147.0% | Severe overshoot |
| Calibrated residual clamp to −11.88 | ≈97% | 60.6% | Correct coordinate scale, incomplete behavior |
| Layerwise affine replacement using negative centroids | ≈98% locally | 48.0% | Matching one coordinate at every layer is still insufficient |
| Exact paired-source replacement, full dose | Natural paired displacement | 49.0% | Cleanest natural-scale source intervention |
| Fixed negative-source centroid | Natural centroid target | 68.2% | Stronger, but less example-specific |

Paired-source doses of 0.25, 0.50, 0.75, and 1.00 produce 11.6%, 24.0%, 36.4%, and 49.0% behavioral movement. The approximately monotonic curve supports using paired natural coordinates rather than arbitrary raw coefficients for circuit discovery.

### 10. Joint prompt- and generation-stage intervention

An initial diagnostic reused the **generation vector at both prompt and generation stages**. Generation-only intervention was more effective than prompt-only intervention, and combined 2× intervention moved 64.0% of the natural gap, but many judge scores hit the scale ceiling. Because the prompt representation was mismatched, this run is diagnostic rather than final evidence.

Twenty held-out circuit-selection questions were freely generated and externally judged. Each stage used its corresponding vector.

| Condition | Sycophancy | Normalized change toward anti-sycophancy | Coherence |
|---|---:|---:|---:|
| Positive baseline | 81.11 | 0% | — |
| Prompt vector, 1× | 73.92 | 14.7% | 87.03 |
| Generation vector, 1× | 78.19 | 6.0% | 86.84 |
| Both, 1× | 67.93 | 26.9% | 87.66 |
| Prompt vector, 2× | 56.02 | 51.2% | 89.47 |
| Generation vector, 2× | 68.41 | 25.9% | 85.89 |
| Both, 2× | 35.75 | 92.5% | 90.12 |
| Natural negative baseline | 32.08 | 100% | 85.07 |

**Supported pilot conclusion:** prompt and generation coordinates make complementary causal contributions. Joint 2× intervention nearly reaches the natural anti-sycophantic baseline while retaining judged coherence.

Limitations: only 20 questions; bounded judge scores make the apparent super-additivity descriptive rather than a clean interaction estimate.

### 11. Sycophancy circuit experiments

#### Natural end-to-end circuit

The valid end-to-end task contrasts positive and negative instructions and targets the stored-answer log-probability margin.

| Circuit size | Faithfulness | Necessity |
|---:|---:|---:|
| 500 | −0.3% | 100.0% |
| 1,000 | 0.2% | 99.0% |
| 3,000 | 5.1% | 99.6% |
| 5,000 | 9.7% | 99.9% |
| 10,000 | 16.9% | 101.1% |

High necessity with low faithfulness means the selected edges are disruptive when removed but are insufficient to reproduce the clean computation.

#### Legacy generation-vector upstream circuit

The legacy experiment had low faithfulness, but its generation vector was targeted at a prompt-last location. It therefore mixes a response-derived vector with a pre-generation target and is **not valid final evidence about generation-vector formation**.

#### Legacy downstream circuit

The downstream experiment imposed the generation vector at layer 20 and scored its transmission to an answer-margin target.

| Circuit size | Faithfulness |
|---:|---:|
| 500 | 90.7% |
| 1,000 | 94.1% |
| 3,000 | 97.8% |

This is strong evidence that an already imposed direction can be transmitted by a compact circuit. It is not proof that the natural behavior is fully mediated by that direction: the target is artificial and stronger than the natural contrast, and evaluation reused a small identity set.

#### Correct prompt-vector formation EAP-IG

The current prompt experiment uses exact paired-coordinate replacement at every prompt token:

- 64 reserved EAP-IG prompts;
- layer 15, prompt vector;
- five integration steps;
- clean projection 14.99;
- corrupted projection −0.086;
- natural negative reference −0.090;
- all 64 paths have the correct and monotonic direction;
- 260,956 scored edges;
- top 3,000 edges contain 71.5% of absolute attribution mass.

The path is curved because the intervention affects both upstream sources and the target coordinate. This is an intervention path, not evidence for two circuits by itself.

**Running:** held-out faithfulness and necessity of the 3,000-edge prompt circuit (job `26006976`).

#### Correct generation-vector formation EAP-IG

The natural paired-coordinate generation experiment is pending on H100 (job `26006461`). The A100 version exhausted memory. No scientific conclusion should yet be drawn from this run.

## What is genuinely novel

| Claim | Relationship to prior work | Current status |
|---|---|---|
| A steering vector need not be a unique natural causal bottleneck | Extends beyond work showing that directions control behavior | Strongly supported for refusal; quantitatively supported for generation sycophancy |
| Prompt- and generation-stage sycophancy use distinct, complementary directions | Persona Vectors primarily provides response-derived trait directions | Supported; final large held-out behavior test remains |
| Exact paired-coordinate interventions distinguish natural-scale mediation from arbitrary steering | Addresses overshoot and off-manifold corruption in vector/circuit analysis | Supported methodologically and geometrically |
| Instruction initializes a sycophantic response state and answer content reinforces it | Adds temporal/stage structure to a static persona-vector account | Supported on 64 held-out stored-answer factorial examples |
| Refusal has a compact, highly faithful and necessary representation circuit | Related refusal circuits exist, so “first refusal circuit” is not defensible | Strongly supported on development data |
| Refusal-vector computation substantially overlaps across languages | Goes beyond a single-language vector/circuit account | Strong preliminary evidence |
| Natural trait formation may be distributed while transmission of an imposed direction is compact | A proposed bridge between vectors and circuits | Suggestive only; correct generation formation and held-out prompt-circuit evaluations are pending |
| Joint prompt and generation interventions can approximate full behavioral reversal | Connects distinct stage representations to causal behavior | Strong pilot, not final result |

## Comparison with literature

| Literature | Established result | Our addition |
|---|---|---|
| Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction* | A harmful–harmless direction can induce or suppress refusal; global directional ablation is causal | The direction has a compact representation circuit, but one-token local ablation/rescue shows that control does not imply a unique scalar bottleneck |
| Chen et al. (2025), *Persona Vectors* | Response-derived directions monitor and control traits including sycophancy | Prompt and generation directions are distinct; natural factorial interventions quantify initialization, maintenance, orthogonal state, and joint stage effects |
| Panickssery et al. (2023), contrastive activation addition | Sycophancy can be changed by activation steering | We study natural-scale mediation and the circuits that form/transmit the directions, not steering alone |
| Syed et al. (2024), EAP-IG; Hanna et al. (2024), activation patching | Attribution and integrated gradients can discover circuits; intervention design changes interpretation | We make the corruption explicitly coordinate-aware and test faithfulness, necessity, behavior, and mediation separately |
| Konen et al. (2024), *There is More to Refusal than Meets the Eye* | Refusal may not be universally one-dimensional across concepts | Our results give a complementary causal caveat: even a powerful direction may not be the only local mediator |
| Acharya et al. (2025) and Duan et al. (2025), refusal-circuit work | Refusal circuitry can be localized and edited | We should not claim the first refusal circuit; our focus is the relationship between a vector, its circuit, natural mediation, and cross-trait differences |
| Paranjape et al. (2026), *The Internal Origins of Sycophancy* | Sycophancy can be localized to late residual/MLP computations | Our distinctive contribution is stage-specific vectors, natural coordinate geometry, and formation/transmission/end-to-end decomposition |
| Tayal et al. (2026), *Off-the-Shelf Persona Vectors* | Persona geometry can be multidimensional; one vector may not fully specify a persona | Our interventions causally separate the selected direction, orthogonal state, and bypass while relating them to circuit discovery |

## Claims we must not make

- A single vector fully explains refusal or sycophancy.
- The selected layer/token is a unique scalar bottleneck.
- We have found a complete end-to-end sycophancy circuit.
- We have proven distinct formation and transmission subgraphs.
- Prompt and generation circuits are disjoint.
- The 20-question sycophancy pilot is a final generalization result.
- High necessity alone means a circuit is faithful.
- The legacy prompt-last generation experiment is a valid generation-formation result.
- We are the first to find a refusal or sycophancy circuit.
- The post-hoc refusal label correction is the primary result.

## Recommended paper story

1. **Problem:** activation vectors are often treated as mechanistic explanations because steering them changes behavior.
2. **Gap:** controllability does not establish natural mediation, nor identify the circuit computing or using the representation.
3. **Framework:** separately test:
   - **formation:** input → trait coordinate;
   - **transmission:** imposed coordinate → output;
   - **end-to-end computation:** input → output;
   - **mediation:** remove and restore the natural coordinate.
4. **Refusal case:** compact vector-reconstruction circuit and substantial behavioral effect, but no unique one-token scalar bottleneck.
5. **Sycophancy case:** distinct prompt and generation coordinates, partial natural mediation, and strong joint causal control.
6. **Emerging synthesis:** the relationship between vectors and circuits depends on temporal scope and representation geometry. A compact steering direction may summarize a distributed natural computation while still being transmitted through a sparse downstream pathway.

## Candidate WiML abstract

> Activation steering directions can causally control language-model behavior, but it remains unclear whether they identify the mechanisms that naturally produce that behavior. We study this question for refusal and sycophancy in Qwen2.5-7B-Instruct by separating three objects: circuits that form a trait representation, circuits that transmit an imposed representation to output behavior, and end-to-end behavioral circuits. We combine difference-in-means trait directions, natural-scale coordinate interventions, EAP-IG edge attribution, and tests of faithfulness, necessity, behavioral effect, and mediation. For refusal, a 1,000-edge circuit reconstructs 87.8% of a prompt-level refusal coordinate and has 100.3% normalized necessity; however, removing the coordinate at its nominal layer and token has little behavioral effect, showing that steering control need not imply a unique scalar bottleneck. For sycophancy, prompt- and generation-stage directions are distinct, with maximum cross-stage cosine below 0.49. Factorial interventions show that the generation direction is a genuine but partial natural mediator, while a pilot using both stage-matched directions closes 92.5% of the judged sycophancy-score gap to the anti-sycophantic baseline with preserved coherence. Preliminary circuit results suggest that transmitting an imposed direction can be sparse even when the natural end-to-end computation is not. These findings motivate vector-mediated circuits as a framework for connecting representation engineering with mechanistic explanations without assuming that a controllable direction fully constitutes a behavior.

## Literature identifiers

- Arditi et al., 2024 — arXiv:2406.11717.
- Chen et al., 2025 — arXiv:2507.21509.
- Panickssery et al., 2023/2024 — arXiv:2312.06681.
- Syed et al., 2024 — arXiv:2403.05328.
- Hanna et al., 2024 — arXiv:2403.17806.
- Konen et al., 2024 — arXiv:2407.02949.
- Acharya et al., 2025 — arXiv:2506.15250.
- Duan et al., 2025 — arXiv:2510.07921.
- Paranjape et al., 2026 — arXiv:2603.12748.
- Tayal et al., 2026 — arXiv:2601.10387.

## Minimum remaining work before a paper

1. Finish the held-out prompt-vector circuit faithfulness/necessity evaluation.
2. Finish the correct natural generation-vector formation EAP-IG and circuit evaluation.
3. Run the untouched 300-prompt behavior tests for the final selected interventions/circuits.
4. Replicate the central results with uncertainty estimates and, ideally, one additional model.
5. Treat the formation/transmission distinction as a hypothesis unless both natural formation circuits and downstream mediation tests support it.

## Source files audited

- `REFUSAL_FINALIZATION_PROTOCOL.md`
- `SYCOPHANCY_DUAL_STAGE_RESULTS.md`
- `SYCOPHANCY_GENERATION_GEOMETRY_RESULTS.md`
- `SYCOPHANCY_GENERATION_MEDIATION_PLAN.md`
- `SYCOPHANCY_END_TO_END_RESULTS.md`
- `SYCOPHANCY_END_TO_END_PLAN.md`
- `VECTOR_RESULTS.md`
- `analysis/sycophancy_vector_geometry/REPORT.md`
- `data/processed/qwen2.5-7b-instruct/trait_circuits/manifest.json`
- `results/qwen2.5-7b-instruct/refusal/final_v1/en/EXPERIMENT_RECORD.md`
- `results/qwen2.5-7b-instruct/refusal/downstream_v1/en/CAUSAL_MEDIATION_NOTE.md`
- `old_results/qwen2.5-7b-instruct/polyrefuse/RQ2_english_vector_results_summary.md`

Legacy Gemma refusal runs and generic EAP-IG benchmark artifacts are method-development evidence, not part of the proposed paper claims.

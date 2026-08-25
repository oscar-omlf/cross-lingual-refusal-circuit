# Project status: vector-mediated circuits for refusal and sycophancy

**Date:** 25 August 2026  
**Model:** Qwen2.5-7B-Instruct  
**Provisional title:** *Towards Vector-Mediated Circuits: When Steering Directions Do—and Do Not—Explain Refusal and Sycophancy*

## One-paragraph story

Steering vectors can control behavior, but this does not prove that a behavior is naturally mediated by one scalar bottleneck or one sparse circuit. We therefore separate four questions: **formation** (input → trait coordinate), **transmission** (imposed coordinate → logits), **end-to-end computation** (input → logits), and **mediation** (does removing/restoring the natural coordinate remove/rescue behavior?). Refusal follows an Arditi-style setup: one prompt-derived direction controls behavior when applied broadly across positions/generation, yet its coordinate at one layer/token is not the sole bottleneck. Persona Vectors instead derives sycophancy directions from generated responses. Testing both approaches revealed distinct prompt- and generation-stage sycophancy coordinates: each has a partial effect, while their joint intervention nearly reaches anti-sycophantic behavior. The selected coordinates have sparse representation/transmission circuits, whereas natural end-to-end sycophancy is much more distributed. This is a strong **“towards” paper**, not yet a complete circuit decomposition.

![Stage-matched sycophancy interventions](figures/wiml_core_results.svg)

**WiML recommendation:** use only this compact plot. The call permits at most
one page including figures and references; additional plots would crowd out the
problem, method, and circuit result.

## Literature and gap

| Work | Established result | Our question/addition |
|---|---|---|
| Arditi et al. (2024), arXiv:2406.11717 | A harmful–harmless residual direction can induce/suppress refusal; global directional ablation is causal | Does that coordinate have a faithful circuit, and is it a natural bottleneck? |
| Panickssery et al. (2023), arXiv:2312.06681 | Contrastive mean activations steer high-level behavior, including sycophancy | Separate controllability from natural mediation |
| Chen et al. (2025), arXiv:2507.21509 | Response-derived persona vectors monitor/control traits such as sycophancy | Distinguish prompt and generation representations and their circuits |
| Hanna et al. (2024), arXiv:2403.17806 | EAP-IG efficiently attributes behavior to model edges | Make the EAP-IG corruption path trait-coordinate-aware; test faithfulness, necessity, behavior, and mediation separately |
| Wang et al. (2025), arXiv:2508.02087 | Sycophancy has identifiable internal mechanisms/circuits | Relate stage-specific representations to formation, transmission, and end-to-end circuits |

## Data

Each trait has **748 disjoint assigned prompts**:

| Phase | Refusal | Sycophancy |
|---|---:|---:|
| Vector extraction | 256 | 256 |
| Vector validation | 64 | 64 |
| EAP-IG scoring | 64 | 64 |
| Circuit selection | 64 | 64 |
| Final behavior | 300 | 300 |

- Refusal extraction: 128 harmful + 128 harmless.
- Sycophancy prompt extraction: 256 questions × positive/negative instructions = 512 prompt forwards.
- Sycophancy source contains 10,099 questions; 9,351 remain unused.
- Exception: the legacy generation vector used 810 effective judged response pairs from 1,000 candidate pairs.
- Confirmatory caveat: 88 refusal final identities appeared in archived
  exploratory artifacts, and 16 sycophancy final identities were touched by an
  invalid pilot. Final reporting must disclose this or freeze replacements.

## Main preliminary results

### 1. Refusal: sparse representation circuit, distributed mediation

| Experiment | Data | Result |
|---|---:|---|
| Vector extraction/selection | 256 extraction; 64 validation | Layer 15, token −2; global projection changes harmful refusal prefixes from 31/32 to 0/32 with harmless KL 0.133 |
| Representation circuit | 64 EAP-IG + 64 evaluation prompts | k=1,000: **87.8% faithfulness, 100.3% necessity** |
| Development behavior | 64 harmful prompts; 256 generations | Normal 93.8%, circuit-only 93.8%, circuit-ablated 9.4%, projected 1.6%; filtered reversal 91.5% (raw 83.9%) |
| Local mediation | 16 reserved harmful prompts | Removing the coordinate once explains only 1.7% of global behavioral change; restoring it after circuit ablation rescues 34.4% [20.7%, 49.8%] |
| Multilingual exploratory result | 32 held-out prompts/language | 1,000-edge consensus faithfulness 86.3–92.6% across English/German/Chinese/Thai; necessity ≈100%; matched random circuits near zero |

The extraction/circuit target is one location, but the successful behavioral
interventions are broad:

- global projection removes the direction at every layer and token: 0/32
  refusal prefixes;
- layer-15 negative steering remains active at every prompt and generated
  token: 2/32 refusals at −1× and 0/32 at −2×;
- exact projection only at layer 15, token −2 barely changes refusal.

**Interpretation:** the circuit strongly reconstructs a refusal representation, but refusal is not mediated only by that scalar at one token. The direction likely follows a trajectory across positions/layers/generation or acts alongside additional state. Exact all-prompt-token projection is queued; therefore broad steering shows control, not yet natural prompt-wide mediation.

### 2. Sycophancy: distinct prompt and generation coordinates

| Experiment | Data | Result |
|---|---:|---|
| Prompt vector | 256 extraction; 16 location-screen questions | Layer 15, token −1; projection removes 83.4% of the positive→negative trait-score gap |
| Generation vector | 810 effective response pairs | Layer 20; behaviorally selected from 7 layers × 5 strengths × 200 answers = 7,000 judged answers |
| Cross-stage geometry | Fixed vectors | Maximum prompt–generation cosine **0.490**: related but clearly distinct |

The first scope experiment used the **response-derived generation vector at
both stages**. On 20 validation questions, moving only the final prompt state
closed 0.9% of the natural behavioral gap; generation-only closed 16.0%;
prompt+generation closed 20.6% at 1× and 64.0% at 2×. This motivated extracting
a genuine prompt-stage vector rather than reusing the response vector.

The stage-matched experiment used 20 separate circuit-selection questions,
8 conditions, 160 generated answers and 320 judge calls:

| Condition | Sycophancy | Natural gap closed | Coherence |
|---|---:|---:|---:|
| Positive baseline | 81.11 | 0.0% | 87.09 |
| Prompt vector only, 1× | 73.92 | 14.7% | 87.03 |
| Generation vector only, 1× | 78.19 | 6.0% | 86.84 |
| Both vectors, 1× | 67.93 | 26.9% | 87.66 |
| Prompt vector only, 2× | 56.02 | 51.2% | 89.47 |
| Generation vector only, 2× | 68.41 | 25.9% | 85.89 |
| **Both vectors, 2×** | **35.75** | **92.5%** | **90.12** |
| Natural negative baseline | 32.08 | 100% | 85.07 |

The prompt vector was extracted/targeted at layer 15, final prompt token −1,
but its behavioral intervention and EAP-IG corruption act at **every prompt
token**. The generation vector acts at layer 20 on every generated-prefix
state. The prompt-vector circuit still targets the final-token coordinate.
A direct one-token-versus-all-prompt-token comparison remains unfinished.

**Interpretation:** prompt-time disposition and response-time maintenance are complementary causal representations, not interchangeable measurements of one vector. Unlike refusal, sycophancy currently requires two stage-matched directions for near-complete reversal.

### 3. Generation-vector geometry: genuine but partial mediation

On **64 held-out questions / 256 teacher-forced transcripts**:

| Intervention at layer-20 response states | Natural effect removed |
|---|---:|
| Exact paired generation-direction replacement | 29.7% |
| Orthogonal-state replacement | 48.5% |
| Direction + orthogonal state | 78.2% |
| Full-state replacement | 80.8% |

- Instruction effects align with the generation vector (cosine 0.730); generated content increasingly reinforces the coordinate.
- Direction and orthogonal effects are nearly additive (2.6 percentage-point residual).
- Naive −2 steering equals ~8 natural coordinate gaps and overshoots; exact paired-coordinate interventions give a smooth 11.6%, 24.0%, 36.4%, 49.0% dose response at ¼, ½, ¾, and full dose.

**Interpretation:** the mean direction is a real causal mediator, but not the whole state. This motivates natural-coordinate corruption rather than arbitrary steering coefficients.

### 4. Circuit evidence: sparse coordinates, distributed natural behavior

All rows use **64 EAP-IG prompts + 64 disjoint circuit-selection prompts** unless marked legacy.

The prompt-coordinate circuit uses the positive prompt plus matched-coordinate
source corruption at every prompt token; the negative prompt defines the target
coordinates but is not itself executed as the corrupt graph input. It therefore
establishes sparse coordinate reconstruction, not yet unique natural formation.

| Circuit target | Size | Graph fraction | Faithfulness | Necessity |
|---|---:|---:|---:|---:|
| Prompt sycophancy coordinate | 100 | 0.038% | **89.8%** | **96.6%** |
| Prompt sycophancy coordinate | 500 | 0.192% | 94.9% | 97.9% |
| Prompt sycophancy coordinate | 1,000 | 0.383% | 97.5% | 98.5% |
| Natural end-to-end sycophancy | 10,000 retained 9,776 | 1.04% | 16.9% | 101.1% |
| Natural end-to-end sycophancy | 50,000 retained 49,911 | 5.33% | 67.9% | 100.4% |
| Natural end-to-end sycophancy | 100,000 retained 99,979 | 10.68% | **76.8%** | **100.2%** |
| Legacy imposed generation-vector → logits | 500–3,000 | — | 90.7–97.8% | — |

![Coordinate-circuit versus end-to-end sparsity](figures/circuit_sparsity.svg)

The legacy downstream result used 64 EAP-IG records and 32 evaluation records with reused identities and an artificial imposed-vector contrast; it is suggestive, not final.

**Central synthesis:** compact circuits can reconstruct or transmit a selected low-dimensional coordinate, while naturally producing the full behavior may require a much broader computation. High necessity without faithfulness in small end-to-end circuits means the selected edges are important when removed but insufficient when retained alone.

## Novel claims we can currently defend

1. **Control ≠ bottleneck:** a direction can strongly steer behavior without being the unique local natural mediator.
2. **Stage-specific sycophancy:** prompt and generation directions are distinct and jointly more effective than either alone.
3. **Natural-coordinate methodology:** paired coordinate replacement exposes overshoot and provides an EAP-IG-compatible, natural-scale corruption.
4. **Sparsity depends on the target:** trait-coordinate circuits can be extremely sparse even when natural end-to-end behavior is distributed.
5. **Preliminary multilingual structure:** refusal-representation circuits strongly overlap and transfer across four languages.

We must **not** yet claim a complete end-to-end circuit, four disjoint sycophancy subcircuits, a universal formation→vector→transmission theorem, or final generalization beyond Qwen2.5-7B.

## Currently queued

At the last successful HPC check, no job was executing; the following jobs
were queued for GPUs:

| Work | Planned data/status |
|---|---|
| Correct prompt-sycophancy circuit behavior + judge | 16-prompt pilot; pending |
| Generation-vector formation EAP-IG + k=100/500/1,000/3,000 evaluation | 64 EAP-IG + 64 evaluation questions; H100 pending |
| Natural end-to-end sycophancy k=10k behavior | 16-prompt pilot; pending |
| Paired-coordinate end-to-end sycophancy EAP-IG/evaluation/behavior | 64 + 64 + 16; pending |
| Refusal final behavior | 95/300 prompts generated; remaining 205 and WildGuard pending |
| Refusal end-to-end EAP-IG/evaluation | 64 + 64; pending |
| Refusal downstream and scope diagnostics | Exploratory; pending |

The partial refusal final run currently has only prefix diagnostics on 49 harmful prompts: normal 85.7%, circuit-only 87.8%, circuit-ablated 0%, global projection 0%. This is **not** a final result until all 300 prompts and WildGuard judging finish.

## Submission assessment

The project is worth developing as a **carefully scoped “towards” paper**. Its strongest story is not “we found the circuit,” but:

> **Steering directions are useful causal coordinates, yet the circuit that reconstructs a coordinate can be far sparser than the natural end-to-end computation producing behavior. Refusal and sycophancy reveal different geometries and temporal scopes, requiring explicit tests of formation, transmission, mediation, and behavior.**

Priority before a full submission: finish the pending prompt/generation behavioral and circuit evaluations, add matched-random controls and uncertainty, preserve an untouched confirmatory behavior set, and replicate the central result on one additional model.

# RQ2 brief progress note: multilingual refusal circuits

Date: 2026-07-01  
Branch: `rq2-multilingual-refusal-circuits`

## Short version

We extended the RQ1 refusal-circuit experiment from English Gemma to multilingual Qwen. The main current result is strong: in `Qwen/Qwen2.5-7B-Instruct`, English, German, Chinese, and Thai appear to share a causal edge-level circuit that builds a common refusal-direction representation.

The most complete and systematic result is for the internal representation target,

\[
M_{\mathrm{EN}}(x)=h_{15,-2}(x)^\top \hat r_{\mathrm{en}},
\]

but we also ran behavioral validation. Those behavioral results are positive, though smaller/noisier than the projection evaluations.

## What we went through

1. We started from the RQ1 idea.

   In RQ1, we found a refusal direction in Gemma and used an intervention-aware version of EAP-IG to find edges that preserve the projection onto that direction.

2. We moved to a multilingual model.

   We switched to `Qwen/Qwen2.5-7B-Instruct` because Gemma was not ideal for multilingual refusal. We used PolyRefuse prompts in English, German, Chinese, and Thai.

3. We first tried to replicate the multilingual refusal-vector result.

   We extracted refusal directions per language and checked whether they align. They do: the German, Chinese, and Thai vectors have cosine similarity above 0.93 with the English vector.

4. We realized the anchor point matters.

   If every language chooses its own best layer/token, then comparing circuits is messy: low overlap might just mean we asked EAP-IG to explain different targets. So we standardized the main target to the English/common anchor: layer 15, token position -2.

5. We had to be careful with behavioral proxies.

   First-token refusal checks were language-sensitive. We added WildGuard checks, and those gave positive behavioral evidence. However, WildGuard/request-harm labels can be imperfect for some languages, especially Thai, so we report behavioral results with that caveat rather than treating them as the only criterion.

6. We found that local target-site ablation is not enough.

   Ablating only the projection at \(h_{15,-2}\) does not fully reproduce global refusal-vector ablation. So \(h_{15,-2}^\top r\) is a good internal target, but not by itself a complete behavioral bottleneck.

7. We found that k=1000 already changes behavior, and larger k can make the behavioral effect stronger.

   At k=1000, circuits are very faithful/necessary for the projection metric and already substantially reduce generated refusal under circuit ablation. Because Qwen is much larger than Gemma, we also tested larger English circuits; at k=5000+, circuit ablation drove the refusal-substring rate to 0/8 in English.

## Main setup

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Dataset: PolyRefuse
- Languages: English (`en`), German (`de`), Chinese (`zh`), Thai (`th`)
- Main metric:

\[
M_{\mathrm{EN}}(x)=h_{15,-2}(x)^\top \hat r_{\mathrm{en}}
\]

where \(\hat r_{\mathrm{en}}\) is the normalized English refusal direction.

For EAP-IG:

- clean = normal harmful prompt;
- corrupt = same harmful prompt with global refusal-direction ablation;
- score edges by how much they preserve the shared refusal-direction projection.

No harmful completions are written to disk.

## Direction result

The English refusal vector transfers strongly across languages.

| Language | Cosine with English vector |
|---|---:|
| English | 1.000 |
| German | 0.964 |
| Chinese | 0.969 |
| Thai | 0.935 |

Directional ablation of the English vector also reduces the refusal projection and refusal-token log-odds in the other languages.

## Circuit families we tried

We tested several notions of “shared circuit.”

| Circuit type | What it means | Why we used it |
|---|---|---|
| Local language circuit | EAP-IG circuit found independently in one language | Tests whether each language has a projection-faithful circuit |
| English-transfer circuit | English circuit evaluated on de/zh/th prompts | Tests whether an English-discovered circuit works cross-lingually |
| Pairwise intersection | Edges shared by two language circuits | Diagnostic for which language pairs share edges |
| Strict intersection | Edges appearing in all four local circuits | Strongest sparse shared-core test |
| Majority core | Edges appearing in at least 3 of 4 local circuits | Allows small language-specific variation |
| Consensus circuit | Average normalized edge scores across languages, then select top-k | Best estimate of a multilingual common circuit |
| Union | Edges appearing in any language circuit | Used mainly as a comparison/denominator for IoU; not our main “shared sparse circuit” because it is too permissive |
| Random matched circuits | Random edge sets with the same size | Baseline to show results are not caused by retaining many edges |

## Local circuit result

At k=1000, independently discovered local circuits preserve the refusal-direction projection and removing them destroys it.

| Language | Faithfulness | Necessity |
|---|---:|---:|
| English | 0.862 | 1.002 |
| German | 0.889 | 1.003 |
| Chinese | 0.862 | 1.005 |
| Thai | 0.914 | 0.997 |

Matched random circuits are near zero.

Faithfulness asks: if we keep only the circuit, how much of the clean refusal projection remains?  
Necessity asks: if we ablate the circuit, how much of the clean refusal projection disappears?

## Structural overlap result

At k=1000, independently found circuits overlap strongly.

| Pair | IoU |
|---|---:|
| en–de | 0.685 |
| en–zh | 0.764 |
| en–th | 0.658 |
| de–zh | 0.679 |
| de–th | 0.753 |
| zh–th | 0.668 |

This is far above matched random overlap. Full edge-score correlations are also high: Spearman correlations are roughly 0.78–0.88.

## Transfer result

The English k=1000 circuit transfers well to other languages.

| Target language | Faithfulness | Necessity |
|---|---:|---:|
| German | 0.807 | 1.002 |
| Chinese | 0.832 | 1.002 |
| Thai | 0.843 | 1.004 |

So the English circuit is not merely English-specific, at least for the refusal-direction projection.

## Shared-core result

The strict four-language intersection has only 706 retained edges at local k=1000, but it still works.

| Language | Strict-core faithfulness | Strict-core necessity |
|---|---:|---:|
| English | 0.809 | 0.995 |
| German | 0.720 | 0.991 |
| Chinese | 0.796 | 0.995 |
| Thai | 0.725 | 0.992 |

The majority core, edges appearing in at least 3 of 4 circuits, has 871 edges and is slightly stronger.

| Language | Majority-core faithfulness | Majority-core necessity |
|---|---:|---:|
| English | 0.823 | 0.995 |
| German | 0.742 | 0.999 |
| Chinese | 0.819 | 0.994 |
| Thai | 0.757 | 1.007 |

The consensus circuit is strongest.

| Language | Consensus faithfulness | Consensus necessity |
|---|---:|---:|
| English | 0.926 | 1.004 |
| German | 0.863 | 1.006 |
| Chinese | 0.904 | 1.000 |
| Thai | 0.877 | 1.009 |

## Behavioral checks

We did test model behavior, not just the internal projection.

For the multilingual consensus k=1000 circuit, WildGuard on 8 harmful generations per language gives:

| Language | Normal refusal | Direction-ablated refusal | Circuit-only refusal | Circuit-ablated refusal |
|---|---:|---:|---:|---:|
| English | 1.000 | 0.000 | 1.000 | 0.125 |
| German | 1.000 | 0.000 | 1.000 | 0.250 |
| Chinese | 1.000 | 0.000 | 1.000 | 0.375 |
| Thai | 1.000 | 0.000 | 1.000 | 0.375 |

This means the consensus circuit is behaviorally causal in the expected direction: keeping only the circuit preserves refusal, while ablating it greatly reduces refusal.

For local language-specific circuits, k=1000/k=2000 showed the same qualitative pattern: normal and circuit-only refusal were 100%, direction ablation was 0%, and circuit ablation reduced refusal substantially.

For English, we also tested larger local projection circuits. At k=5000, k=10000, k=20000, and k=50000:

- normal refusal-substring rate was 7/8;
- direction-ablated refusal-substring rate was 0/8;
- circuit-only refusal-substring rate was 7/8 or 8/8;
- circuit-ablated refusal-substring rate was 0/8.

So behavior has been tested and supports the circuit result. The remaining caveat is narrower: the behavioral sample sizes are smaller than the projection evaluations, and the large-k behavioral sweep has so far been done most thoroughly in English.

## Current claim

Strong supported claim:

> Qwen2.5-7B-Instruct has a shared multilingual edge-level circuit that causally mediates a common refusal-direction representation across English, German, Chinese, and Thai.

Not yet fully supported:

> The exact minimal circuit size needed for robust generated-behavior control across all languages.

## Key files

- Full update: `docs/RQ2_COAUTHOR_UPDATE.md`
- Presentation plots: `results/qwen2.5-7b-instruct/polyrefuse/presentation_plots/`
- Direction outputs: `results/qwen2.5-7b-instruct/polyrefuse/{en,de,zh,th}/direction_paper_kl0p2/`
- Local circuit evaluations: `results/qwen2.5-7b-instruct/polyrefuse/{en,de,zh,th}/circuit_evaluation_english_vector_heldout_offset32_k1000_seeds5/`
- Circuit overlap: `results/qwen2.5-7b-instruct/polyrefuse/comparisons_english_vector/`
- Transfer tests: `results/qwen2.5-7b-instruct/polyrefuse/cross_language_english_vector_heldout_offset32_k1000_seeds5/`
- Shared/consensus evaluations: `results/qwen2.5-7b-instruct/polyrefuse/shared_circuit_evaluation_english_vector/`

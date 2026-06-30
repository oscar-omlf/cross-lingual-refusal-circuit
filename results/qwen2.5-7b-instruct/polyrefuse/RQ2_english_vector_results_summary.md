# RQ2 English-vector multilingual circuit results

Date: 2026-06-30.

This note records the current RQ2 state so the experiment can be reconstructed later without relying on chat history.

## Executive takeaway

This run gives a strong positive result for RQ2. Using Qwen2.5-7B-Instruct and PolyRefuse, we found that the English common-anchor refusal vector is highly aligned with refusal vectors from German, Chinese, and Thai, and that a common edge-level circuit upstream of this refusal-direction representation appears to exist across languages.

The strongest single summary result is the consensus circuit at k=1000. It preserves the refusal-direction projection with faithfulness 0.926 / 0.863 / 0.904 / 0.877 on English / German / Chinese / Thai, while matched random circuits are near zero. Its necessity is also ~1.0 in every language. The strict four-language intersection is also functional: only 706 edges shared by all four independently discovered k=1000 circuits preserve 72–81% of the projection and are ~fully necessary.

Current best claim:

> In Qwen2.5-7B-Instruct, for English, German, Chinese, and Thai PolyRefuse prompts, we find evidence for a shared multilingual edge-level circuit that causally mediates a common refusal-direction representation at layer 15, token -2. The evidence comes from aligned refusal vectors, high structural overlap between independently discovered circuits, strong English-circuit transfer, and functional faithfulness/necessity of strict shared, majority-core, and consensus circuits.

Important scope limit:

> This is a circuit for the residual-stream refusal-direction projection \(h_{15,-2}^{\top}\hat r_{\mathrm{en}}\), not yet a full generated-behavior circuit from prompt to refusal text. Behavioral validation remains a separate robustness step.

## Setup

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Dataset: PolyRefuse processed splits under `data/processed/qwen2.5-7b-instruct/polyrefuse/{en,de,zh,th}`.
- Languages: English (`en`), German (`de`), Chinese (`zh`), Thai (`th`).
- Common scalar metric:

  \[
  M_{\mathrm{en}}(x)=h_{15,-2}(x)^\top \hat r_{\mathrm{en}}
  \]

- The metric uses the English common-anchor refusal direction saved in each language namespace under `shared_metrics/english_vector/`.
- All discovery and evaluation here use the same residual endpoint: layer 15, token position -2.
- Circuit discovery uses our intervention-aware EAP-IG adaptation: the clean/corrupt path is global refusal-direction ablation strength \(\alpha=0\) to clean \(\alpha=1\), not vanilla input-embedding interpolation.
- No generated harmful completions are saved in these jobs. Evaluation used `--skip-behaviour`.

## Metric precheck

The English vector is well aligned with each language's local common-anchor vector and produces a large harmful clean-corrupt projection drop.

| Language | Cosine with English anchor | Harmful clean-corrupt projection drop | Harmful refusal log-odds clean → corrupt |
|---|---:|---:|---:|
| en | 1.000 | 16.42 | 5.61 → -12.27 |
| de | 0.964 | 15.20 | 1.75 → -7.70 |
| zh | 0.969 | 19.63 | 2.34 → -14.88 |
| th | 0.935 | 14.78 | 1.38 → -6.94 |

## Completed jobs

| Phase | Jobs | Status | Runtime |
|---|---:|---|---:|
| EAP-IG discovery | `24302504–24302507` | completed | 5–8 min each |
| Top-k circuit rebuild | `24302512–24302515` | completed | <1 min each |
| Local faithfulness/necessity | `24302520–24302523` | completed | 1h11–1h29 each |
| Structural comparison | `24302530` | completed | ~1 min |
| Shared/intersection construction | `24302531` | completed | 13 sec |
| Consensus construction | `24302532` | completed | 25 sec |
| English-circuit transfer | `24302533–24302535` | completed | 1h11–1h30 each |

## Local circuit results

Each row evaluates the circuit discovered independently for that language, on that language's held-out harmful test prompts. `F` is normalized faithfulness and `N` is normalized necessity for \(M_{\mathrm{en}}\). Random baselines are matched random edge sets with the same retained-edge count.

| k | en F | en N | de F | de N | zh F | zh N | th F | th N |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.268 | 0.684 | 0.235 | 0.682 | 0.223 | 0.629 | 0.253 | 0.675 |
| 25 | 0.301 | 0.715 | 0.331 | 0.787 | 0.250 | 0.757 | 0.341 | 0.737 |
| 50 | 0.424 | 0.720 | 0.482 | 0.801 | 0.369 | 0.761 | 0.498 | 0.803 |
| 100 | 0.553 | 0.859 | 0.623 | 0.900 | 0.588 | 0.881 | 0.590 | 0.906 |
| 200 | 0.672 | 0.982 | 0.695 | 0.984 | 0.660 | 0.971 | 0.621 | 0.982 |
| 500 | 0.742 | 0.995 | 0.768 | 0.993 | 0.860 | 0.983 | 0.762 | 0.983 |
| 1000 | 0.856 | 1.006 | 0.862 | 1.005 | 0.878 | 1.003 | 0.865 | 0.996 |
| 2000 | 0.896 | 1.015 | 0.888 | 0.999 | 0.955 | 1.010 | 0.876 | 1.013 |

Random matched faithfulness was small throughout: typically ~0.006–0.031 across k/language. Random necessity was near zero, sometimes slightly negative.

## Local circuit logit-margin diagnostics

These are secondary sanity checks, not the primary EAP-IG target.

| Language | k | Clean margin | Circuit-only margin | Circuit-ablated margin |
|---|---:|---:|---:|---:|
| en | 1000 | 12.881 | 13.128 | -4.623 |
| de | 1000 | 14.066 | 13.906 | 4.912 |
| zh | 1000 | 16.012 | 15.816 | 5.770 |
| th | 1000 | 12.428 | 12.337 | 6.351 |

The local circuits preserve refusal-token logit margins under circuit-only patching and reduce them under circuit ablation, though not always down to the fully corrupted baseline.

## English-circuit transfer

The English circuit was frozen and evaluated on target-language prompts using the same English-vector metric.

| k | de F | de N | zh F | zh N | th F | th N |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.239 | 0.674 | 0.279 | 0.714 | 0.261 | 0.678 |
| 25 | 0.312 | 0.763 | 0.313 | 0.725 | 0.346 | 0.745 |
| 50 | 0.419 | 0.749 | 0.428 | 0.737 | 0.433 | 0.716 |
| 100 | 0.546 | 0.876 | 0.550 | 0.860 | 0.555 | 0.854 |
| 200 | 0.615 | 0.990 | 0.669 | 0.966 | 0.641 | 0.982 |
| 500 | 0.651 | 0.993 | 0.725 | 0.982 | 0.667 | 0.986 |
| 1000 | 0.778 | 1.002 | 0.844 | 0.998 | 0.784 | 1.005 |
| 2000 | 0.792 | 1.013 | 0.891 | 1.009 | 0.825 | 1.011 |

Interpretation: the English circuit transfers strongly. It is slightly less faithful than each language's own circuit, but far above random and highly necessary.

## Structural circuit overlap

Pairwise structural overlap among independently discovered language circuits is far above random.

| k | Mean IoU | IoU range | Mean overlap coefficient | Mean weighted Jaccard | Random IoU |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.747 | 0.667–1.000 | 0.850 | 0.704 | 0.00000 |
| 25 | 0.697 | 0.613–0.786 | 0.820 | 0.704 | 0.00000 |
| 50 | 0.743 | 0.639–0.887 | 0.850 | 0.704 | 0.00000 |
| 100 | 0.771 | 0.709–0.818 | 0.870 | 0.704 | 0.00017 |
| 200 | 0.755 | 0.717–0.818 | 0.860 | 0.704 | 0.00025 |
| 500 | 0.703 | 0.653–0.786 | 0.825 | 0.704 | 0.00083 |
| 1000 | 0.701 | 0.658–0.764 | 0.824 | 0.704 | 0.00231 |
| 2000 | 0.691 | 0.643–0.752 | 0.817 | 0.704 | 0.00387 |

At k=1000:

| Pair | IoU | Shared edges | Union edges | Overlap coefficient | Weighted Jaccard |
|---|---:|---:|---:|---:|---:|
| en–de | 0.685 | 813 | 1187 | 0.813 | 0.671 |
| en–zh | 0.764 | 866 | 1134 | 0.866 | 0.773 |
| en–th | 0.658 | 794 | 1206 | 0.794 | 0.653 |
| de–zh | 0.679 | 809 | 1191 | 0.809 | 0.677 |
| de–th | 0.753 | 859 | 1141 | 0.859 | 0.774 |
| zh–th | 0.668 | 801 | 1199 | 0.801 | 0.674 |

Full-graph edge-score correlations:

| Pair | Spearman | Pearson |
|---|---:|---:|
| en–de | 0.813 | 0.978 |
| en–zh | 0.875 | 0.997 |
| en–th | 0.781 | 0.976 |
| de–zh | 0.813 | 0.976 |
| de–th | 0.865 | 0.995 |
| zh–th | 0.813 | 0.976 |

## Shared-core construction and functional evaluation

Three shared-circuit families were built and functionally evaluated on all four languages:

- `intersection_en_de_zh_th`: strict edge intersection of all four independently discovered circuits.
- `majority_at_least_3_of_4`: edges appearing in at least three of four language circuits.
- `consensus_mean_scores`: circuit selected by averaging normalized signed EAP-IG scores across languages and applying the greedy frontier selection.

Shared-core sizes:

| k | Strict 4-language shared edges | Majority ≥3/4 edges | Union edges |
|---:|---:|---:|---:|
| 10 | 7 | 9 | 12 |
| 25 | 18 | 22 | 35 |
| 50 | 38 | 43 | 64 |
| 100 | 78 | 91 | 125 |
| 200 | 151 | 181 | 251 |
| 500 | 359 | 426 | 670 |
| 1000 | 706 | 871 | 1341 |
| 2000 | 1388 | 1740 | 2716 |

Consensus circuits from averaged normalized signed scores were also built for all k; their retained edge counts equal the requested k.

Mean functional results across languages:

| Circuit | k | Mean edges | Mean F | F range | Mean N | N range | Random F |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict shared | 100 | 78 | 0.545 | 0.535–0.561 | 0.844 | 0.826–0.858 | 0.014 |
| strict shared | 500 | 359 | 0.703 | 0.662–0.751 | 0.975 | 0.963–0.983 | 0.027 |
| strict shared | 1000 | 706 | 0.763 | 0.720–0.809 | 0.993 | 0.991–0.995 | 0.027 |
| strict shared | 2000 | 1388 | 0.833 | 0.784–0.892 | 1.005 | 0.999–1.011 | 0.020 |
| majority ≥3/4 | 100 | 91 | 0.592 | 0.583–0.598 | 0.868 | 0.858–0.884 | 0.015 |
| majority ≥3/4 | 500 | 426 | 0.726 | 0.692–0.765 | 0.977 | 0.972–0.984 | 0.030 |
| majority ≥3/4 | 1000 | 871 | 0.785 | 0.742–0.823 | 0.999 | 0.994–1.007 | 0.026 |
| majority ≥3/4 | 2000 | 1740 | 0.897 | 0.860–0.945 | 1.001 | 0.997–1.007 | 0.019 |
| consensus | 100 | 100 | 0.589 | 0.579–0.602 | 0.891 | 0.885–0.905 | 0.013 |
| consensus | 500 | 500 | 0.746 | 0.711–0.778 | 0.983 | 0.973–0.992 | 0.029 |
| consensus | 1000 | 1000 | 0.892 | 0.863–0.926 | 1.005 | 1.000–1.009 | 0.028 |
| consensus | 2000 | 2000 | 0.916 | 0.869–0.949 | 1.007 | 1.002–1.012 | 0.018 |

Per-language shared-circuit results at k=1000:

| Circuit | en F/N | de F/N | zh F/N | th F/N |
|---|---:|---:|---:|---:|
| strict shared, 706 edges | 0.809 / 0.995 | 0.720 / 0.991 | 0.796 / 0.995 | 0.725 / 0.992 |
| majority ≥3/4, 871 edges | 0.823 / 0.995 | 0.742 / 0.999 | 0.819 / 0.994 | 0.757 / 1.007 |
| consensus, 1000 edges | 0.926 / 1.004 | 0.863 / 1.006 | 0.904 / 1.000 | 0.877 / 1.009 |

Comparison to each language's local independently discovered circuit at k=1000:

| Language | Local F | Strict F | Majority F | Consensus F | Local N | Strict N | Majority N | Consensus N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 0.856 | 0.809 | 0.823 | 0.926 | 1.006 | 0.995 | 0.995 | 1.004 |
| de | 0.862 | 0.720 | 0.742 | 0.863 | 1.005 | 0.991 | 0.999 | 1.006 |
| zh | 0.878 | 0.796 | 0.819 | 0.904 | 1.003 | 0.995 | 0.994 | 1.000 |
| th | 0.865 | 0.725 | 0.757 | 0.877 | 0.996 | 0.992 | 1.007 | 1.009 |

Comparison to English-circuit transfer at k=1000:

| Target | English transfer F | Strict F | Majority F | Consensus F | English transfer N | Strict N | Majority N | Consensus N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| de | 0.778 | 0.720 | 0.742 | 0.863 | 1.002 | 0.991 | 0.999 | 1.006 |
| zh | 0.844 | 0.796 | 0.819 | 0.904 | 0.998 | 0.995 | 0.994 | 1.000 |
| th | 0.784 | 0.725 | 0.757 | 0.877 | 1.005 | 0.992 | 1.007 | 1.009 |

Interpretation:

- The strict four-language intersection alone is faithful and necessary. At k=1000, the 706 edges shared by all four language circuits preserve 72–81% of the English-vector refusal projection and have necessity ~0.99.
- The majority core is slightly stronger than the strict intersection.
- The consensus circuit is strongest: at k=1000 it reaches 86–93% faithfulness and ~1.0 necessity across all languages, matching or exceeding local circuits in several cases.
- Random matched edge sets remain near zero faithfulness and near-zero necessity.

## Current interpretation

Supported by current results:

1. The English refusal vector at the common anchor is meaningful across all four languages.
2. Independently discovered circuits are faithful and necessary for the English-vector refusal projection in every language.
3. The circuits have very high structural overlap and highly correlated full edge-score landscapes.
4. The English circuit transfers strongly to German, Chinese, and Thai.
5. The strict four-language shared core, majority core, and consensus circuit are each functionally faithful and necessary across languages.

The precise current claim should be:

> We find strong evidence for a shared multilingual edge-level circuit upstream of the common English-vector refusal representation in Qwen2.5-7B-Instruct. Independently discovered circuits overlap strongly, the English circuit transfers to other languages, and strict/majority/consensus shared circuits are faithful and necessary across English, German, Chinese, and Thai.

Caveat: this is still a circuit for the residual refusal-direction projection \(h_{15,-2}^\top \hat r_{\mathrm{en}}\), not yet a full behavior-to-logits or generated-completion refusal circuit.

## SBU accounting and estimates

Accounting uses Slurm `billing` TRES. For A100 jobs here, `billing=128`, so estimated SBU is:

\[
\mathrm{SBU}\approx 128\times \mathrm{elapsed\ hours}.
\]

Past completed jobs:

| Phase | Jobs | Avg runtime/job | Estimated SBU total | Avg SBU/job |
|---|---:|---:|---:|---:|
| Discovery | 4 | 0.104 h | 53.3 | 13.3 |
| Local eval | 4 | 1.325 h | 678.6 | 169.7 |
| English transfer | 3 | 1.360 h | 522.3 | 174.1 |
| CPU rebuild/compare/core/consensus | 7 | 0.008 h | 0.9 | 0.1 |

The new shared-core eval jobs evaluate only four k values (`100 500 1000 2000`), half the previous k grid. Estimated cost:

| Remaining eval set | Jobs | Estimated runtime/job | Estimated SBU |
|---|---:|---:|---:|
| Strict 4-language intersection | 4 | ~0.66 h | ~339 |
| Majority ≥3/4 core | 4 | ~0.66 h | ~339 |
| Consensus mean-score circuit | 4 | ~0.66 h | ~339 |
| Total submitted now | 12 | ~0.66 h | ~1,018 |

Actual shared-core evaluation cost:

| Shared eval set | Jobs | Runtime range | Actual estimated SBU |
|---|---:|---:|---:|
| Strict 4-language intersection | 4 | 48–56 min | 441 |
| Majority ≥3/4 core | 4 | 50–60 min | 462 |
| Consensus mean-score circuit | 4 | 51–60 min | 469 |
| Total completed shared eval | 12 | 48–60 min | 1,372 |

The actual cost was higher than the rough estimate because the half-size k grid still required model loading and repeated clean/corrupt/circuit-only/circuit-ablated passes.

## Completed shared-core evaluation jobs

All use `METRIC_NAME=english_vector`, `NUM_PROMPTS=32`, `K_VALUES="100 500 1000 2000"`, and `RANDOM_SEEDS=0`.

| Shared circuit kind | Language | Job ID | Runtime |
|---|---|---:|---:|
| strict intersection `intersection_en_de_zh_th` | en | 24307681 | 49m23s |
| strict intersection `intersection_en_de_zh_th` | de | 24307683 | 53m00s |
| strict intersection `intersection_en_de_zh_th` | zh | 24307686 | 48m08s |
| strict intersection `intersection_en_de_zh_th` | th | 24307688 | 56m14s |
| majority core `majority_at_least_3_of_4` | en | 24307690 | 51m27s |
| majority core `majority_at_least_3_of_4` | de | 24307692 | 55m16s |
| majority core `majority_at_least_3_of_4` | zh | 24307694 | 49m56s |
| majority core `majority_at_least_3_of_4` | th | 24307696 | 59m41s |
| consensus `consensus_mean_scores` | en | 24307697 | 52m46s |
| consensus `consensus_mean_scores` | de | 24307699 | 56m02s |
| consensus `consensus_mean_scores` | zh | 24307702 | 50m47s |
| consensus `consensus_mean_scores` | th | 24307704 | 60m26s |

## Deferred robustness work

These were intentionally not run in this batch:

- `common_mean` metric branch.
- Local/transfer eval with 64 held-out prompts.
- More random seeds.
- Behavioral generation/WildGuard validation.
- Full behavior-to-logits circuit; current circuit target is only the residual refusal-direction projection.

## Recommended next steps

### 1. Freeze and preserve the current positive result

This should happen before launching another major experiment:

- Commit the code changes needed to reproduce the RQ2 pipeline.
- Commit this summary memo and the core CSV/JSON result tables for:
  - direction transfer / metric precheck,
  - local circuit evaluation,
  - circuit overlap comparison,
  - English-circuit transfer,
  - shared-core / majority-core / consensus evaluations.
- Avoid committing raw logs, model caches, papers, and large transient/debug outputs unless explicitly needed.

### 2. Make paper/poster figures from the current data

Highest-value figures:

- Faithfulness vs k for local circuits, English-transfer circuits, strict shared core, majority core, consensus circuit, and matched random baselines.
- Necessity vs k for the same circuit families.
- Pairwise circuit-overlap heatmap at k=1000.
- Edge-score correlation heatmap across languages.
- A compact component/layer distribution plot for the consensus circuit and strict intersection.

These figures are enough for a presentation-ready RQ2 story.

### 3. Add minimal robustness without exploding cost

If budget/time allows, prioritize:

1. Re-evaluate only the most important circuits at k=1000 with 64 held-out prompts:
   - local circuits,
   - English-transfer circuit,
   - strict intersection,
   - consensus circuit.
2. Add 2–3 random seeds for the matched-random baselines at k=1000 only.
3. Run the `common_mean` metric branch only if we want to show that the result is not an artifact of using the English vector as the scalar target.

### 4. Behavioral validation, if we want a stronger claim

The current evidence is causal for the refusal-direction representation. To claim more about actual refusal behavior, run a small behavioral validation:

- normal model,
- global refusal-direction ablated model,
- circuit-only model,
- circuit-ablated model.

Use refusal-token log-odds and, if possible, WildGuard or another safety classifier on generated completions. Do not write harmful completions to disk; save only aggregate labels/metrics and sanitized examples if necessary.

### 5. Scientific follow-up questions

The current result suggests a nice next set of research questions:

- Is the common circuit mostly late-layer/shared, with early language-specific routing?
- Does the shared circuit survive if we add Arabic or other scripts/language families?
- Does the same common circuit mediate different refusal categories, or only generic refusal?
- Does a behavior-to-logits refusal circuit share the same core as this representation-level circuit?
- Does the circuit transfer across models, or is it model-specific?

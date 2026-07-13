# Results summary

This file is generated from committed result artifacts under `results/`.

## Result inventory

| Result tree | Files | Size |
| --- | --- | --- |
| cross-task | 10 | 0.4 MB |
| faithfulness | 5 | 0.0 MB |
| first_figure | 3 | 0.1 MB |
| gemma-2-2b-it | 125 | 13.4 MB |
| ig_test | 6 | 0.1 MB |
| jaccard | 14 | 0.3 MB |
| manual_overlap | 8 | 0.2 MB |
| pareto | 80 | 1.7 MB |
| qwen2.5-7b-instruct | 475 | 161.6 MB |
| real | 6 | 0.0 MB |
| real_pareto_combined | 3 | 0.1 MB |
| real_rank | 17 | 1.2 MB |


## RQ1: Gemma-2-2B-it English refusal circuit

Goal: find an edge-level circuit that mediates a refusal-direction projection in `google/gemma-2-2b-it`.

- Main target: residual projection onto the selected refusal direction at layer 14, token position -2.
- Main behavioral rerun: EAP-IG top-1000 circuit, which retained 668 edges.

| Condition | n | Refusal substring rate | Refusal-token logit margin |
| --- | --- | --- | --- |
| normal | 32 | 1.000 | -2.803 |
| refusal_direction_ablated | 32 | 0.000 | -14.169 |
| target_direction_ablated | 32 | 0.969 | -4.489 |
| circuit_only | 32 | 0.969 | -3.073 |
| circuit_ablated | 32 | 0.094 | -11.677 |


Projection-level pilot metrics:

| Circuit | k | retained edges | Faithfulness | Necessity |
| --- | --- | --- | --- | --- |
| EAP-IG | 500 | 135 | 0.318 | 0.740 |
| matched random | 500 | 135 | -0.002 | -0.004 |


Interpretation: the broader circuit is behaviorally causal; target-site-only ablation is not enough, so the scalar projection is not by itself the whole behavioral bottleneck.

Key files: `results/gemma-2-2b-it/en/rq1/behavioral_rerun/eapig_top_1000_test_n32_gen32.csv`, `results/gemma-2-2b-it/en/rq1/faithfulness_pilot/faithfulness.csv`, `results/gemma-2-2b-it/en/rq1/faithfulness_pilot/necessity.csv`.

## RQ2: Qwen2.5-7B-Instruct multilingual refusal circuits

Main target used for circuit comparison:

\[M_{\mathrm{EN}}(x)=h_{15,-2}(x)^\top \hat r_{\mathrm{en}}\]

Direction/vector sanity checks:

| Language | cosine with English anchor | harmful projection drop | refusal log-odds clean → corrupt |
| --- | --- | --- | --- |
| en | 1.000 | 16.42 | 5.61 → -12.27 |
| de | 0.964 | 15.20 | 1.75 → -7.70 |
| zh | 0.969 | 19.63 | 2.34 → -14.88 |
| th | 0.935 | 14.78 | 1.38 → -6.94 |


Local held-out circuit evaluations at k=1000:

| Language | edges | Faithfulness | Necessity | circuit-ablated logit margin |
| --- | --- | --- | --- | --- |
| en | 1000 | 0.862 | 1.002 | -2.396 |
| de | 1000 | 0.889 | 1.003 | 4.165 |
| zh | 1000 | 0.862 | 1.005 | 7.183 |
| th | 1000 | 0.914 | 0.997 | 6.445 |


Cross-language structural overlap for independently discovered circuits, all using the same English anchor:

| Pair | shared edges | union edges | IoU | overlap coefficient |
| --- | --- | --- | --- | --- |
| en–de | 813 | 1187 | 0.685 | 0.813 |
| en–zh | 866 | 1134 | 0.764 | 0.866 |
| en–th | 794 | 1206 | 0.658 | 0.794 |
| de–zh | 809 | 1191 | 0.679 | 0.809 |
| de–th | 859 | 1141 | 0.753 | 0.859 |
| zh–th | 801 | 1199 | 0.668 | 0.801 |


Full edge-score correlations:

| Pair | Spearman | Pearson |
| --- | --- | --- |
| en–de | 0.813 | 0.978 |
| en–zh | 0.875 | 0.997 |
| en–th | 0.781 | 0.976 |
| de–zh | 0.813 | 0.976 |
| de–th | 0.865 | 0.995 |
| zh–th | 0.813 | 0.976 |


Shared-core / consensus circuit evaluations at k=1000:

| Circuit family | Language | edges | Faithfulness | Necessity |
| --- | --- | --- | --- | --- |
| strict intersection | en | 706 | 0.809 | 0.995 |
| strict intersection | de | 706 | 0.720 | 0.991 |
| strict intersection | zh | 706 | 0.796 | 0.995 |
| strict intersection | th | 706 | 0.725 | 0.992 |
| majority ≥3/4 | en | 871 | 0.823 | 0.995 |
| majority ≥3/4 | de | 871 | 0.742 | 0.999 |
| majority ≥3/4 | zh | 871 | 0.819 | 0.994 |
| majority ≥3/4 | th | 871 | 0.757 | 1.007 |
| consensus | en | 1000 | 0.926 | 1.004 |
| consensus | de | 1000 | 0.863 | 1.006 |
| consensus | zh | 1000 | 0.904 | 1.000 |
| consensus | th | 1000 | 0.877 | 1.009 |


Behavioral validation:

WildGuard refusal rates for multilingual consensus k=1000 circuits:

| Language | normal | direction ablated | circuit only | circuit ablated |
| --- | --- | --- | --- | --- |
| en | 1.000 | 0.000 | 1.000 | 0.125 |
| de | 1.000 | 0.000 | 1.000 | 0.250 |
| zh | 1.000 | 0.000 | 1.000 | 0.375 |
| th | 1.000 | 0.000 | 1.000 | 0.375 |


English larger-k projection circuit generation substring checks:

| k | normal | circuit only | circuit ablated |
| --- | --- | --- | --- |
| 5000 | 0.875 | 1.000 | 0.000 |
| 10000 | 0.875 | 1.000 | 0.000 |
| 20000 | 0.875 | 1.000 | 0.000 |
| 50000 | 0.875 | 0.875 | 0.000 |


English logit-target pilot:

| k | normal log-odds | circuit-only log-odds | circuit-ablated log-odds | ablated gt0 rate |
| --- | --- | --- | --- | --- |
| 1000 | 5.941 | -5.066 | -10.302 | 0.000 |
| 2000 | 5.941 | -3.961 | -10.247 | 0.000 |
| 5000 | 5.941 | 2.367 | -10.739 | 0.000 |
| 10000 | 5.941 | 2.713 | -11.145 | 0.000 |
| 20000 | 5.941 | 3.217 | -11.332 | 0.000 |
| 50000 | 5.941 | 5.274 | -11.355 | 0.000 |


## Current high-level claim

RQ1 showed a sparse, behaviorally causal refusal circuit in Gemma-2-2B-it. RQ2 extended this to Qwen2.5-7B-Instruct and found strong evidence for a shared multilingual circuit that builds a common refusal-direction representation across English, German, Chinese, and Thai. Behavioral tests support the circuit result, especially for consensus k=1000 and larger English projection circuits.

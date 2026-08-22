# Vector results

All runs use `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, BF16.

## Refusal prompt vector

| Item | Exact setup/result |
|---|---|
| Extraction | DIM(`harmful − harmless`), 128+128 prompts; layers 0–27; positions −5…−1 |
| Selection | 32+32 held-out prompts; Arditi causal selection; addition `+1× raw` at all positions; KL≤0.2; late-layer fraction 0.8 |
| Selected site | `blocks.15.hook_resid_pre`, position −2, raw norm `20.09413768` |

Validation: 32 harmful + 32 harmless prompts; deterministic generation on the
32 harmful prompts, 64 tokens; steering writes the raw vector at every position
of layer 15; projection removes the unit direction globally.

| Condition | Harmful refusal log-odds | Harmless KL | Prefix refusals | WildGuard refusals | WildGuard harmful |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.247 | 0 | 31/32 | 32/32 | 0/32 |
| Global projection | −11.724 | 0.133 | 0/32 | 0/32 | 30/32 |
| Steering −1× | −7.544 | 0.373 | 2/32 | 4/32 | 28/32 |
| Steering −2× | −8.164 | 1.502 | 0/32 | 0/32 | 30/32 |

Conclusion: projection and −2× equally remove refusal, but projection has much
lower harmless-prompt distortion.

## Sycophancy generation vector

| Item | Exact setup/result |
|---|---|
| Extraction | Mean positive-response-token residual minus negative; 1,000+1,000 judged answers; 810 effective aligned pairs |
| Filter | Positive trait/coherence ≥50; negative trait <50 and coherence ≥50 |
| Selected site | HF hidden state 20 = output of block 19 / `blocks.20.hook_resid_pre`; raw norm `23.77709198` |
| Layer sweep | Layers 4/8/12/16/20/24/28; coefficients 0.5/1/1.5/2/2.5; 200 answers per cell = 7,000 |

Layer 20 won at every coefficient:

| +Coefficient | Trait | Coherence |
|---:|---:|---:|
| 0.5 | 8.98 | 99.46 |
| 1.0 | 20.15 | 99.23 |
| 1.5 | 63.24 | 96.78 |
| 2.0 | 91.56 | 85.86 |
| 2.5 | 98.44 | 39.99 |

## Sycophancy prompt vectors

Both extractions use 256 unique questions under matched positive/negative
system instructions: 512 prompt forwards, no generation; DIM at every layer.

| Position | Token | Valid layers | Result |
|---:|---|---|---|
| −1 | final newline | 1–27 | Complete; layer 0 zero |
| −2 | `assistant` | 1–27 | Complete; layer 0 zero |

Exact cosine maxima:

| Comparison | Maximum |
|---|---:|
| −1 vs −2 | `0.7970` at layer 15 |
| −1 vs legacy prompt vector | `0.8554` at layer 18 |
| −1 vs generation vector | `0.4897` at layer 21 |
| −2 vs generation vector | `0.3776` at layer 24 |

Completed causal screen, job `25968596`: 16 held-out questions; layers 1–27;
positions −1/−2; positive-system and negative-system baselines; recurrent
`+2× raw` induction and `−2× raw` ablation; deterministic, batch 4, 256 tokens;
1,760/1,760 answers, no empty outputs. Persona trait/coherence judging remains
pending because the HPC has no OpenAI credential.

The full prompt-versus-generation geometry analysis, tables, and figure are in
[`analysis/sycophancy_vector_geometry/REPORT.md`](analysis/sycophancy_vector_geometry/REPORT.md).

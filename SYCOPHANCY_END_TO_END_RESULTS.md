# Sycophancy circuit results

## Experiments

| Experiment | Contrast/target | Data | Graph |
|---|---|---:|---:|
| Current end-to-end | Natural sycophantic vs anti-sycophantic instruction → answer log-probability margin | 64 EAP-IG + 64 disjoint evaluation questions | Input→logits; 935,747 edges |
| Old upstream | Natural instruction → sycophancy-vector projection | 64 EAP-IG | Input→vector; 471,191 edges |
| Old downstream | Layer-20 generation-vector steering (`0→−2`) → answer margin | 64 EAP-IG + 32 evaluation records | Layer 20→logits; 70,157 edges |

- EAP-IG used 5 steps.
- Current target gap: `0.379` discovery; `0.405` evaluation.
- Old downstream gap: `1.535` discovery; `1.478` evaluation.

## Circuit results

| k | Current retained | Current faithfulness | Current necessity | Old upstream faithfulness | Old downstream faithfulness |
|---:|---:|---:|---:|---:|---:|
| 500 | 413 | −0.3% | 100.0% | 0.5% | 90.7% |
| 1,000 | 857 | 0.2% | 99.0% | −0.6% | 94.1% |
| 3,000 | 2,787 | 5.1% | 99.6% | 6.7% | 97.8% |
| 5,000 | 4,719 | 9.7% | 99.9% | 13.5% | — |
| 10,000 | 9,776 | 16.9% | 101.1% | 17.7% | — |

## Interpretation

- Current end-to-end results closely match old upstream results: persona formation is distributed.
- Old downstream starts from an imposed one-dimensional layer-20 signal, so only its compact transmission circuit must be recovered.
- Old downstream also used a stronger target and reused the same 20 question identities across discovery/evaluation records; its estimate may be optimistic.
- Result: **distributed formation → compact persona-vector bottleneck → sparse downstream transmission**.


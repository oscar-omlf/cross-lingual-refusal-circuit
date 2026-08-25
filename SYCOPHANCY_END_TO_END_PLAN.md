# Sycophancy end-to-end plan

- **Contrast:** clean = sycophantic instruction; corrupt = token-aligned
  anti-sycophantic instruction. Keep the user question and answer fixed.
- **Target:**  
  `(logP(sycophantic answer) − logP(normal answer))clean −`
  `(logP(sycophantic answer) − logP(normal answer))corrupt`.
- **Graph:** complete model input (`blocks.0.hook_resid_pre`) → response logits.
  No trait vector, projection, or steering.
- **Now:** score edges with five-step EAP-IG on 64 `eapig` prompts
  (job `25981840`). The 16-prompt contrast check passed: `+0.441`, 16/16.
- **Next:** build connected `k = 100, 500, 1000, 3000` circuits and evaluate
  faithfulness/necessity concurrently on 64 `circuit_selection` prompts.
- **Behavior:** test the winner on 16 prompts, then 300 prompts: positive
  baseline, negative baseline, circuit-only, and circuit-ablated. Circuit-only
  should recover sycophancy; circuit ablation should remove it while preserving
  coherence. Paid judging requires explicit approval.

# RQ1 presentation brief

## Claim

In `google/gemma-2-2b-it`, a sparse connected edge circuit causally mediates a
validated *downstream refusal-direction representation*. This is not a claim
that all refusal is literally one-dimensional or that we recovered the whole
upstream refusal mechanism.

## Question and model

RQ1 asks whether EAP-IG can find a faithful head/MLP edge-level circuit for

$$
M_{ref}(x)=h_{l^*,t^*}(x)^\top\hat r_{ref}.
$$

The model is the instruction-tuned Gemma-2 2B model, rendered with Gemma's
native chat template. The selected direction is at residual-input layer 14,
post-instruction position -2.

## Dataset and refusal direction

We built balanced instruction splits: 128 harmful + 128 harmless training,
32 + 32 validation, and 64 + 64 test examples. Harmful examples were retained
only when deterministic base completions matched a refusal prefix; harmless
examples were retained only when they did not. Prompts longer than 512 chat
tokens were excluded. No harmful completions are saved in results.

On training prompts, we computed a difference-in-means residual vector for
every layer and every post-instruction template token:

$$
r_{l,t}=\operatorname{mean}(h_{l,t}\mid harmful)-
\operatorname{mean}(h_{l,t}\mid harmless).
$$

Validation selected the candidate whose global directional ablation most
reduced harmful refusal, subject to harmless-direction addition inducing
refusal, low harmless KL shift, and an early-enough source layer. The selected
unit vector is saved in `results/rq1/refusal_direction.pt`.

Direction sanity check: normal harmful prompts had a 100% refusal-prefix rate
on 32 validation examples; directional ablation reduced this to 0%. Adding the
direction to harmless prompts induced refusal in 96.9% of cases.

## Circuit finding

Clean is a harmful prompt under the normal model. Corrupt is the *same token
sequence* under global ablation of the selected refusal direction at residual
inputs, complete attention writes, and MLP writes. This preserves token
alignment and defines the contrast used for circuit patching.

We used the authors' EAP graph (attention-head and MLP nodes, including Q/K/V
attention-input edges), their backwards greedy absolute-score circuit
selection, and recursive pruning. Empty/full circuit closure was verified
before attribution.

The original EAP-IG paper interpolates token embeddings between two different
clean/corrupt strings. That would be degenerate here because our clean and
corrupt token sequences are identical. We therefore use **intervention-aware
EAP-IG**: cache clean/corrupt component-output differences and integrate EAP
gradients along the continuous strength of the global direction-ablation
intervention (fully ablated → normal), using five steps. This is a valid
path-integrated attribution for our explicitly defined intervention, but it is
not a literal replication of the paper's text-pair embedding interpolation.

Because the target is an internal residual activation rather than logits, the
terminal gradient is contracted directly with source differences. The
authors' standard logit-terminal code does not need that adaptation.

## Main result: held-out test evaluation

Edges were scored on 64 harmful training prompts and evaluated on the separate
64 harmful test prompts. For a circuit `C`, faithfulness keeps only `C` clean;
necessity corrupts only `C`:

$$
F(C)=\frac{M_{C\text{-only}}-M_{corrupt}}{M_{clean}-M_{corrupt}},\qquad
N(C)=\frac{M_{clean}-M_{C\text{-ablated}}}{M_{clean}-M_{corrupt}}.
$$

| Method | Requested k | Retained edges | Held-out F | Held-out N |
| --- | ---: | ---: | ---: | ---: |
| EAP-IG | 1,000 | 636 (3.0% of 21,064) | **0.858** | **0.932** |
| EAP | 1,000 | 594 | 0.753 | 0.913 |
| EAP-IG | 2,000 | 1,444 | **0.913** | **0.959** |
| EAP | 2,000 | 1,515 | 0.863 | 0.963 |

Matched random edge sets have faithfulness around 0.007 at the 1,000-edge
request and around 0.038 at the 2,000-edge request. Thus the EAP-IG circuit is
far from a generic consequence of retaining many edges. It reaches the 0.85
faithfulness target with fewer than half the retained edges needed by EAP to
reach comparable faithfulness.

Behavioral sanity check of the original 668-edge EAP-IG circuit (8 test
prompts): normal 87.5% refusal-prefix rate; global direction ablation 0%;
circuit-only 75%; circuit-ablated 0%. These small-sample behavior results are
secondary to the projection metric.

## What to show

1. The held-out faithfulness and necessity plots.
2. The 636-edge/668-edge headline table above.
3. The direction intervention summary.
4. The readable top-80-edge Graphviz circuit view, labelled as a visualization
   of the selected circuit rather than a complete explanation.

Presentation assets: [held-out faithfulness plot](../results/rq1/heldout_test/figures/faithfulness_poster.png),
[held-out necessity plot](../results/rq1/heldout_test/figures/necessity_poster.png),
[behavioral summary plot](../results/rq1/figures/behavioral_sanity_poster.png), and
[top-80-edge circuit view](../results/rq1/figures/eapig_top1000_top80.png).

## Boundaries of the claim

The best circuit preserves about 91% rather than 100% of the target metric, so
there are residual/redundant contributions outside it. We tested one model,
one selected direction, and English harmful/harmless prompts. The result
supports a dominant downstream causal circuit for this representation; it does
not establish that all refusal categories, directions, or languages use the
same circuit.

# Sycophancy generation-vector mediation plan

**Status:** Frozen experimental design  
**Frozen:** 2026-08-23

## Mechanistic decomposition

At generation step \(t\):

`prompt + previous tokens → formation circuit → layer-20 state → transmission circuit → next-token logits`

| Object | Meaning | Graph | Target |
|---|---|---|---|
| Formation circuit \(U\) | Naturally creates a high/low coordinate along the generation direction | Input → layer 20 | \(M_t=\hat d^\top h_{20,t}\) |
| Transmission circuit \(D\) | Converts that coordinate into sycophantic logits | Layer 20 → logits | `logP(sycophantic answer) − logP(normal answer)` |
| End-to-end circuit \(E\) | Captures all instruction-to-behaviour pathways, including bypasses around \(d\) | Input → logits | Same log-probability margin |

The model does not literally transmit a vector. Components create, read, and
transform a hidden-state coordinate defined by the analyst.

Generation positions must be separated:

- **Final prompt:** initiation of the first response token.
- **Response tokens except the last:** generation-vector maintenance; each can
  affect the next token.
- **Final response token:** cannot affect the response already produced.

## Stage 1: geometry and mediation gate

Use all 64 `vector_validation` questions and their stored answers. No generation
or external judging is required.

Teacher-force four conditions per question:

| Instruction | Answer |
|---|---|
| Positive | Sycophantic |
| Negative | The same sycophantic answer |
| Positive | Normal |
| Negative | The same normal answer |

At layer 20, measure:

- the instruction main effect;
- the answer-content main effect;
- their interaction;
- alignment with \(d\);
- the natural positive-minus-negative coordinate gap, variance, and sign
  consistency;
- the position-resolved trajectory from final prompt through the response.

Run these causal interventions:

1. Replace only the positive state's \(d\)-coordinate with its paired natural
   negative coordinate.
2. Replace only the orthogonal remainder.
3. Replace the complete state as an upper-bound control.
4. Repeat the \(d\)-coordinate intervention from negative to positive.

Use the stored-answer margin

\[
Y=\log P(\text{sycophantic answer})-\log P(\text{normal answer})
\]

as the behavioural metric.

| Natural movement along \(d\) | Causal effect of changing \(d\) | Interpretation |
|---|---|---|
| Strong | Strong | Natural mediator |
| Strong | Weak | Behavioural marker |
| Weak | Strong | Controllable direction not naturally used |
| Weak | Weak | Not a useful mechanism |

Do not continue to vector-mediated circuit discovery unless the direction shows
both natural relevance and a meaningful causal effect.

## Normalized steering

Let \(g\) be the natural positive-minus-negative coordinate gap measured on the
validation split. Apply the intervention once at layer 20:

\[
h' = h-\alpha g\hat d
\]

Minimal tests:

- Positive → negative: \(\alpha=0.5,1\).
- Negative → positive: \(\alpha=1\).
- Test \(\alpha>1\) only if \(\alpha=1\) remains coherent but undershoots.

Also express the old raw `−2` intervention in natural-gap units. Do not run a
broad raw-coefficient sweep or use repeated all-layer steering. Pair-specific
coordinate replacement is the primary scientific test; normalized steering
tests whether one deployable coefficient generalizes.

## Stage 2: natural formation EAP-IG

Use all 64 `eapig` questions and both stored answers: 128 teacher-forced
trajectories.

- **Clean:** positive instruction.
- **Corrupt:** token-aligned negative instruction.
- Hold answer tokens fixed within every clean/corrupt pair.
- **Graph:** `blocks.0.hook_resid_pre → blocks.20.hook_resid_pre`.
- Use five EAP-IG interpolation steps along the natural negative-to-positive
  input path.
- **Primary target:** mean \(d\)-projection over response-prefix states:
  generated response tokens except the last.
- Average the sycophantic- and normal-answer trajectories equally:

\[
\frac12[(M_{P,S}-M_{N,S})+(M_{P,N}-M_{N,N})].
\]

Also report the exact all-response-token projection as a diagnostic. Build only
connected source-to-target circuits.

This combines natural positive/negative corruption with a genuine generation
target. It must not use prompt-last as the primary target or artificial
steering as its corrupt condition.

## Stage 3: transmission EAP-IG

Use the same 64 `eapig` questions and both stored answers.

- Start from the positive-instruction transcript.
- At layer 20, replace the response-prefix \(d\)-coordinate with its paired
  natural negative value.
- Interpolate along this coordinate-replacement path with five EAP-IG steps.
- **Graph:** layer 20 → logits.
- **Target:** the contrastive stored-answer log-probability margin, excluding
  the first response token when the intervention begins only on response-token
  states.

This identifies components that transmit a natural-scale change in the
generation direction, rather than components that recover from an extreme
global intervention.

## Stage 4: circuit evaluation and mediation

Use all 64 `circuit_selection` questions.

Required tests:

- Formation-circuit ablation lowers \(M\).
- Formation-circuit ablation also lowers sycophantic behaviour.
- Restoring only \(M\) after formation-circuit ablation rescues behaviour.
- Transmission-circuit ablation blocks the effect of restoring or inducing
  \(M\).
- Composing the formation and transmission circuits recovers part of the
  end-to-end effect.

Exact edge overlap with the end-to-end circuit is not required. The end-to-end
circuit may additionally contain first-token initiation and pathways not
mediated by the generation direction.

## Execution order and data

| Phase | Split | Questions | New generation required? |
|---|---|---:|---|
| Geometry and mediation gate | `vector_validation` | 64 | No |
| Formation and transmission EAP-IG | `eapig` | 64 | No |
| Circuit selection and mediation | `circuit_selection` | 64 | No |
| Final behavioural evaluation | `final_behavior` | 300 | Yes |

After Stage 1 passes, formation and transmission EAP-IG may run concurrently.
Their circuit evaluations may also run concurrently. Only the selected final
circuits should proceed to the 300-question generation and judging stage.

## Minimal code gap

The repository already contains the graphs, response masks, natural
clean/corrupt transcript construction, and contrastive log-probability metric.
The exact protocol needs only:

1. masked generation-position projection with natural prompt-pair corruption;
2. paired natural-coordinate replacement for downstream attribution.

No broader infrastructure rewrite is justified.

# RQ1: refusal-direction circuit

This directory adds the first experiment only:

> Can EAP-IG identify a faithful head/MLP edge-level circuit that causally
> mediates the downstream refusal representation of `google/gemma-2-2b-it`?

It does not make a claim that all kinds of refusal are one-dimensional. The
later refusal-cone and SAE papers in the supplied archive are good reasons to
interpret this as a narrow, first test of a downstream bottleneck.

## What is implemented

`python scripts/rq.py rq1.prepare-data` creates exactly balanced JSONL splits under
`data/processed/rq1/`: 128/128 train, 32/32 validation, and 64/64 test
harmful/harmless instructions. It starts from the checked-out
`external/refusal_direction/dataset/splits` prompt pools (or a user-supplied
`data/raw/rq1/{harmful,harmless}_{split}.{json,jsonl}`), then retains harmful
instructions whose deterministic base completion matches a refusal prefix and
harmless instructions whose completion does not. The processed JSONL is the
only dataset used after this step. It excludes chat-formatted prompts longer
than `--max-length` (512 by default), recording retained token length in each
row, before any generation-based filtering.

`python scripts/rq.py rq1.extract-direction` follows Arditi et al.'s selection
procedure. It derives every post-instruction offset from the loaded Gemma chat
template, computes a difference-in-means candidate at every such position and
every residual-input layer, then evaluates each candidate on held-out data.
For Gemma IT the refusal metric is the paper's log-odds of the next-token mass
on `{235285}` (`I`). The selected direction is the candidate with minimum
harmful ablation/bypass score subject to: harmless addition/induce score > 0,
harmless last-token KL < 0.1 under ablation, and source layer < 0.8L. Addition
is applied at every token position in its selected layer, as in the paper.
It saves a unit vector to `results/rq1/refusal_direction.pt`, selection
metadata to `results/rq1/refusal_direction_metadata.json`, and all validation
rows (including excluded candidates) to `results/rq1/refusal_direction_validation.csv`.

The primary metric is exactly

`M_ref(x) = h[l*, t*](x) dot r_ref_hat`.

The position `t*` may be the final assistant-start newline or any preceding
post-instruction template token. `python scripts/rq.py rq1.run-eapig` builds an EAP graph
that terminates at this residual hook rather than at logits, so all nodes are
attention heads or MLPs upstream of the downstream metric.

Prompts are rendered with the tokenizer's Gemma chat template and then encoded
with `add_special_tokens=False`; the chat template already supplies its control
tokens, so this prevents a duplicate BOS from changing the selected position.

Gemma-2 has grouped-query attention (eight query heads and four key/value
heads). The shared model loader enables TransformerLens's
`ungroup_grouped_query_attention` setting, exposing K/V inputs with the same
eight-head indexing as Q. This is required for the EAP graph's per-head Q/K/V
edge representation and is printed by the smoke test.

## The intervention-aware EAP-IG path

The clean input is a harmful formatted prompt. The corrupt input is the same
token sequence with a global refusal-direction ablation at every residual
input, attention result, and MLP output. This avoids mismatched token lengths.

Because this intervention first changes `blocks.0.hook_resid_pre`, the RQ1
graph uses that post-intervention layer-0 residual activation as its input
source rather than the unchanged token embedding. This is required for a
same-token corrupt baseline to be reconstructible by edge patching. Before any
attribution, `python scripts/rq.py rq1.circuit-closure` enforces the four boundary
conditions: empty/full circuits recover corrupt/clean respectively for both
faithfulness and necessity.

The upstream EAP-IG implementation interpolates between different clean and
corrupt token embeddings. Since RQ1's clean and corrupt text are identical,
that embedding path would have zero length. `src/refusal_circuit/eapig.py`
therefore makes the minimal task-specific adaptation: it caches full
component-output clean/corrupt differences and averages EAP gradients along a
continuous global-ablation-strength path from fully ablated (`alpha=0`) to
normal (`alpha=1`). It is documented as **intervention-aware EAP-IG**, not
silently presented as the original text-pair implementation.

The graph keeps EAP's Q/K/V input edges for attention heads. Thus “head/MLP
edge granularity” means an attention head or MLP output as a source and a
receiving head's Q, K, or V input / a receiving MLP / the terminal residual
metric as a target.

## Environment and smoke test

Create the GPU environment (the YAML specifies CUDA 12.1; change only the
PyTorch CUDA package if the cluster requires another compatible runtime):

```bash
conda env create -f environment.yml
conda activate rq1-refusal-circuit
python scripts/rq.py rq1.smoke
```

The smoke test uses the same `load_model` function and TransformerLens hook
configuration as every experiment. It intentionally loads
`google/gemma-2-2b-it`; it does not fall back to GPT-2. Hugging Face access
must be available and the Gemma licence accepted for the account in use.

The documented local prefix is `.conda/rq1-refusal-circuit`; its installed
packages import `eap` from the local EAP-IG checkout and `refusal_circuit` from
this repository. GPU/model-loading commands are submitted through SLURM rather
than run on the login node.

## Existing EAP-IG code

The upstream demonstration is `EAP-IG/greater_than.ipynb`. The repository's
batch driver is `test_ig_iterations.py`; it loads a TransformerLens model,
builds `eap.graph.Graph`, runs `eap.attribute_mem.attribute`, then evaluates
top-edge graphs with `eap.evaluate_graph.evaluate_graph`. Its CLI requires
`--model`, `--task`, `--attribution_metric`, `--eval_metric`, and
`--batch_size`; its bundled tasks and CSVs are historical GPT-2/Pythia
benchmarks. They are useful installation regressions only, not evidence for
this refusal experiment.

## Run order

```bash
# 1. Filter and persist the required balanced splits.
python scripts/rq.py rq1.prepare-data --batch-size 4

# 2. Train DIM candidates, then select and save the validated direction.
python scripts/rq.py rq1.extract-direction --batch-size 4

# 3. Aggregate-only directional behavioural check; no completions are saved.
python scripts/rq.py rq1.validate-direction --max-new-tokens 32

# 4. Require that empty/full patched circuits reproduce corrupt/clean baselines.
python scripts/rq.py rq1.circuit-closure --num-prompts 8

# 5. One-prompt, one-step numerical attribution calibration.
python scripts/rq.py rq1.run-eapig --num-prompts 1 --ig-steps 1 --output-dir results/rq1/calibration

# 6. Small pilot: 16 harmful test prompts and three path steps.
python scripts/rq.py rq1.run-eapig --num-prompts 16 --ig-steps 3

# 7. Full run: 64 harmful test prompts and five path steps.
python scripts/rq.py rq1.run-eapig --num-prompts 64 --ig-steps 5

# 8. Circuit-only faithfulness, circuit ablation necessity, and matched random
#    edge sets. EAP is included when its score files/circuits exist.
python scripts/rq.py rq1.evaluate --num-prompts 64
```

The fourth command overwrites the pilot score paths by design. Preserve a pilot
by copying `results/rq1/` to a timestamped external artifact directory before
the full run, or pass a different `--output-dir` to both the attribution and
evaluation commands.

## Outputs and interpretation

The full protocol creates:

- `results/rq1/refusal_direction.pt` and `refusal_direction_metadata.json`;
- `eapig_edge_scores.parquet` and, by default, `eap_edge_scores.parquet`;
- `circuits/eapig_top_{50,100,200,500,1000,2000}.json` (and EAP equivalents);
- `faithfulness.csv`, `necessity.csv`, and their two curve plots;
- `baseline_closure.csv`, a required pre-attribution clean/corrupt boundary check;
- `refusal_direction_behaviour.csv`, a separate aggregate-only direction check;
- `top_edges_summary.md`; and
- `behavioural_sanity.csv`, an intentionally small aggregate-only check. It
  records refusal substring rate and first-token margin for normal,
  direction-ablated, circuit-only, and circuit-ablated conditions but never
  writes harmful completions to disk. It uses the EAP-IG top-500 circuit by
  default; choose another available size with `--behaviour-k`.

For a circuit `C`, the evaluator reports

`F(C) = (M_circuit_only - M_corrupt) / (M_clean - M_corrupt)`

and

`N(C) = (M_clean - M_circuit_ablated) / (M_clean - M_corrupt)`.

Random rows have five fixed matched-cardinality samples by default. The
primary result is the projection metric; logit margins and generated-prefix
rates are secondary behavioural sanity checks. Compare EAP-IG against random
and EAP at a fixed requested k. Avoid interpreting a high F alone as unique
mechanistic necessity: redundancy can produce high sufficiency but low N.

## Attribution and provenance

The RQ1 implementation is new code under `src/refusal_circuit`, built on the
local `EAP-IG/eap/graph.py` graph representation. It adapts the published
refusal-direction repository's Gemma prompt format, difference-in-means
candidate extraction, all-layer directional ablation/addition design, and
refusal-prefix matcher. The reference checkout is ignored and pinned in
`external/README.md`; it is never imported by RQ1 runtime code.

The supplied papers were reviewed before implementation: Arditi et al. motivate
the DIM direction and causal interventions; Hanna et al. motivate reporting
faithfulness rather than circuit overlap. The later single-direction,
refusal-cone, and SAE papers motivate the deliberately cautious scope above and
are not implemented as RQ2/RQ3 or SAE features here.

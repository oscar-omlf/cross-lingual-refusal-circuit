# Compressed research handoff context

Last updated: 2026-07-19 (Europe/Amsterdam)

## Repository and operating rules

- Workspace: `/home/omiro1/cross-lingual-refusal-circuit`
- GitHub remote: `https://github.com/oscar-omlf/cross-lingual-refusal-circuit/`
- Upstream EAP-IG project: `https://github.com/hannamw/eap-ig`
- Current checked-out branch: `persona-vectors-circuits`
- Current local tip: `47c6ecc Add persona circuit behavioral pilot`
- The branch is one local commit ahead of `origin/persona-vectors-circuits`; the new pilot commit has not been pushed.
- `origin/main` already contains the cleaned RQ1/RQ2 repository through merge commit `b60a891`.
- Do not put Hugging Face or API tokens in committed files or logs. No token is included here.
- Do not save generated harmful completions. Existing behavioral scripts retain generations in RAM and save only aggregate/instance labels without text.
- Use SLURM for all GPU/model-loading work, not the login node.
- Before inspecting/loading HPC modules for GPU work:

  ```bash
  module purge
  module load 2025
  ```

- Usual environment setup:

  ```bash
  module load CUDA/12.8.0
  export RQ_VENV=/home/omiro1/cross-lingual-refusal-circuit/.venv
  test -x "$RQ_VENV/bin/python"
  ```

- The environment is managed by UV 0.11.29 with Python 3.10.20 and the
  committed `uv.lock`. Rebuild it with
  `/home/omiro1/.local/bin/uv sync --locked` on the login node.

## Research objective and method

The project studies edge-level circuits that causally build behavioral activation directions.

For a normalized direction \(\hat r\), target layer \(\ell^*\), and assistant-start position \(t^*\), the primary scalar metric is

\[
M(x)=h_{\ell^*,t^*}(x)^\top \hat r.
\]

The repository extends EAP-IG to a same-prompt intervention path. The clean and corrupt inputs have identical tokens; corrupt activations come from global directional ablation. EAP-IG integrates edge gradients along ablation strength from fully ablated to clean. Faithfulness and necessity are

\[
F(C)=\frac{M_{\mathrm{circuit\ only}}-M_{\mathrm{corrupt}}}
{M_{\mathrm{clean}}-M_{\mathrm{corrupt}}},
\qquad
N(C)=\frac{M_{\mathrm{clean}}-M_{\mathrm{circuit\ ablated}}}
{M_{\mathrm{clean}}-M_{\mathrm{corrupt}}}.
\]

This finds a circuit up to the selected residual representation, unless a logit-level target is explicitly used.

## Completed RQ1: English refusal in Gemma-2-2B-it

- Model: `google/gemma-2-2b-it`
- Direction: harmful minus harmless residual-stream difference in means.
- Selected site: `blocks.14.hook_resid_pre`, position `-2`.
- Selection followed Arditi-style causal constraints: refusal bypass after ablation, refusal induction after addition, benign KL threshold, and a non-late-layer constraint.
- Saved vector: `results/gemma-2-2b-it/en/rq1/refusal_direction.pt`
- Saved circuits: `results/gemma-2-2b-it/en/rq1/circuits/`
- Key behavior rates:
  - normal harmful refusal: `1.000`
  - global refusal-direction ablation: `0.000`
  - target-site-only direction ablation: `0.969`
  - circuit-only: `0.969`
  - circuit-ablated: `0.094`
- This supports a behaviorally causal refusal circuit in Gemma, while later causal-site work showed that the scalar endpoint alone is not the entire behavioral mechanism.

## Completed RQ2: multilingual refusal in Qwen2.5-7B-Instruct

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Dataset: PolyRefuse.
- Main languages: English, German, Chinese, Thai.
- Common metric anchor: layer 15, position `-2`, English refusal vector.
- Direction cosine with English:
  - en `1.000`
  - de `0.964`
  - zh `0.969`
  - th `0.935`
- Local same-anchor EAP-IG circuits at requested \(k=1000\):
  - en faithfulness/necessity `0.862/1.002`
  - de `0.889/1.003`
  - zh `0.862/1.005`
  - th `0.914/0.997`
- Pairwise same-anchor circuit IoU at \(k=1000\):
  - en-de `0.685`
  - en-zh `0.764`
  - en-th `0.658`
  - de-zh `0.679`
  - de-th `0.753`
  - zh-th `0.668`
- Consensus \(k=1000\) circuit faithfulness/necessity:
  - en `0.926/1.004`
  - de `0.863/1.006`
  - zh `0.904/1.000`
  - th `0.877/1.009`
- WildGuard consensus-circuit harmful-prompt refusal rates:
  - normal `1.000` in all languages
  - direction ablated `0.000` in all languages
  - circuit-only `1.000` in all languages
  - circuit-ablated: en `0.125`, de `0.250`, zh `0.375`, th `0.375`
- Larger English projection circuits (requested k 5k/10k/20k) reduced circuit-ablated refusal to `0.000` while circuit-only stayed `1.000`.
- A separate logit-targeted EAP-IG pilot also found strong behaviorally causal refusal circuits. Projection and logit score artifacts were saved separately.
- Consolidated main-branch results document: `docs/RESULTS_SUMMARY.md` (available on `main`, not necessarily on the current persona branch).

## Persona/evil branch audit

The co-author branch is `persona-vectors-circuits`. It is based on older commit `467f5b0`, before the cleaned main branch, so do not blindly merge it into `main`; port/rebase its useful changes later.

### What the co-author implemented

- Trait actually studied: `evil`, not generic toxicity.
- Model: `google/gemma-2-2b-it`, matching RQ1.
- Source prompts: positive/negative evil persona instructions from the official Persona Vectors artifacts.
- Because Gemma has no system role, persona instructions were folded into the user turn.
- Vector extraction:

  \[
  v_\ell=\mathbb{E}[h_{\ell,-1}\mid\text{evil instruction}]
  -\mathbb{E}[h_{\ell,-1}\mid\text{ethical instruction}].
  \]

- Selected site: layer 13, position `-1` (`blocks.13.hook_resid_pre`).
- Selected by maximum validation Cohen's d, with layers limited to the first 80% of the model.
- Selected validation Cohen's d: `13.0116`.
- Positive/negative projection means: `45.1209/-33.7293`.
- Direction artifact: `results/persona/evil/persona_direction.pt`
- Per-layer raw stack: `results/persona/evil/evil_prompt_last_diff.pt`
- Circuit metric:

  \[
  M_{\mathrm{evil}}(x)=h_{13,-1}(x)^\top\hat v_{\mathrm{evil}}.
  \]

- Circuit discovery used 32 persona-positive test prompts, EAP-IG with 5 IG steps, and also plain EAP.
- Corrupt baseline: same prompt with the selected evil direction globally ablated from every layer's residual input, attention output, and MLP output.

### Relationship to Anthropic Persona Vectors

This is a paper-inspired prompt-last variant, not a faithful reproduction of the paper's main vector.

- Shared with the paper: contrastive difference in means and the official positive/negative trait prompts. The official code computes and saves a `prompt_last_diff` variant.
- Different from the main paper method:
  - Anthropic generates responses under positive and negative persona prompts.
  - It filters for coherent responses that actually express/suppress the trait using a judge.
  - It averages activations across response tokens (`response_avg_diff`) for the main vector.
  - It chooses the layer by causal steering effectiveness.
  - It studies Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, not Gemma.
- The current vector securely represents explicit evil-versus-ethical instruction context; it has not yet been shown to cause evil behavior.
- Dataset split caveat: the five persona instruction pairs recur across splits, and questions overlap heavily between train/validation/test. Circuit discovery and circuit evaluation also used the same 32 test-positive prompts, so saved circuit faithfulness is in-sample.

### Existing evil projection-circuit results

Main EAP-IG rows:

| Requested k | Retained edges | Faithfulness | Necessity |
|---:|---:|---:|---:|
| 200 | 46 | 0.722 | 0.651 |
| 500 | 208 | 0.973 | 0.893 |
| 1000 | 789 | 1.009 | 0.986 |
| 2000 | 1591 | 1.002 | 1.020 |

These establish projection reconstruction/mediation relative to the chosen global-ablation baseline, not evil behavior.

### Evil versus refusal direction

- Selected evil layer-13/-1 vector versus refusal layer-14/-2 vector cosine: `0.1648`.
- Evil layer-14 candidate versus refusal layer-14 vector cosine: `0.2127` (largest across evil layers).
- Same-layer squared shared component is roughly `4.5%`.
- Interpretation: non-random shared safety/harm signal, but clearly not the same direction.
- Refusal is the aligned policy decision to decline a harmful request. An evil persona would more naturally cause malicious compliance, not refusal. A positive cosine can arise because both prompt sets contain harm/ethics semantics, especially because the evil vector was measured before generation.

### Evil versus refusal circuit comparison

EAP-IG recursive-pruned circuit overlap:

| Requested k | Refusal edges | Evil edges | Shared | IoU | Overlap coefficient |
|---:|---:|---:|---:|---:|---:|
| 500 | 135 | 208 | 52 | 0.179 | 0.385 |
| 1000 | 668 | 789 | 299 | 0.258 | 0.448 |
| 2000 | 1398 | 1591 | 676 | 0.292 | 0.484 |

Uniform matched-size random expected IoUs are approximately `0.004`, `0.017`, and `0.037`, respectively.

At requested k=1000:

- terminal projection-edge IoU: `0.658`
- upstream-edge IoU: `0.230`
- full EAP-IG score signed Pearson: `0.855` (dominated by large terminal edges)
- full signed Spearman: `0.024`
- absolute-score Spearman: `0.627`
- upstream-only signed Pearson: `0.345`

Caveat: the terminal sites differ (evil layer 13/-1 versus refusal layer 14/-2), so this is exploratory cross-target overlap rather than a same-anchor circuit equivalence test.

## Current unfinished task: tiny behavioral test of the evil circuit

The user asked for a very small behavioral test that ablates the saved evil circuit before committing to full persona-vector experiments on larger models.

Implemented locally on `persona-vectors-circuits` in commit `47c6ecc`:

- `scripts/persona_behavioral_validation.py`
- `scripts/slurm/persona_behavioral_pilot.sbatch`

The script tests the saved EAP-IG requested-k=1000 circuit (789 retained edges) under four generation conditions:

1. normal
2. global evil-direction ablation
3. circuit-only
4. circuit-ablated

It also computes projection values for normal, global direction-ablated, target-site direction-ablated, circuit-only, and circuit-ablated conditions.

Behavioral judges:

- `allenai/wildguard` for harmful-request, refusal, and harmful-response labels.
- `Qwen/Qwen2.5-7B-Instruct` as a local preliminary judge using Anthropic's evil-trait rubric (0–100 or REFUSAL).
- Both judges are loaded only after Gemma is released; models are sequential to fit GPU memory.
- Prompts, completions, and raw judge outputs are removed before saving.
- Saved outputs contain aggregate metrics and instance labels without text.

The reduced SLURM pilot uses:

- 2 persona-positive prompts
- 32 greedy generation tokens
- MIG partition
- 20-minute wall-time
- classifier batch size 2

Expected output directory:

`results/persona/evil/behavioral_pilot/eapig_top_1000_test_n2/`

Expected files:

- `projection_metrics.csv`
- `behavioral_aggregate.csv`
- `instance_labels_no_text.csv`
- `metadata.json`

### Scheduler blocker

No job is currently running and no job ID was created. Submission was attempted with both a full A100 and a reduced MIG request. Even the two-prompt, 20-minute MIG request was rejected before scheduling with:

```text
sbatch: error: You don't have a sufficiently high budget for the specified partition(s).
sbatch: error: Batch job submission failed: Access/permission denied
```

`sshare` showed roughly 7,500 usage units for account `gpuuva083`; the allocation appears exhausted. All required models were present in the Hugging Face cache:

- `google/gemma-2-2b-it`
- `allenai/wildguard`
- `Qwen/Qwen2.5-7B-Instruct`

Once using an account/allocation with GPU budget, submit from the repository root:

```bash
mkdir -p logs/persona
sbatch scripts/slurm/persona_behavioral_pilot.sbatch
```

Then inspect:

```bash
squeue -u "$USER"
tail -n 100 logs/persona/behavioral_<JOBID>.out
tail -n 100 logs/persona/behavioral_<JOBID>.err
```

If it succeeds, analyze whether circuit ablation lowers `evil_score_mean` and/or `response_harmful_rate` relative to normal, and whether circuit-only retains those behaviors. Because n=2 and the prompts are in-sample, treat it only as a go/no-go diagnostic. If behavior changes in the expected direction, next run a proper held-out behavioral evaluation using the separate Persona Vectors evaluation questions. If behavior does not change, validate/re-extract a response-average, behavior-filtered evil vector before spending resources on larger models.

## Immediate Git actions in the new session

Verify state:

```bash
git status --short --branch
git log -5 --oneline --decorate
```

Expected state:

```text
persona-vectors-circuits...origin/persona-vectors-circuits [ahead 1]
47c6ecc Add persona circuit behavioral pilot
```

Do not merge the persona branch directly into current `main` yet. First run and assess the pilot. Later, rebase or port the persona functionality into the cleaned generic entrypoint structure on a fresh branch from `main`.

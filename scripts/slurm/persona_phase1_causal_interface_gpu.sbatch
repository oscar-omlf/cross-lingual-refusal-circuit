#!/usr/bin/env bash
#SBATCH --job-name=syc-phase1-interface
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq/%x_%j.out
#SBATCH --error=logs/rq/%x_%j.err

set -euo pipefail

module purge
module load 2025
module load CUDA/12.8.0

repo_root="${SLURM_SUBMIT_DIR}"
venv="${RQ_VENV:-${repo_root}/.venv}"
phase1_root="${PHASE1_ROOT:-data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/causal_interface_phase1/run1}"
split_manifest="${PHASE1_SPLIT:-data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/causal_interface_phase1/split_manifest.json}"

cd "${repo_root}"
mkdir -p logs/rq
export PYTHONPATH="${repo_root}/src:${repo_root}/EAP-IG${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${repo_root}/.env"
  set +a
fi
if [[ ! -x "${venv}/bin/python" ]]; then
  echo "Missing pinned environment at ${venv}" >&2
  exit 1
fi

"${venv}/bin/python" scripts/rq.py persona.freeze-causal-interface-split \
  --seed 20260818 \
  --minimum-interface-pairs 100 \
  --output "${split_manifest}"

"${venv}/bin/python" scripts/rq.py persona.run-causal-interface-phase1 \
  --split-manifest "${split_manifest}" \
  --question-indices 0 1 2 3 4 \
  --identity-max-new-tokens 16 \
  --max-new-tokens 192 \
  --max-length 1024 \
  --condition-microbatch-size "${PHASE1_CONDITION_MICROBATCH_SIZE:-2}" \
  --output-root "${phase1_root}" \
  --resume

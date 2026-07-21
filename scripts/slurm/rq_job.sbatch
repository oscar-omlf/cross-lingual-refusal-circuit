#!/usr/bin/env bash
#SBATCH --job-name=rq-job
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq/%x_%j.out
#SBATCH --error=logs/rq/%x_%j.err

# Generic launcher for the refusal-circuit entry point.
#
# Example:
#   sbatch --job-name=rq2-eapig-en \
#     --export=ALL,RQ_COMMAND=rq2.run-eapig,RQ_ARGS='--model Qwen/Qwen2.5-7B-Instruct ...' \
#     scripts/slurm/rq_job.sbatch
#
# For CPU-only jobs, override partition/resources at submission time:
#   sbatch --partition=rome --gres=none --mem=16G --time=00:30:00 ...

set -euo pipefail

module purge
module load 2025
module load CUDA/12.8.0

repo_root="${SLURM_SUBMIT_DIR}"
venv="${RQ_VENV:-${repo_root}/.venv}"
command_name="${RQ_COMMAND:?Set RQ_COMMAND, e.g. rq2.run-eapig}"
command_args="${RQ_ARGS:-}"

cd "${repo_root}"
mkdir -p logs/rq
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Load local credentials without putting secrets in sbatch --export arguments
# or committing them; `.env` is Git-ignored and should be mode 600.
if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${repo_root}/.env"
  set +a
fi

if [[ ! -x "${venv}/bin/python" ]]; then
  echo "Missing UV environment at ${venv}; run 'uv sync --locked' before submission." >&2
  exit 1
fi

# shellcheck disable=SC2086
"${venv}/bin/python" scripts/rq.py "${command_name}" ${command_args}

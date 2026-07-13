#!/usr/bin/env bash
#SBATCH --job-name=rq-job
#SBATCH --account=gpuuva083
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
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${CONDA_ENV_PREFIX:-${repo_root}/.conda/rq1-refusal-circuit}"
command_name="${RQ_COMMAND:?Set RQ_COMMAND, e.g. rq2.run-eapig}"
command_args="${RQ_ARGS:-}"

cd "${repo_root}"
mkdir -p logs/rq
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# shellcheck disable=SC2086
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq.py "${command_name}" ${command_args}

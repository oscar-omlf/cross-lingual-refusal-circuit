#!/usr/bin/env bash
#SBATCH --job-name=persona-prepare
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/persona/prepare_%j.out
#SBATCH --error=logs/persona/prepare_%j.err

# Prompt-based dataset prep loads no model and is CPU-only; it can also be run
# directly on a login node. The Slurm wrapper is provided only for consistency.
set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
TRAIT="${TRAIT:-evil}"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/persona_prepare_dataset.py --trait "${TRAIT}"

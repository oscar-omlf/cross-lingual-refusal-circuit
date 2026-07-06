#!/usr/bin/env bash
#SBATCH --job-name=rq1-smoke
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/rq1/smoke_%j.out
#SBATCH --error=logs/rq1/smoke_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3


repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
# Supply HF_TOKEN through the submission environment; never store it here.
: "${HF_TOKEN:?Export HF_TOKEN before submitting this job (do not put it in this file).}"
export HF_TOKEN
conda run --no-capture-output --prefix "${env_prefix}" python scripts/legacy/rq1_smoke_test.py

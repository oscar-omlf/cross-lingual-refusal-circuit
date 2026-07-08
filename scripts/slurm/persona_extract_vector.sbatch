#!/usr/bin/env bash
#SBATCH --job-name=persona-extract
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=logs/persona/extract_%j.out
#SBATCH --error=logs/persona/extract_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
: "${HF_TOKEN:?Export HF_TOKEN before submitting this job (do not put it in this file).}"
export HF_TOKEN
TRAIT="${TRAIT:-evil}"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/persona_extract_vector.py --trait "${TRAIT}" --batch-size 4

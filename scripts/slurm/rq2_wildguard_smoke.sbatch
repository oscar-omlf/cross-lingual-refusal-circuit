#!/usr/bin/env bash
#SBATCH --job-name=rq2-wildguard-smoke
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rq2/wildguard_smoke_%j.out
#SBATCH --error=logs/rq2/wildguard_smoke_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
cd "${repo_root}"

# ``transformers`` reads a token from the normal Hugging Face cache when the
# account has logged in.  Alternatively submit with an inherited HF_TOKEN;
# never put a token literal in this tracked script.
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_wildguard_smoke_test.py

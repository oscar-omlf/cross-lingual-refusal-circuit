#!/usr/bin/env bash
#SBATCH --job-name=rq2-polyrefuse-data
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rq2/prepare_polyrefuse_%j.out
#SBATCH --error=logs/rq2/prepare_polyrefuse_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
languages="${LANGUAGES:-${LANGUAGE:-en}}"
cd "${repo_root}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Public Qwen model; credentials are never embedded in job files.
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_prepare_polyrefuse.py \
    --model Qwen/Qwen2.5-7B-Instruct --languages ${languages} --batch-size 1 \
    --token-protocol paper_qwen7b \
    --allow-filter-shortfall-splits val test \
    --min-filter-fraction 0.75

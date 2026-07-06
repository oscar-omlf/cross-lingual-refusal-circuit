#!/usr/bin/env bash
#SBATCH --job-name=rq2-refusal-starters
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rq2/refusal_starters_%j.out
#SBATCH --error=logs/rq2/refusal_starters_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
languages="${LANGUAGES:-de ru zh yo ar}"
token_source="${TOKEN_SOURCE:-project}"
cd "${repo_root}"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/legacy/rq2_inspect_refusal_starters.py \
    --model Qwen/Qwen2.5-7B-Instruct --languages ${languages} --label harmful --split val --batch-size 8 \
    --token-source "${token_source}"

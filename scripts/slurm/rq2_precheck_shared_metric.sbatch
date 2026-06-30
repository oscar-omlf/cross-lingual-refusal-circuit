#!/usr/bin/env bash
#SBATCH --job-name=rq2-metric-check
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rq2/shared_metric_precheck_%j.out
#SBATCH --error=logs/rq2/shared_metric_precheck_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
language="${LANGUAGE:?Submit with --export=ALL,LANGUAGE=<code>}"
metric_name="${METRIC_NAME:-english_vector}"
num_prompts="${NUM_PROMPTS:-64}"

cd "${repo_root}"
mkdir -p logs/rq2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_precheck_shared_metric.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --language "${language}" \
    --metric-name "${metric_name}" \
    --split test \
    --num-prompts "${num_prompts}" \
    --batch-size 1

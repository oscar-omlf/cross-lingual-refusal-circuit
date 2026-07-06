#!/usr/bin/env bash
#SBATCH --job-name=rq1-heldout-faith
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rq1/heldout_faithfulness_%j.out
#SBATCH --error=logs/rq1/heldout_faithfulness_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3
repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
: "${HF_TOKEN:?Export HF_TOKEN before submitting this job (do not put it in this file).}"
export HF_TOKEN
conda run --no-capture-output --prefix "${env_prefix}" python scripts/legacy/rq1_evaluate_faithfulness.py \
  --split test --num-prompts 64 --batch-size 4 --circuits-dir results/gemma-2-2b-it/en/rq1/heldout_train/circuits \
  --output-dir results/gemma-2-2b-it/en/rq1/heldout_test --skip-behaviour

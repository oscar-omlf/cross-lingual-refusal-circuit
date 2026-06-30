#!/usr/bin/env bash
#SBATCH --job-name=rq2-direction-transfer
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rq2/direction_transfer_%j.out
#SBATCH --error=logs/rq2/direction_transfer_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
target_language="${TARGET_LANGUAGE:?Submit with --export=ALL,TARGET_LANGUAGE=<code>}"
source_languages="${SOURCE_LANGUAGES:-en de zh th}"
direction_subdir="${DIRECTION_SUBDIR:-direction_paper_kl0p2}"
cd "${repo_root}"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_evaluate_direction_transfer.py \
    --model Qwen/Qwen2.5-7B-Instruct --target-language "${target_language}" \
    --source-languages ${source_languages} --split test --num-prompts 64 --batch-size 1 \
    --direction-subdir "${direction_subdir}"

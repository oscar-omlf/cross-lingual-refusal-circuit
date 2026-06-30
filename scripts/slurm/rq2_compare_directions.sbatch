#!/usr/bin/env bash
#SBATCH --job-name=rq2-dir-compare
#SBATCH --account=gpuuva083
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rq2/compare_directions_%j.out
#SBATCH --error=logs/rq2/compare_directions_%j.err

set -euo pipefail
module purge
module load 2025
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
languages="${LANGUAGES:-en de ru zh}"
anchor_language="${ANCHOR_LANGUAGE:-en}"
direction_subdir="${DIRECTION_SUBDIR:-direction}"
extra_args=()
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  extra_args+=(--output-dir "${OUTPUT_DIR}")
fi
cd "${repo_root}"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_compare_multilingual_directions.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --languages ${languages} \
    --anchor-language "${anchor_language}" \
    --direction-subdir "${direction_subdir}" \
    "${extra_args[@]}"

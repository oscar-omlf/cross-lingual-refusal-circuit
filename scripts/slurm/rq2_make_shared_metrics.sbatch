#!/usr/bin/env bash
#SBATCH --job-name=rq2-shared-metrics
#SBATCH --account=gpuuva083
#SBATCH --partition=rome
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/rq2/shared_metrics_%j.out
#SBATCH --error=logs/rq2/shared_metrics_%j.err

set -euo pipefail
module purge
module load 2025
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
languages="${LANGUAGES:-en de zh th}"
metrics="${METRICS:-english_vector common_mean}"
direction_subdir="${DIRECTION_SUBDIR:-direction_paper_kl0p2}"

cd "${repo_root}"
mkdir -p logs/rq2
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_make_shared_metrics.py \
    --languages ${languages} \
    --metrics ${metrics} \
    --direction-subdir "${direction_subdir}"

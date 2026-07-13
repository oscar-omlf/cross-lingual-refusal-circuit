#!/usr/bin/env bash
#SBATCH --job-name=rq2-circuit-compare
#SBATCH --account=gpuuva083
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rq2/compare_circuits_%j.out
#SBATCH --error=logs/rq2/compare_circuits_%j.err

set -euo pipefail
module purge
module load 2025
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
languages="${LANGUAGES:-en de ru zh}"
method="${METHOD:-eapig}"
cd "${repo_root}"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/legacy/rq2_compare_multilingual_circuits.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --languages ${languages} \
    --method "${method}"

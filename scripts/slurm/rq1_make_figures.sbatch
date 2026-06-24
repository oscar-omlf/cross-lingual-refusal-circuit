#!/usr/bin/env bash
#SBATCH --job-name=rq1-figures
#SBATCH --account=gpuuva083
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/rq1/figures_%j.out
#SBATCH --error=logs/rq1/figures_%j.err

set -euo pipefail
module purge
module load 2025
module load Miniforge3/25.3.0-3
repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
conda run --no-capture-output --prefix "${env_prefix}" python scripts/rq1_make_figures.py

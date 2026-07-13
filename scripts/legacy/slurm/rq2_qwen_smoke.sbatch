#!/usr/bin/env bash
#SBATCH --job-name=rq2-qwen-smoke
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rq2/qwen_smoke_%j.out
#SBATCH --error=logs/rq2/qwen_smoke_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
cd "${repo_root}"
# Qwen2.5 is public; no token is stored in this script or needed for this job.
conda run --no-capture-output --prefix "${env_prefix}" python -m py_compile \
  scripts/rq2_prepare_polyrefuse.py \
  scripts/rq2_extract_multilingual_direction.py \
  scripts/rq2_compare_multilingual_directions.py \
  scripts/rq2_evaluate_direction_transfer.py \
  scripts/rq2_compare_multilingual_circuits.py \
  scripts/rq2_evaluate_cross_language_circuit.py \
  src/refusal_circuit/polyrefuse.py
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/legacy/rq1_smoke_test.py --model Qwen/Qwen2.5-7B-Instruct

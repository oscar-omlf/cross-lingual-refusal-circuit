#!/usr/bin/env bash
#SBATCH --job-name=rq2-eapig
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq2/circuit_discovery_%j.out
#SBATCH --error=logs/rq2/circuit_discovery_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
language="${LANGUAGE:-en}"
base="results/qwen2.5-7b-instruct/polyrefuse/${language}"
data="data/processed/qwen2.5-7b-instruct/polyrefuse/${language}"
cd "${repo_root}"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/legacy/rq1_run_eapig.py \
    --model Qwen/Qwen2.5-7B-Instruct --split train --num-prompts 64 --batch-size 1 --ig-steps 5 \
    --methods eapig eap --data-dir "${data}" \
    --direction-path "${base}/direction/anchor_refusal_direction.pt" \
    --metadata-path "${base}/direction/anchor_direction_metadata.json" \
    --output-dir "${base}/circuit_discovery"

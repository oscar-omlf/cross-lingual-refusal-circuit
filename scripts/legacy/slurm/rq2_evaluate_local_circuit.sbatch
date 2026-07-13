#!/usr/bin/env bash
#SBATCH --job-name=rq2-circuit-eval
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rq2/local_circuit_eval_%j.out
#SBATCH --error=logs/rq2/local_circuit_eval_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
language="${LANGUAGE:-en}"
k_values="${K_VALUES:-50 100 200 500 1000 2000}"
num_prompts="${NUM_PROMPTS:-64}"
circuits_subdir="${CIRCUITS_SUBDIR:-circuits}"
output_subdir="${OUTPUT_SUBDIR:-circuit_evaluation}"
random_seeds="${RANDOM_SEEDS:-0 1 2 3 4}"
methods="${METHODS:-eapig}"
base="results/qwen2.5-7b-instruct/polyrefuse/${language}"
data="data/processed/qwen2.5-7b-instruct/polyrefuse/${language}"
cd "${repo_root}"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/legacy/rq1_evaluate_faithfulness.py \
    --model Qwen/Qwen2.5-7B-Instruct --split test --num-prompts "${num_prompts}" --batch-size 1 \
    --data-dir "${data}" --direction-path "${base}/direction/anchor_refusal_direction.pt" \
    --metadata-path "${base}/direction/anchor_direction_metadata.json" \
    --circuits-dir "${base}/circuit_discovery/${circuits_subdir}" \
    --output-dir "${base}/${output_subdir}" \
    --methods ${methods} \
    --k-values ${k_values} \
    --random-seeds ${random_seeds} \
    --skip-behaviour

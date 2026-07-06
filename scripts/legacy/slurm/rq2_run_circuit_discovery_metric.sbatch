#!/usr/bin/env bash
#SBATCH --job-name=rq2-eapig-metric
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq2/circuit_discovery_metric_%j.out
#SBATCH --error=logs/rq2/circuit_discovery_metric_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
language="${LANGUAGE:-en}"
metric_name="${METRIC_NAME:-english_vector}"
num_prompts="${NUM_PROMPTS:-64}"
ig_steps="${IG_STEPS:-5}"
methods="${METHODS:-eapig}"
k_values="${K_VALUES:-10 25 50 100 200 500 1000 2000}"
base="results/qwen2.5-7b-instruct/polyrefuse/${language}"
data="data/processed/qwen2.5-7b-instruct/polyrefuse/${language}"
direction_path="${DIRECTION_PATH:-${base}/shared_metrics/${metric_name}/refusal_direction.pt}"
metadata_path="${METADATA_PATH:-${base}/shared_metrics/${metric_name}/metadata.json}"
output_dir="${OUTPUT_DIR:-${base}/circuit_discovery_${metric_name}}"

cd "${repo_root}"
mkdir -p logs/rq2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/legacy/rq1_run_eapig.py \
    --model Qwen/Qwen2.5-7B-Instruct --split train --num-prompts "${num_prompts}" --batch-size 1 \
    --ig-steps "${ig_steps}" --methods ${methods} --k-values ${k_values} \
    --data-dir "${data}" \
    --direction-path "${direction_path}" \
    --metadata-path "${metadata_path}" \
    --output-dir "${output_dir}" \
    --skip-paper-circuits

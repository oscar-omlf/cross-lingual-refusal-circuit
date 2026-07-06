#!/usr/bin/env bash
#SBATCH --job-name=rq2-logit-behavior
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rq2/logit_behavioral_validation_%j.out
#SBATCH --error=logs/rq2/logit_behavioral_validation_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
language="${LANGUAGE:-en}"
metric_name="${METRIC_NAME:-english_vector}"
k="${K:-1000}"
num_prompts="${NUM_PROMPTS:-32}"
prompt_offset="${PROMPT_OFFSET:-32}"
generation_prompts="${GENERATION_PROMPTS:-8}"
max_new_tokens="${MAX_NEW_TOKENS:-32}"
batch_size="${BATCH_SIZE:-1}"
circuit_dir="${CIRCUIT_DIR:-results/qwen2.5-7b-instruct/polyrefuse/en/circuit_discovery_logit_english_vector_pilot/circuits_greedy_no_parent_prune}"
output_dir="${OUTPUT_DIR:-}"

cd "${repo_root}"
mkdir -p logs/rq2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

extra_args=()
if [[ -n "${output_dir}" ]]; then
  extra_args+=(--output-dir "${output_dir}")
fi

conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_behavioral_validation_logit_circuit.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --language "${language}" \
    --metric-name "${metric_name}" \
    --k "${k}" \
    --split test \
    --num-prompts "${num_prompts}" \
    --prompt-offset "${prompt_offset}" \
    --generation-prompts "${generation_prompts}" \
    --max-new-tokens "${max_new_tokens}" \
    --batch-size "${batch_size}" \
    --circuit-dir "${circuit_dir}" \
    "${extra_args[@]}"

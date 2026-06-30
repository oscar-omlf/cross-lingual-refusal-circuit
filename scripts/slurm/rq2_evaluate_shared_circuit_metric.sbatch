#!/usr/bin/env bash
#SBATCH --job-name=rq2-shared-eval
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rq2/shared_circuit_eval_%j.out
#SBATCH --error=logs/rq2/shared_circuit_eval_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
language="${LANGUAGE:?Submit with --export=ALL,LANGUAGE=<code>}"
metric_name="${METRIC_NAME:-english_vector}"
shared_kind="${SHARED_KIND:-intersection_en_de_zh_th}"
circuits_subdir="${CIRCUITS_SUBDIR:-circuits_greedy_no_parent_prune_small_to_large}"
k_values="${K_VALUES:-10 25 50 100 200 500 1000 2000}"
num_prompts="${NUM_PROMPTS:-32}"
prompt_offset="${PROMPT_OFFSET:-0}"
random_seeds="${RANDOM_SEEDS:-0}"
methods="${METHODS:-eapig}"
base="results/qwen2.5-7b-instruct/polyrefuse/${language}"
data="data/processed/qwen2.5-7b-instruct/polyrefuse/${language}"
direction_path="${DIRECTION_PATH:-${base}/shared_metrics/${metric_name}/refusal_direction.pt}"
metadata_path="${METADATA_PATH:-${base}/shared_metrics/${metric_name}/metadata.json}"
circuits_dir="${CIRCUITS_DIR:-results/qwen2.5-7b-instruct/polyrefuse/shared_circuits/circuit_discovery_${metric_name}/${shared_kind}/${circuits_subdir}}"
output_dir="${OUTPUT_DIR:-results/qwen2.5-7b-instruct/polyrefuse/shared_circuit_evaluation_${metric_name}/${shared_kind}/${language}}"

cd "${repo_root}"
mkdir -p logs/rq2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq1_evaluate_faithfulness.py \
    --model Qwen/Qwen2.5-7B-Instruct --split test --num-prompts "${num_prompts}" \
    --prompt-offset "${prompt_offset}" --batch-size 1 \
    --data-dir "${data}" --direction-path "${direction_path}" \
    --metadata-path "${metadata_path}" \
    --circuits-dir "${circuits_dir}" \
    --output-dir "${output_dir}" \
    --methods ${methods} \
    --k-values ${k_values} \
    --random-seeds ${random_seeds} \
    --skip-behaviour

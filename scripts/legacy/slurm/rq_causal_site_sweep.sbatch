#!/usr/bin/env bash
#SBATCH --job-name=rq-causal-site
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq_causal_site_sweep_%j.out
#SBATCH --error=logs/rq_causal_site_sweep_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"

model="${MODEL:?Submit with --export=ALL,MODEL=<hf-model>}"
data_dir="${DATA_DIR:?Submit with --export=ALL,DATA_DIR=<processed-data-dir>}"
direction_path="${DIRECTION_PATH:?Submit with --export=ALL,DIRECTION_PATH=<refusal_direction.pt>}"
metadata_path="${METADATA_PATH:?Submit with --export=ALL,METADATA_PATH=<metadata.json>}"
output_dir="${OUTPUT_DIR:?Submit with --export=ALL,OUTPUT_DIR=<output-dir>}"

split="${SPLIT:-test}"
num_prompts="${NUM_PROMPTS:-32}"
prompt_offset="${PROMPT_OFFSET:-0}"
batch_size="${BATCH_SIZE:-4}"
max_length="${MAX_LENGTH:-512}"
dtype="${DTYPE:-bfloat16}"
labels="${LABELS:-harmful harmless}"
positions="${POSITIONS:--5 -4 -3 -2 -1}"
layers="${LAYERS:-all}"

cd "${repo_root}"
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# shellcheck disable=SC2206
label_args=(${labels})
# shellcheck disable=SC2206
position_args=(${positions})
# shellcheck disable=SC2206
layer_args=(${layers})

conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/legacy/rq_causal_site_sweep.py \
    --model "${model}" \
    --data-dir "${data_dir}" \
    --direction-path "${direction_path}" \
    --metadata-path "${metadata_path}" \
    --output-dir "${output_dir}" \
    --split "${split}" \
    --num-prompts "${num_prompts}" \
    --prompt-offset "${prompt_offset}" \
    --batch-size "${batch_size}" \
    --max-length "${max_length}" \
    --dtype "${dtype}" \
    --labels "${label_args[@]}" \
    --positions "${position_args[@]}" \
    --layers "${layer_args[@]}"

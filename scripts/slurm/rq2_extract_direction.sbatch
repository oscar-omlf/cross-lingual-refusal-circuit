#!/usr/bin/env bash
#SBATCH --job-name=rq2-direction
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rq2/extract_direction_%j.out
#SBATCH --error=logs/rq2/extract_direction_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
language="${LANGUAGE:-en}"
batch_size="${BATCH_SIZE:-4}"
kl_threshold="${KL_THRESHOLD:-0.2}"
late_layer_fraction="${LATE_LAYER_FRACTION:-0.8}"
output_dir="${OUTPUT_DIR:-}"
cd "${repo_root}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
extra=()
if [[ -n "${output_dir}" ]]; then
  extra+=(--output-dir "${output_dir}")
fi
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_extract_multilingual_direction.py \
    --model Qwen/Qwen2.5-7B-Instruct --language "${language}" --batch-size "${batch_size}" \
    --kl-threshold "${kl_threshold}" --late-layer-fraction "${late_layer_fraction}" \
    "${extra[@]}"

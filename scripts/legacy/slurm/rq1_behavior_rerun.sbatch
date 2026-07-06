#!/usr/bin/env bash
#SBATCH --job-name=rq1-behavior-rerun
#SBATCH --account=gpuuva083
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rq1/behavior_rerun_%j.out
#SBATCH --error=logs/rq1/behavior_rerun_%j.err

set -euo pipefail
module purge
module load 2025
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
k="${K:-1000}"
num_prompts="${NUM_PROMPTS:-32}"
max_new_tokens="${MAX_NEW_TOKENS:-32}"
split="${SPLIT:-test}"
base="results/gemma-2-2b-it/en/rq1"
circuit_path="${CIRCUIT_PATH:-${base}/circuits/eapig_top_${k}.json}"
output_path="${OUTPUT_PATH:-${base}/behavioral_rerun/eapig_top_${k}_${split}_n${num_prompts}_gen${max_new_tokens}.csv}"

cd "${repo_root}"
mkdir -p logs/rq1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/legacy/rq1_behavioral_sanity.py \
    --model google/gemma-2-2b-it \
    --split "${split}" \
    --num-prompts "${num_prompts}" \
    --max-new-tokens "${max_new_tokens}" \
    --circuit-path "${circuit_path}" \
    --direction-path "${base}/refusal_direction.pt" \
    --metadata-path "${base}/refusal_direction_metadata.json" \
    --data-dir data/processed/gemma-2-2b-it/en/rq1 \
    --output-path "${output_path}"

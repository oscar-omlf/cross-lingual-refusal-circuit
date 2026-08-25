#!/usr/bin/env bash
#SBATCH --job-name=ref_v1_faith
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ref_v1_faith_%j.out
#SBATCH --error=logs/ref_v1_faith_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p logs
for k in 100 500 1000 3000; do
  .venv/bin/python scripts/run_tests.py \
    --config configs/refusal_final_v1_en.json \
    --circuit-path "results/qwen2.5-7b-instruct/refusal/final_v1/en/eapig/circuits/eapig_top_${k}.json" \
    --output-path "results/qwen2.5-7b-instruct/refusal/final_v1/en/faithfulness/k${k}.json"
done

#!/usr/bin/env bash
#SBATCH --job-name=ref_v1_vector
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ref_v1_vector_%j.out
#SBATCH --error=logs/ref_v1_vector_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p logs
.venv/bin/python scripts/extract_vector.py \
  --config configs/refusal_final_v1_en.json

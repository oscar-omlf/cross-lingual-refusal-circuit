#!/usr/bin/env bash
#SBATCH --job-name=syc_prompt_screen
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/syc_prompt_screen_%j.out
#SBATCH --error=logs/syc_prompt_screen_%j.err

set -euo pipefail

REPO_DIR="/home/omiro1/cross-lingual-refusal-circuit-prompt-vector"
cd "$REPO_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p logs
nvidia-smi
"$HOME/.local/bin/uv" run --locked python \
  scripts/sycophancy_prompt_location_screen.py

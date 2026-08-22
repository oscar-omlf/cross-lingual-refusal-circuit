#!/usr/bin/env bash
#SBATCH --job-name=syc_prompt_vector
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/syc_prompt_vector_%j.out
#SBATCH --error=logs/syc_prompt_vector_%j.err

set -euo pipefail

REPO_DIR="/home/omiro1/cross-lingual-refusal-circuit-prompt-vector"
cd "$REPO_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nvidia-smi
"$HOME/.local/bin/uv" run --locked python scripts/extract_vector.py \
  --config configs/sycophancy_prompt.json

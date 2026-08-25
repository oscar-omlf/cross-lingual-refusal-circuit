#!/bin/bash
#SBATCH --job-name=syc_modes_l20
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/syc_modes_l20_%j.out
#SBATCH --error=logs/syc_modes_l20_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-sycophancy-causal-modes-20260823
mkdir -p logs
export PYTHONPATH="$PWD/src:$PWD"

.venv/bin/python scripts/run_sycophancy_causal_modes.py extract \
  --num-prompts 256 \
  --screen-prompts 64 \
  --candidate-layers 20 \
  --ig-steps 5 \
  --max-modes 16 \
  --max-length 1024 \
  --condition-batch-size 8 \
  --output-dir results/qwen2.5-7b-instruct/sycophancy/generation/causal_modes_layer20_anchored

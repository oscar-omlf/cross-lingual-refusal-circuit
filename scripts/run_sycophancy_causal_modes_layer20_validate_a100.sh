#!/bin/bash
#SBATCH --job-name=syc_modes_l20_val
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=logs/syc_modes_l20_val_%j.out
#SBATCH --error=logs/syc_modes_l20_val_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-sycophancy-causal-modes-20260823
mkdir -p logs
export PYTHONPATH="$PWD/src:$PWD"

.venv/bin/python scripts/run_sycophancy_causal_modes.py validate \
  --num-prompts 64 \
  --max-length 1024 \
  --condition-batch-size 8 \
  --random-seeds 0,1,2,3 \
  --bootstrap-samples 10000 \
  --extraction-dir results/qwen2.5-7b-instruct/sycophancy/generation/causal_modes_layer20_anchored \
  --output-dir results/qwen2.5-7b-instruct/sycophancy/generation/causal_modes_layer20_anchored/heldout_validation

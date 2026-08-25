#!/usr/bin/env bash
#SBATCH --job-name=syc_k100_behavior
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/syc_k100_behavior_%A_%a.out
#SBATCH --error=logs/syc_k100_behavior_%A_%a.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p logs
.venv/bin/python scripts/run_sycophancy_circuit_behavior.py \
  --num-shards 10 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --max-new-tokens 256

#!/usr/bin/env bash
#SBATCH --job-name=syc_k3000_pilot
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=logs/syc_k3000_pilot_%j.out
#SBATCH --error=logs/syc_k3000_pilot_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

pilot_args=(
  --num-prompts 16
  --num-shards 1
  --max-new-tokens 256
  --conditions baseline_positive full_projection circuit_ablated
  --circuit results/qwen2.5-7b-instruct/sycophancy/prompt/eapig/circuits/eapig_top_3000.json
  --output-dir results/qwen2.5-7b-instruct/sycophancy/prompt/eapig/behavioral_pilot_k3000_n16
)

.venv/bin/python scripts/run_sycophancy_circuit_behavior.py "${pilot_args[@]}"
.venv/bin/python scripts/run_sycophancy_circuit_behavior.py "${pilot_args[@]}" --aggregate

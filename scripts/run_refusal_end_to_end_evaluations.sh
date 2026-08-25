#!/usr/bin/env bash
#SBATCH --job-name=ref_e2e_eval
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ref_e2e_eval_%A_%a.out
#SBATCH --error=logs/ref_e2e_eval_%A_%a.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

K_VALUES=(1000 3000 10000)
K="${K_VALUES[${SLURM_ARRAY_TASK_ID}]}"

mkdir -p logs
.venv/bin/python scripts/run_tests.py \
  --config configs/refusal_end_to_end_v1_en.json \
  --circuit-path "results/qwen2.5-7b-instruct/refusal/end_to_end_v1/en/eapig/circuits/eapig_top_${K}.json" \
  --output-path "results/qwen2.5-7b-instruct/refusal/end_to_end_v1/en/faithfulness/k${K}.json"

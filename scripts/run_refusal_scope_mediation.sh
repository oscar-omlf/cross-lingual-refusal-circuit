#!/usr/bin/env bash
#SBATCH --job-name=ref_scope_gate
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ref_scope_gate_%j.out
#SBATCH --error=logs/ref_scope_gate_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p logs
.venv/bin/python scripts/run_refusal_scope_mediation.py \
  --config configs/refusal_downstream_v1_en.json \
  --dataset data/processed/qwen2.5-7b-instruct/trait_circuits/refusal/reserve.jsonl \
  --circuit results/qwen2.5-7b-instruct/refusal/final_v1/en/eapig/circuits/eapig_top_1000.json \
  --output results/qwen2.5-7b-instruct/refusal/scope_mediation_v1/en/diagnostic.json \
  --num-examples 16 \
  --max-new-tokens 64 \
  --random-seeds 0,1,2 \
  --skip-generation

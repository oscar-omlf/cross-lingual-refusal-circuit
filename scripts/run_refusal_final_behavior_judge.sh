#!/usr/bin/env bash
#SBATCH --job-name=ref_final_wg
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ref_final_wg_%j.out
#SBATCH --error=logs/ref_final_wg_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

OUTPUT_DIR="results/qwen2.5-7b-instruct/refusal/final_v1/en/behavior_final/k1000"
DATASET="data/processed/qwen2.5-7b-instruct/trait_circuits/refusal/final_behavior.jsonl"
CIRCUIT="results/qwen2.5-7b-instruct/refusal/final_v1/en/eapig/circuits/eapig_top_1000.json"

.venv/bin/python scripts/run_refusal_circuit_behavior.py judge \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --phase final \
  --batch-size 8

.venv/bin/python scripts/run_refusal_circuit_behavior.py aggregate \
  --config configs/refusal_final_v1_en.json \
  --dataset "${DATASET}" \
  --circuit "${CIRCUIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --phase final

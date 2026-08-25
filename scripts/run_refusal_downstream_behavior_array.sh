#!/usr/bin/env bash
#SBATCH --job-name=ref_down_final
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/ref_down_final_%A_%a.out
#SBATCH --error=logs/ref_down_final_%A_%a.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

METRICS_DIR="results/qwen2.5-7b-instruct/refusal/downstream_v1/en/faithfulness"
K="$(.venv/bin/python scripts/select_refusal_downstream_k.py --metrics-dir "${METRICS_DIR}")"
OUTPUT_DIR="results/qwen2.5-7b-instruct/refusal/downstream_v1/en/behavior_final/k${K}"

mkdir -p logs
.venv/bin/python scripts/run_refusal_circuit_behavior.py generate \
  --config configs/refusal_downstream_v1_en.json \
  --dataset data/processed/qwen2.5-7b-instruct/trait_circuits/refusal/final_behavior.jsonl \
  --circuit "results/qwen2.5-7b-instruct/refusal/downstream_v1/en/eapig/circuits/eapig_top_${K}.json" \
  --output-dir "${OUTPUT_DIR}" \
  --phase final \
  --corruption-condition source_projection \
  --max-new-tokens 64 \
  --num-shards 16 \
  --shard-index "${SLURM_ARRAY_TASK_ID}"

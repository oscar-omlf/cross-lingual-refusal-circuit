#!/bin/bash
#SBATCH --job-name=syc_e2e_eval
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/syc_e2e_eval_%A_%a.out
#SBATCH --error=logs/syc_e2e_eval_%A_%a.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
mkdir -p logs

k_values=(500 1000 3000 5000)
k="${k_values[$SLURM_ARRAY_TASK_ID]}"

.venv/bin/python scripts/evaluate_sycophancy_end_to_end_circuit.py \
  --num-prompts 64 \
  --circuit "results/qwen2.5-7b-instruct/sycophancy/end_to_end/eapig_n64_ig5/circuits/eapig_top_${k}.json" \
  --output-dir "results/qwen2.5-7b-instruct/sycophancy/end_to_end/eapig_n64_ig5/circuit_selection/k${k}"

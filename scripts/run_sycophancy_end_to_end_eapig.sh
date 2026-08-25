#!/bin/bash
#SBATCH --job-name=syc_e2e_eapig
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=logs/syc_e2e_eapig_%j.out
#SBATCH --error=logs/syc_e2e_eapig_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
mkdir -p logs

.venv/bin/python scripts/run_sycophancy_end_to_end_eapig.py \
  --num-prompts 64 \
  --ig-steps 5 \
  --max-length 512 \
  --output-dir results/qwen2.5-7b-instruct/sycophancy/end_to_end/eapig_n64_ig5

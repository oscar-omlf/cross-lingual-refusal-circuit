#!/bin/bash
#SBATCH --job-name=syc_e2e_instr
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=logs/syc_e2e_instr_%j.out
#SBATCH --error=logs/syc_e2e_instr_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
mkdir -p logs

.venv/bin/python scripts/validate_sycophancy_instruction_contrast.py \
  --num-prompts 16 \
  --output-dir results/qwen2.5-7b-instruct/sycophancy/end_to_end/instruction_contrast_n16

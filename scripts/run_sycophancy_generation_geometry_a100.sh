#!/bin/bash
#SBATCH --job-name=syc_gen_geometry
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/syc_gen_geometry_%j.out
#SBATCH --error=logs/syc_gen_geometry_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-sycophancy-geometry-20260823
mkdir -p logs

.venv/bin/python scripts/run_sycophancy_generation_geometry.py \
  --num-prompts 64 \
  --max-length 1024 \
  --condition-batch-size 5 \
  --bootstrap-samples 10000 \
  --output-dir results/qwen2.5-7b-instruct/sycophancy/generation/geometry_coordinate_n64

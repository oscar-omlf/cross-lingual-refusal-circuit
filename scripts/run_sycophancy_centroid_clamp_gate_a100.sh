#!/bin/bash
#SBATCH --job-name=syc_affine_gate
#SBATCH --account=gisr129887
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/syc_affine_gate_%j.out
#SBATCH --error=logs/syc_affine_gate_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-sycophancy-geometry-20260823
mkdir -p logs

.venv/bin/python scripts/run_sycophancy_generation_geometry.py \
  --config configs/sycophancy_generation_centroid_clamp.json \
  --num-prompts 64 \
  --max-length 1024 \
  --bootstrap-samples 10000 \
  --clamp-gate-only \
  --clamp-calibration-dataset data/processed/qwen2.5-7b-instruct/trait_circuits/sycophancy/vector_extraction.jsonl \
  --output-dir results/qwen2.5-7b-instruct/sycophancy/generation/centroid_affine_projection_gate_calibrated_n64

#!/usr/bin/env bash
#SBATCH --job-name=syc_k100_aggregate
#SBATCH --account=gisr129887
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/syc_k100_aggregate_%j.out
#SBATCH --error=logs/syc_k100_aggregate_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/run_sycophancy_circuit_behavior.py \
  --num-shards 10 \
  --max-new-tokens 256 \
  --aggregate

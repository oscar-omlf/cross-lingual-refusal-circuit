#!/usr/bin/env bash
#SBATCH --job-name=judge_syc_k3000
#SBATCH --account=gisr129887
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/judge_syc_k3000_%j.out
#SBATCH --error=logs/judge_syc_k3000_%j.err

set -euo pipefail

cd /gpfs/home5/omiro1/cross-lingual-refusal-circuit-prompt-eapig-20260823
set -a
source .env
set +a

common=(--expected-rows 48 --concurrency 20 --checkpoint-rows 48)

.venv/bin/python scripts/judge_sycophancy_responses.py \
  --input results/qwen2.5-7b-instruct/sycophancy/prompt/eapig/behavioral_pilot_k3000_n16/responses.jsonl \
  --output-dir results/qwen2.5-7b-instruct/sycophancy/prompt/eapig/behavioral_pilot_k3000_n16 \
  "${common[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=syc-phase1-judge
#SBATCH --account=gisr129887
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rq/%x_%j.out
#SBATCH --error=logs/rq/%x_%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR}"
venv="${RQ_VENV:-${repo_root}/.venv}"
phase1_root="${PHASE1_ROOT:-data/intermediate/persona/qwen2.5-7b-instruct/sycophantic/causal_interface_phase1/run1}"

cd "${repo_root}"
mkdir -p logs/rq
export PYTHONPATH="${repo_root}/src:${repo_root}/EAP-IG${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -f "${repo_root}/.env" ]]; then
  echo "Missing Git-ignored .env containing OPENAI_API_KEY" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source "${repo_root}/.env"
set +a
if [[ ! -x "${venv}/bin/python" ]]; then
  echo "Missing pinned environment at ${venv}" >&2
  exit 1
fi

"${venv}/bin/python" scripts/rq.py persona.judge \
  --config configs/qwen_sycophancy.json \
  --input "${phase1_root}/responses.jsonl" \
  --output "${phase1_root}/scored.jsonl" \
  --artifact eval \
  --protocol-field behavior_protocol_id \
  --resume

"${venv}/bin/python" scripts/rq.py persona.analyze-causal-interface-phase1 \
  --config configs/qwen_sycophancy.json \
  --scored "${phase1_root}/scored.jsonl" \
  --bootstrap-samples 10000 \
  --seed 20260818 \
  --output "${phase1_root}/aggregate.json"

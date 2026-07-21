#!/usr/bin/env bash
#SBATCH --job-name=syc-s1-judge
#SBATCH --account=gisr129887
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rq/%x_%j.out
#SBATCH --error=logs/rq/%x_%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR}"
venv="${RQ_VENV:-${repo_root}/environments/persona_vectors/.venv}"
cd "${repo_root}"
mkdir -p logs/rq

if [[ ! -f "${repo_root}/.env" ]]; then
  echo "Missing Git-ignored .env containing OPENAI_API_KEY" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source "${repo_root}/.env"
set +a

root="data/intermediate/persona/qwen2.5-7b-instruct/sycophantic"
"${venv}/bin/python" scripts/rq.py persona.judge \
  --artifact extract \
  --input "${root}/generation/pos_responses.jsonl" \
  --output "${root}/judged/pos_scored.jsonl" \
  --resume
"${venv}/bin/python" scripts/rq.py persona.judge \
  --artifact extract \
  --input "${root}/generation/neg_responses.jsonl" \
  --output "${root}/judged/neg_scored.jsonl" \
  --resume

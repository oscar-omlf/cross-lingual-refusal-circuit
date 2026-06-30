#!/usr/bin/env bash
#SBATCH --job-name=rq2-intersections
#SBATCH --account=gpuuva083
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rq2/intersection_circuits_%j.out
#SBATCH --error=logs/rq2/intersection_circuits_%j.err

set -euo pipefail
module purge
module load 2025
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
languages="${LANGUAGES:-en de ru}"
metric_name="${METRIC_NAME:-english_vector}"
k_values="${K_VALUES:-25 50 100 200 500 1000 2000}"
include_pairs="${INCLUDE_PAIRS:-0}"

cd "${repo_root}"
mkdir -p logs/rq2
extra=()
if [[ "${include_pairs}" == "1" ]]; then
  extra+=(--include-pairs)
fi
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_make_intersection_circuits.py \
    --languages ${languages} \
    --k-values ${k_values} \
    --discovery-subdir "circuit_discovery_${metric_name}" \
    "${extra[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=rq2-rebuild-circuits
#SBATCH --account=gpuuva083
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rq2/rebuild_metric_circuits_%j.out
#SBATCH --error=logs/rq2/rebuild_metric_circuits_%j.err

set -euo pipefail
module purge
module load 2025
module load Miniforge3/25.3.0-3

repo_root="${SLURM_SUBMIT_DIR}"
env_prefix="${repo_root}/.conda/rq1-refusal-circuit"
language="${LANGUAGE:-en}"
metric_name="${METRIC_NAME:-english_vector}"
k_values="${K_VALUES:-10 25 50 100 200 500 1000 2000}"
method="${METHOD:-eapig}"
selection="${SELECTION:-greedy_no_parent_prune}"
output_subdir="${OUTPUT_SUBDIR:-circuits_greedy_no_parent_prune_small_to_large}"

cd "${repo_root}"
mkdir -p logs/rq2
conda run --no-capture-output --prefix "${env_prefix}" \
  python scripts/rq2_rebuild_circuits_from_scores.py \
    --languages "${language}" \
    --method "${method}" \
    --k-values ${k_values} \
    --selection "${selection}" \
    --discovery-subdir "circuit_discovery_${metric_name}" \
    --metadata-subpath "shared_metrics/${metric_name}/metadata.json" \
    --output-subdir "${output_subdir}"

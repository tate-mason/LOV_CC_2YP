#!/bin/bash
#SBATCH --job-name=observed_theory
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/observed_theory_%j.log

# Usage: sbatch Code/final_simple/server/run_observed_state_theory.sh [NOVELTY_ONLY_FIT_DIR]
set -euo pipefail
umask 077
server_root="/scratch/dtm63837/Kilts_Panel"
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit this job with sbatch on the research server.' >&2
    exit 1
fi
if [[ $# -gt 1 ]]; then
    echo 'Usage: sbatch run_observed_state_theory.sh [NOVELTY_ONLY_FIT_DIR]' >&2
    exit 1
fi
fit_dir="${1:-$server_root/rebuilds/run_49942107/fits/separate_experiments_50281568/novelty_only}"
output_dir="$server_root/rebuilds/observed_state_theory_${SLURM_JOB_ID}"
ml Python/3.12.3-GCCcore-13.3.0
export PYTHONPYCACHEPREFIX="$server_root/python_cache_observed_theory_${SLURM_JOB_ID}"
python3 "$project_dir/Code/final_simple/server/run_observed_state_theory.py" \
    --fit-dir "$fit_dir" --output "$output_dir"

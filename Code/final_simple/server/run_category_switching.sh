#!/bin/bash
#SBATCH --job-name=category_switch
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/category_switch_%j.log
set -euo pipefail
umask 077
server_root="/scratch/dtm63837/Kilts_Panel"
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch on the research server.' >&2
    exit 1
fi
if [[ $# -gt 2 ]]; then
    echo 'Usage: sbatch run_category_switching.sh [PRIOR_NOVELTY_ONLY_FIT] [PREVIOUS_RESULTS_ONLY]' >&2
    exit 1
fi
fit_dir="${1:-$server_root/rebuilds/run_49942107/fits/separate_experiments_50281568/novelty_only}"
output_dir="$server_root/rebuilds/category_switching_${SLURM_JOB_ID}"
ml Python/3.12.3-GCCcore-13.3.0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$server_root/python_cache_category_${SLURM_JOB_ID}"
previous=()
if [[ $# -eq 2 ]]; then previous=(--previous-results "$2"); fi
python3 "$project_dir/Code/final_simple/server/run_category_switching.py" \
    --fit-dir "$fit_dir" --output "$output_dir" --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "${previous[@]}"

#!/bin/bash
#SBATCH --job-name=yogurt_week_fix
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/observed_week_refit_%j.log

# sbatch [cluster partition overrides] this_script FIT_DIR FULL_PANEL_PARQUET
set -euo pipefail
umask 077
server_root=/scratch/dtm63837/Kilts_Panel
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch on the research server.' >&2
    exit 1
fi
if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 FIT_DIR FULL_PANEL_PARQUET [runner options]" >&2
    exit 1
fi
ml Python/3.12.3-GCCcore-13.3.0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPYCACHEPREFIX="$server_root/python_cache_observed_week_refit"
PYTHONPATH="$project_dir/Code/final_simple" python3 -m unittest -v checks.test_observed_weeks
python3 -u "$project_dir/Code/final_simple/server/run_observed_week_refit.py" "$@"

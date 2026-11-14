#!/bin/bash
#SBATCH --job-name=bundle_stats
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/bundle_stats_%j.log
set -euo pipefail
umask 077
server_root=/scratch/dtm63837/Kilts_Panel
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch on the research server.' >&2
    exit 1
fi
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 EXISTING_SUMMARY_DIRECTORY" >&2
    exit 1
fi
ml Python/3.12.3-GCCcore-13.3.0
python3 "$project_dir/Code/final_simple/server/bundle_statistics.py" \
    --summary-dir "$1" \
    --output "$server_root/results/bundle_stats_${SLURM_JOB_ID}/RESULTS_ONLY"

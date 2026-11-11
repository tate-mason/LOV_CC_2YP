#!/bin/bash
#SBATCH --job-name=novelty_info
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/information_pricing_%j.log
set -euo pipefail
umask 077
server_root=/scratch/dtm63837/Kilts_Panel
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch on the research server.' >&2
    exit 1
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch $0 EXISTING_FIT_DIRECTORY [plain|berry|other|all]" >&2
    exit 1
fi
fit_dir="$1"
focal_flavor="${2:-plain}"
ml Python/3.12.3-GCCcore-13.3.0
export MPLCONFIGDIR="$server_root/information_pricing_cache/matplotlib"
export PYTHONPYCACHEPREFIX="$server_root/information_pricing_cache/python"
echo "Binary specification: focal flavor versus not focal in both periods; two successor histories."
python3 "$project_dir/Code/final_simple/demand/information_pricing.py" \
    --fit-dir "$fit_dir" \
    --output "$server_root/results/information_pricing_${SLURM_JOB_ID}/RESULTS_ONLY" \
    --flavor "$focal_flavor" --nodes 9 --delta 1

#!/bin/bash
#SBATCH --job-name=summCols # Job name (testBowtie2)
#SBATCH --partition=hugemem_p   # Partition name (batch, highmem_p, or gpu_p)
#SBATCH --nodes=1           # Number of compute nodes for resources to be spread out over (increase only if using MPI enabled software)
#SBATCH --ntasks=1          # 1 task (process) for below commands
#SBATCH --cpus-per-task=8   # CPU core count per task, by default 1 CPU core per task
#SBATCH --mem=900G          # Memory per node (4GB); by default using M as unit
#SBATCH --time=6:00:00      # Time limit hrs:min:sec or days-hours:minutes:seconds
#SBATCH --output=%x_%j.txt  # Standard output log, e.g., testBowtie2_12345.out
#SBATCH --mail-user=dtm63837@uga.edu    # Where to send mail
#SBATCH --mail-type=END,FAIL            # Mail events (BEGIN, END, FAIL, ALL)

set -euo pipefail

ml Python/3.13.5-GCCcore-14.3.0 # Load Python

pip install -qqq -r /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/shell_files/requirements.txt


REPO_DIR=/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP
git -C "$REPO_DIR" pull --ff-only
echo "Code revision: $(git -C "$REPO_DIR" rev-parse --short HEAD)"

SCRIPT_DIR=/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Code/full_draft_1231
export POLARS_MAX_THREADS=4
RETAIL_DIR="$REPO_DIR/Code/dat_expo/retail_dat"

# Rebuild the existing scanner extract with the corrected annual join first.
# set -e stops the job if either rebuild fails, avoiding a stale-input merge.
echo "START rebuilding scanner market files with store-and-year join"
python -u "$RETAIL_DIR/retail_merge.py"
echo "DONE rebuilding scanner market files"
echo "START rebuilding full_retail.parquet"
python -u "$RETAIL_DIR/retail_concat.py"
echo "DONE rebuilding full_retail.parquet"
echo "START panel-scanner merge"

# Run diagnostics even if the merge exits at its duplicate-value audit.
merge_status=0
python -u "$SCRIPT_DIR/data_merge.py" || merge_status=$?

diagnostic_status=0
python -u "$SCRIPT_DIR/diagnose_merge.py" --year 2022 || diagnostic_status=$?

echo "Merge exit status: $merge_status; diagnostic exit status: $diagnostic_status"
# Preserve failure status so SLURM does not report an unsuccessful merge as OK.
if [ "$merge_status" -ne 0 ]; then
    exit "$merge_status"
fi
exit "$diagnostic_status"

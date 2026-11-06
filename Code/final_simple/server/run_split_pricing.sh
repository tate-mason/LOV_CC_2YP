#!/bin/bash
#SBATCH --job-name=plain_berry_pricing		# Job name (testBowtie2)
#SBATCH --partition=hugemem_p		# Partition name (batch, highmem_p, or gpu_p)
#SBATCH --nodes=1			# Number of compute nodes for resources to be spread out over (increase only if using MPI enabled software)
#SBATCH --ntasks=1			# 1 task (process) for below commands
#SBATCH --cpus-per-task=1	 	# CPU core count per task, by default 1 CPU core per task
#SBATCH --mem=300G			# Memory per node (4GB); by default using M as unit
#SBATCH --time=36:00:00              	# Time limit hrs:min:sec or days-hours:minutes:seconds
#SBATCH --output=%x_%j.txt              # Standard output log, e.g., testBowtie2_12345.out
#SBATCH --mail-user=dtm63837@uga.edu    # Where to send mail
#SBATCH --mail-type=END,FAIL          	# Mail events (BEGIN, END, FAIL, ALL)

set -euo pipefail

# === Parameters: edit these paths on the server === #
project_dir="/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP"
# est_struc.sh uses relative --output paths; these defaults match runs from shell_files.
controlled_fit="$project_dir/shell_files/res_het_hist"
uncontrolled_fit="$project_dir/shell_files/res_het_nocf"

# Weekly household inputs are temporary: remove them on success or failure.
cleanup_weekly_inputs() {
    rm -f -- "$controlled_fit/weekly_inputs.csv" "$controlled_fit"/.weekly_inputs_*.csv
}
trap cleanup_weekly_inputs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
pricing_output="/scratch/dtm63837/Kilts_Panel/plain_berry_pricing_ex"

ml Python/3.12.3-GCCcore-13.3.0
cd "$project_dir"

# Check saved fit artifacts before rebuilding household inputs on the server.
for fit_directory in "$controlled_fit" "$uncontrolled_fit"; do
    if [[ ! -f "$fit_directory/fit.json" || ! -f "$fit_directory/point_estimates.csv" ]]; then
        echo "Missing saved fit in $fit_directory. Set controlled_fit/uncontrolled_fit to the estimation output directories." >&2
        exit 1
    fi
done

# 1. Reconstruct the deleted export using saved settings; verify its likelihood.
python3 Code/final_simple/demand/regenerate_weekly_inputs.py --fit-dir "$controlled_fit"

# 2. Export both fitted population coefficient distributions.
python3 Code/final_simple/demand/history_distributions.py --fit-dir "$controlled_fit"
python3 Code/final_simple/demand/history_distributions.py --fit-dir "$uncontrolled_fit"

# 3. Compare plain and berry as focal sellers; retain all three demand categories.
python3 Code/final_simple/demand/plain_berry_pricing.py \
    --household-input "$controlled_fit/weekly_inputs.csv" \
    --fit-dir "$controlled_fit" --demand-delta 0.92 \
    --output "$pricing_output"

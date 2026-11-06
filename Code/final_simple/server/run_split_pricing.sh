#!/bin/bash
#SBATCH --job-name=split_pricing		# Job name (testBowtie2)
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
controlled_fit="/scratch/dtm63837/Kilts_Panel/res_het_hist"
uncontrolled_fit="/scratch/dtm63837/Kilts_Panel/res_het_nocf"
pricing_output="/scratch/dtm63837/Kilts_Panel/split_pricing_ex"

ml Python/3.12.3-GCCcore-13.3.0
cd "$project_dir"

# 1. Reconstruct the deleted export using saved settings; verify its likelihood.
python3 Code/final_simple/demand/regenerate_weekly_inputs.py --fit-dir "$controlled_fit"

# 2. Export both fitted population coefficient distributions.
python3 Code/final_simple/demand/history_distributions.py --fit-dir "$controlled_fit"
python3 Code/final_simple/demand/history_distributions.py --fit-dir "$uncontrolled_fit"

# 3. Price each flavor separately against the other two combined.
python3 Code/final_simple/demand/split_pricing.py \
    --household-input "$controlled_fit/weekly_inputs.csv" \
    --fit-dir "$controlled_fit" --focal-flavor all --demand-delta 0.92 \
    --output "$pricing_output"

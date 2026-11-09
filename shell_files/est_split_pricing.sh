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

# Refuse empirical execution on this computer, even with local path overrides.
if [[ "$(uname -s)" != "Linux" || ! -d /scratch/dtm63837/Kilts_Panel ]]; then
    echo "Server-only empirical workflow. Copy back results only, never research records." >&2
    exit 1
fi

ml Python/3.12.3-GCCcore-13.3.0 # Load software module and run bowtie2 below

pip install -qqq -r /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/shell_files/requirements.txt

set -e

git pull

project_dir="/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP"
# These are result directories under the repository, not under Kilts_Panel itself.
controlled_fit="$project_dir/shell_files/res_het_hist"
uncontrolled_fit="$project_dir/shell_files/res_het_nocf"

# Job-private research records live outside the code checkout and are deleted on exit.
mkdir -p /scratch/dtm63837/Kilts_Panel/Data/temporary
weekly_work=$(mktemp -d /scratch/dtm63837/Kilts_Panel/Data/temporary/pricing.XXXXXX)
weekly_input="$weekly_work/weekly_inputs.csv"

# Weekly household inputs are temporary: remove them on success or failure.
cleanup_weekly_inputs() {
    rm -f -- "$weekly_input" "$weekly_work"/.weekly_inputs_*.csv
    rmdir -- "$weekly_work"
}
trap cleanup_weekly_inputs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for fit_directory in "$controlled_fit" "$uncontrolled_fit"; do
    if [[ ! -f "$fit_directory/fit.json" || ! -f "$fit_directory/point_estimates.csv" ]]; then
        echo "Missing saved fit in $fit_directory. Set controlled_fit/uncontrolled_fit to the estimation output directories." >&2
        exit 1
    fi
done

# Rebuild the deleted household export on the server, using the saved fit settings.
# Demand parameters and standard errors are retained; likelihood must match.
python3 "$project_dir/Code/final_draft/est_code/regenerate_weekly_inputs.py" \
        --fit-dir "$controlled_fit" --weekly-output "$weekly_input"

python3 "$project_dir/Code/final_draft/est_code/history_distributions.py" \
        --fit-dir "$controlled_fit"
python3 "$project_dir/Code/final_draft/est_code/history_distributions.py" \
        --fit-dir "$uncontrolled_fit"
python3 "$project_dir/Code/final_simple/demand/plain_berry_pricing.py" \
        --household-input "$weekly_input" \
        --fit-dir "$controlled_fit" \
        --demand-delta 0.92 \
        --output "/scratch/dtm63837/Kilts_Panel/results/plain_berry_pricing_ex"

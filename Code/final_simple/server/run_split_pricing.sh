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

set -euo pipefail

# === Parameters: edit these paths on the server === #
project_dir="/scratch/dtm63837/Kilts_Panel/LOV_CC_2YP"
# est_struc.sh uses relative --output paths; these defaults match runs from shell_files.
controlled_fit="$project_dir/shell_files/res_het_hist"
uncontrolled_fit="$project_dir/shell_files/res_het_nocf"

# Set RUN_PRICING=0 for figures only. Reuse an existing server weekly file
# with WEEKLY_INPUT=/scratch/.../weekly_inputs.csv; it will never be deleted.
run_pricing="${RUN_PRICING:-1}"
run_distributions="${RUN_DISTRIBUTIONS:-1}"
weekly_input="${WEEKLY_INPUT:-}"
weekly_work=""
cleanup_weekly_inputs() {
    if [[ -n "$weekly_work" ]]; then
        rm -f -- "$weekly_work/weekly_inputs.csv" "$weekly_work"/.weekly_inputs_*.csv
        rmdir -- "$weekly_work"
    fi
}
trap cleanup_weekly_inputs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
pricing_output="/scratch/dtm63837/Kilts_Panel/plain_berry_pricing_ex"

ml Python/3.12.3-GCCcore-13.3.0
cd "$project_dir"

if [[ "$run_distributions" == "1" ]]; then
    python3 Code/final_simple/demand/history_distributions.py --fit-dir "$controlled_fit"
    python3 Code/final_simple/demand/history_distributions.py --fit-dir "$uncontrolled_fit"
fi

if [[ "$run_pricing" == "1" ]]; then
    if [[ -n "$weekly_input" ]]; then
        # Validate the server path before allowing the pricing reader to open it.
        # run_households also validates this path through server_data_policy.py.
        if [[ ! -f "$weekly_input" ]]; then
            echo "WEEKLY_INPUT does not exist; supply an existing server weekly file." >&2
            exit 1
        fi
        echo "Reusing existing server weekly input; skipping regeneration."
    else
        mkdir -p /scratch/dtm63837/Kilts_Panel/Data/temporary
        weekly_work=$(mktemp -d /scratch/dtm63837/Kilts_Panel/Data/temporary/pricing.XXXXXX)
        weekly_input="$weekly_work/weekly_inputs.csv"
        python3 Code/final_simple/demand/regenerate_weekly_inputs.py \
            --fit-dir "$controlled_fit" --weekly-output "$weekly_input"
    fi
    python3 Code/final_simple/demand/plain_berry_pricing.py \
        --household-input "$weekly_input" \
        --fit-dir "$controlled_fit" --demand-delta 0.92 \
        --output "$pricing_output"
fi

# 4. Generate seminar mechanism figures on the server from aggregate results.
python3 Code/final_draft/seminar_pres/make_policy_figures.py --output "$pricing_output"

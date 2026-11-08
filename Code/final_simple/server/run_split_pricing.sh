#!/bin/bash
#SBATCH --job-name=gamma_pricing
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/pricing_%j.log
#SBATCH --mail-user=dtm63837@uga.edu
#SBATCH --mail-type=END,FAIL

# Reuse a completed full_rebuild run; no raw-data build or demand estimation.
# sbatch Code/final_simple/server/run_split_pricing.sh \
#   /scratch/dtm63837/Kilts_Panel/rebuilds/run_JOB_ID [existing_weekly_inputs.csv]
set -euo pipefail
umask 077
server_root="/scratch/dtm63837/Kilts_Panel"
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this pricing-only job with sbatch on the research server." >&2
    exit 1
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch $0 EXISTING_REBUILD_RUN_DIR [EXISTING_WEEKLY_INPUT]" >&2
    exit 1
fi
run_dir="$1"
controlled_fit="$run_dir/fits/controlled"
uncontrolled_fit="$run_dir/fits/no_control"
summary_dir="$run_dir/data/summary_outputs"
weekly_input="${2:-$controlled_fit/weekly_inputs.csv}"
job_dir="$run_dir/pricing_only_${SLURM_JOB_ID}"
pricing_output="$job_dir/pricing/plain_berry"
export_dir="$job_dir/RESULTS_ONLY"

ml Python/3.12.3-GCCcore-13.3.0

# Check every configured research path before reading input or creating outputs.
python3 - "$project_dir" "$run_dir" "$weekly_input" "$job_dir" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1])/'Code/privacy'))
from server_data_policy import require_server, require_data_path
require_server()
for value in sys.argv[2:]:
    require_data_path(value)
run = Path(sys.argv[2])
for relative in ['fits/controlled/fit.json', 'fits/controlled/point_estimates.csv']:
    if not (run/relative).is_file():
        raise SystemExit('Missing existing fit: '+str(run/relative))
if Path(sys.argv[4]).exists():
    raise SystemExit('Refusing to overwrite a previous pricing job: '+sys.argv[4])
PY
mkdir -p "$job_dir"
export MPLCONFIGDIR="$job_dir/matplotlib_cache"
export PYTHONPYCACHEPREFIX="$job_dir/python_cache"
cd "$job_dir"

# full_rebuild removes its weekly export. If absent, reconstruct only that
# temporary export from the existing prepared sample and saved demand fit.
# Raw data construction, matching, summaries and demand estimation are skipped.
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
if [[ ! -f "$weekly_input" ]]; then
    if [[ $# -eq 2 ]]; then
        echo "Explicit weekly input does not exist: $weekly_input" >&2
        exit 1
    fi
    weekly_work=$(mktemp -d "$job_dir/weekly.XXXXXX")
    weekly_input="$weekly_work/weekly_inputs.csv"
    python3 "$project_dir/Code/final_simple/demand/regenerate_weekly_inputs.py" \
        --fit-dir "$controlled_fit" --weekly-output "$weekly_input"
fi

# Original pricing plus low/medium/high type probabilities with fixed beliefs.
python3 "$project_dir/Code/final_simple/demand/plain_berry_pricing.py" \
    --household-input "$weekly_input" --fit-dir "$controlled_fit" \
    --demand-delta 0.92 --output "$pricing_output"
python3 "$project_dir/Code/final_draft/seminar_pres/make_policy_figures.py" \
    --output "$pricing_output"

# Reuse the rebuild's estimates and summaries, staging only allowed results.
python3 "$project_dir/Code/privacy/export_rebuild_results.py" \
    --summary-dir "$summary_dir" --controlled-fit "$controlled_fit" \
    --uncontrolled-fit "$uncontrolled_fit" --pricing-dir "$pricing_output" \
    --output "$export_dir"
printf '\nPricing complete. Recover ONLY: %s\n' "$export_dir"

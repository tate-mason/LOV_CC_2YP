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
#   /scratch/dtm63837/Kilts_Panel/rebuilds/run_JOB_ID
set -euo pipefail
umask 077
server_root="/scratch/dtm63837/Kilts_Panel"
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this pricing-only job with sbatch on the research server." >&2
    exit 1
fi
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 EXISTING_REBUILD_RUN_DIR" >&2
    exit 1
fi
run_dir="$1"
uncontrolled_fit="$run_dir/fits/no_control"
job_dir="$run_dir/pricing_uncontrolled_${SLURM_JOB_ID}"
weekly_input="$job_dir/weekly_inputs.csv"
pricing_output="$job_dir/pricing/plain_berry"
export_dir="$job_dir/RESULTS_ONLY"

ml Python/3.12.3-GCCcore-13.3.0
python3 -c 'import numpy, pandas, scipy, matplotlib, polars'

# Check every configured research path before reading input or creating outputs.
python3 - "$project_dir" "$run_dir" "$weekly_input" "$job_dir" <<'PY'
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(sys.argv[1])/'Code/privacy'))
from server_data_policy import require_server, require_data_path
require_server()
for value in sys.argv[2:]:
    require_data_path(value)
run = Path(sys.argv[2])
for relative in ['fits/no_control/fit.json', 'fits/no_control/point_estimates.csv']:
    if not (run/relative).is_file():
        raise SystemExit('Missing existing fit: '+str(run/relative))
fit = json.loads((run/'fits/no_control/fit.json').read_text())
if fit['settings'].get('no_price_control') is not True or fit.get('price_control', False):
    raise SystemExit('Pricing requires the uncontrolled fit (no_price_control=true).')
if fit['settings'].get('novelty_measure', 'previous_share') != 'previous_share':
    raise SystemExit('Pricing requires the original previous-share novelty measure.')
if Path(sys.argv[4]).exists():
    raise SystemExit('Refusing to overwrite a previous pricing job: '+sys.argv[4])
PY
mkdir -p "$job_dir"
export MPLCONFIGDIR="$job_dir/matplotlib_cache"
export PYTHONPYCACHEPREFIX="$job_dir/python_cache"
cd "$job_dir"

# Reconstruct a temporary weekly export from the existing prepared sample
# and saved uncontrolled demand fit; verify its saved likelihood.
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
# Always regenerate from the selected fit: do not reuse another fit's table.
weekly_work=$(mktemp -d "$job_dir/weekly.XXXXXX")
weekly_input="$weekly_work/weekly_inputs.csv"
python3 "$project_dir/Code/final_simple/demand/regenerate_weekly_inputs.py" \
    --fit-dir "$uncontrolled_fit" --weekly-output "$weekly_input"

# Original pricing plus supply theory calibrated to actual observed prices,
# with estimated low/medium/high gamma and fixed type probabilities.
python3 "$project_dir/Code/final_simple/demand/plain_berry_pricing.py" \
    --household-input "$weekly_input" --fit-dir "$uncontrolled_fit" \
    --demand-delta 0.92 --output "$pricing_output"
python3 "$project_dir/Code/final_draft/seminar_pres/make_policy_figures.py" \
    --output "$pricing_output"

# Export only the uncontrolled fit and allowed aggregate pricing results.
python3 "$project_dir/Code/final_simple/server/export_uncontrolled_pricing.py" \
    --fit-dir "$uncontrolled_fit" --pricing-dir "$pricing_output" --output "$export_dir"
printf '\nPricing complete. Recover ONLY: %s\n' "$export_dir"

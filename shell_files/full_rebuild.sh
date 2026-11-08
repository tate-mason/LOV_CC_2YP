#!/bin/bash
#SBATCH --job-name=yogurt_full_rebuild
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=900G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/rebuild_%j.log
#SBATCH --mail-user=dtm63837@uga.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
umask 077

# === Parameters: edit these server locations and scheduler limits === #
server_root="/scratch/dtm63837/Kilts_Panel"
project_dir="$server_root/LOV_CC_2YP"
panel_raw="$server_root/nielsen_extracts/nielsen_extracts/nielsen_extracts/HMS"
retail_raw="$server_root/nielsen_extracts/nielsen_extracts/RMS"
flavor_reference="$retail_raw/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv"
first_year=2006
last_year=2015
draws=128
starts=3

# Nothing runs locally, and nothing is downloaded by this job.
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this job with sbatch on the research server. Local execution is forbidden." >&2
    exit 1
fi

ml Python/3.12.3-GCCcore-13.3.0
# Install requirements in your server Python environment before submitting.
python3 -c 'import numpy, pandas, polars, pyarrow, scipy, matplotlib, sympy, jinja2'

# Each full rebuild starts in a NEW server directory: no old samples or fits reused.
run_dir="$server_root/rebuilds/run_${SLURM_JOB_ID}"
if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse an existing rebuild directory: $run_dir" >&2
    exit 1
fi
mkdir -p "$run_dir"
data_dir="$run_dir/data"
summary_dir="$data_dir/summary_outputs"
controlled_fit="$run_dir/fits/controlled"
uncontrolled_fit="$run_dir/fits/no_control"
pricing_output="$run_dir/pricing/plain_berry"
export_dir="$run_dir/RESULTS_ONLY"
mkdir -p "$data_dir" "$controlled_fit" "$uncontrolled_fit"
export MPLCONFIGDIR="$run_dir/matplotlib_cache"
export PYTHONPYCACHEPREFIX="$run_dir/python_cache"

# All paths are checked before any research input is opened.
python3 - "$project_dir" "$panel_raw" "$retail_raw" "$flavor_reference" "$run_dir" "$first_year" "$last_year" <<'PY'
from pathlib import Path
import sys
project=Path(sys.argv[1]).resolve()
sys.path.insert(0, str(project/'Code/privacy'))
from server_data_policy import require_server, require_data_path
require_server()
for value in sys.argv[2:6]:
    require_data_path(value)
for value in sys.argv[2:5]:
    if not Path(value).exists():
        raise SystemExit('Missing configured server input: '+value)
# A fresh build must not silently omit a missing raw year.
panel, retail = map(Path, sys.argv[2:4])
required = []
for year in range(int(sys.argv[6]), int(sys.argv[7])+1):
    required.extend(panel/str(year)/'Annual_Files'/f'{table}_{year}.tsv'
                    for table in ['panelists', 'purchases', 'trips'])
    required.extend(retail/str(year)/'Annual_Files'/f'{table}_{year}.tsv'
                    for table in ['products_extra', 'rms_versions', 'stores'])
    required.extend(retail/str(year)/'Movement_Files'/f'2510_{year}'/f'{module}_{year}.tsv'
                    for module in ['3603', '3612'])
required.append(panel/'Master_Files/Latest/products.tsv')
required.append(retail/'Master_Files_2006-2020/Latest/products.tsv')
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit('Missing raw input files; full rebuild stopped before processing:\n'+'\n'.join(missing))
PY

# Delete only weekly records created by THIS run, on success/failure/INT/TERM.
cleanup_weekly_inputs() {
    rm -f -- "$controlled_fit/weekly_inputs.csv" "$controlled_fit"/.weekly_inputs_*.csv \
              "$uncontrolled_fit/weekly_inputs.csv" "$uncontrolled_fit"/.weekly_inputs_*.csv
}
trap cleanup_weekly_inputs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Work outside the repository. Source files are executed by absolute path.
cd "$run_dir"

printf '\n[1/5] Raw conversion, cleaning, annual combination, panel/retail merges, yogurt match\n'
python3 "$project_dir/Code/final_simple/data/build_data.py" \
    --data-dir "$data_dir" --panel-raw "$panel_raw" --retail-raw "$retail_raw" \
    --first-year "$first_year" --last-year "$last_year" \
    --steps convert_panel clean_panel combine_panel merge_panel panel_weeks \
            convert_retail clean_retail combine_retail merge_retail match_yogurt

printf '\n[2/5] Summary statistics, figures, flavor mapping, prepared estimation sample\n'
python3 "$project_dir/Code/final_simple/data/summaries.py" \
    --data-dir "$data_dir" --flavors "$flavor_reference" \
    --output "$summary_dir" --first-year "$first_year" --last-year "$last_year" \
    --household-batches 16

printf '\n[3/5] Split novelty/habit estimation: controlled and no-control fits\n'
python3 "$project_dir/Code/final_simple/demand/estimation.py" \
    --sample "$summary_dir/yogurt_estimation_sample.parquet" \
    --flavors "$summary_dir/estimation_flavor_reference.csv" \
    --first-year "$first_year" --last-year "$last_year" \
    --model heterogeneous_history --capacity 7 --draws "$draws" --starts "$starts" \
    --se-method hessian --output "$controlled_fit"
python3 "$project_dir/Code/final_simple/demand/estimation.py" \
    --sample "$summary_dir/yogurt_estimation_sample.parquet" \
    --flavors "$summary_dir/estimation_flavor_reference.csv" \
    --first-year "$first_year" --last-year "$last_year" \
    --model heterogeneous_history --capacity 7 --draws "$draws" --starts "$starts" \
    --se-method hessian --no-price-control --output "$uncontrolled_fit"
# The estimator exports both population history distributions automatically.

printf '\n[4/5] Plain and berry pricing, using this run\x27s controlled split fit\n'
python3 "$project_dir/Code/final_simple/demand/plain_berry_pricing.py" \
    --household-input "$controlled_fit/weekly_inputs.csv" --fit-dir "$controlled_fit" \
    --demand-delta 0.92 --output "$pricing_output"

# Generate aggregate mechanism figures before staging the results-only export.
python3 "$project_dir/Code/final_draft/seminar_pres/make_policy_figures.py" \
    --output "$pricing_output"

printf '\n[5/5] Stage an explicit allowlist of results for local recovery\n'
python3 "$project_dir/Code/privacy/export_rebuild_results.py" \
    --summary-dir "$summary_dir" --controlled-fit "$controlled_fit" \
    --uncontrolled-fit "$uncontrolled_fit" --pricing-dir "$pricing_output" \
    --output "$export_dir"

printf '\nFull rebuild complete. Recover ONLY this directory: %s\n' "$export_dir"
printf 'Raw data, panels, household records and logs remain on the server: %s\n' "$run_dir"

#!/bin/bash
#SBATCH --job-name=yogurt_two_specs
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=900G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/after_data_%j.log
#SBATCH --mail-user=dtm63837@uga.edu
#SBATCH --mail-type=END,FAIL

# sbatch Code/final_simple/server/run_after_data.sh EXISTING_REBUILD_RUN_DIR [DESCRIPTOR_REFERENCE]
set -euo pipefail
umask 077
server_root="/scratch/dtm63837/Kilts_Panel"
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit with sbatch on the research server; empirical I/O is server-only." >&2
    exit 1
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch $0 EXISTING_REBUILD_RUN_DIR [DESCRIPTOR_REFERENCE]" >&2
    exit 1
fi
reference="${2:-$server_root/nielsen_extracts/nielsen_extracts/RMS/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv}"
# Prefer the exact supplemented descriptor reference from the completed build.
if [[ $# -eq 1 && -f "$1/data/summary_outputs/recovered_flavor_reference.csv" ]]; then
    reference="$1/data/summary_outputs/recovered_flavor_reference.csv"
fi
ml Python/3.12.3-GCCcore-13.3.0
python3 -c 'import numpy, pandas, polars, pyarrow, scipy, matplotlib, jinja2'
# Put Python caches on the research server before importing any pipeline modules.
export PYTHONPYCACHEPREFIX="$server_root/python_cache_after_data"
python3 "$project_dir/Code/final_simple/server/run_after_data.py" "$1" --flavors "$reference"

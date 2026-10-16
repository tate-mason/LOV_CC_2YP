#!/usr/bin/env bash
# Submit from this folder or its parent:
#   sbatch 2026-10-16/run_slurm.sh --draws 128 --starts 3
# Add cluster-specific options (e.g. --account or --partition) before the script.
#SBATCH --job-name=flavor-comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=flavor-comparison-%j.out
#SBATCH --error=flavor-comparison-%j.err

set -euo pipefail

# SLURM runs a spool copy of this file, so locate code from the submission
# directory rather than from the script's location inside a batch job.
if [[ -n "${RUN_DIR:-}" ]]; then
  run_dir="$RUN_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  run_dir="$SLURM_SUBMIT_DIR"
  if [[ ! -f "$run_dir/estimate.py" ]]; then
    run_dir="$run_dir/2026-10-16"
  fi
else
  run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
cd -- "$run_dir"
if [[ ! -f estimate.py ]]; then
  echo 'Cannot find estimate.py; submit from 2026-10-16 or its parent, or set RUN_DIR.' >&2
  exit 1
fi

# Activate your environment before sbatch, or set PYTHON to its Python binary.
python_bin="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
# Fits are serial; keep numerical libraries within the requested CPU allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"

run_id="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)-$$}"
echo "Starting flavor comparison: job=$run_id host=$(hostname) directory=$PWD"
"$python_bin" --version
# A supplied --output overrides this default; output must stay inside this folder.
srun_command=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun_command=(srun --ntasks=1)
fi
"${srun_command[@]}" "$python_bin" -u estimate.py --output "slurm_results/$run_id" "$@"
echo "Finished flavor comparison: job=$run_id"

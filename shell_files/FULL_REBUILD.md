# Full server rebuild

Submit from the server after syncing code and installing
`Code/final_simple/requirements.txt` in the selected Python environment:

```bash
sbatch /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/shell_files/full_rebuild.sh
```

Review the raw HMS/RMS paths, flavor-reference path, year range, and SLURM
memory/time limits in the file first. Defaults follow the existing jobs;
actual raw input availability and scheduler limits have not been checked remotely.

The job runs the readable final_simple code in this order:

1. Raw conversion, cleaning, annual combination, panel/retail merging,
   Saturday weeks, yogurt matching.
2. Demographic/flavor/dynamics summary statistics and figures, plus the prepared
   estimation sample and resolved flavor mapping.
3. Heterogeneous split novelty/habit estimation, controlled and uncontrolled,
   with Hessian inference and population coefficient distributions.
4. Plain and berry focal-seller pricing using the new controlled fit.
5. Explicitly allowlisted export of estimates, aggregate tables, LaTeX stats
   tables and figures. Household CSVs and weekly tables are excluded.

A NEW run directory is created at
`/scratch/dtm63837/Kilts_Panel/rebuilds/run_JOBID/`. All generated data, household
records, staging files, fits, plots and caches stay there, outside the code
checkout. Existing converted data and saved fits are not reused. The job stops
if required raw files are missing, a step fails, or pricing rejects an
unconverged/boundary/mismatched fit. Weekly inputs from both fits are deleted on
success, failure and handled interruption. Source raw data and generated panels
remain on the server. SIGKILL or machine failure cannot run an exit handler.

Only recover `run_JOBID/RESULTS_ONLY/` to the local computer. Do not copy the
run directory, data directory, fit directories or job logs wholesale. The export
uses explicit result filenames, skips symlinks and staging folders, and excludes
all Parquet and household-level CSVs. It performs no downloads itself.

Validation here was source/argument inspection, shell syntax and fail-closed
local boundary checks. No research inputs were opened and no SLURM job was run.

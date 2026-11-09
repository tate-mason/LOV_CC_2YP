#!/bin/bash
#SBATCH --job-name=novelty_demand
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/novelty_%j.log
#SBATCH --mail-user=dtm63837@uga.edu
#SBATCH --mail-type=END,FAIL

# sbatch Code/final_simple/server/run_novelty_estimation.sh EXISTING_REBUILD_RUN_DIR
# Reuses prepared data; estimates controlled and uncontrolled demand only.
set -euo pipefail
umask 077
server_root="/scratch/dtm63837/Kilts_Panel"
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this job with sbatch on the research server." >&2
    exit 1
fi
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 EXISTING_REBUILD_RUN_DIR" >&2
    exit 1
fi
run_dir="$1"
job_dir="$run_dir/novelty_estimation_${SLURM_JOB_ID}"
ml Python/3.12.3-GCCcore-13.3.0
python3 -c 'import numpy, pandas, scipy, matplotlib, polars'

# Reuse exact sample paths and optimization settings from each previous fit.
# All empirical inputs and outputs are validated by the existing privacy policy.
python3 - "$project_dir" "$run_dir" "$job_dir" <<'PY'
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(sys.argv[1])/'Code/privacy'))
from server_data_policy import require_server, require_data_path
require_server()
project, run, job = map(Path, sys.argv[1:])
for path in [run, job]:
    require_data_path(path)
if job.exists():
    raise SystemExit('Refusing to overwrite an existing novelty job.')
for label in ['controlled', 'no_control']:
    saved = json.loads((run/'fits'/label/'fit.json').read_text())
    settings = saved['settings']
    for key in ['sample', 'flavors', 'products_extra']:
        if settings.get(key) is not None:
            require_data_path(settings[key])
    if not Path(settings['sample']).is_file():
        raise SystemExit('Missing prepared server sample.')
PY
mkdir -p "$job_dir"
export MPLCONFIGDIR="$job_dir/matplotlib_cache"
export PYTHONPYCACHEPREFIX="$job_dir/python_cache"
cd "$job_dir"
python3 - "$project_dir" "$run_dir" "$job_dir" <<'PY'
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import sys
project, run, job = map(Path, sys.argv[1:])
sys.path.insert(0, str(project/'Code/final_simple/demand'))
sys.path.insert(0, str(project/'Code/privacy'))
from estimation import run_estimation
from server_data_policy import require_server, require_data_path
from export_rebuild_results import FIT_RESULTS
require_server()
for label in ['controlled', 'no_control']:
    old = json.loads((run/'fits'/label/'fit.json').read_text())
    settings = dict(old['settings'])
    for key in ['sample', 'flavors', 'products_extra']:
        if settings.get(key) is not None:
            settings[key] = Path(settings[key])
    settings.update(model='heterogeneous_history', novelty_measure='lagged_switch_away',
                    no_price_control=(label == 'no_control'), output=job/'fits'/label)
    run_estimation(SimpleNamespace(**settings))

# Copy explicit population-level results only. No weekly inputs, logs, bootstrap
# records, saved source paths, or household-level predictions enter this export.
export = job/'RESULTS_ONLY'
require_data_path(export)
export.mkdir()
for label in ['controlled', 'no_control']:
    source = job/'fits'/label
    target = export/label
    target.mkdir()
    for name in sorted(FIT_RESULTS):
        item = source/name
        if item.is_file() and not item.is_symlink():
            shutil.copyfile(item, target/name)
    fit = json.loads((source/'fit.json').read_text())
    keep = ['model', 'novelty_measure', 'price_control', 'history_specification',
            'households', 'weeks', 'capacity', 'inference', 'converged', 'message',
            'nll', 'parameters', 'aic', 'bic']
    (target/'fit_summary.json').write_text(json.dumps({k: fit[k] for k in keep}, indent=2)+'\n')
(export/'README.txt').write_text(
    'Lagged binary switch-away novelty; cumulative own-share stock unchanged.\n'
    'Only population estimates, inference, aggregate distributions and fit summaries.\n'
    'Copy this RESULTS_ONLY directory only. Research inputs remain on server.\n')
print('Estimation complete. Recover ONLY: '+str(export), flush=True)
PY

#!/bin/bash
#SBATCH --job-name=separate_specs
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/dtm63837/Kilts_Panel/separate_specs_%j.log

# sbatch Code/final_simple/server/run_separate_experiments.sh EXISTING_FIT_DIR [FULL_PANEL_PARQUET]
set -euo pipefail
umask 077
server_root="/scratch/dtm63837/Kilts_Panel"
project_dir="$server_root/LOV_CC_2YP"
if [[ "$(uname -s)" != Linux || ! -d "$server_root" || -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch on the research server.' >&2
    exit 1
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch $0 EXISTING_FIT_DIR [FULL_PANEL_PARQUET]" >&2
    exit 1
fi
ml Python/3.12.3-GCCcore-13.3.0
python3 - "$project_dir" "$1" "${2:-}" <<'PY'
from pathlib import Path
import json
import os
import subprocess
import sys
project, fit_dir = map(Path, sys.argv[1:3])
sys.path.insert(0, str(project/'Code/privacy'))
from server_data_policy import require_server, require_data_path
require_server()
require_data_path(fit_dir)
settings = json.loads((fit_dir/'fit.json').read_text())['settings']
inputs = {k: Path(settings[k]) for k in ['sample','flavors']}
panel = sys.argv[3] or settings.get('panel')
if not panel:
    raise SystemExit('Saved fit has no full panel path. Supply FULL_PANEL_PARQUET as the second argument.')
inputs['panel'] = Path(panel)
for path in inputs.values():
    require_data_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
# Preflight every dependency before starting any potentially long estimation.
import numpy, pandas, scipy, polars, pyarrow, matplotlib, linearmodels
output = fit_dir.parent/('separate_experiments_'+os.environ['SLURM_JOB_ID'])
require_data_path(output)
if output.exists():
    raise FileExistsError('Refusing to overwrite '+str(output))
output.mkdir()
os.environ['MPLCONFIGDIR'] = str(output/'matplotlib_cache')
os.environ['PYTHONPYCACHEPREFIX'] = str(output/'python_cache')
source = project/'Code/final_simple/experiments'
common = [arg for key,path in inputs.items() for arg in ['--'+key,str(path)]]
for key in ['first_year','last_year']:
    common += ['--'+key.replace('_','-'),str(settings.get(key,2006 if key=='first_year' else 2015))]
failed = []
# Finish the quick regressions first; retain them even if a later fit fails.
jobs = [('cross_price', 'cross_price.py', [])]
for spec in ['novelty_only','random_beta_common_alpha']:
    options = ['--spec',spec]
    for key,default in [('draws',128),('starts',3),('maxiter',600),('seed',306),('capacity',7),('flavor_spec','three')]:
        options += ['--'+key.replace('_','-'),str(settings.get(key,default))]
    if settings.get('no_price_control',False):
        options += ['--no-price-control']
    jobs.append((spec,'estimate.py',options))
for label,script,options in jobs:
    print('Running '+label,flush=True)
    result = subprocess.run([sys.executable,str(source/script),*common,*options,
                             '--output',str(output/label)],cwd=output)
    if result.returncode:
        failed.append(label)
print('Outputs: '+str(output),flush=True)
if failed:
    raise SystemExit('Failed experiments: '+', '.join(failed))
PY

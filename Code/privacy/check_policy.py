"""Verify fail-closed policy without reading or writing research data."""
from pathlib import Path
import ast
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server_data_policy import require_server, server_only

calls=[]
@server_only
def forbidden():
    calls.append('executed')
with patch('server_data_policy.sys.platform', 'darwin'):
    try:
        forbidden()
    except PermissionError:
        pass
    else:
        raise AssertionError('Local execution allowed')
assert not calls
root=Path(__file__).resolve().parents[2]
for relative in ['Code/final_simple/demand/estimation.py','Code/final_simple/data/summaries.py',
                 'Code/final_simple/data/build_data.py','Code/final_simple/demand/regenerate_weekly_inputs.py',
                 'Code/final_simple/demand/split_pricing.py','Code/final_simple/demand/plain_berry_pricing.py',
                 'Code/final_simple/demand/information_pricing.py']:
    path=root/relative
    spec=importlib.util.spec_from_file_location(path.stem, path)
    sys.path.insert(0,str(path.parent))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function=next(getattr(module,n) for n in ['run_estimation','run_summaries','run_build','regenerate','run_households','run_comparison','run'] if hasattr(module,n))
    with patch('server_data_policy.sys.platform', 'darwin'):
        try:
            function(SimpleNamespace())
        except PermissionError:
            pass
        else:
            raise AssertionError(relative+' did not reject local execution')
print('All seven current empirical entry points reject local execution before touching inputs or outputs.')

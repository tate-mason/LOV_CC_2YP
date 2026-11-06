"""Fail closed before empirical I/O on a non-server machine.

No environment-variable override is provided. Results-only readers and theoretical
calculations do not require this guard. Research data remain under the known
server filesystem; a copied /scratch path on a Mac does not authorize execution.
"""
from functools import wraps
from pathlib import Path
import sys
import inspect

SERVER_ROOT = Path('/scratch/dtm63837/Kilts_Panel')


def require_server():
    if sys.platform != 'linux' or not SERVER_ROOT.is_dir():
        raise PermissionError(
            'Empirical data access is server-only. Run this workflow on the research '
            'server; copy back only estimates, predictions without household IDs, '
            'and aggregate results. Local input/output overrides are not permitted.')


def require_data_path(path):
    resolved = Path(path).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if not resolved.is_relative_to(SERVER_ROOT) or resolved.is_relative_to(checkout):
        raise PermissionError('Research inputs and outputs must stay on the server, outside LOV_CC_2YP.')


def require_legacy_server():
    require_server()
    require_data_path(Path.cwd())


def server_only(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        require_server()
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
        path_names = {'sample', 'data_dir', 'output', 'output_path', 'directory', 'work',
                      'household_input', 'weekly_output', 'input_path', 'path', 'retail_path',
                      'reference_path', 'products_path', 'flavors', 'products_extra',
                      'raw_dir', 'output_dir', 'panel_raw', 'retail_raw', 'retail_merged_input',
                      'recovery_output', 'destination'}
        for name, value in bound.arguments.items():
            if name in path_names and isinstance(value, (str, Path)):
                require_data_path(value)
            if name in {'args', 'settings'} and hasattr(value, '__dict__'):
                for key, path in vars(value).items():
                    if key in path_names and isinstance(path, (str, Path)):
                        require_data_path(path)
        return function(*args, **kwargs)
    return guarded

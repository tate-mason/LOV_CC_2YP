"""Urgent corrected three-flavor refit using existing server data, never a raw rebuild."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

project = next(p for p in Path(__file__).resolve().parents
               if (p / 'Code/privacy/server_data_policy.py').is_file())
sys.path.insert(0, str(project / 'Code/privacy'))
from server_data_policy import require_server, require_data_path
sys.path.insert(0, str(project / 'Code/final_simple'))


FIT_EXPORTS = {
    'point_estimates.csv', 'estimates.csv', 'implied_wtp.csv',
    'gamma_distribution.csv', 'gamma_distribution.pdf', 'gamma_distribution.png',
    'history_distribution_summary.csv', 'history_distribution_grid.csv',
    'history_distributions.pdf', 'history_distributions.png',
    'hessian.csv', 'hessian_diagnostics.json', 'starts_checkpoint.json',
    'iteration_checkpoint.json', 'sample_audit.json',
}


def export_stage(fit_dir, destination, status):
    """Copy named population estimates and aggregates only; never weekly inputs/logs."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(FIT_EXPORTS):
        source = fit_dir / name
        if source.is_file() and not source.is_symlink():
            shutil.copyfile(source, destination / name)
    if (fit_dir / 'fit.json').is_file():
        fit = json.loads((fit_dir / 'fit.json').read_text())
        keys = ['model', 'price_control', 'flavor_spec', 'flavor_categories',
                'novelty_measure', 'stock_measure', 'households', 'weeks', 'capacity',
                'observation_mode', 'calendar_audit', 'converged', 'message',
                'inference', 'nll', 'aic', 'bic']
        (destination / 'fit_summary.json').write_text(json.dumps(
            {k: fit.get(k) for k in keys}, indent=2) + '\n')
    (destination / 'stage_status.json').write_text(json.dumps(status, indent=2) + '\n')


def main():
    require_server()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit_dir', type=Path, help='Original server fit directory containing fit.json.')
    parser.add_argument('panel', type=Path, help='Existing full-shopping merged_panel.parquet.')
    parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    parser.add_argument('--quick-draws', type=int, default=32)
    parser.add_argument('--draws', type=int, default=128)
    parser.add_argument('--maxiter', type=int, default=300)
    parser.add_argument('--quick-only', action='store_true')
    parser.add_argument('--sample', type=Path, help='Only if the original prepared-sample path moved.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    for path in [args.fit_dir, args.panel, args.sample, args.output]:
        if path is not None:
            require_data_path(path)
    if args.workers < 1 or args.maxiter < 1 or any(n < 2 or n % 2 for n in [args.quick_draws, args.draws]):
        parser.error('Use positive workers/maxiter and even draw counts >= 2.')
    original = json.loads((args.fit_dir / 'fit.json').read_text())
    saved = dict(original['settings'])
    if original['model'] != 'heterogeneous_history' or saved.get('flavor_spec', 'three') != 'three':
        raise ValueError('Target the three-flavor heterogeneous-history presentation fit.')
    if not saved.get('no_price_control', False):
        raise ValueError('This urgent runner targets the uncontrolled presentation fit.')
    output = args.output or args.fit_dir.parent / ('observed_weeks_' + os.environ.get(
        'SLURM_JOB_ID', time.strftime('%Y%m%d_%H%M%S')))
    require_data_path(output)
    output.mkdir(parents=True, exist_ok=False)
    print('NEW RUN: ' + str(output), flush=True)
    print('Results-only export: ' + str(output / 'RESULTS_ONLY'), flush=True)
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    import pandas as pd
    from demand import estimation as est
    from demand.parallel_score import parallel_objective
    from demand.history import saved_stock_measure
    for key in ['sample', 'flavors', 'products_extra']:
        if saved.get(key):
            saved[key] = Path(saved[key])
            require_data_path(saved[key])
    saved.update(model=original['model'], panel=args.panel, observation_mode='observed_panel',
                 output=output, capacity=7, no_price_control=True, starts=1,
                 maxiter=args.maxiter, initial_estimates=args.fit_dir / 'point_estimates.csv',
                 stock_measure=saved_stock_measure(saved))
    saved.setdefault('novelty_measure', 'previous_share')
    saved.setdefault('stock_retention', .9)
    saved.setdefault('flavor_spec', 'three')
    if args.sample is not None:
        saved['sample'] = args.sample
    settings = SimpleNamespace(**saved)
    began = time.monotonic()
    frame = est.prepare_data(settings)
    audit = dict(frame.attrs['calendar_audit'])
    audit['original_purchase_weeks'] = original['weeks']
    audit['original_households'] = original['households']
    (output / 'sample_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    # Research table remains here on the server and is never part of RESULTS_ONLY.
    frame.to_csv(output / 'weekly_inputs.csv', index=False)
    data = est.make_arrays(frame, settings.capacity)
    print(f'Preparation: {time.monotonic()-began:.0f}s; {len(frame):,} weeks; '
          f'{data["n_households"]:,} households; {args.workers} CPU workers.', flush=True)
    # One compilation-free, exact objective distributed across whole households.
    with parallel_objective(est, data, args.workers):
        for stage, draws, inference in [('quick_32', args.quick_draws, 'none'),
                                        ('refined_128', args.draws, 'hessian')]:
            settings.output = output / stage
            settings.draws = draws
            settings.se_method = inference
            settings.output.mkdir()
            (settings.output / 'sample_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
            print(f'\nSTAGE {stage}: {draws} draws, one warm start, SE={inference}', flush=True)
            est.fit_model(data, settings, frame)
            fit = json.loads((settings.output / 'fit.json').read_text())
            status = dict(stage=stage, draws=draws, converged=fit['converged'],
                          preliminary=stage == 'quick_32', elapsed_seconds=time.monotonic()-began)
            export_stage(settings.output, output / 'RESULTS_ONLY' / stage, status)
            print('RECOVER ONLY: ' + str(output / 'RESULTS_ONLY' / stage), flush=True)
            print(json.dumps(status), flush=True)
            settings.initial_estimates = settings.output / 'point_estimates.csv'
            if args.quick_only:
                break
    print('DONE: ' + str(output / 'RESULTS_ONLY'), flush=True)


if __name__ == '__main__':
    main()

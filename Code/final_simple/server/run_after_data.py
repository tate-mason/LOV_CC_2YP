"""Server-only two-specification pipeline starting from a completed data build."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

project = next(p for p in Path(__file__).resolve().parents
               if (p/'Code/privacy/server_data_policy.py').is_file())
sys.path.insert(0, str(project/'Code/privacy'))
from server_data_policy import require_server, require_data_path


def run_pipeline(args):
    require_server()
    require_data_path(args.rebuild_dir)
    require_data_path(args.flavors)
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Submit with the accompanying sbatch shell.')
    data = args.rebuild_dir/'data'
    for name in ['merged_panel.parquet', 'yogurt_estimation_sample.parquet']:
        require_data_path(data/name)
        if not (data/name).is_file():
            raise FileNotFoundError('Completed data build missing '+str(data/name))
    if not args.flavors.is_file():
        raise FileNotFoundError('Missing descriptor reference: '+str(args.flavors))
    work = args.rebuild_dir/('after_data_'+os.environ['SLURM_JOB_ID'])
    require_data_path(work)
    if work.exists():
        raise FileExistsError('Refusing to overwrite an existing after-data run.')
    work.mkdir()
    os.environ['MPLCONFIGDIR'] = str(work/'matplotlib_cache')
    os.environ['PYTHONPYCACHEPREFIX'] = str(work/'python_cache')
    source = project/'Code/final_simple'
    def call(script, *arguments):
        subprocess.run([sys.executable, str(source/script), *map(str, arguments)],
                       cwd=work, check=True)
    reports = {}
    # Complete both demand specifications before running the slower pricing jobs.
    # This retains both sets of estimates even if a later pricing stage fails.
    for spec in ['three', 'five']:
        folder = work/spec
        summary = folder/'data/summary_outputs'
        fit = folder/'fits/no_control'
        print('\nSUMMARY / ESTIMATION: '+spec, flush=True)
        call('data/summaries.py', '--data-dir', data, '--flavors', args.flavors,
             '--flavor-spec', spec, '--output', summary,
             '--first-year', args.first_year, '--last-year', args.last_year,
             '--household-batches', args.household_batches)
        call('demand/estimation.py', '--sample', summary/'yogurt_estimation_sample.parquet',
             '--flavors', summary/'estimation_flavor_reference.csv',
             '--flavor-spec', spec, '--output', fit, '--no-price-control',
             '--novelty-measure', 'previous_share', '--model', 'heterogeneous_history',
             '--capacity', 7, '--first-year', args.first_year, '--last-year', args.last_year,
             '--draws', args.draws, '--starts', args.starts, '--seed', args.seed,
             '--maxiter', args.maxiter, '--se-method', 'hessian')
        reports[spec] = dict(status='estimated')
        # Stage population estimates immediately, without exporting research inputs.
        from export_uncontrolled_pricing import FIT_RESULTS
        estimate_export = work/'RESULTS_ONLY'/spec/'no_control_fit'
        estimate_export.mkdir(parents=True)
        for name in sorted(FIT_RESULTS):
            item = fit/name
            if item.is_file() and not item.is_symlink():
                shutil.copyfile(item, estimate_export/name)
        fit_info = json.loads((fit/'fit.json').read_text())
        allowed = ['model', 'price_control', 'flavor_spec', 'flavor_categories',
                   'novelty_measure', 'households', 'weeks', 'converged', 'inference',
                   'capacity', 'nll', 'aic', 'bic', 'message']
        (estimate_export/'fit_summary.json').write_text(json.dumps(
            {k: fit_info[k] for k in allowed}, indent=2)+'\n')
        target = work/'RESULTS_ONLY'/spec/'summary_statistics'
        for item in summary.rglob('*'):
            if (item.is_file() and not item.is_symlink() and
                    item.suffix.lower() in {'.tex', '.pdf', '.png'} and
                    not any(p.startswith('summary-work-') for p in item.relative_to(summary).parts)):
                destination = target/item.relative_to(summary)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, destination)
    failed = False
    for spec in ['three', 'five']:
        folder = work/spec
        fit = folder/'fits/no_control'
        pricing = folder/'pricing/plain_berry'
        print('\nPRICING: '+spec, flush=True)
        try:
            call('demand/plain_berry_pricing.py', '--household-input', fit/'weekly_inputs.csv',
                 '--fit-dir', fit, '--demand-delta', 1.0, '--output', pricing)
            subprocess.run([sys.executable, str(project/'Code/final_draft/seminar_pres/make_policy_figures.py'),
                            '--output', str(pricing)], cwd=work, check=True)
            # The exporter requires a new directory; merge only its allowlisted output.
            staged = folder/'RESULTS_ONLY'
            call('server/export_uncontrolled_pricing.py', '--fit-dir', fit,
                 '--pricing-dir', pricing, '--output', staged)
            shutil.copytree(staged, work/'RESULTS_ONLY'/spec, dirs_exist_ok=True)
            provenance = json.loads((staged/'pricing_provenance.json').read_text())
            reports[spec] = dict(status='complete', information_calibration={
                flavor: case['information_calibration_status']
                for flavor, case in provenance['cases'].items()})
        except subprocess.CalledProcessError as error:
            failed = True
            reports[spec] = dict(status='pricing_failed', returncode=error.returncode)
            print('Pricing failed for '+spec+'; continuing the other specification.', flush=True)
        finally:
            # Only the exact weekly file created by this job is removed.
            (fit/'weekly_inputs.csv').unlink(missing_ok=True)
    # Compact cross-spec comparison contains only aggregate revenue and estimates.
    import pandas as pd
    comparisons, parameters = [], []
    for spec in ['three', 'five']:
        estimate_file = work/'RESULTS_ONLY'/spec/'no_control_fit/estimates.csv'
        if estimate_file.is_file():
            parameters.append(pd.read_csv(estimate_file).assign(flavor_spec=spec))
        for flavor in ['plain', 'berry']:
            file = work/'RESULTS_ONLY'/spec/'pricing'/flavor/'pricing_policy_benchmarks.csv'
            if file.is_file():
                comparisons.append(pd.read_csv(file).assign(flavor_spec=spec, flavor=flavor))
    if comparisons:
        pd.concat(comparisons, ignore_index=True).to_csv(
            work/'RESULTS_ONLY/pricing_specification_comparison.csv', index=False)
    if parameters:
        pd.concat(parameters, ignore_index=True).to_csv(
            work/'RESULTS_ONLY/estimate_specification_comparison.csv', index=False)
    (work/'RESULTS_ONLY/run_status.json').write_text(json.dumps(reports, indent=2)+'\n')
    print('\nRecover ONLY: '+str(work/'RESULTS_ONLY'), flush=True)
    print(json.dumps(reports, indent=2), flush=True)
    if failed:
        raise SystemExit('One or more pricing stages failed; exported estimates and summaries are retained.')


if __name__ == '__main__':
    require_server()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rebuild_dir', type=Path)
    parser.add_argument('--flavors', type=Path, required=True)
    parser.add_argument('--first-year', type=int, default=2006)
    parser.add_argument('--last-year', type=int, default=2015)
    parser.add_argument('--household-batches', type=int, default=16)
    parser.add_argument('--draws', type=int, default=128)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=600)
    parser.add_argument('--seed', type=int, default=306)
    run_pipeline(parser.parse_args())

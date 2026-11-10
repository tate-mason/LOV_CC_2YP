"""Server-only allowlisted export with uncontrolled-fit provenance."""
from pathlib import Path
import argparse
import json
import shutil
import sys

root = next(p for p in Path(__file__).resolve().parents
            if (p/'Code/privacy/server_data_policy.py').is_file())
sys.path.insert(0, str(root/'Code/privacy'))
from server_data_policy import require_server, require_data_path
from export_rebuild_results import FIT_RESULTS, PRICING_RESULTS


def export_results(fit_dir, pricing_dir, output):
    require_server()
    for path in [fit_dir, pricing_dir, output]:
        require_data_path(path)
    fit = json.loads((fit_dir/'fit.json').read_text())
    manifest = json.loads((pricing_dir/'run_manifest.json').read_text())
    if fit['settings'].get('no_price_control') is not True or fit.get('price_control', False):
        raise ValueError('Results export requires an uncontrolled fit.')
    if fit['settings'].get('novelty_measure', 'previous_share') != 'previous_share':
        raise ValueError('Results export requires original previous-share novelty.')
    # Verify the run actually used the requested parameter files, before export.
    import hashlib
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    expected_fit = digest(fit_dir/'fit.json')
    expected_parameters = digest(fit_dir/'point_estimates.csv')
    cases = {}
    for flavor in ['plain', 'berry']:
        report = manifest['results'][flavor]
        if report['price_control'] is not False:
            raise ValueError(flavor+': pricing used controls; refusing export.')
        if (report['fit_sha256'] != expected_fit or
                report['parameter_sha256'] != expected_parameters):
            raise ValueError(flavor+': pricing provenance does not match requested fit.')
        keys = ['price_control', 'fit_sha256', 'parameter_sha256', 'reconstructed_nll',
                'households', 'capacity', 'history_model', 'demand_delta', 'type_treatment',
                'flavor_spec', 'flavor_categories']
        cases[flavor] = {key: report[key] for key in keys}
        cases[flavor]['information_calibration_status'] = report['gamma_information'].get('status', 'complete')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing results export.')
    output.mkdir(parents=True)
    # Exact names and locations only; never copy an entire research directory.
    for source, destination, names in [
            (fit_dir, output/'no_control_fit', FIT_RESULTS),
            (pricing_dir, output/'pricing', {'flavor_comparison.csv'})]:
        destination.mkdir()
        for name in sorted(names):
            item = source/name
            if item.is_file() and not item.is_symlink():
                shutil.copyfile(item, destination/name)
    for flavor in ['plain', 'berry']:
        destination = output/'pricing'/flavor
        destination.mkdir()
        for name in sorted(PRICING_RESULTS | {'information_calibration_status.json',
                                            'information_calibration_moments.csv'}):
            item = pricing_dir/flavor/name
            if item.is_file() and not item.is_symlink():
                shutil.copyfile(item, destination/name)
    # Presentation figure names, using the common persistent-type evaluation.
    for flavor in ['plain', 'berry']:
        if cases[flavor]['information_calibration_status'] != 'complete':
            continue
        target = output/'empirical_information'/flavor
        target.mkdir(parents=True)
        for source_name, target_name in [('profits', 'revenues'),
                                         ('information_value', 'information_value'),
                                         ('first_prices', 'first_prices'),
                                         ('terminal_prices', 'terminal_prices')]:
            source = pricing_dir/flavor/('information_friction_'+source_name+'.pdf')
            if not source.is_file() or source.is_symlink():
                raise ValueError('Missing presentation figure: '+str(source))
            shutil.copyfile(source, target/(target_name+'.pdf'))
    provenance = dict(price_control=False, novelty_measure='previous_share',
                      flavor_spec=fit['settings'].get('flavor_spec', 'three'),
                      cases=cases, script_sha256=manifest['script_sha256'],
                      solver_sha256=manifest['solver_sha256'])
    (output/'pricing_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (output/'README.txt').write_text(
        'Uncontrolled demand fit; original previous-share novelty.\n'
        'Plain/berry coupons and policy benchmarks use the same uncontrolled fit.\n'
        'Check pricing_provenance.json for auxiliary calibration status.\n'
        'Rejected calibrations have diagnostics, not information-value figures.\n'
        'Copy only this RESULTS_ONLY directory. No household records, weekly\n'
        'inputs, full internal manifests, or research logs are included.\n'
        'Slide sources and existing local assets have not been overwritten.\n')


if __name__ == '__main__':
    require_server()
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['fit-dir', 'pricing-dir', 'output']:
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    export_results(args.fit_dir, args.pricing_dir, args.output)

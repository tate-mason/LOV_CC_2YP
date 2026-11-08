"""Server-only results staging. Uses explicit names; never copies whole data folders."""
from pathlib import Path
import argparse
import shutil
from server_data_policy import require_server, require_data_path

FIT_RESULTS = {
    'point_estimates.csv', 'estimates.csv', 'covariance.csv', 'first_stage.csv',
    'implied_wtp.csv', 'history_distribution_summary.csv', 'history_distribution_grid.csv',
    'history_distributions.pdf', 'history_distributions.png',
}
PRICING_RESULTS = {
    'information_friction_aggregate.csv', 'information_friction_by_group.csv',
    'information_friction_states.csv',
    'information_friction_settings.json', 'information_friction_results.tex',
    'information_friction_profits.pdf', 'information_friction_profits.png',
    'information_friction_first_prices.pdf', 'information_friction_first_prices.png',
    'information_friction_terminal_prices.pdf', 'information_friction_terminal_prices.png',
    'information_friction_information_value.pdf', 'information_friction_information_value.png',
    'focal_flavor_comparison.csv', 'targeted_revenue_aggregate.csv',
    'pricing_policy_benchmarks.csv', 'pricing_policy_benchmarks.pdf', 'pricing_policy_benchmarks.png',
    'terminal_offers_by_purchase.csv', 'terminal_offers_by_purchase.pdf', 'terminal_offers_by_purchase.png',
    'terminal_offer_within_household_gap.csv',
    'pricing_mechanism.pdf', 'pricing_mechanism.png',
    'targeted_revenue_by_switching_group.csv', 'household_group_summary.csv',
    'household_coupon_results.pdf', 'household_coupon_results.png',
    'discounts_by_consumer_type.pdf', 'discounts_by_consumer_type.png',
}


def export_results(summary, controlled, uncontrolled, pricing, output):
    require_server()
    for path in [summary, controlled, uncontrolled, pricing, output]:
        require_data_path(path)
    if output.exists():
        raise FileExistsError('Results export must use a new directory.')
    output.mkdir(parents=True)
    for source, label in [(controlled, 'controlled_fit'), (uncontrolled, 'no_control_fit'),
                          (pricing, 'pricing'), (summary, 'summary_statistics')]:
        for item in source.rglob('*'):
            if not item.is_file() or item.is_symlink():
                continue
            if source == summary:
                # Summary LaTeX tables and figures are aggregate results, not CSV records.
                keep = item.suffix.lower() in {'.tex', '.pdf', '.png'} and not any(
                    part.startswith('summary-work-') for part in item.relative_to(source).parts)
            else:
                keep = item.name in (PRICING_RESULTS if source == pricing else FIT_RESULTS)
            if keep:
                target = output / label / item.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, target)
    (output/'README.txt').write_text(
        'Results-only export: estimates, aggregate tables, and figures.\n'
        'No raw panels, household CSVs, weekly inputs, Parquet, or logs are included.\n')


if __name__ == '__main__':
    require_server()
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['summary-dir', 'controlled-fit', 'uncontrolled-fit', 'pricing-dir', 'output']:
        parser.add_argument('--'+flag, required=True, type=Path)
    args = parser.parse_args()
    export_results(args.summary_dir, args.controlled_fit, args.uncontrolled_fit, args.pricing_dir, args.output)

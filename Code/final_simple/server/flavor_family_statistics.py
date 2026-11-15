"""Server-only flavor-family statistics from an existing detailed weekly export.

Empirical input stays on the server. Only pooled statistics and descriptor-level
aggregates are written. No demand estimation or matching is rerun.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'privacy'))
from server_data_policy import require_server, require_data_path
from bundle_statistics import aggregate_plans, write_table
from flavor_families import classify, ALIASES, MODIFIERS, COMPONENTS


def recode_weekly(weekly, mapping):
    """Pure transformation; combine aliases before building baskets or shares."""
    return (weekly.with_columns(pl.col('flavor').replace_strict(mapping).alias('flavor'))
            .group_by(['household_code', 'week', 'flavor'])
            .agg(pl.col('quantity').sum()))


def summary_plans(weekly):
    plans = aggregate_plans(weekly)
    household_flavors = weekly.group_by(['household_code', 'flavor']).agg(pl.col('quantity').sum())
    plans['concentration'] = (household_flavors
        .with_columns((pl.col('quantity') / pl.col('quantity').sum().over('household_code')).alias('share'))
        .group_by('household_code').agg(pl.len().alias('distinct_flavors'),
                                       (pl.col('share') ** 2).sum().alias('hhi'))
        .select(pl.col('distinct_flavors').mean().alias('mean_distinct_flavors'),
                pl.col('hhi').mean().alias('mean_household_hhi')))
    return plans


def headline(results):
    overall = results['overall'].row(0, named=True)
    household = results['households'].row(0, named=True)
    concentration = results['concentration'].row(0, named=True)
    rows = dict(households=overall['households'], purchase_weeks=overall['purchase_weeks'],
                mean_distinct_flavors=concentration['mean_distinct_flavors'],
                mean_household_hhi=concentration['mean_household_hhi'],
                mixed_purchase_weeks_pct=overall['mixed_pct'])
    for clock in ['purchase', 'calendar']:
        spell = results[clock + '_spells'].row(0, named=True)
        rows.update({clock + '_eligible_pairs': overall[clock + '_pairs'],
                     clock + '_change_pct': overall[clock + '_change_pct'],
                     clock + '_mean_changes_per_household': household['mean_' + clock + '_changes'],
                     clock + '_turnover_pct': overall[clock + '_turnover_pct'],
                     clock + '_spell_count': spell['spells'],
                     clock + '_mean_spell_length': spell['mean_length'],
                     clock + '_median_spell_length': spell['median_length']})
    return rows


def run(summary_dir, output):
    require_server()
    require_data_path(summary_dir)
    require_data_path(output)
    source = summary_dir / 'yogurt_weekly_flavor_quantities.csv'
    require_data_path(source)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing output directory')
    weekly = pl.scan_csv(source, schema_overrides={
        'household_code': pl.String, 'week': pl.Date, 'flavor': pl.String, 'quantity': pl.Float64})
    weekly = weekly.select('household_code', 'week', 'flavor', 'quantity')
    invalid = weekly.select((pl.col('household_code').is_null() | pl.col('week').is_null()
        | pl.col('flavor').is_null() | pl.col('quantity').is_null()
        | ~pl.col('quantity').is_finite() | (pl.col('quantity') <= 0)).sum()).collect().item()
    if invalid:
        raise ValueError('Missing keys or invalid quantities in the weekly export')
    weekly = weekly.group_by(['household_code', 'week', 'flavor']).agg(pl.col('quantity').sum())
    # Descriptor-level totals are aggregates, never household or weekly records.
    labels = weekly.group_by('flavor').agg(pl.col('quantity').sum().alias('packages'),
                                          pl.len().alias('household_flavor_weeks')).collect()
    if not labels.height:
        raise ValueError('No purchases in the detailed weekly export')
    mapping, audit_rows = {}, []
    total_packages = labels['packages'].sum()
    for row in labels.iter_rows(named=True):
        family, rule = classify(row['flavor'])
        mapping[row['flavor']] = family
        audit_rows.append(dict(original_flavor=row['flavor'], family=family, rule=rule,
                               packages=row['packages'],
                               percent_packages=100 * row['packages'] / total_packages,
                               household_flavor_weeks=row['household_flavor_weeks']))
    family_weekly = recode_weekly(weekly, mapping)
    summaries = {}
    by_flavor = {}
    for label, frame in [('detailed', weekly), ('families', family_weekly)]:
        results = {name: plan.collect(engine='streaming') for name, plan in summary_plans(frame).items()}
        summaries[label] = headline(results)
        by_flavor[label] = {clock: results[clock + '_by_flavor'] for clock in ['purchase', 'calendar']}
    for key in ['households', 'purchase_weeks', 'purchase_eligible_pairs', 'calendar_eligible_pairs']:
        if summaries['detailed'][key] != summaries['families'][key]:
            raise AssertionError('Flavor normalization changed the sample or purchase clock')
    # A coarsening cannot introduce basket changes or lower household HHI.
    for clock in ['purchase', 'calendar']:
        before, after = [summaries[k][clock + '_change_pct'] for k in ['detailed', 'families']]
        if before is not None and after > before + 1e-9:
            raise AssertionError('Normalization introduced basket changes')
    if summaries['families']['mean_household_hhi'] + 1e-9 < summaries['detailed']['mean_household_hhi']:
        raise AssertionError('Coarsening unexpectedly reduced concentration')
    output.mkdir(parents=True, exist_ok=False)
    # Recover means, medians and rates only; keep sample and spell counts internal.
    export_metrics = [key for key in summaries['detailed']
                      if key not in {'households', 'purchase_weeks'}
                      and not key.endswith(('_eligible_pairs', '_spell_count'))]
    comparison = pl.DataFrame({
        'metric': export_metrics,
        **{label: [float(summaries[label][key]) if summaries[label][key] is not None else None
                   for key in export_metrics] for label in ['detailed', 'families']}})
    write_table(comparison, output, 'summary_comparison', 'Detailed labels versus normalized flavor families')
    audit = pl.DataFrame(audit_rows).sort('packages', descending=True)
    audit.select('original_flavor', 'family', 'rule').sort('original_flavor').write_csv(
        output / 'flavor_family_mapping.csv')
    coverage = audit.group_by('rule').agg(pl.col('percent_packages').sum()).sort('rule')
    write_table(coverage, output, 'mapping_coverage', 'Flavor-family normalization coverage')
    for label, clocks in by_flavor.items():
        for clock, frame in clocks.items():
            write_table(frame.drop('spells'), output, label + '_presence_' + clock,
                        label.title() + ': presence spells in ' + clock + ' weeks')
    (output / 'definitions.json').write_text(json.dumps(dict(
        method='Conservative ingredient-component families; unrecognized descriptors retained separately',
        aliases=ALIASES, removed_modifiers=sorted(MODIFIERS), components=COMPONENTS,
        blends='Retain all recognized ingredients as a sorted composite; strawberry-banana differs from strawberry',
        desserts='Retain recognized dessert styles and toppings; strawberry cheesecake differs from strawberry',
        unknown='Any unexplained descriptor text retains the original label; no records dropped or pooled into other',
        quantities='Combine aliases within household/week before all statistics; package shares define household HHI',
        clocks='Purchase clock skips calendar gaps; calendar clock requires adjacent weeks',
        spells='Flavor presence allows mixed baskets; pool all observed spells without censoring correction',
        source=str(source), mapping_sha256=hashlib.sha256(Path(__file__).with_name('flavor_families.py').read_bytes()).hexdigest(),
        recovery='Means, medians, percentages and label-only mapping; no sample counts, flavor counts, spell counts, household IDs or weekly records; never recover the source weekly export'), indent=2) + '\n')
    print('Complete. Recover only aggregate output: ' + str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.summary_dir, args.output)

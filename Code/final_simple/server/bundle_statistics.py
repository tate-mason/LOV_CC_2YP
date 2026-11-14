"""Server-only, aggregate-only summaries of mixed-flavor weekly yogurt baskets.

Reuse the detailed and aggregated_flavors/yogurt_weekly_flavor_quantities.csv
exports from data/summaries.py. No household output is written by this script.
"""
from pathlib import Path
import argparse
import json
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'privacy'))
from server_data_policy import require_server, require_data_path


def aggregate_plans(weekly):
    """Pure lazy transformations; input is one positive quantity per HH/week/flavor.

    Pair rates weight eligible pairs equally. Counts per household include zeros.
    Presence spells are pooled across household/flavor spells, not HH averages.
    """
    baskets = (weekly.group_by(['household_code', 'week'])
        .agg(pl.col('flavor').sort().alias('flavors'))
        .sort(['household_code', 'week'])
        .with_columns(
            pl.col('flavors').shift().over('household_code').alias('previous'),
            pl.col('week').diff().over('household_code').dt.total_days().alias('gap_days'),
            pl.col('week').cum_count().over('household_code').alias('purchase_index'))
        .with_columns(
            (pl.col('gap_days') == 7).fill_null(False).alias('adjacent'),
            pl.col('previous').is_not_null().alias('pair'))
        .with_columns(
            (pl.col('flavors') != pl.col('previous')).fill_null(False).alias('changed'),
            pl.col('flavors').list.set_difference('previous').list.len().alias('added'),
            pl.col('previous').list.set_difference('flavors').list.len().alias('dropped'),
            (1 - pl.col('flavors').list.set_intersection('previous').list.len()
             / pl.col('flavors').list.set_union('previous').list.len()).alias('turnover')))
    pair_expressions = []
    for key, mask in [('purchase', pl.col('pair')), ('calendar', pl.col('adjacent'))]:
        pair_expressions.extend([
            mask.sum().alias(key+'_pairs'),
            (100 * pl.col('changed').filter(mask).mean()).alias(key+'_change_pct'),
            pl.col('added').filter(mask).mean().alias(key+'_added'),
            pl.col('dropped').filter(mask).mean().alias(key+'_dropped'),
            (100 * pl.col('turnover').filter(mask).mean()).alias(key+'_turnover_pct')])
    overall = baskets.select(
        pl.len().alias('purchase_weeks'),
        pl.col('household_code').n_unique().alias('households'),
        (100 * (pl.col('flavors').list.len() > 1).mean()).alias('mixed_pct'),
        *pair_expressions)
    household_means = (baskets.group_by('household_code').agg(
        (pl.col('changed') & pl.col('pair')).sum().alias('purchase_changes'),
        (pl.col('changed') & pl.col('adjacent')).sum().alias('calendar_changes'))
        .select(pl.col('purchase_changes').mean().alias('mean_purchase_changes'),
                pl.col('calendar_changes').mean().alias('mean_calendar_changes')))
    # Every flavor's presence is tracked, including when other flavors coexist.
    presence = (weekly.select('household_code', 'week', 'flavor')
        .join(baskets.select('household_code', 'week', 'purchase_index'),
              on=['household_code', 'week'], how='left')
        .sort(['household_code', 'flavor', 'week'])
        .with_columns(
            (pl.col('purchase_index').diff().over(['household_code', 'flavor']) != 1)
                .fill_null(True).alias('new_purchase_spell'),
            (pl.col('week').diff().over(['household_code', 'flavor']).dt.total_days() != 7)
                .fill_null(True).alias('new_calendar_spell'))
        .with_columns(
            pl.col('new_purchase_spell').cast(pl.UInt64).cum_sum()
                .over(['household_code', 'flavor']).alias('purchase_spell'),
            pl.col('new_calendar_spell').cast(pl.UInt64).cum_sum()
                .over(['household_code', 'flavor']).alias('calendar_spell')))
    plans = {'overall': overall, 'households': household_means}
    for clock in ['purchase', 'calendar']:
        spells = presence.group_by(['household_code', 'flavor', clock+'_spell']).len()
        plans[clock+'_spells'] = spells.select(
            pl.len().alias('spells'), pl.col('len').mean().alias('mean_length'),
            pl.col('len').median().alias('median_length'))
        plans[clock+'_by_flavor'] = spells.group_by('flavor').agg(
            pl.len().alias('spells'), pl.col('len').mean().alias('mean_length'),
            pl.col('len').median().alias('median_length')).sort('flavor')
    return plans


def write_table(frame, output, stem, caption):
    """Only explicitly aggregated frames reach this writer."""
    frame.write_csv(output / (stem+'.csv'))
    (output / (stem+'.tex')).write_text(frame.to_pandas().to_latex(
        index=False, escape=True, float_format='%.2f', caption=caption))


def summarize(source, output):
    # Check both paths before empirical I/O, including schema inspection.
    require_server()
    require_data_path(source)
    require_data_path(output)
    weekly = (pl.scan_csv(source, schema_overrides={
        'household_code': pl.String, 'week': pl.Date, 'flavor': pl.String,
        'quantity': pl.Float64})
        .select('household_code', 'week', 'flavor', 'quantity'))
    invalid = weekly.select((
        pl.col('household_code').is_null() | pl.col('week').is_null()
        | pl.col('flavor').is_null() | ~pl.col('quantity').is_finite()
        | (pl.col('quantity') <= 0) | pl.col('quantity').is_null()).sum()).collect().item()
    if invalid:
        raise ValueError('Weekly quantities contain missing keys or nonpositive/nonfinite quantities')
    # Consolidate duplicate flavor entries; sets are insensitive to package amounts.
    weekly = weekly.group_by(['household_code', 'week', 'flavor']).agg(pl.col('quantity').sum())
    plans = aggregate_plans(weekly)
    # Persist only aggregates; intermediate lazy frames stay on the server.
    results = {name: plan.collect(engine='streaming') for name, plan in plans.items()}
    o = results['overall'].row(0, named=True)
    if not o['purchase_weeks']:
        raise ValueError('No weekly purchases in selected summary export')
    h = results['households'].row(0, named=True)
    rows = [
        ('Households', o['households']), ('Yogurt purchase weeks', o['purchase_weeks']),
        ('Mixed-flavor purchase weeks (%)', o['mixed_pct']),
        ('Mean basket-set changes per household (successive purchase weeks)', h['mean_purchase_changes']),
        ('Mean basket-set changes per household (adjacent calendar weeks)', h['mean_calendar_changes'])]
    for clock, label in [('purchase', 'Successive purchase weeks'), ('calendar', 'Adjacent calendar weeks')]:
        rows.extend([
            (label+': eligible pairs', o[clock+'_pairs']),
            (label+': basket flavor-set changes (%)', o[clock+'_change_pct']),
            (label+': mean flavors added', o[clock+'_added']),
            (label+': mean flavors dropped', o[clock+'_dropped']),
            (label+': mean flavor-set turnover (%)', o[clock+'_turnover_pct'])])
        r = results[clock+'_spells'].row(0, named=True)
        rows.extend([
            (label+': flavor-presence spells', r['spells']),
            (label+': mean flavor-presence spell length', r['mean_length']),
            (label+': median flavor-presence spell length', r['median_length'])])
    output.mkdir(parents=True, exist_ok=True)
    write_table(pl.DataFrame({'metric': [r[0] for r in rows], 'value': [float(r[1]) if r[1] is not None else None for r in rows]}),
                output, 'bundle_dynamics', 'Yogurt basket changes and flavor-presence spells')
    for clock in ['purchase', 'calendar']:
        write_table(results[clock+'_by_flavor'], output, 'flavor_presence_'+clock,
                    'Flavor-presence spells ('+clock+' weeks)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_server()
    require_data_path(args.summary_dir)
    require_data_path(args.output)
    # Fresh job directory only: preserve previous results and user files.
    if args.output.exists():
        raise ValueError('Refusing to overwrite existing output directory')
    sources = {label: args.summary_dir / relative / 'yogurt_weekly_flavor_quantities.csv'
               for label, relative in [('detailed', ''), ('aggregated_flavors', 'aggregated_flavors')]}
    for source in sources.values():
        require_data_path(source)
        if not source.is_file():
            raise FileNotFoundError('Missing saved weekly summary export: '+str(source))
    for label, source in sources.items():
        summarize(source, args.output / label)
    (args.output / 'definitions.json').write_text(json.dumps(dict(
        basket='set of flavors with positive quantity in a yogurt purchase week; mixed baskets included',
        change='any addition or removal of a flavor; quantities alone do not change the set',
        turnover='1 - intersection size / union size (Jaccard distance)',
        purchase_clock='successive observed yogurt purchase weeks, allowing calendar gaps',
        calendar_clock='adjacent calendar weeks only; gaps break spells and are excluded from pair rates',
        presence_spell='a flavor appears in successive eligible baskets; other flavors may coexist',
        weighting='pair rates pool pairs; spell lengths pool spells; household counts include zero changes',
        censoring='first/last observed spells included; descriptive observed lengths, no censoring adjustment',
        sample='inherits the saved summary sample and flavor grouping; no raw-data rebuild or demand refit',
        recovery='all files in this output directory are aggregates; no household or weekly records'), indent=2)+'\n')
    print('Complete. Recover only aggregate output: '+str(args.output))


if __name__ == '__main__':
    main()

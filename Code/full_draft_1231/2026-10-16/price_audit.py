"""Audit raw price failures before imputation; optionally verify RMS join keys."""
import argparse
import json

import numpy as np
import pandas as pd

import demand_core as core


def status(values):
    numeric = pd.to_numeric(values, errors='coerce')
    return pd.Series(np.select(
        [values.isna(), numeric.isna(), np.isinf(numeric), numeric.eq(0), numeric.lt(0)],
        ['missing', 'malformed', 'infinite', 'zero', 'negative'], default='valid'), index=values.index)


def key_id(values):
    return values.astype('string').str.strip().str.replace(r'\.0$', '', regex=True).str.lstrip('0').replace('', '0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=core.DEFAULT_DATA_PATH)
    parser.add_argument('--output', default='price_audit')
    parser.add_argument('--price-column', default='price')
    parser.add_argument('--flavor-column', default='flavor')
    parser.add_argument('--prefiltered', action='store_true')
    parser.add_argument('--rms', help='Optional original RMS Parquet: scan only join keys using Polars')
    parser.add_argument('--rms-price-column', default='price', help='Original RMS price field, if present')
    parser.add_argument('--spend-column', help='Documented HMS total expenditure for this purchase line')
    parser.add_argument('--coupon-column', help='Documented total coupon value for this line; subtract only if expenditure is pre-coupon')
    args = parser.parse_args()
    output = core.local_path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a fresh output directory.')
    frame = core.read_table(core.input_path(args.input))
    stages = [{'stage': 'input', 'rows': len(frame)}]
    for column, values in {'household_size': [1], 'serving_per_container_cd': [67181961, 65622705],
                           'product_module_code_hms': [3612, 3603]}.items():
        if column in frame:
            frame = frame[pd.to_numeric(frame[column], errors='coerce').isin(values)].copy()
        elif not args.prefiltered:
            parser.error(f'Missing {column}; use --prefiltered for an already restricted sample.')
        stages.append({'stage': column, 'rows': len(frame)})
    if frame.empty:
        parser.error('No rows after sample restrictions.')
    required = [args.price_column, 'quantity'] + [c for c in [args.spend_column, args.coupon_column] if c]
    if set(required) - set(frame):
        parser.error(f'Missing columns: {sorted(set(required)-set(frame))}')
    if args.coupon_column and not args.spend_column:
        parser.error('--coupon-column requires --spend-column.')
    output.mkdir(parents=True, exist_ok=True)
    frame['_price_status'] = status(frame[args.price_column])
    frame['_invalid_price'] = frame._price_status.ne('valid')
    numeric = pd.to_numeric(frame[args.price_column], errors='coerce')
    frame['_valid_price'] = numeric.where(~frame._invalid_price)
    quantity = pd.to_numeric(frame.quantity, errors='coerce')
    quantities = quantity.where(np.isfinite(quantity) & quantity.gt(0))
    stages.append({'stage': 'audit_sample', 'rows': len(frame)})
    pd.DataFrame(stages).to_csv(output / 'sample_stages.csv', index=False)
    frame.groupby('_price_status', dropna=False).agg(rows=('_invalid_price', 'size'),
        recorded_quantity=('quantity', lambda q: pd.to_numeric(q, errors='coerce').sum())).to_csv(output / 'price_status.csv')
    # Descriptive grouping: patterns suggest where to investigate, not causes.
    if 'week_end' in frame:
        week = core.dates(frame.week_end)
        frame['_year'] = week.dt.year
        frame['_month'] = week.dt.to_period('M').astype(str)
    groups = [args.flavor_column, 'store_code_uc', 'upc', '_year', '_month',
              'market_code', 'dma_code', 'retailer_code', 'retailer_code_hms', 'deal_flag_uc']
    for column in dict.fromkeys(groups):
        if column in frame:
            table = frame.groupby(column, dropna=False).agg(rows=('_invalid_price', 'size'),
                invalid_rows=('_invalid_price', 'sum'), invalid_fraction=('_invalid_price', 'mean'),
                valid_price_median=('_valid_price', 'median'))
            table.sort_values('invalid_rows', ascending=False).to_csv(output / f'by_{column}.csv')
    candidates = [c for c in frame if any(term in c.lower() for term in
                  ['price', 'paid', 'spend', 'cost', 'coupon', 'dollars', 'units']) and not c.startswith('_')]
    summaries = []
    for column in candidates:
        value = pd.to_numeric(frame[column], errors='coerce')
        value = value.where(np.isfinite(value))
        summaries.append({'column': column, 'numeric_nonmissing': int(value.notna().sum()),
                          'available_when_price_invalid': int((value.notna() & frame._invalid_price).sum()),
                          'zero_rows': int(value.eq(0).sum()), 'negative_rows': int(value.lt(0).sum()),
                          'p01': value.quantile(.01), 'median': value.median(), 'p99': value.quantile(.99)})
    pd.DataFrame(summaries).to_csv(output / 'candidate_fields.csv', index=False)
    report = {'rows': len(frame), 'price_column': args.price_column,
              'price_status_counts': frame._price_status.value_counts().to_dict(),
              'invalid_quantity_rows': int(quantities.isna().sum()),
              'columns': list(frame.columns),
              'interpretation': 'Missing price is not proof of a failed RMS match. Key verification requires original RMS. No prices are replaced by this audit.'}
    if args.spend_column:
        spend = pd.to_numeric(frame[args.spend_column], errors='coerce')
        coupon = pd.to_numeric(frame[args.coupon_column], errors='coerce') if args.coupon_column else 0
        per_cup = ((spend - coupon) / quantities).replace([np.inf, -np.inf], np.nan)
        comparison = pd.DataFrame({'recorded_price': numeric, 'quantity': quantity,
                                   'line_spend': spend, 'candidate_net_paid_per_cup': per_cup,
                                   'price_status': frame._price_status})
        comparison['price_to_candidate_ratio'] = numeric / per_cup.where(per_cup > 0)
        comparison.to_csv(output / 'payment_comparison.csv', index=False)
        report['candidate_paid_price'] = {'positive_finite_rows': int((per_cup.gt(0) & np.isfinite(per_cup)).sum()),
            'median': float(per_cup.median()), 'formula': '(line expenditure - specified line coupon)/quantity',
            'interpretation': 'Requires documented units, pack conversion, and coupon accounting. Transaction prices can reflect endogenous discounts; not automatically shelf prices for unchosen alternatives.'}
    if args.rms:
        import polars as pl
        keys = ['store_code_uc', 'upc', 'week_end']
        if set(keys) - set(frame):
            parser.error('RMS verification requires store_code_uc, upc, week_end in purchases.')
        # Restrict the lazy scan to identifiers; never collect wide retail rows.
        scan = pl.scan_parquet(str(core.input_path(args.rms)))
        source_price = args.rms_price_column
        columns = keys + ([source_price] if source_price in scan.collect_schema().names() and source_price not in keys else [])
        rms = scan.select(columns).unique().collect(engine='streaming').to_pandas()
        left = frame[keys].copy()
        left['week_end'] = core.dates(left.week_end)
        rms['week_end'] = core.dates(rms.week_end)
        for key in keys[:2]:
            left[key], rms[key] = key_id(left[key]), key_id(rms[key])
        valid_keys = left.notna().all(axis=1)
        retail_keys = pd.MultiIndex.from_frame(rms[keys].dropna().drop_duplicates())
        left['_exact_rms_key'] = valid_keys & pd.MultiIndex.from_frame(left[keys]).isin(retail_keys)
        left['_price_status'] = frame._price_status.to_numpy()
        left['_missing_join_key'] = ~valid_keys
        # Adjacent dates test date conventions; they do not authorize fuzzy joins.
        nearby = np.zeros(len(left), dtype=bool)
        for days in [-7, -1, 1, 7]:
            shifted = left[keys].copy()
            shifted['week_end'] = shifted.week_end.to_numpy(dtype='datetime64[ns]') + np.timedelta64(days, 'D')
            nearby |= pd.MultiIndex.from_frame(shifted).isin(retail_keys)
        left['_nearby_rms_key_only'] = nearby & ~left._exact_rms_key.to_numpy() & valid_keys.to_numpy()
        left.groupby(['_price_status', '_missing_join_key', '_exact_rms_key', '_nearby_rms_key_only'], dropna=False).size().rename('rows').to_csv(output / 'rms_key_cross_tab.csv')
        left.loc[~left._exact_rms_key].head(200).to_csv(output / 'unmatched_key_examples.csv', index=False)
        if source_price in rms and source_price not in keys:
            rms['_source_price_status'] = status(rms[source_price])
            rms['_source_positive_price'] = pd.to_numeric(rms[source_price], errors='coerce').where(rms._source_price_status.eq('valid'))
            source = rms.dropna(subset=keys).groupby(keys).agg(
                distinct_source_rows=(source_price, 'size'),
                positive_source_values=('_source_positive_price', 'count'),
                source_price_min=('_source_positive_price', 'min'),
                source_price_max=('_source_positive_price', 'max')).reset_index()
            matched = left.merge(source, on=keys, how='left', validate='many_to_one')
            matched['_any_positive_source_price'] = matched.positive_source_values.fillna(0).gt(0)
            matched['_conflicting_positive_source_prices'] = matched.source_price_max.gt(matched.source_price_min)
            matched.groupby(['_price_status', '_exact_rms_key', '_any_positive_source_price',
                             '_conflicting_positive_source_prices'], dropna=False).size().rename('rows').to_csv(output / 'rms_source_price_cross_tab.csv')
            matched.loc[matched._price_status.ne('valid') & matched._any_positive_source_price].head(200).to_csv(
                output / 'invalid_target_with_valid_source_examples.csv', index=False)
            report['rms_source_price_note'] = 'A positive source price at an exact normalized key with invalid target price suggests checking the merge, column selection, or subsequent cleaning. Multiple distinct positive source prices at a key flag a scanner-grain ambiguity; no source value is selected automatically.'
        report['rms_exact_key_rows'] = int(left._exact_rms_key.sum())
        report['rms_note'] = 'Keys normalized for dates, whitespace, numeric .0 and leading zeros. Exact key existence does not establish nonmissing RMS price values. Nearby dates are diagnostic only. RMS keys are collected in memory.'
    example_columns = list(dict.fromkeys([c for c in ['household_code', 'store_code_uc', 'upc', 'week_end', args.flavor_column,
                           'quantity', args.price_column, '_price_status'] + candidates if c in frame]))
    frame.loc[frame._invalid_price, example_columns].groupby('_price_status', group_keys=False).head(20).to_csv(output / 'invalid_examples.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ['rows', 'price_column', 'price_status_counts']}, indent=2))
    print(f'Audit saved to {output}')


if __name__ == '__main__':
    main()

"""Observed shopping-week calendar. Empirical reads execute on the server only."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

_root = next(p for p in Path(__file__).resolve().parents
             if (p / 'Code/privacy/server_data_policy.py').is_file())
sys.path.insert(0, str(_root / 'Code/privacy'))
from server_data_policy import server_only, require_data_path


def flag_yogurt_lines(source, known_yogurt_upcs=None):
    """Retail-known yogurt UPCs supplement HMS module classifications.

    The original match selects RMS modules, not HMS modules. HMS metadata alone
    therefore cannot define the pre-match yogurt total used for reconciliation.
    """
    text = pl.col('upc').cast(pl.String).str.strip_chars()
    canonical = text.str.replace(r'\.0+$', '').str.strip_chars_start('0')
    source = source.with_columns(pl.when(text.str.contains(r'^[0-9]+(?:\.0+)?$'))
        .then(pl.when(canonical == '').then(pl.lit('0')).otherwise(canonical))
        .otherwise(text).alias('upc'))
    if known_yogurt_upcs is not None:
        known = known_yogurt_upcs.select('upc').unique().with_columns(pl.lit(True).alias('_retail_yogurt'))
        source = source.join(known, on='upc', how='left', validate='m:1')
        retail_known = pl.col('_retail_yogurt').fill_null(False)
    else:
        retail_known = pl.lit(False)
    module_known = pl.col('product_module_code').cast(pl.Int64, strict=False).is_in([3603, 3612]).fill_null(False)
    return source.with_columns((module_known | retail_known).alias('_is_yogurt'))


@server_only
def load_observed_weeks(panel, households, settings, known_yogurt_upcs=None):
    """Aggregate the full shopping panel, restricted to existing demand households.

    One observed week requires at least one recorded shopping trip. Exposure is
    the store with most distinct trips (store code breaks ties). Yogurt totals
    BEFORE retail matching distinguish true zeros from unmatched purchases.
    """
    require_data_path(panel)
    source = pl.scan_parquet(panel)
    dma = getattr(settings, 'household_dma_column', 'dma_cd')
    required = {'household_code', 'household_size', 'panel_year', 'purchase_date',
                'trip_code_uc', 'store_code_uc', 'product_module_code', 'upc', 'quantity', dma}
    missing = required - set(source.collect_schema().names())
    if missing:
        raise ValueError(f'Full shopping panel is missing columns: {sorted(missing)}')
    source = (source.filter(
        pl.col('panel_year').is_between(settings.first_year, settings.last_year)
        & (pl.col('household_size').cast(pl.Float64, strict=False) == 1)
        & pl.col('household_code').is_in(pl.Series(households).implode()))
        .select(*sorted(required))
        .with_columns(pl.col('purchase_date').cast(pl.String).str.slice(0, 10).str.to_date()))
    source = source.with_columns((pl.col('purchase_date') + pl.duration(
        days=(6 - pl.col('purchase_date').dt.weekday()) % 7)).alias('week'))
    source = flag_yogurt_lines(source, known_yogurt_upcs)
    yogurt = pl.col('_is_yogurt')
    quantity = pl.col('quantity').cast(pl.Float64, strict=False)
    # Returns/invalid yogurt quantities cannot establish a true zero.
    invalid = yogurt & (quantity.is_null() | ~quantity.is_finite() | (quantity <= 0))
    stores = (source.group_by(['household_code', 'week', 'store_code_uc']).agg(
        pl.col('trip_code_uc').n_unique().alias('trips'),
        pl.when(yogurt & (quantity > 0)).then(quantity).otherwise(0.).sum().alias('raw_yogurt'),
        invalid.fill_null(False).any().alias('invalid_yogurt'),
        pl.col(dma).first().alias('dma_cd'), pl.col(dma).n_unique().alias('dma_values'))
        .collect(engine='streaming').to_pandas())
    if stores.empty:
        raise ValueError('No full-panel weeks for the selected demand households.')
    stores['week'] = pd.to_datetime(stores['week'])
    keys = ['household_code', 'week']
    if (stores['dma_values'] != 1).any() or (stores.groupby(keys)['dma_cd'].nunique() != 1).any():
        raise ValueError('Consumer DMA must be unique within each observed household-week.')
    totals = stores.groupby(keys, as_index=False).agg(
        raw_yogurt=('raw_yogurt', 'sum'), invalid_yogurt=('invalid_yogurt', 'any'))
    exposure = (stores.sort_values(keys + ['trips', 'store_code_uc'],
                                  ascending=[True, True, False, True])
                .drop_duplicates(keys)[keys + ['store_code_uc', 'dma_cd']])
    return exposure.merge(totals, on=keys, validate='one_to_one')


def include_observed_weeks(weekly, calendar, flavors):
    """Pure calculation; missing matched yogurt purchases never become outside choices."""
    keys = ['household_code', 'week']
    if calendar.duplicated(keys).any() or weekly.duplicated(keys).any():
        raise ValueError('Expected one row per household-week.')
    covered = weekly[keys].merge(calendar[keys], on=keys, how='left', indicator=True)
    if (covered['_merge'] != 'both').any():
        raise ValueError('Matched purchase weeks are absent from the full-panel calendar.')
    rename = {c: 'calendar_' + c for c in ['store_code_uc', 'dma_cd']}
    result = calendar.rename(columns=rename).merge(weekly, on=keys, how='left', validate='one_to_one')
    for c in ['store_code_uc', 'dma_cd']:
        if c not in result:
            result[c] = result['calendar_' + c]
        else:
            if c == 'dma_cd':
                known = result[c].notna()
                if (result.loc[known, c].astype(float) != result.loc[known, 'calendar_' + c].astype(float)).any():
                    raise ValueError('Full-panel and matched-purchase consumer DMA disagree.')
            result[c] = result[c].fillna(result['calendar_' + c])
    result[flavors] = result[flavors].fillna(0.)
    matched = result[flavors].sum(axis=1)
    surplus = matched > result['raw_yogurt'] + 1e-8
    if surplus.any():
        # Aggregate diagnostics only: no household codes, UPCs or record contents.
        diagnostic = dict(affected_weeks=int(surplus.sum()),
                          affected_share=float(surplus.mean()),
                          matched_total=float(matched.loc[surplus].sum()),
                          panel_total=float(result.loc[surplus, 'raw_yogurt'].sum()),
                          panel_zero_weeks=int((result.loc[surplus, 'raw_yogurt'] == 0).sum()))
        raise ValueError('Matched yogurt quantity exceeds full-panel yogurt quantity. '
                         'Check that the panel and prepared sample belong to the same build; '
                         'possible duplicate matched lines or catalog disagreement. Aggregate diagnostic: '
                         + str(diagnostic))
    incomplete = result['invalid_yogurt'] | ~np.isclose(matched, result['raw_yogurt'])
    audit = dict(observed_panel_weeks=len(result),
                 unmatched_or_invalid_yogurt_weeks=int(incomplete.sum()),
                 zero_purchase_weeks=int(((result['raw_yogurt'] == 0) & ~incomplete).sum()),
                 matched_purchase_weeks=int(((matched > 0) & ~incomplete).sum()))
    result = result.loc[~incomplete].copy()
    result['zero_purchase_week'] = result['raw_yogurt'] == 0
    return result.drop(columns=list(rename.values()) + ['raw_yogurt', 'invalid_yogurt']), audit


def fill_zero_week_prices(weekly, market_prices, flavors):
    """Use same-week consumer-DMA flavor means for uncovered zero-week store quotes.

    Purchase-week pricing is unchanged. No temporal interpolation or free prices.
    """
    keys = ['week', 'dma_cd']
    market = market_prices.rename(columns={f: 'market_' + f for f in flavors})
    result = weekly.merge(market, on=keys, how='left', validate='many_to_one')
    used = np.zeros(len(result), dtype=bool)
    for f in flavors:
        price = 'price_' + f
        missing = ~np.isfinite(result[price]) | (result[price] <= 0)
        fill = result['zero_purchase_week'] & missing & (result['market_' + f] > 0)
        result.loc[fill, price] = result.loc[fill, 'market_' + f]
        used |= fill.to_numpy()
    result['zero_week_market_price'] = used
    return result.drop(columns=['market_' + f for f in flavors])

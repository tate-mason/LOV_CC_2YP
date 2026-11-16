"""Quick, separate server-only IV substitution tests: berry and other versus plain.

Q_f = b_own*p_f + b_plain*p_plain + household FE + week FE + error.
Instrument BOTH prices with corresponding same-week leave-other-DMA prices.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import polars as pl
from linearmodels.iv import IV2SLS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'demand'))
sys.path.insert(0, str(ROOT.parents[1]/'Code/privacy'))
from server_data_policy import server_only, require_data_path
import estimation as base


@server_only
def flavor_instruments(settings, weekly):
    """Separate leave-DMA flavor quote means; do not reuse the pooled baseline IV."""
    source = pl.scan_parquet(settings.sample)
    schema = source.collect_schema().names()
    source = source.filter(pl.col('panel_year').is_between(settings.first_year, settings.last_year))
    source = source.with_columns(base.normalized_upc_expr())
    if 'flavor' not in schema:
        mapping = base.read_flavors(settings.flavors)
        source = source.join(pl.from_pandas(mapping).lazy(),on='upc',how='left')
    price = next(c for c in ['price','price_rms'] if c in schema)
    source = source.with_columns(pl.col(price).cast(pl.Float64).alias('quote_price'),
        pl.col('purchase_date').cast(pl.String).str.slice(0,10).str.to_date().alias('date'))
    source = source.with_columns((pl.col('date')+pl.duration(days=(6-pl.col('date').dt.weekday())%7)).alias('week'))
    quotes = source.filter(pl.col('quote_price').is_finite() & (pl.col('quote_price')>0)).select(
        ['store_code_uc','week','upc','dma_cd','flavor','quote_price']).unique().collect(engine='streaming').to_pandas()
    quotes['dma_cd'] = base.check_dma(quotes.dma_cd,'instrument quotes')
    quotes['week'] = pd.to_datetime(quotes.week)
    # Equal weight per deduplicated store/week/UPC quote, not per purchase record.
    market = quotes.groupby(['week','flavor','dma_cd']).quote_price.agg(['sum','count']).reset_index()
    totals = market.groupby(['week','flavor'])[['sum','count']].sum().rename(columns={'sum':'total_sum','count':'total_count'}).reset_index()
    exposure = weekly[['week','dma_cd']].drop_duplicates().merge(
        pd.DataFrame({'flavor':['other','berry','plain']}),how='cross')
    exposure = exposure.merge(totals,on=['week','flavor'],how='left',validate='many_to_one')
    exposure = exposure.merge(market,on=['week','flavor','dma_cd'],how='left',validate='many_to_one')
    denominator = exposure.total_count-exposure['count'].fillna(0)
    exposure['instrument'] = (exposure.total_sum-exposure['sum'].fillna(0))/denominator.where(denominator>0)
    wide = exposure.pivot(index=['week','dma_cd'],columns='flavor',values='instrument').rename(columns=lambda f:'iv_'+f).reset_index()
    weekly = weekly.drop(columns=['iv_other','iv_berry','iv_plain'],errors='ignore')
    return weekly.merge(wide,on=['week','dma_cd'],how='left',validate='many_to_one')


def fit_iv(y, prices, instruments, clusters):
    """Inputs must already have household/week effects removed."""
    return IV2SLS(y, None, prices, instruments).fit(cov_type='clustered', clusters=clusters)


@server_only
def run(args):
    for path in [args.sample, args.flavors, args.output]:
        require_data_path(path)
    if args.panel is not None:
        require_data_path(args.panel)
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    base.FLAVORS[:] = ['other', 'berry', 'plain']
    settings = SimpleNamespace(**vars(args), products_extra=None, max_households=None,
        household_dma_column='dma_cd', habit_weight=.7, no_price_control=True,
        stock_measure=base.DEFAULT_STOCK_MEASURE, stock_retention=base.DEFAULT_STOCK_RETENTION,
        novelty_measure='previous_share', observation_mode='observed_panel', flavor_spec='three', capacity=7)
    frame = base.prepare_data(settings)
    frame = flavor_instruments(settings,frame)
    reports, first_stages, metadata = [], [], []
    for outcome in ['berry', 'other']:
        columns = [outcome, 'price_'+outcome, 'price_plain', 'iv_'+outcome, 'iv_plain']
        keep = (np.isfinite(frame[columns]).all(axis=1)
                & (frame[['price_'+outcome, 'price_plain']] > 0).all(axis=1)
                & frame[['household_code','week','dma_cd']].notna().all(axis=1))
        sample = frame.loc[keep].copy()
        if sample.empty or sample.duplicated(['household_code','week']).any():
            raise ValueError('Empty or duplicate household/week regression sample')
        # Same observations and the same FE projection for outcome, both prices, both IVs.
        residual = base.absorb_effects(sample[columns].to_numpy(float),
                                      [sample.household_code, sample.week])
        clusters = pd.DataFrame({name:pd.factorize(sample[name])[0] for name in ['dma_cd','week']})
        if (clusters.nunique() < 2).any():
            raise ValueError('Need at least two DMA and week clusters')
        prices = pd.DataFrame(residual[:,1:3],columns=['price_'+outcome,'price_plain'])
        instruments = pd.DataFrame(residual[:,3:5],columns=['iv_'+outcome,'iv_plain'])
        fit = fit_iv(residual[:,0],prices,instruments,clusters)
        interval = fit.conf_int()
        for term in fit.params.index:
            reports.append(dict(outcome=outcome,term=term,estimate=fit.params[term],
                std_error=fit.std_errors[term],pvalue=fit.pvalues[term],
                ci_lower=interval.loc[term,'lower'],ci_upper=interval.loc[term,'upper']))
        first_stages.append(fit.first_stage.diagnostics.reset_index(names='price').assign(outcome=outcome))
        metadata.append(dict(outcome=outcome,weeks=len(sample),households=sample.household_code.nunique(),
            dma_clusters=sample.dma_cd.nunique(),week_clusters=sample.week.nunique(),
            zero_quantity_share=float(sample[outcome].eq(0).mean())))
    args.output.mkdir(parents=True)
    pd.DataFrame(reports).to_csv(args.output/'cross_price_estimates.csv',index=False)
    pd.concat(first_stages,ignore_index=True).to_csv(args.output/'first_stage_diagnostics.csv',index=False)
    (args.output/'specification.json').write_text(json.dumps(dict(
        equations=['Q_berry ~ p_berry + p_plain','Q_other ~ p_other + p_plain'],
        effects=['household','week'],clusters=['DMA','week'],
        instruments='Separate same-week flavor quote means outside own DMA; both IVs in both first stages; quote mix is not fixed',
        interpretation='Positive plain-price coefficient supports substitution; contemporaneous levels, not a lagged response',
        inference='Asymptotic clustered IV covariance; no finite-sample FE degrees-of-freedom correction',
        quantity='Prepared weekly flavor purchase count, including zero purchases',
        samples=metadata),indent=2)+'\n')
    print(pd.DataFrame(reports).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['sample','flavors','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--first-year',type=int,default=2006)
    p.add_argument('--last-year',type=int,default=2015)
    run(p.parse_args())

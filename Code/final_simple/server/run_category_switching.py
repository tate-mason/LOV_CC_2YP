"""Server-only refit: weekly yogurt participation, flavor tastes, no habit."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

PROJECT = next(p for p in Path(__file__).resolve().parents
               if (p/'Code/privacy/server_data_policy.py').is_file())
sys.path.insert(0, str(PROJECT/'Code/privacy'))
from server_data_policy import require_server, require_data_path


def prepare(settings, flavors):
    """Read only on server. Keep all weekly states before likelihood exclusions."""
    require_server()
    for path in [settings.sample, settings.panel, settings.flavors]:
        require_data_path(path)
    import numpy as np
    import pandas as pd
    import polars as pl
    from demand import estimation as base
    from demand.observed_weeks import load_observed_weeks

    sample = pl.scan_parquet(settings.sample)
    columns = sample.collect_schema().names()
    sample = sample.filter(pl.col('panel_year').is_between(settings.first_year, settings.last_year))
    if 'flavor' not in columns:
        mapping = base.read_flavors(settings.flavors)
        sample = sample.with_columns(base.normalized_upc_expr()).join(
            pl.from_pandas(mapping).lazy(), on='upc', how='left')
    price_name = next((key for key in ['price', 'price_rms'] if key in columns), None)
    if price_name is None:
        raise ValueError('Prepared sample lacks a quoted price column.')
    sample = sample.with_columns(
        pl.col('purchase_date').cast(pl.String).str.slice(0, 10).str.to_date(),
        pl.col(price_name).cast(pl.Float64).alias('_price'))
    sample = sample.with_columns((pl.col('purchase_date')+pl.duration(
        days=(6-pl.col('purchase_date').dt.weekday()) % 7)).alias('week'))
    cohort = sample.filter(pl.col('household_size').cast(pl.Float64, strict=False) == 1)
    ids = cohort.select('household_code').unique().collect(engine='streaming')['household_code'].to_numpy()
    calendar = load_observed_weeks(settings.panel, ids, settings,
        known_yogurt_upcs=sample.select('upc').unique().with_columns(base.normalized_upc_expr()))
    counts = (cohort.filter(pl.col('flavor').is_in(flavors))
        .group_by(['household_code', 'week', 'flavor']).agg(pl.col('quantity').sum())
        .collect(engine='streaming').to_pandas())
    counts['week'] = pd.to_datetime(counts.week)
    counts = counts.pivot(index=['household_code', 'week'], columns='flavor', values='quantity')
    counts = counts.reindex(columns=flavors, fill_value=0).fillna(0).reset_index()
    calendar['week'] = pd.to_datetime(calendar.week)
    weekly = calendar.merge(counts, on=['household_code', 'week'], how='left', validate='one_to_one')
    weekly[flavors] = weekly[flavors].fillna(0.)
    matched = weekly[flavors].sum(axis=1).to_numpy(float)
    if np.any(matched > weekly.raw_yogurt.to_numpy()+1e-8):
        raise ValueError('Matched yogurt exceeds full-panel yogurt; verify source build.')
    # Category participation is known from the full panel even when the flavor
    # basket is incompletely retail-matched. Such weeks omit only the flavor term.
    weekly['purchase'] = (weekly.raw_yogurt > 0).astype(int)
    weekly['state_known'] = (weekly.raw_yogurt > 0) | ~weekly.invalid_yogurt
    flavor_known = np.isclose(matched, weekly.raw_yogurt) & ~weekly.invalid_yogurt
    weekly.loc[~flavor_known, flavors] = 0.

    # Common menu exposures, not chosen basket prices. Zero-yogurt weeks receive
    # store/week quotes or a same-week DMA fallback, never a time interpolation.
    quotes = (sample.filter(pl.col('_price').is_finite() & (pl.col('_price') > 0))
        .select('store_code_uc', 'week', 'upc', 'flavor',
                pl.col('dma_cd').cast(pl.Float64).cast(pl.Int64).alias('dma_cd'), '_price').unique())
    store = (quotes.select('store_code_uc', 'week', 'upc', 'flavor', '_price').unique()
        .group_by(['store_code_uc', 'week', 'flavor']).agg(pl.col('_price').mean())
        .collect(engine='streaming').to_pandas())
    store = store.pivot(index=['store_code_uc', 'week'], columns='flavor', values='_price')
    store = store.reindex(columns=flavors).rename(columns={f:'price_'+f for f in flavors}).reset_index()
    store['week'] = pd.to_datetime(store.week)
    market = (quotes.group_by(['dma_cd', 'week', 'flavor']).agg(pl.col('_price').mean())
        .collect(engine='streaming').to_pandas())
    market = market.pivot(index=['dma_cd', 'week'], columns='flavor', values='_price')
    market = market.reindex(columns=flavors).rename(columns={f:'market_'+f for f in flavors}).reset_index()
    market['week'] = pd.to_datetime(market.week)
    weekly = weekly.merge(store, on=['store_code_uc', 'week'], how='left', validate='many_to_one')
    weekly = weekly.merge(market, on=['dma_cd', 'week'], how='left', validate='many_to_one')
    fallback = 0
    for f in flavors:
        missing = ~np.isfinite(weekly['price_'+f]) | (weekly['price_'+f] <= 0)
        usable = missing & np.isfinite(weekly['market_'+f]) & (weekly['market_'+f] > 0)
        fallback += int(usable.sum())
        weekly.loc[usable, 'price_'+f] = weekly.loc[usable, 'market_'+f]
    weekly = weekly.sort_values(['household_code', 'week']).reset_index(drop=True)
    previous = weekly.groupby('household_code').purchase.shift()
    previous_known = weekly.groupby('household_code').state_known.shift().fillna(False).astype(bool)
    weekly['history_sign'] = 1-2*previous
    first = weekly.drop_duplicates('household_code', keep='first').copy()
    prices = weekly[['price_'+f for f in flavors]].to_numpy(float)
    covered = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid = covered & weekly.state_known.to_numpy() & previous_known.to_numpy()
    scoring = weekly.loc[valid].copy()
    eligible = scoring.groupby('household_code').size()
    eligible = eligible[eligible >= 2].index
    scoring = scoring[scoring.household_code.isin(eligible)].copy()
    first = first[first.household_code.isin(eligible)].copy()
    unknown_initial = int((~first.state_known).sum())
    eligible = first.loc[first.state_known, 'household_code']
    first = first[first.state_known].copy()
    scoring = scoring[scoring.household_code.isin(eligible)].copy()
    if scoring.empty:
        raise ValueError('No usable households after weekly state/price coverage checks.')
    labels = sorted(scoring.household_code.unique())
    data = dict(hh=pd.Categorical(scoring.household_code, categories=labels).codes,
        n_households=len(labels), prices=scoring[['price_'+f for f in flavors]].to_numpy(float),
        available=np.ones((len(scoring), 3), dtype=bool),
        purchase=scoring.purchase.to_numpy(float), history_sign=scoring.history_sign.to_numpy(float))
    audit = dict(observation_unit='observed shopping week; all trips within week pooled',
        timing='consecutive observed shopping weeks; no adjacency requirement',
        observed_weeks=len(weekly), scoring_weeks=len(scoring), cohort_households=len(ids),
        fitted_households=len(labels), initial_inside=int(first.purchase.sum()),
        initial_outside=int((1-first.purchase).sum()), excluded_unknown_initial=unknown_initial,
        zero_yogurt_scoring_weeks=int((scoring.purchase == 0).sum()),
        category_only_scoring_weeks=int(((scoring.purchase == 1)&(scoring[flavors].sum(axis=1)==0)).sum()),
        market_price_fallback_cells=fallback,
        excluded_price_coverage_weeks=int((~covered).sum()),
        cohort='existing single-person matched-yogurt cohort; not expanded to never-yogurt households',
        price_exposure='mean quoted UPC price per flavor/store/week; same-week DMA fallback',
        history='previous actual observed shopping week, constructed before price exclusions')
    return data, audit


def run(args):
    require_server()
    for p in [args.fit_dir, args.output]:
        require_data_path(p)
    if args.previous_results:
        require_data_path(args.previous_results)
    fit = json.loads((args.fit_dir/'fit.json').read_text())
    if fit.get('spec') != 'novelty_only' or fit.get('habit') is not False:
        raise ValueError('Use the prior novelty_only fit as the input/provenance source.')
    saved = dict(fit['settings'])
    for key in ['sample', 'panel', 'flavors']:
        saved[key] = Path(saved[key]); require_data_path(saved[key])
    if args.output.exists():
        raise FileExistsError('Output already exists; use a new directory.')
    args.output.mkdir(parents=True)
    os.environ['MPLCONFIGDIR'] = str(args.output/'matplotlib_cache')
    for folder in ['Code/final_simple', 'Code/final_simple/demand', 'Code/final_simple/supply']:
        sys.path.insert(0, str(PROJECT/folder))
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    from demand.category_switching import PARAMETERS, FLAVORS, draws, objective, pooled_prediction
    from demand.parallel_score import parallel_objective
    from category_report import build
    settings = SimpleNamespace(**dict(saved, household_dma_column='dma_cd'))
    data, audit = prepare(settings, FLAVORS)
    exported = args.output/'RESULTS_ONLY'; exported.mkdir()
    (exported/'sample_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)
    old = pd.read_csv(args.fit_dir/'point_estimates.csv').set_index('parameter').estimate
    initial = np.array([old[k] for k in PARAMETERS])
    bounds = [(-20,20), (-20,20), (-20,20), (-6,3), (0,3), (-20,20), (0,3)]
    adapter = SimpleNamespace(demand_objective_gradient=lambda t,d,z,m,household=False:
        objective(t,d,z,household=household))
    random = np.random.default_rng(args.seed+1)
    converged, final_draws = [], None
    # A quick stage supplies a starting point; only a converged full-draw fit
    # generates pricing. Starting values are not treated as new estimates.
    with parallel_objective(adapter, data, args.workers):
        for stage, number, starts in [('quick', args.quick_draws, 1), ('refined', args.draws, args.starts)]:
            integration = draws(number, args.seed)
            results = []
            def function(t):
                return adapter.demand_objective_gradient(t,data,integration,None)
            for start in range(starts):
                x0 = initial if start == 0 else np.clip(initial+random.normal(0,.2,len(initial)),
                    np.array(bounds)[:,0]+1e-5,np.array(bounds)[:,1]-1e-5)
                count = [0]
                def progress(t):
                    count[0] += 1
                    if count[0] % 10 == 0:
                        print(f'{stage} start {start+1}: iteration {count[0]}', flush=True)
                result = minimize(function,x0,jac=True,bounds=bounds,method='L-BFGS-B',
                    callback=progress,options=dict(maxiter=args.maxiter,ftol=1e-10,gtol=1e-6))
                results.append(result)
                checkpoint = dict(stage=stage,draws=number,start=start+1,converged=bool(result.success),
                    nll=float(result.fun*data['n_households']),message=str(result.message),parameters=result.x.tolist())
                (exported/'optimization_status.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
                print(json.dumps(checkpoint),flush=True)
            valid = [r for r in results if r.success]
            best = min(valid or results,key=lambda r:r.fun)
            initial = best.x
            if stage == 'refined':
                converged, final_draws = valid, integration
        if not converged:
            raise RuntimeError('No full-draw start converged; pricing was not generated.')
        table = pd.DataFrame(dict(parameter=PARAMETERS,estimate=best.x,
            at_bound=[min(v-lo,hi-v)<1e-4 for v,(lo,hi) in zip(best.x,bounds)]))
        table.to_csv(exported/'point_estimates.csv',index=False)
        value, gradient = function(best.x)
        (exported/'fit_summary.json').write_text(json.dumps(dict(
            model='category_switching_with_flavor_tastes',converged=True,habit=False,
            nll=float(value*data['n_households']),draws=args.draws,seed=args.seed,
            gradient_max=float(np.abs(gradient).max()),flavors=FLAVORS,
            likelihood='one Bernoulli yogurt/no-yogurt outcome per observed shopping week',
            package_counts_used_as_choices=False,
            category_history='yogurt versus no yogurt, not flavor switching',
            price_control=False,observation_unit='observed shopping week',learning=False,
            population_distribution_same_across_initial_states=True,
            input_parameter_sha256=hashlib.sha256((args.fit_dir/'point_estimates.csv').read_bytes()).hexdigest()),indent=2)+'\n')
        if table.at_bound.any():
            raise RuntimeError('New fit has boundary parameters. Estimates saved; pricing withheld for review.')
    prediction,participation = pooled_prediction(best.x,data,final_draws,return_participation=True)
    diagnostics = []
    for sign,label in [(-1,'previous yogurt'),(1,'previous no yogurt')]:
        mask = data['history_sign'] == sign
        if mask.any():
            diagnostics.append(dict(history=label,weeks=int(mask.sum()),
                observed_yogurt_rate=float(data['purchase'][mask].mean()),
                predicted_yogurt_rate=float(participation[mask].mean())))
    pd.DataFrame(diagnostics).to_csv(exported/'fit_check_by_history.csv',index=False)
    comparison = table[['parameter','estimate']].rename(columns={'estimate':'new_category_estimate'})
    comparison['previous_flavor_switching_estimate'] = comparison.parameter.map(old)
    comparison.to_csv(exported/'parameter_comparison.csv',index=False)
    build(exported, audit, previous_results=args.previous_results)
    print('Complete. Recover only '+str(exported),flush=True)


if __name__ == '__main__':
    require_server()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous-results',type=Path)
    parser.add_argument('--quick-draws',type=int,default=32)
    parser.add_argument('--draws',type=int,default=128)
    parser.add_argument('--starts',type=int,default=3)
    parser.add_argument('--maxiter',type=int,default=600)
    parser.add_argument('--seed',type=int,default=306)
    parser.add_argument('--workers',type=int,default=int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    args = parser.parse_args()
    if min(args.starts,args.maxiter,args.workers)<1:
        parser.error('Need positive starts/maxiter/workers.')
    run(args)

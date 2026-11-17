"""Separate server-only demand variants. Does not alter baseline estimation or pricing."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'demand'))
sys.path.insert(0, str(ROOT.parents[1]/'Code/privacy'))
from server_data_policy import server_only, require_data_path
import estimation as base
from flavor_specs import categories
from demand_models import SPECS, names, draws, objective, alpha_moments


@server_only
def run(args):
    for path in [args.sample, args.flavors, args.output]:
        require_data_path(path)
    if args.panel is not None:
        require_data_path(args.panel)
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    if args.starts < 1 or args.maxiter < 1 or args.capacity < 1:
        raise ValueError('Need positive starts, iterations and capacity')
    flavors = categories(args.flavor_spec)
    base.FLAVORS[:] = flavors
    settings = SimpleNamespace(**vars(args), products_extra=None, max_households=None,
        household_dma_column='dma_cd', habit_weight=0. if args.spec == 'novelty_only' else .7,
        stock_measure=base.DEFAULT_STOCK_MEASURE, stock_retention=base.DEFAULT_STOCK_RETENTION)
    # Price controls are reconstructed in memory; no research inputs are exported.
    frame = base.prepare_data(settings)
    first_stage = None
    if not args.no_price_control:
        frame, first_stage = base.add_price_controls(frame)
    data = base.make_arrays(frame, args.capacity)
    parameters = names(args.spec, flavors, data['control'])
    integration = draws(args.draws, args.seed, args.spec, flavors)
    bounds = [(0., 3.) if n.startswith('sd_') else (-6., 3.) if n == 'mu_log_minus_alpha' else (-20., 20.) for n in parameters]
    start = np.array([.3 if n.startswith('sd_') else 0. for n in parameters])
    rng = np.random.default_rng(args.seed+1)
    fits = []
    for k in range(args.starts):
        initial = start if k == 0 else np.clip(start+rng.normal(0,.5,len(start)), np.array(bounds)[:,0], np.array(bounds)[:,1])
        fit = minimize(objective, initial, args=(data,integration,args.spec,flavors), jac=True,
                       method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=args.maxiter, ftol=1e-10, gtol=1e-6))
        fits.append(fit)
        print(f'Start {k+1}: converged={fit.success}; NLL={fit.fun*data["n_households"]:.6f}', flush=True)
    converged = [f for f in fits if f.success]
    best = min(converged or fits, key=lambda f:f.fun)
    args.output.mkdir(parents=True)
    table = pd.DataFrame(dict(parameter=parameters, estimate=best.x,
        at_bound=[min(abs(v-lo),abs(v-hi)) < 1e-4 for v,(lo,hi) in zip(best.x,bounds)]))
    table.to_csv(args.output/'point_estimates.csv',index=False)
    inference = dict(status='not run: fit did not converge')
    if best.success:
        errors,inference = base.hessian_standard_errors(best.x,
            lambda t:objective(t,data,integration,args.spec,flavors)[1],
            data['n_households'],bounds,parameters,args.output)
        if errors is not None:
            table = table.merge(errors,on='parameter',validate='one_to_one')
    table.to_csv(args.output/'estimates.csv',index=False)
    pd.DataFrame([alpha_moments(dict(zip(parameters, best.x)))]).to_csv(
        args.output/'transformed_alpha.csv', index=False)
    if first_stage is not None:
        first_stage.to_csv(args.output/'first_stage.csv',index=False)
    metadata = dict(spec=args.spec, flavors=flavors, converged=bool(best.success),
        nll=float(best.fun*data['n_households']), households=data['n_households'], weeks=len(frame),
        habit=args.spec != 'novelty_only', alpha='common' if args.spec == 'random_beta_common_alpha' else 'lognormal',
        beta='independent normal for every flavor' if args.spec == 'random_beta_common_alpha' else 'common',
        gamma='heterogeneous normal novelty', inference=inference,
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.output/'fit.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',choices=SPECS,required=True)
    for name in ['sample','flavors','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--panel',type=Path)
    p.add_argument('--flavor-spec',choices=['three','five'],default='three')
    p.add_argument('--observation-mode',choices=['observed_panel','purchase_weeks'],default='observed_panel')
    p.add_argument('--novelty-measure',choices=['previous_share'],default='previous_share')
    p.add_argument('--first-year',type=int,default=2006)
    p.add_argument('--last-year',type=int,default=2015)
    p.add_argument('--capacity',type=int,default=7)
    p.add_argument('--draws',type=int,default=128)
    p.add_argument('--starts',type=int,default=3)
    p.add_argument('--maxiter',type=int,default=600)
    p.add_argument('--seed',type=int,default=306)
    p.add_argument('--no-price-control',action='store_true')
    run(p.parse_args())

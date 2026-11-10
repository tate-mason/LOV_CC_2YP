"""Server-only supply calibration to observed prices and fitted gamma types.

Same binary utility and fixed-belief timing as theory/gamma_uncertainty.py.
No coupon prices or counterfactual demand forecasts enter calibration targets.
"""
from pathlib import Path
import sys
_policy_root = next(p for p in Path(__file__).resolve().parents
                    if (p / 'Code/privacy/server_data_policy.py').is_file())
sys.path.insert(0, str(_policy_root / 'Code/privacy'))
from server_data_policy import server_only
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit, lambertw

GROUPS = ['low', 'middle', 'high']
PRIORS = [('Equal types', [1/3]*3), ('Low type more likely', [.5,.25,.25]),
          ('Medium type more likely', [.25,.5,.25]), ('High type more likely', [.25,.25,.5])]


def known_prices(gamma, h0, alpha, beta, delta):
    """Theory's known-type analytic prices; cost=0, alpha signed negative."""
    terminal_z = lambertw(np.exp(beta+np.array([-gamma, gamma])-1)).real
    terminal_price, terminal_value = -(1+terminal_z)/alpha, -terminal_z/alpha
    effective = -delta*(terminal_value[0]-terminal_value[1])
    intercept = beta+(-gamma if h0 == 'j' else gamma)
    first_z = float(lambertw(np.exp(intercept+alpha*effective-1)).real)
    p1 = effective-(1+first_z)/alpha
    s1 = float(expit(intercept+alpha*p1))
    profit = s1*p1+delta*(s1*terminal_value[0]+(1-s1)*terminal_value[1])
    return dict(profit=float(profit), expected_p1=float(p1),
                expected_p2=float(s1*terminal_price[0]+(1-s1)*terminal_price[1]))


def maximize(fun, bounds, points=161):
    grid = np.linspace(*bounds, points)
    values = np.array([fun(p) for p in grid])
    candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
    for i in range(1, points-1):
        if values[i] >= values[i-1] and values[i] >= values[i+1]:
            r = minimize_scalar(lambda p: -fun(p), bounds=(grid[i-1], grid[i+1]),
                                method='bounded', options={'xatol': 1e-10})
            candidates.append((-r.fun, r.x))
    value, price = max(candidates)
    if min(price-bounds[0], bounds[1]-price) < 1e-6:
        raise RuntimeError('Supply price optimum at numerical boundary')
    return float(price), float(value)


def uncertainty(gamma, prior, h0, alpha, beta, delta):
    prior = np.asarray(prior, float)
    scale = (30+abs(beta)+np.max(np.abs(gamma)))/abs(alpha)
    def shares(p, h):
        return expit(beta+alpha*p+(-gamma if h == 'j' else gamma))
    p2j, wj = maximize(lambda p: float(prior @ shares(p, 'j'))*p, (0., scale))
    p20, w0 = maximize(lambda p: float(prior @ shares(p, '0'))*p, (0., scale))
    def objective(p):
        s = float(prior @ shares(p, h0))
        return s*p+delta*(s*wj+(1-s)*w0)
    p1, value = maximize(objective, (-scale, scale))
    _, wide_value = maximize(objective, (-2*scale, 2*scale), 321)
    if not np.isclose(value, wide_value, atol=1e-7, rtol=1e-8):
        raise AssertionError('Supply price wider-grid audit failed')
    s1 = float(prior @ shares(p1, h0))
    return dict(profit=value, expected_p1=p1, expected_p2=s1*p2j+(1-s1)*p20,
                p2j=p2j, p20=p20, belief_j=prior.tolist(), belief_0=prior.tolist())


class CalibrationError(ValueError):
    """Observed price moments do not support an admissible binary calibration."""
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('Binary supply calibration rejected; see information_calibration_status.json')


def calibration_rank(jacobian):
    """Reject numerically flat directions, not merely exact floating-point zeros."""
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return int(np.sum(singular > max(1e-12, singular[0]*1e-7)))


def calibrate(moments, gamma, delta):
    """Weighted relative price-moment fit, common alpha/beta across type/h0 cells."""
    target = moments[['observed_p1', 'observed_p2']].to_numpy()
    weight = np.sqrt(moments.households.to_numpy()/moments.households.sum())[:, None]
    def predictions(x):
        alpha, beta = -np.exp(x[0]), x[1]
        return np.array([[r['expected_p1'], r['expected_p2']] for r in [
            known_prices(gamma[GROUPS.index(row.group)], row.h0, alpha, beta, delta)
            for row in moments.itertuples()]])
    def residual(x):
        return ((predictions(x)-target)/target*weight).ravel()
    start = -np.log(target.mean())
    fits = [least_squares(residual, [np.clip(start, -7., 4.), b],
                          bounds=([-8., -15.], [5., 15.]), x_scale='jac', max_nfev=4000,
                          xtol=1e-11, ftol=1e-11, gtol=1e-11)
            for b in [-1., 2., 5.]]
    # Choose among admissible solutions, not an inadmissible lower objective.
    attempts = [dict(success=bool(f.success), active_bounds=f.active_mask.tolist(),
                     jacobian_rank=calibration_rank(f.jac),
                     singular_values=np.linalg.svd(f.jac, compute_uv=False).tolist(),
                     parameters=f.x.tolist(), objective=float(np.sum(f.fun**2)),
                     message=str(f.message)) for f in fits]
    valid = [f for f in fits if f.success and not np.any(f.active_mask)
             and calibration_rank(f.jac) == 2 and np.isfinite(f.fun).all()]
    if not valid:
        raise CalibrationError(dict(status='failed', attempts=attempts,
                                    bounds=[[-8., -15.], [5., 15.]]))
    fit = min(valid, key=lambda r: np.sum(r.fun**2))
    result = moments.copy()
    result[['fitted_p1', 'fitted_p2']] = predictions(fit.x)
    result['relative_error_p1'] = (result.fitted_p1-result.observed_p1)/result.observed_p1
    result['relative_error_p2'] = (result.fitted_p2-result.observed_p2)/result.observed_p2
    return -float(np.exp(fit.x[0])), float(fit.x[1]), result, float(np.sqrt(np.mean(fit.fun**2)))


@server_only
def run_calibration(frame, households, delta, output):
    # frame is already permuted so berry is the flavor flavor. Require actual
    # consecutive-week observations; never carry one price into future periods.
    first = frame.sort_values(['household_code', 'week']).groupby('household_code').head(1)
    first = first.set_index('household_code').reindex(households.household_code)
    weekly = frame.set_index(['household_code', 'week'])
    observed = []
    for lag in [7, 14]:
        index = pd.MultiIndex.from_arrays([first.index, first.week+pd.Timedelta(days=lag)],
                                         names=['household_code', 'week'])
        observed.append(weekly.reindex(index).price_berry.to_numpy(float))
    targets = pd.DataFrame(dict(group=households.group.to_numpy(),
        h0=np.where(first.berry.to_numpy() > 0, 'j', '0'), p1=observed[0], p2=observed[1]))
    valid = np.isfinite(targets[['p1', 'p2']]).all(axis=1) & (targets[['p1', 'p2']] > 0).all(axis=1)
    targets = targets.loc[valid]
    if targets.empty:
        status = dict(status='failed', reason='No observed flavor prices at both t0+7 and t0+14',
                      coupon_pricing_affected=False)
        (output/'information_calibration_status.json').write_text(json.dumps(status, indent=2)+'\n')
        raise CalibrationError(status)
    moments = targets.groupby(['group', 'h0']).agg(
        households=('p1', 'size'), observed_p1=('p1', 'mean'), observed_p2=('p2', 'mean')).reset_index()
    gamma = households.groupby('group').switching_mean.mean().reindex(GROUPS).to_numpy(float)
    if not np.isfinite(gamma).all():
        raise ValueError('Missing fitted low/medium/high gamma types')
    try:
        alpha, beta, fitted_moments, rms = calibrate(moments, gamma, delta)
    except CalibrationError as error:
        moments.to_csv(output / 'information_calibration_moments.csv', index=False)
        status = dict(error.diagnostics, estimated_gamma=gamma.tolist(),
                      price_moment_cells=len(moments), delta=delta,
                      coupon_pricing_affected=False)
        (output/'information_calibration_status.json').write_text(json.dumps(status, indent=2)+'\n')
        raise
    fitted_moments.to_csv(output / 'information_calibration_moments.csv', index=False)
    rows, type_rows = [], []
    for scenario, prior in PRIORS:
        for h0 in ['j', '0']:
            known = [known_prices(g, h0, alpha, beta, delta) for g in gamma]
            full = {key: float(np.dot(prior, [r[key] for r in known])) for key in known[0]}
            unknown = uncertainty(gamma, prior, h0, alpha, beta, delta)
            persistent_unknown = []
            for k, label in enumerate(['low', 'medium', 'high']):
                s1 = float(expit(beta+alpha*unknown['expected_p1']+(-gamma[k] if h0 == 'j' else gamma[k])))
                s2j = float(expit(beta+alpha*unknown['p2j']-gamma[k]))
                s20 = float(expit(beta+alpha*unknown['p20']+gamma[k]))
                true_profit = s1*unknown['expected_p1']+delta*(
                    s1*unknown['p2j']*s2j+(1-s1)*unknown['p20']*s20)
                if known[k]['profit']+1e-7 < true_profit:
                    raise AssertionError('Known-type bound on persistent-type revenue failed')
                persistent_unknown.append(true_profit)
                type_rows.append(dict(scenario=scenario, h0=h0, type=label,
                    gamma=float(gamma[k]), probability=prior[k],
                    known_profit=known[k]['profit'], known_p1=known[k]['expected_p1'],
                    known_expected_p2=known[k]['expected_p2'],
                    unknown_policy_persistent_profit=true_profit))
            persistent_unknown_revenue = float(np.dot(prior, persistent_unknown))
            information_gain = full['profit'] - persistent_unknown_revenue
            for case, result in [('known', full), ('unknown', unknown)]:
                rows.append(dict(scenario=scenario, h0=h0, case=case,
                    **{key: result[key] for key in full},
                    persistent_type_revenue=(full['profit'] if case == 'known' else persistent_unknown_revenue),
                    value_of_information=information_gain))
    pd.DataFrame(type_rows).to_csv(output / 'information_friction_by_group.csv', index=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'information_friction_aggregate.csv', index=False)
    settings = dict(alpha=alpha, beta=beta, delta=delta, cost=0.,
        types=['low', 'medium', 'high'], estimated_gamma=gamma.tolist(), priors=PRIORS,
        beliefs='fixed in every state', baseline='known type supply pricing',
        calibrated_parameters=['alpha', 'beta'], calibration='household-count-weighted relative price moments',
        weighted_residual_rms=rms, households_in_price_targets=int(valid.sum()),
        households_without_complete_price_targets=int((~valid).sum()),
        observed_prices='actual weekly prices at t0+7 and t0+14 days',
        outcome_units='per model consumer, not population revenue totals',
        revenue_evaluation='same persistent consumer types under both policies',
        profit_column='fixed-belief optimization objective; use persistent_type_revenue for policy comparisons',
        assumption='observed mean offered prices target known-type optimal supply prices',
        gamma_mapping='fitted novelty group means used as symmetric binary theory gamma')
    (output / 'information_friction_settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    # State probabilities and estimated support are aggregate results only.
    pd.DataFrame([dict(scenario=name, type=label, gamma=float(gamma[k]), probability=prior[k],
                       belief_j=prior[k], belief_0=prior[k])
                  for name, prior in PRIORS for k, label in enumerate(['low','medium','high'])]
                 ).to_csv(output / 'information_friction_states.csv', index=False)
    (output/'information_calibration_status.json').write_text(json.dumps(
        dict(status='complete', alpha=alpha, beta=beta, weighted_residual_rms=rms), indent=2)+'\n')
    return table, settings

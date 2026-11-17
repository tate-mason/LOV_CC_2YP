"""Pure normal-gamma theory: quintile-informed versus population-uniform pricing."""
from pathlib import Path
from statistics import NormalDist
import argparse
import json
import sys

import numpy as np
import pandas as pd

from pricing_model import expit, scalar_maximum
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'demand'))
from binary_welfare import evaluate_policy
from plot_supply_bins import plot_bins


def solve_distribution(gamma, weights, h0, alpha, beta, delta):
    """Fixed-belief terminal offers; persistent-type introductory objective.

    No posterior updating occurs after either outcome. Terminal offers maximize
    revenue under the unchanged input distribution. The introductory decision
    integrates each persistent type's probability times its continuation revenue.
    """
    g, mass = np.asarray(gamma, float), np.asarray(weights, float)
    if (g.ndim != 1 or mass.shape != g.shape or len(g) == 0
            or not np.isfinite(g).all() or not np.isfinite(mass).all()
            or (mass <= 0).any() or not np.isclose(mass.sum(), 1.)
            or h0 not in ['j', '0'] or alpha >= 0 or not 0 <= delta <= 1
            or not np.isfinite([alpha, beta, delta]).all()):
        raise ValueError('Need finite types, positive normalized weights and valid pricing parameters.')
    scale = (30+abs(beta)+np.max(abs(g)))/abs(alpha)
    def share(p, h):
        return expit(beta+alpha*p+(-g if h == 'j' else g))
    terminal = [scalar_maximum(lambda p: p*float(mass@share(p, h)), (0., scale))
                for h in ['j', '0']]
    pj, p0 = terminal[0][0], terminal[1][0]
    rj, r0 = pj*share(pj, 'j'), p0*share(p0, '0')
    def objective(p):
        q = share(p, h0)
        return float(mass@(p*q+delta*(q*rj+(1-q)*r0)))
    p1, value = scalar_maximum(objective, (-scale, scale))
    _, wide = scalar_maximum(objective, (-2*scale, 2*scale), 321)
    np.testing.assert_allclose(value, wide, atol=1e-7)
    actual = evaluate_policy(g, h0, alpha, beta, delta, p1, pj, p0)
    np.testing.assert_allclose(value, mass@actual['revenue'], rtol=1e-11, atol=1e-12)
    q = share(p1, h0)
    derivative = float(mass@(q+alpha*q*(1-q)*(p1+delta*(rj-r0))))
    return dict(p1=p1, p2j=pj, p20=p0, objective=value,
                reported_revenue=float(mass@actual['revenue']), first_order_residual=derivative)


def run(output, mean=0., sd=1.25, nodes=21, beta=2., alpha=-1., delta=1.,
        exercise='pure theory; no estimated inputs', individual=True):
    if sd <= 0 or nodes < 2 or alpha >= 0 or not 0 <= delta <= 1:
        raise ValueError('Need positive SD, >=2 nodes, negative alpha and delta in [0,1].')
    x, w = np.polynomial.legendre.leggauss(nodes)
    probability = ((np.arange(5)[:, None]+(x+1)/2)/5).ravel()
    gamma = np.array([NormalDist(mean, sd).inv_cdf(float(p)) for p in probability])
    weights = np.tile(w/10, 5)
    labels = np.repeat([f'bin_{k+1}' for k in range(5)], nodes)
    for label in np.unique(labels):
        np.testing.assert_allclose(weights[labels == label].sum(), .2)

    rows = []
    diagnostics = []
    for h0 in ['j', '0']:
        common = solve_distribution(gamma, weights, h0, alpha, beta, delta)
        diagnostics.append(dict(h0=h0, regime='common', group='population', **common))
        for group in dict.fromkeys(labels):
            mask = labels == group
            conditional = weights[mask]/weights[mask].sum()
            informed = solve_distribution(gamma[mask], conditional, h0, alpha, beta, delta)
            diagnostics.append(dict(h0=h0, regime='informed', group=group, **informed))
            for regime, policy in [('informed', informed), ('common', common)]:
                outcomes = evaluate_policy(gamma[mask], h0, alpha, beta, delta,
                                           policy['p1'], policy['p2j'], policy['p20'])
                for k, g in enumerate(gamma[mask]):
                    rows.append(dict(scenario='Pure theory', h0=h0, regime=regime,
                        group=group, gamma=g, weight=weights[mask][k],
                        **{key:float(np.broadcast_to(value, (mask.sum(),))[k])
                           for key, value in outcomes.items()}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output/'type_results.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output/'solver_diagnostics.csv', index=False)
    plot_bins(results, output, individual=individual)
    (output/'settings.json').write_text(json.dumps(dict(
        exercise=exercise, mean_gamma=mean, sd_gamma=sd,
        beta=beta, alpha=alpha, delta=delta, bins=5, nodes_per_bin=nodes,
        bin_probability=.2, information='quintile known versus full-population uniform',
        beliefs='fixed within each observed bin or full population, in both histories',
        learning=False,
        introductory_objective='integral of type-specific purchase probability times continuation revenue',
        terminal_objective='static revenue under unchanged distribution; no history-based updating',
        evaluation='same persistent-type revenue as introductory objective',
        uniform_path='one population-weighted p1 to E[p2] per initial history',
        welfare='persistent gamma; policy-specific histories; zero marginal cost'), indent=2)+'\n')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mean', type=float, default=0.)
    parser.add_argument('--sd', type=float, default=1.25)
    parser.add_argument('--nodes', type=int, default=21)
    args = parser.parse_args()
    run(args.output, args.mean, args.sd, args.nodes)

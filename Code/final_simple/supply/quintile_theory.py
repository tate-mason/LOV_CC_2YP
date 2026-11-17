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

    def solve(g, mass, h0):
        scale = (30+abs(beta)+np.max(abs(g)))/abs(alpha)
        def share(p, h):
            return expit(beta+alpha*p+(-g if h == 'j' else g))
        terminal = [scalar_maximum(lambda p: p*float(mass@share(p, h)), (0., scale))
                    for h in ['j', '0']]
        def objective(p):
            s = float(mass@share(p, h0))
            return s*p+delta*(s*terminal[0][1]+(1-s)*terminal[1][1])
        p1, value = scalar_maximum(objective, (-scale, scale))
        _, wide = scalar_maximum(objective, (-2*scale, 2*scale), 321)
        np.testing.assert_allclose(value, wide, atol=1e-7)
        return p1, terminal[0][0], terminal[1][0]

    rows = []
    for h0 in ['j', '0']:
        common = solve(gamma, weights, h0)
        for group in dict.fromkeys(labels):
            mask = labels == group
            conditional = weights[mask]/weights[mask].sum()
            informed = solve(gamma[mask], conditional, h0)
            for regime, policy in [('informed', informed), ('common', common)]:
                outcomes = evaluate_policy(gamma[mask], h0, alpha, beta, delta, *policy)
                for k, g in enumerate(gamma[mask]):
                    rows.append(dict(scenario='Pure theory', h0=h0, regime=regime,
                        group=group, gamma=g, weight=weights[mask][k],
                        **{key:float(np.broadcast_to(value, (mask.sum(),))[k])
                           for key, value in outcomes.items()}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output/'type_results.csv', index=False)
    plot_bins(results, output, individual=individual)
    (output/'settings.json').write_text(json.dumps(dict(
        exercise=exercise, mean_gamma=mean, sd_gamma=sd,
        beta=beta, alpha=alpha, delta=delta, bins=5, nodes_per_bin=nodes,
        bin_probability=.2, information='quintile known versus full-population uniform',
        beliefs='fixed within each observed bin or full population, in both histories',
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

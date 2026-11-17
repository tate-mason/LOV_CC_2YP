"""Persistent-type introductory pricing with explicitly unchanged terminal beliefs."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'supply'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'demand'))
from quintile_theory import solve_distribution
from binary_welfare import evaluate_policy

def expit(z):
    return 1/(1+np.exp(-z))


class PersistentPricingChecks(unittest.TestCase):
    def test_joint_revenue_optimum_and_unchanged_terminal_beliefs(self):
        g, w = np.array([-3., -.4, 1.5]), np.array([.2, .5, .3])
        alpha, beta, delta = -1.3, .7, .9
        for h0 in ['j', '0']:
            solution = solve_distribution(g, w, h0, alpha, beta, delta)
            # Independent global search of the reported outcome accounting,
            # keeping terminal offers fixed as required by no learning.
            grid = np.linspace(-20., 20., 40001)
            out = evaluate_policy(g[None, :], h0, alpha, beta, delta, grid[:, None],
                                  solution['p2j'], solution['p20'])
            independent = out['revenue']@w
            self.assertGreaterEqual(solution['objective']+1e-12, independent.max())
            self.assertLess(abs(solution['p1']-grid[independent.argmax()]), .001)
            self.assertAlmostEqual(solution['objective'], solution['reported_revenue'], places=12)
            self.assertLess(abs(solution['first_order_residual']), 1e-6)
            for history, key in [('j', 'p2j'), ('0', 'p20')]:
                sign = -1 if history == 'j' else 1
                terminal_grid = np.linspace(0., 20., 20001)
                terminal_revenue = terminal_grid*(expit(
                    beta+alpha*terminal_grid[:, None]+sign*g)@w)
                self.assertLess(abs(solution[key]-terminal_grid[terminal_revenue.argmax()]), .001)
            # The former product-of-means objective has a different derivative.
            p = solution['p1']
            q = expit(beta+alpha*p+(-g if h0 == 'j' else g))
            rj = solution['p2j']*expit(beta+alpha*solution['p2j']-g)
            r0 = solution['p20']*expit(beta+alpha*solution['p20']+g)
            old_derivative = w@q+(w@(alpha*q*(1-q)))*(p+delta*(w@(rj-r0)))
            self.assertGreater(abs(old_derivative), 1e-4)
        inside = solve_distribution(g, w, 'j', alpha, beta, delta)
        outside = solve_distribution(g, w, '0', alpha, beta, delta)
        np.testing.assert_allclose([inside['p2j'], inside['p20']],
                                   [outside['p2j'], outside['p20']], atol=1e-12)

    def test_degenerate_type_and_zero_discount(self):
        for h0 in ['j', '0']:
            for delta in [0., 1.]:
                result = solve_distribution(np.array([.8]), np.ones(1), h0, -1.2, 1., delta)
                duplicate = solve_distribution(np.full(3,.8), np.array([.2,.3,.5]), h0, -1.2, 1., delta)
                np.testing.assert_allclose([result[k] for k in ['p1','p2j','p20','objective']],
                    [duplicate[k] for k in ['p1','p2j','p20','objective']], atol=1e-7)
                if delta == 0:
                    self.assertAlmostEqual(result['p1'], result['p2j' if h0 == 'j' else 'p20'], places=5)


if __name__ == '__main__':
    unittest.main()

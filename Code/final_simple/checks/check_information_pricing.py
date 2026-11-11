"""Mathematical utility arrays only: no household panels or empirical I/O."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.special import expit, logsumexp, gammaln
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand.information_pricing_model import solve_environment, _terminal_common
from supply.pricing_model import static_price


def environment(**changes):
    values = dict(prices=np.array([1.1, 1.4, 1.2]), available=np.ones(3, bool),
        beta=np.array([1.5, 1.8, 1.3]), novelty=np.array([1., 0., 1.]),
        stock=np.array([0., 1., 0.]), alpha=1.2, habit=.3,
        gamma=np.array([-1., 0., 1.]), weights=np.array([.25, .5, .25]), capacity=2)
    values.update(changes)
    return values


class InformationPricingChecks(unittest.TestCase):
    def test_terminal_roundoff_prefers_refined_stationary_price(self):
        # Mathematical utility columns, not household records. This fixture
        # reproduced the server assertion before the tie-breaking fix.
        rng = np.random.default_rng(217)
        for _ in range(8):
            offsets = rng.normal(size=(9, 36))
        intercepts = -20.+10.*offsets
        weights = np.ones(9)/9
        price, revenue = _terminal_common(intercepts, weights, 1., 7, 65)
        q = expit(intercepts-price)
        slope = (weights[:, None]*q*(1-price*(1-q))).sum(axis=0)
        np.testing.assert_allclose(slope, 0., atol=2e-9)
        # Independently check achieved revenue against a dense price search.
        for state in [0, 6, 17]:
            grid = np.linspace(.5, max(2., price[state]*2), 10001)
            quantity = 7*(weights[:, None]*expit(intercepts[:, state, None]-grid)).sum(axis=0)
            self.assertGreaterEqual(revenue[state]+1e-12, np.max(grid*quantity))

    def test_degenerate_type_regimes_coincide(self):
        for g in [-1., 0., 1.]:
            result = solve_environment(**environment(gamma=np.array([g]), weights=np.ones(1)))
            np.testing.assert_allclose(result['full']['p1'], result['limited']['p1'], atol=2e-7)
            np.testing.assert_allclose(result['p2_full'][0], result['p2_limited'], atol=2e-7)
            self.assertAlmostEqual(result['gain'], 0., places=8)

    def test_static_limit(self):
        env = environment(delta=0., habit=0., gamma=np.zeros(1), weights=np.ones(1))
        result = solve_environment(**env)
        rivals = np.r_[env['beta'][[0, 2]]-env['alpha']*env['prices'][[0, 2]], 0.]
        intercept = env['beta'][1]-logsumexp(rivals)
        expected = static_price(intercept, -env['alpha'])
        np.testing.assert_allclose(result['full']['p1'], expected, atol=1e-7)
        np.testing.assert_allclose(result['p2_full'], expected, atol=1e-7)

    def test_fixed_beliefs_history_adaptive_and_dominance(self):
        result = solve_environment(**environment())
        expected = np.broadcast_to(result['weights'][:, None], result['limited_beliefs'].shape)
        np.testing.assert_array_equal(result['limited_beliefs'], expected)
        self.assertGreater(np.ptp(result['p2_limited']), 1e-3)
        self.assertGreaterEqual(result['gain'], -1e-8)
        self.assertGreater(np.ptp(result['full']['p1']), 1e-3)
        self.assertEqual(np.ptp(result['limited']['p1']), 0.)
        # Following history does not imply posterior updating about type.
        self.assertGreater(np.ptp(result['limited']['state_probability'][:, 0]), 1e-3)

    def test_independent_direct_basket_enumeration(self):
        env = environment()
        result = solve_environment(**env)
        states = result['states']; shares = states[:, :3].astype(float)
        totals = shares.sum(axis=1, keepdims=True)
        shares = np.divide(shares, totals, out=np.zeros_like(shares), where=totals > 0)
        history2 = env['stock']+shares
        novelty2 = shares.sum(axis=1, keepdims=True)-shares
        coef = gammaln(env['capacity']+1)-gammaln(states+1).sum(axis=1)
        prices = np.linspace(-2., 7., 3001)
        for k, g in enumerate(env['gamma']):
            terminal = env['beta']+g*novelty2+env['habit']*history2-env['alpha']*env['prices']
            terminal[:, 1] += env['alpha']*env['prices'][1]
            for regime, terminal_price in [('full', result['p2_full'][k]), ('limited', result['p2_limited'])]:
                u2 = terminal.copy();u2[:, 1] -= env['alpha']*terminal_price
                q2 = np.exp(u2[:, 1]-logsumexp(np.column_stack([u2, np.zeros(len(states))]), axis=1))
                continuation = env['capacity']*terminal_price*q2
                u1 = env['beta']+g*env['novelty']+env['habit']*env['stock']-env['alpha']*env['prices']
                u1 = np.tile(np.r_[u1, 0.], (len(prices), 1))
                u1[:, 1] += env['alpha']*(env['prices'][1]-prices)
                logp = u1-logsumexp(u1, axis=1, keepdims=True)
                mass = np.exp(coef+logp @ states.T)
                value = env['capacity']*prices*np.exp(logp[:, 1])+mass @ continuation
                if regime == 'full':
                    self.assertGreaterEqual(result['full']['revenue'][k]+1e-7, value.max())
                chosen = result[regime]['p1'][k]
                u = u1[0].copy();u[1] += env['alpha']*(prices[0]-chosen)
                lp = u-logsumexp(u)
                chosen_mass = np.exp(coef+states @ lp)
                actual = env['capacity']*chosen*np.exp(lp[1])+chosen_mass @ continuation
                self.assertAlmostEqual(actual, result[regime]['revenue'][k], places=8)

    def test_five_categories_and_menu(self):
        result = solve_environment(**environment(prices=np.array([1., 1.4, 1.2, 0., 1.6]),
            beta=np.array([1.5, 1.8, 1.3, 1., .9]), available=np.array([1, 1, 1, 0, 1], bool),
            novelty=np.array([1., 0., 1., 1., 1.]), stock=np.array([0., 1., 0., 0., 0.])))
        self.assertTrue((result['states'][:, 3] == 0).all())
        np.testing.assert_allclose(result['full']['state_probability'].sum(axis=1), 1.)

    def test_focal_flavor_permutation(self):
        env = environment()
        reference = solve_environment(**env)
        order = np.array([0, 2, 1])
        for key in ['prices', 'available', 'beta', 'novelty', 'stock']:
            env[key] = env[key][order]
        permuted = solve_environment(**env, focal=2)
        for regime in ['full', 'limited']:
            for field in ['p1', 'expected_p2', 'revenue']:
                np.testing.assert_allclose(reference[regime][field], permuted[regime][field], atol=1e-8)

    def test_history_specifications_and_grid_refinement(self):
        base = environment()
        cumulative = solve_environment(**base)
        refined = solve_environment(**base, search_points=129)
        np.testing.assert_allclose(cumulative['full']['p1'], refined['full']['p1'], atol=2e-7)
        np.testing.assert_allclose(cumulative['p2_limited'], refined['p2_limited'], atol=2e-7)
        average = solve_environment(**base, stock_measure='average', occasions=2)
        decay = solve_environment(**base, stock_measure='decaying', stock_retention=.9)
        self.assertGreater(np.max(np.abs(average['p2_full']-cumulative['p2_full'])), 1e-3)
        self.assertGreater(np.max(np.abs(decay['p2_full']-cumulative['p2_full'])), 1e-3)

if __name__ == '__main__':
    unittest.main()

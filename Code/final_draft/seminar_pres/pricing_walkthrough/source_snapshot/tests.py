"""Binary mathematical checks only; no household records or empirical I/O."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.special import expit, logsumexp, wrightomega
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand.binary_information_pricing_model import solve_binary
from demand.gamma_calibration import known_prices, uncertainty


def setup(**changes):
    values = dict(prices=np.array([1., 1.4, 1.2]), available=np.ones(3, bool),
                  beta=np.array([1.2, 1.5, 1.]), stock=np.array([.2, .8, .1]),
                  alpha=1.1, habit=.2, gamma=np.array([-1.25, 0., 1.25]),
                  weights=np.ones(3)/3, initial_inside=True)
    values.update(changes)
    return values


class BinaryPricingChecks(unittest.TestCase):
    def test_exact_supply_model_limit(self):
        for state in ['j', '0']:
            env = setup(prices=np.array([1.]), available=np.ones(1, bool),
                        beta=np.array([2.]), stock=np.zeros(1), focal=0,
                        alpha=1., habit=0., initial_inside=(state == 'j'))
            actual = solve_binary(**env)
            for k, g in enumerate(env['gamma']):
                expected = known_prices(g, state, -1., 2., 1.)
                np.testing.assert_allclose([actual['full']['p1'][k], actual['full']['expected_p2'][k],
                                           actual['full']['revenue'][k]],
                                          [expected['expected_p1'], expected['expected_p2'], expected['profit']], atol=2e-7)
            common = uncertainty(env['gamma'], env['weights'], state, -1., 2., 1.)
            np.testing.assert_allclose(actual['limited']['p1'], common['expected_p1'], atol=2e-7)
            np.testing.assert_allclose(actual['p2_limited'], [common['p2j'], common['p20']], atol=2e-7)
            self.assertAlmostEqual(actual['limited_objective'], common['profit'], places=7)

    def test_two_choices_two_successors(self):
        for size in [3, 5]:
            result = solve_binary(**setup(prices=np.ones(size), available=np.ones(size, bool),
                                          beta=np.ones(size), stock=np.zeros(size)))
            np.testing.assert_array_equal(result['states'], [[1, 0], [0, 1]])
            self.assertEqual(result['full']['state_probability'].shape, (3, 2))
            np.testing.assert_allclose(result['full']['state_probability'].sum(axis=1), 1.)
            self.assertEqual(result['p2_limited'].shape, (2,))

    def test_outside_mapping_and_habit(self):
        env = setup()
        result = solve_binary(**env)
        outside = logsumexp(np.r_[env['beta'][[0, 2]]-env['alpha']*env['prices'][[0, 2]]
                                      +env['habit']*env['stock'][[0, 2]], 0.])
        self.assertAlmostEqual(result['outside_index'], outside)
        # h=j gives -gamma, h=0 gives +gamma; own habit rises only after plain.
        expected = result['binary_intercept']+np.column_stack([
            -env['gamma']+env['habit']*(env['stock'][1]+1),
            env['gamma']+env['habit']*env['stock'][1]])
        np.testing.assert_allclose(result['terminal_utility_intercept'], expected)

    def test_type_certain_regimes_agree(self):
        result = solve_binary(**setup(gamma=np.array([-.8]), weights=np.ones(1)))
        self.assertAlmostEqual(result['gain'], 0., places=9)
        np.testing.assert_allclose(result['full']['p1'], result['limited']['p1'], atol=1e-8)
        np.testing.assert_allclose(result['p2_full'][0], result['p2_limited'], atol=1e-8)

    def test_dense_independent_first_price_search_and_evaluation(self):
        for initial in [False, True]:
            env = setup(initial_inside=initial, habit=.7)
            result = solve_binary(**env)
            grid = np.linspace(-5., 10., 20001)
            for k in range(3):
                q = expit(result['first_utility_intercept'][k]-env['alpha']*grid)
                values = grid*q+q*result['terminal_revenue_full'][k, 0]+(1-q)*result['terminal_revenue_full'][k, 1]
                self.assertGreaterEqual(result['full']['revenue'][k]+1e-9, values.max())
                common_p1 = result['limited']['p1'][k]
                s1 = expit(result['first_utility_intercept'][k]-env['alpha']*common_p1)
                revenue = common_p1*s1+s1*result['terminal_revenue_limited'][k, 0]+(1-s1)*result['terminal_revenue_limited'][k, 1]
                self.assertAlmostEqual(revenue, result['limited']['revenue'][k], places=10)
            qbar = env['weights'] @ expit(result['first_utility_intercept'][:, None]-env['alpha']*grid)
            perceived = env['weights'] @ result['terminal_revenue_limited']
            subjective = grid*qbar+qbar*perceived[0]+(1-qbar)*perceived[1]
            self.assertGreaterEqual(result['limited_objective']+1e-9, subjective.max())
            self.assertGreaterEqual(result['gain'], -1e-10)
            np.testing.assert_array_equal(result['limited_beliefs'], np.ones((3, 2))/3)

    def test_no_continuation_and_negative_habit(self):
        for phi in [-.4, .4]:
            result = solve_binary(**setup(delta=0., habit=phi))
            expected = (1+wrightomega(result['first_utility_intercept']-1))/1.1
            np.testing.assert_allclose(result['full']['p1'], expected, atol=1e-9)
            self.assertGreaterEqual(result['gain'], -1e-10)

    def test_focal_permutation_and_unavailable_rival(self):
        env = setup(); reference = solve_binary(**env)
        order = [0, 2, 1]
        for key in ['prices', 'available', 'beta', 'stock']:
            env[key] = env[key][order]
        permuted = solve_binary(**env, focal=2)
        np.testing.assert_allclose(reference['full']['revenue'], permuted['full']['revenue'], atol=1e-9)
        one = solve_binary(**setup(available=np.array([False, True, False])))
        self.assertAlmostEqual(one['outside_index'], 0.)

if __name__ == '__main__':
    unittest.main()

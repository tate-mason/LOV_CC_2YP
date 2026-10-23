"""Numerical checks for the demand derivatives and committed pricing policy."""
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
import toy_model as tm


class ToyModelChecks(unittest.TestCase):
    def test_fixed_menu_regions_exclude_dominated_crossing(self):
        # H and L cross at 1/2, where the uniform menu dominates both.
        values = np.array([[3., 0.], [0., 3.], [2., 2.]])
        regions = tm.fixed_menu_regions(values)
        np.testing.assert_allclose(
            regions, [(0, 1/3, 1), (1/3, 2/3, 2), (2/3, 1, 0)]
        )

    def test_calibrated_fixed_menu_transitions(self):
        types = [tm.gamma_H, tm.gamma_L]
        policies = [tm.solve_common([g], [1.]) for g in types]
        policies.append(tm.solve_common(types, [.5, .5]))
        values = np.array([tm.evaluate_policy(p.p1, p.p2, types) for p in policies])
        regions = tm.fixed_menu_regions(values)
        self.assertEqual([r[2] for r in regions], [1, 2, 0])
        for (_, boundary, a), (_, _, b) in zip(regions[:-1], regions[1:]):
            weights = np.array([boundary, 1-boundary])
            self.assertAlmostEqual(values[a] @ weights, values[b] @ weights, places=10)
            for q, expected in [(boundary-1e-5, a), (boundary+1e-5, b)]:
                self.assertEqual(int(np.argmax(values @ [q, 1-q])), expected)

    def test_first_period_share_derivative(self):
        prices = np.array([1.7, 2.3])
        for gamma in (0.1, 2.0):
            _, analytic = tm.shares_jac(prices, gamma)
            numeric = approx_derivative(lambda p: tm.shares_jac(p, gamma)[0], prices)
            np.testing.assert_allclose(analytic, numeric, atol=1e-9)

    def test_targeting_dominance_and_endpoints(self):
        high = tm.solve_common([tm.gamma_H], [1.0])
        low = tm.solve_common([tm.gamma_L], [1.0])
        common = tm.solve_common([tm.gamma_H, tm.gamma_L], [0.5, 0.5])
        self.assertGreaterEqual((high.profit + low.profit) / 2, common.profit - 1e-8)
        for weight, own in [(0.0, low), (1.0, high)]:
            endpoint = tm.solve_common([tm.gamma_H, tm.gamma_L], [weight, 1-weight])
            self.assertAlmostEqual(endpoint.profit, own.profit, places=7)
        np.testing.assert_allclose(common.type_profits,
            tm.evaluate_policy(common.p1, common.p2, [tm.gamma_H, tm.gamma_L]), atol=1e-10)

    def test_terminal_equal_markup_and_outside_share(self):
        policy = tm.solve_common([tm.gamma_H], [1.0])
        for j, prices in enumerate(policy.p2):
            margins = prices - tm.costs
            _, outside = tm.logit_probs(tm.val_state(prices, j, tm.gamma_H))
            np.testing.assert_allclose(margins, -1 / (tm.alpha * outside), atol=2e-6)

    def test_committed_menu_stationarity(self):
        types = [tm.gamma_H, tm.gamma_L]
        policy = tm.solve_common(types, [0.5, 0.5])
        flat = np.r_[policy.p1, policy.p2.ravel()]
        grad = approx_derivative(lambda x: np.mean(tm.evaluate_policy(
            x[:tm.J], x[tm.J:].reshape(tm.J, tm.J), types)), flat).ravel()
        np.testing.assert_allclose(grad, 0, atol=2e-6)

    def test_no_prior_purchase_or_first_period_type_signal(self):
        prices = np.array([1.7, 2.3])
        expected, _ = tm.logit_probs(tm.beta_0 + tm.alpha * prices)
        for gamma in (tm.gamma_H, tm.gamma_L):
            shares, _ = tm.shares_jac(prices, gamma)
            np.testing.assert_allclose(shares, expected)
        self.assertFalse(hasattr(tm, "init_state_dist"))
        # Variety preferences still affect demand after the first purchase.
        self.assertFalse(np.allclose(tm.shares_jac(prices, tm.gamma_H, 0)[0],
                                     tm.shares_jac(prices, tm.gamma_L, 0)[0]))

    def test_newborn_fixed_discount_cutoff(self):
        types = [tm.gamma_H, tm.gamma_L]
        regular = tm.solve_newborn([tm.gamma_H], [1.0])
        offer = tm.solve_newborn([tm.gamma_L], [1.0])
        offer.p1 = regular.p1.copy()
        gains = (tm.evaluate_newborn(offer.p1, offer.p2, types)
                 - tm.evaluate_newborn(regular.p1, regular.p2, types))
        self.assertLess(gains[0], 0)
        self.assertGreater(gains[1], 0)
        cutoff = -gains[0] / (gains[1] - gains[0])
        self.assertTrue(0 < cutoff < 1)
        self.assertAlmostEqual(np.array([1-cutoff, cutoff]) @ gains, 0, places=12)
        for q in (0.01, 0.5, 0.99):
            common = tm.solve_newborn(types, [1-q, q])
            for menu in (regular, offer):
                value = np.array([1-q, q]) @ tm.evaluate_newborn(menu.p1, menu.p2, types)
                self.assertGreaterEqual(common.profit + 1e-8, value)


if __name__ == '__main__':
    unittest.main()

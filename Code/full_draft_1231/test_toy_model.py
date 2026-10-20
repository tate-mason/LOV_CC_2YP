"""Numerical checks for the demand derivatives and backward-induction policy."""
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
import toy_model as tm


class ToyModelChecks(unittest.TestCase):
    def test_pooled_share_derivative(self):
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

    def test_posterior_and_envelope_gradient(self):
        policy = tm.solve_common([tm.gamma_H, tm.gamma_L], [0.5, 0.5])
        s = np.array([tm.shares_jac(policy.p1, g)[0] for g in (tm.gamma_H, tm.gamma_L)])
        np.testing.assert_allclose(policy.posterior[0], s[0] / s.sum(axis=0))
        # Holding the terminal policy fixed gives the same first-order gradient
        # as the optimized continuation value (envelope theorem).
        grad = approx_derivative(lambda p: np.mean(tm.evaluate_policy(
            p, policy.p2, [tm.gamma_H, tm.gamma_L])), policy.p1).ravel()
        np.testing.assert_allclose(grad, 0, atol=2e-6)

    def test_newborn_has_no_initial_type_signal(self):
        policy = tm.solve_newborn([tm.gamma_H, tm.gamma_L], [0.5, 0.5])
        np.testing.assert_allclose(policy.posterior, 0.5)
        np.testing.assert_allclose(policy.type_profits,
            tm.evaluate_newborn(policy.p1, policy.p2, [tm.gamma_H, tm.gamma_L]))

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

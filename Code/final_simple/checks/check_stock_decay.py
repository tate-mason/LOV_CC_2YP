"""Pure mathematical checks; no household panels or empirical I/O."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand.history import DEFAULT_STOCK_RETENTION, advance_stock


class StockDecayChecks(unittest.TestCase):
    def test_calendar_gaps_match_discounted_sum(self):
        # Three abstract impulses, observed at mathematical times 0, 1, 4.
        a = np.array([.25, .5, .25])
        b = np.array([0., 1., 0.])
        stock1 = advance_stock(np.zeros(3), a)
        stock4 = advance_stock(stock1, b, weeks=3)
        np.testing.assert_allclose(stock1, .9*a)
        np.testing.assert_allclose(stock4, .9**4*a + .9**3*b)
        # Splitting a gap into empty intermediate transitions is equivalent.
        intermediate = advance_stock(stock1, b)
        np.testing.assert_allclose(advance_stock(intermediate, np.zeros(3), 2), stock4)

    def test_bounded_persistent_stock(self):
        rho = DEFAULT_STOCK_RETENTION
        for size in [3, 5]:
            share = np.eye(size)[1]
            stock = np.zeros(size)
            for n in range(1, 1001):
                stock = advance_stock(stock, share)
                expected = share*rho*(1-rho**n)/(1-rho)
                np.testing.assert_allclose(stock, expected, atol=1e-12)
                self.assertLessEqual(float(stock.sum()), 9.+1e-12)
            np.testing.assert_allclose(stock, share*9., atol=1e-12)

    def test_zero_and_legacy_retention(self):
        stock, share = np.array([2., 3., 4.]), np.array([.5, .5, 0.])
        np.testing.assert_array_equal(advance_stock(stock, share, 20, 1.), stock+share)
        np.testing.assert_array_equal(advance_stock(stock, share, retention=0.), np.zeros(3))
        np.testing.assert_allclose(advance_stock(stock, np.zeros(3)), .9*stock)

    def test_basket_broadcasting_and_pricing_transition(self):
        stock = np.array([1., 2., 0.])
        baskets = np.array([[0., 0., 0.], [0., 1., 0.], [.5, 0., .5]])
        terminal = advance_stock(stock, baskets)
        np.testing.assert_allclose(terminal, [[.9, 1.8, 0.], [.9, 2.7, 0.], [1.35, 1.8, .45]])
        # Observation zero establishes the first priced state, then h1 establishes h2.
        initial = advance_stock(np.zeros(3), baskets[1])
        np.testing.assert_allclose(advance_stock(initial, baskets[2]),
                                   .9**2*baskets[1]+.9*baskets[2])

    def test_invalid_parameters_rejected(self):
        for rho in [-.1, 1.1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                advance_stock(0., 0., retention=rho)
        for weeks in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                advance_stock(0., 0., weeks=weeks)


if __name__ == '__main__':
    unittest.main()

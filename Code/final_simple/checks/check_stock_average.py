"""Mathematical stock-vector checks only; no empirical records or I/O."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand.history import update_history, saved_stock_measure

class AverageStockChecks(unittest.TestCase):
    def test_arithmetic_mean_and_calendar_gaps(self):
        for dimension in [3, 5]:
            impulses = np.eye(dimension)
            mean = np.zeros(dimension)
            for n in range(dimension):
                mean = update_history(mean, impulses[n], n, weeks=10+n)
                np.testing.assert_allclose(mean, impulses[:n+1].sum(axis=0)/(n+1))
                self.assertTrue(((mean >= 0) & (mean <= 1)).all())

    def test_repetition_and_empty_history(self):
        share = np.array([0., 1., 0.])
        np.testing.assert_array_equal(update_history(np.zeros(3), np.zeros(3), 0), np.zeros(3))
        np.testing.assert_array_equal(update_history(share, share, 1000), share)
        np.testing.assert_array_equal(update_history(share, np.zeros(3), 10), share)

    def test_basket_transition(self):
        initial = np.array([0., 1., 0.])
        baskets = np.array([[0., 0., 0.], [1., 0., 0.], [.5, 0., .5]])
        np.testing.assert_allclose(update_history(initial, baskets, 1),
            [[0., 1., 0.], [.5, .5, 0.], [.25, .5, .25]])

    def test_saved_fit_compatibility(self):
        self.assertEqual(saved_stock_measure({}), 'cumulative')
        self.assertEqual(saved_stock_measure({'stock_retention': .9}), 'decaying')
        self.assertEqual(saved_stock_measure({'stock_measure': 'average', 'stock_retention': .9}), 'average')
        np.testing.assert_allclose(update_history([1., 0.], [0., 1.], 1, measure='cumulative'), [1., 1.])
        np.testing.assert_allclose(update_history([1., 0.], [0., 1.], 1, measure='decaying'), [.9, .9])

if __name__ == '__main__':
    unittest.main()

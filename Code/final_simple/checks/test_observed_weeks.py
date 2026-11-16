"""Calendar/count/history and exact parallel-score checks using in-memory toy arrays only."""
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import polars as pl

from demand.observed_weeks import include_observed_weeks, fill_zero_week_prices, flag_yogurt_lines
from demand import estimation as est
from demand.parallel_score import partition_households, parallel_objective


class ObservedWeekChecks(unittest.TestCase):
    def test_retail_yogurt_upcs_supplement_hms_modules_without_duplicating_lines(self):
        source = pl.DataFrame(dict(upc=['0001', '2', '3', '4'],
                                  product_module_code=[9999, 3612, 9999, None])).lazy()
        known = pl.DataFrame({'upc': ['1', '1', '4']}).lazy()
        actual = flag_yogurt_lines(source, known).collect()
        self.assertEqual(actual.height, 4)
        self.assertEqual(actual['_is_yogurt'].to_list(), [True, True, False, True])

    def calendar(self):
        return pd.DataFrame(dict(
            household_code=[10]*5,
            week=pd.date_range('2010-01-02', periods=5, freq='7D'),
            store_code_uc=[100]*5, dma_cd=[1]*5,
            raw_yogurt=[2., 0., 1., 3., 0.], invalid_yogurt=[False]*5))

    def purchases(self):
        return pd.DataFrame(dict(
            household_code=[10, 10], week=pd.to_datetime(['2010-01-02', '2010-01-23']),
            store_code_uc=[100, 100], dma_cd=[1, 1],
            other=[2., 0.], berry=[0., 2.], plain=[0., 0.]))

    def test_zero_weeks_are_seven_outside_unmatched_weeks_are_excluded(self):
        frame, audit = include_observed_weeks(self.purchases(), self.calendar(), est.FLAVORS)
        self.assertEqual(audit['unmatched_or_invalid_yogurt_weeks'], 2)
        self.assertEqual(audit['zero_purchase_weeks'], 2)
        self.assertEqual(frame['week'].tolist(), list(pd.to_datetime(['2010-01-02', '2010-01-09', '2010-01-30'])))
        frame['outside'] = np.maximum(7 - frame[est.FLAVORS].sum(axis=1), 0)
        np.testing.assert_array_equal(frame.outside, [5, 7, 7])
        frame = est.add_stocks(frame)
        self.assertTrue(np.isfinite(frame.filter(regex='stock_|novelty_|xi_')).all().all())
        np.testing.assert_allclose(frame.filter(like='stock_').drop(columns='stock_occasions'),
                                   [[0, 0, 0], [1, 0, 0], [1, 0, 0]])

    def test_zero_week_is_in_likelihood_and_history_has_no_nan(self):
        calendar = self.calendar().iloc[:3].copy()
        calendar['raw_yogurt'] = [2., 0., 2.]
        purchases = self.purchases().copy()
        purchases.loc[1, 'week'] = calendar.iloc[2]['week']
        frame, _ = include_observed_weeks(purchases, calendar, est.FLAVORS)
        for f in est.FLAVORS:
            frame['price_' + f] = 1.
        frame['outside'] = np.maximum(7-frame[est.FLAVORS].sum(axis=1), 0)
        frame = est.add_stocks(frame)
        np.testing.assert_array_equal(frame.filter(like='novelty_'), [[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        data = est.make_arrays(frame, 7)
        names = est.parameter_names('heterogeneous_history', True)
        theta = np.zeros(len(names))
        draws = est.make_draws(4, 306, 'heterogeneous_history')
        value = est.household_loglike(theta, data, draws, 'heterogeneous_history')[0]
        from scipy.special import logsumexp
        denominator = logsumexp([-1., -1., -1., 0.])
        self.assertAlmostEqual(value, -4. - 21.*denominator)
        value, gradient = est.demand_objective_gradient(theta, data, draws, 'heterogeneous_history')
        for k in range(len(theta)):
            plus, minus = theta.copy(), theta.copy()
            plus[k] += 1e-5
            minus[k] -= 1e-5
            finite = (est.demand_objective_gradient(plus, data, draws, 'heterogeneous_history')[0]
                      - est.demand_objective_gradient(minus, data, draws, 'heterogeneous_history')[0])/2e-5
            self.assertAlmostEqual(gradient[k], finite, places=6)

    def test_fallback_only_changes_zero_weeks(self):
        weeks = pd.to_datetime(['2010-01-02', '2010-01-09'])
        frame = pd.DataFrame(dict(week=weeks, dma_cd=[1, 1], zero_purchase_week=[True, False],
                                  price_other=[np.nan, np.nan], price_berry=[2., 2.], price_plain=[np.nan, np.nan]))
        market = pd.DataFrame(dict(week=weeks, dma_cd=[1, 1], other=[3., 3.], berry=[4., 4.], plain=[5., 5.]))
        actual = fill_zero_week_prices(frame, market, est.FLAVORS)
        np.testing.assert_allclose(actual.iloc[0][['price_other', 'price_berry', 'price_plain']].to_numpy(float), [3, 2, 5])
        self.assertTrue(np.isnan(actual.iloc[1]['price_other']))

    def test_parallel_objective_preserves_whole_household_integration(self):
        frame = pd.DataFrame(dict(household_code=[10, 10, 20, 20, 20],
            week=pd.to_datetime(['2010-01-02', '2010-01-09', '2010-01-02', '2010-01-09', '2010-01-16']),
            other=[2., 0., 1., 0., 3.], berry=[0., 0., 1., 2., 0.], plain=[0.]*5,
            outside=[5., 7., 5., 5., 4.]))
        for f in est.FLAVORS:
            frame['price_' + f] = np.linspace(1., 2., len(frame))
        data = est.make_arrays(est.add_stocks(frame), 7)
        names = est.parameter_names('heterogeneous_history', True)
        theta = np.array([.3 if n.startswith('sd_') else .1 for n in names])
        draws = est.make_draws(8, 306, 'heterogeneous_history')
        expected = est.demand_objective_gradient(theta, data, draws, 'heterogeneous_history')
        blocks = partition_households(data, 2)
        self.assertEqual([len(b['hh']) for b in blocks], [2, 3])
        with parallel_objective(est, data, 2):
            actual = est.demand_objective_gradient(theta, data, draws, 'heterogeneous_history')
        self.assertAlmostEqual(actual[0], expected[0], places=12)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)

    @unittest.skipIf(sys.platform == 'linux' and Path('/scratch/dtm63837/Kilts_Panel').is_dir(),
                     'Local-machine rejection check; this is the research server.')
    def test_server_guard_still_rejects_local_empirical_read(self):
        from demand.observed_weeks import load_observed_weeks
        with self.assertRaises(PermissionError):
            load_observed_weeks('/not/read.parquet', [10], SimpleNamespace())


if __name__ == '__main__':
    unittest.main()

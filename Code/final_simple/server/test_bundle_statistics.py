"""In-memory arithmetic fixtures only; no research files or empirical I/O."""
from datetime import date, timedelta
import unittest

import polars as pl

from bundle_statistics import aggregate_plans


class BundleStatisticsTests(unittest.TestCase):
    def fixture(self):
        # Two artificial sequences: mixed baskets, a calendar gap, and no changes.
        entries = [('a', 0, ['x', 'y']), ('a', 1, ['x', 'z']),
                   ('a', 3, ['x', 'z']), ('a', 4, ['y', 'z']),
                   ('b', 0, ['x', 'y']), ('b', 1, ['x', 'y'])]
        rows = [(hh, date(2020, 1, 4)+timedelta(weeks=w), flavor, 1.)
                for hh, w, flavors in entries for flavor in flavors]
        return pl.DataFrame(rows, schema=['household_code', 'week', 'flavor', 'quantity'],
                            orient='row').lazy()

    def test_mixed_baskets_and_calendar_gaps(self):
        results = {k: v.collect() for k, v in aggregate_plans(self.fixture()).items()}
        r = results['overall'].row(0, named=True)
        self.assertEqual(r['mixed_pct'], 100.)
        self.assertEqual(r['purchase_pairs'], 4)
        self.assertEqual(r['calendar_pairs'], 3)
        self.assertEqual(r['purchase_change_pct'], 50.)
        self.assertAlmostEqual(r['calendar_change_pct'], 200/3)
        self.assertAlmostEqual(r['purchase_turnover_pct'], 100/3)
        self.assertEqual(r['purchase_added'], .5)
        self.assertEqual(r['purchase_dropped'], .5)
        self.assertEqual(results['households']['mean_purchase_changes'].item(), 1.)
        self.assertEqual(results['purchase_spells']['spells'].item(), 6)
        self.assertEqual(results['purchase_spells']['mean_length'].item(), 2.)
        self.assertEqual(results['calendar_spells']['spells'].item(), 8)
        self.assertEqual(results['calendar_spells']['mean_length'].item(), 1.5)
        # Only aggregates, no household/week-level results can reach the writer.
        for frame in results.values():
            self.assertFalse({'household_code', 'week', 'purchase_index'} & set(frame.columns))

    def test_no_eligible_pairs(self):
        weekly = self.fixture().filter(pl.col('week') == date(2020, 1, 4))
        plans = aggregate_plans(weekly)
        r = plans['overall'].collect().row(0, named=True)
        self.assertEqual(r['purchase_pairs'], 0)
        self.assertIsNone(r['purchase_change_pct'])
        self.assertEqual(plans['households'].collect()['mean_purchase_changes'].item(), 0.)
        self.assertEqual(plans['purchase_spells'].collect()['mean_length'].item(), 1.)

    def test_package_changes_alone_do_not_change_sets(self):
        weekly = self.fixture().filter(pl.col('household_code') == 'b').with_columns(
            pl.when(pl.col('week') > date(2020, 1, 4)).then(7.).otherwise(1.).alias('quantity'))
        r = aggregate_plans(weekly)['overall'].collect().row(0, named=True)
        self.assertEqual(r['purchase_change_pct'], 0.)
        self.assertEqual(r['purchase_turnover_pct'], 0.)


if __name__ == '__main__':
    unittest.main()

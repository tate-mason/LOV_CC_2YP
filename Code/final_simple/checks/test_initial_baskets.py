"""Synthetic, in-memory fixtures only. No research records or empirical I/O."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'demand'))
from initial_states import first_observed_baskets


class FirstBasketChecks(unittest.TestCase):
    def test_skip_zero_weeks_and_keep_first_basket_despite_calendar_gap(self):
        # Entirely invented identifiers, dates and counts; no files are read/written.
        weekly = pd.DataFrame(dict(household_code=['synthetic_A']*3,
            week=pd.to_datetime(['2000-01-01', '2000-01-15', '2000-02-12']),
            plain=[0, 1, 0], berry=[0, 2, 1]))
        calendar = weekly[['household_code', 'week']].copy()
        calendar['raw_yogurt'] = weekly[['plain', 'berry']].sum(axis=1)
        result = first_observed_baskets(weekly, ['plain', 'berry'], calendar)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.week.iloc[0], pd.Timestamp('2000-01-15'))
        self.assertEqual(result.plain.iloc[0], 1)
        self.assertEqual(result.berry.iloc[0], 2)
        self.assertTrue(result.state_observed.iloc[0])

    def test_unknown_first_purchase_is_not_replaced_by_later_basket(self):
        weekly = pd.DataFrame(dict(household_code=['synthetic_B'],
            week=pd.to_datetime(['2000-02-12']), plain=[1], berry=[0]))
        calendar = pd.DataFrame(dict(household_code=['synthetic_B']*3,
            week=pd.to_datetime(['2000-01-01', '2000-01-15', '2000-02-12']),
            raw_yogurt=[0, 2, 1]))
        result = first_observed_baskets(weekly, ['plain', 'berry'], calendar)
        self.assertEqual(result.week.iloc[0], pd.Timestamp('2000-01-15'))
        self.assertFalse(result.state_observed.iloc[0])
        self.assertTrue(pd.isna(result.plain.iloc[0]))


if __name__ == '__main__':
    unittest.main()

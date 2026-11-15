"""Synthetic in-memory tests only; never read empirical files."""
from datetime import date, timedelta
import unittest

import polars as pl

from flavor_families import classify
from flavor_family_statistics import recode_weekly, summary_plans, headline


class FlavorFamiliesTests(unittest.TestCase):
    def test_aliases_and_marketing_variants(self):
        for label in ['st', 'strb', 'strawberry', 'st mist', 'strawberry sunrise']:
            self.assertEqual(classify(label)[0], 'strawberry')
        self.assertEqual(classify('bb patch')[0], 'blueberry')
        self.assertEqual(classify('french vanilla')[0], 'vanilla')

    def test_real_blends_and_desserts_remain_distinct(self):
        self.assertEqual(classify('st & bn')[0], classify('banana strawberry')[0])
        self.assertNotEqual(classify('st & bn')[0], classify('st')[0])
        self.assertNotEqual(classify('st orange sunris')[0], classify('st')[0])
        self.assertNotEqual(classify('st cheesecake')[0], classify('st')[0])
        self.assertNotEqual(classify('bb')[0], classify('st')[0])
        self.assertNotEqual(classify('black cherry')[0], classify('cherry')[0])

    def test_unknowns_stay_separate_and_are_not_guessed(self):
        self.assertEqual(classify('h-p'), ('unresolved: h-p', 'retain_exact_label'))
        self.assertEqual(classify('st xyz'), ('unresolved: st xyz', 'retain_exact_label'))
        self.assertNotEqual(classify('unknown a')[0], classify('unknown b')[0])

    def test_recode_combines_aliases_and_extends_presence(self):
        entries = [(0, 'st', 1.), (0, 'strawberry', 2.),
                   (1, 'st mist', 3.), (3, 'strawberry sunrise', 1.),
                   (4, 'st & bn', 2.)]
        rows = [('synthetic', date(2020, 1, 4) + timedelta(weeks=week), flavor, quantity)
                for week, flavor, quantity in entries]
        weekly = pl.DataFrame(rows, schema=['household_code', 'week', 'flavor', 'quantity'], orient='row').lazy()
        mapping = {label: classify(label)[0] for _, label, _ in entries}
        recoded = recode_weekly(weekly, mapping)
        self.assertEqual(recoded.select(pl.col('quantity').sum()).collect().item(), 9.)
        self.assertEqual(recoded.collect().height, 4)
        results = {name: plan.collect() for name, plan in summary_plans(recoded).items()}
        h = headline(results)
        self.assertEqual(h['purchase_weeks'], 4)
        self.assertEqual(h['mean_distinct_flavors'], 2)
        self.assertAlmostEqual(h['mixed_purchase_weeks_pct'], 0.)
        self.assertAlmostEqual(h['purchase_change_pct'], 100 / 3)
        self.assertEqual(h['purchase_mean_spell_length'], 2.)  # Strawberry 3, blend 1.
        self.assertAlmostEqual(h['calendar_mean_spell_length'], 4 / 3)
        self.assertAlmostEqual(h['mean_household_hhi'], (7 / 9) ** 2 + (2 / 9) ** 2)
        for frame in results.values():
            self.assertFalse({'household_code', 'week', 'purchase_index'} & set(frame.columns))


if __name__ == '__main__':
    unittest.main()

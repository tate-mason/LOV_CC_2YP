"""Checks for the sibling summary script, using synthetic purchases only."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import polars as pl


class SummaryChecks(unittest.TestCase):
    def test_demographics_and_mixed_baskets(self):
        # The override lets a prepared replacement be tested before installation.
        script = Path(os.environ.get("SUMMARY_SCRIPT", Path(__file__).parent / "../data_code/summ_stats.py"))
        spec = importlib.util.spec_from_file_location("summary", script)
        summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = pl.DataFrame({
                "household_code": [1, 1, 1, 1, 2], "panel_year": [2010]*5,
                "purchase_date": ["2010-01-02", "2010-01-02", "2010-01-09", "2010-01-23", "2010-01-03"],
                "trip_code_uc": [1, 1, 2, 3, 4], "upc": [1, 2, 2, 3, 1], "quantity": [2, 1, 1, 3, 1],
                "race": [1, 1, 1, 1, 2], "hispanic_origin": [2, 2, 2, 2, 1],
                "household_income": [5, 5, 5, 5, 8], "female_head_age": [3, 3, 3, 3, 0],
                "male_head_age": [4, 4, 4, 4, 2], "household_size": [2]*5})
            source.write_parquet(root / "sample.parquet")
            scan = summary.scan_purchases(root / "sample.parquet", 2006, 2015)
            households = summary.household_summary(scan).set_index("household_code")
            self.assertEqual(len(households), 2)
            self.assertEqual(households.loc[1, "trips"], 3)
            self.assertEqual(households.loc[2, "race"], "Hispanic")
            self.assertEqual(households.loc[2, "race_original"], "2")
            summary.demographic_tables(households.reset_index(), root, "test")
            race = pd.read_csv(root / "test_race.csv")
            np.testing.assert_allclose(race["percent_all_households"], 50.)
            reference = pl.DataFrame({"upc": ["1", "2", "3"], "flavor": ["other", "berry", "plain"]})
            weekly, _ = summary.yogurt_weeks(scan, reference)
            self.assertEqual(weekly.loc[weekly.household_code == 2, "week"].iloc[0], pd.Timestamp("2010-01-09"))
            shares, transitions = summary.flavor_summaries(weekly, root)
            self.assertEqual(weekly.loc[0, "flavors_in_basket"], 2)
            self.assertTrue(transitions.isna().all().all())
            self.assertAlmostEqual(shares.percent_packages.sum(), 100.)
            self.assertTrue((root / "test_income.tex").exists())


if __name__ == "__main__":
    unittest.main()

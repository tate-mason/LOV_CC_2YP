"""Synthetic checks for sparse summaries and exact results across batch sizes."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd
import polars as pl

import summ_stats as summary


class MemorySummaryChecks(unittest.TestCase):
    def test_batches_preserve_global_statistics(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = pl.DataFrame({
                "household_code": [1, 1, 1, 1, 2, 2, 3], "panel_year": [2010] * 7,
                "purchase_date": ["2010-01-02", "2010-01-09", "2010-01-09", "2010-01-23", "2010-01-02", "2010-01-09", "2010-01-02"],
                "trip_code_uc": [1, 2, 2, 3, 4, 5, 6], "upc": [1, 1, 2, 3, 2, 3, 1],
                "quantity": [2, 1, 1, 3, 1, 2, 1], "race": [1, 1, 1, 1, 2, 2, 3],
                "hispanic_origin": [2, 2, 2, 2, 1, 1, 2], "household_income": [5, 5, 5, 5, 8, 8, 9],
                "female_head_age": [3, 3, 3, 3, 0, 0, 4], "male_head_age": [4, 4, 4, 4, 2, 2, 5],
                "household_size": [2] * 7, "store_code_uc": [100] * 7,
                "original_metadata": ["preserved"] * 7,
            })
            for name in ["merged_panel", "yogurt_estimation_sample"]:
                source.write_parquet(root / f"{name}.parquet")
            reference = root / "reference.csv"
            reference.write_text("upc,flavor_descr,module\n001,vanilla,3612\n002,strawberry,3612\n003,plain,3612\n004,unused,3612\n")
            for batches in [1, 3]:
                out, work = root / f"out{batches}", root / f"work{batches}"
                out.mkdir()
                work.mkdir()
                args = SimpleNamespace(data_dir=root, flavors=reference, products_extra=root / "absent",
                                       first_year=2006, last_year=2015, household_batches=batches, no_figures=True)
                summary.run(args, out, work)
            for name, keys in [
                ("full_panel_households", ["household_code"]), ("matched_yogurt_households", ["household_code"]),
                ("yogurt_household_flavors", ["household_code"]), ("yogurt_dynamics", ["metric"]),
                ("full_panel_overview", ["metric"]), ("matched_yogurt_overview", ["metric"]),
                ("yogurt_flavor_shares", ["flavor"]), ("full_panel_year_coverage", ["panel_year"]),
            ]:
                left = pd.read_csv(root / "out1" / f"{name}.csv").sort_values(keys).reset_index(drop=True)
                right = pd.read_csv(root / "out3" / f"{name}.csv").sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(left, right)
            prepared = pl.read_parquet(root / "out3/yogurt_estimation_sample.parquet")
            self.assertEqual(prepared.height, 6)
            self.assertTrue(set(source.columns).issubset(prepared.columns))
            self.assertEqual(set(prepared["household_code"].to_list()), {1, 2})
            self.assertEqual(prepared["original_metadata"].unique().to_list(), ["preserved"])
            self.assertEqual(set(prepared["upc"].to_list()), {"1", "2", "3"})
            self.assertEqual(prepared["flavor"].null_count(), 0)
            hh = pd.read_csv(root / "out3/yogurt_household_flavors.csv").set_index("household_code")
            self.assertEqual(set(hh.index), {1, 2})
            self.assertAlmostEqual(hh.loc[1, "flavor_HHI"], 19 / 49)
            self.assertAlmostEqual(hh.loc[2, "flavor_HHI"], 5 / 9)
            metrics = pd.read_csv(root / "out3/yogurt_dynamics.csv").set_index("metric")["value"]
            self.assertEqual(metrics["Eligible adjacent single-flavor week pairs"], 1)
            self.assertEqual(metrics["Switch rate in eligible pairs (%)"], 100)
            self.assertEqual(metrics["Median gap between purchase weeks"], 1)
            # Sparse quantity output has one row per observed flavor/week, not
            # one column for every flavor in the full reference.
            quantities = pl.read_csv(root / "out3/yogurt_weekly_flavor_quantities.csv")
            self.assertEqual(quantities.height, 6)
            self.assertEqual(set(quantities.columns), {"household_code", "week", "flavor", "quantity"})

    def test_recovery_preserves_existing_labels_and_uses_earliest_year(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            ref, extra, output = root / "ref.csv", root / "extra.parquet", root / "result.csv"
            ref.write_text("upc,flavor_descr\n001,vanilla\n002,NA\n")
            pl.DataFrame({"upc": [1, 2, 3, 3, 4], "flavor_descr": ["plain", "berry", "cherry", "strawberry", None],
                          "panel_year": [2006, 2006, 2007, 2006, 2006]}).write_parquet(extra)
            summary.recover_flavor_reference(ref, extra, output)
            result = summary.flavor_reference(output, detailed=True).sort("upc")
            self.assertEqual(result.to_dict(as_series=False),
                             {"upc": ["1", "2", "3"], "flavor": ["vanilla", "unknown", "strawberry"]})


if __name__ == "__main__":
    unittest.main()

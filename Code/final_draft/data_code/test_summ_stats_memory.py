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
                "household_size": [2] * 7, "store_code_uc": [100] * 7, "price": [1.] * 7,
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
            # Price filtering precedes the two-week household rule.
            unpriced = source.with_columns(pl.when(
                ((pl.col("household_code") == 2) & (pl.col("purchase_date") == "2010-01-09"))
                | ((pl.col("household_code") == 1) & (pl.col("purchase_date") == "2010-01-09") & (pl.col("upc") == 1))
                ).then(None).otherwise(pl.col("price")).alias("price"))
            unpriced.write_parquet(root / "yogurt_estimation_sample.parquet")
            out, work = root / "filtered", root / "filtered_work"
            out.mkdir()
            work.mkdir()
            args.household_batches = 2
            summary.run(args, out, work)
            filtered = pl.read_parquet(out / "yogurt_estimation_sample.parquet")
            self.assertEqual(filtered.height, 3)
            self.assertEqual(filtered["household_code"].unique().to_list(), [1])
            self.assertTrue(filtered["price"].is_not_null().all())
            self.assertEqual(pl.read_csv(out / "yogurt_price_coverage.csv")["purchase_lines_without_price_coverage"][0], 2)
            # Sparse quantity output has one row per observed flavor/week, not
            # one column for every flavor in the full reference.
            quantities = pl.read_csv(root / "out3/yogurt_weekly_flavor_quantities.csv")
            self.assertEqual(quantities.height, 6)
            self.assertEqual(set(quantities.columns), {"household_code", "week", "flavor", "quantity"})

    def test_demographics_match_previous_fieldwise_sort(self):
        # Unsorted histories, missing fields, conflicting same-date values and
        # repeated purchase lines must preserve the earliest nonmissing values.
        source = pl.DataFrame({
            "household_code": [1, 2, 1, 1, 1, 2],
            "panel_year": [2011, 2010, 2010, 2010, 2010, 2011],
            "purchase_date": ["2011-01-01", "2010-01-02", "2010-01-02", "2010-01-02", "2010-01-02", "2011-01-01"],
            "trip_code_uc": [1, 2, 3, 3, 3, 4], "upc": [1, 2, 3, 3, 3, 4],
            "quantity": [1.] * 6,
        }).with_columns(pl.col("purchase_date").str.to_date())
        fields = ["race", "income", "hispanic_origin", "household_size", "head_age",
                  "female_head_age", "male_head_age", "dma_cd"]
        source = source.with_columns([pl.Series(c, ["9", None, None, "2", "3", "7"]) for c in fields])
        expected = source.lazy().group_by("household_code").agg([
            pl.col(c).sort_by(["panel_year", "purchase_date"]).drop_nulls().first().alias(c)
            for c in fields
        ]).collect().sort("household_code")
        actual = summary.household_summary(source.lazy()).collect().sort("household_code")
        self.assertEqual(actual.select(["household_code"] + fields).to_dicts(), expected.to_dicts())
        self.assertEqual(actual["purchase_lines"].to_list(), [4, 2])
        with tempfile.TemporaryDirectory() as directory:
            parts = summary.partition_purchases(source.lazy(), Path(directory), "check", 16)
            restored = pl.concat(parts).collect()
            self.assertEqual(restored.height, source.height)
            counts = restored.group_by("household_code").len().sort("household_code")
            self.assertEqual(counts["len"].to_list(), [4, 2])
            batched = pl.concat([summary.household_summary(part) for part in parts]).collect().sort("household_code")
            self.assertEqual(batched.to_dicts(), actual.to_dicts())

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

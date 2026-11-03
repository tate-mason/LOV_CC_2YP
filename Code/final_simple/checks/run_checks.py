"""Run synthetic verification while prohibiting CSV reads.

Run from final_simple: python3 checks/run_checks.py
The complete legacy export tests are still available with unittest discovery;
two of those reopen their generated CSVs and are excluded by this runner.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import polars as pl

modules = ["checks.test_est", "checks.test_price_control", "checks.test_hessian_se",
           "checks.test_wtp", "checks.test_empirical_timing", "checks.test_output_records", "checks.test_supply_derivatives", "checks.test_data_build", "checks.test_data_summaries"]
excluded = {"test_mixture_fit_outputs", "test_bootstrap_refits_first_stage_and_reports_empirical_sd"}


def individual_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from individual_tests(item)
        else:
            yield item


def no_csv_reads(*arguments, **keywords):
    raise AssertionError("CSV reads are deliberately disabled in these checks.")


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for module in modules:
        loaded = unittest.defaultTestLoader.loadTestsFromName(module)
        for test in individual_tests(loaded):
            if test._testMethodName not in excluded:
                suite.addTest(test)
    with patch.object(pd, "read_csv", side_effect=no_csv_reads), \
         patch.object(pl, "read_csv", side_effect=no_csv_reads), \
         patch.object(pl, "scan_csv", side_effect=no_csv_reads):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("CSV reads were blocked. Checks use generated arrays and temporary synthetic Parquet inputs.")
    print("Two CSV-reopening legacy export checks are retained but excluded from this runner.")
    sys.exit(0 if result.wasSuccessful() else 1)

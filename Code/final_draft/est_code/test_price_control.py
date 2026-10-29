"""Checks for DMA exclusion, first-stage OLS, and two-stage inference."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from unittest.mock import patch

from price_control import other_dma_instruments, absorb_effects, add_price_controls
from bootstrap_se import resample_households, bootstrap_standard_errors
from est import FLAVORS, make_arrays, optimize_demand


def synthetic_panel():
    rng = np.random.default_rng(120)
    n_households, periods = 60, 8
    size = n_households * periods
    frame = pd.DataFrame({"household_code": np.repeat(np.arange(n_households), periods),
                          "week": np.tile(pd.date_range("2010-01-02", periods=periods, freq="7D"), n_households),
                          "dma_cd": np.repeat(np.arange(n_households) % 4 + 1, periods)})
    for flavor in FLAVORS:
        frame["xi_" + flavor] = rng.normal(size=size)
        frame["iv_" + flavor] = rng.uniform(.5, 2.5, size=size)
        frame["price_" + flavor] = (1 + .8*frame["iv_" + flavor]
                                    + .1*frame["xi_" + flavor] + rng.normal(0, .2, size))
    controlled, _ = add_price_controls(frame)
    utility = (np.array([0., .4, -.3])
               - controlled[["price_"+f for f in FLAVORS]].to_numpy()
               + .8*controlled[["cf_"+f for f in FLAVORS]].to_numpy())
    probs = np.exp(utility - utility.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    counts = np.array([rng.multinomial(8, p) for p in probs])
    for j, flavor in enumerate(FLAVORS):
        frame[flavor] = counts[:, j]
    return frame


class PriceControlChecks(unittest.TestCase):
    def test_excludes_consumer_dma_even_when_exposure_store_elsewhere(self):
        week = pd.Timestamp("2010-01-02")
        weekly = pd.DataFrame({"store_code_uc": [99, 99, 99], "week": [week]*3,
                               "dma_cd": [1, 2, 4]})
        exposure = pd.DataFrame({"store_code_uc": [99, 99], "week": [week]*2,
                                 "upc": ["a", "b"], "flavor": ["berry", "berry"]})
        prices = pd.DataFrame({"upc": ["a"]*3+["b"]*3, "week": [week]*6,
                               "dma_code": [1, 2, 3]*2, "dma_price": [1., 3., 5., 2., 4., 6.]})
        result = other_dma_instruments(weekly, exposure, prices)
        np.testing.assert_allclose(result["iv_berry"], [4.5, 3.5, 3.5])
        # If one UPC has no outside-DMA observation, the entire flavor IV is missing.
        missing = prices.loc[~((prices["upc"] == "b") & (prices["dma_code"] != 1))]
        result = other_dma_instruments(weekly, exposure, missing)
        self.assertTrue(np.isnan(result.loc[0, "iv_berry"]))

    def test_absorption_matches_explicit_dummy_ols(self):
        rng = np.random.default_rng(20)
        weeks = pd.Series([1, 1, 2, 2, 2, 3, 3, 4, 4, 4])
        dma = pd.Series([1, 2, 1, 2, 3, 1, 3, 1, 2, 3])
        x = rng.normal(size=(len(weeks), 3))
        dummies = pd.concat([pd.get_dummies(weeks, dtype=float),
                             pd.get_dummies(dma, dtype=float)], axis=1).to_numpy()
        expected = x - dummies @ np.linalg.lstsq(dummies, x, rcond=None)[0]
        np.testing.assert_allclose(absorb_effects(x, [weeks, dma]), expected, atol=1e-8)

    def test_first_stage_residuals_orthogonal_and_in_utility(self):
        frame = synthetic_panel()
        controlled, report = add_price_controls(frame)
        self.assertTrue((report["iv_coefficient"] > .7).all())
        self.assertTrue((report["iv_cluster_F"] > 10).all())
        for flavor in FLAVORS:
            residual = controlled["cf_" + flavor]
            self.assertAlmostEqual(float(residual @ frame["iv_" + flavor]), 0., places=7)
            np.testing.assert_allclose(residual.groupby(frame["week"]).mean(), 0., atol=1e-8)
        data = make_arrays(controlled, 0)
        self.assertTrue(data["control"])
        np.testing.assert_allclose(data["cf"], controlled[["cf_"+f for f in FLAVORS]])

    def test_resampling_preserves_histories_and_relabels_duplicates(self):
        frame = synthetic_panel()
        sampled = resample_households(frame, np.random.default_rng(1))
        self.assertEqual(sampled["household_code"].nunique(), 60)
        np.testing.assert_array_equal(sampled.groupby("household_code").size(), 8)
        self.assertFalse(sampled.duplicated(["household_code", "week"]).any())

    def test_bootstrap_refits_first_stage_and_reports_empirical_sd(self):
        frame = synthetic_panel()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(seed=306, bootstrap_reps=20, no_price_control=False,
                                   output=Path(directory))
            def estimate(sample, initial):
                # Explicitly depend on the recomputed residual, so frozen residuals
                # would produce a different bootstrap distribution.
                theta = np.array([sample["cf_other"].std(), sample["cf_berry"].std()])
                return OptimizeResult(x=theta, fun=1., success=True, message="test fit")
            with patch("bootstrap_se.add_price_controls", wraps=add_price_controls) as first_stage:
                with contextlib.redirect_stdout(io.StringIO()):
                    table, report = bootstrap_standard_errors(frame, args, estimate, np.zeros(2), ["a", "b"])
            self.assertEqual(first_stage.call_count, 20)
            self.assertEqual(report["usable"], 20)
            draws = pd.read_csv(Path(directory) / "bootstrap_draws.csv")
            np.testing.assert_allclose(table["std_error"], draws[["a", "b"]].std(ddof=1))
            self.assertTrue((table["std_error"] > 0).all())

    def test_controlled_demand_optimizer_runs(self):
        controlled, _ = add_price_controls(synthetic_panel())
        args = SimpleNamespace(model="naive", draws=8, seed=306, starts=2, maxiter=150)
        fit, _, _ = optimize_demand(make_arrays(controlled, 0), args, quiet=True)
        self.assertTrue(fit.success)
        self.assertEqual(len(fit.x), 4)  # Two betas, mean log(-alpha), price control.
        self.assertTrue(np.isfinite(fit.fun))


if __name__ == "__main__":
    unittest.main()

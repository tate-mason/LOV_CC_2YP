"""Small checks that need no Nielsen data: python3 -m unittest -v test_est.py."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import polars as pl
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from est import read_flavors, prepare_data, fit_model, add_stocks, make_arrays, household_loglike, make_draws, parameter_names


class EstimationChecks(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "household_code": [10, 10, 10],
            "week": pd.to_datetime(["2010-01-02", "2010-01-09", "2010-01-23"]),
            "other": [2., 0., 1.], "berry": [0., 2., 1.], "plain": [0., 0., 0.],
            "price_other": [1., 1.2, 1.1], "price_berry": [1.2, 1., 1.3],
            "price_plain": [1.5, 1.4, 1.3], "outside": [5., 5., 5.]})

    def test_history_excludes_current_counts_and_accumulates_all_occasions(self):
        frame = add_stocks(self.frame())
        stock = frame[["stock_other", "stock_berry", "stock_plain"]]
        np.testing.assert_allclose(stock.iloc[0], [0., 0., 0.])
        np.testing.assert_allclose(stock.iloc[1], [1., 0., 0.])
        np.testing.assert_allclose(stock.iloc[2], [1., 1., 0.])
        changed = self.frame()
        changed.loc[2, "plain"] = 100
        np.testing.assert_allclose(add_stocks(changed).filter(like="stock_"),
                                   frame.filter(like="stock_"))

    def test_xi_sign_and_unlimited_history(self):
        frame = add_stocks(self.frame(), .7)
        self.assertAlmostEqual(frame.loc[1, "xi_other"], -.7)
        self.assertAlmostEqual(frame.loc[1, "xi_berry"], .3)
        # Time gaps do not change occasion-time histories when delta=1.
        old = self.frame()
        old.loc[2, "week"] = pd.Timestamp("2011-04-03")
        old = add_stocks(old)
        np.testing.assert_allclose(old.filter(like="xi_"), frame.filter(like="xi_"))
        self.assertEqual(frame.loc[2, "novelty_other"], 0.)
        self.assertAlmostEqual(frame.loc[2, "xi_other"], -.7)

    def test_parquet_pipeline_and_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            def append(household, week, upc, store, dma, price, quantity=1):
                rows.append(dict(household_code=household, panel_year=2010,
                                 purchase_date=str(pd.to_datetime(str(week), format="%Y%m%d").date()),
                                 trip_code_uc=len(rows) + 1, upc=str(upc), store_code_uc=store,
                                 quantity=quantity, price=float(price), dma_cd=dma,
                                 flavor={1: "other", 2: "berry", 3: "plain"}[upc]))
            append(10, 20100102, 1, 100, 1, 3, 2)
            append(10, 20100102, 2, 200, 1, 4)
            append(10, 20100109, 2, 100, 1, 4, 2)
            for week in [20100102, 20100109]:
                for upc in [1, 2, 3]:
                    append(20, week, upc, 200, 2, 2 + upc)
                    append(30, week, upc, 300, 3, 6 + upc)
                    append(40, week, upc, 100, 1, 2 + upc)
            # Repeat a cheap quote: it must not overweight the overall IV.
            append(20, 20100102, 1, 200, 2, 3)
            sample = root / "prepared.parquet"
            original = pl.DataFrame(rows)
            original.write_parquet(sample)
            args = SimpleNamespace(
                sample=sample, flavors=root / "not_needed.csv", rms=root / "does_not_exist.parquet",
                first_year=2010, last_year=2010, max_households=1, habit_weight=.7,
                capacity=0, no_price_control=True, household_dma_column="dma_cd", se_method="none",
                model="heterogeneous", draws=8, starts=1, maxiter=5, seed=306, output=root / "output")
            frame = prepare_data(args)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.household_code.unique().tolist(), [10])
            self.assertEqual(frame.loc[0, "other"], 2)
            self.assertEqual(frame.loc[0, "berry"], 1)
            self.assertEqual(frame.loc[0, "store_code_uc"], 100)
            self.assertEqual(frame.loc[0, "price_other"], 3)
            np.testing.assert_allclose(frame.filter(like="stock_").iloc[0], 0)
            fit = fit_model(make_arrays(frame, 0), args)
            self.assertTrue(np.isfinite(fit.fun))
            self.assertTrue((args.output / "fit.json").exists())
            incomplete = original.filter(~((pl.col("purchase_date") == "2010-01-02")
                                             & (pl.col("store_code_uc") == 100) & (pl.col("upc") == "3")))
            incomplete.write_parquet(sample)
            with self.assertRaisesRegex(ValueError, "at least two usable"):
                prepare_data(args)
            original.write_parquet(sample)
            args.no_price_control = False
            controlled = prepare_data(args)
            # Pool DMA 2's quotes 3,4,5 and DMA 3's 7,8,9; mean = 6.
            for name in ["iv_overall", "iv_other", "iv_berry", "iv_plain"]:
                np.testing.assert_allclose(controlled[name], 6.)
            # Changing every own-DMA quote cannot change the instrument.
            changed = original.with_columns(pl.when(pl.col("dma_cd") == 1)
                                            .then(pl.col("price") + 100).otherwise(pl.col("price")).alias("price"))
            changed.write_parquet(sample)
            # Store 200 is shared across DMAs in this fixture. Update both copies
            # of its quoted price to keep local store/week/UPC prices consistent.
            changed = changed.with_columns(pl.when(pl.col("store_code_uc") == 200)
                                            .then(pl.col("upc").cast(pl.Float64) + 2).otherwise(pl.col("price")).alias("price"))
            changed.write_parquet(sample)
            np.testing.assert_allclose(prepare_data(args)["iv_overall"], 6.)

    def test_persistent_household_integration(self):
        data = make_arrays(add_stocks(self.frame()), 7)
        draws = make_draws(8, 306)
        names = parameter_names("heterogeneous", True)
        theta = np.array([.4 if n.startswith("sd_") else .2 for n in names])
        actual = household_loglike(theta, data, draws, "heterogeneous")[0]
        # Independently calculate each fixed type's full household history.
        by_draw = []
        for z in draws:
            alpha = -np.exp(.2 + .4*z[0])
            gamma = .2 + .4*z[1:]
            utility = np.array([.2, .4, .4]) + alpha*data["prices"] + gamma*data["xi"]
            utility = np.column_stack([utility, np.zeros(3)])
            logp = utility - logsumexp(utility, axis=1, keepdims=True)
            by_draw.append(np.sum(data["counts"]*logp))
        self.assertAlmostEqual(actual, logsumexp(by_draw)-np.log(8))

    def test_four_model_specifications(self):
        for model, gamma, alpha_sd, gamma_sd in [
            ("naive", False, False, False),
            ("homogeneous", True, False, False),
            ("mixed_naive", False, True, False),
            ("heterogeneous", True, True, True),
        ]:
            with self.subTest(model=model):
                names = parameter_names(model, False)
                self.assertEqual("mean_gamma_other" in names, gamma)
                self.assertEqual("sd_log_minus_alpha" in names, alpha_sd)
                self.assertEqual("sd_gamma_other" in names, gamma_sd)
                self.assertFalse(any("sd_beta" in n for n in names))

    def test_mixed_naive_integrates_only_alpha(self):
        data = make_arrays(add_stocks(self.frame()), 0)
        draws = make_draws(8, 306)
        theta = np.array([.1, -.2, -.4, .6])
        actual = household_loglike(theta, data, draws, "mixed_naive")
        independent = []
        for z in draws:
            alpha = -np.exp(-.4 + .6*z[0])
            utility = np.array([0., .1, -.2]) + alpha*data["prices"]
            logp = utility - logsumexp(utility, axis=1, keepdims=True)
            independent.append(np.sum(data["counts"]*logp))
        np.testing.assert_allclose(actual, [logsumexp(independent)-np.log(len(draws))])
        changed = dict(data, xi=data["xi"] + 100.)
        np.testing.assert_allclose(actual, household_loglike(theta, changed, draws, "mixed_naive"))
        np.testing.assert_allclose(
            household_loglike(theta[:3], data, draws, "naive"),
            household_loglike(np.r_[theta[:3], 0.], data, draws, "mixed_naive"))

    def test_zero_heterogeneity_nests_homogeneous_model(self):
        data = make_arrays(add_stocks(self.frame()), 0)
        fixed = np.array([.1, -.2, -.4, .3, -.2, .5])
        random = np.r_[fixed, np.zeros(4)]
        draws = make_draws(8, 306)
        np.testing.assert_allclose(
            household_loglike(fixed, data, draws, "homogeneous"),
            household_loglike(random, data, draws, "heterogeneous"))


if __name__ == "__main__":
    unittest.main()

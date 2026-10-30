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
            flavor = pd.DataFrame({"upc": [1, 2, 3],
                                   "flavor_descr": ["vanilla", "berry", "plain"],
                                   "product_descr": ["yogurt"]*3})
            flavor.loc[flavor["upc"] != 3].to_csv(root / "flavors.csv", index=False)
            pl.DataFrame({"upc": [1, 3, 3], "flavor_descr": ["berry", "vanilla", "plain"],
                          "panel_year": [2006, 2011, 2010]}).write_parquet(root / "products_extra.parquet")
            # Two stores in one household-week must produce ONE basket.
            pl.DataFrame({
                "household_code": [10, 10, 10, 20], "panel_year": [2010]*4,
                "purchase_date": ["2010-01-02", "2010-01-02", "2010-01-09", "2010-01-09"],
                "trip_code_uc": [1, 2, 3, 4], "upc": [1, 2, 2, 3],
                "store_code_uc": [100, 200, 100, 100], "quantity": [2, 1, 2, 1],
            }).write_parquet(root / "sample.parquet")
            rows = []
            for week in [20100102, 20100109]:
                for store in [100, 200]:
                    for upc in [1, 2, 3]:
                        rows.append(dict(upc=upc, store_code_uc=store, week_end=week,
                                         price=2.+upc, product_module_code=3612))
            pl.DataFrame(rows).write_parquet(root / "rms.parquet")
            args = SimpleNamespace(
                flavors=root / "flavors.csv", sample=root / "sample.parquet",
                rms=root / "rms.parquet", products_extra=root / "products_extra.parquet", first_year=2010, last_year=2010,
                max_households=2, habit_weight=.7, capacity=0, no_price_control=True,
                se_method="none",
                model="heterogeneous", draws=8, starts=1, maxiter=5, seed=306,
                output=root / "output")
            frame = prepare_data(args)
            # Read a full prepared purchase table and exact saved flavor mapping.
            mapping = read_flavors(args.flavors, args.products_extra)
            mapping.to_csv(root / "estimation_flavor_reference.csv", index=False)
            raw = pl.read_parquet(args.sample).with_columns(pl.col("upc").cast(pl.String))
            raw.join(pl.from_pandas(mapping), on="upc", how="left").write_parquet(root / "prepared.parquet")
            args.sample = root / "prepared.parquet"
            args.flavors = root / "estimation_flavor_reference.csv"
            prepared_frame = prepare_data(args)
            pd.testing.assert_frame_equal(frame, prepared_frame)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.household_code.unique().tolist(), [10])
            self.assertEqual(frame.loc[0, "other"], 2)
            self.assertEqual(frame.loc[0, "berry"], 1)
            self.assertEqual(frame.loc[0, "store_code_uc"], 100)
            self.assertEqual(frame.loc[0, "price_other"], 3.0)
            np.testing.assert_allclose(frame.filter(like="stock_").iloc[0], 0)
            fit = fit_model(make_arrays(frame, 0), args)
            self.assertTrue(np.isfinite(fit.fun))
            self.assertTrue((args.output / "fit.json").exists())
            # The instrument must include an unvisited store in a third DMA.
            panel = pl.read_parquet(args.sample).with_columns(pl.lit(1).alias("dma_cd"))
            panel.write_parquet(args.sample)
            rms = pl.read_parquet(args.rms).with_columns(
                pl.when(pl.col("store_code_uc") == 100).then(1).otherwise(2).alias("dma_code"))
            extra = rms.filter(pl.col("store_code_uc") == 100).with_columns(
                pl.lit(300, dtype=pl.Int64).alias("store_code_uc"),
                pl.lit(3, dtype=pl.Int32).alias("dma_code"),
                (pl.col("price") + 4).alias("price"))
            pl.concat([rms, extra], how="vertical_relaxed").write_parquet(args.rms)
            # Two observed weeks but only one complete menu must also be dropped.
            incomplete = pl.read_parquet(args.rms).filter(
                ~((pl.col("week_end") == 20100102) & (pl.col("upc") == 3)))
            original_rms = pl.read_parquet(args.rms)
            incomplete.write_parquet(args.rms)
            with self.assertRaisesRegex(ValueError, "at least two usable"):
                prepare_data(args)
            original_rms.write_parquet(args.rms)
            args.no_price_control = False
            args.household_dma_column = "dma_cd"
            args.retail_dma_column = "dma_code"
            controlled = prepare_data(args)
            np.testing.assert_allclose(controlled["iv_other"], 5.0)

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

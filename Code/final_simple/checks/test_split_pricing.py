"""Synthetic timing and backward-induction checks; no CSV input or output."""
import math
import unittest

import numpy as np
import pandas as pd

from demand import split_pricing as pricing


class EmpiricalTimingTests(unittest.TestCase):
    def test_first_observation_becomes_t0(self):
        rows = []
        for household, initial in [("a", [1., 3., 0.]), ("b", [0., 0., 0.])]:
            for t in range(3):
                counts = initial if t == 0 else [0., 0., 7.]
                row = dict(household_code=household,
                           week=pd.date_range("2020-01-04", periods=3, freq="7D")[t])
                for j, flavor in enumerate(pricing.FLAVORS):
                    row[flavor] = counts[j]
                    row["stock_"+flavor] = 0. if t == 0 else 10.+t
                    row["price_"+flavor] = 1.+j+t
                    row["cf_"+flavor] = .1*j+t
                # Eligibility comes from t0, even when berry is available later.
                if household == "b" and t == 0:
                    row["price_berry"] = np.nan
                rows.append(row)
        frame = pd.DataFrame(rows).sample(frac=1., random_state=7)
        households = pd.DataFrame({"household_code": ["b", "a"]})
        context = pricing.forecast_context(frame, households)
        np.testing.assert_allclose(context["stocks"], [[0., 0., 0.], [.25, .75, 0.]])
        np.testing.assert_allclose(context["novelty"], [[0., 0., 0.], [.75, .25, 1.]])
        np.testing.assert_allclose(context["prices"], [[1., 0., 3.], [1., 2., 3.]])
        np.testing.assert_allclose(context["cf"], [[0., .1, .2], [0., .1, .2]])
        np.testing.assert_array_equal(context["eligible"], [False, True])
        self.assertTrue(households.initial_observed_week.eq(pd.Timestamp("2020-01-04")).all())
        self.assertTrue(households.pricing_week1.eq(pd.Timestamp("2020-01-11")).all())
        self.assertTrue(households.pricing_week2.eq(pd.Timestamp("2020-01-18")).all())
        later = frame.week > pd.Timestamp("2020-01-04")
        changed = frame.copy()
        changed.loc[later, [*pricing.FLAVORS, *["stock_"+f for f in pricing.FLAVORS]]] = 999.
        for prefix in ["price_", "cf_"]:
            changed.loc[later, [prefix+f for f in pricing.FLAVORS]] = 123.
        repeated = pricing.forecast_context(changed, households.copy())
        for key in context:
            np.testing.assert_array_equal(context[key], repeated[key])

    def test_p1_h0_then_terminal_p2_h1_matches_scalar_enumeration(self):
        # h0 is an observed berry-only basket: stock=1 and novelty for rivals=1.
        households = pd.DataFrame({"household_code": ["synthetic"], "group": ["low"]})
        context = dict(stocks=np.array([[0., 1., 0.]]),
                       novelty=np.array([[1., 0., 1.]]),
                       prices=np.array([[2., 2., 2.]]),
                       available=np.ones((1, 3), dtype=bool),
                       cf=np.zeros((1, 3)), eligible=np.ones(1, dtype=bool))
        discounts = np.linspace(0., 1., 101)
        a, g, capacity, delta = 1., -.8, 2, .92
        b_n, b_s = -.6, .35
        beta = np.ones(3)*.5
        table, error = pricing.targeted_coupons(
            context, households, np.ones((1, 1)), np.array([a]), np.array([[b_n, b_s]]),
            beta, discounts, capacity=capacity, delta=delta)
        self.assertLess(error, 1e-12)
        baskets = pricing.compositions(capacity)
        terminal_values = []
        for basket in baskets:
            x = basket[:3]/max(basket[:3].sum(), 1)
            history = b_n*(x.sum()-x)+b_s*(context["stocks"][0]+x)
            values = []
            for discount in discounts:
                prices = context["prices"][0].copy()
                prices[1] *= 1-discount
                utilities = np.r_[beta-a*prices+history, 0.]
                probability = np.exp(utilities-pricing.logsumexp(utilities))
                values.append(capacity*prices[1]*probability[1])
            terminal_values.append(max(values))
        total_values = []
        for discount in discounts:
            prices = context["prices"][0].copy()
            prices[1] *= 1-discount
            history = b_n*context["novelty"][0]+b_s*context["stocks"][0]
            utilities = np.r_[beta-a*prices+history, 0.]
            probability = np.exp(utilities-pricing.logsumexp(utilities))
            first = capacity*prices[1]*probability[1]
            mass = np.array([
                math.factorial(capacity)/math.prod(math.factorial(int(n)) for n in basket)
                *np.prod(probability**basket) for basket in baskets])
            total_values.append(first+delta*(mass @ terminal_values))
        policies = table.set_index("policy")
        self.assertEqual(set(policies.index), {"no_coupon", "personalized_dynamic"})
        dynamic = policies.loc["personalized_dynamic"]
        self.assertEqual(dynamic.discount1, discounts[np.argmax(total_values)])
        self.assertAlmostEqual(dynamic.discounted_revenue, max(total_values), places=12)



class SplitPosteriorTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for i in range(8):
            stock = np.zeros(3)
            previous = np.zeros(3)
            for t in range(3):
                counts = np.eye(3)[(i+t) % 3]
                novelty = previous.sum()-previous
                row = dict(household_code=str(i), week=pd.Timestamp('2020-01-04')+pd.Timedelta(weeks=t), outside=0.)
                for j, f in enumerate(pricing.FLAVORS):
                    row.update({f: counts[j], 'price_'+f: 1.+j/2,
                                'novelty_'+f: novelty[j], 'stock_'+f: stock[j]})
                rows.append(row)
                stock += counts
                previous = counts
        parameters = dict(inside_constant=.5, beta_berry=.2, beta_plain=-.1,
                          mu_log_minus_alpha=-.3, sd_log_minus_alpha=.4,
                          mean_novelty=-.2, sd_novelty=.7, mean_own_stock=.1, sd_own_stock=.3)
        return pd.DataFrame(rows), parameters

    def test_posterior_matches_scalar_and_estimator(self):
        frame, p = self.fixture()
        draws = pricing.persistent_draws(8, 306)
        households, weights, alpha, history, beta, nll = pricing.posterior(frame, p, draws, False)
        scalar = []
        for _, block in frame.groupby('household_code', sort=True):
            logs = []
            for a, (b_n, b_s) in zip(alpha, history):
                ll = 0.
                for _, row in block.iterrows():
                    u = [beta[j]-a*row['price_'+f]+b_n*row['novelty_'+f]+b_s*row['stock_'+f]
                         for j, f in enumerate(pricing.FLAVORS)]+[0.]
                    probs = np.exp(u-pricing.logsumexp(u))
                    ll += sum(row[f]*np.log(probs[j]) for j, f in enumerate(pricing.FLAVORS+['outside']))
                logs.append(ll)
            scalar.append(logs)
        scalar = np.array(scalar)
        expected = np.exp(scalar-pricing.logsumexp(scalar, axis=1, keepdims=True))
        np.testing.assert_allclose(weights, expected, atol=1e-13)
        self.assertAlmostEqual(nll, -sum(pricing.logsumexp(row)-np.log(len(draws)) for row in scalar), places=12)
        np.testing.assert_allclose(households.habit_mean, weights @ history[:, 1])
        from demand import estimation as est
        for f in pricing.FLAVORS:
            frame["xi_"+f] = .3*frame["novelty_"+f]-.7*frame["stock_"+f]
        data = est.make_arrays(frame, 1)
        names = est.parameter_names("heterogeneous_history", True, False)
        estimator_draws = est.make_draws(8, 306, "heterogeneous_history")
        np.testing.assert_array_equal(draws, estimator_draws)
        expected_ll = est.household_loglike(np.array([p[n] for n in names]), data,
                                           estimator_draws, "heterogeneous_history")
        np.testing.assert_allclose(pricing.logsumexp(scalar, axis=1)-np.log(len(draws)),
                                   expected_ll, atol=1e-12)
        # Both coefficients affect probabilities; logit collapsing is exact.
        u = np.array([.3, -.2, .7, 0.])
        probs = np.exp(u-pricing.logsumexp(u))
        for focal in range(3):
            inclusive = pricing.logsumexp(np.delete(u, focal))
            self.assertAlmostEqual(pricing.expit(u[focal]-inclusive), probs[focal])

    def test_three_focal_cases_both_type_treatments(self):
        frame, parameters = self.fixture()
        households, weights, alpha, history, beta, _ = pricing.posterior(
            frame, parameters, pricing.persistent_draws(8, 306), False)
        for focal in pricing.FLAVORS:
            forecast = frame.copy()
            intercepts = beta.copy()
            if focal != 'berry':
                for destination, source in {'berry': focal, focal: 'berry'}.items():
                    for prefix in ['', 'price_', 'stock_', 'novelty_']:
                        forecast[prefix+destination] = frame[prefix+source]
                    intercepts[pricing.FLAVORS.index(destination)] = beta[pricing.FLAVORS.index(source)]
            context = pricing.forecast_context(forecast, households.copy())
            for treatment in ['posterior-mean', 'posterior-integrated']:
                table, error = pricing.targeted_coupons(context, households, weights, alpha,
                    history, intercepts, np.array([0., .5, 1.]), capacity=2,
                    delta=.92, type_treatment=treatment)
                self.assertLess(error, 1e-12)
                self.assertTrue((table.revenue_change >= -1e-10).all())


class PlainBerryComparisonTests(unittest.TestCase):
    def test_only_focal_sellers_change_and_full_demand_is_retained(self):
        from demand import plain_berry_pricing as comparison
        from types import SimpleNamespace
        from pathlib import Path
        from unittest.mock import patch
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(household_input=Path('/server/weekly_inputs.csv'),
                fit_dir=Path('/server/split_fit'), output=Path(directory),
                discounts=np.array([0., .5, 1.]), demand_delta=.92,
                type_treatment='posterior-mean', batch_size=24, save_state_policies=False)
            def solve(case):
                self.assertEqual(case.household_input, settings.household_input)
                self.assertEqual(case.fit_dir, settings.fit_dir)
                return dict(aggregate=[dict(policy='personalized_dynamic', discounted_revenue=1.)])
            captured = []
            def capture(frame, path, **kwargs):
                captured.append(frame.copy())
            with patch.object(comparison.pricing, 'run_households', side_effect=solve) as solver, \
                    patch.object(pd.DataFrame, 'to_csv', capture):
                manifest = comparison.run_comparison(settings)
            self.assertEqual([call.args[0].focal_flavor for call in solver.call_args_list], ['plain', 'berry'])
            self.assertEqual(set(captured[0].focal_flavor), {'plain', 'berry'})
            self.assertEqual(manifest['demand_categories'], ['other', 'berry', 'plain'])
            self.assertEqual(set(manifest['results']), {'plain', 'berry'})
            self.assertFalse((settings.output/'other').exists())
            self.assertEqual(settings.output, Path(directory))


if __name__ == "__main__":
    unittest.main()

"""Synthetic timing and backward-induction checks; no CSV input or output."""
import math
import unittest

import numpy as np
import pandas as pd

from pricing_exercises import run_empirical_targeting as pricing


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
        beta = np.ones(3)*.5
        table, error = pricing.targeted_coupons(
            context, households, np.ones((1, 1)), np.array([a]), np.array([g]),
            beta, discounts, capacity=capacity, delta=delta)
        self.assertLess(error, 1e-12)
        baskets = pricing.compositions(capacity)
        terminal_values = []
        for basket in baskets:
            x = basket[:3]/max(basket[:3].sum(), 1)
            xi = .3*(x.sum()-x)-.7*(context["stocks"][0]+x)
            values = []
            for discount in discounts:
                prices = context["prices"][0].copy()
                prices[1] *= 1-discount
                utilities = np.r_[beta-a*prices+g*xi, 0.]
                probability = np.exp(utilities-pricing.logsumexp(utilities))
                values.append(capacity*prices[1]*probability[1])
            terminal_values.append(max(values))
        first_values, total_values = [], []
        for discount in discounts:
            prices = context["prices"][0].copy()
            prices[1] *= 1-discount
            xi = .3*context["novelty"][0]-.7*context["stocks"][0]
            utilities = np.r_[beta-a*prices+g*xi, 0.]
            probability = np.exp(utilities-pricing.logsumexp(utilities))
            first = capacity*prices[1]*probability[1]
            mass = np.array([
                math.factorial(capacity)/math.prod(math.factorial(int(n)) for n in basket)
                *np.prod(probability**basket) for basket in baskets])
            first_values.append(first)
            total_values.append(first+delta*(mass @ terminal_values))
        policies = table.set_index("policy")
        dynamic = policies.loc["personalized_dynamic"]
        myopic = policies.loc["personalized_myopic"]
        self.assertEqual(dynamic.discount1, discounts[np.argmax(total_values)])
        self.assertAlmostEqual(dynamic.discounted_revenue, max(total_values), places=12)
        self.assertEqual(myopic.discount1, discounts[np.argmax(first_values)])
        self.assertGreater(dynamic.discount1, myopic.discount1)


if __name__ == "__main__":
    unittest.main()

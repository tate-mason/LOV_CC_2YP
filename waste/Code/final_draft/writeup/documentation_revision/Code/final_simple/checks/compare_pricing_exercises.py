"""Compare every pricing exercise using synthetic inputs only.

Run: python3 checks/compare_pricing_exercises.py /path/to/final_draft
Research scripts are independent of final_draft; only this comparison reads it.
"""
from pathlib import Path
import ast
import importlib.util
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demand import pricing
from supply import pricing_model, unknown_type


def load_reference(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(original):
    old_pricing = load_reference("reference_pricing", original / "pricing_exercises/run_empirical_targeting.py")
    old_supply = load_reference("reference_supply", original / "pricing_exercises/run_supply_exercises.py")
    old_type = load_reference("reference_type", original / "pricing_exercises/unknown_type/run_unknown_type.py")
    rng = np.random.default_rng(53)
    rows = []
    for household in range(8):
        stock = np.zeros(3)
        previous = np.zeros(3)
        for week in range(4):
            counts = rng.multinomial(3, [.2, .4, .2, .2])
            if counts[:3].sum() == 0:
                counts[1], counts[3] = 1, 2
            novelty = 1-previous
            row = dict(household_code=str(household), week=pd.date_range('2010-01-02', periods=4, freq='7D')[week], outside=counts[3])
            for j, flavor in enumerate(pricing.FLAVORS):
                row[flavor] = float(counts[j])
                row['price_'+flavor] = 1 + rng.uniform(0, 1)
                row['cf_'+flavor] = rng.normal(0, .1)
                row['xi_'+flavor] = .3*novelty[j]-.7*stock[j]
                row['stock_'+flavor] = stock[j]
                row['novelty_'+flavor] = novelty[j]
            rows.append(row)
            previous = counts[:3]/max(counts[:3].sum(), 1)
            stock += previous
    frame = pd.DataFrame(rows)
    parameters = dict(inside_constant=.8, beta_berry=-.2, beta_plain=-.1, mu_log_minus_alpha=-.3, sd_log_minus_alpha=.4, mean_gamma=-.1, sd_gamma=.2, price_control=.5)
    draws = pricing.persistent_draws(8, 19)
    households, weights, alpha, gamma, beta, nll = pricing.posterior(frame, parameters, draws, True)
    old_households, old_weights, old_alpha, old_gamma, old_beta, old_nll = old_pricing.posterior(frame, parameters, draws, True)
    pd.testing.assert_frame_equal(households, old_households)
    np.testing.assert_allclose(weights, old_weights)
    np.testing.assert_allclose(nll, old_nll)
    context = pricing.forecast_context(frame, households)
    old_context = old_pricing.forecast_context(frame, old_households)
    for key in context:
        np.testing.assert_array_equal(context[key], old_context[key])
    # Include a missing competitor menu and an ineligible household.
    context['available'][0, 2] = False
    context['prices'][0, 2] = 0.
    context['eligible'][1] = False
    for treatment in ['posterior-mean', 'posterior-integrated']:
        for capacity in [1, 2, 7]:
            for discount_factor in [0., .95]:
                arguments = (context, households, weights, alpha, gamma, beta, np.linspace(0, 1, 11))
                options = dict(capacity=capacity, delta=discount_factor, type_treatment=treatment, control_coefficient=.5, batch_size=3)
                new, new_error = pricing.targeted_coupons(*arguments, **options)
                old, old_error = old_pricing.targeted_coupons(*arguments, **options)
                # The simple loop groups rows by household; the old batch loop groups by policy.
                order = ['household_code', 'policy']
                new = new.sort_values(order).reset_index(drop=True)
                old = old.sort_values(order).reset_index(drop=True)
                pd.testing.assert_frame_equal(new, old, check_exact=False, rtol=1e-11, atol=1e-11)
                assert max(new_error, old_error) < 1e-12
    print('Matched all household-pricing output columns: both type treatments, three capacities, two discount factors, missing menus and eligibility.')


    # Compare the actual baseline functions used by supply_simulation, without
    # importing its unrelated symbolic/derivative dependencies.
    baseline = ast.parse((ROOT / 'supply/supply_simulation.py').read_text())
    names = {'expit', 'golden_maximum', 'scalar_maximum', 'static_price',
             'supply_solution', 'outcome_tables'}
    functions = [node for node in baseline.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(vars(pricing_model))
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'baseline_calculations', 'exec'), namespace)
    for gamma in [-2., -1.25, 0., 1.25, 2.]:
        for prior in [0., .25, .5, .75, 1.]:
            expected = old_supply.supply_solution(gamma, prior)
            for actual in [pricing_model.supply_solution(gamma, prior), namespace['supply_solution'](gamma, prior)]:
                assert actual.keys() == expected.keys()
                for key in actual:
                    np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
                for a, b in zip(pricing_model.outcome_tables(actual), old_supply.outcome_tables(expected)):
                    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))
    print('Matched baseline prices, FOCs, history and expected outcomes at 25 calibrations.')
    for scenario, prior in unknown_type.PRIORS.items():
        np.testing.assert_array_equal(prior, old_type.PRIORS[scenario])
        for h0 in ['0', 'j']:
            new = unknown_type.solve(h0, prior)
            old = old_type.solve(h0, prior)
            for a, b in zip(new[:2], old[:2]):
                pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b), check_exact=False, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(new[2], old[2], rtol=1e-12, atol=1e-12)
    print('Matched all unknown-type prior/state combinations, posteriors, prices, probabilities and profits.')


if __name__ == '__main__':
    original = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'final_draft'
    with patch.object(pd, 'read_csv', side_effect=AssertionError('Empirical CSV reads are forbidden.')):
        compare(original.resolve())
    print('All pricing comparisons passed using synthetic data only.')

"""Compare simplified calculations with final_draft using generated data only.

Optional: python3 checks/compare_original.py /path/to/final_draft
The research files do not depend on final_draft; this verification does.
No CSV data are read. Original files are imported as calculation references.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from demand import estimation
from demand import pricing
from supply import supply_simulation


def load_reference(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(original):
    sys.path.insert(0, str(original / 'est_code'))
    old_estimation = load_reference('original_estimation', original / 'est_code/est.py')
    old_pricing = load_reference('original_pricing', original / 'pricing_exercises/run_empirical_targeting.py')
    old_supply = load_reference('original_supply', original / 'pricing_exercises/run_supply_exercises.py')
    rng = np.random.default_rng(53)
    rows = []
    for household in range(8):
        for week in range(4):
            counts = rng.multinomial(3, [.2, .4, .2, .2])
            if counts[:3].sum() == 0:
                counts[1], counts[3] = 1, 2
            row = dict(household_code=str(household), week=pd.date_range('2010-01-02', periods=4, freq='7D')[week], outside=counts[3])
            for j, flavor in enumerate(estimation.FLAVORS):
                row[flavor] = float(counts[j])
                row['price_'+flavor] = 1 + rng.uniform(0, 1)
                row['cf_'+flavor] = rng.normal(0, .1)
            rows.append(row)
    raw = pd.DataFrame(rows)
    frame = estimation.add_stocks(raw)
    pd.testing.assert_frame_equal(frame, old_estimation.add_stocks(raw))
    models = ['naive', 'homogeneous', 'mixed_naive', 'heterogeneous', 'mixture', 'separate_history', 'heterogeneous_history']
    likelihood_checks = 0
    for capacity in [0, 7]:
        for controlled in [False, True]:
            inputs = frame if controlled else frame.drop(columns=['cf_'+f for f in estimation.FLAVORS])
            data = estimation.make_arrays(inputs, capacity)
            old_data = old_estimation.make_arrays(inputs, capacity)
            for model in models:
                names = estimation.parameter_names(model, bool(capacity), controlled)
                assert names == old_estimation.parameter_names(model, bool(capacity), controlled)
                theta = np.full(len(names), .2)
                draws = estimation.make_draws(16, 306, model)
                np.testing.assert_array_equal(draws, old_estimation.make_draws(16, 306, model))
                np.testing.assert_allclose(estimation.household_loglike(theta, data, draws, model), old_estimation.household_loglike(theta, old_data, draws, model), rtol=1e-12, atol=1e-12)
                value, gradient = estimation.demand_objective_gradient(theta, data, draws, model)
                old_value, old_gradient = old_estimation.demand_objective_gradient(theta, old_data, draws, model)
                np.testing.assert_allclose(value, old_value, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(gradient, old_gradient, rtol=1e-12, atol=1e-12)
                likelihood_checks += 1
    print(f'Matched likelihoods and analytic gradients: {likelihood_checks} model/capacity/control combinations.')

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

    for gamma in [-2., 0., 2.]:
        for prior in [.25, .5, .75]:
            new = supply_simulation.supply_solution(gamma, prior)
            old = old_supply.supply_solution(gamma, prior)
            assert new.keys() == old.keys()
            for key in new:
                np.testing.assert_allclose(new[key], old[key], rtol=1e-12, atol=1e-12)
            for actual, expected in zip(supply_simulation.outcome_tables(new), old_supply.outcome_tables(old)):
                pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected))
    print('Matched all three theoretical supply cases and outcome tables at nine gamma/prior calibrations.')


if __name__ == '__main__':
    original = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'final_draft'
    def no_csv(*args, **kwargs):
        raise AssertionError('CSV reads are forbidden in this comparison.')
    with patch.object(pd, 'read_csv', side_effect=no_csv):
        compare(original)
    print('Original-vs-simple comparisons passed without CSV reads.')

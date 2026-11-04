# Pricing exercises in final_simple

Edit the Parameters section near the top of each script, then run it.
Optional command-line flags override these variables for server jobs.
All research scripts run independently of final_draft.

| final_draft/pricing_exercises | final_simple counterpart |
|---|---|
| run_supply_exercises.py, SUPPLY_EXERCISES.md | supply/supply_simulation.py; original supply equations and cases |
| run_empirical_targeting.py, EMPIRICAL_TARGETING.md | demand/pricing.py; household offers and Bayesian sensitivity |
| unknown_type/run_unknown_type.py | supply/unknown_type.py |
| run_pricing_exercises.py | Run the task-specific scripts directly |
| test_empirical_timing.py | checks/test_empirical_timing.py |
| README.md | This guide and the project README |

The unknown-gamma script uses supply/pricing_model.py, which contains the same binary
supply calculations as supply_simulation.py without symbolic derivative imports.
The shared file has no editable research parameters; choose them in the entry
points. This keeps the information exercises runnable with NumPy, pandas and
matplotlib alone.

```bash
python3 supply/unknown_type.py
python3 supply/unknown_type.py --self-test
python3 checks/compare_pricing_exercises.py /path/to/final_draft
```

## Gamma uncertainty: one policy against a mixture

The firm observes h0. It never observes gamma. The support is {-1.25,0,+1.25};
priors are (.5,.25,.25), (.25,.5,.25), (.25,.25,.5), and exact equal thirds.
For each prior and known h0 it chooses one p1 and two p2 prices, conditional on
observed h1. Choices update beliefs about gamma; prices do not condition on its
true value. Profit is maximized after averaging over possible types.

`results/unknown_type/optimal_policies.csv` reports all eight solutions and
prior probabilities directly. One PNG/PDF page per prior shows the gamma
distribution, optimal prices for each known h0, and expected profit. The output
has only the unknown-gamma exercise. Known-type and commitment comparisons are
not output cases. The unrelated extra initial-state runner has been withdrawn.

Numerical tables and graphs match final_draft. Source syntax, paths, hashes,
and readable run notes may differ. The original baseline supply and empirical
scripts remain available; no empirical dataset is guessed or re-estimated.

## Additional supply information figures

Run `python3 supply/supply_information.py`. Edit its Parameters section to change
the probability-grid size or calibration. It adds six horizontal three-panel
figures with the original supply_information layout. The support stays fixed;
the probability of each gamma is varied in turn, splitting the remainder equally.
Each graph conditions on a known h0. Panels show optimal prices, expected profit,
and the value of observing h1. Revealed-gamma and commitment benchmarks enter
only these information comparisons. The four earlier policy pages are retained.

`results/unknown_type/supply_information.png` varies Pr(gamma=+1.25) with known
h0=outside. Other files identify focal gamma and h0 in their names. The CSV
`supply_information_results.csv` contains all plotted numerical values.

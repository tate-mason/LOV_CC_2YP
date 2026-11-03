# Start here

This is a self-contained, simpler version of the data, demand, and supply code in
`final_draft`. The layout follows `LOV_bundle_intro.py`: imports, editable
parameters, numbered calculation sections, and the run steps at the bottom.
The scientific calculations and model options are preserved. The original
research directory has not been replaced.

The code is organized by research task:

```text
final_simple/
    data/
        build_data.py          # raw conversion, cleaning, and all merges
        summaries.py           # demographics, flavors, HHI, and demand input
    demand/
        estimation.py          # demand estimation and inference
        pricing.py             # empirical dynamic berry offers
    supply/
        supply_sympy.py        # equations and exact derivatives
        supply_simulation.py   # numerical solutions and figures
    checks/
    README.md
    requirements.txt
```

Each research script has editable parameters at the top and visible run steps at
the bottom. `supply/supply_simulation.py` imports its formulas from `supply/supply_sympy.py`;
the other research calculations are contained in their own files. None imports
from `final_draft`. Package `__init__.py` files only identify the folders; they
contain no research calculations or settings.

Start with [the data guide](data/README.md) for a full rebuild. The usual order is
**data/build_data.py → data/summaries.py → demand/estimation.py → demand/pricing.py**.
The theoretical supply exercises run separately from the empirical data pipeline.

`checks/` contains verification, not additional research steps. You can leave
that folder alone unless you want to run checks or change the mathematics.
`requirements.txt` lists the external Python packages.

## How to work with a script

1. Open its **Parameters: edit these values** section near the top.
2. Change the ordinary Python variables you need, as in your simulation example.
3. Read the corresponding `run_build`, `run_summaries`, `run_estimation`,
   `run_pricing`, or `run_supply` function near the
   bottom to see the exact order of research steps.
4. Run the file. Function definitions above that section explain each step.

For example, in `demand/estimation.py`:

```python
sample = Path("/your/data/yogurt_estimation_sample.parquet")
output = Path("/your/results/development")
model = "heterogeneous"
max_households = 100
draws = 32
starts = 1
se_method = "none"
```

Then run:

```bash
python3 demand/estimation.py
```

A smaller sample and fewer draws are development settings. For a research fit,
choose your full sample, check convergence, and increase draws/starts as needed.
Those settings affect what is estimated and how accurately it is solved.

Optional command-line flags remain available for server jobs. For example,
`python3 demand/estimation.py --draws 128` overrides the Python variable for that run.
You do not have to use flags: editing the Parameters section is sufficient.
`python3 demand/estimation.py --help` lists all available overrides.

The `if __name__ == "__main__":` line at the bottom means “run these steps when
I execute this file.” Importing a function for a check does not run estimation.
`SimpleNamespace` is just a record bundling the variables you edited:
`settings.draws` means your chosen `draws`, not a different hidden setting.

## What the estimator contains

The sections cover price controls; input preparation; histories; probabilities
and likelihood; parameter optimization; standard errors; economic interpretation;
and outputs. Nothing is estimated by the pricing script.

| `model` setting | Specification |
|---|---|
| `naive` | Common price coefficient; no history coefficient |
| `homogeneous` | Common price coefficient and gamma |
| `mixed_naive` | Household-specific price coefficient; no gamma |
| `heterogeneous` | Household-specific price coefficient and gamma |
| `mixture` | Two-component gamma population; heterogeneous price sensitivity |
| `separate_history` | Separate common novelty/stock effects; heterogeneous price sensitivity |
| `heterogeneous_history` | Separate heterogeneous novelty/stock effects and price sensitivity |

`se_method` can be `hessian`, `bootstrap`, or `none`. Hessian inference holds
estimated price controls fixed; the household bootstrap refits price controls
and demand. These are different inference assumptions, not interchangeable labels.

The original default `capacity = 0` estimates flavor choice conditional on
buying yogurt. `capacity = 7` adds the constructed outside count and estimates
participation under that convention. The empirical pricing exercise requires
a converged `heterogeneous` fit with positive integer capacity from 1 through 12.
To use capacity 7 for pricing, estimate that model with capacity 7; do not simply
change the saved fit's metadata. Each model/capacity should have its own output
folder.

The prepared input is the Parquet purchase table produced by
`data/summaries.py`. Raw conversion, cleaning, merging, flavor recovery, and
summaries are included in `data/`. The optional legacy flavor-reference CSV
reader and optional product metadata recovery remain available.

## What empirical pricing does

Set `household_input` to a saved `weekly_inputs.csv` and `fit_dir` to the
corresponding estimation output folder. Both start as `None` so importing or
running the script does not guess a dataset. For example:

```python
household_input = Path("/your/results/outside_model/weekly_inputs.csv")
fit_dir = Path("/your/results/outside_model")
```

Then run `python3 demand/pricing.py`.

Timing follows the discussion of the theoretical model:

- The first observed basket is relabeled from observed t=1 to model t=0 and
  establishes initial history h0.
- The firm chooses p1(h0); the resulting basket establishes h1.
- The firm chooses terminal p2(h1), conditional on the realized basket and
  estimated household type.

The first observed menu, posted prices, and price-control residuals are carried
forward. Type estimation still uses the complete observed estimation sample.
The code checks that the supplied records reproduce the saved likelihood.
This is a retrospective counterfactual with known estimated types.

The main specification uses household posterior means. The optional
`posterior-integrated` setting retains uncertainty about type and updates its
weights after the first basket. No-coupon prices provide a revenue benchmark;
there is only one optimized policy, dynamic pricing.

Only the low/high gamma quartiles receive personalized offers. The middle half
receives no coupons. Each eligible household is optimized separately; quartiles
are reporting/eligibility groups, not pooled pricing groups. With capacity 7,
all 120 possible baskets enter the expectation. Terminal offers are selected
before their revenues are averaged over histories. The firm earns berry revenue. The revenue graph shows one bar per type for
first-period revenue plus 0.95 times expected terminal revenue by default.
`demand_delta` controls both optimization and the graph's terminal discounting.
Type colors match the distribution chart. Offers are chosen to maximize the
specified revenue objective; the labels do not claim that switching or retention
was empirically demonstrated.

## What theoretical supply does

The numerical script includes the observed-initial-state baseline, hidden
initial state, and commitment to both prices. It retains both terminal branches,
unrestricted theoretical prices, marginal costs, history-specific price ratios,
information/flexibility values, and the original comparative statics.

Run `python3 supply/supply_sympy.py` to print equations without running a numerical grid.
Run `python3 supply/supply_simulation.py` to solve the cases, draw the figures, and write
the benchmark appendix as LaTeX source. A LaTeX compiler is separate; this code
does not automatically invoke one.

Set `skip_derivatives = True` to run just the three supply cases. For a custom
calibration or derivative grid, set `no_appendix = True`: the written appendix
contains benchmark-specific statements. The derivative figures still use your
chosen parameters. The theoretical binary switching model and empirical
three-flavor stock model retain their different utility specifications; the
refactor does not substitute one for the other.

## Reading outputs without learning JSON

Demand and supply outputs go under your selected `output` path. Their default
is a project-level `results/` folder with a separate folder for each exercise.
The raw data build writes under your selected `data_dir`; summaries default to
`data_dir/summary_outputs`. These locations are explicit in each Parameters section.

Start with:

- `run_notes.txt`: readable settings, convergence, and diagnostics.
- `estimates.csv` / `point_estimates.csv`: fitted coefficients and inference.
- The PDF figures: the model's plots.
- `weekly_inputs.csv`: household-week counts, prices, and histories for auditing.

`fit.json` is an automatic estimation record. `demand/pricing.py` reads it automatically
to learn which model and settings produced the estimates. `run_manifest.json`
records a pricing/supply run. You do not need to write or edit either JSON file.
Keep the automatic records with their output tables so scripts can verify that
the pieces match. `starts_checkpoint.json` records optimizer progress; it is
not an automatic resume instruction.

Use different output folders for different specifications. Reusing a folder
can leave earlier artifacts whose names are no longer relevant to the new run.

## Checks and development

Install the listed packages in your usual environment if needed:

```bash
python3 -m pip install -r requirements.txt
```

Run the synthetic checks from this directory:

```bash
python3 checks/run_checks.py
```

This runner blocks CSV reads and uses generated arrays/temporary synthetic
Parquet data. Some checks write temporary output tables without reopening them.
Two legacy checks that reopen generated CSVs remain in `checks/`, but are
excluded from this runner. The original complete export self-test is also still
available in `demand/pricing.py --self-test`; it creates and rereads temporary synthetic
CSVs, so it was not used during this no-CSV development work.

This refactor does not re-estimate the empirical model or claim new empirical
results. See `checks/VALIDATION.md` for the checks actually completed.

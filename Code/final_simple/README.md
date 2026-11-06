# Start here

This is a self-contained, simpler version of the data, demand, and supply code in
`final_draft`. The layout follows `LOV_bundle_intro.py`: imports, editable
parameters, numbered calculation sections, and the run steps at the bottom.
The scientific calculations and model options are preserved.
The current empirical setup uses separate novelty and habit coefficients; start
with [the split-model reading guide](demand/README.md) for the revised workflow.
For the full teaching text, open [Python 101](../final_draft/code_guides_pdf/python101_walkthrough.pdf):
worked lessons 11–16 cover the current split model and server workflow, followed
by annotated chapters for these readable demand scripts. The original
research directory has not been replaced.

The code is organized by research task:

```text
final_simple/
    data/
        build_data.py          # raw conversion, cleaning, and all merges
        summaries.py           # demographics, flavors, HHI, and demand input
    demand/
        estimation.py          # demand estimation and inference
        plain_berry_pricing.py # main comparison: plain and berry focal sellers
        split_pricing.py       # split model; all three focal flavors
        regenerate_weekly_inputs.py # server: rebuild the deleted weekly export
        history_distributions.py    # export distributions from saved split fits
        pricing.py             # earlier combined-gamma berry comparison
    supply/
        supply_sympy.py        # equations and exact derivatives
        supply_simulation.py   # numerical solutions and figures
    server/run_split_pricing.sh # regenerate, export distributions, compare plain/berry
    checks/
    README.md
    requirements.txt
```

Each research script has editable parameters at the top and visible run steps at
the bottom. `supply/supply_simulation.py` imports its formulas from `supply/supply_sympy.py`;
the estimation and pricing calculations are contained in their own files.
The distribution-export and weekly-regeneration entry points reuse functions
from `demand/estimation.py`. None imports from `final_draft`. Package `__init__.py` files only identify the folders; they
contain no research calculations or settings.

Start with [the data guide](data/README.md) for a full rebuild. The usual order is
**data/build_data.py → data/summaries.py → demand/estimation.py → demand/split_pricing.py**.
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
model = "heterogeneous_history"
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

`capacity = 0` estimates flavor choice conditional on buying yogurt. The current
split-model default, `capacity = 7`, adds the constructed outside count and estimates
participation under that convention. The empirical pricing exercise requires
a converged `heterogeneous_history` fit with positive integer capacity from 1 through 12.
The earlier `pricing.py` remains available for combined-gamma fits.
To use capacity 7 for pricing, estimate that model with capacity 7; do not simply
change the saved fit's metadata. Each model/capacity should have its own output
folder.

The prepared input is the Parquet purchase table produced by
`data/summaries.py`. Raw conversion, cleaning, merging, flavor recovery, and
summaries are included in `data/`. The optional legacy flavor-reference CSV
reader and optional product metadata recovery remain available.

## What empirical pricing does

The current script is `demand/split_pricing.py`. Set `household_input` and
`fit_dir` in its Parameters section; neither path is guessed. It reads the
existing split-model fit, recovers persistent household price, novelty, and
habit types, and verifies the saved likelihood before running pricing.

It solves three separate seller problems: other, berry, and plain each serve
as the focal inside product against the other two flavors combined. The outside
alternative also includes constructed nonpurchase for positive-capacity fits.
Only the focal price changes; rival prices stay fixed. These are separate
counterfactuals, not a simultaneous equilibrium.

Bottom/top novelty quartiles receive household-specific offers, while the
middle group remains at posted prices. Both novelty and habit coefficients
enter every demand calculation. Timing is observed h0 → p1(h0) → realized
basket h1 → terminal p2(h1), with exact basket integration and backward
induction. The default treats posterior means as known; posterior uncertainty
integration is optional. Coupons range from 0% through 100% in 1% steps.
The revenue discount factor is .92, matching the theoretical supply default.

Results are saved in `other/`, `berry/`, and `plain/` folders, with a root
`focal_flavor_comparison.csv`, `run_manifest.json`, and readable `run_notes.txt`.
Read [the demand guide](demand/README.md) for distribution definitions, the
inside/outside mapping, assumptions, output columns, and server instructions.

Deleted `weekly_inputs.csv` should be regenerated **on the server** with
`demand/regenerate_weekly_inputs.py`. This rebuilds histories, menus and price
controls from the saved settings, verifies the original likelihood and sample
counts, then writes the export. It retains fitted demand estimates and inference.
`server/run_split_pricing.sh` does this before exporting distributions and
running all three focal cases. Local checks access generated inputs only.

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
excluded from this runner. New split-pricing and export-regeneration checks are included in the no-CSV
runner. The original complete export self-test is also still
available in `demand/pricing.py --self-test`; it creates and rereads temporary synthetic
CSVs, so it was not used during this no-CSV development work.

This refactor does not re-estimate the empirical model or claim new empirical
results. See `checks/VALIDATION.md` for the checks actually completed.

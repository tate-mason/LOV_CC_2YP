# Revised information comparison: habit retained, only novelty hidden

- Added a new pricing-only server runner, pure numerical solver and aggregate-only
  supply-style plotting pipeline. Existing demand estimates and saved history
  construction remain unchanged; no empirical data were accessed locally.
- Both regimes adapt terminal prices to the realized purchase basket. The firm
  knows price sensitivity and habit in both; only novelty information differs.
  Prices can rise or fall; all novelty groups are optimized.
- Main setup now states this empirical extension. Earlier binary and shared-schedule
  numerical results are in backup and are not described as the new results.
- Added an aggregate import command and conditional result slides. Until server
  output is imported, the main deck explicitly marks the new results as pending.
- Supply-style figure templates include optimal prices by gamma, price ratios and
  conditional profit, type/common price paths, information gains, and changes
  relative to observed prices. Consumer type persists during revenue evaluation.
- Pure tests cover analytic limits, direct enumeration, menu expansion, fixed
  beliefs, history updates, grid refinement and full-information dominance.
  All seven empirical entry points remain blocked locally by policy checks.

Earlier entries below document prior iterations and their then-current results.

# Price paths restored to the main sequence

The calibrated gamma curves are main slide 28 in the seminar deck, followed by
a dedicated type-specific/common price-path slide 29. The paths separate low,
middle and high types and initially inside/outside histories. The zoom remains
in the appendix. Both short sources retain their distinct introductions. The
seminar deck now has 35 main slides, the title, and 23 appendix slides (59 pages).
No numerical results changed.

# Calibrated gamma-price plots

Replaced the calibrated plain expected price-path main slide with two optimal
price-versus-gamma panels matching the theory format. Saved plain calibration
parameters generate the curves; markers and shading identify the three estimated
group means. Added appendix links to a zoom and the retained price paths.
All six exported type/state paths match; terminal and initial FOCs pass over
the full gamma grid. No empirical I/O or server rerun was required. The canonical
short deck now has 34 main slides, the title, and 24 appendix slides (59 pages).

# November 11: aligned theory and calibrated pricing sections

- Both sections now use known-type personalized offers versus hidden-type common
  offers with fixed beliefs, evaluated on the same persistent consumer types.
- Corrected the theoretical common-policy revenue evaluation. Original policies
  and fixed-belief objectives match the saved results exactly; all eight
  state/prior comparisons now report persistent-type revenue differences.
- Rebuilt the plain information figures and price paths from the named
  `Results/pricing_no_disc` aggregates, with independent numerical reconstruction
  of all eight cells. No empirical estimates or policies were rerun or overwritten.
- The binary illustration, its 32-history price calibration, and the separate
  full-demand coupon exercise are explicitly distinguished. The estimated middle
  type is not zero. The coupon result remains +7.23% for plain.
- Updated both short-deck pricing sections, retaining their distinct introductions.
  The canonical deck has 34 main slides plus title and 22 appendix slides.
- Added `build_aligned_pricing.py`, aggregate asset provenance, and
  `PRICING_SPEAKER_NOTES.md`. New average-stock demand remains a pending server
  refit; the presentation correctly labels the saved cumulative-stock estimates.
- Both PDFs compile; revised pricing slides were visually inspected. Appendix
  links and frame-label uniqueness pass. Fixed two pre-existing build problems:
  duplicate introduction labels and list items outside a list environment.

The remaining notes below record earlier revisions and their then-current inputs.

# Pricing and presentation changes

The updated presentation is `Code/final_draft/seminar_pres/pres_short.pdf`, with an identical PDF in `Code/final_draft/writeup/shortened_presentation/pres_short.pdf`.

## Presentation

- Inserted **Optimal Price Ratios and Profit** immediately after **Optimal Prices**, with price ratios on the left and realized-history profit on the right. It is main slide 10 (PDF page 11).
- Regenerated known-type theory figures, the theory calibration table, and fixed-belief type-information figures at **delta = 1**. The original type probabilities are retained: equal thirds and the three permutations of (0.50, 0.25, 0.25).
- Updated both `pres_short.tex` copies, figure references, navigation, main-slide count (33), and reading guides. The rebuilt short deck has 64 pages including its title and appendix.
- Standardized public labels to **Common pricing** and **Personalized pricing**. Removed the displayed no-coupon policy. Price-path charts show **p1 → E[p2]**, with no separate conditional terminal-price series. The axis label is simply p2.
- Updated main-deck and appendix pricing graphics, information comparisons, and tables. Full-deck `pres.tex` figure references and labels were updated as well; the rebuilt PDFs are the short decks.
- Rebased the empirical policy comparison and conclusion onto common pricing: **9.48% for plain** and **7.26% for berry**, using the already recovered aggregate results. The former 13.35% plain figure compared against posted prices and is no longer displayed.

## Pricing code

- Added `Code/pricing_presentation.py` as a shared presentation layer for current and historical pricing runners. It centralizes labels, policy filtering, price paths, comparisons, and PNG/PDF output.
- Added `refresh_pricing_slides.py`, which reads only three explicitly named aggregate CSVs per flavor and creates presentation assets. It records provenance and never reads household panels, purchases, or weekly inputs.
- Updated the active split-history pricing solver to collect common-pricing revenue by type group at the same optimal common prices. This adds aggregate accounting, not a different demand model or optimization objective. An identity checks that group revenues sum to the original common-policy revenue grid.
- Public active-run policy tables now contain common and personalized pricing; reported gains use common pricing as the reference. The posted-price calculation remains an internal feasibility/dominance diagnostic.
- Standardized plotting functions and displayed labels across legacy/snapshot pricing runners and independent toy-model figure builders. Legacy solvers that do not estimate a common policy display their personalized path without pretending posted prices are optimized common pricing.
- Retained existing figure filenames as compatibility aliases where slide and export code already uses them. Their displayed titles and labels use the new wording.

## Flavor and outside

- Active plain/berry pricing uses `flavor`, `--flavor`, `flavor_comparison.csv`, and a `flavor` comparison column.
- Flavor means the selected plain or berry category. Outside means **everything except that flavor**, including other yogurt categories and nonpurchase. P(outside) = 1 − P(flavor).
- Full multinomial alternatives and basket-history states remain intact. The demand estimator's normalized nonpurchase alternative remains distinct from this broader pricing outside.
- Updated the active server comparison/export code and allowlist to match the new names and figures.

## No time discounting

- Changed production pricing/theory defaults and relevant server launchers to **delta = 1**. Updated validation and mathematical assumptions to admit that boundary.
- Updated matching production defaults in older pricing/theory and toy-model entry points. Explicit nonunit values in sensitivity tests remain available.
- The active theoretical figure runner now uses the existing shared monotone scalar price solver, avoiding the broken local SciPy optimizer import while retaining the same first-order condition.

## Demand notation

- Documented the distinction between persistent normal draws `z`, the transitory T1EV utility shock epsilon, and the first-stage price residual nu.
- Persistent draws determine alpha/gamma/phi and stay fixed through the household history. Epsilon is integrated analytically to obtain logit probabilities. In the price-control notation, nu is the price-equation error and its estimated residual enters only controlled demand fits; the current uncontrolled fits omit it.

## Validation and limits

- All changed/new Python sources parse; modified shell launchers pass `bash -n`; source whitespace checks pass.
- Original persistent draws, posterior, basket-state construction, and forecast functions match the original code by AST comparison. Original pricing utility, probability, revenue-grid, and action-selection assignments also match.
- Theory checks passed for probabilities, profit reconstruction, first-order conditions, limiting cases, and degenerate priors. Independent scalar price solvers agree at delta = 1. The 1,202 known-state theory grid solutions have maximum first-order-condition residual approximately 7.77e-16; conditional profits reconstruct expected profits.
- Fixed-belief theory checks passed: wider price searches, independent terminal maximization, finite-difference first-order conditions, unchanged beliefs, full-information bounds, and degenerate priors.
- The short presentation compiled successfully. Its text was checked for the new policy names, removed baseline labels, corrected gains, and placement of the ratio/profit slide. The ratio/profit and empirical price-path slides were inspected visually.
- **Empirical estimates and optimal policies were not rerun locally.** The restyled empirical presentation assets use recovered aggregate results from the original **delta = 0.92** pricing run. `assets/pricing/presentation_provenance.json` records this explicitly. New server runs use delta = 1; fresh empirical results still require research-server execution.
- The general empirical numerical and privacy-checker entry points could not start because the local SciPy binary fails to import. No empirical records were opened, generated, copied, or downloaded.
- Source changes were made without changing Git history. Unrelated files and deletions in the workspace were left alone.

## Modified source files

- `Code/final_draft/RESULTS_ONLY/pricing/focal_flavor_comparison.tex`
- `Code/final_draft/pricing_exercises/run_empirical_targeting.py`
- `Code/final_draft/pricing_exercises/run_split_targeting.py`
- `Code/final_draft/pricing_exercises/run_supply_exercises.py`
- `Code/final_draft/pricing_exercises/unknown_type/plot_supply_information.py`
- `Code/final_draft/pricing_exercises/unknown_type/run_unknown_type.py`
- `Code/final_draft/seminar_pres/CHANGES_pricing_plots.md`
- `Code/final_draft/seminar_pres/make_policy_figures.py`
- `Code/final_draft/seminar_pres/pres.tex`
- `Code/final_draft/seminar_pres/pres_short.tex`
- `Code/final_draft/seminar_pres/pres_short_README.md`
- `Code/final_draft/seminar_pres/refresh_pricing_slides.py`
- `Code/final_draft/supply_code/make_comparative_static_figures.py`
- `Code/final_draft/supply_code/make_latex_document.py`
- `Code/final_draft/supply_code/model/assumptions.py`
- `Code/final_draft/supply_code/model/numerical.py`
- `Code/final_draft/supply_code/structural_switching_reference.py`
- `Code/final_draft/writeup/1101_writeup/build_assets.py`
- `Code/final_draft/writeup/1108_writeup/assets/results/empirical_information/build_from_aggregates.py`
- `Code/final_draft/writeup/1108_writeup/assets/results/pricing/benchmarks.tex`
- `Code/final_draft/writeup/1108_writeup/assets/results/pricing/focal_flavor_comparison.tex`
- `Code/final_draft/writeup/1108_writeup/draft.tex`
- `Code/final_draft/writeup/1108_writeup/theory/gamma_uncertainty.py`
- `Code/final_draft/writeup/documentation_revision/Code/final_simple/checks/reference_supply.py`
- `Code/final_draft/writeup/documentation_revision/Code/final_simple/demand/plain_berry_pricing.py`
- `Code/final_draft/writeup/documentation_revision/Code/final_simple/demand/pricing.py`
- `Code/final_draft/writeup/documentation_revision/Code/final_simple/demand/split_pricing.py`
- `Code/final_draft/writeup/documentation_revision/Code/final_simple/supply/pricing_model.py`
- `Code/final_draft/writeup/documentation_revision/Code/final_simple/supply/supply_information.py`
- `Code/final_draft/writeup/documentation_revision/Code/final_simple/supply/supply_simulation.py`
- `Code/final_draft/writeup/documentation_revision/Code/final_simple/supply/supply_sympy.py`
- `Code/final_draft/writeup/documentation_revision/Code/final_simple/supply/unknown_type.py`
- `Code/final_draft/writeup/shortened_presentation/build_short_version.py`
- `Code/final_draft/writeup/shortened_presentation/pres_short.tex`
- `Code/final_draft/writeup/shortened_presentation/pres_short_README.md`
- `Code/final_draft/writeup/two_spec_revision/Code/final_simple/demand/plain_berry_pricing.py`
- `Code/final_draft/writeup/two_spec_revision/Code/final_simple/demand/split_pricing.py`
- `Code/final_draft/writeup/uncontrolled_pricing_revision/plain_berry_pricing.py`
- `Code/final_draft/writeup/uncontrolled_pricing_revision/split_pricing.py`
- `Code/final_simple/README.md`
- `Code/final_simple/checks/reference_supply.py`
- `Code/final_simple/checks/test_split_pricing.py`
- `Code/final_simple/demand/README.md`
- `Code/final_simple/demand/gamma_calibration.py`
- `Code/final_simple/demand/plain_berry_pricing.py`
- `Code/final_simple/demand/pricing.py`
- `Code/final_simple/demand/split_pricing.py`
- `Code/final_simple/server/README.md`
- `Code/final_simple/server/export_uncontrolled_pricing.py`
- `Code/final_simple/server/run_after_data.py`
- `Code/final_simple/server/run_split_pricing.sh`
- `Code/final_simple/supply/pricing_model.py`
- `Code/final_simple/supply/supply_information.py`
- `Code/final_simple/supply/supply_simulation.py`
- `Code/final_simple/supply/supply_sympy.py`
- `Code/final_simple/supply/unknown_type.py`
- `Code/full_draft_1231/toy_model.py`
- `Code/full_draft_1231/updates/draft_2/presentation/build_pricing_figures.py`
- `Code/full_draft_1231/updates/toy_model/archive_unrestricted/toy_model.py`
- `Code/full_draft_1231/updates/toy_model/fixed_regular_price_variant/build_pricing_figures.py`
- `Code/full_draft_1231/updates/toy_model/fixed_regular_price_variant/toy_model.py`
- `Code/pricing_presentation.py`
- `Code/privacy/export_rebuild_results.py`
- `shell_files/est_split_pricing.sh`
- `shell_files/full_rebuild.sh`

## Generated presentation assets

- `Code/final_draft/seminar_pres/assets/pricing/berry/discounts_by_consumer_type.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/discounts_by_consumer_type.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/discounts_by_household_type.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/discounts_by_household_type.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/household_coupon_results.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/household_coupon_results.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_first_prices.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_first_prices.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_information_value.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_information_value.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_price_paths.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_price_paths.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_profits.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_profits.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_terminal_prices.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_friction_terminal_prices.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_theory_flavor.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_theory_flavor.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_theory_outside.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/information_theory_outside.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/price_paths.csv`
- `Code/final_draft/seminar_pres/assets/pricing/berry/pricing_mechanism.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/pricing_mechanism.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/pricing_policy_benchmarks.csv`
- `Code/final_draft/seminar_pres/assets/pricing/berry/pricing_policy_benchmarks.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/pricing_policy_benchmarks.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/pricing_theory_comparison.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/pricing_theory_comparison.png`
- `Code/final_draft/seminar_pres/assets/pricing/berry/terminal_offers_by_purchase.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/berry/terminal_offers_by_purchase.png`
- `Code/final_draft/seminar_pres/assets/pricing/flavor_comparison.csv`
- `Code/final_draft/seminar_pres/assets/pricing/flavor_comparison.tex`
- `Code/final_draft/seminar_pres/assets/pricing/plain/discounts_by_consumer_type.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/discounts_by_consumer_type.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/discounts_by_household_type.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/discounts_by_household_type.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/household_coupon_results.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/household_coupon_results.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_first_prices.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_first_prices.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_information_value.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_information_value.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_price_paths.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_price_paths.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_profits.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_profits.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_terminal_prices.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_friction_terminal_prices.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_theory_flavor.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_theory_flavor.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_theory_outside.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/information_theory_outside.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/price_paths.csv`
- `Code/final_draft/seminar_pres/assets/pricing/plain/pricing_mechanism.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/pricing_mechanism.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/pricing_policy_benchmarks.csv`
- `Code/final_draft/seminar_pres/assets/pricing/plain/pricing_policy_benchmarks.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/pricing_policy_benchmarks.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/pricing_theory_comparison.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/pricing_theory_comparison.png`
- `Code/final_draft/seminar_pres/assets/pricing/plain/terminal_offers_by_purchase.pdf`
- `Code/final_draft/seminar_pres/assets/pricing/plain/terminal_offers_by_purchase.png`
- `Code/final_draft/seminar_pres/assets/pricing/presentation_provenance.json`
- `Code/final_draft/seminar_pres/assets/theory_delta1/all_state_outcomes.csv`
- `Code/final_draft/seminar_pres/assets/theory_delta1/all_state_price_ratio_derivatives.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/all_state_price_ratio_derivatives.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/all_state_price_ratios.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/all_state_price_ratios.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/all_state_realized_profit.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/all_state_realized_profit.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/all_state_shares.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/all_state_shares.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_p1_by_gamma.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_p1_by_gamma.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_profit_by_gamma.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_profit_by_gamma.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/first_prices.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/first_prices.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_first_prices.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_first_prices.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_information_value.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_information_value.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_price_paths.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_price_paths.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_profits.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_profits.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_terminal_prices.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_friction_terminal_prices.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_theory_flavor.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_theory_flavor.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_theory_outside.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_theory_outside.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_value.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/information_value.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/profits.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/profits.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/results.tex`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/settings.json`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/solutions.json`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/terminal_prices.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/gamma_uncertainty/terminal_prices.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/induce_harvest_price_ratio_derivative.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/induce_harvest_price_ratio_derivative.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/matched_price_ratio_derivative.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/matched_price_ratio_derivative.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/optimal_prices_by_gamma.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/optimal_prices_by_gamma.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/profit_by_gamma.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/profit_by_gamma.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/solutions.csv`

## Rebuilt PDFs and TeX auxiliary files

- `Code/final_draft/seminar_pres/pres_short.pdf`
- `Code/final_draft/writeup/shortened_presentation/pres_short.aux`
- `Code/final_draft/writeup/shortened_presentation/pres_short.log`
- `Code/final_draft/writeup/shortened_presentation/pres_short.nav`
- `Code/final_draft/writeup/shortened_presentation/pres_short.pdf`
- `Code/final_draft/writeup/shortened_presentation/pres_short.snm`
- `Code/final_draft/writeup/shortened_presentation/pres_short.toc`

## Follow-up: probability notation and duplicate slide

- Defined `Pr(gamma=gamma_r)=mu_r`, for r in {H,0,L}, in theory and empirical information slides. Equal-case display is `mu_H=mu_0=mu_L=0.33`; calculation keeps exact thirds. Defined gamma_L=-1.25, gamma_0=0, gamma_H=1.25.
- Updated the type-probability plot source to label bars with mu_L, mu_0 and mu_H, displaying two-decimal probabilities.
- Removed duplicate main Plain: Price Paths frame short-43; retained short-42. The deck now has 33 main slides and 64 pages total.

## Follow-up: mu on plot axes and supply regeneration

- Prior-comparison pricing and supply axes display the dominant probability (mu_L=0.5, mu_0=0.5, or mu_H=0.5), or mu_r=0.33 for the equal case, with Pr(gamma=gamma_r)=mu_r as the axis label. Exact thirds remain in computation.
- Supply probability sweeps now label the x-axis mu_L, mu_0, or mu_H as appropriate, with the probability definition. The 0.33 tick represents exact one-third.
- Regenerated known-type supply figures, fixed-belief figures, all four hidden-type prior figures, and six probability-sweep figures at delta=1. Hidden-type runs passed denser-grid checks; probability sweeps passed policy-ordering and denser-grid checks.
- Refreshed both short-deck PDFs and the standard active supply output folders. Empirical graphics remain a styling refresh of recovered aggregates, not an empirical delta=1 rerun.

Additional regenerated supply assets:

- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_manifest.json`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_negative_h0_inside.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_negative_h0_inside.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_negative_h0_outside.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_negative_h0_outside.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_positive_h0_inside.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_positive_h0_inside.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_results.csv`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_zero_h0_inside.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_zero_h0_inside.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_zero_h0_outside.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/information_sweeps/supply_information_zero_h0_outside.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/optimal_policies.csv`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/optimal_policy_equal.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/optimal_policy_equal.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/optimal_policy_favor_negative.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/optimal_policy_favor_negative.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/optimal_policy_favor_positive.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/optimal_policy_favor_positive.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/optimal_policy_favor_zero.pdf`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/optimal_policy_favor_zero.png`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/run_manifest.json`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/run_notes.txt`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/supply_expected_outcomes.csv`
- `Code/final_draft/seminar_pres/assets/theory_delta1/unknown_type/supply_history_outcomes.csv`

Standard active copies: `Code/final_simple/results/unknown_type/`, `Code/final_simple/results/supply/derivatives/figures/`, and `Code/final_draft/writeup/1108_writeup/assets/results/gamma_uncertainty/`.

## Final probability-label simplification

Prior-comparison tick labels now show only the dominant probability: `mu_L=0.5`, `mu_0=0.5`, or `mu_H=0.5`; the equal case shows `mu_r=0.33`. The other probabilities are unchanged. The same compact labels appear in supply-prior figure titles and the slide notation. Pricing/supply figures and both short-deck PDFs were refreshed with these labels.

## Initial-state wording

Changed the figure headings to “Initially inside” and “Initially outside”; price-path panel headings use “Inside” and “Outside”. Plot filenames and calculations are unchanged. Refreshed pricing, fixed-belief supply, probability-sweep figures, and both short-deck PDFs.

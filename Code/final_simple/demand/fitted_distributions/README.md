# Distribution artifacts and specifications

The existing figures here are saved results from earlier fits; adding code for five groups does not recompute them. The after-data job exports fresh population distributions separately under `RESULTS_ONLY/three/no_control_fit` and `RESULTS_ONLY/five/no_control_fit`. Use fit/provenance metadata rather than inferring the specification from a figure filename. The current main analysis is uncontrolled.

# Existing split-fit population distributions

These are copies of the distribution exports previously generated from the
saved `shell_files/res_het_hist` (with control) and `res_het_nocf` (without
control) fits. No demand model was re-estimated and no household inputs were
accessed to copy these exports.

Open `with_control/history_distributions.pdf` first. Each folder also contains
PNG plots, population means/SDs/sign shares/quantiles, and a density/CDF grid.
These are fitted household heterogeneity, not sampling confidence intervals.
The server workflow regenerates them from your selected saved fits.

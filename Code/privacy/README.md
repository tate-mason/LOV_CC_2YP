# Server-only research data policy

Raw purchases, household panels, weekly inputs, matched retail quotes, household
histories and temporary data partitions must remain on the research server.
Only estimates, predictions without household identifiers, and aggregate results
may be copied back to this computer. User-authored files are preserved.

`server_data_policy.py` rejects empirical I/O unless Python runs on Linux with
the established `/scratch/dtm63837/Kilts_Panel` server filesystem. There is no
environment-variable bypass. Guards run before loaders, processing workflows,
raw exports and regeneration, including direct function calls with local paths.
Legacy scripts with module-level I/O reject execution before data-stack imports.
Shell, Stata and MATLAB empirical jobs carry equivalent platform/filesystem guards.
Pure theoretical calculations and saved-result distribution exporters remain usable.
Keep `Code/privacy` alongside the research code when deploying to the server.

Run `python3 Code/privacy/check_policy.py` to verify the six current empirical
entry points reject local execution. This check does not create dataset fixtures.
Synthetic numerical tests remain separate; data-pipeline tests that perform
file I/O are now server-only too. Do not run the former full local empirical
check suite to work around the new guards.

The audit manifest lists guarded Python/Stata files and one pre-existing,
non-executable legacy Python file with a syntax error. MATLAB scripts and shell
jobs also have explicit guards. A second static check inventories all sources.

The pricing jobs still delete regenerated weekly inputs and staging files at
exit on the server. They do not delete the server's source data. Do not download
whole result directories indiscriminately: older folders can contain raw or
household-level exports. Recover only vetted estimate tables, aggregate results,
figures and current prediction exports without IDs.

This policy changes code execution; it does not remove other clones, backups,
historical Git objects, or user-authored files. Git history was left in place
at the user's direction. Existing results were not opened during this audit.

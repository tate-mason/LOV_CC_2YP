# LOV_CC_2YP

| Location | Contents |
|---|---|
| [Code/final_simple](Code/final_simple/README.md) | Current demand, pricing, and server workflow |
| [Code/final_draft/seminar_pres](Code/final_draft/seminar_pres/pres_short_README.md) | Seminar decks and short-deck build instructions |
| [Code/final_draft/writeup](Code/final_draft/writeup/) | Recent writeups and presentation revisions |
| [separate_experiments_50281568](separate_experiments_50281568/) | New fitted specifications, including the no-habit estimates |
| [supply_review](supply_review/) | Pricing mathematics and population-input benchmark figures |
| [Results](Results/README.md) | Exported estimates, aggregate results, and archived result bundles |
| [shell_files](shell_files/FULL_REBUILD.md) | Server submission scripts and rebuild instructions |
| [LitReview](LitReview/) | Literature sources and review |
| [old](old/README.md) | Superseded drafts, exploratory code, source revisions, and legacy outputs |
| [waste](waste/README.md) | Verified redundant source copies and generated build artifacts; nothing deleted |

The observed-basket theory analogue is submitted with
`Code/final_simple/server/run_observed_state_theory.sh`; see the
[server run guide](Code/final_simple/server/OBSERVED_STATE_THEORY.md).
The local no-habit plots remain labeled as population-input benchmarks until
the observed-initialization server run is completed.

The [old manifest](old/manifest.csv) and [waste manifest](waste/manifest.csv)
record original paths and reasons for every relocation. Environments, referenced
legacy modules, current estimate inputs, and uncertain data/diagnostic files
were retained in place. Earlier `Writeup/` and most legacy `Output/` material
now live under `old/`; a few unclassified diagnostics remain at their old paths.

Follow the [research data boundary](AGENTS.md) and [server data policy](Code/privacy/README.md) when working with empirical inputs and outputs.

---

# Netflix Algorithmic Choice Paper - Code Companion

## Purpose
This repository holds the code used to simulate and analyze a search model inspired by recommendation algorithms like those used by Netflix. The code is designed to first be simple, with hopes of extending it to more complex scenarios in the future.

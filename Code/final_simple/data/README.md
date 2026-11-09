# Server-only execution

This entire empirical pipeline is now server-only. The shared guard in
`Code/privacy/server_data_policy.py` runs before empirical reads or writes.
Local `--data-dir` and `--output` overrides cannot authorize processing here.
Only estimates, ID-free predictions and aggregates may be recovered locally.

# The data workflow

There are two scripts. Both contain their actual calculations; neither launches
the old scripts or imports from `final_draft`.

| Script | Purpose |
|---|---|
| `build_data.py` | Convert raw panel/retail files, clean columns, stack years, join tables, attach weeks, match yogurt prices |
| `summaries.py` | Recover flavor labels, describe households/purchases, calculate concentration and transitions, save the demand input |

Open each script's **Parameters** section before running. A Polars `LazyFrame`
is a plan for a calculation; `sink_parquet` executes that plan and writes it to
disk without requiring a pandas copy of the whole dataset. Those operations are
retained because the source files are large.

## 1. Build the matched purchase data

In `build_data.py`, edit `data_dir`, `panel_raw`, `retail_raw`, and the years.
The existing server paths are provided as ordinary Python variables.

`run_steps` starts empty. Copy the example full list into it for a complete
rebuild, or select only the steps whose prerequisites you already have.
The script runs selected steps in the order below, regardless of list order.
Importing the file does not start a build.

| Step | Input | Output under `data_dir` |
|---|---|---|
| `convert_panel` | Annual/master HMS TSVs | Annual panel Parquet and master tables |
| `clean_panel` | Converted panel tables | `cleaned_*.parquet` with selected columns |
| `combine_panel` | Cleaned annual files | `combined/purchases.parquet`, `trips.parquet`, `panelists.parquet` |
| `merge_panel` | Combined panel tables and product metadata | `merged_panel.parquet` |
| `panel_weeks` | Merged panel purchase dates | Same file with Saturday `week_end` |
| `convert_retail` | Annual/master RMS TSVs and movement modules | Annual/master Parquet in `RMS/` |
| `clean_retail` | Converted retail tables | Selected columns in `RMS/cleaned_*.parquet` |
| `combine_retail` | Annual retail tables | `RMS/combined/*.parquet` |
| `merge_retail` | Movements, versions, products, stores, extra labels, brands | `RMS/merged_rms.parquet` |
| `match_yogurt` | Merged panel and retail tables | `yogurt_estimation_sample.parquet` |

From the `final_simple` folder:

```bash
python3 data/build_data.py
```

Optional server flags can override the Python variables:

```bash
python3 data/build_data.py --data-dir /your/Data \
  --steps merge_panel panel_weeks merge_retail match_yogurt
```

The conversion loops report missing files and continue, as in the old scripts.
Check those messages for complete year coverage. Annual concatenation retains
the old wildcard rule: all matching annual files are included. A stale annual
file therefore still belongs to the input unless you remove it from that source
folder or choose another build directory.

The panel merge anchors on purchases and inner joins trips, then left joins
household/year demographics and product metadata. The retail merge retains the
original first-row deduplication rules and metadata join keys. Counts, null keys,
metadata coverage, entity coverage, year coverage, prices, and file sizes are
reported. A row-count ratio alone does not prove that a join is correct.

The yogurt match remains an **inner join** on panel year, Saturday week-end,
UPC, and store. Only module codes 3603 and 3612 are used by default. The year key
is retained: a purchase's panel year and its retail calendar year must agree,
including at year boundaries. No unmatched purchase is turned into an outside
choice or assigned an imputed price.

The two old empty placeholders, `panel_dat.py` and `retail_dat.py`, had no
calculations to move. The eight other raw-build scripts' operations are now
named functions and explicit steps in `build_data.py`.

## 2. Recover flavors, summarize purchases, and prepare demand input

Edit `data_dir`, `flavors`, `products_extra`, `output`, years, and
`household_batches`, and `flavor_spec` in `summaries.py`.

```bash
python3 data/summaries.py
```

The script runs these research steps in order:

1. Supplement absent UPCs from annual product metadata when available.
2. Construct full-panel and matched-yogurt household samples.
3. Summarize demographics and annual coverage with household partitions.
4. Calculate detailed-flavor shares, household HHI, transitions, and spells.
5. Repeat grouped calculations for the selected `flavor_spec`: original three or expanded five categories.
6. Save the labeled purchase table and resolved mapping for demand estimation.

Both the recovery and original classification rules remain: existing reference
entries are retained even if blank; earliest valid annual metadata supplements
absent UPCs; plain/unflavored classification precedes berry terms. References can
be legacy CSV or Parquet. No reference data were opened during development.

Set `recover_only = True` to run only the recovery utility formerly provided by
`recover_flavors.py`. Set `recovery_output` to your desired separate file, or
leave it `None` to use the summary output folder. The original reference is
never overwritten by recovery.

Set `hhi_only = True` to redraw detailed/grouped HHI figures from existing saved
household tables. Set `no_figures = True` for tables only. Normal summaries now
save a small household Parquet checkpoint alongside the existing CSV output, so
HHI and household averages can reuse it. Older CSV-only HHI folders are still
supported.

`household_batches` is a memory setting, not a sampling rule. Households are
partitioned physically and retained in full. Changing the partition count must
not change statistics; synthetic checks compare one and three partitions.

Full-panel and matched-yogurt demographic summaries restrict purchase records
to single-person households. Matched-yogurt summaries additionally require
positive matched prices and at least two distinct purchase weeks. Weeks need
not be consecutive. Transitions and spells require adjacent calendar weeks and
single-flavor baskets; mixed baskets and gaps interrupt those histories.

The prepared demand input deliberately retains other households' retail quote
coverage. `demand/estimation.py` applies its own single-person demand restriction
while retaining those quotes for price menus and instruments.

The main handoff to demand is:

```text
data_dir/summary_outputs/yogurt_estimation_sample.parquet
    -> demand/estimation.py's sample parameter
```

Keep the output mapping, `estimation_flavor_reference.csv`, with it. The saved
purchase table already contains the resolved `flavor` column, so normal demand
estimation does not need to read and reclassify the reference again.

## Changes made explicit

The scientific join/sample/history rules are retained. Three implementation
changes are documented rather than hidden:

- **Retail path correction:** old retail code wrote `Data/RMS/merged_rms.parquet`
  while old master matching read `Data/merged_rms.parquet`. Both now use the
  former. `retail_merged_input` can select an existing legacy file elsewhere.
- **Temporary year chunks:** matching now uses a private temporary directory
  and the selected year files explicitly. Leftovers from interrupted builds
  cannot enter another run's concatenation.
- **Parquet checkpoints:** regular summaries reuse Parquet for internal
  calculations instead of reopening their own CSV exports. Existing CSV/LaTeX
  tables and PDF figures remain; the checkpoint is additional.

The original data directory is not moved or rebuilt by organizing the code.
Choose a separate build/output location if you want to retain an earlier build.

## Verification

`python3 checks/run_checks.py` now includes data checks. They use generated
Parquet/in-memory tables, and mock raw readers without opening TSV/CSV sources.
`python3 checks/compare_data_original.py /path/to/final_draft` additionally compares
original casts/joins and all exported summary tables with the new calculations.
Legacy CSV intermediates in that comparison are supplied from generated memory
captures; no CSV file is read.


Purchase-level estimation samples belong in the server data directory, outside
the code checkout. Both raw matching and prepared-sample summary producers now
reject an estimation-sample destination inside the checkout before opening
purchase inputs. The sample filename and summary staging folders are also
ignored by Git. This does not erase copies in earlier Git commits.

## Current alongside run: skip the data build

Use `server/run_after_data.sh EXISTING_REBUILD_RUN_DIR` from the server checkout. It reuses the completed conversion/merges/match and reruns this summary stage independently for `three` and `five`, followed by uncontrolled estimation and pricing. The five groups are other, berry, plain, other fruits, desserts. Shared `flavor_specs.py` gives dessert labels priority over fruit, with plain/unflavored first. The original three-group mapping remains available without changing its classification. New outputs have separate directories; earlier files are not replaced.

The numerical baseline is preserved, but groups added in the expanded model change within-yogurt shares, concentration, transitions, menus and estimated fixed effects. Compare both runs instead of combining their tables. A full descriptor reference is required to classify the new groups; the old three-group mapping alone cannot recover the detail.

The older verification commands above describe historical fixture checks. They do not authorize local empirical I/O. Current local validation uses pure numerical kernels and the privacy guard; run data-producing checks only under the research-server policy.

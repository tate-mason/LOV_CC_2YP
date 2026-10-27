## === Import Necessary Packages === ##
import glob
import os
import polars as pl

## === Define Locals === ##

annual = 0
master = 1

years = range(2006, 2016)
base_raw_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/nielsen_extracts/nielsen_extracts/HMS"
base_out_dir = "/scratch/dtm63837/Kilts_Panel/Data"
os.makedirs(base_out_dir, exist_ok=True)

base_mapping = {
    "panelists": "panelists",
    "purchases": "purchases",
    "trips": "trips",
}
master_mappings = {"products": "products", "retailers": "retailers"}
## === Conversion Function === ##

if annual:
    for y in years:
        print(f"\n--- Converting year {y} ---")

        for base_tsv, out_name in base_mapping.items():
            tsv_filename = f"{base_tsv}_{y}.tsv"
            tsv_path = os.path.join(base_raw_dir, str(y), "Annual_Files", tsv_filename)
            parquet_out = os.path.join(base_out_dir, f"{out_name}_{y}.parquet")

            if os.path.exists(tsv_path):
                (
                    pl.scan_csv(
                        tsv_path,
                        separator="\t",
                        quote_char=None,
                        infer_schema_length=10000,
                    )
                    .rename(str.lower)
                    .sink_parquet(parquet_out)
                )
                print(f"Converted: {tsv_filename} -> {out_name}_{y}.parquet")
            else:
                print(f"WARNING: File not found {tsv_path}")
if master:
    for base_tsv, out_name in master_mappings.items():
        tsv_filename = f"{base_tsv}.tsv"
        tsv_path = os.path.join(base_raw_dir, "Master_Files/Latest", tsv_filename)
        parquet_out = os.path.join(base_out_dir, f"{out_name}.parquet")

        if os.path.exists(tsv_path):
            (
                pl.scan_csv(
                    tsv_path,
                    separator="\t",
                    quote_char=None,
                    infer_schema_length=10000,
                    encoding="utf8-lossy",
                )
                .rename(str.lower)
                .sink_parquet(parquet_out)
            )

## === Cleaning === ##

parquet_files = glob.glob(os.path.join(base_out_dir, "*.parquet"))

for filepath in sorted(parquet_files):
    filename = os.path.basename(filepath)
    cols = pl.scan_parquet(filepath).collect_schema().names()
    print(f"\n--- {filename} ({len(cols)} columns) ---")
    print(cols)

columns_to_keep = {
    "panelists": [
        "household_cd",
        "panel_year",
        "household_income",
        "household_size",
        "male_head_age",
        "female_head_age",
        "male_head_employment",
        "female_head_employment",
        "race",
        "hispanic_origin",
        "dma_cd",
        "dma_name",
    ],
    "purchases": [
        "trip_code_uc",
        "upc",
        "quantity",
        "total_price_paid",
        "deal_flag_uc",
    ],
    "trips": [
        "trip_code_uc",
        "household_code",
        "purchase_date",
        "store_code_uc",
        "panel_year",
    ],
}

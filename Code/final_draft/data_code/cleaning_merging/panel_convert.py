## === Import Necessary Packages === ##
import glob
import os
import polars as pl

## === Define Locals === ##

years = range(2006, 2016)
base_raw_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/nielsen_extracts/nielsen_extracts/HMS"
base_out_dir = "/scratch/dtm63837/Kilts_Panel/Data"
os.makedirs(base_out_dir, exist_ok=True)

base_mapping = {
    "panelists": "panelists",
    "purchases": "purchases",
    "trips": "trips",
}

## === Conversion Function === ##

for y in years:
    print(f"\n--- Converting year {y} ---")

    for base_tsv, out_name in base_mapping.items():
        tsv_filename = f"{base_tsv}_{y}.tsv"
        tsv_path = os.path.join(base_raw_dir, str(y), "Annual_Files", tsv_filename)
        parquet_out = os.path.join(base_out_dir, f"{out_name}_{y}.parquet")

        if os.path.exists(tsv_path):
            (
                pl.scan_csv(
                    tsv_path, separator="\t", quote_char=None, infer_schema_length=0
                )
                .rename(str.lower)
                .sink_parquet(parquet_out)
            )
            print(f"Converted: {tsv_filename} -> {out_name}_{y}.parquet")
        else:
            print(f"WARNING: File not found {tsv_path}")
master_mappings = {"products": "products", "retailers": "retailers"}
for y in years:
    for base_tsv, out_name in master_mappings.items():
        tsv_filename = f"{base_tsv}_{y}.tsv"
        tsv_path = os.path.join(
            base_raw_dir, str(y), "Master_Files/Latest", tsv_filename
        )
        parquet_out = os.path.join(base_out_dir, f"{out_name}_{y}.parquet")

        if os.path.exists(tsv_path):
            (
                pl.scan_csv(
                    tsv_path, separator="\t", quote_char=None, infer_schema_length=0
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

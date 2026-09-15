import polars as pl
import gc
import os

years = [2022, 2023, 2024]

base_raw_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/RMS"
base_out_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/RMS"

os.makedirs(base_out_dir, exist_ok=True)

files_to_convert = {
    'productattributes.tsv': 'attributes',
    'productdesc.tsv': 'description',
    'producthierarchy.tsv': 'hierarchy',
    'stores.tsv':           'stores'
}

for y in years:
    print(f'=== Converting files for {y} ===')
    for tsv_name, out_name in files_to_convert.items():
        tsv_path    = f"{base_raw_dir}/{y}/Annual_Files/{tsv_name}"
        parquet_out = f"{base_out_dir}/{out_name}_{y}.parquet"

        if os.path.exists(tsv_path):
            (pl.scan_csv(
                tsv_path,
                separator = "\t",
                quote_char = None,
                infer_schema_length = 0
            )
            .rename(str.lower)
            .sink_parquet(parquet_out)
            )
            print(f"Converted {tsv_name} -> {out_name}_{y}.parquet")
        else:
            print(f"WARNING: File not found {tsv_path}")
    print(f'===Converting Movement Files===')
    tsv_path       = f"{base_raw_dir}/GROCERY/YOGURT_{y}.tsv"
    parquet_out    = f"{base_out_dir}/movement_{y}.parquet"

    (pl.scan_csv(
        tsv_path,
        separator  = "\t",
        quote_char = None,
        infer_schema_length = 0
    )
    .rename(str.lower)
    .sink_parquet(parquet_out)
    )




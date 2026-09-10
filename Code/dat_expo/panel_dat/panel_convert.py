import os
import polars as pl

years = [2022, 2023, 2024]
base_raw_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS"
base_out_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS"
os.makedirs(base_out_dir, exist_ok=True)

files_to_convert = {
    'panelist.tsv': 'panelists',
    'purchase.tsv': 'purchases',
    'trip.tsv': 'trips',
    'productattributes.tsv': 'product_attr',
    'productdesc.tsv': 'product_desc',
    'retailer.tsv': 'retailer',
    'producthierarchy.tsv': 'hierarchy'
}

for y in years:
    print(f"--- Converting year {y} ---")
    for tsv_name, out_name in files_to_convert.items():
        tsv_path = f"{base_raw_dir}/{y}/Annual_Files/{tsv_name}"
        parquet_out = f"{base_out_dir}/{out_name}_{y}.parquet"
        
        if os.path.exists(tsv_path):
            (
                pl.scan_csv(
                    tsv_path,
                    separator='\t',
                    quote_char=None,
                    infer_schema_length=0
                )
                .rename(str.lower)
                .sink_parquet(parquet_out)
            )
            print(f"Converted {tsv_name} -> {out_name}_{y}.parquet")
        else:
            print(f"WARNING: File not found {tsv_path}")

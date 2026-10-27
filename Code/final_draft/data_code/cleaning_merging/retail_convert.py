import glob
import os
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/nielsen_extracts/RMS"
output_dir = "/scratch/dtm63837/Kilts_Panel/Data/RMS"
os.makedirs(output_dir, exist_ok=True)

annual = 0
master = 0

years = range(2006, 2016)

annual_mapping = {
    "products_extra": "products_extra",
    "rms_versions": "rms",
    "stores": "stores",
}
movement_mapping = {
    "3603": "3603",
    "3612": "3612",
}
master_mapping = {"brand_variations": "brand_variations", "products": "products"}

if annual:
    for y in years:
        for base_mapping, out_name in annual_mapping.items():
            tsv_filename = f"{base_mapping}_{y}.tsv"
            tsv_path = os.path.join(input_dir, str(y), "Annual_Files", tsv_filename)
            parquet_out = os.path.join(output_dir, f"{out_name}_{y}.parquet")

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
                print(f"FAILED: File not found {tsv_filename}")
        for base_mapping, out_name in movement_mapping.items():
            tsv_filename = f"{base_mapping}_{y}.tsv"
            tsv_path = os.path.join(
                input_dir, str(y), f"Movement_Files/2510_{y}", tsv_filename
            )
            parquet_out = os.path.join(output_dir, f"{out_name}_{y}.parquet")

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
                print(f"FAILED: File not found {tsv_filename}")

if master:
    for base_mapping, out_name in master_mapping.items():
        tsv_filename = f"{base_mapping}.tsv"
        tsv_path = os.path.join(
            input_dir, "Master_Files_2006-2020/Latest", tsv_filename
        )
        parquet_out = os.path.join(output_dir, f"{out_name}.parquet")

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
            print(f"Converted: {tsv_filename} -> {out_name}.parquet")
        else:
            print(f"FAILED: File not found {tsv_filename}")


parquet_files = glob.glob(os.path.join(output_dir, "*.parquet"))

for filepath in sorted(parquet_files):
    filename = os.path.basename(filepath)
    cols = pl.scan_parquet(filepath).collect_schema().names()
    print(f"\n--- {filename} ({len(cols)} columns) ---")
    print(cols)

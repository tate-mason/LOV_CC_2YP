import glob
import os
import polars as pl

input_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/nielsen_extracts/RMS"
output_dir = "/scratch/dtm63837/Kilts_Panel/Data/RMS"
os.makedirs(output_dir, exist_ok=True)

annual = 1
master = 1

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

    # Check if file is empty before reading
    if os.path.getsize(filepath) == 0:
        print(f"\n--- {filename} is EMPTY (0 bytes), skipping ---")
        continue

    try:
        cols = pl.scan_parquet(filepath).collect_schema().names()
        print(f"\n--- {filename} ({len(cols)} columns) ---")
        print(cols)
    except Exception as e:
        print(f"\n--- Could not read {filename} (corrupted or partial file): {e} ---")

columns_to_keep = {
    "3603": ["store_code_uc", "upc", "week_end", "price"],
    "3612": ["store_code_uc", "upc", "week_end", "price"],
    "products": [
        "upc",
        "upc_ver_uc",
        "product_module_code",
        "brand_code_uc",
        "brand_descr",
        "size1_code_uc",
        "size1_amount",
        "size1_units",
    ],
    "products_extra": [
        "upc",
        "upc_ver_uc",
        "panel_year",
        "flavor_code",
        "flavor_descr",
    ],
    "stores": [
        "store_code_uc",
        "year",  # Keep as string 'year' during initial extraction
        "retailer_code",
        "channel_code",
        "dma_code",
        "dma_descr",
    ],
}

# Sort keys by length descending so "products_extra" is tested before "products"
table_keys_by_length = sorted(columns_to_keep.keys(), key=len, reverse=True)

for filepath in glob.glob(os.path.join(output_dir, "*.parquet")):
    filename = os.path.basename(filepath)

    # Skip files that have already been cleaned
    if filename.startswith("cleaned_"):
        continue

    # Remove extension -> "products_extra_2011"
    clean_name = os.path.splitext(filename)[0]

    # Match against dictionary keys taking exact boundaries into account
    matched_table = None
    for key in table_keys_by_length:
        if clean_name == key or clean_name.startswith(f"{key}_"):
            matched_table = key
            break

    if matched_table:
        target_cols = columns_to_keep[matched_table]
        cleaned_path = os.path.join(output_dir, f"cleaned_{filename}")

        (pl.scan_parquet(filepath).select(target_cols).sink_parquet(cleaned_path))
        print(
            f"Cleaned {filename} (Matched: {matched_table}) -> saved to cleaned_{filename}"
        )

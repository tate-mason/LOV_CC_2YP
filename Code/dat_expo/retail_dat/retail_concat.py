import os
import polars as pl

output_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/RMS/output_markets"

market_codes = [
    "Atlanta", "Chicago", "Houston", "Denver", 
    "Phoenix", "Philadelphia", "San_Diego", "Des_Moines"
]

market_dfs = []
for m in market_codes:
    file_path = os.path.join(output_dir, f'retail_{m}.parquet')

    if os.path.exists(file_path):
        lazy_df = (
            pl.scan_parquet(file_path)
            .rename(str.lower)
            .with_columns(pl.lit(m).alias("market_name"))
        )
        market_dfs.append(lazy_df)

    else:
        print(f"Skipping missing file: {file_path}")

if market_dfs:
    combined_lazy = pl.concat(market_dfs, how='diagonal_relaxed')
    combined_lazy.sink_parquet(os.path.join(output_dir, 'full_retail.parquet'))
    print('--> Saved master retail parquet file (full_retail.parquet')

import os
import polars as pl

market_codes = [
    "Atlanta", "Chicago", "Houston", "Denver", 
    "Phoenix", "Philadelphia", "San_Diego", "Des_Moines"
]
for m in market_codes:
    retail_m = (
        pl.scan_parquet(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/RMS/output_markets/retail_{m}')
        .rename(str.lower)
        .collect()
    )

    print(f"Joined: {retail_m.height}")

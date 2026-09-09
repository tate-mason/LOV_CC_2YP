import polars as pl
import os

output_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets"
os.makedirs(output_dir, exist_ok=True)
# data sets being loaded
dat = ['purchases', 'trips', 'product_attr', 'product_desc', 'retailer']
years = [2022, 2023, 2024]
valid_codes = {
    "Atlanta": [f'{x:05d}' for x in range(13010, 13300)],
    "Chicago": [f'{x:05d}' for x in range(17031, 18128)],
    "Houston": [f'{x:05d}' for x in range(48015, 48482)],
    "Denver":  [f'{x:05d}' for x in range(8001, 8126)],
    "Phoenix": [f'{x:05d}' for x in range(4007, 4026)],
    "Philadelphia": [
        f'{x:05d}' for x in set().union(
            range(34001, 34004),
            range(42017, 42102)
        )
    ],
    "San Diego": ["06073"],
    "Des Moines":[f'{x:05d}' for x in range(19001, 19199)]
}

market_df = []
for m in valid_codes:
    file_path = f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets/master_{m}.parquet'

    lazy_df   = pl.scan_parquet(file_path).with_columns(pl.lit(m).alias("market_name"))
    market_df.append(lazy_df)

combined_lazy = pl.concat(market_df, how='diagonal_relaxed')
combined_lazy.sink_parquet(os.path.join(output_dir, 'full_panel.parquet'))
print('--> Saved master panel parquet file')




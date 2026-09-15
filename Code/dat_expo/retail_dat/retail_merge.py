import os
import polars as pl

rms_dir    = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/RMS"
output_dir = f"{rms_dir}/output_markets"

os.makedirs(output_dir, exist_ok=True)

years = [2022,2023,2024]
market_codes = {
    "Atlanta": [f'{x:05d}' for x in range(13010, 13301)],
    "Chicago": [f'{x:05d}' for x in range(17031, 18129)],
    "Houston": [f'{x:05d}' for x in range(48015, 48483)],
    "Denver":  [f'{x:05d}' for x in range(8001, 8127)],
    "Phoenix": [f'{x:05d}' for x in range(4007, 4027)],
    "Philadelphia": [
        f'{x:05d}' for x in set().union(
            range(34001, 34005),
            range(42017, 42103)
        )
    ],
    "San_Diego": ["06073"],
    "Des_Moines": [f'{x:05d}' for x in range(19001, 19200)]
}

files_to_merge = {
    'productattributes.parquet': 'attributes',
    'productdesc.parquet': 'description',
    'producthierarchy.parquet': 'hierarchy',
    'stores.parquet':           'stores'
}
for m, codes in market_codes.items():
    yearly_lfs = []

    for f, name in files_to_merge.items():
        for y in years:
            file_path = f"{rms_dir}/{name}_{y}.parquet"

            lazy_df = (
                pl.scan_parquet(file_path)
                .with_columns(
                    (pl.col('fips_state_code') + pl.col('fips_county_code')).alias('fips_code')
                )
                .filter(pl.col('fips_full').is_in(codes))
                .drop('fips_full')
            )
            yearly_lfs.append(lazy_df)
        combined_lazy = pl.concat(yearly_lfs, how='diagonal_relaxed')
        out_file = os.path.join(output_dir, f'{m}_{name}.parquet')
        combined_lazy.sink_parquet(out_file)
        print(f'--> Saved market data for {name} in {m}')

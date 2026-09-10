import os
from itertools import product
import polars as pl
import scipy as sp
import gc

output_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets"
os.makedirs(output_dir, exist_ok=True)
# data sets being loaded
dat = ['purchases', 'trips', 'product_attr', 'product_desc', 'retailer']
years = [2022, 2023, 2024]
valid_codes = {
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
    "Des_Moines":[f'{x:05d}' for x in range(19001, 19200)]
}

market_dfs = {}
dat_filter = ['panelists']

for m, codes in valid_codes.items():
    market_dfs[m] = {}

    for d in dat_filter:
        yearly_lfs = []
        for y in years:
            file_path = f"/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{d}_{y}.parquet"

            lazy_df = (
                pl.scan_parquet(file_path)
                .rename(str.lower)
                .select([
                    'household_code',
                    'panel_year',
                    'projection_factor',
                    'household_income',
                    'household_size',
                    'male_head_age',
                    'female_head_age',
                    'male_head_employment',
                    'female_head_employment',
                    'race',
                    'hispanic_origin',
                    'panelist_zip_code',
                    'fips_state_code',
                    'fips_county_code'])
                .with_columns(
                    (
                        pl.col('fips_state_code').cast(pl.Utf8).str.zfill(2) +
                        pl.col('fips_county_code').cast(pl.Utf8).str.zfill(3)
                    ).alias('fips_full')
                )
                .filter(pl.col('fips_full').is_in(codes))
                .filter(pl.col('household_size').cast(pl.Int64)==1)
                .drop('fips_full')  # Optional: drop the temp column
            )
            yearly_lfs.append(lazy_df)
        combined_lazy = pl.concat(yearly_lfs, how='diagonal_relaxed')
        df_dataset = combined_lazy.collect()
        market_dfs[m][d] = df_dataset 

        out_file = os.path.join(output_dir, f'{m}_{d}.parquet')
        df_dataset.write_parquet(out_file)
        print(f'--> Saved {out_file} ({df_dataset.height:,} rows)')

for d in dat:
    yearly_lfs = []
    for y in years:
        file_path = f"/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{d}_{y}.parquet"

        lazy_df = (
            pl.scan_parquet(file_path)
            .rename(str.lower)
        )
        yearly_lfs.append(lazy_df)
    combined_lazy = pl.concat(yearly_lfs, how='diagonal_relaxed')
    df_dataset = combined_lazy.collect()
    market_dfs[d] = df_dataset 

    out_file = os.path.join(output_dir, f'{d}.parquet')
    df_dataset.write_parquet(out_file)
    print(f'--> Saved {out_file} ({df_dataset.height:,} rows)')

# merging trips and panelists
trips     = pl.scan_parquet(os.path.join(output_dir, 'trips.parquet'))
purchases = pl.scan_parquet(os.path.join(output_dir, 'purchases.parquet'))
products  = (
    pl.scan_parquet(
        os.path.join(
            output_dir, 'product_attr.parquet'
        )
    )
    .select(['upc',
                'year',
                'brand',
                'brand_cd',
                'protein_gram',
                'protein_gram_cd',
                'total_fat_gram',
                'total_fat_gram_cd',
                'sugar_gram',
                'sugar_gram_cd',
                'product_size',
                'product_size_cd'
                ])
    .rename({'year': 'panel_year'})
    .join(
        pl.scan_parquet(
            os.path.join(
                output_dir, 'product_desc.parquet'
            )
        )
        .rename({'year': 'panel_year'}),
        on = 'upc',
        how='left',
        suffix = '_prod'
    )
)
retailers = pl.scan_parquet(os.path.join(output_dir, 'retailer.parquet'))
# Build a single set containing all numbers across all ranges

datasets = {
    'trips': trips,
    'purchases': purchases,
    'retailer': retailers,
    'products': products
}

for name, d in datasets.items():
    print(f"{name} cols:", d.collect_schema().names())

for m in valid_codes:
    trip_panelists = trips.join(
        pl.scan_parquet(
            os.path.join(output_dir, f'{m}_panelists.parquet')
        ), 
        on     = ['panel_year','household_code'], 
        how    = 'inner',
        suffix = '_panelist'

    )

    tpp    = trip_panelists.join(
        purchases,
        on     = ['panel_year','trip_code_uc'],
        how    = 'inner',
        suffix = '_purchase'
    )

    tpp_r  = tpp.join(
        retailers,
        on    = 'retailer_code',
        how   = 'left',
        suffix = '_retailer'
    )

    master    = (
        tpp_r
        .join(
            products,
            on     = 'upc',
            how    = 'left',
            suffix = '_product'
        )
        .sink_parquet(
            os.path.join(output_dir, f'master_{m}.parquet')
        )
    )

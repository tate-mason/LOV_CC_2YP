import os
import polars as pl

hms_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS"
output_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets"
os.makedirs(output_dir, exist_ok=True)

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
    "Des_Moines": [f'{x:05d}' for x in range(19001, 19200)]
}

# 1. Process and filter Panelists per Market
for m, codes in valid_codes.items():
    yearly_lfs = []
    for y in years:
        file_path = f"{hms_dir}/panelists_{y}.parquet"

        lazy_df = (
            pl.scan_parquet(file_path)
            .rename(str.lower)
            .select([
                'household_code', 'panel_year', 'projection_factor',
                'household_income', 'household_size', 'male_head_age',
                'female_head_age', 'male_head_employment', 'female_head_employment',
                'race', 'hispanic_origin', 'panelist_zip_code',
                'fips_state_code', 'fips_county_code'
            ])
            .with_columns([
                pl.col('household_code').cast(pl.Int64),
                pl.col('panel_year').cast(pl.Int64),
                pl.col('household_size').cast(pl.Int64),
                pl.col('fips_state_code').cast(pl.Utf8).str.zfill(2),
                pl.col('fips_county_code').cast(pl.Utf8).str.zfill(3),
            ])
            .with_columns(
                (pl.col('fips_state_code') + pl.col('fips_county_code')).alias('fips_full')
            )
            .filter(pl.col('fips_full').is_in(codes))
            .filter(pl.col('household_size') == 1)
            .drop('fips_full')
        )
        yearly_lfs.append(lazy_df)

    combined_lazy = pl.concat(yearly_lfs, how='diagonal_relaxed')
    out_file = os.path.join(output_dir, f'{m}_panelists.parquet')
    combined_lazy.sink_parquet(out_file)
    print(f'--> Saved market panelists for {m}')

# 2. Process General Datasets (Trips, Purchases, Products, etc.)
general_dats = ['purchases', 'trips', 'product_attr', 'product_desc', 'retailer', 'hierarchy']

for d in general_dats:
    yearly_lfs = []
    for y in years:
        file_path = f"{hms_dir}/{d}_{y}.parquet"
        lazy_df = pl.scan_parquet(file_path).rename(str.lower)
        
        # Explicitly cast common join keys to Int64 if they exist
        cols = lazy_df.collect_schema().names()
        casts = []
        for k in ['household_code', 'panel_year', 'year', 'trip_code_uc', 'upc', 'retailer_code']:
            if k in cols:
                casts.append(pl.col(k).cast(pl.Int64))
        
        if casts:
            lazy_df = lazy_df.with_columns(casts)

        yearly_lfs.append(lazy_df)

    combined_lazy = pl.concat(yearly_lfs, how='diagonal_relaxed')
    out_file = os.path.join(output_dir, f'{d}.parquet')
    combined_lazy.sink_parquet(out_file)
    print(f'--> Saved combined {d}.parquet')

# 3. Master Join Step per Market
trips = pl.scan_parquet(os.path.join(output_dir, 'trips.parquet'))
purchases = pl.scan_parquet(os.path.join(output_dir, 'purchases.parquet'))
retailers = pl.scan_parquet(os.path.join(output_dir, 'retailer.parquet'))

hierarchy = (
    pl.scan_parquet(os.path.join(output_dir, 'hierarchy.parquet'))
    .with_columns(
        pl.col('year').alias('panel_year') if 'year' in pl.scan_parquet(os.path.join(output_dir, 'hierarchy.parquet')).collect_schema().names() else pl.col('panel_year')
    )
)


products = (
    pl.scan_parquet(os.path.join(output_dir, 'product_attr.parquet'))
    .rename({'year': 'panel_year'}, missing_ok=True)
    .join(
        pl.scan_parquet(os.path.join(output_dir, 'product_desc.parquet'))
        .rename({'year': 'panel_year'}, missing_ok=True),
        on=['panel_year', 'upc'],
        how='left',
        suffix='_prod'
    )
)

for m in valid_codes:
    panelists = pl.scan_parquet(os.path.join(output_dir, f'{m}_panelists.parquet'))

    trip_panelists = trips.join(
        panelists,
        on=['panel_year', 'household_code'],
        how='inner',
        suffix='_panelist'
    )

    tpp = trip_panelists.join(
        purchases,
        on=['panel_year', 'trip_code_uc'],
        how='inner',
        suffix='_purchase'
    )

    tpp_r = tpp.join(
        retailers,
        on='retailer_code',
        how='left',
        suffix='_retailer'
    )

    (
        tpp_r
        .join(products, on=['panel_year', 'upc'], how='left', suffix='_product')
        .join(hierarchy, on=['panel_year', 'upc'], how='left', suffix='_hierarchy')
        .sink_parquet(os.path.join(output_dir, f'master_{m}.parquet'))
    )
    print(f'--> Finished master build for {m}')

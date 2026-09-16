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
    'movement.parquet': 'movement'
}
for f, name in files_to_merge.items():
    yearly_lfs = []
    for y in years:
        file_path = f"{rms_dir}/{name}_{y}.parquet"

        lazy_df = (
            pl.scan_parquet(file_path)
        )
        yearly_lfs.append(lazy_df)
    combined_lazy = pl.concat(yearly_lfs, how='diagonal_relaxed')
    out_file = os.path.join(output_dir, f'{name}.parquet')
    combined_lazy.sink_parquet(out_file)
    print(f'--> Saved data for {name}')

for m, codes in market_codes.items():
    yearly_lfs = []
    for y in years:
        file_path = f"{rms_dir}/stores_{y}.parquet"

        lazy_df = (
            pl.scan_parquet(file_path)
            .rename(str.lower)
            .with_columns([
                pl.col('fips_state_code').cast(pl.Utf8).str.zfill(2),
                pl.col('fips_county_code').cast(pl.Utf8).str.zfill(3),
            ])
            .with_columns(
                (pl.col('fips_state_code') + pl.col('fips_county_code')).alias('fips_full')
            )
            .filter(pl.col('fips_full').is_in(codes))
            .drop('fips_full')
        )
        yearly_lfs.append(lazy_df)

    combined_lazy = pl.concat(yearly_lfs, how='diagonal_relaxed')
    out_file = os.path.join(output_dir, f'{m}_stores.parquet')
    combined_lazy.sink_parquet(out_file)
    print(f'--> Saved market stores for {m}')

attr_lf = pl.scan_parquet(os.path.join(output_dir, 'attributes.parquet'))
if 'year' in attr_lf.collect_schema().names():
    attr_lf = attr_lf.rename({'year': 'panel_year'})

desc_lf = pl.scan_parquet(os.path.join(output_dir, 'description.parquet'))
if 'year' in desc_lf.collect_schema().names():
    desc_lf = desc_lf.rename({'year': 'panel_year'})
hierarchy   = (
        pl.scan_parquet(os.path.join(output_dir, 'hierarchy.parquet'))
        .with_columns(
            pl.col('year').alias('panel_year') if 'year' in pl.scan_parquet(os.path.join(output_dir, 'hierarchy.parquet')).collect_schema().names() else pl.col('panel_year')
        )
)
movement    = pl.scan_parquet(os.path.join(output_dir, 'movement.parquet'))

products    = attr_lf.join(
    desc_lf,
    on     = ['panel_year', 'upc'],
    how    = 'left',
    suffix = '_prod' 
)

for m in market_codes:
    stores = (
            pl.scan_parquet(os.path.join(output_dir, f'{m}_stores.parquet'))
            .with_columns(
                pl.col('year').alias('panel_year') if 'year' in pl.scan_parquet(os.path.join(output_dir, f'{m}_stores.parquet')).collect_schema().names() else pl.col('panel_year')
            )
    )

    store_movement = stores.join(
        movement,
        on     = 'store_code_uc',
        how    = 'left',
        suffix = '_mvmt'
    )

    sm_p           = store_movement.join(
        products,
        on     = ['upc', 'panel_year'],
        how    = 'left',
        suffix = '_p'
    )

    smp_h          = sm_p.join(
        hierarchy,
        on     = ['upc', 'panel_year'],
        how    = 'left',
        suffix = '_hier'
    ).sink_parquet(
        os.path.join(
            output_dir, f'retail_{m}.parquet'
        )
    )





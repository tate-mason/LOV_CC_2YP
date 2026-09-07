import os
from itertools import product
import polars as pl
import scipy as sp
import gc

output_dir = "/scratch/dtm63837/Kilts_Panel/nielsen_extracts/output_markets"
os.makedirs(output_dir, exist_ok=True)
# data sets being loaded
dat = ['purchases', 'trips', 'product_attr', 'product_desc']
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

market_dfs = {}
dat_filter = ['panelists', 'retailer']

for m, codes in valid_codes.items():
    market_dfs[m] = {}

    for d in dat_filter:
        yearly_lfs = []
        for y in years:
            file_path = f"/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{d}_{y}.parquet"

            lazy_df = (
                pl.scan_parquet(file_path)
                .rename(str.lower)
                .filter(pl.col('fips_county_code').cast(pl.Utf8).is_in(codes))
            )
            yearly_lfs.append(lazy_df)
        combined_lazy = pl.concat(yearly_lfs, how='diagonal_relaxed')
        df_dataset = combined_lazy.collect()
        market_dfs[m][d] = df_dataset 

        out_file = os.path.join(output_dir, f'{m}_{d}.parquet')
        df_dataset.sink_parquet(out_file)
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
    df_dataset.sink_parquet(out_file)
    print(f'--> Saved {out_file} ({df_dataset.height:,} rows)')

# merging trips and panelists
# Build a single set containing all numbers across all ranges
#trip_panelists = trips.join(panelists, on = ['panel_year', 'household_code'], how='left')
# Atlanta, Chicago, Denver, Des Moines, San Diego, Philly, Houston, Phoenix 
# .is_in(13010:13299), (17031:18127), (48015:48481), (08001:08125), (04007:04025), ((34001:34003), (42017:42101)), 06073, (19001:19197)
del trips, panelists
gc.collect()

# merge purchases and trip_panelists

#tpp = trip_panelists.join(purchases, on = 'trip_code_uc', how='left')
#del trip_panelists, purchases
#gc.collect()
#
#tpp_r = tpp.join(retailer, on = 'retailer_code', how='left')
#del tpp, retailer
#gc.collect()
#
#products = product_attr.join(product_desc, on = 'upc', how = 'left')
#del product_attr, product_desc
#gc.collect()
#
#master = (
#        tpp_r.join(products, on = 'upc')
#        .collect(streaming=False)
#        .write_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/master_panel.parquet')
#)
#del tpp_r, products
#gc.collect()
#
#print("done")

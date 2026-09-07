from itertools import product
import polars as pl
import scipy as sp
import gc

# data sets being loaded
dat = ['panelists', 'purchases', 'trips', 'product_attr', 'product_desc', 'retailer']
years = [2022, 2023, 2024]

# loop for loading all parquet
frame = {}
for d, y in product(dat, years):
    print(f'Loading {d}, {y}')
    lazy_df = pl.scan_parquet(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{d}_{y}.parquet').rename(str.lower)
    frame[(d,y)] = lazy_df
    print(f'{d} loaded')
combined_frame = {}
for d in dat:
    yearly_lfs   = [frame[(d,y)] for y in years]
    combined_lfs = pl.concat(yearly_lfs, how='diagonal_relaxed')
    combined_frame[d] = combined_lfs
    globals()[f'{d}'] = combined_lfs 

print(panelists.columns)

# merging trips and panelists
# Build a single set containing all numbers across all ranges
valid_codes = set().union(
    range(13010, 13300),
    range(17031, 18128),
    range(48015, 48482),
    range(8001, 8126),
    range(4007, 4026),
    range(34001, 34004),
    range(42017, 42102),
    [6073],
    range(19001, 19198)
)
panelists = panelists.filter(pl.col('fips_county_code').is_in(valid_codes))
trip_panelists = trips.join(panelists, on = ['panel_year', 'household_code'], how='left')
# Atlanta, Chicago, Denver, Des Moines, San Diego, Philly, Houston, Phoenix 
# .is_in(13010:13299), (17031:18127), (48015:48481), (08001:08125), (04007:04025), ((34001:34003), (42017:42101)), 06073, (19001:19197)
del trips, panelists
gc.collect()

# merge purchases and trip_panelists

tpp = trip_panelists.join(purchases, on = 'trip_code_uc', how='left')
del trip_panelists, purchases
gc.collect()

tpp_r = tpp.join(retailer, on = 'retailer_code', how='left')
del tpp, retailer
gc.collect()

products = product_attr.join(product_desc, on = 'upc', how = 'left')
del product_attr, product_desc
gc.collect()

master = (
        tpp_r.join(products, on = 'upc')
        .collect(streaming=False)
        .write_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/master_panel.parquet')
)
del tpp_r, products
gc.collect()

print("done")

from itertools import product
import polars as pl
import scipy as sp
import gc

# data sets being loaded
dat = ['panelists', 'purchases', 'trips', 'product_attr', 'product_desc', 'retailers']
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

# bring naming convention in line with other files
panelists = panelists.rename({"household_cd": "household_code"})


# merging trips and panelists
trip_panelists = trips.join(panelists, on = ['panel_year', 'household_code'], how='left')
del trips, panelists
gc.collect()

# merge purchases and trip_panelists

tpp = trip_panelists.join(purchases, on = 'trip_code_uc', how='left')
del trip_panelists, purchases
gc.collect()

tpp_r = tpp.join(retailers, on = 'retailer_code', how='left')
del tpp, retailers
gc.collect()

products = product_attr.join(product_desc, on = ['upc', 'upc_ver_uc'], how = 'left')
del product_attr, product_desc
gc.collect()

master = (
        tpp_r.join(products, on = ['upc', 'upc_ver_uc'])
        .sink_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/master_panel.parquet')
)
del tpp_r, products
gc.collect()

print("done")

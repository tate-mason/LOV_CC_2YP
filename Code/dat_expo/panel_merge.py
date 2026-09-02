from itertools import product
import polars as pl
import scipy as sp
import gc

# data sets being loaded
dat = ['panelists', 'purchases', 'trips']
years = [2017, 2018, 2019]

# loop for loading all parquet
frame = {}
for d, y in product(dat, years):
    print(f'Loading {d}, {y}')
    lazy_df = pl.scan_parquet(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/{d}_{y}.parquet')
    frame[(d,y)] = lazy_df
    print(f'{d} loaded')
combined_frame = {}
for d in dat:
    yearly_lfs   = [frame[(d,y)] for y in years]
    combined_lfs = pl.concat(yearly_lfs, how='vertical')
    combined_frame[d] = combined_lfs
    globals()[f'{d}'] = combined_lfs 

products  = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/products.parquet')
retailers = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/retailers.parquet')
# bring naming convention in line with other files
panelists = panelists.rename({"Household_Cd": "household_code"})


# merging trips and panelists
trip_panelists = trip.join(panelists, on = ['panel_year', 'household_code'], how='left')
del trips, panelists
gc.collect()

# merge purchases and trip_panelists

tpp = trip_panelists.join(purchases, on = ['household_code', 'trip_code_uc'], how='left')
del trip_panelists, purchases
gc.collect()

tpp_r = tpp.join(retailers, on = 'retailer_code', how='left')
del tpp, retailers
gc.collect()

master = (
        tpp_r.join(products, on = ['upc', 'upc_ver_uck'])
        .sink_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet')
)
del tpp_r, products
gc.collect()

print("done")

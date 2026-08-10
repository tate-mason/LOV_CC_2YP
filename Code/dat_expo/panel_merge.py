import pandas as pd
import polars as pl
import scipy as sp
import gc

# data sets being loaded
dat = ['panelists', 'purchases', 'products', 'trips']

# loop for loading all parquet
frame = {}
for d in dat:
    print(f'Loading {d}')
    frame[d] = pl.scan_parquet(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/{d}.parquet')
    print(f'{d} loaded')

# print column names
for name, d in frame.items():
    print(name, d.columns)
    globals()[name] = d

# bring naming convention in line with other files
panelists = panelists.rename({"Household_Cd": "household_code"})


# merging trips and purchases
trip_purchase = trips.join(purchases, on='trip_code_uc', how='left')
print('finished merging trips and purchases')
# delete base dataset
del trips, purchases
gc.collect()

# merge panelists and trip_purchases

tpp = trip_purchase.join(panelists, on='household_code', how = 'left')
del panelists, trip_purchase
gc.collect()

master = tpp.join(products, on='upc', how='left').sink_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet')
del products
gc.collect()

panel = pl.read_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet')
print(panel['purchase_date'].n_unique())
print("done")

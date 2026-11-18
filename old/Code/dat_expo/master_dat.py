
# Empirical I/O is permitted only on the research server, never on this computer.
from pathlib import Path as _PolicyPath
import sys as _policy_sys
_policy_root = next(p for p in _PolicyPath(__file__).resolve().parents if (p / "Code/privacy/server_data_policy.py").is_file())
_policy_sys.path.insert(0, str(_policy_root / "Code/privacy"))
from server_data_policy import server_only, require_server, require_legacy_server
require_legacy_server()
import polars as pl
import pandas as pd
import pyarrow as pa
import gc

# Try to load in individual TSV files and merge one by one to find problem

purchases = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/purchases.parquet')
print(purchases.columns)

trips = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/trips.parquet')
print(trips.collect()['purchase_date'].head())
trips = trips.with_columns(
    (pl.col('purchase_date').str.replace_all('-',''))
    .cast(pl.Int64)
    .alias('week_end')
)
print(trips.columns)

movement_3603 = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/RMS/movement_3603.parquet')
print(movement_3603.columns)

movement_3612 = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/RMS/movement_3612.parquet')
print(movement_3612.columns)

movement = pl.concat([movement_3603, movement_3612])

purchases_trips = purchases.join(trips, on='trip_code_uc', how='left')
del purchases, trips
gc.collect()

pt = purchases_trips.collect()
mv = movement.collect()

print(pt[['store_code_uc', 'upc', 'purchase_date', 'week_end']].head())
print(mv[['store_code_uc', 'upc', 'week_end']].head())


print(pt.dtypes)
print(mv.dtypes)


pt_movement = purchases_trips.join(movement, on=['store_code_uc', 'upc', 'week_end'])
del movement_3603, movement_3612, movement
gc.collect()

print(pt_movement.columns)

panelists = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/panelists.parquet')
panelists = panelists.rename({c: c.lower() for c in panelists.collect_schema().names()})
panelists = panelists.rename({"household_cd": "household_code"})

pptm = pt_movement.join(panelists, on='household_code', how='left')
print(pptm.columns)

products = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/RMS/2014/products.parquet')

ppptm = pptm.join(products, on = 'upc', how='left').collect()
print(ppptm.columns)

print(purchases_trips.collect().shape)
print(pt_movement.collect().shape)
print(pptm.collect().shape)
print(ppptm.shape)
print(ppptm.null_count())
print(ppptm.head())


ppptm.write_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master.parquet')
print('Data Saved')


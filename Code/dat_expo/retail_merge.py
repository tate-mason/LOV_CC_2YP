import polars as pl
import gc

# Loading data

dat = [
    'rms',
    'products',
    'movement',
    'stores'
]

# loop for loading all
frame = {}
for d in dat:
    print(f'loading {d}')
    frame[d] = pl.scan_parquet(f'/scratch/dtm63837/Kilts_Panel/RMS/{d}.parquet')
    print(f'{d} loaded')

for name, d in frame.items():
    print(name, d.collect_schema().names())
    globals()[name] = d

# Merge RMS and products

stores   = stores.rename({'year':'panel_year'})
market_stores = stores.select('store_code_uc').unique()
filtered_movement = movement.join(market_stores, on='store_code_uc', how='inner')

del movement
gc.collect()

step1 = products.join(rms, on=['upc', 'upc_ver_uc'], how='inner')
n1    = step1.select(pl.len()).collect(engine='streaming').item()
print('step1 rows:', n1)

step1.sink_parquet('/scratch/dtm63837/Kilts_Panel/RMS/step1.parquet')

del products, rms, step1
gc.collect()

step2 = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/RMS/step1.parquet').join(filtered_movement, on = 'upc', how='left')
n2    = step2.select(pl.len()).collect(engine='streaming').item()
print('step2 rows:', n2)

step2.sink_parquet('/scratch/dtm63837/Kilts_Panel/RMS/step2.parquet')

del step2, filtered_movement
gc.collect()

master = pl.scan_parquet('/scratch/dtm63837/Kilts_Panel/RMS/step2.parquet').join(stores, on = ['panel_year', 'store_code_uc'], how='left')
n3     = master.select(pl.len()).collect(engine='streaming').item()
print('master rows:', n3)
print('week_end', master.pl.col('week_end').unique())

master.sink_parquet('/scratch/dtm63837/Kilts_Panel/RMS/master_retail.parquet')

del master, stores
gc.collect()

print('-'*60)
print('Retail Data Merged')
print('-'*60)

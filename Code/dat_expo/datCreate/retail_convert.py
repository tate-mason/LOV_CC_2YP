import polars as pl
import gc
import os

outpath = '/scratch/dtm63837/Kilts_Panel/RMS/'

print("Loading RMS Data:")
rms = ( 
    pl.scan_csv('/scratch/dtm63837/Kilts_Panel/RMS/2014/Annual_Files/rms_versions_2014.tsv', separator = '\t', quote_char=None, encoding='utf8-lossy')
)
rms.sink_parquet(f'{outpath}rms.parquet')
print("RMS Data Converted")

print('Loading Store Data')
stores = (
    pl.scan_csv('/scratch/dtm63837/Kilts_Panel/RMS/2014/Annual_Files/stores_2014.tsv', separator = '\t', quote_char=None, encoding='utf8-lossy')
    .filter(pl.col('dma_code').is_in([524, 602, 751, 825]))
)
stores.sink_parquet(f'{outpath}stores.parquet')
print('Store Data Converted')

print('Loading Movement Data')
raw_movement3603 =(
    pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/RMS/2014/Movement_Files/2510_2014/3603_2014.tsv', separator='\t', quote_char=None, encoding='utf8-lossy')
    #.sink_parquet('../../movement_3603.parquet')
)

raw_movement3612 = (
    pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/RMS/2014/Movement_Files/2510_2014/3612_2014.tsv', separator='\t', quote_char=None, encoding='utf8-lossy')
    #.sink_parquet('../../movement_3612.parquet')
)

raw_move = (
    pl.concat([raw_movement3603, raw_movement3612])
)
raw_move.sink_parquet(f'{outpath}movement.parquet')

print('Movement Data Converted')

print("loading products")
products = (
    pl.scan_csv('/scratch/dtm63837/Kilts_Panel/RMS/Master_Files_2006-2020/Latest/products.tsv', separator='\t', quote_char=None, encoding='utf8-lossy')
    .select(['upc', 'upc_ver_uc', 'product_group_code', 'product_module_code', 'brand_code_uc', 'size1_code_uc', 'size1_units'])
)
products.sink_parquet(f'{outpath}products.parquet')
print('Products converted')

#print(products.columns)
#print(rms.columns)
#print(stores.columns)
#print(raw_movement3603.columns)
#print(raw_movement3612.columns)

print('-'*60)
print('All Retail Files Coverted from .csv to .parquet -- Move to Merge')
print('-'*60)

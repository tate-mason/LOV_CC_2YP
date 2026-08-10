import polars as pl

print("loading panelists")
panelists = (
    pl.scan_csv('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/2014/Annual_Files/panelists_2014.tsv', separator='\t')
    #.filter(pl.col('dma_code').is_in([524, 602, 751, 825]))
    .sink_parquet('../../panelists.parquet')
)
print('Panelists converted')

print('loading products')
products = (
    pl.scan_csv(
        '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/Master_Files/Latest/products.tsv', 
        separator='\t', 
        quote_char=None,
        encoding='utf8-lossy'
    )
    .sink_parquet('../../products.parquet')
)
print('Products converted')

print('loading purchases')
purchases = (
    pl.scan_csv('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/2014/Annual_Files/purchases_2014.tsv', separator='\t')
    .sink_parquet('../../purchases.parquet')
)
print('Purchases converted')

print('loading trips')
trip = (
    pl.scan_csv('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/2014/Annual_Files/trips_2014.tsv', separator='\t')
    .sink_parquet('../../trips.parquet')
)
print('Trips converted')

print('-'*60)
print('All Files Converted from .csv to .parquet - Move to Merge')
print('-'*60)

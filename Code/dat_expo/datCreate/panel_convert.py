import polars as pl

years = [2017, 2018, 2019]

for y in years:
    print(f"loading panelists {y}")
    panelists = (
        pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{y}/Annual_Files/panelists_{y}.tsv', separator='\t')
        .sink_parquet(f'../../nielsen_extracts/panelists_{y}.parquet')
    )
    print(f'Panelists converted for {y}')

    print('loading purchases')
    purchases = (
        pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{y}/Annual_Files/purchases_{y}.tsv', separator='\t')
        .sink_parquet(f'../../nielsen_extracts/purchases_{y}.parquet')
    )
    print('Purchases converted')

    print('loading trips')
    trip = (
        pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{y}/Annual_Files/trips_{y}.tsv', separator='\t')
        .sink_parquet(f'../../nielsen_extracts/trips_{y}.parquet')
    )
    print('Trips converted')

print(f'loading products')
products = (
    pl.scan_csv(
        '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/Master_Files/Latest/products.tsv', 
        separator='\t', 
        quote_char=None,
        encoding='utf8-lossy'
    )
    .sink_parquet('../../nielsen_extracts/products.parquet')
)
print('Products converted')

print(f'loading retailers')
retailers = (
    pl.scan_csv(
        '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/Master_Files/Latest/retailers.tsv',
        separator='\t',
        quote_char=None,
        encoding='utf8-lossy'
    )
    .sink_parquet('../../nielsen_extracts/retailers.parquet')
)
print('Retailers converted')

print('-'*60)
print('All Files Converted from .tsv to .parquet - Move to Merge')
print('-'*60)

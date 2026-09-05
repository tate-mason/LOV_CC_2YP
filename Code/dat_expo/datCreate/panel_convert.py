import polars as pl

years = [2022, 2023, 2024]

for y in years:
    print(f"loading panelists {y}")
    panelists = (
        pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{y}/Annual_Files/panelist.tsv',
        separator='\t',
        quote_char=None,
        infer_schema_length=0).rename(str.lower)
        .sink_parquet(f'../../nielsen_extracts/HMS/panelists_{y}.parquet')
    )
    print(f'Panelists converted for {y}')

    print('loading purchases')
    purchases = (
        pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{y}/Annual_Files/purchase.tsv', 
        separator='\t',
        quote_char=None,
        infer_schema_length=0).rename(str.lower)
        .sink_parquet(f'../../nielsen_extracts/HMS/purchases_{y}.parquet')
    )
    print('Purchases converted')

    print('loading trips')
    trip = (
        pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{y}/Annual_Files/trip.tsv', 
        separator='\t',
        quote_char=None,
        infer_schema_length=0).rename(str.lower)
        .sink_parquet(f'../../nielsen_extracts/HMS/trips_{y}.parquet')
    )
    print('Trips converted')
    
    print('loading product attributes')
    prod_att = (
    		pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{y}/Annual_Files/productattributes.tsv', 
            separator='\t',
            quote_char=None,
            infer_schema_length=0).rename(str.lower)
        .sink_parquet(f'../../nielsen_extracts/HMS/product_attr_{y}.parquet')
    )
    print('Product Attributes Converted')
    
    print('loading product descriptions')
    prod_desc = (
		pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{y}/Annual_Files/productdesc.tsv', 
        separator='\t',
        quote_char=None,
        infer_schema_length=0).rename(str.lower)
        .sink_parquet(f'../../nielsen_extracts/HMS/product_desc_{y}.parquet')
    )
    print('descriptions converted')
    
    print('loading retailers')
    retailers = (
    		pl.scan_csv(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{y}/Annual_Files/retailer.tsv', 
            separator='\t',
            quote_char=None,
            infer_schema_length=0).rename(str.lower)
        .sink_parquet(f'/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/retailer_{y}.parquet')
    )
    print('retailers converted')




print('-'*60)
print('All Files Converted from .tsv to .parquet - Move to Merge')
print('-'*60)

import polars as pl

df = pl.scan_parquet(f'../../nielsen_extracts/HMS/trips_2022.parquet')

print(df.columns)

#print(df.select(['dma_code', 'dma_descr']))



import polars as pl

df = pl.scan_parquet(f'../../nielsen_extracts/HMS/panelists_2022.parquet')

print(df.columns.to_list())

#print(df.select(['dma_code', 'dma_descr']))



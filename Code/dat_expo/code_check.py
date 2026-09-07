import polars as pl

df = pl.read_excel('/scratch/dtm63837/Kilts_Panel/Reference_Documentation/2021-Onward_Documentation/FIPS-SMM-DMA-Conversion-Table-2021.xlsx')

print(df.columns)

#print(df.select(['dma_code', 'dma_descr']))



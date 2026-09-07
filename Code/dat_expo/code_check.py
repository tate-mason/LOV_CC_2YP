import polars as pl

df = pl.read_excel('/scratch/dtm63837/Kilts_Panel/Reference_Documentation/2021-Onward_Documetation/FIPS-SMM-DMA-Conversion-Table-2021.xlsx')

print(df.columns)



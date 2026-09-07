import polars as pl
pl.Config.set_fmt_str_lengths(1000)  # Max characters per string/cell
pl.Config.set_tbl_formatting("ASCII_FULL")

# 2. Prevent row/column truncation in the terminal/notebook
pl.Config.set_tbl_rows(-1)       # Show ALL rows (-1 means unlimited)
pl.Config.set_tbl_cols(-1)       # Show ALL columns
pl.Config.set_fmt_table_cell_list_len(-1)  # Show ALL elements in a list column

df = pl.read_excel('/scratch/dtm63837/Kilts_Panel/Reference_Documentation/2021-Onward_Documetation/FIPS-SMM-DMA-Conversion-Table-2021.xlsx')

print(df.columns)
print(df.schema)

res_DMA = (
    df 
    .with_columns(
        pl.col('DMA Name').str.strip_chars(),
        pl.col('DMA code').str.strip_chars()
    )
    .group_by('DMA Name')
    .agg(
        pl.col('DMA code')
        .filter(pl.col('DMA code').is_not_null() & (pl.col('DMA code') != ""))
        .unique()
        .sort()
        .list.join(", ")
        .alias('associated_dma'),
        # unique vals
        pl.col('DMA code')
        .n_unique()
        .alias('Unique_DMA')
    )
)

res_FIPS = (
    df
    .with_columns(
        pl.col('County Name').str.strip_chars(),
        pl.col('FIPS Code').str.strip_chars()
    )
    .group_by('County Name')
    .agg(
        pl.col('FIPS Code')
        .filter(pl.col('FIPS Code').is_not_null() & (pl.col('FIPS Code') != ""))
        .unique()
        .sort()
        .list.join(", ")
        .alias('Associated_FIPS'),
        # unique vals
        pl.col('FIPS Code')
        .n_unique()
        .alias('Unique_FIPS')
    )
)

print(res_DMA)
print('='*50)
print(res_FIPS)

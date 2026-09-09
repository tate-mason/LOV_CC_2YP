from itertools import product
import pandas as pd
pd.set_option(
    'display.max_rows', None,
    'display.max_columns', None
)

df = pd.read_excel('/scratch/dtm63837/Kilts_Panel/Reference_Documentation/2021-Onward_Documetation/FIPS-SMM-DMA-Conversion-Table-2021.xlsx', dtype=str)

mapping_df = df.iloc[:,[1, 2, 3, 4,5]].drop_duplicates().reset_index(drop=True)

print(mapping_df)

dat   = ['panelists', 'trips', 'retailers', 'product_attr', 'product_desc', 'purchases']
years = [2022, 2023, 2024]

for d, y in product(dat, years):
    market_df = pd.read_parquet(f"/scratch/dtm63837/Kilts_Panel/nielsen_extracts/HMS/{d}_{y}.parquet")
    print(market_df.columns.to_list())


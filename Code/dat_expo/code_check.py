import pandas as pd
pd.set_option('display.max_rows', None)

df = pd.read_excel('/scratch/dtm63837/Kilts_Panel/Reference_Documentation/2021-Onward_Documetation/FIPS-SMM-DMA-Conversion-Table-2021.xlsx', dtype=str)

mapping_df = df.iloc[:,[4,5]].drop_duplicates().reset_index(drop=True)

print(mapping_df)

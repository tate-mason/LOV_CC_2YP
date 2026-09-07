import pandas as pd

df = pd.read_excel('/scratch/dtm63837/Kilts_Panel/Reference_Documentation/2021-Onward_Documetation/FIPS-SMM-DMA-Conversion-Table-2021.xlsx', dtype=str)

mapping_df = df.iloc[:,[4,5]].drop_duplicates().reset_index(drop=True)

for i in range(len(mapping_df)):
    dma_code  = df.iloc[i, 4]
    dma_label = df.iloc[i, 5]

    print(f'DMA Code: {dma_code} --> DMA: {dma_label}')

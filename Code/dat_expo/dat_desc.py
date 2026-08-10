import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)

df = pl.read_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet').to_pandas()

df = df.convert_dtypes(dtype_backend='numpy_nullable')
df.columns = df.columns.str.lower()
df_sub = df[df['household_size']==1]
df_sub = df_sub[df_sub['dma_cd'].isin([524, 602, 751, 825])]
df_yogurt = df_sub[df_sub['product_group_code'] == 2510]
df_yogurt = df_yogurt[df_yogurt['product_module_code'] == 3603]

print('Size types:', df_yogurt.groupby('size1_amount')['size1_code_uc'].unique())

print('Product modules:', df_yogurt.groupby('product_module_descr')['product_module_code'].unique())
print('UPC', df_yogurt.groupby('upc_descr')['upc'].unique())
print(df_yogurt['size1_amount'].dtype)
print('Product Type', df_yogurt.groupby('upc')['upc_descr'].unique())

#df_cereal = df_sub[df_sub['product_group_code'] == 1005]
#print('Product modules:', df_cereal.groupby('product_module_descr')['product_module_code'].unique())
#print('UPC', df_cereal.groupby('upc_descr')['upc'].unique())
#print('Size types', df_cereal.groupby('size1_amount')['size1_code_uc'].unique())


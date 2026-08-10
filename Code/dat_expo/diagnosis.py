import polars as pl
import pandas as pd
from rich.traceback import install; install()
from rich.console import Console
console = Console()

agent_panel  = pl.read_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet').to_pandas()
product_panel = pl.read_parquet('/scratch/dtm63837/Kilts_Panel/RMS/master_retail.parquet').to_pandas()

agent_panel['purchase_date'] = agent_panel['purchase_date'].str.replace('-','',regex=False)
agent_panel['purchase_date'] = pd.to_datetime(agent_panel['purchase_date'], format='%Y%m%d')
agent_panel['week_end']      = agent_panel['purchase_date'] + pd.offsets.Week(weekday=5, n=0) # convert purchase date to week_end to make it work in merge
product_panel['week_end']    = pd.to_datetime(product_panel['week_end'], format='%Y%m%d')

agent_panel['store_code_uc'] = agent_panel['store_code_uc'].astype(int)
agent_panel['upc']           = agent_panel['upc'].astype(float)

console.print(agent_panel['store_code_uc'].isna().sum())
console.print(agent_panel['upc'].isna().sum())

console.print(product_panel['week_end'].head)
console.print(product_panel['week_end'].isna().sum())
console.print(product_panel['store_code_uc'].isna().sum())


product_panel = product_panel.dropna(subset=(['week_end', 'store_code_uc']))
agent_panel   = agent_panel.dropna(subset=('upc'))

console.print(product_panel['week_end'].isna().sum())
console.print(agent_panel['upc'].isna().sum())
console.print(product_panel['store_code_uc'].isna().sum())

agent_panel["upc"] = agent_panel["upc"].astype("Int64")
product_panel['store_code_uc'] = product_panel['store_code_uc'].astype("Int64")

console.print(agent_panel["upc"].dtype, product_panel["upc"].dtype)
console.print(agent_panel["upc"].head())
console.print(product_panel["upc"].head())
console.print(agent_panel["store_code_uc"].dtype, product_panel["store_code_uc"].dtype)
console.print(agent_panel["store_code_uc"].head())
console.print(product_panel["store_code_uc"].head())
console.print(len(set(agent_panel["upc"]) & set(product_panel["upc"])))
console.print(len(set(agent_panel["store_code_uc"]) & set(product_panel["store_code_uc"])))
console.print(len(set(agent_panel["week_end"]) & set(product_panel["week_end"])))

console.print(agent_panel["upc"].head(10).tolist())
console.print(product_panel["upc"].head(10).tolist())

console.print(agent_panel['store_code_uc'].head(10).tolist())
console.print(product_panel['store_code_uc'].head(10).tolist())

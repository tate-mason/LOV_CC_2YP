# data loading
import polars as pl
import pandas as pd
# numerical and statistical analysis
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import poisson as poisson_dist
#output
from rich.console import Console
from rich.traceback import install; install()
console = Console()

hms_path          = '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet'
rms_path          = '/scratch/dtm63837/Kilts_Panel/RMS/master_retail.parquet'
agent_panel   = (
        pl.read_parquet(hms_path)
        .rename({'DMA_Cd':'dma_code'})
        .filter((pl.col('product_module_code') == 3612) | (pl.col('product_module_code') == 3603))
        .filter(pl.col('dma_code').is_in([524, 602, 751, 825]))
        .to_pandas() # load in parquet and convert to pandas 
)
product_panel = (
        pl.read_parquet(rms_path)
        .filter((pl.col('product_module_code') == 3612) | (pl.col('product_module_code') == 3603))
        .to_pandas()                                                                                                          
)

agent_panel['purchase_date']   = agent_panel['purchase_date'].str.replace('-','',regex=False)
agent_panel['purchase_date']   = pd.to_datetime(agent_panel['purchase_date'], format='%Y%m%d')
agent_panel['week_end']        = agent_panel['purchase_date'] + pd.offsets.Week(weekday=5, n=0) # convert purchase date to week_end to make it work in merge
agent_panel["upc"]             = agent_panel["upc"].astype("Int64")
product_panel['store_code_uc'] = product_panel['store_code_uc'].astype("Int64")
product_panel['week_end']      = pd.to_datetime(product_panel['week_end'], format='%Y%m%d')
product_panel                  = product_panel.dropna(subset=('week_end'))

console.print(agent_panel['store_code_uc'].dtype, product_panel['store_code_uc'].dtype)
console.print(agent_panel['upc'].dtype, product_panel['upc'].dtype)
console.print(product_panel['week_end'].dt.dayofweek.value_counts())
console.print(agent_panel['week_end'].min(), agent_panel['week_end'].max())
console.print(product_panel['week_end'].min(), product_panel['week_end'].max())

print('store overlap:', len(set(agent_panel['store_code_uc']) & set(product_panel['store_code_uc'])),
      'of', agent_panel['store_code_uc'].nunique(), 'agent stores')

print('upc overlap:', len(set(agent_panel['upc']) & set(product_panel['upc'])),
      'of', agent_panel['upc'].nunique(), 'agent upcs')

print('week_end overlap:', len(set(agent_panel['week_end']) & set(product_panel['week_end'])),
      'of', agent_panel['week_end'].nunique(), 'agent weeks')

agent_keys   = set(zip(agent_panel['store_code_uc'], agent_panel['week_end'], agent_panel['upc']))
product_keys = set(zip(product_panel['store_code_uc'], product_panel['week_end'], product_panel['upc']))
print('combined key overlap:', len(agent_keys & product_keys), 'of', len(agent_keys), 'agent rows')

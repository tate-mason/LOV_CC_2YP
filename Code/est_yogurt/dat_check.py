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

console.print(agent_panel['store_code_uc'].dtype, product_panel['store_code_uc'].dtype)
console.print(agent_panel['upc'].dtype, product_panel['upc'].dtype)
console.print(product_panel['week_end'].dt.dayofweek.value_counts())
console.print(agent_panel['week_end'].min(), agent_panel['week_end'].max())
console.print(product_panel['week_end'].min(), product_panel['week_end'].max())

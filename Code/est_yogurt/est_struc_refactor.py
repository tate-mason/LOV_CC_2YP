# data loading
import polars as pl
import pandas as pd
# numerical and statistical analysis
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import poisson as poisson_dist
import statsmodels.formula.api as smf
#output
from rich.console import Console
from rich.traceback import install; install()
console = Console()

import time

console.print('='*60)
console.print('Data Loading and Manipulation')
console.print('='*60)

hms_path          = '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet'
rms_path          = '/scratch/dtm63837/Kilts_Panel/RMS/master_retail.parquet'
out_path          = '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master.parquet'

# Building the merged dataset from HMS and RMS
def build_merged_panel(hms_path, rms_path, out_path, force):
    import os
    if os.path.exists(out_path) and not force:
        master_df = pl.read_parquet(out_path)
    else:
        agent_panel   = (
                pl.read_parquet(hms_path)
                .rename({'DMA_Cd':'dma_code'})
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

        agent_panel['store_code_uc']   = agent_panel['store_code_uc'].astype('Int64')
        product_panel['store_code_uc'] = product_panel['store_code_uc'].astype('Int64')
        agent_panel['upc']   = agent_panel['upc'].astype('Int64')
        product_panel['upc'] = product_panel['upc'].astype('Int64')

        product_panel['week_end']      = pd.to_datetime(product_panel['week_end'], format='%Y%m%d')
        product_panel                  = product_panel.dropna(subset=['week_end'])
        master_df = agent_panel.merge(product_panel, on=['store_code_uc', 'week_end', 'upc'], how='left')                  # merge the HMS and RMS on store x week x product
        master_df = pl.from_pandas(master_df) # make polars
        master_df.write_parquet(out_path)     # write df
    return master_df # return merged data

master_df = build_merged_panel(hms_path, rms_path, out_path, force=False) # call function
master_df = master_df.rename({'product_module_code_x':'product_module_code',
                              'product_group_code_x':'product_group_code', 
                              'size1_code_uc_x':'size1_code_uc',
                              'size1_units_x':'size1_units',
                              'dma_code_x':'dma_code'
                              })
master_df = master_df.to_pandas() # make pandas format
master_df.columns = master_df.columns.str.lower() # column names lower

flavors   = pd.read_csv('/scratch/dtm63837/Kilts_Panel/RMS/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv') # load in flavors documentation
master_df = master_df.merge(flavors, on = 'upc', how = 'left') # merge flavors data

master_df = master_df.assign(
    flavor = np.select(
        [
            master_df['flavor_code'].isin([139, 44642, 75721, 2180]), # apple
            master_df['flavor_code'].isin([22053, 24357, 52953, 74408, 17159, 23721]), # blueberry
            master_df['flavor_code'].isin([11214, 20888, 17849, 17849]), # banana
            master_df['flavor_code'].isin([904, 13314, 1169, 1174, 5651]), # cherry
            master_df['flavor_code'].isin([73560, 3075, 73560]), # key lime
            master_df['flavor_code'].isin([3107, 22916, 3122, 6061]), # lemon
            master_df['flavor_code'].isin([3943, 3060, 70529, 10808, 3985, 23346]), # peach
            master_df['flavor_code'].isin([6352, 41654, 41681, 78681, 41634, 6912]), # raspberry
            master_df['flavor_code'].isin([23344, 16007, 16102, 66438, 16194, 30581, 45574, 72000, 17110]), # strawberry
            master_df['flavor_code'].isin([5537, 5539, 66938, 5658, 72317]), # vanilla
            master_df['flavor_code'].isin([66438, 66684, 71101, 72483,19061, 16102,  61082, 61487, 57428, 67420, 78857, 1154, 26050, 1216]), # mixed flavors
            master_df['flavor_code'].isin([57129, 76690, 16200, 62349, 16199, 16182, 72290, 32300, 72289, 16102, 72292, 3465, 68109, 52953, 72288]), # mixed berry
            master_df['flavor_code'].isin([4167]) #plain 
        ],
        [1,2,3,4,5,6,7,8,9,10,11,12,13],
        default=np.nan # na any non-specified
    )
)
master_df['flavor_binary'] = (
    master_df['flavor'] == 13 # if plain, flavor_binary == 1
).astype(int)

full_panel = master_df.copy() # copy of all people
master_df = master_df[master_df['product_module_code'].isin([3612, 3603])] # filter to yogurt purchasers
master_df = master_df.dropna(subset=['price']) # drop na for price

#==================================#
# Price Endogeneity - IV           #
#==================================#

market_price = (
    master_df.groupby(['upc', 'week_end', 'dma_code'])['price']
    .mean()
    .reset_index()
)
totals = (
    market_price.groupby(['upc', 'week_end'])['price']
    .agg(['sum', 'count'])
    .reset_index()
    .rename(columns={'sum': 'price_sum_all', 'count': 'n_markets_all'})
)
market_price = market_price.merge(totals, on=['upc', 'week_end'], how='left')
market_price['price_iv'] = (
    (market_price['price_sum_all'] - market_price['price']) /
    (market_price['n_markets_all'] - 1)
)
market_price['price_iv'] = market_price['price_iv'].replace([np.inf, -np.inf], np.nan)

master_df = master_df.merge(
    market_price[['upc', 'week_end', 'dma_code', 'price_iv']],
    on=['upc', 'week_end', 'dma_code'],
    how='left'
)
iv_res = smf.ols('price ~ price_iv + size1_amount + C(week_end)', data=master_df, missing='drop').fit() # run ols and drop NA
master_df['iv_resid'] = iv_res.resid # recover residuals
master_df = master_df.dropna(subset=['iv_resid'])   # was missing the reassignment -- dropna alone doesn't mutate in place

#=================================#
# 

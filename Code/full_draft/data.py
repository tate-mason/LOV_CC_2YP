"""
    Code for data section
    - Summ stats for full sample
    - Summ stats for yogurt sample
    - Switching heatmaps 
    - Switching behavior over time graph
"""

#===================================#
# Loading Packages & Dependencies   #
#===================================#

# Data management
import polars as pl
import pandas as pd
# Numerical manipulation
import scipy as sp
import numpy as np
# Graphing
import matplotlib.pyplot as plt
import seaborn as sns
# Output formatting
from rich.traceback import install; install()
from rich.console import Console
console=Console() # alias for function

#==================================#
# Loading Data                     #
#==================================#

hms_path  = '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet' # HomeScan
rms_path  = '/scratch/dtm63837/Kilts_Panel/RMS/master_retail.parquet' # MarketScan
out_path  = '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master.parquet' # Merged
flav_path = '/scratch/dtm63837/Kilts_Panel/RMS/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv' # Flavors

#=== Agent Panel Operations ===#

# Loading agent panel

agent_panel   = (
    pl.read_parquet(hms_path)      # call the local path set above
    .rename({'DMA_Cd':'dma_code'}) # rename for easier filtering/merging later
    .to_pandas()                   # convert from LazyFrame to pandas DataFrame
)

# Agent panel cleaning
agent_panel                             = agent_panel.convert_dtypes(dtype_backend = 'numpy_nullable') # make data numpy compatible
agent_panel.columns                     = agent_panel.columns.str.lower() # make column names lowercase

agent_panel                    = agent_panel[agent_panel['household_size'] == 1] # subset to single agent hh
agent_panel                    = agent_panel[agent_panel.groupby('household_code')['trip_code_uc'].transform('count') > 2] # at least 2 shopping trips
agent_panel                    = agent_panel[agent_panel['size1_units'] == 'OZ'] # keep only yogurt measured in ounces
agent_panel                    = agent_panel[(agent_panel['size1_amount'] > 5) | (agent_panel['size1_amount'] < 8)] # restrict to cups of yogurt

agent_panel['purchase_date']   = agent_panel['purchase_date'].str.replace('-','',regex=False)   # get rid of hyphens in purchase date
agent_panel['purchase_date']   = pd.to_datetime(agent_panel['purchase_date'], format='%Y%m%d')  # convert to YearMonthDay format
agent_panel['week_end']        = agent_panel['purchase_date'] + pd.offsets.Week(weekday=5, n=0) # create a week_end variable like RMS has 

agent_panel['store_code_uc']   = agent_panel['store_code_uc'].astype('Int64') # convert code to Int64 datatype to match RMS
agent_panel['upc']             = agent_panel['upc'].astype('Int64')           # same as above

#=== Product Panel Operations ===#

# Loading product panel

product_panel = (
    pl.read_parquet(rms_path) # load data in via local path
    .filter((pl.col('product_module_code')==3612) | (pl.col('product_module_code')==3603)) # filter to yogurt
    .to_pandas()              # convert from LazyFrame to DataFrame
) 

# Product panel cleaning

product_panel['week_end']      = pd.to_datetime(product_panel['week_end'], format='%Y%m%d') # convert date format
product_panel                  = product_panel.dropna(subset=['week_end'])                  # drop NA values for dates

product_panel['store_code_uc'] = product_panel['store_code_uc'].astype('Int64')             # convert store code to Int64 type
product_panel['upc']           = product_panel['upc'].astype('Int64')                       # same as above

#=== Merging Flavor Date ===#

flavors      = pd.read_csv(flav_path) # load in flavors documentation

# Agent merge and clean

agent_master = agent_panel.merge(flavors, on='upc', how='left') # merge flavors on UPC codes with a left join
agent_panel  = agent_master.dropna(subset=['quantity', 'product_group_code', 'flavor_code', 'flavor_descr']) # drop NA for key var after merge
agent_master['plain'] = (
        agent_master['flavor_code'] == 4167 # 1 if plain 0 if flavored
).astype(int)
agent_master['yogurt_purchase'] = (
    (df_yogurt['product_module_code'].isin([3612,3603]) & (df_yogurt['quantity']>0)) # create dummy for HH who bought at least one yogurt product
).astype(int)
agent_master['no_yogurt']       = (
    1 - df_yogurt['yogurt_purchase'] # 0 when purchased, 1 when no purchase
)
outside_option = agent_master.groupby(['household_code', 'trip_code_uc']).agg({
    'no_yogurt': 'sum',        # sum of no purchase occ.
    'yogurt_purchase': 'count' # count of yogurt purchasers by trip code
}).assign(
    total_occasions = lambda x: x['no_purchase'] + x['yogurt_purchase'] # total purchase occ.
)
outside_option['outside_option_rate'] = (
    outside_option['no_purchase'] / outside_option['total_occasions']
) # variable for the rate of taking outside option (sum of no purchase occ. / total occ.)
console.print(agent_master['plain'].value_counts())           # print # of individuals who purchased plain v other
console.print(agent_master['yogurt_purchase'].value_counts()) # number of purchasers vs non-purchasers

# Product merge and clean
product_master = prduct_panel.merge(flavors, on='upc', how='left')
product_master['plain'] = (
    product_master['flavor_code'] == 4167 # same as above
).astype(int)

#==========================#
# Summary Statistics       #
#==========================#

#=== Agent Data Stats ===#

"""
Full Household:
    - n. HH
    - mean trip number per HH
    - mean yogurt purchases per HH
    - mean/median income per HH
    - n. each race
    - mean taking outside option
    - coupon users
Yogurt Only:
    - percent ever-switch 
    - average time on a flavor
    - mean times switching
    - mean consecutive buys
"""

agent_yogurt = agent_master.copy() # copy full sample
agent_yogurt = agent_yogurt[agent_yogurt['product_module_code'].isin([3612, 3603])] # subset to yogurt purchases

console.print(
    f'Number of households in full sample:                  {agent_master['household_code'].nunique()}\n',
    f'Mean number of trips per HH:                          {agent_master.groupby('household_code')['trip_code_uc'].nunique().mean()}\n',
    f'Number of yogurt purchases among purchasers per trip: {agent_yogurt.groupby(['household_code', 'trip_code_uc', 'flavor'])['quantity'].mean()}\n',
    f'Mean household income:                                {agent_master['household_income'].mean()}\n',
    f'Median household income:                              {agent_master['household_income'].median()}\n',
    f'Racial makeup of sample:                              {agent_master.groupby('race')['household_code'].nunique()}\n',
    f'Percent taking outside option each trip:              {outside_option['outside_option_rate'].mean()}\n',
    f'Percent purchasing with coupon:                       {agent_master.groupby(['household_code', 'deal_flag_uc'])['yogurt_buy'].mean()}\n',
)

#=== Switching Stats ===#

agent_yogurt = agent_yogurt.sort_values(['household_code', 'trip_code_uc']) # sort by time and household
agent_yogurt['new_flavor'] = (
    (agent_yogurt['plain']          != df_yogurt.groupby('household_code')['plain'].shift(1)) |
    (agent_yogurt['household_code'] != df_yogurt['household_code'].shift(1))
).astype(int) # dummy for if a household switched flavors between trips 
agent_yogurt['flavor_spell_id']      = agent_yogurt.groupby('household_code')['new_flavor'].cumsum() # count of periods on new flavor
agent_yogurt['flavor_spell_buys']    = agent_yogurt.groupby('household_code')['flavor_spell_id'].cumcount() + 1 # consecutive periods on flavor
agent_yogrut['prev_flavor']          = agent_yogurt.groupby('household_code')['plain'].shift(1) # last purchased flavor
agent_yogurt['spell_length']         = agent_yogurt.groupby(['household_code', 'flavor_spell_id'])['flavor_spell_buys'].transform('max') # get the number of buys in the flavor spell
agent_yogurt['switched']             = (
    agent_yogurt['plain']           != agent_yogurt['prev_flavor']
).astype(int) # ever-switch indicator
agent_yogurt['returned']             = agent_yogurt.groupby('household_code')['plain'].transform(
        lambda x: x.shift(1).isin(x.shift(-1))
) # indicator for returning to a previous flavor
switching_sample = agent_yogurt[agent_yogurt['swithced'] == 1][[
    'household_code',
    'trip_code_uc',
    'plain',
    'spell_length'
]] # filter to switchers

console.print(
    f'Mean consecutive buys by flavor x hh: {agent_yogurt.groupby(['household_code','plain'])['flavor_spell_buys'].mean()}\n',
    f'Mean times switching by flavor x hh:  {agent_yogurt.groupby(['household_code', 'plain'])['new_flavor'].count().mean()}\n',
    f'Percent of HH who ever-switch:        {(switching_sample['household_code'].count() / agent_yogurt['household_code'].count())}\n',
    f'Average time spent on each flavor:    {agent_yogurt.groupby(['prev_flavor', 'plain'])['spell_length'].mean()}'
)

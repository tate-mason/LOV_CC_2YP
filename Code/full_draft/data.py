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

# Loading merged panel

merged_panel = (
    pl.read_parquet(out_path)
    .rename({'dma_code_x': 'dma_code'}) # Clean it up first
    .filter(pl.col('size1_amount').is_between(5, 8))
    .filter(pl.col('dma_code').is_in([524, 602, 751, 825]))
    .to_pandas()
)

# Loading agent panel

agent_panel   = (
    pl.read_parquet(hms_path)      # call the local path set above
    .rename({'DMA_Cd':'dma_code'}) # rename for easier filtering/merging later
    .filter(pl.col('dma_code').is_in([524, 602, 751, 825])) # filter to 4 markets
    .to_pandas()                   # convert from LazyFrame to pandas DataFrame
)


# Agent panel cleaning
agent_panel                             = agent_panel.convert_dtypes(dtype_backend = 'numpy_nullable') # make data numpy compatible
agent_panel.columns                     = agent_panel.columns.str.lower() # make column names lowercase

agent_panel                    = agent_panel[agent_panel['household_size'] == 1] # subset to single agent hh
agent_panel                    = agent_panel[agent_panel.groupby('household_code')['trip_code_uc'].transform('count') > 2] # at least 2 shopping trips
agent_panel                    = agent_panel[agent_panel['size1_units'] == 'OZ'] # keep only yogurt measured in ounces
agent_panel                    = agent_panel[agent_panel['size1_amount'].between(5,8)] # restrict to cups of yogurt

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

#=== Merging Flavor Data ===#

flavors      = pd.read_csv(flav_path) # load in flavors documentation

# Agent merge and clean
print("Columns in flavors.csv:", flavors.columns.tolist())
print("Columns in merged_panel:", merged_panel.columns.tolist())
print("dtypes:", merged_panel['upc'].dtype, flavors['upc'].dtype)

agent_master  = agent_panel.merge(flavors, on='upc', how='left') # merge flavors on UPC codes with a left join
agent_master  = agent_master.dropna(subset=['quantity', 'product_group_code', 'flavor_code', 'flavor_descr']) # drop NA for key var after merge
agent_master  = agent_master.assign(
    flavor_class = np.select(
        [
            agent_master['flavor_code'].isin([139, 44642, 75721, 2180]), # apple
            agent_master['flavor_code'].isin([22053, 24357, 52953, 74408, 17159, 23721]), # blueberry
            agent_master['flavor_code'].isin([11214, 20888, 17849, 17849]), # banana
            agent_master['flavor_code'].isin([904, 13314, 1169, 1174, 5651]), # cherry
            agent_master['flavor_code'].isin([73560, 3075, 73560]), # key lime
            agent_master['flavor_code'].isin([3107, 22916, 3122, 6061]), # lemon
            agent_master['flavor_code'].isin([3943, 3060, 70529, 10808, 3985, 23346]), # peach
            agent_master['flavor_code'].isin([6352, 41654, 41681, 78681, 41634, 6912]), # raspberry
            agent_master['flavor_code'].isin([23344, 16007, 16102, 66438, 16194, 30581, 45574, 72000, 17110]), # strawberry
            agent_master['flavor_code'].isin([5537, 5539, 66938, 5658, 72317]), # vanilla
            agent_master['flavor_code'].isin([66438, 66684, 71101, 72483,19061, 16102,  61082, 61487, 57428, 67420, 78857, 1154, 26050, 1216]), # mixed flavors
            agent_master['flavor_code'].isin([57129, 76690, 16200, 62349, 16199, 16182, 72290, 32300, 72289, 16102, 72292, 3465, 68109, 52953, 72288]), # mixed berry
            agent_master['flavor_code'].isin([4167]) # flavor
        ],
        [1,2,3,4,5,6,7,8,9,10,11,12,13],
        default=np.nan
    )
)
agent_master = agent_master.assign(
    flavor = np.select(
        [
            agent_master['flavor_class'].isin([2,8,9,12]), # berry
            agent_master['flavor_class'] == 13,
        ],
        [1,2],
        default=0
    )
)

agent_master['yogurt_purchase'] = (
    (agent_master['product_module_code'].isin([3612,3603]) & (agent_master['quantity']>0)) # create dummy for HH who bought at least one yogurt product
).astype(int)
agent_master['no_yogurt']       = (
    1 - agent_master['yogurt_purchase'] # 0 when purchased, 1 when no purchase
)
# 1. Identify if ANY yogurt was purchased on a given store trip
trip_yogurt = agent_master.groupby(['household_code', 'trip_code_uc'])['yogurt_purchase'].max().reset_index()

# 2. A trip took the outside option if max(yogurt_purchase) == 0
trip_yogurt['chose_outside_option'] = (trip_yogurt['yogurt_purchase'] == 0).astype(int)

# 3. Overall rate of taking the outside option across all trips
outside_option_rate = trip_yogurt['chose_outside_option'].mean()

# Merged merge and clean

merged_master  = merged_panel.merge(flavors, on='upc', how='left') # merge flavors on UPC codes with a left join
merged_master = merged_master.loc[:, ~merged_master.columns.duplicated()]
merged_master = merged_master.dropna(subset=['quantity', 'product_group_code'])
merged_master  = merged_master.assign(
    flavor_class = np.select(
        [
            merged_master['flavor_code'].isin([139, 44642, 75721, 2180]), # apple
            merged_master['flavor_code'].isin([22053, 24357, 52953, 74408, 17159, 23721]), # blueberry
            merged_master['flavor_code'].isin([11214, 20888, 17849, 17849]), # banana
            merged_master['flavor_code'].isin([904, 13314, 1169, 1174, 5651]), # cherry
            merged_master['flavor_code'].isin([73560, 3075, 73560]), # key lime
            merged_master['flavor_code'].isin([3107, 22916, 3122, 6061]), # lemon
            merged_master['flavor_code'].isin([3943, 3060, 70529, 10808, 3985, 23346]), # peach
            merged_master['flavor_code'].isin([6352, 41654, 41681, 78681, 41634, 6912]), # raspberry
            merged_master['flavor_code'].isin([23344, 16007, 16102, 66438, 16194, 30581, 45574, 72000, 17110]), # strawberry
            merged_master['flavor_code'].isin([5537, 5539, 66938, 5658, 72317]), # vanilla
            merged_master['flavor_code'].isin([66438, 66684, 71101, 72483,19061, 16102,  61082, 61487, 57428, 67420, 78857, 1154, 26050, 1216]), # mixed flavors
            merged_master['flavor_code'].isin([57129, 76690, 16200, 62349, 16199, 16182, 72290, 32300, 72289, 16102, 72292, 3465, 68109, 52953, 72288]), # mixed berry
            merged_master['flavor_code'].isin([4167]) # flavor
        ],
        [1,2,3,4,5,6,7,8,9,10,11,12,13],
        default=np.nan
    )
)
merged_master = merged_master.assign(
    flavor = np.select(
        [
            merged_master['flavor_class'].isin([2,8,9,12]), # berry
            merged_master['flavor_class'] == 13,
        ],
        [1,2],
        default=0
    )
)

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
agent_yogurt = agent_yogurt[agent_yogurt['product_group_code'] == 2510] # subset to yogurt purchases
multipack_pattern   = r'MULTI|MULTIPACK|\bPK\b|\bCT\b'
agent_yogurt = agent_yogurt[
    ~agent_yogurt['upc_descr'].str.contains(multipack_pattern,case=False, na=False)
]
console.print(
    f'Number of households in full sample:                  {agent_master['household_code'].nunique()}\n',
    f'Number of yogurt purchasing households:               {agent_yogurt['household_code'].nunique()}\n',
    f'Mean number of trips per HH:                          {agent_master.groupby('household_code')['trip_code_uc'].nunique().mean()}\n',
    f'Number of yogurt purchases among purchasers per trip: {agent_yogurt.groupby(['household_code', 'trip_code_uc'])['quantity'].sum().mean()}\n',
    f'Mean household income:                                {agent_master['household_income'].mean()}\n',
    f'Median household income:                              {agent_master['household_income'].median()}\n',
    f'Racial makeup of sample:                              {agent_master.groupby('race')['household_code'].nunique()}\n',
    f'Percent taking outside option each trip:              {outside_option_rate:.2f}\n',
    f'Percent purchasing with coupon:                       {agent_yogurt.groupby(['household_code', 'trip_code_uc'])['deal_flag_uc'].max().mean()*100:.2f}\n',
)

#=== Switching Stats ===#

agent_yogurt = agent_yogurt.sort_values(['household_code', 'trip_code_uc']) # sort by time and household
agent_yogurt['new_flavor'] = (
    (agent_yogurt['flavor']          != agent_yogurt.groupby('household_code')['flavor'].shift(1)) |
    (agent_yogurt['household_code'] != agent_yogurt['household_code'].shift(1))
).astype(int) # dummy for if a household switched flavors between trips 
agent_yogurt['flavor_spell_id']      = agent_yogurt.groupby('household_code')['new_flavor'].cumsum() # count of periods on new flavor
agent_yogurt['flavor_spell_buys']    = agent_yogurt.groupby(['household_code','flavor_spell_id']).cumcount() + 1 # consecutive periods on flavor
agent_yogurt['prev_flavor']          = agent_yogurt.groupby('household_code')['flavor'].shift(1) # last purchased flavor
agent_yogurt['spell_length']         = agent_yogurt.groupby(['household_code', 'flavor_spell_id'])['flavor_spell_buys'].transform('max') # get the number of buys in the flavor spell
agent_yogurt['switched']             = (
    agent_yogurt['flavor']           != agent_yogurt['prev_flavor']
).astype(int) # ever-switch indicator
agent_yogurt['returned']             = agent_yogurt.groupby('household_code')['flavor'].transform(
        lambda x: x.shift(1).isin(x.shift(-1))
) # indicator for returning to a previous flavor
agent_yogurt['next_flavor']          = agent_yogurt.groupby('household_code')['flavor'].shift(-1) # get the next flavor

switching_sample = agent_yogurt[agent_yogurt['switched'] == 1][[
    'household_code',
    'trip_code_uc',
    'flavor',
    'prev_flavor',
    'next_flavor',
    'spell_length'
]] # filter to switchers
switches_coupon = agent_yogurt[(agent_yogurt['switched'] == 1) & (agent_yogurt['deal_flag_uc'] == 1)][[
    'household_code',
    'trip_code_uc',
    'flavor',
    'prev_flavor',
    'next_flavor',
    'spell_length'
]] # filtering to HH who switched and used a deal in purchase

console.print(
    f'Mean consecutive buys by flavor x hh: {agent_yogurt['spell_length'].mean()}\n',
    f'Mean times switching by flavor x hh:  {agent_yogurt.groupby(['household_code', 'flavor'])['switched'].sum().mean()}\n',
    f'Percent of HH who ever-switch:        {(agent_yogurt.groupby('household_code')['flavor'].nunique()>1).mean()*100}\n',
    f'Average time spent on each flavor:    {agent_yogurt['spell_length'].mean()}\n'
    f'Percent switching due to coupon:      {len(switches_coupon)/(len(switching_sample))}\n',
)

#=== Switching Graphs for Agents ===#

# t-1 --> t

heat_flav = (
        switching_sample.groupby(['prev_flavor', 'flavor'])['spell_length']
        .mean()
        .unstack()
) # grouping switchers by flavor sequence, taking the mean, and then pivoting rows to columns, like a matrix
heat_flav = heat_flav.rename(columns={0:"Other", 1:"Berry", 2:"Plain"}) # labeling columns with flavor names
cell_labs = np.array(
    [[f'{val:.1f} trips' for val in row] for row in heat_flav.to_numpy()]
) # applying cell labels applying the word 'trips' after the spell value 
fig, ax   = plt.subplots(figsize=(10,8)) # defining the canvas and plot area
sns.heatmap(heat_flav,
            yticklabels=['Other', 'Berry', 'Plain'],
            annot   =cell_labs,
            fmt     ='',
            cmap    ='YlOrRd',
            ax      =ax,
            ) # heatmap using spell lengths, labeling y axis, applying cell labels, no formatting, color specification (Yellow,Orange,Red gradient), axis
ax.set_xlabel('Flavor Switched To') # x label
ax.set_ylabel('Flavor Switched From') # y label
ax.set_title('Mean Spell Length Upon Switching') # title: plot shows how long you stay on the switched to flavor
plt.tight_layout() # auto adjusts the spacing and margins
plt.savefig('../Output/Plots/3_flav_heatmap.pdf', format='pdf', bbox_inches='tight') # save the heatmap to output/plots
plt.close() # close the plot in python

#=== Product Stats ===#

# Mapping Nielsen DMA Codes to Market Names
dma_map = {
    524: 'Atlanta',
    602: 'Chicago',
    751: 'Denver',
    825: 'San Diego'
}

merged_master['market_name'] = merged_master['dma_code'].map(dma_map)

# 1. Generate Summary Table (LaTeX & Console Output)
dma_price_stats = (
    merged_master.groupby(['market_name', 'flavor'])['price']
    .agg(
        Mean_Price='mean',
        Std_Dev='std',
        Min_Price='min',
        Max_Price='max',
        Obs='count'
    )
    .reset_index()
)

# Format flavor labels
flavor_map = {0: 'Other', 1: 'Berry', 2: 'Plain'}
dma_price_stats['flavor_label'] = dma_price_stats['flavor'].map(flavor_map)

# Pivot table for clean LaTeX formatting
dma_pivot = dma_price_stats.pivot(index='market_name', columns='flavor_label', values='Mean_Price')
dma_pivot.columns = [f'Mean Price ({col})' for col in dma_pivot.columns]

console.print("\n=== DMA PRICE VARIATION SUMMARY ===")
console.print(dma_pivot)

# Save as LaTeX Table for paper
dma_pivot.to_latex('../Output/Tables/dma_price_summary.tex', float_format="%.3f")


# 2. Plot DMA Cross-Market Price Time Series
weekly_dma_price = (
    merged_master.groupby(['week_end', 'market_name'])['price']
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.lineplot(
    data=weekly_dma_price,
    x='week_end',
    y='price',
    hue='market_name',
    style='market_name',
    linewidth=1.8,
    ax=ax
)

ax.set_xlabel('Week', fontsize=11)
ax.set_ylabel('Mean Unit Price ($)', fontsize=11)
ax.set_title('Weekly Yogurt Price Variation Across Markets (2014)', fontsize=12, fontweight='bold')
ax.legend(title='Market (DMA)', frameon=True)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('../Output/Plots/dma_price_time_series.pdf', format='pdf', bbox_inches='tight')
plt.close()


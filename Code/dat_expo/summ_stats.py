import pandas as pd
import polars as pl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from rich.traceback import install; install() 
from rich.console import Console
console = Console() # alias function

"""
Data exploration and summary statistics.

    Subset to hh_size=1
    - Mean products purchased
    - Variance in n_purchased
    - n_HH
    - mean income
    - corr between income and purchases
    - mean purchases
    - racial makeup

    List Columns:
    - variance in unique upc's purchased by hh and date
    - variance in quantity purchased by hh and date
    - variance in product code (robustness of product measure)

    - sum, mean, variance of quantity of different products
    - correlation between purchases in t and t-1

"""

pd.set_option('display.max_rows', None) # allow for full output from prints

# === Read in Panel Parquet and Flavors CSV === #
df         = pl.read_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet').to_pandas() # panel read
flavors    = pd.read_csv('/scratch/dtm63837/Kilts_Panel/RMS/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv') # flavors read
df         = df.merge(flavors, on='upc', how='left') # merge on product codes
df         = df.convert_dtypes(dtype_backend='numpy_nullable') # make data numpy compatible
df.columns = df.columns.str.lower() # make column names lowercase
df         = df.dropna(subset = ['quantity', 'product_group_code', 'flavor_code', 'flavor_descr']) # drop NA from important columns
#console.print(df.columns)

df_sub              = df[df['household_size']==1] # subset to single agent households
df_sub['upc_descr'] = df_sub['upc_descr'].astype(str) # make the product descriptions strings
df_yogurt           = df_sub[df_sub.groupby('household_code')['trip_code_uc'].transform('count') > 2] # at least two trips
df_yogurt           = df_yogurt[df_yogurt['size1_units'] == 'OZ'] # restrict to oz only

console.print(df_yogurt['product_group_code'].dtype)
console.print(df_yogurt['quantity'].dtype)

df_yogurt['yogurt_purchase'] = (
    (df_yogurt['product_group_code'] == 2510) & (df_yogurt['quantity'] > 0)
).astype(int) # create dummy variable for if a household bought yogurt
df_yogurt['no_purchase']     = 1 - df_yogurt['yogurt_purchase'] # 0 when purchased, 1 if did not

# descriptives on yogurt purchases and quality

console.print(df_yogurt['yogurt_purchase'].value_counts())
console.print(df_yogurt['yogurt_purchase'].unique())
console.print(df_yogurt['quantity'].min())
console.print((df_yogurt['quantity'] <= 0).sum())

outside_option = df_yogurt.groupby(['household_code', 'trip_code_uc']).agg({
    'no_purchase': 'sum',
    'yogurt_purchase': 'count'
}).assign(
    total_occasions = lambda x: x['no_purchase'] + x['yogurt_purchase']
) # creating the outside option series (sum of no purchase occasions, count of total purchases)

outside_option['outside_option_rate'] = (
    outside_option['no_purchase'] / outside_option['total_occasions']
) # creating the rate of taking the outside option (sum of no purchase / count of purchases)

# === Summary Statistics === #
"""
n. Households,
Mean n. Trips,
Mean n. Purchases (yogurt),
Mean inc,
Median inc,
n. each rate,
Mean % taking outside option
"""

console.print(f'Number of households:    {df_yogurt['household_code'].nunique()}\n',
      f'Number of times seen:            {df_yogurt.groupby('household_code')['trip_code_uc'].nunique().mean()}\n',
      f'Yogurt purchases:                {df_yogurt.groupby('household_code')['trip_code_uc'].value_counts().mean()}\n',
      f'Mean household income:           {df_yogurt['household_income'].mean()}\n',
      f'Median household income:         {df_yogurt['household_income'].median()}\n',
      f'Racial Makeup of Sample:         {df_yogurt.groupby('race')['household_code'].nunique()}\n',
      f'Percent Taking Outside Option:   {outside_option['outside_option_rate'].mean()}\n',
      )

df_yogurt_filtered = df_yogurt[df_yogurt['product_group_code'] == 2510] # filtering to just yogurt
#console.print("Flavors:", df_yogurt_filtered.groupby('flavor_code')['flavor_descr'].unique()) # checking all flavors

"""
Variable creation:

    flavor = {}. Flavor of yogurt:
        1 = apple
        2 = blueberry
        3 = banana
        4 = cherry
        5 = key lime
        6 = lemon
        7 = peach
        8 = raspberry
        9 = strawberry
        10 = vanilla
        11 = mixed
        12 = plain
        0 = other
        
    flavor will be augmented over time, main analysis splits into 3 buckets desc. later
"""

df_yogurt = df_yogurt.assign(
    flavor = np.select(
        [
            df_yogurt['flavor_code'].isin([139, 44642, 75721, 2180]), # apple
            df_yogurt['flavor_code'].isin([22053, 24357, 52953, 74408, 17159, 23721]), # blueberry
            df_yogurt['flavor_code'].isin([11214, 20888, 17849, 17849]), # banana
            df_yogurt['flavor_code'].isin([904, 13314, 1169, 1174, 5651]), # cherry
            df_yogurt['flavor_code'].isin([73560, 3075, 73560]), # key lime
            df_yogurt['flavor_code'].isin([3107, 22916, 3122, 6061]), # lemon
            df_yogurt['flavor_code'].isin([3943, 3060, 70529, 10808, 3985, 23346]), # peach
            df_yogurt['flavor_code'].isin([6352, 41654, 41681, 78681, 41634, 6912]), # raspberry
            df_yogurt['flavor_code'].isin([23344, 16007, 16102, 66438, 16194, 30581, 45574, 72000, 17110]), # strawberry
            df_yogurt['flavor_code'].isin([5537, 5539, 66938, 5658, 72317]), # vanilla
            df_yogurt['flavor_code'].isin([66438, 66684, 71101, 72483,19061, 16102,  61082, 61487, 57428, 67420, 78857, 1154, 26050, 1216]), # mixed flavors
            df_yogurt['flavor_code'].isin([57129, 76690, 16200, 62349, 16199, 16182, 72290, 32300, 72289, 16102, 72292, 3465, 68109, 52953, 72288]), # mixed berry
            df_yogurt['flavor_code'].isin([4167]) # plain
        ],
        [1,2,3,4,5,6,7,8,9,10,11,12,13],
        default=np.nan
    )
)

#console.print(df_yogurt['flavor_descr'].value_counts())

df_yogurt = df_yogurt.dropna(subset=['flavor']) # dropping misc. flavors

"""
    All flavors
"""

#print('Variance in brand:', df_yogurt[df_yogurt['product_group_code'] == 2510].groupby(['household_code', 'trip_code_uc'])['brand_code_uc'].nunique().var())
#brand_choices=df_yogurt[df_yogurt['product_group_code'] == 2510].groupby(['household_code', 'trip_code_uc'])['brand_code_uc']
#plt.hist(brand_choices.var())
#plt.savefig('../Output/Plots/brand_var.pdf')
#plt.close()
#print('Variance in product:', df_yogurt[df_yogurt['product_group_code']==2510].groupby(['household_code', 'trip_code_uc'])['product_group_code'].nunique().var())
#unique_flav = df_yogurt[df_yogurt['product_group_code']==2510].groupby(['household_code', 'trip_code_uc'])['flavor'].nunique()
#print(unique_flav.dtype)
#print('Variance in flavor:', unique_flav.var())
#
#flavor_choices = df_yogurt[df_yogurt['product_group_code']==2510].groupby(['household_code', 'trip_code_uc'])['flavor']
#plt.hist(flavor_choices.var())
#plt.savefig('../Output/Plots/flavor_var_hist.pdf')
#plt.close()
#
#print('Most Popular Flavor:', df_yogurt['flavor'].mode()[0], df_yogurt[df_yogurt['flavor'] == df_yogurt['flavor'].mode()[0]]['flavor_descr'].iloc[0])
#
#df_sort = df_yogurt[df_yogurt['product_group_code']==2510].sort_values(['household_code','trip_code_uc'])
#df_sort['brand_last'] = df_sort.groupby('household_code')['brand_code_uc'].shift(1)
#df_sort['repeat_purchase'] = (df_sort['brand_code_uc']==df_sort['brand_last'])
#print('Mean repeat purchase percentage of brand', df_sort['repeat_purchase'].mean())
#
#df_sort_flavor = df_yogurt[df_yogurt['product_group_code']==2510].sort_values(['household_code','trip_code_uc'])
#df_sort_flavor['flavor_last'] = df_sort_flavor.groupby('household_code')['flavor'].shift(1)
#df_sort_flavor['repeat_purchase'] = (df_sort_flavor['flavor']==df_sort_flavor['flavor_last'])
#print('Mean repeat purchase of yogurt type', df_sort_flavor['repeat_purchase'].mean())
#
#df_sort_flavor.groupby('flavor')['repeat_purchase'].mean().plot(kind='bar')
#plt.savefig('../Output/Plots/repeat_flavor_bar.pdf')
#plt.close()
#
#brand_counts = df_yogurt[df_yogurt['product_group_code']==2510].groupby(['household_code', 'brand_code_uc']).size().reset_index(name='n_purchased')
#total_counts = df_yogurt[df_yogurt['product_group_code']==2510].groupby('household_code').size().reset_index(name='n_total')
#
#max_brand = brand_counts.groupby('household_code')['n_purchased'].max().reset_index()
#res = brand_counts.merge(total_counts, on = 'household_code')
#res['brand_concentration'] = res['n_purchased']/res['n_total']
#print(f'Mean Brand Concentration Within Household: {res['brand_concentration'].mean()}')
#
#flavor_counts = df_yogurt[df_yogurt['product_group_code']==2510].groupby(['household_code', 'flavor']).size().reset_index(name='flav_n_purchased')
#total_flavor  = df_yogurt[df_yogurt['product_group_code']==2510].groupby('household_code').size().reset_index(name='flav_n_total')
#
#max_flavor = flavor_counts.groupby('household_code')['flav_n_purchased'].max().reset_index()
#res_fl = flavor_counts.merge(total_flavor, on = 'household_code')
#res['flavor_concentration'] = res_fl['flav_n_purchased']/res_fl['flav_n_total']
#print(f'Mean Flavor Concentration Within Household: {res['flavor_concentration'].mean()}')
#
#df_run = df_yogurt[df_yogurt['product_group_code']==2510].sort_values(['household_code', 'trip_code_uc'])
#df_run['new_run'] = (
#    (df_run['flavor'] != df_run.groupby('household_code')['flavor'].shift(1)) |
#    (df_run['household_code'] != df_run['household_code'].shift(1))
#).astype(int)
#
#df_run['run_id'] = df_run.groupby('household_code')['new_run'].cumsum()
#df_run['consecutive_buys'] = df_run.groupby(['household_code', 'run_id']).cumcount() + 1
#
#print('Mean consecutive buys by household:', df_run['consecutive_buys'].mean())
#print('Variance in consecutive buys by household:', df_run['consecutive_buys'].var())
#
#print('Mean consecutive_buys by flavor x Household:', df_run.groupby('flavor')['consecutive_buys'].mean())
#
#df_run['prev_flavor'] = df_run.groupby('household_code')['flavor'].shift(1)
#df_run['switched'] =(
#    df_run['flavor'] != df_run['prev_flavor']
#).astype(int)
#
#df_run['returned'] = df_run.groupby('household_code')['flavor'].transform(
#    lambda x: x.shift(1).isin(x.shift(-1))
#)
#df_run['run_length'] = df_run.groupby(['household_code', 'run_id'])['consecutive_buys'].transform('max')
#switches = df_run[df_run['switched'] == 1][[
#    'household_code',
#    'trip_code_uc',
#    'prev_flavor',
#    'flavor',
#    'run_length'
#]]
#
#print('Mean percentage returned after switch', df_run['returned'].mean())
#print('Average time spent on one flavor:', df_run['run_length'].mean())
#
#print('Time spent on switched flavor:', switches.groupby(['prev_flavor', 'flavor'])['run_length'].mean())
#
#switches.groupby(['prev_flavor','flavor'])['run_length'].mean().hist(bins=20)
#plt.xlabel('Mean Switch Length')
#plt.ylabel('Count')
#plt.title('Distribution of Mean Consumption Spell Length by Flavor Transition')
#plt.tight_layout()
#plt.savefig('../Output/Plots/run_length_hist.pdf', format='pdf', bbox_inches='tight')
#plt.close()
#
#pivot = switches.groupby(['prev_flavor', 'flavor'])['run_length'].mean().unstack()
#fig, ax = plt.subplots(figsize=(10,8))
#sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
#ax.set_xlabel('Flavor Switched To')
#ax.set_ylabel('Flavor Switched From')
#ax.set_title('Mean Spell Length by Flavor Transition')
#plt.tight_layout()
#plt.savefig('../Output/Plots/run_length_heatmap.pdf', format='pdf', bbox_inches='tight')
#plt.close()
#
#"""
#    Plain vs. Other
#"""
#
#df_yogurt['plain'] = (df_yogurt['flavor']==13).astype(int)
#
#df_plain = df_yogurt.sort_values(['household_code', 'purchase_date'])
#df_plain['new_run'] = (
#    (df_plain['plain'] != df_plain.groupby('household_code')['plain'].shift(1)) |
#    (df_plain['household_code'] != df_plain['household_code'].shift(1))
#).astype(int)
#
#df_plain['run_id'] = df_plain.groupby('household_code')['new_run'].cumsum()
#df_plain['consecutive_buys'] = df_plain.groupby(['household_code', 'run_id']).cumcount() + 1
#
#print('Mean consecutive buys (plain vs other) by hh:', df_plain.groupby('plain')['consecutive_buys'].mean())
#
#df_plain['prev_flavor'] = df_plain.groupby('household_code')['plain'].shift(1)
#df_plain['run_length'] = df_plain.groupby(['household_code', 'run_id'])['consecutive_buys'].transform('max')
#
#df_plain['switched'] = (
#    df_plain['plain'] != df_plain['prev_flavor']
#).astype(int)
#
#df_plain['returned'] = df_plain.groupby('household_code')['flavor'].transform(
#    lambda x: x.shift(1).isin(x.shift(-1))
#)
#
#switches_plain = df_plain[df_plain['switched'] == 1][[
#    'household_code',
#    'purchase_date',
#    'prev_flavor',
#    'plain',
#    'run_length'
#]]
#
#print('Mean percentage return after switch', df_plain['returned'].mean())
#print('Average time spent on either type:', df_plain['run_length'].mean())
#
#print('Time spent on switched flavor:', switches_plain.groupby(['prev_flavor', 'plain'])['run_length'].mean())
#
#switches_plain.groupby(['prev_flavor','plain'])['run_length'].mean().hist(bins=20)
#plt.xlabel('Mean Switch Length')
#plt.ylabel('Count')
#plt.title('Distribution of Mean Consumption Spell Length by Flavor Transition')
#plt.tight_layout()
#plt.savefig('../Output/Plots/run_length_hist_plain.pdf', format='pdf', bbox_inches='tight')
#plt.close()
#
#pivot_plain = switches_plain.groupby(['prev_flavor', 'plain'])['run_length'].mean().unstack()
#fig, ax = plt.subplots(figsize=(10,8))
#sns.heatmap(pivot_plain, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
#ax.set_xlabel('Flavor switched to')
#ax.set_ylabel('Flavor switched from')
#ax.set_title('Mean Spell Length, Plain vs. Other')
#plt.tight_layout()
#plt.savefig('../Output/Plots/plain_heatmap.pdf', format='pdf', bbox_inches='tight')
#plt.close()
#
"""
    Plain, Berry, Other
"""

df_yogurt['flavor_3'] = np.select(
    [
        df_yogurt['flavor'].isin([2, 8, 9, 12]), # berry
        (df_yogurt['flavor'] == 13), # plain
    ],
    [1,2],
    default=0
) # Creating flavor codes, 0 = other, 1 = berry, 2 = plain

df_yogurt['size_cat'] = np.select(
    [
        (df_yogurt['size1_amount'] > 4) & (df_yogurt['size1_amount'] < 7), # cups
        (df_yogurt['size1_amount'] >= 32), # tubs
    ],
    [1,2],
    default = 0
) # size indicators, 0 = weird size, 1 = cup, 2 = tub

df_3flav = df_yogurt.sort_values(['household_code', 'trip_code_uc']) # sorting by time and household
df_3flav['new_run_flav'] = (
    (df_3flav['flavor_3'] != df_3flav.groupby('household_code')['flavor_3'].shift(1)) |
    (df_3flav['household_code'] != df_3flav['household_code'].shift(1))
).astype(int) # creating dummy for if HH is in a continuous flavor spell

df_3flav['flav_run_id']           = df_3flav.groupby('household_code')['new_run_flav'].cumsum() # create spell by sum of periods in a spell
df_3flav['flav_consecutive_buys'] = df_3flav.groupby(['household_code', 'flav_run_id']).cumcount() + 1 # consecutive buy counter


# Mean consecutive purcahses of a flavor by household
console.print('Mean consecutive buys (plain v berry v other) by hh:', df_3flav.groupby('flavor_3')['flav_consecutive_buys'].mean()) 

df_3flav['prev_flavor']   = df_3flav.groupby('household_code')['flavor_3'].shift(1) # defining the previous flavor bought by a household
df_3flav['run_length']    = df_3flav.groupby(['household_code', 'flav_run_id'])['flav_consecutive_buys'].transform('max') # the total length of a purchase spell

df_3flav['switched'] = (
    df_3flav['flavor_3'] != df_3flav['prev_flavor']
).astype(int) # indicator for switching (1 if current flavor != last flavor)

df_3flav['returned'] = df_3flav.groupby('household_code')['flavor_3'].transform(
    lambda x: x.shift(1).isin(x.shift(-1))
) # indicator for if a household returns to a prior flavor

switches_3flav = df_3flav[df_3flav['switched'] == 1][[
    'household_code',
    'trip_code_uc',
    'prev_flavor',
    'flavor_3',
    'run_length',
    'size_cat'
]] # filtering to HH who did switch

switches_coupon_3flav = df_3flav[(df_3flav['switched'] == 1) & (df_3flav['deal_flag_uc'] == 1)][[
    'household_code',
    'trip_code_uc',
    'prev_flavor',
    'flavor_3',
    'run_length',
    'size_cat'
]] # filtering to HH who switched and used a deal in purchase

"""
Stats around switching:
Count of households who ever switched flavors,
Count of households who switched and used a deal,
Mean percentage of households who return to their last flavor after a switch,
Mean time spent on flavor switch,

Graphs around switching:
Histogram of switch lengths by flavor,
Heatmap of switching behaviors between each flavor
"""

console.print('Percent ever-switched:', (switches_3flav['household_code'].count())/(df_3flav['household_code'].count()))
console.print('Ever-switcher by size type:', (switches_3flav.groupby('size_cat')['household_code'].count())/(df_3flav.groupby('size_cat')['household_code'].count()))
console.print('Percent switching due to coupon:', (switches_coupon_3flav['household_code'].count())/(df_3flav['household_code'].count()))
console.print('Mean percentage return after switch:', df_3flav['returned'].mean())
console.print('Average time spent on any type:', df_3flav['run_length'].mean())

console.print('Time spent on switched to flavor:', switches_3flav.groupby(['prev_flavor', 'flavor_3',])['run_length'].mean())

switches_3flav.groupby(['prev_flavor', 'flavor_3'])['run_length'].mean().hist(bins=20)
plt.xlabel('Mean Switch Length by Flavor')
plt.ylabel('Count')
plt.title('Distribution of Mean Consumption Spell Length by Flavor')
plt.tight_layout()
plt.savefig('../Output/Plots/run_length_hist_3flav.pdf', format='pdf', bbox_inches='tight')
plt.close()

pivot_3flav = switches_3flav.groupby(['prev_flavor', 'flavor_3'])['run_length'].mean().unstack()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(pivot_3flav, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
ax.set_xlabel('Flavor Switched To')
ax.set_ylabel('Flavor Switched From')
ax.set_title('Mean Spell Length, Plain vs. Berry vs. Other')
plt.tight_layout()
plt.savefig('../Output/Plots/3flav_heatmap.pdf', format='pdf', bbox_inches='tight')
plt.close()

tally_3flav = switches_3flav.groupby(['prev_flavor', 'flavor_3'])['run_length'].count().unstack().fillna(0) # length of flavor spell, NA = 0
row_norm    = tally_3flav.sum(axis=1) # sum of entries for normalization
tally_norm  = tally_3flav.div(row_norm, axis=0) # divide tally_3flav / row_norm to normalize

"""
Print:
Spell matrix,
Normalization row,
Distribution of raw matrix,
Distribution of normalized matrix
"""

console.print(tally_3flav)
console.print(row_norm)

console.print(sp.stats.describe(tally_3flav))
console.print(sp.stats.describe(tally_norm))

repetoire_size = switches_3flav.groupby('household_code')['flavor_3'].nunique() # size
repetoire_set  = switches_3flav.groupby('household_code')['flavor_3'].unique() # composition

rep_size_shh = switches_3flav.groupby(['household_code', 'size_cat'])['flavor_3'].nunique() # size (hh x container size)
rep_set_shh  = switches_3flav.groupby(['household_code', 'size_cat'])['flavor_3'].unique()  # composition (hh x container size)

buckets = repetoire_size.value_counts() # how many picked only one flavor, two flavors, all flavors
console.print(buckets)

buckets_shh = rep_size_shh.value_counts() # how many within the size buckets picked different combos
console.print(buckets_shh)

hh_flavors = switches_3flav.groupby('household_code')['flavor_3'].agg(lambda x: tuple(sorted(set(x))))
combo_map  = {
    (0,): 'Other',
    (1,): 'Berry',
    (2,): 'Plain',
    (0,1): 'Other, Berry',
    (0,2): 'Other, Plain',
    (1,2): 'Berry, Plain',
    (0,1,2): 'Other, Berry, Plain'
}

buckets_rep = hh_flavors.map(combo_map).value_counts()
console.print(buckets_rep)

"""
Barplot of counts of agents who pick x flavors:
    - flavor x hh
    - flavor x size x hh
"""

ax = sns.barplot(x=buckets.index, y=buckets.values, legend=False)
ax.set_xlabel('Amount Chosen')
ax.set_ylabel('Count')
ax.set_title('Chosen Flavor Buckets')
ax.bar_label(ax.containers[0])
plt.tight_layout()
plt.savefig('../Output/Plots/flavor_pair_comps.pdf', format='pdf', bbox_inches='tight')
plt.close()

ax = sns.barplot(x=buckets_shh.index, y=buckets_shh.values, legend=False)
ax.set_xlabel('Amount Chosen')
ax.set_ylabel('Count')
ax.set_title('Flavor Buckets Purchased by Product Size')
ax.bar_label(ax.containers[0])
plt.tight_layout()
plt.savefig('../Output/Plots/size_pair_comps.pdf', format='pdf', bbox_inches='tight')
plt.close()

two_flav             = repetoire_size[repetoire_size == 2].index # create mask for those who switched between 2
two_flav_sets        = repetoire_set[two_flav] # filter to those households
two_flav_sets_sorted = two_flav_sets.apply(lambda x: tuple(sorted(x))) # sorting the filtered data to refine to flavor pairs

two_flav_shh         = rep_size_shh[rep_size_shh == 2].index # mask for two buckets
two_flav_sets_shh    = rep_set_shh[two_flav_shh] # filter with mask
two_flav_sorted_shh  = two_flav_sets_shh.apply(lambda x: tuple(sorted(x))) # sort filtered data to count different permutations of sets as same set

pair_counts          = two_flav_sets_sorted.value_counts() # counting the values within each pair
pair_counts_shh      = two_flav_sorted_shh.value_counts() # counting how many within each pair

console.print(pair_counts)
console.print(pair_counts_shh)

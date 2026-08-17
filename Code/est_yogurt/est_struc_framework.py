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
                              })
master_df = master_df.to_pandas() # make pandas format
master_df.columns = master_df.columns.str.lower() # column names lower

flavors   = pd.read_csv('/scratch/dtm63837/Kilts_Panel/RMS/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv') # load in flavors documentation
master_df = master_df.merge(flavors, on = 'upc', how = 'left')

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
        default=np.nan
    )
)
master_df['flavor_binary'] = (
    master_df['flavor'] == 13 # if plain, flavor_binary == 1
).astype(int)

full_panel = master_df.copy()
master_df = master_df[master_df['product_module_code'].isin([3612, 3603])]
master_df = master_df.dropna(subset=['price'])

console.print('flavor_binary counts (yogurt only):')
console.print(master_df['flavor_binary'].value_counts()) # checking counts of flavor_binary values

import time
t0 = time.time()
choice_set_index = {
    key: group[['upc','price','flavor_binary']]
    for key,group in master_df.groupby(['store_code_uc','week_end'])
}
print(f'choice_set_index build: {time.time() - t0:.2f}s')

trips_df = full_panel[[
    'upc', 'product_module_code', 'trip_code_uc', 'household_code',
    'week_end', 'purchase_date', 'store_code_uc', 'flavor_binary',
    'quantity', 'household_income', 'male_head_age', 'female_head_age',
    'type_of_residence', 'race', 'price', 'dma_code'
]]

yog    = full_panel[full_panel['product_module_code'].isin([3612,3603])]
hh_yog = yog['household_code'].unique()   # households that buy yogurt at least once — still a reasonable scope restriction

yog_buyers = trips_df[trips_df['household_code'].isin(hh_yog)].copy()
yog_buyers['is_yogurt'] = (
    yog_buyers['product_module_code'].isin([3612, 3603])
).astype(int)

occ_lists = (
    yog_buyers[yog_buyers['is_yogurt']==1]
    .groupby(['household_code', 'trip_code_uc'])
    .apply(lambda g: list(np.repeat(g['upc'].values, g['quantity'].values.astype(int))))
)
occ_lists.name = 'yogurt_buy'

hh_trips = yog_buyers.merge(
    occ_lists, on=['household_code', 'trip_code_uc'], how='left'
)
trip_level = hh_trips.explode('yogurt_buy')

known_store_weeks = pd.DataFrame(
    list(choice_set_index.keys()), columns=['store_code_uc','week_end']
)
n_before   = trip_level['trip_code_uc'].nunique()
trip_level = trip_level.merge(known_store_weeks,
                              on=['store_code_uc','week_end'],
                              how = 'inner' )
n_after    = trip_level['trip_code_uc'].nunique()
console.print(f'trips retained after covered store filter: {n_after} of {n_before}')

trip_level = trip_level.sort_values(['household_code', 'week_end'])
trip_level['new_flavor'] = (
    (trip_level['flavor_binary'] != trip_level.groupby('household_code')['flavor_binary'].shift(1)) |
    (trip_level['household_code'] != trip_level['household_code'].shift(1))
).astype(int)
trip_level['flav_spell_id'] = trip_level.groupby('household_code')['new_flavor'].cumsum()
trip_level['cons_buys']     = trip_level.groupby(['household_code', 'flav_spell_id']).cumcount() + 1
trip_level['weeks_since_last_flavor'] = trip_level['cons_buys'] - 1
trip_level['weeks_since_last_trip']   = trip_level.groupby('household_code')['week_end'].diff().dt.days / 7
trip_level['since_last_trip']         = trip_level['weeks_since_last_trip'].fillna(0)
trip_level['head_age']                = trip_level['male_head_age'].fillna(trip_level['female_head_age'])
trip_level['single_male_head']        = trip_level['male_head_age'].notna().astype(int)

trip_level = trip_level.dropna(subset = ['type_of_residence', 'race', 'head_age'])
trip_level = trip_level[(trip_level['price'] > 0.01) | (trip_level['price'].isna())]

no_purchase_share = trip_level.groupby('trip_code_uc')['yogurt_buy'].first().isna().mean()

console.print(f'no-purchase trip share (post-restriction): {no_purchase_share:.4f}')
console.print(f'households remaining: {trip_level["household_code"].nunique()}')

console.print('='*60)
console.print('Built merged panel for estimation')
console.print('='*60)

# ================================================ #
# SECTION 1: model functions -- no data-loading
# ================================================ #

console.print('='*60)
console.print('Model Functions')
console.print('='*60)

# function to update theta within utility
def update_theta(theta_prev, x_chosen):
    # last period for testing
    theta = x_chosen # weighted average of previous choices and most recent choice
    return theta

# function to compute LOV within utility
def comp_Xi(x, theta):
    Xi = np.abs(x - theta) # euclidean distance of current from past
    return Xi

# gives the utility function (deterministic)
def utility_func(x, const, beta, gamma, alpha, theta, price):
    Xi = comp_Xi(x, theta) # calling LOV variable
    u = const + beta*x + gamma*np.log(1 + Xi) - alpha*price # defining utility
    return u

# ============================================= #
# SECTION 2: household contribution given params
# ============================================= #

#R   = 30
rng = np.random.default_rng(219) # setting seed

trip_keys  = set(zip(trip_level['store_code_uc'], trip_level['week_end']))
known_keys = set(choice_set_index.keys())
console.print(f'{len(trip_keys & known_keys)} of {len(trip_keys)} store-week combos have a known assortment')

all_trip_ids = trip_level['trip_code_uc'].unique()
#fixed_uniforms_index = {
#    trip_id: rng.uniform(size=R)
#    for trip_id in all_trip_ids
#}

trip_flavor_share = (
    trip_level[trip_level['yogurt_buy'].notna()]     # only actual purchases, not the NaN placeholder rows
    .groupby('trip_code_uc')['flavor_binary']
    .mean()
    .to_dict()
)

hh_index = {
    key: group.sort_values(['trip_code_uc', 'yogurt_buy'])
    for key, group in trip_level.groupby('household_code')
}

market_price = (
    master_df.groupby(['upc', 'week_end', 'dma_code'])['price']
    .mean()
    .reset_index()
)
totals = (
    market_price.groupby(['upc', 'week_end'])['price']
    .agg(['sum', 'count'])
    .reset_index()
    .rename(columns={'sum':'price_sum_all', 'count':'n_markets_all'})
)
market_price = market_price.merge(totals, on=['upc', 'week_end'], how='left')
market_price['price_iv'] = (
    (market_price['price_sum_all'] - market_price['price']) /
    (market_price['n_markets_all'] - 1)
)
market_price['price_iv'] = market_price['price_iv'].replace([np.inf, -np.inf], np.nan)

trip_level = trip_level.merge(
    market_price[['upc', 'week_end', 'dma_code', 'price_iv']],
    on=['upc', 'week_end', 'dma_code'],
    how='left'
)


iv_res = smf.OLS('price ~ price_iv + size1_amount + C(week_end)', data=trip_level).fit()
iv_resid = iv_res.resid

console.print(iv_res.summary())

console.print(trip_level[['price_iv', 'household_income', 'head_age']].isna().sum())
console.print(len(trip_level), iv_res.nobs)

trip_level['iv_resid'] = iv_resid


#z_cols = ['household_income', 'weeks_since_last_flavor', 'since_last_trip', 'head_age']
#
#for col in z_cols:
#    mean = trip_level[col].mean()
#    std  = trip_level[col].std()
#    trip_level[col + '_z'] = (trip_level[col] - mean) / std

def household_contribution(
        hh_id, trip_level_df,
        choice_set_index, hh_index,
        const, beta, gamma, alpha, theta_i0=0.0):

    hh_df = hh_index[hh_id] 
    if len(hh_df) == 0:
        return 0.0

    theta = theta_i0
    log_lik = 0.0

    for occ in hh_df.itertuples():
        store = occ.store_code_uc
        week  = occ.week_end

        if (store, week) not in choice_set_index:
            continue # skips stores not covered

        choice_set  = choice_set_index[(store, week)]
        flavor_binary       = choice_set['flavor_binary'].to_numpy()

        #d_ht = np.array([
        #    1.0, occ.household_income_z, occ.weeks_since_last_flavor_z,
        #    occ.since_last_trip_z, occ.single_male_head,
        #    occ.head_age_z, occ.type_of_residence, occ.race
        #])
        #lambda_ht = np.exp(d_ht @ delta)

        #u_fixed = fixed_uniforms_index[occ.trip_code_uc]
        #J_draws = np.maximum(poisson_dist.ppf(u_fixed, lambda_ht).astype(int), 1)

        if occ.yogurt_buy:
            chosen_upc    = occ.yogurt_buy  
            chosen_mask   = (choice_set['upc'] == chosen_upc).to_numpy()
            if not chosen_mask.any():
                continue
            chosen_idx    = np.where(chosen_mask)[0][0]
            x_chosen      = occ.flavor_binary   # replaces chosen_flavor
        else:
            chosen_idx = None
            x_chosen   = None

        u = utility_func(
            x=flavor_binary, const= const, beta=beta, gamma=gamma, alpha=alpha,
            theta=theta, price=choice_set['price'].to_numpy()
        )
        u_all = np.append(u, 0.0)
        IV    = logsumexp(u_all)
        prob  = np.exp(u_all - IV)

        #if chosen_idx is not None:
        #    sim_probs = 1 - (1 - prob[chosen_idx]) ** J_draws
        #else:
        #    sim_probs = prob[-1] ** J_draws

        chosen_prob = prob[chosen_idx] if chosen_idx is not None else prob[-1]

        if chosen_idx is not None and x_chosen is not None:
            theta = update_theta(theta, x_chosen)

        log_lik += np.log(chosen_prob)

    return log_lik

# =================================================================== #
# SECTION 3: total pop utility
# =================================================================== #

def obj_test(
        theta_vec, trip_level_df, choice_set_index, hh_index
):
    ll_contrib = []
    const, beta, gamma, alpha = theta_vec[:4]

    total_log_lik = 0.0

    hh_list = trip_level['household_code'].unique()
    for hh_id in trip_level_df['household_code'].unique():
        contrib = household_contribution(
            hh_id, trip_level_100, choice_set_index, hh_index,
            const, beta, gamma, alpha
        )
        if not np.isfinite(contrib):
            console.print(f'[red]non-finite contribution[/red] household = {hh_id}: {contrib}')

        ll_contrib.append(contrib)
    contribs = np.array(ll_contrib)

    console.print(contribs.min(), contribs.max(), np.sort(contribs), np.std(contribs))


def total_objective(
        theta_vec, trip_level_df, choice_set_index, hh_index):

    const, beta, gamma, alpha = theta_vec[:3]
    #delta = theta_vec[4:]

    total_log_lik = 0.0
    hh_list = trip_level['household_code'].unique()

    for hh_id in trip_level_df['household_code'].unique():
        ll = household_contribution(
            hh_id, trip_level_df, choice_set_index, hh_index,
            const, beta, gamma, alpha
        )
        if not np.isfinite(ll):
            console.print(f'[red]non-finite contribution[/red] household={hh_id}: {contrib}')

        total_log_lik += ll

    return -total_log_lik
# ========================================================== #
# SECTION 4: optimization
# ========================================================== #

sample_hh_100   = trip_level['household_code'].unique()[:100]
sample_hh_1000  = trip_level['household_code'].unique()[:1000]

trip_level_100  = trip_level[trip_level['household_code'].isin(sample_hh_100)]
trip_level_1000 = trip_level[trip_level['household_code'].isin(sample_hh_1000)]


purchase_share = (trip_level_100.groupby('trip_code_uc')['yogurt_buy'].first().notna()).mean()
console.print(f'purchase-trip share in this sample: {purchase_share:.4f}')

purchase_share_1000 = (trip_level_1000.groupby('trip_code_uc')['yogurt_buy'].first().notna()).mean()
console.print(f'purchase-trip share in this sample: {purchase_share_1000:.4f}')
console.print(trip_level_100['price'].describe())
console.print(trip_level_1000['price'].describe())

console.print(trip_level_100['price'].isna().sum())
console.print(trip_level_1000['price'].isna().sum())

console.print(np.where(trip_level_1000['price'] < .05))

x0 = np.array([0.0, 2.0, 9.0, 0.5])
bounds = (
    [(None, None), (None, None), (None, None), (None, None)])

res_100 = minimize(
    total_objective,
    x0     = x0,
    args   = (trip_level_100, choice_set_index, hh_index),
    method = 'L-BFGS-B',
    bounds = bounds
)
 
res_1000 = minimize(
    total_objective,
    x0     = x0,
    args   = (trip_level_1000, choice_set_index, hh_index),
    method = 'L-BFGS-B',
    bounds = bounds,
)

param_names = ['Constant', 'β', 'γ', 'α']
sample_labels = ['100 households', '1000 households']
for label, res in zip(sample_labels, [res_100, res_1000]):
    console.print(f'--- {label} ---')
    for name, val in zip(param_names, res.x):
        console.print(f'{name}: {val:.4f}')
    console.print('success:', res.success)
    console.print('final objective:', res.fun)
    console.print('jacobian:', res.jac)
    console.print(res.message)

# combat with simulated data and estimate off that
# try weighting lambda 50/50
# dummy for flavor_binary, flavored
# think of as product fixed effect (excluding outside option)
# update theta with only x_t-1
# share of occasions flavor purchased
# use product intro to add if one period behind works but other doesn't

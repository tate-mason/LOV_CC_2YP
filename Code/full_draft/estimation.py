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
from rich.table import Table
console = Console()

console.print('='*60)
console.print('Data Loading and Manipulation')
console.print('='*60)

hms_path          = '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet'
rms_path          = '/scratch/dtm63837/Kilts_Panel/RMS/master_retail.parquet'
out_path          = '/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master.parquet'

# Building the merged dataset from HMS and RMS

agent_panel = (
    pl.read_parquet(hms_path)
    .rename({'DMA_Cd':'dma_code'})
    .filter(pl.col('dma_code').is_in([524, 602, 751, 825]))
    .filter(pl.col('size1_amount').is_between(5, 8))
    .with_columns([
        pl.col('purchase_date').str.replace_all('-', '').str.to_date("%Y%m%d").alias('purchase_date'),
        pl.col('store_code_uc').cast(pl.Int64),
        pl.col('upc').cast(pl.Int64)
    ])
    .to_pandas()
)

agent_panel['week_end'] = agent_panel['purchase_date'] + pd.offsets.Week(weekday=5, n=0)

product_panel = (
    pl.read_parquet(rms_path)
    .filter(pl.col('product_module_code').is_in([3612, 3603]))
    .with_columns([
        pl.col('week_end').cast(pl.String).str.to_date('%Y%m%d'),
        pl.col('store_code_uc').cast(pl.Int64),
        pl.col('upc').cast(pl.Int64)
    ])
    .to_pandas()
    .dropna(subset=['week_end'])
)

master_df = agent_panel.merge(
    product_panel, on = ['store_code_uc', 'week_end', 'upc'], how = 'left'
)

master_df = master_df.rename(columns = {'product_module_code_x':'product_module_code',
                              'product_group_code_x':'product_group_code', 
                              'size1_code_uc_x':'size1_code_uc',
                              'size1_units_x':'size1_units',
                              'dma_code_x':'dma_code'
                              })
master_df.columns = master_df.columns.str.lower()

flavors   = pd.read_csv('/scratch/dtm63837/Kilts_Panel/RMS/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv') # load in flavors documentation
master_df = master_df.merge(flavors, on = 'upc', how = 'left')

other_codes = [139, 44642, 75721, 2180, 11214, 20888, 17849, 904, 13314, 1169, 1174, 5651, 73560, 3075, 3107, 22916, 3122, 6061, 3943, 3060, 70529, 10808, 3985, 23346, 5537, 5539, 66938, 5658, 72317, 66438, 66684, 71101, 72483, 19061, 16102, 61082, 61487, 57428, 67420, 78857, 1154, 26050, 1216]
berry_codes = [22053, 24357, 52953, 74408, 17159, 23721, 6352, 41654, 41681, 78681, 41634, 6912, 23344, 16007, 16102, 66438, 16194, 30581, 45574, 72000, 17110, 57129, 76690, 16200, 62349, 16199, 16182, 72290, 32300, 72289, 72292, 3465, 68109, 72288]
plain_codes = [4167]

conditions = [
    master_df["flavor_code"].isin(other_codes),
    master_df["flavor_code"].isin(berry_codes),
    master_df["flavor_code"].isin(plain_codes),
]
master_df["flavor"] = np.select(conditions, [0, 1, 2], default=np.nan)

console.print('flavor counts (yogurt only):')
console.print(master_df['flavor'].value_counts())
master_df['yogurt_buy'] = (
    (master_df['product_module_code'].isin([3612,3603]) & (master_df['quantity']>0)) # create dummy for HH who bought at least one yogurt product
).astype(int)

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

master_df = master_df[(master_df['price'] > 0.01) | (master_df['price'].isna())]

iv_res = smf.ols('price ~ price_iv + size1_amount + C(week_end)', data=master_df, missing='drop').fit()
master_df['iv_resid'] = iv_res.resid
master_df = master_df.dropna(subset=['iv_resid'])   # was missing the reassignment -- dropna alone doesn't mutate in place

trip_level             = master_df.copy()
trip_level['head_age'] = trip_level['male_head_age'].fillna(trip_level['female_head_age'])
trip_level             = trip_level.dropna(subset=['type_of_residence', 'race', 'head_age'])
trip_level             = trip_level[(trip_level['price'] > 0.01) | (trip_level['price'].isna())]

console.print('='*60)
console.print('Built merged panel for estimation')
console.print('='*60)
"""
    Vectorized Modal Choice and State Update
    - pre compute last chosen flavor at household x trip level before est
"""

def map_flavor_category(flavor):
    # adjust mapping to match categorial defintion
    if pd.isna(flavor):
        return 'outside'
    elif flavor == 0:
        return 'other'
    elif flavor == 1:
        return 'berry'
    elif flavor == 2:
        return 'plain'

# map individual item purchases to category strings
trip_level['category_chosen'] = trip_level['flavor'].map(map_flavor_category)
trip_level = trip_level.drop_duplicates(subset=['household_code', 'trip_code_uc']).copy()
# model trip category tiebreaker
rng                           = np.random.default_rng(219)

def get_modal_flavor(group):
    # Filter out outside option - purchase occurred
    purchases = group[group['yogurt_buy'].notna()]
    if purchases.empty:
        return np.nan
    counts    = purchases['flavor'].value_counts()
    if counts.empty:
        return np.nan
    max_count = counts.max()
    top_modes = counts[counts==max_count].index.to_numpy()
    return rng.choice(top_modes) # random tiebreaker if tied

modal_choices = (
    trip_level.sort_values(['household_code', 'week_end', 'trip_code_uc'])
    .groupby(['household_code', 'trip_code_uc', 'week_end'])
    .apply(get_modal_flavor)
    .reset_index(name='modal_x')
)
# vectorized state update sequence with lag
modal_choices = modal_choices.sort_values(['household_code', 'week_end'])
modal_choices['theta_prev'] = (
    modal_choices.groupby('household_code')['modal_x']
    .shift(1)
    .fillna(0.0) # default initial state
)
flavor_map_num = {'other':0.0, 'berry':1.0, 'plain':2.0}

"""
    Aggregated Store-Week Choice Sets
    - construct fixed alternative attributes for each category per store x week
"""

# Create category mapping on master
master_df['category'] = master_df['flavor'].map(map_flavor_category)

# Group assortments by store x week x category
cat_choice_sets = (
    master_df.groupby(['store_code_uc', 'week_end', 'category'])
    .agg(
        price     = ('price', 'mean'),
        iv_resid  = ('iv_resid', 'mean')
    )
    .reset_index()
)

CATEGORIES = ['other', 'berry', 'plain', 'outside']
cat_map = {c: i for i, c in enumerate(CATEGORIES)}

choice_set_matrix = {}
for (store, week), group in cat_choice_sets.groupby(['store_code_uc', 'week_end']):
    mat = np.zeros((4, 2))  # Matrix shape is now (4, 2) since flavor is fixed structurally
    
    # Calculate store-week average price to fill missing categories cleanly
    mean_store_price = group['price'].mean()
    mat[:3, 0] = mean_store_price # Default fallback price
    
    for row in group.itertuples():
        if row.category in cat_map and row.category != 'outside':
            idx = cat_map[row.category]
            # Store price and residual; ignore NaN residuals
            res_val = 0.0 if np.isnan(row.iv_resid) else row.iv_resid
            mat[idx] = [row.price, res_val]
            
    choice_set_matrix[(store, week)] = mat

"""
    Estimation Loop
    - join theta_prev and category index back to trips
"""

trips_processed = trip_level.merge(
    modal_choices[['household_code', 'trip_code_uc', 'theta_prev']],
    on  = ['household_code', 'trip_code_uc'],
    how = 'left'
)
# Default unmapped/outside choices safely to index 3
trips_processed['choice_idx'] = trips_processed['category_chosen'].map(cat_map).fillna(3).astype(np.int64)

# pre-pack inputs in arrays by hh

hh_packed_data = {}
for hh_id, group in trips_processed.groupby('household_code'):
    stores  = group['store_code_uc'].to_numpy()
    weeks   = group['week_end'].to_numpy()
    choices = group['choice_idx'].to_numpy(dtype=np.int64)
    thetas  = group['theta_prev'].to_numpy(dtype=np.float64)

    valid_mask = np.array([(s,w) in choice_set_matrix for s, w in zip(stores, weeks)])
    if not valid_mask.any():
        continue
    
    hh_packed_data[hh_id] = {
        'matrices': [choice_set_matrix[(s,w)] for s,w in zip(stores[valid_mask], weeks[valid_mask])],
        'choices' : choices[valid_mask],
        'thetas'  : thetas[valid_mask]
    }

def total_objective(params, hh_packed_data):
    # Parameter order: [const, beta, gamma, alpha, sigma]
    const, beta_ber, beta_pl, gamma, alpha, sigma = params

    d_berry = np.array([0.0, 1.0, 0.0])
    d_plain = np.array([0.0, 0.0, 1.0])
    
    # Fixed structural flavor representations: other = 0.0, berry = 1.0, plain = 2.0
    cat_flavors = np.array([0.0, 1.0, 2.0]) 
    
    total_ll = 0.0 

    for hh_data in hh_packed_data.values():
        matrices = hh_data['matrices']
        choices  = hh_data['choices']
        thetas   = hh_data['thetas']

        for X_mat, y_idx, theta in zip(matrices, choices, thetas):
            prices = X_mat[:, 0]
            resids = X_mat[:, 1]

            # 1. Compute Xi for inside alternatives (rows 0, 1, 2)
            Xi = np.zeros(4)
            Xi[:3] = np.abs(cat_flavors - theta)

            # 2. Compute Utility using fixed cat_flavors
            u = np.zeros(4)
            u[:3] = (
                const
                + beta_ber * d_berry 
                + beta_pl  * d_plain
                + gamma * np.log(1.0 + Xi[:3])
                + alpha * prices[:3]
                + sigma * resids[:3]
            )
            u[3] = 0.0  # Outside option normalized baseline

            # 3. Log-Sum-Exp & Choice Probability
            u_max = np.max(u)
            log_sum_exp = u_max + np.log(np.sum(np.exp(u - u_max)))

            log_prob = u[y_idx] - log_sum_exp
            
            if not np.isfinite(log_prob):
                log_prob = -700.0

            total_ll += log_prob

    return -total_ll

x0 = np.zeros(6)
bounds = [
    (None, None),  # beta_0
    (None, None),  # beta_ber
    (None, None),
    (None, None),  # gamma (unconstrained)
    (None, 0.0),  # alpha (strictly positive magnitude)
    (None, None)   # sigma
]

res = minimize(
    total_objective,
    x0 = x0,
    args = (hh_packed_data,),
    method = 'L-BFGS-B',
    bounds = bounds,
    options={'ftol':1e-6}
)

cov_matrix = res.hess_inv.todense()
se         = np.sqrt(np.diag(cov_matrix))

z = res.x / se
p = 2 * (1 - sp.stats.norm.cdf(np.abs(z)))

table = Table(title="Structural Estimation Results", show_header=True, header_style="bold magenta")
table.add_column("Parameter", style="cyan", justify="left")
table.add_column("Estimate", justify="right")
table.add_column("Std. Error", justify="right")
table.add_column("z-stat", justify="right")
table.add_column("p-value", justify="right")

param_names = ['β_0 (Intercept)', 'β_ber (Berry)', 'β_pl (Plain)', 'γ (Habit/Variety)', 'α (Price)', 'σ (Control Func)']

for name, val, se, z, p in zip(param_names, res.x, se, z, p):
    p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
    if p < 0.001:
        p_str += " ***"
    elif p < 0.05:
        p_str += " **"

    table.add_row(
        name,
        f"{val:.4f}",
        f"{se:.4f}" if not np.isnan(se) else "N/A",
        f"{z:.3f}" if not np.isnan(z) else "N/A",
        p_str
    )

console.print(table)
console.print(f"[bold]Optimization Success:[/bold] {res.success}")
console.print(f"[bold]Final Log-Likelihood Objective:[/bold] {res.fun:.4f}")

WTP = res.x[2] / res.x[4]
console.print(f'[bold]Willingness to Pay:[/bold] {WTP}')

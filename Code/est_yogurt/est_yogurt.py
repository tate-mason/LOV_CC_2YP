#data
import pandas as pd
import polars as pl
#numerical and statistical
import numpy as np
import scipy as sp
from scipy.special import logsumexp, expit
from scipy.optimize import minimize
#plotting
import matplotlib.pyplot as plt
import seaborn as sns
#estimation
import statsmodels.api as sm

import pyblp
pyblp.options.digits  = 2
pyblp.options.verbose = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
#output
from rich.console import Console
from rich.traceback import install; install()
console = Console()

console.print('='*60)
console.print('Data Loading and Manipulation')
console.print('='*60)

df_agent          = pl.read_parquet('/scratch/dtm63837/Kilts_Panel/nielsen_extracts/master_panel.parquet').to_pandas() # load in parquet and convert to pandas
#df_product        = pl.read_parquet('/scratch/dtm63837/Kilts_Panel/RMS/master_retail.parquet').to_pandas()
flavors           = pd.read_csv('/scratch/dtm63837/Kilts_Panel/RMS/Reference_Documentation/2006-2020_Documentation/Latest_Flavor_2010.csv') # load in flavors documentation
panel_df          = df_agent.merge(flavors, on='upc', how='left') # merge flavor and panel
panel_df          = panel_df.convert_dtypes(dtype_backend='numpy_nullable') # change data backend for numpy compatability
#retail_df         = df_product.merge(flavors, on='upc', how='left') # merge flavors in
#retail_df         = df_product.convert_dtypes(dtype_backend='numpy_nullable') # make numpy data
panel_df.columns  = panel_df.columns.str.lower() # making column names lowercase for convenience
#retail_df.columns = retail_df.columns.str.lower() # column names lowercase
panel_df          = panel_df.dropna(subset = ['quantity', 'product_group_code', 'flavor_code', 'flavor_descr']) # dropping NA households for important variables

console.print(panel_df.head)
console.print(panel_df.shape)
console.print(panel_df.info())
console.print(panel_df.describe())
console.print(panel_df.isna().sum())

#console.print(retail_df.head)
#console.print(retail_df.shape)
#console.print(retail_df.info())
#console.print(retail_df.describe())
#console.print(retail_df.isna().sum())

df_yogurt              = panel_df[panel_df['household_size'] == 1]
df_yogurt['upc_descr'] = df_yogurt['upc_descr'].astype(str)
df_yogurt           = df_yogurt[df_yogurt['size1_units'] == 'OZ']

df_yogurt['date'] = pd.to_datetime(df_yogurt['purchase_date'])
#df_yogurt['purchase_week'] = df_yogurt['date'].dt.to_period['W']


df_yogurt['yogurt_purchase'] = (
    (df_yogurt['product_group_code'] == 2510) & (df_yogurt['quantity'] > 0)
).astype(int)
df_yogurt['no_purchase']     = 1 - df_yogurt['yogurt_purchase']

console.print(f'Number of households:    {df_yogurt['household_code'].nunique()}\n',
      f'Number of times seen:            {df_yogurt.groupby('household_code')['trip_code_uc'].nunique().mean()}\n',
      f'Yogurt purchases:                {df_yogurt.groupby('household_code')['trip_code_uc'].value_counts().mean()}\n',
      f'Mean household income:           {df_yogurt['household_income'].mean()}\n',
      f'Median household income:         {df_yogurt['household_income'].median()}\n',
      f'Racial Makeup of Sample:         {df_yogurt.groupby('race')['household_code'].nunique()}\n',
      f'Percent Taking Outside Option:   {df_yogurt.groupby('trip_code_uc')['no_purchase'].nunique().mean()}\n',
      )

df_yogurt = (df_yogurt['product_group_code'] == 2510)

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

df_yogurt = df_yogurt.dropna(subset=['flavor']) # dropping misc. flavors

df_yogurt['flavor_binary'] = (
    df_yogurt['flavor'] == 13
).astype(int)
console.print(df_yogurt.groupby(['date', 'household_code', 'flavor_binary'])['quantity'].sum().reset_index())

df_yogurt['size_cat'] = np.select(
    [
        (df_yogurt['size1_amount'] > 4) & (df_yogurt['size1_amount'] < 7), # cups
        (df_yogurt['size1_amount'] >= 32), # tubs
    ],
    [1,2],
    default = 0
) # size indicators, 0 = weird size, 1 = cup, 2 = tub

df_yogurt.sort_values(['household_code', 'trip_code_uc']) # sorting values by household and trip
df_yogurt['new_flavor']    = (
    (df_yogurt['flavor_binary']  != df_yogurt.groupby('household_code')['flavor_binary'].shift(1)) |
    (df_yogurt['household_code'] != df_yogurt['household_code'].shift(1))
).astype(int) # flavor switched
df_yogurt['flav_spell_id'] = df_yogurt.groupby('household_code')['new_flavor'].cumsum() # ID for flavor spell
df_yogurt['cons_buys']     = df_yogurt.groupby(['household_code', 'flav_spell_id']).cumcount() + 1 # consecutive periods on flavor
df_yogurt['prev_flavor']   = df_yogurt.groupby('household_code')['flavor_binary'].shift(1) # flavor before switch
df_yogurt['spell_len']     = df_yogurt.groupby(['household_code', 'flav_spell_id'])['cons_buys'].transform('max') # length of spell before switch
df_yogurt['switched']      = (
    df_yogurt['new_flavor'] != df_yogurt['prev_flavor']
).astype(int) # dummy for switching flavor
df_yogurt['returned']      = (
    lambda x: x.shift(1).isin(x.shift(-1))
) # variable to determine if you return to your past flavor
    
console.print('='*60)
console.print('Linear Estimation')
console.print('='*60)

# Overall
df_yogurt.sort_values(['household_code', 'trip_code_uc'])
df_yogurt['switched_t1'] = df_yogurt['switched'].shift(1)

X = df_yogurt[['household_income', 'total_price_paid', 'male_head_age', 'female_head_age', 'switched_t1']] # covariates
X = X.astype({col: 'float64' for col in X.columns}) # bringing datatypes to numpy backend instead of pandas
X = X.dropna() # dropping NA rows
X = sm.add_constant(X) # adding constant

y = df_yogurt['switched'] # dependent variable - switching
y = y.loc[X.index] # aligning size with X

model = sm.OLS(y, X).fit() # calling model
console.print(model.summary()) # results of model

console.print('='*60)
console.print('ML Estimation')
console.print('='*60)

# LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = .2, random_state=219
) # defining 80% of data for training 20% for testing
pipe_logit = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
]) # setting the ML pipeline with a balanced scaler and logit model

pipe_logit.fit(X_train, y_train) # fitting on training data
preds_logit = pipe_logit.predict(X_test) # predicting on test data
console.print(classification_report(y_test, preds_logit)) # showing how well the model does on actual outcomes

# RandomForestClassifier
rf_pipe = Pipeline([
    ('model', RandomForestClassifier(random_state=219, class_weight='balanced'))
]) # using Random Forest to more robustly test

rf_pipe.fit(X_train, y_train) # fitting on training
preds_rf = rf_pipe.predict(X_test) # predicting on testing
console.print(classification_report(y_test, preds_rf)) # seeing discrepency between prediction and real

fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(pipe_logit, X_test, y_test, ax=ax, name='LogisticRegression') # ROC Curve to show pred accuracy for Logit
RocCurveDisplay.from_estimator(rf_pipe, X_test, y_test, ax=ax, name='RandomForest') # ROC curve for Random Forest
plt.savefig('../Output/Plots/ML_ROC.pdf', format='pdf', bbox_inches='tight')
plt.close

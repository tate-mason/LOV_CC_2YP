# Empirical Economics 101: Discrete Choice & Demand Estimation Pipeline

## 1. High-Level Concept

Every time a household visits a grocery store, they look at available products, evaluate prices and attributes, and make a discrete decision: buy **Other (0)**, **Berry (1)**, **Plain (2)**, or **Nothing / Outside Option (3)**.

To model this, we process panel data in a multi-step empirical framework:

1. Combine household shopping trips (**HMS**) with store-level availability and pricing (**RMS**).
2. Construct an **Instrumental Variable (IV)** control function to correct for price endogeneity.
3. Compute a dynamic state variable ($\theta_{i, t-1}$) capturing each household's previous trip choice to account for habit persistence.
4. Pre-pack $4 \times 3$ store-week choice matrices to represent available shelf options.
5. Estimate utility parameters via **Maximum Likelihood Estimation (MLE)**.

---

## 2. Step-by-Step Technical Breakdown

### Step 1: Loading & Cleaning Panel Data

We process two big-data extracts using Polars for memory efficiency before converting to Pandas for statistical routines.

* `pl.read_parquet(path)`: Loads compressed Parquet data files.
* `.rename({'DMA_Cd': 'dma_code'})`: Standardizes geographic market identifiers to lowercase.
* `.filter(pl.col('dma_code').is_in([524, 602, 751, 825]))`: Restricts analysis to four primary geographic markets.
* `str.replace_all('-', '')` & `str.to_date("%Y%m%d")`: Converts string dates to standard calendar date objects.
* `cast(pl.Int64)`: Casts key join columns (`upc`, `store_code_uc`) to 64-bit integers to prevent join type mismatches.
* `.to_pandas()`: Converts the processed Polars DataFrame into a Pandas DataFrame for statistical modeling.
* `agent_panel['purchase_date'] + pd.offsets.Week(weekday=5, n=0)`: Aligns daily household purchase dates to RMS Saturday week-ending dates.
* `.dropna(subset=['week_end'])`: Removes observations lacking valid week alignment.

---

### Step 2: Merging Panels & Classifying Categories

We align household purchases with store shelf prices and group thousands of individual UPC barcodes into three distinct product categories.

* `agent_panel.merge(product_panel, on=['store_code_uc', 'week_end', 'upc'], how='left')`: Merges household visits with store prices on matching store, week, and product barcode.
* `np.select(conditions, [0, 1, 2], default=np.nan)`: Maps product UPCs into category integer codes: `0` (Other), `1` (Berry), and `2` (Plain). Unmapped or non-yogurt items default to `NaN`.

---

### Step 3: Controlling for Price Endogeneity (Control Function IV)

Promotions and unobserved display ads create endogeneity between price and demand. We construct a Hausman-style IV using average prices of the same UPC in other geographic markets during the same week.

* `.groupby(['upc', 'week_end', 'dma_code'])['price'].mean()`: Computes the average item price per market-week.
* `(price_sum_all - price) / (n_markets_all - 1)`: Calculates the mean price across all **other** markets, isolating cost shocks from local store demand shocks.
* `smf.ols('price ~ price_iv + size1_amount + C(week_end)', data=master_df).fit()`: Runs a First-Stage OLS regression predicting item price using the IV, product size, and week fixed effects.
* `iv_res.resid`: Extracts first-stage residuals ($\hat{\nu}_{jt}$). Including $\hat{\nu}_{jt}$ directly in the utility function controls for unobserved price endogeneity.

---

### Step 4: Constructing Dynamic Habit States ($\theta_{i, t-1}$)

To model brand loyalty or variety-seeking behavior, we record what each household bought on their previous trip.

* `trip_level.groupby(['household_code', 'trip_code_uc', 'week_end']).apply(get_modal_flavor)`: Finds the most frequently purchased flavor category for a given trip.
* `if counts.empty: return np.nan`: Handles trips where no mapped yogurt was purchased (outside option).
* `rng.choice(top_modes)`: Breaks ties randomly using NumPy's random number generator if multiple categories tied for the maximum purchase count.
* `.groupby('household_code')['modal_x'].shift(1).fillna(0.0)`: Shifts modal choices down by one trip per household to generate $\theta_{i, t-1}$, initializing cold starts to `0.0`.

---

### Step 5: Constructing Pre-Packed Choice Sets

We construct a fixed-dimension $4 \times 3$ array for every store-week pair to represent shelf options during a trip:

* **4 Rows (Alternatives):** `0: Other`, `1: Berry`, `2: Plain`, `3: Outside Option`
* **3 Columns (Attributes):** `[Price, IV Residual, Category Index]`
* `mat = np.zeros((4, 3))`: Initializes a blank $4 \times 3$ matrix per store-week.
* `for row in group.itertuples():`: Iterates through store-week aggregated data to populate attribute values into row index `idx`.
* `choice_set_matrix[(store, week)] = mat`: Stores the finished matrix in a dictionary keyed by `(store_code_uc, week_end)` for $O(1)$ constant-time lookup during optimization.

---

### Step 6: Structural Likelihood Optimization

We specify structural consumer utility as:


$$U_{ij} = \beta_j - \alpha \cdot \text{Price}_j + \sigma \cdot \hat{\nu}_j + \gamma \cdot \ln(1 + \vert{}\text{Category}_j - \theta_{prev}\vert{})$$

* `np.abs(flavors - theta)`: Measures the distance between available product categories and the household's lagged state ($\theta_{i, t-1}$).
* `u[3] = 0.0`: Normalizes the utility of the outside option to zero for model identification.
* `u_max = np.max(u)` & `u_max + np.log(np.sum(np.exp(u - u_max)))`: Implements the numerically stable **Log-Sum-Exp** formulation to prevent floating-point overflow during exponentiation.
* `log_prob = u[y_idx] - log_sum_exp`: Evaluates the log-probability of observing choice $y_{idx}$.
* `minimize(total_objective, x0=x0, method='L-BFGS-B')`: Optimizes parameter values $(\beta_0, \beta_1, \beta_2, \gamma, \alpha, \sigma)$ to minimize negative log-likelihood.

---

## 3. Core Architectural Rules

| Issue | Solution Implemented |
| --- | --- |
| **Data Bloat** | Collapsed item-level purchase expansions into a single observation per household trip. |
| **Price Endogeneity** | Extracted control function residuals ($\hat{\nu}$) using out-of-market price IVs. |
| **Numerical Instability** | Applied Log-Sum-Exp normalization to logit choice denominators. |
| **Execution Speed** | Pre-packed store-week availability into indexed $4 \times 3$ NumPy matrices before running `minimize()`. |

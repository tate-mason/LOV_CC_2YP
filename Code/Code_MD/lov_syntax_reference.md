# Syntax Reference: LOV Estimation Pipeline

Line-by-line explanations of the Python/pandas/numpy/scipy syntax used in your script, grouped by section. Each entry: **what it does**, **why this form specifically**, **minimal standalone example**.

---

## 1. Data Loading & File Handling

### `pl.read_parquet(path).to_pandas()`
Reads a parquet file with Polars (fast reader), then converts to a pandas DataFrame in one chained call.
```python
df = pl.read_parquet('file.parquet').to_pandas()
```
`.to_pandas()` only works on an eager Polars `DataFrame`, not a `LazyFrame` (`scan_parquet` result) — those need `.collect()` first.

### `os.path.exists(path)`
Checks whether a file/folder exists at that path — returns `True`/`False`. Standard way to gate "only do this if the file isn't already there."
```python
if os.path.exists('out.parquet'):
    df = pd.read_parquet('out.parquet')   # cheap: skip re-doing work
else:
    df = expensive_function()
```

### `def f(a, b, force=False):`
`force=False` is a **default argument** — if the caller doesn't supply `force`, it's `False` automatically. Callers can override: `f(1, 2, force=True)`.

---

## 2. Pandas: Filtering Rows

### Boolean masking with `&` and `|`
```python
mask = (df['col_a'] == 5) & (df['col_b'] == 10)
subset = df[mask]
```
- `&` = "and", `|` = "or" — element-wise (row-by-row) versions, **not** Python's `and`/`or` (those don't work on whole columns).
- **Always parenthesize each condition** — `df['a']==5 & df['b']==10` silently misparses due to operator precedence.

### `.loc[row_condition, column_list]`
Selects specific rows *and* specific columns in one call.
```python
df.loc[df['store_code_uc'] == 123, ['upc', 'price']]
```
Left of the comma = which rows; right of the comma = which columns. Omit the column list (`df.loc[mask]`) to keep all columns.

### `.isin([...])`
Checks membership in a list/array — the vectorized equivalent of "is this value in this set?" for every row at once.
```python
df[df['household_code'].isin(hh_yog)]   # keep rows whose household is in hh_yog
```
Use this instead of chaining many `==`/`|` conditions when checking against more than 2-3 values.

### `.dropna(subset=['col'])`
Removes every row where `col` is `NaN` (missing). `subset` limits the check to that column only — without it, a row missing *any* column gets dropped.
```python
df = df.dropna(subset=['flavor'])
```
**Watch for order-of-operations bugs**: if a merge introduces `NaN` for rows you didn't intend to filter (e.g., a left-join on a reference table that only covers a subset of categories), `dropna` after that merge silently removes more than you think.

---

## 3. Pandas: Combining Data

### `.merge(other_df, on=[...], how='left')`
SQL-style join. `on` = shared column(s) to match on; `how` controls which rows survive:
```python
merged = purchases.merge(prices, on=['store', 'week', 'upc'], how='left')
```
| `how` | Keeps |
|---|---|
| `'left'` | every row from the left frame, `NaN` where no match on the right |
| `'inner'` | only rows where both sides match |

**Duplicate join keys multiply rows** — if a key isn't unique on one side, every match creates a separate output row (this is what caused the billions-of-rows blowups earlier in this project).

### `.astype('Int64')` / `.astype(str)`
Converts a column's data type.
```python
df['week_end'] = df['purchase_date'].astype(str).str.replace('-', '', regex=False).astype('Int64')
```
- `'Int64'` (capital I) = pandas' **nullable** integer type — tolerates `NaN`, unlike plain `int64`.
- `.astype(str)` first, so `.str.replace()` (a string-only method) can run.

### `.str.replace(old, new, regex=False)`
String find-and-replace on every value in a column. `regex=False` treats `old` as a literal string, not a regex pattern — important when your target string (like `-`) could be misread as regex syntax.
```python
'2023-01-15' -> '20230115'
```

### `.drop_duplicates(subset='col')`
Keeps only the first row for each unique value of `col`, discards the rest.
```python
trip_level = hh_trips.drop_duplicates(subset='trip_code_uc')
```
Only safe when every other column is identical across duplicates for a given key — verify with a `groupby(...).nunique()` check first if unsure.

---

## 4. Pandas: Grouping & Aggregating

### `.groupby('col')['other_col'].sum()`
Splits rows into groups by `col`, then computes a summary statistic per group.
```python
yog_trip_count = trip_level.groupby('household_code')['yogurt_trip'].sum()
```
Result: one row per unique `household_code`, value = sum of `yogurt_trip` (a boolean column — `True`/`False` sum as `1`/`0`) within that group.

### `.unique()`
Returns the distinct values of a column/array, order not guaranteed.
```python
hh_list = df['household_code'].unique()
```

### `.value_counts()`
Counts how many times each distinct value appears — a quick frequency table.
```python
df['flavor_binary'].value_counts()
# 0    21000
# 1     4500
```

---

## 5. NumPy: Array Operations

### `np.select([conditions], [choices], default=np.nan)`
Vectorized if/elif/else across an entire array — evaluates conditions **in order**, first match wins.
```python
flavor = np.select(
    [df['code'].isin([1,2]), df['code'].isin([3,4])],
    [1, 2],
    default=np.nan
)
```
If a value could match two conditions, only the **first** listed condition's choice is used — order matters.

### `np.append(arr, value)`
Adds a value (or array) to the end of an array, returning a **new** array (doesn't modify in place).
```python
u_all = np.append(u, 0.0)   # append the outside-option utility as the last element
```

### `np.where(condition)`
Returns the *indices* where `condition` is `True` (not the values themselves).
```python
chosen_idx = np.where(mask)[0][0]   # first index where mask is True
```
`[0]` unpacks the tuple `np.where` returns (one array per dimension); the second `[0]` grabs the first matching index.

### `np.abs(x - theta)`
Element-wise absolute value — equivalent to (but cheaper than) `np.sqrt((x-theta)**2)`.

---

## 6. SciPy: Optimization & Numerical Stability

### `scipy.special.logsumexp(array)`
Computes `log(sum(exp(array)))` in a numerically stable way (avoids overflow from exponentiating large numbers directly).
```python
IV = logsumexp(u_all)
prob = np.exp(u_all - IV)   # softmax: converts utilities to probabilities that sum to 1
```
Subtracting `IV` before `exp()` is what keeps this stable — never write `np.exp(u)/np.sum(np.exp(u))` directly for this reason.

**The math this code is computing.** With Gumbel-distributed shocks $\varepsilon_{ijt}$, the standard McFadden result gives closed-form choice probabilities:

$$P_{ijt} = \frac{e^{U_{ijt}}}{\sum_{k} e^{U_{ikt}}}$$

The denominator's log is the **inclusive value** (log-sum term):

$$IV_{it} = \log\sum_k e^{U_{ikt}}$$

so that $P_{ijt} = \exp(U_{ijt} - IV_{it})$ — exactly what `logsumexp` + subtraction implements. Computing $IV$ this way avoids ever forming $e^{U}$ directly for large $U$ (which overflows), because `logsumexp` internally shifts by the max value in the array before exponentiating, then shifts back — same math, numerically safe.

### Log-likelihood for MLE
Once each period's choice probability is known, the sample log-likelihood is the sum, across all observations, of the log-probability of the *actually chosen* alternative:

$$\ell(\theta) = \sum_{i}\sum_{t} \log P_{i,j^*_{it},t}(\theta)$$

where $j^*_{it}$ is household $i$'s chosen alternative at time $t$ and $\theta = (\beta, \gamma, \alpha, \lambda)$ is the parameter vector. Since `scipy.optimize.minimize` minimizes by default, the code works with the **negative** log-likelihood:

$$\text{neg\_ll}(\theta) = -\ell(\theta) = -\sum_i\sum_t \log P_{i,j^*_{it},t}(\theta)$$

so that "minimizing the objective" is equivalent to maximizing the likelihood. This is why every `neg_ll`-style function in this pipeline returns `-total_ll` rather than `total_ll`.

### `scipy.optimize.minimize(func, x0, args=(...), method=..., bounds=[...])`
Searches for the parameter vector that minimizes `func`.
```python
res = minimize(total_objective, x0=x0, args=(trip_level, master_df, w),
                method='L-BFGS-B', bounds=bounds)
```
- `func`'s **first** argument must be the parameter vector being searched over — everything else needed goes in `args` (passed through unchanged on every call).
- `bounds` is a list of `(low, high)` tuples, one per parameter, in the same order as `x0`. Use `None` for "no limit" on either side: `(0, None)` means "≥ 0, no upper bound."
- Result object: `res.x` (best parameters found), `res.fun` (objective value there), `res.success` (did it actually converge).

---

## 7. Python Fundamentals Used Throughout

### `def f(a, b):` ... `return x, y, z`
Functions can return multiple values at once — they arrive as a tuple, commonly unpacked immediately:
```python
def household_contribution(...):
    return log_lik, n_data, n_model, n_trips

log_lik, n_data, n_model, n_trips = household_contribution(...)
```

### `for _, row in df.iterrows():`
`.iterrows()` yields `(index, row)` pairs for each row. `_` is a naming convention meaning "I don't need this value" — here, the row's pandas index.
```python
for _, row in df.iterrows():
    print(row['upc'])
```
Slow at scale (interprets each row in plain Python) — fine for correctness-testing, worth vectorizing later for performance.

### `[expr for x, y in some_list]` (list comprehension)
Builds a new list by transforming each element of an existing iterable, in one line.
```python
model_shares = [m for m, e in hh_moment_gaps]
```
Equivalent to:
```python
model_shares = []
for m, e in hh_moment_gaps:
    model_shares.append(m)
```

### `f"{name}: {val:.4f}"` (f-string with format spec)
Embeds variables directly into a string. `:.4f` formats a float to 4 decimal places.
```python
print(f"beta: {0.512345:.4f}")   # -> "beta: 0.5123"
```

### `dict` construction and appending to a list of dicts
```python
results = []
results.append({'upc': upc_val, 'prob': p, 'chosen': is_chosen})
# later: pd.DataFrame(results)  -- converts the whole list to a table in one shot
```
Building a list of dicts and converting once at the end is much faster than appending rows to a DataFrame one at a time inside a loop.

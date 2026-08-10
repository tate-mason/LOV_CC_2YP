# Syntax Reference: LOV Monte Carlo Bundle (`LOV_bundle.py`)

Line-by-line explanations of the Python/numpy/scipy/statsmodels syntax used in this script, grouped by section. Each entry: **what it does**, **why this form specifically**, **minimal standalone example**. Companion to `lov_syntax_reference.md` — items already covered there (`logsumexp`, `np.abs`, `iterrows`, f-strings, `scipy.optimize.minimize` basics) aren't repeated here except where this script uses them differently.

---

## 1. Output Logging & File Setup

### `class Tee` + `sys.stdout = Tee(...)`
Redirects every `print()` call in the script to **both** the console and a file, simultaneously.
```python
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.terminal = sys.__stdout__
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

sys.stdout = Tee('mle_results.txt')
```
- `sys.__stdout__` (double underscore) is Python's *original* stdout reference — used here instead of `sys.stdout` to avoid infinite recursion if `Tee` were reassigned again later.
- Once `sys.stdout = Tee(...)` runs, **every subsequent `print()` in the file is silently duplicated to disk** — this is why your SLURM `.out` logs and `mle_results.txt` match. Combine with `python -u` so nothing gets buffered before hitting either destination.
- `flush()` is required because `Tee` is standing in for a real file object — without it, Python may hold output in memory rather than writing immediately, which is exactly the buffering problem `python -u` also guards against.

### `os.makedirs(path, exist_ok=True)`
Creates a directory (and any missing parent directories) — does nothing if it already exists, instead of raising an error.
```python
os.makedirs('../Output/Tables', exist_ok=True)
```
Without `exist_ok=True`, rerunning the script a second time would crash with `FileExistsError`. Safe to call every run.

### Two `save_tex_table` definitions (name collision)
This script defines `save_tex_table` **twice** — once near the top (`latex_str` signature), and again lower down (`rows, headers, title, filename` signature). Python doesn't warn about this: the second definition silently overwrites the first, so only the second version is ever callable.
```python
def save_tex_table(latex_str, filename):        # defined first — becomes dead code
    ...

def save_tex_table(rows, headers, title, filename, caption=""):  # this is the one that actually runs
    ...
```
Worth renaming one (e.g. `save_tex_table_raw` vs. `save_tex_table_grid`) if you ever need both, since right now the first is unreachable.

---

## 2. NamedTuples for Structured Results

### `namedtuple('cons_res', [...])`
Defines a lightweight, read-only record type — lets you return several related arrays from a function and access them by name instead of by position.
```python
cons_res = namedtuple('cons_res', ['IV_S', 'prob_S', 'U_S', 'x1_bar_S', 'x2_bar_S', 's_0_S'])
...
return cons_res(IV_S.mean(axis=0), prob_S.mean(axis=0), ...)
```
Callers can then do `res.prob_S` instead of `res[1]` — self-documenting, and immune to bugs from reordering fields, so long as the constructor call keeps the same order as the field list. **Note:** positional order in the `return cons_res(...)` call must match the field-name order declared above; nothing checks this for you if a field's meaning ever changes.

---

## 3. Building Multi-Dimensional Result Arrays

### `np.zeros((S, T, J+1))` and friends
Pre-allocates the full output array before the simulation loop fills it in — avoids the cost (and bugs) of growing an array inside a loop.
```python
V_S = np.zeros((S, T, J+1))       # simulations × periods × (products + outside option)
CCP_M = np.zeros((M, len(gamma), T, J + 1))   # markets × gamma-values × periods × choices
```
Shape order matters for every later indexing operation (`CCP_M[m, g_idx]` relies on `m` being axis 0 and `g_idx` axis 1) — keep a comment noting what each axis means, since a 4D array like `CCP_M` gives no hint on its own.

### Nested loop writing into a pre-allocated array
```python
for m in range(M):
    for g_idx, g in enumerate(gamma):
        res = ccp_iv_base(S, T, T_prior, J, prod_space1[m], prod_space2[m], beta, g)
        CCP_M[m, g_idx]  = res.prob_S
```
`enumerate(gamma)` gives both the index (`g_idx`, used for array positioning) and the value (`g`, used in the actual computation) in one pass — needed here because `gamma` is a list of arbitrary values (`[0, 6, 9, 12]`), not a simple range, so you can't recover `g` from `g_idx` alone.

---

## 4. Random Draws for Simulation

### `rng.gumbel(0, 1, size=(T, J+1))`
Draws Gumbel-distributed random shocks — one full grid of taste shocks per period per alternative (including the outside option in column 0), used as the ε in each utility draw.
```python
epsilon_ijt = rng.gumbel(0, 1, size=(T,J+1))
```
Gumbel (not normal) shocks are what makes the resulting choice probabilities collapse to the closed-form logit/softmax expression via `logsumexp` — this is the standard McFadden random-utility setup, not an arbitrary distributional choice.

### Pre-drawing shocks once, outside the estimation loop
```python
epsilon_fixed = rng_mle.gumbel(0, 1, size=(M, S, T, J+1))
...
def neg_ll(params, chosen_all, epsilon_fixed, x1_all, x2_all, ...):
    for m in range(M):
        for s in range(S):
            eps_ijt = epsilon_fixed[m, s]   # same draws every call
```
This is the "epsilon handling" rule from your notes made concrete: `epsilon_fixed` is generated **once**, before `scipy.optimize.minimize` ever runs, and every subsequent call to `neg_ll` — no matter what parameter values the optimizer is trying — indexes into the *same* fixed array. If shocks were redrawn inside `neg_ll`, the likelihood surface would be different (noisy) on every evaluation, and the optimizer would never converge cleanly since it can't tell whether a step improved the objective or just got a luckier draw.

### Two separate `default_rng` streams (`rng_mle`, `rng_data`)
```python
rng_mle = np.random.default_rng(seed=45)
rng_data = np.random.default_rng(seed=99)
```
Using different seeded generators for "the shocks" vs. "the product characteristics" keeps the two sources of randomness independent and separately reproducible — you could change one seed to re-draw only the epsilon shocks (or only the product grid) without disturbing the other.

---

## 5. Building the Choice Vector

### `np.concatenate([[u_out], u])`
Joins the outside-option utility (a scalar, wrapped in its own list to make it a 1-element array) with the vector of inside-good utilities into one array.
```python
u_out = epsilon_ijt[t, 0]
u_all = np.concatenate([[u_out], u])   # shape (J+1,), outside option at index 0
```
The `[u_out]` (list containing one scalar) is necessary because `np.concatenate` requires array-like inputs of matching dimensionality — passing the bare scalar `u_out` would raise an error.

### `np.argmax(u_all)` with index-0-as-outside-option convention
```python
chosen_idx = np.argmax(u_all)
if chosen_idx > 0:
    x1_chosen[t] = x1[chosen_idx-1]   # shift by 1 to undo the outside-option offset
```
Because the outside option was concatenated at position 0, every inside good's *true* index in `x1`/`x2` (which don't include an outside-option slot) is `chosen_idx - 1`. Easy off-by-one source if this offset isn't tracked consistently everywhere `chosen_idx` is used.

### Conditional running-mean update, guarding the outside option
```python
if chosen_idx > 0:
    x1_chosen[t] = x1[chosen_idx-1]
    x2_chosen[t] = x2[chosen_idx-1]
else:
    x1_chosen[t] = x1_bar_t     # outside option: carry forward the existing mean, don't pollute it
    x2_chosen[t] = x2_bar_t
```
This directly implements the "outside option periods should not pollute the running mean" principle from your notes — when the outside option is chosen, the running mean is fed *itself* rather than a new characteristic value, so a period of non-purchase doesn't drag the habit stock toward some meaningless value.

---

## 6. Reshaping for the OLS Step

### Building regression inputs with parallel lists + `np.concatenate`
```python
dep, x1v, x2v = [], [], []
for m in range(M):
    for j in range(1, J+1):
        dep.append(np.log(sj) - np.log(s0))
        x1v.append(np.full(T, x1_j))
        x2v.append(np.full(T, x2_j))
dep = np.concatenate(dep)
x1v = np.concatenate(x1v)
```
Pattern: accumulate a list of arrays (one per market/product combination), then flatten with a single `np.concatenate` call at the end — much cheaper than repeatedly resizing one big array inside the loop. `np.full(T, x1_j)` repeats the scalar `x1_j` into a length-`T` array so it lines up row-for-row with the `T` time-period values in `dep`.

### `np.log(sj) - np.log(s0))`
This constructs the classic **log-share-difference** dependent variable — `log(s_j/s_0) = log(s_j) - log(s_0)` — the standard Berry-inversion-style outcome for a multinomial logit share regression. Written as the note in your memory flags: this OLS approach is a *naive/comparison* regression here, not the estimator you actually trust, since Berry inversion breaks down once θ̄ evolves — this script runs it anyway as a baseline to contrast against the MLE results below it.

### `sm.OLS(dep, rhs).fit()`
Fits ordinary least squares. Note there's no intercept column added to `rhs` here (`np.column_stack([x1v, x2v])` or `[x1v, x2v, xiv]`) — `sm.OLS` does **not** add one automatically (unlike some R defaults), so this regression is implicitly forced through the origin unless `sm.add_constant(rhs)` is used.
```python
rhs = np.column_stack([x1v, x2v, xiv])
res_LOV = sm.OLS(dep, rhs).fit()
res_LOV.params    # coefficient estimates, in column order
res_LOV.bse        # standard errors, same order
res_LOV.rsquared   # R²
```

---

## 7. The MLE Objective Function

### The math `neg_ll` is evaluating

Inside the innermost loop, for each simulation $s$, market $m$, and period $t$, the code builds:

$$\Xi_{jt} = \sqrt{(x_{1j} - \bar x_{1,t})^2 + (x_{2j} - \bar x_{2,t})^2}$$

```python
xi = np.sqrt((x1_all[m] - x1_bar_t)**2 + (x2_all[m] - x2_bar_t)**2)
```
— the Euclidean distance between product $j$'s two characteristics and the household's running-mean habit stock $(\bar x_1, \bar x_2)_t$ in each dimension. (Note this is the two-characteristic analogue of the scalar $\Xi_{ijt}=|X_{jt}-\theta_{it}|$ from your notebook derivation — here it's a genuine Euclidean norm because there are two characteristics, not one.)

Utility for inside good $j$:

$$U_{jt} = \beta_1 x_{1j} + \beta_2 x_{2j} + \gamma\log\!\left(1+\Xi_{jt}^2\right) + \varepsilon_{jt}, \qquad U_{0t} = \varepsilon_{0t}$$

```python
u = beta1*x1_all[m] + beta2*x2_all[m] + gamma*np.log(1 + xi**2) + eps_ijt[t, 1:]
u_out = eps_ijt[t, 0]
u_all = np.concatenate([[u_out], u])
```
Note this specification uses $\Xi^2$ inside the log (`xi**2`), not $|\Xi|$ as in the single-characteristic notebook derivation — worth keeping straight, since the identification argument about the log breaking β/γ collinearity via curvature still applies, but the argument itself is now a squared Euclidean distance rather than an absolute deviation.

Choice probability (softmax over $J+1$ alternatives, outside option included at index 0):

$$P_{jt} = \frac{e^{U_{jt}}}{\sum_{k=0}^{J} e^{U_{kt}}} = \exp\!\big(U_{jt} - IV_t\big), \qquad IV_t = \log\sum_{k=0}^{J} e^{U_{kt}}$$

```python
IV = logsumexp(V, axis=1)
prob_S_cand[s] = np.exp(V - IV[:, None])
```

**Log-likelihood, summed across markets, simulated households, and periods:**

$$\ell(\beta_1,\beta_2,\gamma) = \sum_{m=1}^{M}\sum_{s=1}^{S}\sum_{t=1}^{T} \log P_{s,j^*_{mst},t}$$

where $j^*_{mst}$ is the alternative actually chosen by simulated household $s$ in market $m$ at time $t$ (`chosen_all[m][s,t]`). The code accumulates this market-by-market:

```python
chosen_probs_m = prob_S_cand[np.arange(S)[:, None], np.arange(T)[None, :], chosen_all[m]]
total_ll += np.sum(np.log(chosen_probs_m + 1e-300))
```
and `neg_ll` returns $-\ell$, since `scipy.optimize.minimize` minimizes rather than maximizes — the optimizer is therefore searching for

$$\hat\theta = \arg\min_{\theta}\; -\ell(\theta) \;\equiv\; \arg\max_\theta\; \ell(\theta)$$

which is exactly the parameter vector $\hat\theta=(\hat\beta_1,\hat\beta_2,\hat\gamma)$ reported in `print_res`. The `include_lov=False` branch is the same objective with $\gamma$ fixed at 0 — i.e. the restricted model $U_{jt}=\beta_1x_{1j}+\beta_2x_{2j}+\varepsilon_{jt}$ — which is what makes the two `res_no_LOV` / `res_LOV` calls a direct nested-model comparison (you could in principle likelihood-ratio test one against the other using `-2(\ell_{\text{restricted}} - \ell_{\text{full}})`, though the script doesn't currently do that step).

### Toggling parameters with a boolean flag
```python
def neg_ll(params, ..., include_lov):
    if include_lov:
        beta1, beta2, gamma = params
    else:
        beta1, beta2 = params
        gamma = 0.0
```
One objective function serves two model specifications (with/without the LOV term) by branching on how many parameters `params` is expected to contain. Keeps `est()` simple — it just passes a different-length `x0` and a different `include_lov` flag — but means `params`'s length must always match `include_lov`, with nothing enforcing that agreement except correct calling code.

### Early-exit penalty for invalid parameter regions
```python
if gamma < 0 or beta1 < 0 or beta2 < 0:
    return 1e10
```
Instead of using `scipy.optimize.minimize`'s `bounds` argument, this enforces non-negativity by having the objective function itself return a huge value (effectively infinite cost) whenever the optimizer strays into disallowed territory. Works with derivative-free methods like Nelder-Mead (which ignore `bounds` unless you switch methods), but means the optimizer gets no gradient information about *why* that region is bad — it just sees a cliff. `1e10` matches the `buff` constant defined near the top of the file, though the constant itself isn't reused here (worth swapping in `buff` for consistency).

### Fancy indexing to pull out chosen-alternative probabilities
```python
chosen_probs_m = prob_S_cand[np.arange(S)[:, None], np.arange(T)[None, :], chosen_all[m]]
```
This is **advanced/fancy indexing** with broadcasting, pulling one specific value out of the 3D array `prob_S_cand` (shape `S × T × (J+1)`) for *every* simulation-period combination at once, without a Python loop:
- `np.arange(S)[:, None]` → column vector of simulation indices, shape `(S, 1)`
- `np.arange(T)[None, :]` → row vector of period indices, shape `(1, T)`
- `chosen_all[m]` → array of chosen alternative indices, shape `(S, T)`

NumPy broadcasts the first two against each other to produce every `(s, t)` pair, then uses `chosen_all[m]`'s matching `(s, t)` entry as the index into the last axis. Net effect: `chosen_probs_m[s, t] = prob_S_cand[s, t, chosen_all[m][s, t]]` for all `s, t` simultaneously — the vectorized equivalent of a double for-loop with an if-lookup inside.

### `np.log(chosen_probs_m + 1e-300)`
Adds a tiny constant (`tol` elsewhere in the file) before taking the log, purely to prevent `log(0) = -inf` if any simulated choice probability underflows to exactly zero. Matches the `tol = 1e-300` constant defined at the top of the script (again, not directly referenced here — could be swapped in for clarity).

---

## 8. Running the Full Estimation Sweep

### `scipy.optimize.minimize(..., x0=[1.0, 1.0, 5.0], method='Nelder-Mead')`
```python
res_LOV = sp.optimize.minimize(
    neg_ll, x0=[1.0, 1.0, 5.0],
    args=(chosen_all, epsilon_fixed, x1_all, x2_all, M, S, T, T_prior, J, True),
    method='Nelder-Mead'
)
```
Nelder-Mead is a derivative-free ("simplex") method — doesn't need `neg_ll` to be differentiable, which matters here since the objective involves `np.argmax` (choice simulation) buried inside the earlier `ccp_iv_mle` function, a genuinely non-smooth operation that gradient-based optimizers (`L-BFGS-B`, etc.) would struggle with directly. Trade-off: generally slower to converge and more sensitive to starting values (`x0`) than gradient methods, which is worth remembering if convergence ever looks shaky across different `gamma` regimes.

### List comprehension inside `np.mean`, over a list of arrays
```python
print(f'Percent taking outside option: {np.mean([c == 0 for c in chosen_all]):.2f}')
```
`chosen_all` is a *list* of `(S, T)` arrays (one per market `m`). `[c == 0 for c in chosen_all]` produces a list of boolean arrays; `np.mean` on a list of same-shaped arrays implicitly stacks them before averaging — giving the overall fraction of outside-option choices pooled across all markets, simulations, and periods in one line.

### Driving the whole sweep with a single loop over `gamma`
```python
for g in gamma:
    mle = comp_ll(S, T, T_prior, J, M, beta, g, epsilon_fixed, x1_all, x2_all)
```
`comp_ll` bundles "simulate choices under true gamma `g` → estimate both specifications → print comparison" into one call per regime — this is the loop that produces the `beta`/`gamma` recovery comparisons across your `[0, 6, 9, 12]` gamma grid, letting you see directly whether estimation degrades as true gamma shrinks toward 0 (your known identification concern).

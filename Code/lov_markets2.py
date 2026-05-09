import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from prettytable import PrettyTable
from scipy.special import logsumexp
import statsmodels.api as sm
from itertools import product as iproduct
from scipy.optimize import least_squares
from scipy.stats import t as t_dist

import os
os.makedirs('../Output/Tables', exist_ok=True)
os.makedirs('../Output/Plots', exist_ok=True)
for m in range(1, 11):
    os.makedirs(f'../Output/Plots/Markets_{m}', exist_ok=True)

# =============================================================================
# Parameterization
# =============================================================================

M         = 10                              # markets
J         = 5                               # products
T         = 100                             # time periods
T_prior   = 10                              # burn-in history
S         = 1000                            # simulations per cell
B         = 500                             # bootstrap draws

rng = np.random.default_rng(seed=219)

beta = 2.0
gamma      = np.array([0, 6, 9, 12])


Results = namedtuple('Results', [
    'inclusive_value',
    'prob_choose',      # (T, J+1)  — index 0 is outside option
    'utility',          # (T, J+1)
    'x1_bar_path',
    's_0',
])

# =============================================================================
# Consumer simulation
# =============================================================================

def consumer_choice_prob(S, T, T_prior, J, x1, beta, gamma):
    IV_S     = np.zeros((S, T))
    prob_S   = np.zeros((S, T, J + 1))
    U_S      = np.zeros((S, T, J + 1))
    x1_bar_S = np.zeros((S, T))
    s_0_S    = np.zeros((S, T))

    for s in range(S):
        eps      = rng.gumbel(0, 1, size=(T, J + 1))  # col 0 = outside option

        # --- burn-in ---
        prior_choices = np.zeros(T_prior)
        x_bar_prior   = 0.0
        for t in range(T_prior):
            Xi_h      = x1 - x_bar_prior
            U_in      = 1 + beta * x1 + gamma * np.log(1 + Xi_h**2) + rng.gumbel(0, 1, J)
            U_out     = 1 + rng.gumbel(0, 1)
            ch        = np.argmax(np.concatenate([[U_out], U_in]))
            if ch > 0:
                prior_choices[t] = x1[ch - 1]
            else:
                prior_choices[t] = x_bar_prior   # outside option: carry forward
            x_bar_prior = np.mean(prior_choices[:t + 1])

        # --- main periods ---
        x_chosen = np.zeros(T)
        x_bar_t  = np.zeros(T)
        V        = np.zeros((T, J + 1))

        x_bar_t[0] = x_bar_prior

        for t in range(T):
            if t > 0:
                x_bar_t[t] = np.mean(x_chosen[:t])

            Xi_t  = np.sqrt((x1 - x_bar_t[t])**2)
            U_in  = 1 + beta * x1 + gamma * np.log(1 + Xi_t**2) + eps[t, 1:]
            U_out = 1 + eps[t, 0]
            U_all = np.concatenate([[U_out], U_in])
            V[t]  = U_all

            ch = np.argmax(U_all)
            s_0_S[s, t] = int(ch == 0)

            if ch > 0:
                x_chosen[t] = x1[ch - 1]
            else:
                x_chosen[t] = x_bar_t[t]   # outside option: carry forward mean

        IV   = logsumexp(V, axis=1)
        prob = np.exp(V - IV[:, None])

        IV_S[s]     = IV
        prob_S[s]   = prob
        U_S[s]      = V
        x1_bar_S[s] = x_bar_t

    return Results(
        inclusive_value = IV_S.mean(axis=0),
        prob_choose     = prob_S.mean(axis=0),
        utility         = U_S.mean(axis=0),
        x1_bar_path     = x1_bar_S.mean(axis=0),
        s_0             = s_0_S.mean(axis=0),
    )

# =============================================================================
# Product characteristics
# =============================================================================

product_spaces = rng.uniform(0, 100, size=(M,J)) # each market has its own product space

# =============================================================================
# Utility, IV, and outside option share paths for each market x gamma pair
# =============================================================================

CCP_M  = np.zeros((M, len(gamma), T, J + 1))
theta_M = np.zeros((M, len(gamma), T))

for m in range(M):
    for g_idx, g in enumerate(gamma):
        res = consumer_choice_prob(S, T, T_prior, J, product_spaces[m], beta, g)
        CCP_M[m, g_idx]   = res.prob_choose
        theta_M[m, g_idx] = res.x1_bar_path

        # utility plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for j in range(J):
            sns.lineplot(x=np.arange(T), y=res.utility[:, j+1], label=f'Product {j+1}', ax=ax)
        ax.set_title(f'Market {m+1} - Utility Paths ($\\gamma={g}$, $\\beta={beta}$)')
        ax.set_xlabel('Time Period'); ax.set_ylabel('Average Utility'); ax.legend()
        plt.savefig(f'../Output/Plots/Markets_{m+1}/utility_gamma_{g}.pdf'); plt.close()

        # inclusive value plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=np.arange(T), y=res.inclusive_value, ax=ax)
        ax.set_title(f'Market {m+1} - Inclusive Value ($\\gamma={g}$, $\\beta={beta}$)')
        ax.set_xlabel('Time Period'); ax.set_ylabel('Inclusive Value')
        plt.savefig(f'../Output/Plots/Markets_{m+1}/inclusive_value_gamma_{g}.pdf'); plt.close()

        # outside share plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=np.arange(T), y=res.s_0, ax=ax)
        ax.set_title(f'Market {m+1} - Outside Share ($\\gamma={g}$, $\\beta={beta}$)')
        ax.set_xlabel('Time Period'); ax.set_ylabel('Outside Option Share')
        plt.savefig(f'../Output/Plots/Markets_{m+1}/outside_share_gamma_{g}.pdf'); plt.close()


# =============================================================================
# Build CCP_M and theta_M 
# =============================================================================

CCP_M   = np.zeros((M, len(gamma), T, J + 1))
theta_M = np.zeros((M, len(gamma), T))

for m in range(M):
    for g_idx, g in enumerate(gamma):
        res = consumer_choice_prob(S, T, T_prior, J, product_spaces[m], beta, g)
        CCP_M[m, g_idx] = res.prob_choose
        theta_M[m, g_idx] = res.x1_bar_path

# =============================================================================
# Regressions
# =============================================================================

print("\n" + "="*60)
print(f"Regressions [true β={beta}]")
print("="*60)

def save_tex_table(rows, headers, title, filename, caption=""):
    col_fmt = "c" * len(headers)
    header_row = " & ".join(headers) + " \\\\"
    lines = [
        "\\begin{tabular}{" + col_fmt + "}",
        "\\toprule",
        header_row,
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f'../Output/Tables/{filename}.tex', 'w') as f:
        f.write("\n".join(lines))
# no LOV
# =============================================================================
# Regressions
# =============================================================================

print("\n" + "="*60)
print(f"Regressions [true β={beta}]")
print("="*60)

# --- Naive OLS (no LOV) ---
for g_idx, g in enumerate(gamma):
    dep, x1v = [], []
    for m in range(M):
        s0 = CCP_M[m, g_idx, :, 0]
        for j in range(1, J+1):
            s_j  = CCP_M[m, g_idx, :, j]
            x1_j = product_spaces[m, j-1]
            dep.append(np.log(s_j) - np.log(s0))
            x1v.append(np.full(T, x1_j))

    dep = np.concatenate(dep)
    x1v = np.concatenate(x1v)
    results_naive = sm.OLS(dep, sm.add_constant(x1v)).fit()

    summ_tab_naive = PrettyTable()
    summ_tab_naive.title = fr"Naive OLS (No LOV) — True β={beta}, True γ={g}"
    summ_tab_naive.field_names = [r"Constant", r"β_hat", r"SE(β_hat)", "R2"]
    summ_tab_naive.add_row([
        f"{results_naive.params[0]:.4f}",
        f"{results_naive.params[1]:.4f}",
        f"{results_naive.bse[1]:.4f}",
        f"{results_naive.rsquared:.4f}",
    ])
    print(f"\n{summ_tab_naive}")
    save_tex_table(
        [[f"{results_naive.params[0]:.4f}", f"{results_naive.params[1]:.4f}", f"{results_naive.bse[1]:.4f}", f"{results_naive.rsquared:.4f}"]],
        headers=[r"Constant", r"$\hat{\beta}$", r"SE($\hat{\beta}$)", "$R^2$"],
        title=f"Naive OLS",
        filename=f'naive_regression_summary_gamma_{g}',
    )

# --- OLS with LOV ---
for g_idx, g in enumerate(gamma):
    dep, x1v, xiv = [], [], []
    for m in range(M):
        s0 = CCP_M[m, g_idx, :, 0]
        for j in range(1, J+1):
            s_j   = CCP_M[m, g_idx, :, j]
            x1_j  = product_spaces[m, j-1]
            xi_jt = np.sqrt((x1_j - theta_M[m, g_idx])**2)
            dep.append(np.log(s_j) - np.log(s0))
            x1v.append(np.full(T, x1_j))
            xiv.append(np.log(1 + xi_jt**2))

    dep = np.concatenate(dep)
    x1v = np.concatenate(x1v)
    xiv = np.concatenate(xiv)
    results = sm.OLS(dep, sm.add_constant(np.column_stack([x1v, xiv]))).fit()

    summ_tab = PrettyTable()
    summ_tab.title = fr"OLS with LOV — True β={beta}, True γ={g}"
    summ_tab.field_names = [r"Constant", r"β_hat", r"γ_hat", r"SE(β_hat)", r"SE(γ_hat)", "R2"]
    summ_tab.add_row([
        f"{results.params[0]:.4f}",
        f"{results.params[1]:.4f}",
        f"{results.params[2]:.4f}",
        f"{results.bse[1]:.4f}",
        f"{results.bse[2]:.4f}",
        f"{results.rsquared:.4f}",
    ])
    print(f"\n{summ_tab}")
    save_tex_table(
        [[f"{results.params[0]:.4f}", f"{results.params[1]:.4f}", f"{results.params[2]:.4f}", f"{results.bse[1]:.4f}", f"{results.bse[2]:.4f}", f"{results.rsquared:.4f}"]],
        headers=[r"Constant", r"$\hat{\beta}$", r"$\hat{\gamma}$", r"SE($\hat{\beta}$)", r"SE($\hat{\gamma}$)", "$R^2$"],
        title=f"OLS with LOV — γ={g}",
        filename=f'regression_summary_gamma_{g}',
    )



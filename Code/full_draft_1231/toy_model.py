import numpy as np
from scipy.optimize import minimize
from rich.traceback import install

install()
from rich.console import Console

console = Console()

# Plug in Recovered Parameter Estimates

beta_0 = 0.95
beta_B = -0.77
beta_P = -1.42
gamma_H = -0.11
gamma_L = -0.22
alpha = -0.98
delta = 0.95
lambda_1 = 0.4  # Prior Pr(r=H)
costs = np.array([1.0, 1.2])
J = len(costs)


# Logit probs
def log_probs(V):
    eV = np.exp(V)
    return eV / (1.0 + np.sum(eV))


# Utility
def val_p1(p, gamma_r):
    return beta_0 + alpha * p


def val_p2(p, prev_j, k, gamma_r):
    switching = 1.0 if prev_j != k else 0.0
    return beta_0 + gamma_r * switching + alpha * p


# PERFECT INFO
def perfect_info_profit(p1_flat, p2_flat):
    p1 = p1_flat.reshape((2, J))
    p2 = p2_flat.reshape((2, J, J))

    total_profit = 0.0
    types = [
        {"gamma": gamma_H, "weight": lambda_1},
        {"gamma": gamma_L, "weight": 1 - lambda_1},
    ]

    for r_idx, t in enumerate(types):
        V1 = val_p1(p1[r_idx], t["gamma"])
        s1 = log_probs(V1)

        m1 = p1[r_idx] - costs
        pi1 = np.sum(s1 * m1)

        pi2 = 0.0
        for j in range(J):
            V2 = np.array([val_p2(p2[r_idx, j, k], j, k, t["gamma"]) for k in range(J)])
            s2 = log_probs(V2)
            m2 = p2[r_idx, j] - costs
            w_rj = np.sum(s2 * m2)
            pi2 += s1[j] * w_rj
        total_profit += t["weight"] * (pi1 + delta * pi2)
    return -total_profit


def imperfect_info_profit(params):
    p1 = params[:J]
    p2 = params[J:].reshape((J, J))

    s1_H = log_probs(val_p1(p1, gamma_H))
    s1_L = log_probs(val_p1(p1, gamma_L))

    s1_tilde = lambda_1 * s1_H + (1 - lambda_1) * s1_L
    m1 = p1 - costs
    pi1 = np.sum(s1_tilde * m1)

    pi2 = 0.0
    for j in range(J):
        lambda_2_j = (lambda_1 * s1_H[j]) / s1_tilde[j]
        s2_H = log_probs(np.array([val_p2(p2[j, k], j, k, gamma_H) for k in range(J)]))
        s2_L = log_probs(np.array([val_p2(p2[j, k], j, k, gamma_L) for k in range(J)]))

        s2_tilde = lambda_2_j * s2_H + (1 - lambda_2_j) * s2_L
        m2 = p2[j] - costs

        pi2 += s1_tilde[j] * np.sum(s2_tilde * m2)

    total_profit = pi1 + delta * pi2
    return -total_profit


# OPTIMIZE
## PERFECT

init_p1_perf = np.tile(costs + 1.0, (2, 1))
init_p2_perf = np.tile(costs + 1.0, (2, J, 1))

init_params_perf = np.concatenate([init_p1_perf.flatten(), init_p2_perf.flatten()])


def perfect_info_obj(params):
    p1_flat = params[: 2 * J]
    p2_flat = params[2 * J :]
    return perfect_info_profit(p1_flat, p2_flat)


res_perfect = minimize(perfect_info_obj, init_params_perf, method="L-BFGS-B")
p1_opt_perf = res_perfect.x[: 2 * J].reshape((2, J))
p2_opt_perf = res_perfect.x[2 * J :].reshape((2, J, J))

console.print("--- PERFECT INFORMATION OPTIMAL PRICES AND PROFIT ---")
console.print("PERIOD 1 PRICES TYPE H:", p1_opt_perf[0])
console.print("PERIOD 1 PRICES TYPE L:", p1_opt_perf[1])
console.print("MAX PROFIT (PERFECT INFO)", -res_perfect.fun)


## IMPERFECT
init_p1 = costs + 1.0
init_p2 = np.tile(costs + 1.0, (J, 1))
init_params = np.concatenate([init_p1, init_p2.flatten()])

res_imperfect = minimize(imperfect_info_profit, init_params, method="L-BFGS-B")
console.print("--- IMPERFECT INFORMATION OPTIMAL PRICES AND PROFIT ---")
console.print("PERIOD 1 PRICES", res_imperfect.x[:J])
console.print("MAX PROFIT", -res_imperfect.fun)

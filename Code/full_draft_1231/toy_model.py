import numpy as np
from scipy.optimize import minimize
from rich.traceback import install
import pandas as pd

install()
from rich.console import Console

console = Console()

from joblib import Parallel, delayed

N_SIMS = 10000
np.random.seed(219)

lambda_1 = np.random.uniform(low=0.05, high=0.95, size=N_SIMS)

# Plug in Recovered Parameter Estimates

beta_0 = 0.95
beta_B = -0.77
beta_P = -1.42
gamma_H = -0.11
gamma_L = -0.22
alpha = -0.98
delta = 0.95
costs = np.array([1.0, 1.2])
J = len(costs)


# Logit probs
def logit_probs(V):
    eV = np.exp(V)
    return eV / (1.0 + np.sum(eV))


# Utility
def val_p1(p):
    return beta_0 + alpha * p


def val_p2(p, prev_j, k, gamma_r):
    switching = 1.0 if prev_j != k else 0.0
    return beta_0 + gamma_r * switching + alpha * p


def run_simulation_draw(sim_id, l1):
    def perfect_info_profit(params):
        p1 = params[: 2 * J].reshape((2, J))
        p2 = params[2 * J :].reshape((2, J, J))

        types = [{"g": gamma_H, "w": l1}, {"g": gamma_L, "w": 1.0 - l1}]
        tot_prof = 0.0

        for r_idx, t in enumerate(types):
            s1 = logit_probs(val_p1(p1[r_idx]))
            m1 = p1[r_idx] - costs
            pi1 = np.sum(s1 * m1)

            pi2 = 0.0
            for j in range(J):
                V2 = np.array([val_p2(p2[r_idx, j, k], j, k, t["g"]) for k in range(J)])
                s2 = logit_probs(V2)
                m2 = p2[r_idx, j] - costs
                pi2 += s1[j] * np.sum(s2 * m2)

            tot_prof += t["w"] * (pi1 + delta * pi2)
        return -tot_prof

    def imperfect_info_profit(params):
        p1 = params[:J]
        p2 = params[J:].reshape((J, J))

        s1_H = logit_probs(val_p1(p1))
        s1_L = logit_probs(val_p1(p1))
        s1_tilde = l1 * s1_H + (1 - l1) * s1_L
        pi1 = np.sum(s1_tilde * (p1 - costs))

        pi2 = 0.0

        for j in range(J):
            l2_j = (l1 * s1_H[j]) / s1_tilde[j]

            s2_H = logit_probs(
                np.array([val_p2(p2[j, k], j, k, gamma_H) for k in range(J)])
            )
            s2_L = logit_probs(
                np.array([val_p2(p2[j, k], j, k, gamma_L) for k in range(J)])
            )

            s2_tilde = l2_j * s2_H + (1.0 - l2_j) * s2_L
            pi2 += s1_tilde[j] * np.sum(s2_tilde * (p2[j] - costs))
        return -(pi1 + delta * pi2)

    init_perf = np.concatenate(
        [
            np.tile(costs + 1.0, (2, 1)).flatten(),
            np.tile(costs + 1.0, (2, J, 1)).flatten(),
        ]
    )
    init_imperf = np.concatenate([costs + 1.0, np.tile(costs + 1.0, (J, 1)).flatten()])

    res_perf = minimize(perfect_info_profit, init_perf, method="L-BFGS-B")
    res_imperf = minimize(imperfect_info_profit, init_imperf, method="L-BFGS-B")

    perf_p1 = res_perf.x[: 2 * J].reshape((2, J))
    imperf_p1 = res_imperf.x[:J]

    return {
        "sim_id": sim_id,
        "lambda_l": l1,
        "perf_p1_H_j1": perf_p1[0, 0],
        "perf_p1_H_j2": perf_p1[0, 1],
        "perf_p1_L_j1": perf_p1[1, 0],
        "perf_p1_L_j2": perf_p1[1, 1],
        "perf_profit": -res_perf.fun,
        "imperf_p1_j1": imperf_p1[0],
        "imperf_p1_j2": imperf_p1[1],
        "imperf_profit": -res_imperf.fun,
    }


results_list = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_simulation_draw)(i, lambda_1[i]) for i in range(N_SIMS)
)
results_df = pd.DataFrame(results_list)
# Select the core metrics from describe() and transpose for better readability
summary_stats = results_df.drop(columns=["sim_id"]).describe().T

# Rename index for publication-ready labels
summary_stats.index = [
    "Prior P(r=H) [λ₁]",
    "Perf. Info: P1 Price (Type H, Good 1)",
    "Perf. Info: P1 Price (Type H, Good 2)",
    "Perf. Info: P1 Price (Type L, Good 1)",
    "Perf. Info: P1 Price (Type L, Good 2)",
    "Perf. Info: Total Expected Profit",
    "Imperf. Info: P1 Price (Good 1)",
    "Imperf. Info: P1 Price (Good 2)",
    "Imperf. Info: Total Expected Profit",
]

# Display as a clean Markdown table in Jupyter / Console output
console.print(
    summary_stats[["mean", "std", "min", "25%", "50%", "75%", "max"]].to_markdown(
        floatfmt=".4f"
    )
)

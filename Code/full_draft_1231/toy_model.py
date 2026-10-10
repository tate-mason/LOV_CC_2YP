import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ==========================================================
# Model Parameters
# ==========================================================

beta_0 = np.array([2.2, 2.0])
gamma_H = 2.0
gamma_L = 0.1
alpha = -1.5
delta = 0.95
costs = np.array([1.2, 1.0])
J = len(costs)

init_state_dist = np.array([0.5, 0.5])

PRICE_MIN = 1.0
PRICE_MAX = 8.0

# ==========================================================
# LOGIT PROBS & U()
# ==========================================================


def logit_probs(V):
    V_full = np.insert(V, 0, 0.0)
    eV = np.exp(V_full - np.max(V_full))
    probs = eV / np.sum(eV)
    return np.array(probs[1:]), float(probs[0])


def val_state(p, prev_j, gamma_r):
    """UTILITY WITH INITIAL STATE j_0"""

    switching = np.array([0.0 if k == prev_j else 1.0 for k in range(J)])
    return beta_0 + gamma_r * switching + alpha * p


# ==========================================================
# PRICING SOLVERS
# ==========================================================


def solve_perfect_info(l1):
    """Solves optimal prices under perfect info given population share l1 =
    N_H/N"""

    def obj(params):
        p1 = params[: 2 * J].reshape((2, J))
        p2 = params[2 * J :].reshape((2, J, J))

        types = [{"g": gamma_H, "w": l1}, {"g": gamma_L, "w": 1.0 - l1}]
        tot_prof = 0.0

        for r_idx, t in enumerate(types):
            if t["w"] == 0:
                continue
            s1 = np.zeros(J)
            for j0 in range(J):
                V1_j0 = val_state(p1[r_idx], j0, t["g"])
                s1_j0, _ = logit_probs(V1_j0)
                s1 += init_state_dist[j0] * s1_j0

            m1 = p1[r_idx] - costs
            pi1 = np.sum(s1 * m1)
            pi2 = 0.0
            for j in range(J):
                V2 = val_state(p2[r_idx, j], j, t["g"])
                s2, _ = logit_probs(V2)
                m2 = p2[r_idx, j] - costs
                pi2 += s1[j] * np.sum(s2 * m2)

            tot_prof += t["w"] * (pi1 + delta * pi2)
        return -tot_prof

    init_p = np.concatenate(
        [
            np.tile(costs + 1.2, (2, 1)).flatten(),
            np.tile(costs + 1.2, (2, J, 1)).flatten(),
        ]
    )
    bounds = [(PRICE_MIN, PRICE_MAX)] * len(init_p)
    res = minimize(obj, init_p, method="L-BFGS-B", bounds=bounds)
    p1_opt = res.x[: 2 * J].reshape((2, J))
    p2_opt = res.x[2 * J :].reshape((2, J, J))

    return p1_opt, p2_opt, -res.fun


def solve_imperfect_info(l1):
    """Solve optimal prices under imperfect info with Bayesian Updating"""

    def obj(params):
        p1 = params[:J]
        p2 = params[J:].reshape((J, J))

        s1_H = np.zeros(J)
        s1_L = np.zeros(J)

        for j0 in range(J):
            s1_H_j0, _ = logit_probs(val_state(p1, j0, gamma_H))
            s1_L_j0, _ = logit_probs(val_state(p1, j0, gamma_L))
            s1_H += init_state_dist[j0] * s1_H_j0
            s1_L += init_state_dist[j0] * s1_L_j0

        s1_tilde = l1 * s1_H + (1.0 - l1) * s1_L
        pi1 = np.sum(s1_tilde * (p1 - costs))

        pi2 = 0.0
        for j in range(J):
            l2_j = (l1 * s1_H[j]) / s1_tilde[j]

            s2_H, _ = logit_probs(val_state(p2[j], j, gamma_H))
            s2_L, _ = logit_probs(val_state(p2[j], j, gamma_L))

            s2_tilde = l2_j * s2_H + (1.0 - l2_j) * s2_L
            pi2 += s1_tilde[j] * np.sum(s2_tilde * ([p2[j] - costs]))
        return -(pi1 + delta * pi2)

    init_p = np.concatenate([costs + 1.2, np.tile(costs + 1.2, (J, 1)).flatten()])
    bounds = [(PRICE_MIN, PRICE_MAX)] * len(init_p)
    res = minimize(obj, init_p, method="L-BFGS-B", bounds=bounds)

    p1_opt = res.x[:J]
    p2_opt = res.x[J:].reshape((J, J))
    return p1_opt, p2_opt, -res.fun


# ============================================================
# GRID SEARCH OVER POPULATION COMPOSITIONS
# ============================================================

population_compositions = [
    (0.00, "100% LOW"),
    (0.25, "25% HIGH, 75% LOW"),
    (0.50, "50% HIGH, 50% LOW"),
    (0.75, "75% HIGH, 25% LOW"),
    (1.00, "100% HIGH"),
]

results = []

for l1, composition_label in population_compositions:
    perf_p1, perf_p2, perf_prof = solve_perfect_info(l1)
    imperf_p1, imperf_p2, imperf_prof = solve_imperfect_info(l1)

    s1_H = np.zeros(J)
    s1_L = np.zeros(J)

    for j0 in range(J):
        s1_H_j0, _ = logit_probs(val_state(imperf_p1, j0, gamma_H))
        s1_L_j0, _ = logit_probs(val_state(imperf_p1, j0, gamma_L))
        s1_H += init_state_dist[j0] * s1_H_j0
        s1_L += init_state_dist[j0] * s1_L_j0

    s1_tilde = l1 * s1_H + (1.0 - l1) * s1_L
    theo_l2 = (l1 * s1_H) / s1_tilde

    results.append(
        {
            "Population Mix": composition_label,
            "% HIGH (N_H/N)": f"{l1 * 100:.0f}%",
            "% LOW (N_L / N)": f"{(1 - l1) * 100:.0f}%",
            "Imperfect P1 (Good 1)": imperf_p1[0],
            "Imperfect P1 (Good 2)": imperf_p1[1],
            "Posterior | G1": theo_l2[0],
            "Posterior | G2": theo_l2[1],
            "Imperfect Profit": imperf_prof,
            "Perf Profit": perf_prof,
            "Value of Info": perf_prof - imperf_prof,
        }
    )

# ======================================================
# DISPLAY RESULTS
# ======================================================

print("=" * 20)
print("OPTIMAL PRICING AND PROFITS ACROSS TYPE DISTRIBUTIONS")
print("=" * 20)

df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False, floatfmt=".4f"))

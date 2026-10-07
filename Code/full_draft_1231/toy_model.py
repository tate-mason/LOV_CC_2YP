import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Set random seed for reproducibility
np.random.seed(42)

# =====================================================================
# 1. PARAMETERS & ESTIMATES RECOVERED FROM MODEL
# =====================================================================
beta_0 = 1.0  # Baseline utility intercept[cite: 1]
gamma_H = 1.5  # Utility from switching (Type H)[cite: 1]
gamma_L = 0.5  # Utility from switching (Type L)[cite: 1]
alpha = -0.8  # Disutility of price[cite: 1]
delta = 0.95  # Discount factor[cite: 3]
lambda_1 = 0.5  # Prior belief: 50/50 POC ratio[cite: 4]
costs = np.array([1.0, 1.2])  # Marginal costs for Goods 1 and 2[cite: 2]
J = len(costs)


# =====================================================================
# 2. LOGIT HELPER FUNCTIONS & OPTIMAL PRICING SOLVERS
# =====================================================================
def logit_probs(V):
    eV = np.exp(V - np.max(V))  # Numerical stability
    return eV / (1.0 + np.sum(eV))


def val_p1(p):
    return beta_0 + alpha * p[cite:1]


def val_p2(p, prev_j, k, gamma_r):
    switching = 1.0 if prev_j != k else 0.0
    return beta_0 + gamma_r * switching + alpha * p[cite:1]


# --- A. Perfect Information Solver ---
def solve_perfect_info():
    def perfect_info_profit(params):
        p1 = params[: 2 * J].reshape((2, J))
        p2 = params[2 * J :].reshape((2, J, J))

        types = [{"g": gamma_H, "w": lambda_1}, {"g": gamma_L, "w": 1.0 - lambda_1}]
        tot_prof = 0.0

        for r_idx, t in enumerate(types):
            s1 = logit_probs(val_p1(p1[r_idx]))
            m1 = p1[r_idx] - costs
            pi1 = np.sum(s1 * m1)[cite:3]

            pi2 = 0.0
            for j in range(J):
                V2 = np.array([val_p2(p2[r_idx, j, k], j, k, t["g"]) for k in range(J)])
                s2 = logit_probs(V2)
                m2 = p2[r_idx, j] - costs
                pi2 += s1[j] * np.sum(s2 * m2)[cite:3]

            tot_prof += t["w"] * (pi1 + delta * pi2)[cite:3]

        return -tot_prof

    init_p = np.concatenate(
        [
            np.tile(costs + 1.0, (2, 1)).flatten(),
            np.tile(costs + 1.0, (2, J, 1)).flatten(),
        ]
    )
    res = minimize(perfect_info_profit, init_p, method="L-BFGS-B")

    p1_opt = res.x[: 2 * J].reshape((2, J))
    p2_opt = res.x[2 * J :].reshape((2, J, J))
    exp_profit = -res.fun
    return p1_opt, p2_opt, exp_profit


# --- B. Imperfect Information Solver ---
def solve_imperfect_info():
    def imperfect_info_profit(params):
        p1 = params[:J]
        p2 = params[J:].reshape((J, J))

        s1_H = logit_probs(val_p1(p1))
        s1_L = logit_probs(val_p1(p1))
        s1_tilde = lambda_1 * s1_H + (1.0 - lambda_1) * s1_L[cite:4]
        pi1 = np.sum(s1_tilde * (p1 - costs))[cite:4]

        pi2 = 0.0
        for j in range(J):
            # Bayesian update lambda_2(j)[cite: 4]
            l2_j = (lambda_1 * s1_H[j]) / s1_tilde[j][cite:4]

            s2_H = logit_probs(
                np.array([val_p2(p2[j, k], j, k, gamma_H) for k in range(J)])
            )
            s2_L = logit_probs(
                np.array([val_p2(p2[j, k], j, k, gamma_L) for k in range(J)])
            )

            s2_tilde = l2_j * s2_H + (1.0 - l2_j) * s2_L[cite:4]
            pi2 += s1_tilde[j] * np.sum(s2_tilde * (p2[j] - costs))[cite:4]

        return -(pi1 + delta * pi2)[cite:4]

    init_p = np.concatenate([costs + 1.0, np.tile(costs + 1.0, (J, 1)).flatten()])
    res = minimize(imperfect_info_profit, init_p, method="L-BFGS-B")

    p1_opt = res.x[:J]
    p2_opt = res.x[J:].reshape((J, J))
    exp_profit = -res.fun
    return p1_opt, p2_opt, exp_profit


# Solve Firm Optimization
perf_p1, perf_p2, perf_exp_profit = solve_perfect_info()
imperf_p1, imperf_p2, imperf_exp_profit = solve_imperfect_info()


# =====================================================================
# 3. CONSUMER SIMULATION ENGINE (10,000 PATHS)
# =====================================================================
def draw_gumbel(shape):
    return np.random.gumbel(loc=0.0, scale=1.0, size=shape)


def simulate_consumer_paths(N, p1_matrix, p2_tensor, is_imperfect=False):
    """
    Simulates N individual consumer decisions over 2 periods.
    """
    # 50/50 Type Assignment
    types = np.random.choice(["High", "Low"], size=N, p=[lambda_1, 1.0 - lambda_1])
    gamma_i = np.where(types == "High", gamma_H, gamma_L)

    # Period 1
    p1_choices = np.zeros(N, dtype=int)
    p1_margins = np.zeros(N)

    for i in range(N):
        r_idx = 0 if types[i] == "High" else 1
        p1_prices = p1_matrix if is_imperfect else p1_matrix[r_idx]

        V1 = val_p1(p1_prices)
        eps1_inside = draw_gumbel((1, J))
        eps1_outside = draw_gumbel((1, 1))

        U1 = np.hstack([eps1_outside, V1 + eps1_inside])
        choice = np.argmax(U1)
        p1_choices[i] = choice
        if choice > 0:
            p1_margins[i] = p1_prices[choice - 1] - costs[choice - 1]

    # Period 2
    p2_choices = np.zeros(N, dtype=int)
    p2_margins = np.zeros(N)

    for i in range(N):
        prev_choice = p1_choices[i]
        r_idx = 0 if types[i] == "High" else 1

        if prev_choice == 0:
            # Chosen outside option in P1
            p2_prices = p1_matrix if is_imperfect else p1_matrix[r_idx]
            switching = np.zeros(J)
        else:
            prev_prod = prev_choice - 1
            p2_prices = (
                p2_tensor[prev_prod] if is_imperfect else p2_tensor[r_idx, prev_prod]
            )
            switching = np.array([0.0 if k == prev_prod else 1.0 for k in range(J)])

        V2 = beta_0 + gamma_i[i] * switching + alpha * p2_prices[cite:1]
        eps2_inside = draw_gumbel((1, J))
        eps2_outside = draw_gumbel((1, 1))

        U2 = np.hstack([eps2_outside, V2 + eps2_inside])
        choice = np.argmax(U2)
        p2_choices[i] = choice
        if choice > 0:
            p2_margins[i] = p2_prices[choice - 1] - costs[choice - 1]

    df = pd.DataFrame(
        {
            "type": types,
            "p1_choice": p1_choices,
            "p2_choice": p2_choices,
            "p1_margin": p1_margins,
            "p2_margin": p2_margins,
        }
    )

    # Label behavior
    def classify_behavior(r):
        if r["p1_choice"] == 0 or r["p2_choice"] == 0:
            return "Outside Option"
        elif r["p1_choice"] == r["p2_choice"]:
            return "Repeat Buyer (Loyal)"
        else:
            return "Switched Products"

    df["behavior"] = df.apply(classify_behavior, axis=1)
    return df


# Run simulations for 10,000 consumers under both market regimes
N_SIMS = 10000
df_perf = simulate_consumer_paths(N_SIMS, perf_p1, perf_p2, is_imperfect=False)
df_imperf = simulate_consumer_paths(N_SIMS, imperf_p1, imperf_p2, is_imperfect=True)

# =====================================================================
# 4. RESULTS & SUMMARY STATISTIC TABLES
# =====================================================================

# Summary Table 1: Optimal Pricing Strategies
pricing_summary = pd.DataFrame(
    {
        "Metric / Price Variable": [
            "P1 Price: Good 1",
            "P1 Price: Good 2",
            "P2 Price (after Good 1): Good 1",
            "P2 Price (after Good 1): Good 2",
            "P2 Price (after Good 2): Good 1",
            "P2 Price (after Good 2): Good 2",
            "Expected Total Profit per Consumer",
        ],
        "Perfect Info (Type H)": [
            f"${perf_p1[0, 0]:.4f}",
            f"${perf_p1[0, 1]:.4f}",
            f"${perf_p2[0, 0, 0]:.4f}",
            f"${perf_p2[0, 0, 1]:.4f}",
            f"${perf_p2[0, 1, 0]:.4f}",
            f"${perf_p2[0, 1, 1]:.4f}",
            f"${perf_exp_profit:.4f}",
        ],
        "Perfect Info (Type L)": [
            f"${perf_p1[1, 0]:.4f}",
            f"${perf_p1[1, 1]:.4f}",
            f"${perf_p2[1, 0, 0]:.4f}",
            f"${perf_p2[1, 0, 1]:.4f}",
            f"${perf_p2[1, 1, 0]:.4f}",
            f"${perf_p2[1, 1, 1]:.4f}",
            f"${perf_exp_profit:.4f}",
        ],
        "Imperfect Info (Uniform P1)": [
            f"${imperf_p1[0]:.4f}",
            f"${imperf_p1[1]:.4f}",
            f"${imperf_p2[0, 0]:.4f}",
            f"${imperf_p2[0, 1]:.4f}",
            f"${imperf_p2[1, 0]:.4f}",
            f"${imperf_p2[1, 1]:.4f}",
            f"${imperf_exp_profit:.4f}",
        ],
    }
)

# Summary Table 2: Realized Consumer Pathways (Imperfect Info Case)
path_breakdown = (
    pd.crosstab(df_imperf["type"], df_imperf["behavior"], normalize="index") * 100
)

# Summary Table 3: Bayesian Learning Validation
# Compare Firm's Priors vs. Posterior Beliefs vs. Simulated Empirical Realization
s1_H = logit_probs(val_p1(imperf_p1))
s1_L = logit_probs(val_p1(imperf_p1))
s1_tilde = lambda_1 * s1_H + (1.0 - lambda_1) * s1_L[cite:4]
theo_l2 = (lambda_1 * s1_H) / s1_tilde  # Theoretical Bayes Updates[cite: 4]

emp_l2_g1 = (df_imperf[df_imperf["p1_choice"] == 1]["type"] == "High").mean()
emp_l2_g2 = (df_imperf[df_imperf["p1_choice"] == 2]["type"] == "High").mean()

learning_summary = pd.DataFrame(
    {
        "Period 1 Choice Observed": ["Good 1 Chosen", "Good 2 Chosen"],
        "Prior Belief P(H)": [f"{lambda_1:.4f}", f"{lambda_1:.4f}"],
        "Theoretical Posterior λ₂(j)": [f"{theo_l2[0]:.4f}", f"{theo_l2[1]:.4f}"],
        "Empirical Realized Share % High": [f"{emp_l2_g1:.4f}", f"{emp_l2_g2:.4f}"],
    }
)

# Output Tables
print("=========================================================================")
print("TABLE 1: OPTIMAL FIRM PRICING & EXPECTED PROFITS")
print("=========================================================================")
print(pricing_summary.to_markdown(index=False))

print("\n=========================================================================")
print("TABLE 2: REALIZED CONSUMER BEHAVIOR PATTERNS (% BY TYPE, IMPERFECT INFO)")
print("=========================================================================")
print(path_breakdown.to_markdown(floatfmt=".2f"))

print("\n=========================================================================")
print("TABLE 3: FIRM BAYESIAN LEARNING & BELIEF UPDATING VALIDATION")
print("=========================================================================")
print(learning_summary.to_markdown(index=False))

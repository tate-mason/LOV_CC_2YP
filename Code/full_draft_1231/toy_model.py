import numpy as np
import pandas as pd
from scipy.optimize import minimize

np.random.seed(42)

# =====================================================================
# 1. ECONOMICS & PARAMETER SETUP
# =====================================================================
# Asymmetric product baseline utilities to induce type self-selection
beta_0 = np.array([2.5, 2.0])  # Good 1 has higher baseline utility
gamma_H = 1.0  # High type switching utility gain
gamma_L = 0.1  # Low type switching utility gain
alpha = -1.5  # Price sensitivity (elast. centered around ~$2.00-$3.00)
delta = 0.95  # Discount factor
lambda_1 = 0.5  # Prior belief P(High Type)
costs = np.array([1.0, 1.2])  # Marginal costs for Good 1 and Good 2
J = len(costs)

PRICE_MIN = 1.0
PRICE_MAX = 8.0


# =====================================================================
# 2. CHOICE PROBABILITIES & UTILITY FUNCTIONS
# =====================================================================
def logit_probs(V):
    # Logit choice probabilities with outside option V_0 = 0
    V_full = np.insert(V, 0, 0.0)
    eV = np.exp(V_full - np.max(V_full))
    probs = eV / np.sum(eV)
    return probs[1:], probs[0]  # (inside_shares, outside_share)


def val_p1(p):
    return beta_0 + alpha * p


def val_p2(p, prev_j, gamma_r):
    # prev_j is index 0 or 1. Switching occurs when new choice k != prev_j
    switching = np.array([0.0 if k == prev_j else 1.0 for k in range(J)])
    return beta_0 + gamma_r * switching + alpha * p


def safe_div(num, den):
    return num / np.maximum(den, 1e-12)


# =====================================================================
# 3. FIRM PRICING OPTIMIZATION
# =====================================================================
def solve_perfect_info():
    def obj(params):
        p1 = params[: 2 * J].reshape((2, J))
        p2 = params[2 * J :].reshape((2, J, J))

        types = [{"g": gamma_H, "w": lambda_1}, {"g": gamma_L, "w": 1.0 - lambda_1}]
        tot_prof = 0.0

        for r_idx, t in enumerate(types):
            s1, _ = logit_probs(val_p1(p1[r_idx]))
            m1 = p1[r_idx] - costs
            pi1 = np.sum(s1 * m1)

            pi2 = 0.0
            for j in range(J):
                V2 = val_p2(p2[r_idx, j], j, t["g"])
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


def solve_imperfect_info():
    def obj(params):
        p1 = params[:J]
        p2 = params[J:].reshape((J, J))

        s1_H, _ = logit_probs(val_p1(p1))
        s1_L, _ = logit_probs(val_p1(p1))

        # Pooled Period 1 Share
        s1_tilde = lambda_1 * s1_H + (1.0 - lambda_1) * s1_L
        pi1 = np.sum(s1_tilde * (p1 - costs))

        pi2 = 0.0
        for j in range(J):
            # Bayesian posterior update lambda_2(j)
            l2_j = safe_div(lambda_1 * s1_H[j], s1_tilde[j])

            s2_H, _ = logit_probs(val_p2(p2[j], j, gamma_H))
            s2_L, _ = logit_probs(val_p2(p2[j], j, gamma_L))

            s2_tilde = l2_j * s2_H + (1.0 - l2_j) * s2_L
            pi2 += s1_tilde[j] * np.sum(s2_tilde * (p2[j] - costs))

        return -(pi1 + delta * pi2)

    init_p = np.concatenate([costs + 1.2, np.tile(costs + 1.2, (J, 1)).flatten()])
    bounds = [(PRICE_MIN, PRICE_MAX)] * len(init_p)
    res = minimize(obj, init_p, method="L-BFGS-B", bounds=bounds)

    p1_opt = res.x[:J]
    p2_opt = res.x[J:].reshape((J, J))
    return p1_opt, p2_opt, -res.fun


perf_p1, perf_p2, perf_exp_profit = solve_perfect_info()
imperf_p1, imperf_p2, imperf_exp_profit = solve_imperfect_info()


# =====================================================================
# 4. SIMULATION ENGINE (10,000 CONSUMER PATHWAYS)
# =====================================================================
def draw_gumbel(shape):
    return np.random.gumbel(loc=0.0, scale=1.0, size=shape)


def simulate_consumer_paths(N, p1_matrix, p2_tensor, is_imperfect=False):
    types = np.random.choice(["High", "Low"], size=N, p=[lambda_1, 1.0 - lambda_1])
    gamma_i = np.where(types == "High", gamma_H, gamma_L)

    p1_choices = np.zeros(N, dtype=int)
    p2_choices = np.zeros(N, dtype=int)

    for i in range(N):
        r_idx = 0 if types[i] == "High" else 1
        p1_prices = p1_matrix if is_imperfect else p1_matrix[r_idx]

        # Period 1 Choice
        V1 = val_p1(p1_prices)
        U1 = np.hstack([draw_gumbel((1, 1)), V1 + draw_gumbel((1, J))])
        c1 = np.argmax(U1)
        p1_choices[i] = c1

        # Period 2 Choice
        if c1 == 0:
            p2_prices = p1_prices
            V2 = val_p1(p2_prices)
        else:
            prev_prod = c1 - 1
            p2_prices = (
                p2_tensor[prev_prod] if is_imperfect else p2_tensor[r_idx, prev_prod]
            )
            V2 = val_p2(p2_prices, prev_prod, gamma_i[i])

        U2 = np.hstack([draw_gumbel((1, 1)), V2 + draw_gumbel((1, J))])
        p2_choices[i] = np.argmax(U2)

    df = pd.DataFrame({"type": types, "p1_choice": p1_choices, "p2_choice": p2_choices})

    def classify(r):
        if r["p1_choice"] == 0 or r["p2_choice"] == 0:
            return "Outside Option"
        return (
            "Repeat Buyer (Loyal)"
            if r["p1_choice"] == r["p2_choice"]
            else "Switched Products"
        )

    df["behavior"] = df.apply(classify, axis=1)
    return df


N_SIMS = 10000
df_imperf = simulate_consumer_paths(N_SIMS, imperf_p1, imperf_p2, is_imperfect=True)

# =====================================================================
# 5. SUMMARY TABLES & VERIFICATION
# =====================================================================
pricing_summary = pd.DataFrame(
    {
        "Metric / Price Variable": [
            "P1 Price: Good 1",
            "P1 Price: Good 2",
            "P2 Price (after Good 1): Good 1",
            "P2 Price (after Good 1): Good 2",
            "P2 Price (after Good 2): Good 1",
            "P2 Price (after Good 2): Good 2",
            "Expected Total Profit",
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
        "Imperfect Info": [
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

path_breakdown = (
    pd.crosstab(df_imperf["type"], df_imperf["behavior"], normalize="index") * 100
)

s1_H, _ = logit_probs(val_p1(imperf_p1))
s1_L, _ = logit_probs(val_p1(imperf_p1))
s1_tilde = lambda_1 * s1_H + (1.0 - lambda_1) * s1_L
theo_l2 = safe_div(lambda_1 * s1_H, s1_tilde)

g1_mask = df_imperf["p1_choice"] == 1
g2_mask = df_imperf["p1_choice"] == 2

emp_l2_g1 = (df_imperf[g1_mask]["type"] == "High").mean() if g1_mask.sum() > 0 else 0.0
emp_l2_g2 = (df_imperf[g2_mask]["type"] == "High").mean() if g2_mask.sum() > 0 else 0.0

learning_summary = pd.DataFrame(
    {
        "P1 Choice": ["Good 1 Chosen", "Good 2 Chosen"],
        "Prior P(H)": [f"{lambda_1:.4f}", f"{lambda_1:.4f}"],
        "Theoretical Posterior λ₂": [f"{theo_l2[0]:.4f}", f"{theo_l2[1]:.4f}"],
        "Empirical % High": [f"{emp_l2_g1:.4f}", f"{emp_l2_g2:.4f}"],
    }
)

print("=========================================================================")
print("TABLE 1: OPTIMAL FIRM PRICING & EXPECTED PROFITS")
print("=========================================================================")
print(pricing_summary.to_markdown(index=False))

print("\n=========================================================================")
print("TABLE 2: CONSUMER BEHAVIOR PATTERNS (% BY TYPE, IMPERFECT INFO)")
print("=========================================================================")
print(path_breakdown.to_markdown(floatfmt=".2f"))

print("\n=========================================================================")
print("TABLE 3: BAYESIAN LEARNING VALIDATION")
print("=========================================================================")
print(learning_summary.to_markdown(index=False))

"""
empirical_pricing_plain_vs_all.py
=================================
Purpose:
    Conducts the empirical pricing exercise:
    Pricing PLAIN YOGURT (the inside good) against ALL OTHER CHOICES
    (berry, other yogurt, or no yogurt) using the estimated demand parameters.

Comparison:
    - UNIFORM PRICING (Uninformed baseline):
        The firm observes purchase history but not the consumer's switching type.
        Everyone with the same history gets the same price offer.
    - PRICE DISCRIMINATION (Informed counterfactual):
        The firm observes history AND the consumer's switching quintile (Q1 - Q5).
        Offers targeted price paths to each quintile.

Parameters from the Nielsen Demand Estimation (Table 2 in mason_draft.pdf):
    - Plain intercept:       beta  = -1.2851  (inside constant + beta_plain)
    - Mean price sensitivity:alpha = -3.5304  (-exp(mu_a + 0.5 * sd_a^2))
    - Normal switching utility:
        Mean:               mu_g  = -0.8382
        Std Dev:            sd_g  =  2.0992
        Trimming:           Retain central 90% (5th to 95th percentiles)
                            Divide retained mass into 5 equal quintiles (Q1 to Q5).

Style:
    Written for a beginner CS student.
    Uses straightforward loops, explicit simulation points, and clean output tables.

How to run:
    python3 dumbed_down_code/empirical_pricing_plain_vs_all.py
"""

import math

# =============================================================================
# 1. HELPER FUNCTIONS: NORMAL DISTRIBUTION & CHOICE PROBABILITIES
# =============================================================================

def standard_normal_cdf(x):
    """Cumulative probability P(Z <= x) for standard normal."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def normal_inverse_cdf(p, mean=0.0, sd=1.0):
    """Finds value x such that P(X <= x) = p using bisection search."""
    low = mean - 8.0 * sd
    high = mean + 8.0 * sd
    for _ in range(120):
        mid = (low + high) / 2.0
        prob = standard_normal_cdf((mid - mean) / sd)
        if prob < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0

def calculate_plain_share(price, history, gamma, beta, alpha):
    """
    Computes plain yogurt choice probability s = 1 / (1 + exp(-v)).
    
    If history == '0' (last week bought something else):
        v = beta + alpha * price + gamma
    If history == 'j' (last week bought plain yogurt):
        v = beta + alpha * price - gamma
    """
    if history == '0':
        v = beta + alpha * price + gamma
    elif history == 'j':
        v = beta + alpha * price - gamma
    else:
        raise ValueError("history must be 'j' or '0'")

    if v > 50.0:
        return 1.0
    elif v < -50.0:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-v))

def calculate_consumer_surplus_period(price, history, gamma, beta, alpha):
    """
    Consumer surplus in one period relative to the outside option (CS^Z):
        CS = log(1 + exp(v)) / (-alpha)
    """
    if history == '0':
        v = beta + alpha * price + gamma
    else:
        v = beta + alpha * price - gamma

    if v > 50.0:
        log_term = v
    elif v < -50.0:
        log_term = 0.0
    else:
        log_term = math.log(1.0 + math.exp(v))

    return log_term / (-alpha)

# =============================================================================
# 2. DISCRETIZE THE ESTIMATED & TRIMMED GAMMA DISTRIBUTION
# =============================================================================

def build_trimmed_empirical_population(mean=-0.838193, sd=2.099165, points_per_quintile=25):
    """
    Builds the 5 quintiles from the estimated switching distribution:
        - Trim the bottom 5% and top 5% (retain between 0.05 and 0.95).
        - Divide the middle 90% into 5 equal quintiles:
            Q1:  5% to 23%
            Q2: 23% to 41%
            Q3: 41% to 59%
            Q4: 59% to 77%
            Q5: 77% to 95%
    """
    population = []
    
    for q_id in range(1, 6):
        # 0.18 percentile width for each quintile (5 * 0.18 = 0.90)
        prob_start = 0.05 + (q_id - 1) * 0.18
        
        for i in range(points_per_quintile):
            p = prob_start + (i + 0.5) * (0.18 / points_per_quintile)
            gamma_val = normal_inverse_cdf(p, mean=mean, sd=sd)
            # Total retained probability mass is normalized to 1.0, so each point gets:
            weight = 1.0 / (5.0 * points_per_quintile)
            population.append({
                'gamma': gamma_val,
                'weight': weight,
                'quintile': q_id
            })
            
    return population

# =============================================================================
# 3. FIRM OPTIMIZATION SOLVER (GRID SEARCH OVER NONNEGATIVE PRICES)
# =============================================================================

def solve_period2_price(history, consumer_group, beta, alpha, cost=0.0):
    """
    Period 2 optimal price search (p2 >= 0):
        Max Revenue = sum_c weight_c * [ p2 * share(p2) ]
    """
    best_price = 0.0
    best_revenue = -1e9

    # Candidate prices from $0.00 to $4.00 in half-cent steps
    candidate_prices = [i * 0.005 for i in range(0, 801)]
    total_weight = sum(c['weight'] for c in consumer_group)

    for p in candidate_prices:
        rev = 0.0
        for c in consumer_group:
            w = c['weight'] / total_weight
            s = calculate_plain_share(p, history, c['gamma'], beta, alpha)
            rev += w * (p - cost) * s

        if rev > best_revenue:
            best_revenue = rev
            best_price = p

    return best_price, best_revenue

def solve_period1_price(h0, consumer_group, p2_inside, p2_outside, beta, alpha, delta=1.0, cost=0.0):
    """
    Period 1 optimal price search (p1 >= 0):
        Max Expected 2-Period Revenue
    """
    best_price = 0.0
    best_total_revenue = -1e9

    candidate_prices = [i * 0.005 for i in range(0, 801)]
    total_weight = sum(c['weight'] for c in consumer_group)

    for p1 in candidate_prices:
        two_period_rev = 0.0

        for c in consumer_group:
            g = c['gamma']
            w = c['weight'] / total_weight

            s1 = calculate_plain_share(p1, h0, g, beta, alpha)

            s2_j = calculate_plain_share(p2_inside, 'j', g, beta, alpha)
            s2_0 = calculate_plain_share(p2_outside, '0', g, beta, alpha)

            r2_j = (p2_inside - cost) * s2_j
            r2_0 = (p2_outside - cost) * s2_0

            consumer_rev = s1 * (p1 - cost) + delta * (s1 * r2_j + (1.0 - s1) * r2_0)
            two_period_rev += w * consumer_rev

        if two_period_rev > best_total_revenue:
            best_total_revenue = two_period_rev
            best_price = p1

    return best_price, best_total_revenue

def evaluate_policy_on_group(consumer_group, h0, p1, p2_inside, p2_outside, beta, alpha, delta=1.0, cost=0.0):
    """Computes expected Revenue, Consumer Surplus (CS^Z), and Total Welfare."""
    total_rev = 0.0
    total_cs = 0.0
    total_weight = sum(c['weight'] for c in consumer_group)

    for c in consumer_group:
        g = c['gamma']
        w = c['weight'] / total_weight

        s1 = calculate_plain_share(p1, h0, g, beta, alpha)
        cs1 = calculate_consumer_surplus_period(p1, h0, g, beta, alpha)

        s2_j = calculate_plain_share(p2_inside, 'j', g, beta, alpha)
        s2_0 = calculate_plain_share(p2_outside, '0', g, beta, alpha)

        cs2_j = calculate_consumer_surplus_period(p2_inside, 'j', g, beta, alpha)
        cs2_0 = calculate_consumer_surplus_period(p2_outside, '0', g, beta, alpha)

        exp_rev2 = s1 * (p2_inside - cost) * s2_j + (1.0 - s1) * (p2_outside - cost) * s2_0
        exp_cs2 = s1 * cs2_j + (1.0 - s1) * cs2_0

        total_rev += w * (s1 * (p1 - cost) + delta * exp_rev2)
        total_cs += w * (cs1 + delta * exp_cs2)

    return {
        'revenue': total_rev,
        'consumer_surplus': total_cs,
        'total_welfare': total_rev + total_cs
    }

# =============================================================================
# 4. MAIN RUNNER
# =============================================================================

def main():
    print("=" * 80)
    print("EMPIRICAL PRICING: PLAIN YOGURT VS ALL OTHER CHOICES")
    print("Calibrated from Nielsen demand estimates (mason_draft.pdf):")
    print("  beta_plain = -1.2851,  alpha = -3.5304,  mu_gamma = -0.8382,  sd_gamma = 2.0992")
    print("=" * 80)

    beta = -1.2851
    alpha = -3.5304
    cost = 0.0
    delta = 1.0

    population = build_trimmed_empirical_population(mean=-0.838193, sd=2.099165, points_per_quintile=25)

    quintiles = {}
    for q in range(1, 6):
        quintiles[q] = [c for c in population if c['quintile'] == q]

    summary_table_rows = []

    for h0 in ['0', 'j']:
        h0_name = "OUTSIDE (last week did not buy plain)" if h0 == '0' else "INSIDE (last week bought plain)"
        print()
        print("*" * 80)
        print(f"INITIAL STATE: {h0_name}")
        print("*" * 80)

        # 1. Uniform Pricing
        uniform_p2_j, _ = solve_period2_price('j', population, beta, alpha, cost)
        uniform_p2_0, _ = solve_period2_price('0', population, beta, alpha, cost)
        uniform_p1, _ = solve_period1_price(h0, population, uniform_p2_j, uniform_p2_0, beta, alpha, delta, cost)

        print("\n--- UNIFORM PRICING POLICY ---")
        print(f"Period 1 Price:              p1*    = ${uniform_p1:.3f}")
        print(f"Period 2 Price (after plain): p2*(j) = ${uniform_p2_j:.3f}")
        print(f"Period 2 Price (after other): p2*(0) = ${uniform_p2_0:.3f}")

        # 2. Price Discrimination
        informed_policies = {}
        for q in range(1, 6):
            group = quintiles[q]
            p2_j_q, _ = solve_period2_price('j', group, beta, alpha, cost)
            p2_0_q, _ = solve_period2_price('0', group, beta, alpha, cost)
            p1_q, _ = solve_period1_price(h0, group, p2_j_q, p2_0_q, beta, alpha, delta, cost)
            informed_policies[q] = {
                'p1': p1_q,
                'p2j': p2_j_q,
                'p20': p2_0_q
            }

        print("\n--- PRICE DISCRIMINATION (TARGETED BY QUINTILE) ---")
        print(f"{'Quintile':<10}{'Mean Gamma':<15}{'p1*':<12}{'p2*(j)':<12}{'p2*(0)':<12}")
        print("-" * 60)
        for q in range(1, 6):
            group = quintiles[q]
            avg_g = sum(c['gamma'] for c in group) / len(group)
            pol = informed_policies[q]
            print(f"Q{q:<9}{avg_g:<15.3f}${pol['p1']:<11.3f}${pol['p2j']:<11.3f}${pol['p20']:<11.3f}")

        # 3. Welfare Comparison
        total_u_cs = 0.0
        total_u_rev = 0.0
        total_u_tw = 0.0

        total_i_cs = 0.0
        total_i_rev = 0.0
        total_i_tw = 0.0

        print("\n--- QUINTILE WELFARE & REVENUE CHANGES ---")
        print(f"{'Quintile':<10}{'Metric':<18}{'Uniform':<12}{'Informed':<12}{'Change (%)':<12}")
        print("-" * 65)

        for q in range(1, 6):
            group = quintiles[q]
            u_out = evaluate_policy_on_group(group, h0, uniform_p1, uniform_p2_j, uniform_p2_0, beta, alpha, delta, cost)
            pol = informed_policies[q]
            i_out = evaluate_policy_on_group(group, h0, pol['p1'], pol['p2j'], pol['p20'], beta, alpha, delta, cost)

            total_u_cs += 0.20 * u_out['consumer_surplus']
            total_u_rev += 0.20 * u_out['revenue']
            total_u_tw += 0.20 * u_out['total_welfare']

            total_i_cs += 0.20 * i_out['consumer_surplus']
            total_i_rev += 0.20 * i_out['revenue']
            total_i_tw += 0.20 * i_out['total_welfare']

            pct_cs = 100.0 * (i_out['consumer_surplus'] - u_out['consumer_surplus']) / abs(u_out['consumer_surplus'])
            pct_rev = 100.0 * (i_out['revenue'] - u_out['revenue']) / abs(u_out['revenue'])
            pct_tw = 100.0 * (i_out['total_welfare'] - u_out['total_welfare']) / abs(u_out['total_welfare'])

            print(f"Q{q:<9}{'Consumer Surplus':<18}${u_out['consumer_surplus']:<11.4f}${i_out['consumer_surplus']:<11.4f}{pct_cs:>+8.2f}%")
            print(f"Q{q:<9}{'Revenue':<18}${u_out['revenue']:<11.4f}${i_out['revenue']:<11.4f}{pct_rev:>+8.2f}%")
            print(f"Q{q:<9}{'Total Welfare':<18}${u_out['total_welfare']:<11.4f}${i_out['total_welfare']:<11.4f}{pct_tw:>+8.2f}%")
            print("-" * 65)

        avg_pct_cs = 100.0 * (total_i_cs - total_u_cs) / abs(total_u_cs)
        avg_pct_rev = 100.0 * (total_i_rev - total_u_rev) / abs(total_u_rev)
        avg_pct_tw = 100.0 * (total_i_tw - total_u_tw) / abs(total_u_tw)

        state_label = "Outside" if h0 == '0' else "Inside"
        summary_table_rows.append((state_label, "Consumer surplus", total_u_cs, total_i_cs, avg_pct_cs))
        summary_table_rows.append((state_label, "Revenue", total_u_rev, total_i_rev, avg_pct_rev))
        summary_table_rows.append((state_label, "Total welfare", total_u_tw, total_i_tw, avg_pct_tw))

    # =========================================================================
    # REPRODUCE TABLE 3 FROM MASON_DRAFT.PDF
    # =========================================================================
    print()
    print("=" * 80)
    print("REPRODUCTION OF TABLE 3 (mason_draft.pdf): PLAIN YOGURT AVERAGES")
    print("=" * 80)
    print(f"{'Initial State':<15}{'Outcome':<20}{'Uniform ($)':<15}{'Informed ($)':<15}{'Change (%)':<12}")
    print("-" * 75)
    for state, metric, u_val, i_val, pct in summary_table_rows:
        print(f"{state:<15}{metric:<20}{u_val:<15.4f}{i_val:<15.4f}{pct:>+8.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    main()

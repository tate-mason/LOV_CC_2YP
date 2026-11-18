"""
theory_pricing_simulation.py
============================
Purpose:
    Solve the 2-period theoretical pricing model numerically and compare:
      1. UNIFORM PRICING (the firm knows only consumer history, not type).
      2. PRICE DISCRIMINATION (the firm knows history AND switching quintile).

Style:
    Written for a beginner CS / Econ student.
    Uses simple for-loops, explicit functions, and straightforward grid searches.
    No complex matrix broadcasting or obscure one-liners.

How to run:
    python3 dumbed_down_code/theory_pricing_simulation.py
"""

import math

# =============================================================================
# 1. BASIC HELPER FUNCTIONS (STEP BY STEP)
# =============================================================================

def standard_normal_cdf(x):
    """
    Cumulative distribution function (CDF) for standard normal N(0, 1).
    Tells us the probability that a standard normal variable is <= x.
    Uses Python's built-in math.erf.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def normal_inverse_cdf(p, mean=0.0, sd=1.0):
    """
    Finds the value x such that P(X <= x) = p for N(mean, sd^2).
    Uses binary search (bisection) so a student can easily understand how it works!
    """
    low = mean - 6.0 * sd
    high = mean + 6.0 * sd
    for _ in range(100):
        mid = (low + high) / 2.0
        prob = standard_normal_cdf((mid - mean) / sd)
        if prob < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0

def calculate_choice_probability(price, history, gamma, beta, alpha):
    """
    Calculates the probability that a consumer buys the inside good j.
    
    Inputs:
        price: float (price charged by the firm)
        history: string ('j' if consumer bought inside last period, '0' if outside)
        gamma: float (consumer's switching preference)
        beta: float (base utility of inside good)
        alpha: float (price sensitivity, alpha < 0)
    
    Formula:
        If history == '0' (came from outside):
            utility difference v = beta + alpha * price + gamma
        If history == 'j' (came from inside):
            utility difference v = beta + alpha * price - gamma
            
        Probability s = exp(v) / (1 + exp(v)) = 1 / (1 + exp(-v))
    """
    # Step 1: determine sign of history
    if history == '0':
        v = beta + alpha * price + gamma
    elif history == 'j':
        v = beta + alpha * price - gamma
    else:
        raise ValueError("history must be 'j' or '0'")

    # Step 2: compute probability using 1 / (1 + exp(-v))
    # Clip v to prevent overflow if v is huge or tiny
    if v > 50.0:
        return 1.0
    elif v < -50.0:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-v))

def calculate_consumer_surplus_period(price, history, gamma, beta, alpha):
    """
    Calculates consumer surplus relative to the outside option in ONE period (CS^Z):
        CS = log(1 + exp(v)) / (-alpha)
    """
    if history == '0':
        v = beta + alpha * price + gamma
    else:
        v = beta + alpha * price - gamma

    # Safe log(1 + exp(v))
    if v > 50.0:
        log_term = v
    elif v < -50.0:
        log_term = 0.0
    else:
        log_term = math.log(1.0 + math.exp(v))

    return log_term / (-alpha)

# =============================================================================
# 2. NUMERICAL INTEGRATION: DISCRETIZING THE GAMMA DISTRIBUTION
# =============================================================================

def build_five_quintile_population(mean=0.0, sd=1.25, points_per_quintile=21):
    """
    Creates a list of (gamma_value, weight, quintile_id).
    We divide the distribution into 5 equal quintiles:
        Quintile 1: 0% to 20%
        Quintile 2: 20% to 40%
        Quintile 3: 40% to 60%
        Quintile 4: 60% to 80%
        Quintile 5: 80% to 100%
        
    Inside each quintile, we take evenly spaced points.
    """
    population = []
    
    for q_id in range(1, 6):
        prob_start = (q_id - 1) * 0.20
        prob_end = q_id * 0.20
        
        # Take evenly spaced evaluation points inside this quintile
        for i in range(points_per_quintile):
            # Centered midpoint in each little slice
            p = prob_start + (i + 0.5) * (0.20 / points_per_quintile)
            gamma_val = normal_inverse_cdf(p, mean=mean, sd=sd)
            weight = 1.0 / (5.0 * points_per_quintile)
            population.append({
                'gamma': gamma_val,
                'weight': weight,
                'quintile': q_id
            })
            
    return population

# =============================================================================
# 3. FIRM OPTIMIZATION SOLVER (GRID SEARCH)
# =============================================================================

def solve_optimal_period2_price(history, consumer_group, beta, alpha, cost=0.0):
    """
    Finds the price p2 >= 0 that maximizes Period 2 expected revenue:
        Revenue(p2) = sum_{consumer} weight * [ p2 * share(p2) ]
    
    Takes the long way: loops over a fine grid of candidate prices from 0.0 to 10.0!
    """
    best_price = 0.0
    best_revenue = -1e9

    # Generate candidate prices: 0.00, 0.01, 0.02, ..., 10.00
    candidate_prices = [i * 0.005 for i in range(0, 2001)]

    total_weight = sum(c['weight'] for c in consumer_group)

    for p in candidate_prices:
        expected_revenue = 0.0
        for c in consumer_group:
            share = calculate_choice_probability(p, history, c['gamma'], beta, alpha)
            # Probability-weighted revenue from this type
            norm_weight = c['weight'] / total_weight
            expected_revenue += norm_weight * (p - cost) * share

        if expected_revenue > best_revenue:
            best_revenue = expected_revenue
            best_price = p

    return best_price, best_revenue

def solve_optimal_period1_price(h0, consumer_group, p2_inside, p2_outside, beta, alpha, delta=1.0, cost=0.0):
    """
    Finds the price p1 >= 0 that maximizes 2-period expected revenue:
        Expected Revenue = sum_{consumer} weight * [
            s1 * (p1 - cost) + delta * { s1 * r2(j) + (1 - s1) * r2(0) }
        ]
        where:
            r2(j) = p2_inside * s2(p2_inside, 'j')
            r2(0) = p2_outside * s2(p2_outside, '0')
    """
    best_price = 0.0
    best_total_revenue = -1e9

    candidate_prices = [i * 0.005 for i in range(0, 2001)]
    total_weight = sum(c['weight'] for c in consumer_group)

    for p1 in candidate_prices:
        expected_two_period_revenue = 0.0

        for c in consumer_group:
            g = c['gamma']
            norm_weight = c['weight'] / total_weight

            # Period 1 purchase probability
            s1 = calculate_choice_probability(p1, h0, g, beta, alpha)

            # Period 2 purchase probabilities conditional on period 1 choice
            s2_after_j = calculate_choice_probability(p2_inside, 'j', g, beta, alpha)
            s2_after_0 = calculate_choice_probability(p2_outside, '0', g, beta, alpha)

            # Period 2 revenues for this specific consumer type
            r2_j = (p2_inside - cost) * s2_after_j
            r2_0 = (p2_outside - cost) * s2_after_0

            # 2-period revenue for this consumer
            consumer_rev = s1 * (p1 - cost) + delta * (s1 * r2_j + (1.0 - s1) * r2_0)

            expected_two_period_revenue += norm_weight * consumer_rev

        if expected_two_period_revenue > best_total_revenue:
            best_total_revenue = expected_two_period_revenue
            best_price = p1

    return best_price, best_total_revenue

# =============================================================================
# 4. POLICY EVALUATION (EVALUATING WELFARE FOR A GIVEN POLICY)
# =============================================================================

def evaluate_policy_on_group(consumer_group, h0, p1, p2_inside, p2_outside, beta, alpha, delta=1.0, cost=0.0):
    """
    Calculates expected Revenue, Consumer Surplus (CS^Z), and Total Welfare (TW)
    for a given policy (p1, p2_inside, p2_outside) on a group of consumers.
    """
    total_revenue = 0.0
    total_cs = 0.0
    total_weight = sum(c['weight'] for c in consumer_group)

    for c in consumer_group:
        g = c['gamma']
        norm_weight = c['weight'] / total_weight

        # Period 1 choice
        s1 = calculate_choice_probability(p1, h0, g, beta, alpha)
        cs1 = calculate_consumer_surplus_period(p1, h0, g, beta, alpha)

        # Period 2 choices
        s2_j = calculate_choice_probability(p2_inside, 'j', g, beta, alpha)
        s2_0 = calculate_choice_probability(p2_outside, '0', g, beta, alpha)

        cs2_j = calculate_consumer_surplus_period(p2_inside, 'j', g, beta, alpha)
        cs2_0 = calculate_consumer_surplus_period(p2_outside, '0', g, beta, alpha)

        # Expected period 2 outcomes
        expected_rev2 = s1 * (p2_inside - cost) * s2_j + (1.0 - s1) * (p2_outside - cost) * s2_0
        expected_cs2 = s1 * cs2_j + (1.0 - s1) * cs2_0

        # Total 2-period outcomes for this type
        consumer_rev = s1 * (p1 - cost) + delta * expected_rev2
        consumer_cs = cs1 + delta * expected_cs2

        total_revenue += norm_weight * consumer_rev
        total_cs += norm_weight * consumer_cs

    total_welfare = total_revenue + total_cs
    return {
        'revenue': total_revenue,
        'consumer_surplus': total_cs,
        'total_welfare': total_welfare
    }

# =============================================================================
# 5. MAIN SIMULATION AND COMPARISON RUNNER
# =============================================================================

def run_theory_comparison():
    print("=" * 80)
    print("THEORETICAL MODEL: UNIFORM PRICING VS PRICE DISCRIMINATION")
    print("Parameters: beta=2.0, alpha=-1.0, c=0.0, delta=1.0, gamma ~ N(0, 1.25^2)")
    print("=" * 80)

    beta = 2.0
    alpha = -1.0
    cost = 0.0
    delta = 1.0

    # Build population: 5 quintiles
    population = build_five_quintile_population(mean=0.0, sd=1.25, points_per_quintile=21)

    # Separate population into 5 quintile subsets
    quintiles = {}
    for q in range(1, 6):
        quintiles[q] = [c for c in population if c['quintile'] == q]

    # We run for both initial histories: h0 = '0' (starts outside) and h0 = 'j' (starts inside)
    for h0 in ['0', 'j']:
        h0_name = "OUTSIDE (h0 = 0)" if h0 == '0' else "INSIDE (h0 = j)"
        print()
        print("*" * 80)
        print(f"INITIAL CONSUMER STATE: {h0_name}")
        print("*" * 80)

        # ---------------------------------------------------------------------
        # 1. UNIFORM PRICING (Firm does NOT observe quintile)
        # ---------------------------------------------------------------------
        # Firm solves terminal period 2 using the full population
        uniform_p2_j, _ = solve_optimal_period2_price('j', population, beta, alpha, cost)
        uniform_p2_0, _ = solve_optimal_period2_price('0', population, beta, alpha, cost)
        # Firm solves period 1 using the full population
        uniform_p1, _ = solve_optimal_period1_price(h0, population, uniform_p2_j, uniform_p2_0, beta, alpha, delta, cost)

        print("\n--- UNIFORM PRICING POLICY (Same offers to everyone) ---")
        print(f"Period 1 Price:             p1*   = ${uniform_p1:.3f}")
        print(f"Period 2 Price after Inside: p2*(j)= ${uniform_p2_j:.3f}")
        print(f"Period 2 Price after Outside:p2*(0)= ${uniform_p2_0:.3f}")

        # ---------------------------------------------------------------------
        # 2. PRICE DISCRIMINATION (Firm OBSERVES the quintile)
        # ---------------------------------------------------------------------
        informed_policies = {}
        for q in range(1, 6):
            group = quintiles[q]
            p2_j_q, _ = solve_optimal_period2_price('j', group, beta, alpha, cost)
            p2_0_q, _ = solve_optimal_period2_price('0', group, beta, alpha, cost)
            p1_q, _ = solve_optimal_period1_price(h0, group, p2_j_q, p2_0_q, beta, alpha, delta, cost)
            informed_policies[q] = {
                'p1': p1_q,
                'p2j': p2_j_q,
                'p20': p2_0_q
            }

        print("\n--- PRICE DISCRIMINATION POLICIES (Targeted by Quintile) ---")
        print(f"{'Quintile':<10}{'Mean Gamma':<15}{'p1*':<12}{'p2*(j)':<12}{'p2*(0)':<12}")
        print("-" * 60)
        for q in range(1, 6):
            group = quintiles[q]
            avg_g = sum(c['gamma'] for c in group) / len(group)
            pol = informed_policies[q]
            print(f"Q{q:<9}{avg_g:<15.3f}${pol['p1']:<11.3f}${pol['p2j']:<11.3f}${pol['p20']:<11.3f}")

        # ---------------------------------------------------------------------
        # 3. EVALUATE WELFARE AND COMPARE
        # ---------------------------------------------------------------------
        print("\n--- WELFARE & REVENUE COMPARISON BY QUINTILE ---")
        print(f"{'Quintile':<10}{'Metric':<18}{'Uniform':<12}{'Informed':<12}{'Change (%)':<12}")
        print("-" * 65)

        total_u_cs = 0.0
        total_u_rev = 0.0
        total_u_tw = 0.0

        total_i_cs = 0.0
        total_i_rev = 0.0
        total_i_tw = 0.0

        for q in range(1, 6):
            group = quintiles[q]
            # Outcomes under Uniform policy
            u_out = evaluate_policy_on_group(group, h0, uniform_p1, uniform_p2_j, uniform_p2_0, beta, alpha, delta, cost)
            # Outcomes under Informed policy
            pol = informed_policies[q]
            i_out = evaluate_policy_on_group(group, h0, pol['p1'], pol['p2j'], pol['p20'], beta, alpha, delta, cost)

            # Weight is 0.20 for each quintile
            total_u_cs += 0.20 * u_out['consumer_surplus']
            total_u_rev += 0.20 * u_out['revenue']
            total_u_tw += 0.20 * u_out['total_welfare']

            total_i_cs += 0.20 * i_out['consumer_surplus']
            total_i_rev += 0.20 * i_out['revenue']
            total_i_tw += 0.20 * i_out['total_welfare']

            # Percentage changes: 100 * (Informed - Uniform) / |Uniform|
            pct_cs = 100.0 * (i_out['consumer_surplus'] - u_out['consumer_surplus']) / abs(u_out['consumer_surplus'])
            pct_rev = 100.0 * (i_out['revenue'] - u_out['revenue']) / abs(u_out['revenue'])
            pct_tw = 100.0 * (i_out['total_welfare'] - u_out['total_welfare']) / abs(u_out['total_welfare'])

            print(f"Q{q:<9}{'Consumer Surplus':<18}${u_out['consumer_surplus']:<11.4f}${i_out['consumer_surplus']:<11.4f}{pct_cs:>+8.2f}%")
            print(f"Q{q:<9}{'Revenue':<18}${u_out['revenue']:<11.4f}${i_out['revenue']:<11.4f}{pct_rev:>+8.2f}%")
            print(f"Q{q:<9}{'Total Welfare':<18}${u_out['total_welfare']:<11.4f}${i_out['total_welfare']:<11.4f}{pct_tw:>+8.2f}%")
            print("-" * 65)

        # Population Averages
        avg_pct_cs = 100.0 * (total_i_cs - total_u_cs) / abs(total_u_cs)
        avg_pct_rev = 100.0 * (total_i_rev - total_u_rev) / abs(total_u_rev)
        avg_pct_tw = 100.0 * (total_i_tw - total_u_tw) / abs(total_u_tw)

        print(f"{'OVERALL':<10}{'Consumer Surplus':<18}${total_u_cs:<11.4f}${total_i_cs:<11.4f}{avg_pct_cs:>+8.2f}%")
        print(f"{'OVERALL':<10}{'Revenue':<18}${total_u_rev:<11.4f}${total_i_rev:<11.4f}{avg_pct_rev:>+8.2f}%")
        print(f"{'OVERALL':<10}{'Total Welfare':<18}${total_u_tw:<11.4f}${total_i_tw:<11.4f}{avg_pct_tw:>+8.2f}%")
        print("=" * 65)

if __name__ == '__main__':
    run_theory_comparison()

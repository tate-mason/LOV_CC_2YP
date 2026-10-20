"""Two-period monopoly with myopic logit consumers and variety preferences.

Run python3 toy_model.py --help. Prices are net transaction prices. A period-one
outside choice exits permanently; initial inside history is unobserved by the
pricing rule and distributed according to init_state_dist. Profits are per
initial consumer. 'Uniform' means common across types, not across products.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

beta_0 = np.array([2.2, 2.0])
gamma_H, gamma_L = 2.0, 0.1
alpha, delta = -1.5, 0.95
costs = np.array([1.2, 1.0])
J = len(costs)
init_state_dist = np.array([0.5, 0.5])
PRICE_MIN, PRICE_MAX = 1.0, 8.0


def logit_probs(V):
    full = np.r_[0.0, V]
    e = np.exp(full - full.max())
    prob = e / e.sum()
    return prob[1:], float(prob[0])


def val_state(p, prev_j, gamma_r):
    return beta_0 + gamma_r * (np.arange(J) != prev_j) + alpha * np.asarray(p)


def shares_jac(p, gamma, state=None):
    """Average conditional Jacobians, never apply logit derivative to a mixture."""
    states = range(J) if state is None else [state]
    weights = init_state_dist if state is None else [1.0]
    s, jac = np.zeros(J), np.zeros((J, J))
    for h, weight in zip(states, weights):
        sh, _ = logit_probs(val_state(p, h, gamma))
        s += weight * sh
        jac += weight * alpha * (np.diag(sh) - np.outer(sh, sh))
    return s, jac


def maximize(fun):
    """Bounded multistart optimization with analytic gradients and status checks."""
    candidates = []
    for markup in (0.3, 1.2, 3.0):
        res = minimize(lambda p: -fun(p)[0], costs + markup,
                       jac=lambda p: -fun(p)[1], method='L-BFGS-B',
                       bounds=[(PRICE_MIN, PRICE_MAX)] * J,
                       options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 400})
        if res.success:
            candidates.append(res)
    if not candidates:
        raise RuntimeError('All pricing optimization starts failed')
    return min(candidates, key=lambda res: res.fun).x


@dataclass
class Policy:
    p1: np.ndarray
    p2: np.ndarray  # history j, destination k
    profit: float
    type_profits: np.ndarray
    posterior: np.ndarray


def evaluate_policy(p1, p2, gammas):
    """Expected discounted profit conditional on each type, before initial history."""
    values = []
    for gamma in gammas:
        s1, _ = shares_jac(p1, gamma)
        w = np.array([shares_jac(p2[j], gamma, j)[0] @ (p2[j] - costs)
                      for j in range(J)])
        values.append(s1 @ (p1 - costs + delta * w))
    return np.array(values)


def solve_common(gammas, weights):
    """Backward induction: reoptimize terminal prices at each induced posterior.

    The envelope derivative holds optimized terminal prices fixed but retains
    type-specific continuation profits, thus including the learning effect.
    """
    gammas, weights = np.asarray(gammas), np.asarray(weights, dtype=float)
    if len(gammas) != len(weights) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Nonnegative type weights must sum to one')
    cache = {}

    def backward(p1):
        key = tuple(p1)
        if key in cache:
            return cache[key]
        demand = [shares_jac(p1, g) for g in gammas]
        s1 = np.array([v[0] for v in demand])
        joint = weights[:, None] * s1
        posterior = joint / joint.sum(axis=0)
        p2, continuation = [], []
        for j in range(J):
            def terminal(p):
                vals, grads = [], []
                for g in gammas:
                    s, jac = shares_jac(p, g, j)
                    vals.append(s @ (p - costs))
                    grads.append(s + jac.T @ (p - costs))
                return posterior[:, j] @ vals, posterior[:, j] @ np.array(grads)
            price = maximize(terminal)
            p2.append(price)
            continuation.append([shares_jac(price, g, j)[0] @ (price - costs)
                                 for g in gammas])
        w = np.array(continuation).T
        effective = p1 - costs + delta * w
        by_type = np.sum(s1 * effective, axis=1)
        grad = sum(weights[r] * (s1[r] + demand[r][1].T @ effective[r])
                   for r in range(len(gammas)))
        result = (weights @ by_type, grad, np.array(p2), by_type, posterior)
        cache[key] = result
        return result

    p1 = maximize(lambda p: backward(p)[:2])
    profit, _, p2, type_profits, posterior = backward(p1)
    return Policy(p1, p2, profit, type_profits, posterior)


def solve_perfect_info(l1):
    """Legacy tuple interface; each type is solved even at zero population weight."""
    policies = [solve_common([g], [1.0]) for g in (gamma_H, gamma_L)]
    return (np.array([p.p1 for p in policies]), np.array([p.p2 for p in policies]),
            l1 * policies[0].profit + (1 - l1) * policies[1].profit)


def solve_imperfect_info(l1):
    p = solve_common([gamma_H, gamma_L], [l1, 1 - l1])
    return p.p1, p.p2, p.profit


def solve_newborn(gammas, weights):
    """Common prices for a newborn with no prior product and unobserved type.

    There is no switching bonus in the first purchase. Consequently that purchase
    conveys no type information, and the terminal posterior equals the prior.
    The firm can condition terminal prices on the purchased product, not type.
    """
    gammas, weights = np.asarray(gammas), np.asarray(weights)
    p2, values = [], []
    for j in range(J):
        def terminal(p):
            value, gradient = 0.0, np.zeros(J)
            for g, weight in zip(gammas, weights):
                s, jac = shares_jac(p, g, j)
                value += weight * (s @ (p - costs))
                gradient += weight * (s + jac.T @ (p - costs))
            return value, gradient
        p = maximize(terminal)
        p2.append(p)
        values.append([shares_jac(p, g, j)[0] @ (p - costs) for g in gammas])
    p2, w = np.array(p2), np.array(values).T
    average_w = weights @ w

    def first(p):
        s, _ = logit_probs(beta_0 + alpha * p)
        jac = alpha * (np.diag(s) - np.outer(s, s))
        effective = p - costs + delta * average_w
        return s @ effective, s + jac.T @ effective

    p1 = maximize(first)
    s1, _ = logit_probs(beta_0 + alpha * p1)
    by_type = np.array([s1 @ (p1 - costs + delta * wr) for wr in w])
    return Policy(p1, p2, weights @ by_type, by_type, np.tile(weights[:, None], (1, J)))


def evaluate_newborn(p1, p2, gammas):
    s1, _ = logit_probs(beta_0 + alpha * p1)
    return np.array([s1 @ (p1 - costs + delta * np.array([
        shares_jac(p2[j], g, j)[0] @ (p2[j] - costs) for j in range(J)]))
        for g in gammas])


def newborn_discount_analysis(output):
    """Choose between fixed high-type regular and low-type discounted menus.

    The discount is a period-two offer; the first-period price is held fixed.
    q is Pr(L), not the probability of randomly sending an offer. At q=1/2
    discount recipients are equally likely to be H or L. Sending probability
    itself yields an affine objective and has no interior optimum.
    """
    types = [gamma_H, gamma_L]
    regular = solve_newborn([gamma_H], [1.0])
    discounted = solve_newborn([gamma_L], [1.0])
    # Compare terminal discounts only: high types receive introductory investment
    # in period one, so their first price need not exceed low-type prices.
    discounted.p1 = regular.p1.copy()
    if np.any(discounted.p1 > regular.p1 + 1e-7) or np.any(discounted.p2 > regular.p2 + 1e-7):
        raise RuntimeError('Low-type menu is not a pure discount from the high-type menu')
    regular_values = evaluate_newborn(regular.p1, regular.p2, types)
    discount_values = evaluate_newborn(discounted.p1, discounted.p2, types)
    gains = discount_values - regular_values
    slope = gains[1] - gains[0]
    root = -gains[0] / slope if abs(slope) > 1e-12 else np.nan
    threshold = root if 0 <= root <= 1 else np.nan
    rows = []
    for q in np.arange(1, 100) / 100:
        common = solve_newborn(types, [1-q, q])
        regular_profit = (1-q) * regular_values[0] + q * regular_values[1]
        discount_profit = (1-q) * discount_values[0] + q * discount_values[1]
        rows.append(dict(probability_L=q, probability_H=1-q,
            regular_profit=regular_profit, discount_profit=discount_profit,
            discount_gain=discount_profit-regular_profit,
            offer_discount=discount_profit > regular_profit + 1e-9,
            best_fixed_menu_profit=max(regular_profit, discount_profit),
            reoptimized_common_profit=common.profit,
            common_p1_good1=common.p1[0], common_p1_good2=common.p1[1]))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'newborn_discount_probability.csv', index=False)
    price_rows = []
    common50 = solve_newborn(types, [0.5, 0.5])
    for name, policy in [('regular_H_menu', regular), ('discount_L_menu', discounted),
                          ('newborn_common_50_50', common50)]:
        for period, history, price in [(1, 'no prior product', policy.p1)] + [
                (2, f'good{j+1}', policy.p2[j]) for j in range(J)]:
            price_rows.append(dict(menu=name, period=period, history=history,
                                   price_good1=price[0], price_good2=price[1]))
    pd.DataFrame(price_rows).to_csv(output / 'newborn_prices.csv', index=False)
    report = ('# Newborn discount decision\n\n'
        'The newborn has no previous product. Their type is unobserved and '
        'Pr(L)=q, Pr(H)=1-q. The first purchase has no switching bonus and therefore '
        'does not reveal type. Type remains unobserved in period two.\n\n'
        'The regular menu is the two-period policy optimized for a known H type. '
        'The discount menu replaces terminal prices with the known-L optimum while keeping '
        'the regular first-period price. All its prices '
        'are verified to be weakly below regular prices. Both menus are fixed while q varies. '
        'Period-two offers depend on the first product purchased. The firm knows both '
        'types exogenously for purposes of constructing these benchmark menus.\n\n'
        f'Profit change from discounting an H newborn: {gains[0]:.8f}\n\n'
        f'Profit change from discounting an L newborn: {gains[1]:.8f}\n\n'
        f'Break-even Pr(L): {threshold:.8f}\n\n'
        f'Break-even Pr(H): {1-threshold:.8f}\n\n'
        f'50/50 regular profit: {regular_values.mean():.8f}\n\n'
        f'50/50 discount profit: {discount_values.mean():.8f}\n\n'
        f'50/50 reoptimized common profit: {common50.profit:.8f}\n\n'
        'Expected gain = (1-q)*Delta_H + q*Delta_L. The exact cutoff is '
        '-Delta_H/(Delta_L-Delta_H). Offer the fixed discount when the expected '
        'gain is positive; at zero the firm is indifferent. No fixed adoption cost '
        'or classifier error is added.\n\n'
        'The CSV sweeps q from 0.01 through 0.99 and also solves freely reoptimized '
        'common prices at every q. The fixed-menu cutoff does not imply a discontinuity '
        'in those freely optimized prices. It depends on the specified regular and '
        'discount menus.\n\n'
        'If the probability being varied instead means the random offer probability a, '
        'expected profit is regular_profit + a*(discount_profit-regular_profit). '
        'Then the optimum is a=0 or a=1 (or any a at indifference), rather than an '
        'interior threshold in a.\n')
    (output / 'newborn_results.md').write_text(report)
    print(report)
    return frame, threshold


def run_analysis(output, points=21, gamma_max=3.0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    types = [gamma_H, gamma_L]
    personal = [solve_common([g], [1.0]) for g in types]
    uniform = solve_common(types, [0.5, 0.5])
    mix_cache = {}

    def mixture(lam):
        if lam not in mix_cache:
            common = solve_common(types, [lam, 1 - lam])
            perfect = lam * personal[0].profit + (1 - lam) * personal[1].profit
            mix_cache[lam] = common, perfect, perfect - common.profit
        return mix_cache[lam]

    rows = []
    grid = np.unique(np.r_[np.linspace(0, 1, points), 0.5])
    for lam in grid:
        p, perfect, gain = mixture(float(lam))
        row = dict(high_share=lam, common_profit=p.profit, targeted_profit=perfect,
                   targeting_gain=gain,
                   common_profit_H=p.type_profits[0], common_profit_L=p.type_profits[1])
        for j in range(J):
            row[f'p1_good{j+1}'] = p.p1[j]
            row[f'posterior_H_given_good{j+1}'] = p.posterior[0, j]
        rows.append(row)
    mix = pd.DataFrame(rows)
    mix.to_csv(output / 'population_mix.csv', index=False)

    gamma_rows = []
    for g in np.unique(np.r_[np.linspace(0, gamma_max, points), types]):
        p = solve_common([g], [1.0])
        common_value = evaluate_policy(uniform.p1, uniform.p2, [g])[0]
        row = dict(gamma=g, targeted_profit=p.profit, uniform_50_50_profit=common_value,
                   gain_for_type=p.profit - common_value)
        for j in range(J):
            row[f'targeted_p1_good{j+1}'] = p.p1[j]
        gamma_rows.append(row)
    gamma_df = pd.DataFrame(gamma_rows)
    gamma_df.to_csv(output / 'profit_by_gamma.csv', index=False)
    policy_rows = []
    for label, p in [('uniform_50_50', uniform), ('targeted_H', personal[0]), ('targeted_L', personal[1])]:
        for period, history, prices in [(1, 'initial history pooled', p.p1)] + [
                (2, f'good{j+1}', p.p2[j]) for j in range(J)]:
            policy_rows.append(dict(policy=label, period=period, history=history,
                                    **{f'price_good{k+1}': prices[k] for k in range(J)}))
    pd.DataFrame(policy_rows).to_csv(output / 'prices_50_50.csv', index=False)

    newborn, newborn_cutoff = newborn_discount_analysis(output)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(mix.high_share, mix.common_profit, label='Common across types')
    axes[0].plot(mix.high_share, mix.targeted_profit, label='Observed type')
    axes[0].set(xlabel='Population share of high type', ylabel='Profit per initial consumer')
    axes[1].plot(mix.high_share, mix.targeting_gain, label='Gross gain from targeting')
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set(xlabel='Population share of high type', ylabel='Profit gain')
    axes[2].plot(gamma_df.gamma, gamma_df.targeted_profit, label='Prices optimized for gamma')
    axes[2].plot(gamma_df.gamma, gamma_df.uniform_50_50_profit, label='Fixed 50/50 common policy')
    axes[2].set(xlabel='Variety preference gamma', ylabel='Conditional expected profit')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / 'profits.png', dpi=180)
    fig.savefig(output / 'profits.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(newborn.probability_L, newborn.regular_profit, label='Regular H menu')
    axes[0].plot(newborn.probability_L, newborn.discount_profit, label='Discount L menu')
    axes[0].plot(newborn.probability_L, newborn.reoptimized_common_profit, label='Reoptimized common menu')
    axes[0].set(xlabel='Probability newborn is low type', ylabel='Profit per newborn')
    axes[1].plot(newborn.probability_L, newborn.discount_gain, label='Gain from fixed discount')
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set(xlabel='Probability newborn is low type', ylabel='Discount profit minus regular profit')
    if np.isfinite(newborn_cutoff):
        for ax in axes:
            ax.axvline(newborn_cutoff, color='gray', linestyle=':',
                       label=f'Cutoff {newborn_cutoff:.3f}')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'newborn_discount_cutoff.png', dpi=180)
    fig.savefig(output / 'newborn_discount_cutoff.pdf')
    plt.close(fig)
    report = ('# Toy model numerical results\n\n'
              'Uniform means common across types; prices vary by product and period-one purchase history.\n'
              'Consumers are myopic. Period-one outside choices exit. Initial history is pooled.\n'
              'Profits are per initial consumer, with N normalized to one.\n\n'
              f'50/50 common profit: {uniform.profit:.8f}\n\n'
              f'50/50 targeted profit: {mixture(0.5)[1]:.8f}\n\n'
              f'50/50 targeting gain: {mixture(0.5)[2]:.8f}\n\n'
              'These are incumbent-history benchmarks. Newborn consumers have no purchase history; '
              'their discount decision is reported separately in newborn_results.md. No fixed adoption cost '
              'or noisy classifier is assumed.\n\n'
              'profit_by_gamma.csv reports conditional expected profits, not a probability density: '
              'a density over profits requires specifying a distribution over gamma. The population experiment '
              'uses two point masses, H and L.\n\n'
              'Optimization uses three starts and checks solver status; this is numerical evidence, '
              'not a proof of global optimality. See prices_50_50.csv for all prices.\n')
    (output / 'results.md').write_text(report)
    print(report)
    print(pd.DataFrame(policy_rows).to_string(index=False))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='updates/toy_model/results')
    parser.add_argument('--points', type=int, default=21)
    parser.add_argument('--gamma-max', type=float, default=3.0)
    args = parser.parse_args()
    if args.points < 3 or args.gamma_max < 0:
        parser.error('Need points >= 3, gamma-max >= 0')
    run_analysis(args.output_dir, args.points, args.gamma_max)

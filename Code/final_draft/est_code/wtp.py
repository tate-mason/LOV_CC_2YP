"""Population WTP moments under independent gamma and lognormal -alpha."""
import numpy as np
import pandas as pd


def implied_wtp(estimates, habit_weight=.7):
    """Return population moments, not sampling standard errors, in price units."""
    p = dict(zip(estimates.parameter, estimates.estimate))
    mu, sigma = p['mu_log_minus_alpha'], p.get('sd_log_minus_alpha', 0.)
    inverse_mean = np.exp(-mu + sigma**2 / 2)
    inverse_second = np.exp(-2 * mu + 2 * sigma**2)
    rows = []

    def add(term, mean, second, interpretation):
        wtp_mean = mean * inverse_mean
        variance = max(0., second * inverse_second - wtp_mean**2)
        rows.append(dict(term=term, mean_wtp=wtp_mean, population_sd=np.sqrt(variance),
                         interpretation=interpretation))

    for name in ['inside_constant', 'beta_berry', 'beta_plain']:
        if name in p:
            interpretation = ('Other flavor vs outside, at zero history and control residual'
                              if name == 'inside_constant' else 'Flavor premium relative to other')
            add(name, p[name], p[name]**2, interpretation)
    if 'mean_gamma' in p:
        mean, sd = p['mean_gamma'], p.get('sd_gamma', 0.)
        second = mean**2 + sd**2
        if 'mixture_weight' in p:
            weight = p['mixture_weight']
            mean2 = mean + p['gamma_mean_gap']
            second2 = mean2**2 + p['sd_gamma_2']**2
            add('gamma_component_1', mean, second, 'Per unit Xi, conditional on component 1')
            add('gamma_component_2', mean2, second2, 'Per unit Xi, conditional on component 2')
            mean, second = weight * mean + (1 - weight) * mean2, weight * second + (1 - weight) * second2
        add('gamma', mean, second, 'Per unit increase in history index Xi')
        add('gamma_novelty', (1 - habit_weight) * mean, (1 - habit_weight)**2 * second,
            'Per unit novelty N, holding own-flavor stock fixed')
        add('gamma_own_stock', -habit_weight * mean, habit_weight**2 * second,
            'Per unit own-flavor stock S, holding novelty fixed')
    return pd.DataFrame(rows)

"""Separate estimation specifications; numerical functions perform no empirical I/O."""
import numpy as np
from scipy.special import logsumexp

SPECS = ('novelty_only', 'random_beta_common_alpha')


def alpha_moments(parameters):
    """Population moments of signed alpha, not of its positive magnitude."""
    mu = parameters['mu_log_minus_alpha']
    variance = parameters.get('sd_log_minus_alpha', 0.)**2
    mean = -np.exp(mu + variance/2)
    sd = np.exp(mu + variance/2)*np.sqrt(np.expm1(variance))
    return dict(mean_alpha=float(mean), population_sd_alpha=float(sd))


def names(spec, flavors, control=False):
    if spec not in SPECS:
        raise ValueError('Unknown experimental specification')
    result = ['inside_constant'] + ['beta_'+f for f in flavors[1:]]
    result += ['mu_log_minus_alpha', 'mean_novelty', 'sd_novelty']
    if spec == 'novelty_only':
        result += ['sd_log_minus_alpha']
    else:
        result += ['sd_beta_'+f for f in flavors]
        result += ['mean_own_stock', 'sd_own_stock']
    return result + (['price_control'] if control else [])


def draws(number, seed, spec, flavors):
    if number < 2 or number % 2:
        raise ValueError('Use an even draw count >=2')
    dimensions = 2 if spec == 'novelty_only' else len(flavors)+2
    half = np.random.default_rng(seed).standard_normal((number//2, dimensions))
    return np.vstack([half, -half])


def objective(theta, data, integration, spec, flavors):
    """Integrate persistent coefficients over complete household histories."""
    if not data['outside']:
        raise ValueError('An outside option is required to identify all flavor beta distributions')
    parameters = names(spec, flavors, data['control'])
    p = dict(zip(parameters, theta))
    mean_beta = p['inside_constant'] + np.array([0.] + [p['beta_'+f] for f in flavors[1:]])
    likelihoods, scores = [], []
    for z in integration:
        beta = mean_beta.copy()
        alpha = -np.exp(p['mu_log_minus_alpha'] + p.get('sd_log_minus_alpha', 0.)*z[1])
        novelty = p['mean_novelty'] + p['sd_novelty']*z[0]
        utility = beta + alpha*data['prices'] + novelty*data['novelty']
        if spec == 'random_beta_common_alpha':
            beta_draw = np.array([p['sd_beta_'+f] for f in flavors])*z[1:len(flavors)+1]
            utility += beta_draw
            utility += (p['mean_own_stock']+p['sd_own_stock']*z[-1])*data['stock']
        utility += p.get('price_control', 0.)*data['cf']
        utility = np.where(data['available'], utility, -np.inf)
        utility = np.column_stack([utility, np.zeros(len(utility))])
        logp = utility-logsumexp(utility, axis=1, keepdims=True)
        ll = (data['counts']*np.where(data['counts'] > 0, logp, 0.)).sum(axis=1)
        likelihoods.append(np.bincount(data['hh'], weights=ll, minlength=data['n_households']))
        residual = data['counts'][:, :len(flavors)]-data['counts'].sum(axis=1)[:, None]*np.exp(logp[:, :len(flavors)])
        price_score = alpha*(residual*data['prices']).sum(axis=1)
        novelty_score = (residual*data['novelty']).sum(axis=1)
        columns = dict(inside_constant=residual.sum(axis=1), mu_log_minus_alpha=price_score,
                       mean_novelty=novelty_score, sd_novelty=novelty_score*z[0],
                       sd_log_minus_alpha=price_score*z[1], price_control=(residual*data['cf']).sum(axis=1))
        columns.update({'beta_'+f:residual[:, j] for j,f in enumerate(flavors) if j})
        if spec == 'random_beta_common_alpha':
            columns.update({'sd_beta_'+f:residual[:, j]*z[j+1] for j,f in enumerate(flavors)})
            columns['mean_own_stock'] = (residual*data['stock']).sum(axis=1)
            columns['sd_own_stock'] = columns['mean_own_stock']*z[-1]
        scores.append(np.column_stack([np.bincount(data['hh'], weights=columns[n], minlength=data['n_households']) for n in parameters]))
    likelihoods = np.asarray(likelihoods)
    normalizer = logsumexp(likelihoods, axis=0)
    weights = np.exp(likelihoods-normalizer)
    score = (weights[:, :, None]*np.asarray(scores)).sum(axis=0)
    return -float((normalizer-np.log(len(integration))).mean()), -score.mean(axis=0)

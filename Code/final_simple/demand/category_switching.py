"""Pure MNL kernels: yogurt-category switching, flavor tastes, no habit.

No empirical I/O. An observation is one shopping week, including no yogurt.
Types persist across each household's whole history. Other is the reference flavor.
"""
import numpy as np

PARAMETERS = ['inside_constant', 'beta_berry', 'beta_plain',
              'mu_log_minus_alpha', 'sd_log_minus_alpha', 'mean_novelty', 'sd_novelty']
FLAVORS = ['other', 'berry', 'plain']


def logsumexp(x, axis=None, keepdims=False):
    maximum = np.max(x, axis=axis, keepdims=True)
    result = maximum+np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


def draws(number, seed):
    if number < 2 or number % 2:
        raise ValueError('Need an even integration draw count >=2.')
    half = np.random.default_rng(seed).standard_normal((number//2, 2))
    return np.vstack([half, -half])


def objective(theta, data, integration, household=False):
    """Mean negative household log likelihood and analytic score.

    One Bernoulli outcome per observed shopping week: any yogurt or no yogurt.
    Flavor utilities enter only through their inclusive value. Neither package
    quantity nor which flavor was purchased defines the binary outcome or history.
    """
    b0, berry, plain, mu_a, sd_a, mu_g, sd_g = theta
    beta = b0+np.array([0., berry, plain])
    hh, n = data['hh'], data['n_households']
    sign = data['history_sign']
    likelihoods, gradients = [], []
    for za, zg in integration:
        alpha = -np.exp(mu_a+sd_a*za)
        gamma = mu_g+sd_g*zg
        utility = beta+alpha*data['prices']+gamma*sign[:, None]
        utility = np.where(data['available'], utility, -np.inf)
        all_utility = np.column_stack([utility, np.zeros(len(hh))])
        logp = all_utility-logsumexp(all_utility, axis=1, keepdims=True)
        purchase = data['purchase']
        inside_lse = logsumexp(utility, axis=1, keepdims=True)
        log_category = inside_lse[:, 0]-logsumexp(all_utility, axis=1)
        ll = purchase*log_category+(1-purchase)*logp[:,3]
        likelihoods.append(np.bincount(hh, weights=ll, minlength=n))
        conditional = np.exp(utility-inside_lse)
        residual = purchase[:,None]*conditional-np.exp(logp[:,:3])
        intercept_score = residual.sum(axis=1)
        price_score = alpha*(residual*data['prices']).sum(axis=1)
        gamma_score = sign*intercept_score
        scores = [intercept_score, residual[:, 1], residual[:, 2],
                  price_score, price_score*za, gamma_score, gamma_score*zg]
        gradients.append(np.column_stack([
            np.bincount(hh, weights=score, minlength=n) for score in scores]))
    likelihoods = np.asarray(likelihoods)
    normalizer = logsumexp(likelihoods, axis=0)
    posterior = np.exp(likelihoods-normalizer)
    score = (posterior[:, :, None]*np.asarray(gradients)).sum(axis=0)
    loglike = normalizer-np.log(len(integration))
    if household:
        return loglike, score
    return -float(loglike.mean()), -score.mean(axis=0)


def pooled_prediction(theta, data, integration, return_participation=False):
    """Aggregate fit checks only; caller aggregates before persisting anything."""
    b0, berry, plain, mu_a, sd_a, mu_g, sd_g = theta
    beta = b0+np.array([0., berry, plain])
    probability = np.zeros((len(data['hh']), 4))
    participation = np.zeros(len(data['hh']))
    for za, zg in integration:
        u = beta-np.exp(mu_a+sd_a*za)*data['prices']
        u += (mu_g+sd_g*zg)*data['history_sign'][:, None]
        u = np.where(data['available'], u, -np.inf)
        u = np.column_stack([u, np.zeros(len(u))])
        current = np.exp(u-logsumexp(u, axis=1, keepdims=True))
        probability += current/len(integration)
        participation += current[:,:3].sum(axis=1)/len(integration)
    return (probability, participation) if return_participation else probability

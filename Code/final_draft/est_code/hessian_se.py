"""Observed-information inference, conditional on first-stage fits."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def observed_covariance(theta, gradient, households, bounds):
    """Differentiate the mean negative-log-likelihood gradient, then scale by I."""
    theta = np.asarray(theta, dtype=float)
    steps = np.cbrt(np.finfo(float).eps) * np.maximum(1., np.abs(theta))
    lower, upper = np.asarray(bounds, dtype=float).T
    if np.any(np.minimum(theta - lower, upper - theta) <= np.maximum(steps, 1e-4)):
        raise ValueError("parameter at or near bounds; normal inference withheld")
    hessian = np.empty((len(theta), len(theta)))
    for j, step in enumerate(steps):
        delta = np.zeros(len(theta))
        delta[j] = step
        hessian[:, j] = (gradient(theta + delta) - gradient(theta - delta)) / (2 * step)
    hessian = (hessian + hessian.T) / 2
    if not np.isfinite(hessian).all():
        raise ValueError("nonfinite observed information; SEs withheld")
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] <= max(0., eigenvalues[-1] * 1e-10):
        raise ValueError("observed information not positive definite or ill-conditioned; SEs withheld")
    return np.linalg.solve(hessian, np.eye(len(theta))) / households


def hessian_standard_errors(theta, gradient, households, bounds, names, output):
    report = dict(method="observed-information Hessian",
                  conditional_on="estimated price controls, retail instruments, and fixed integration draws",
                  first_stage_uncertainty="not included", robust=False)
    try:
        covariance = observed_covariance(theta, gradient, households, bounds)
    except ValueError as error:
        report["status"] = str(error)
        return None, report
    pd.DataFrame(covariance, index=names, columns=names).to_csv(output / "covariance.csv")
    se = np.sqrt(np.diag(covariance))
    critical = norm.ppf(.975)
    table = pd.DataFrame(dict(parameter=names, std_error=se,
                              ci_lower=theta - critical * se, ci_upper=theta + critical * se))
    report["status"] = "computed; normal 95% intervals; first-stage uncertainty omitted"
    return table, report

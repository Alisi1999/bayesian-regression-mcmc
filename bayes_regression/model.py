"""Gaussian Bayesian linear regression posterior used in the group project."""
import numpy as np
from scipy.special import gammaln


def log_inv_gamma_pdf(sigma2, a0, b0):
    if sigma2 <= 0:
        return -np.inf
    return (a0 * np.log(b0) - gammaln(a0)
            - (a0 + 1.0) * np.log(sigma2) - b0 / sigma2)


def log_posterior(beta, log_sigma2, X, y, tau2=10.0, a0=2.0, b0=1.0):
    beta, X, y = np.asarray(beta), np.asarray(X), np.asarray(y)
    n, p = X.shape
    sigma2 = np.exp(log_sigma2)
    if sigma2 <= 1e-12 or sigma2 > 1e6:
        return -np.inf
    resid = y - X @ beta
    likelihood = (-0.5 * n * np.log(2.0 * np.pi * sigma2)
                  - 0.5 * np.sum(resid ** 2) / sigma2)
    prior_beta = (-0.5 * p * np.log(2.0 * np.pi * tau2)
                  - 0.5 * np.sum(beta ** 2) / tau2)
    return (likelihood + prior_beta
            + log_inv_gamma_pdf(sigma2, a0, b0)
            + log_sigma2)  # Jacobian for sigma2 = exp(log_sigma2)

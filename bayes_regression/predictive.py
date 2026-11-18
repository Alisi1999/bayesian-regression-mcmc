"""Posterior predictive samples for linear regression."""
import numpy as np


def posterior_predictive_samples(beta_samples, log_sigma2_samples, X_new, n_draws, seed=0):
    rng = np.random.default_rng(seed)
    beta_samples = np.asarray(beta_samples)
    log_sigma2_samples = np.asarray(log_sigma2_samples)
    X_new = np.asarray(X_new)
    if n_draws > len(beta_samples):
        raise ValueError("n_draws exceeds available posterior samples")
    idx = rng.choice(len(beta_samples), size=n_draws, replace=False)
    beta_draws = beta_samples[idx]
    sigma2_draws = np.exp(log_sigma2_samples[idx])
    mean_part = beta_draws @ X_new.T
    noise = rng.normal(0, np.sqrt(sigma2_draws)[:, None], mean_part.shape)
    return mean_part + noise

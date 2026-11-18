"""Small smoke tests, requiring no network connection."""
import numpy as np
from bayes_regression import run_mh_chain, autocorr, posterior_predictive_samples


def test_mcmc_chain_shapes():
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(40), rng.normal(size=40)])
    y = X @ np.array([1., .5]) + rng.normal(0, .3, 40)
    chain = run_mh_chain(X, y, n_iter=40, burn=10, seed=2)
    assert chain["betas_post"].shape == (30, 2)
    assert chain["log_sigma2_post"].shape == (30,)
    assert 0 <= chain["accept_rate_beta"] <= 1


def test_short_chain_autocorrelation_is_bounded():
    assert len(autocorr(np.arange(8), max_lag=300)) == 8


def test_posterior_predictions():
    beta = np.zeros((30, 2))
    log_var = np.zeros(30)
    X = np.ones((4, 2))
    preds = posterior_predictive_samples(beta, log_var, X, n_draws=10)
    assert preds.shape == (10, 4)

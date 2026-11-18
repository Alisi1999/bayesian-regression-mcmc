"""Component-wise random-walk Metropolis-Hastings sampler from group coursework."""
import numpy as np
from .model import log_posterior


def run_mh_chain(
    X, y, n_iter=12000, burn=4000, beta_step=0.02,
    log_sigma2_step=0.05, tau2=10.0, a0=2.0, b0=1.0,
    init_beta=None, init_log_sigma2=None, seed=0
):
    X, y = np.asarray(X), np.asarray(y)
    _, p = X.shape
    if not 0 <= burn < n_iter:
        raise ValueError("burn must be in [0, n_iter)")
    rng = np.random.default_rng(seed)
    beta = np.zeros(p) if init_beta is None else np.asarray(init_beta).copy()
    log_sigma2 = 0.0 if init_log_sigma2 is None else float(init_log_sigma2)
    betas = np.zeros((n_iter, p))
    log_sigma2s = np.zeros(n_iter)
    current_lp = log_posterior(beta, log_sigma2, X, y, tau2=tau2, a0=a0, b0=b0)
    if not np.isfinite(current_lp):
        raise RuntimeError("Initial log-posterior is not finite")
    accepted_beta, accepted_variance = 0, 0

    for t in range(n_iter):
        for j in range(p):
            proposal = beta.copy()
            proposal[j] += rng.normal(0.0, beta_step)
            proposal_lp = log_posterior(
                proposal, log_sigma2, X, y, tau2=tau2, a0=a0, b0=b0
            )
            if np.log(rng.uniform()) < proposal_lp - current_lp:
                beta, current_lp = proposal, proposal_lp
                accepted_beta += 1

        variance_proposal = log_sigma2 + rng.normal(0, log_sigma2_step)
        proposal_lp = log_posterior(
            beta, variance_proposal, X, y, tau2=tau2, a0=a0, b0=b0
        )
        if np.log(rng.uniform()) < proposal_lp - current_lp:
            log_sigma2, current_lp = variance_proposal, proposal_lp
            accepted_variance += 1
        betas[t] = beta
        log_sigma2s[t] = log_sigma2

    return {
        "betas": betas,
        "log_sigma2": log_sigma2s,
        "betas_post": betas[burn:],
        "log_sigma2_post": log_sigma2s[burn:],
        "accept_rate_beta": accepted_beta / (n_iter * p),
        "accept_rate_sigma2": accepted_variance / n_iter,
        "burn": burn, "n_iter": n_iter,
    }

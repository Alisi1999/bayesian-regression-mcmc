"""Fast, offline example using synthetic data (not the original housing experiment)."""
import numpy as np
from bayes_regression import (
    run_mh_chain, build_diagnostics_table, posterior_predictive_samples
)


def main():
    rng = np.random.default_rng(11)
    X = np.column_stack([np.ones(80), rng.normal(size=(80, 2))])
    y = X @ np.array([1.0, 0.3, -0.2]) + rng.normal(0, 0.4, 80)
    chains = [run_mh_chain(
        X, y, n_iter=6000, burn=1000,
        beta_step=.08, log_sigma2_step=.25, seed=i,
    ) for i in range(4)]
    diagnostics = build_diagnostics_table(chains, ["Intercept", "x1", "x2"])
    print(diagnostics.to_string(index=False))
    beta = np.vstack([ch["betas_post"] for ch in chains])
    variance = np.concatenate([ch["log_sigma2_post"] for ch in chains])
    preds = posterior_predictive_samples(beta, variance, X[:5], n_draws=50)
    print("Posterior predictive samples shape:", preds.shape)
    print("These diagnostics summarize the synthetic-data run. The California Housing experiment has separate diagnostics.")


if __name__ == "__main__":
    main()

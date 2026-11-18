"""Bayesian regression study using California Housing.

Default: original-scale 4-chain experiment plus proposal-scale sensitivity runs.
Use --quick for a small demonstration (not converged research results).
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayes_regression import (
    load_and_prepare_data, fit_ols_baseline, run_mh_chain,
    build_diagnostics_table, plot_traces_for_all_params,
    plot_running_means_for_all_params, plot_acf_for_params,
    posterior_predictive_samples
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small illustrative run; not for research results")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    X_train, X_test, y_train, y_test, param_names = load_and_prepare_data()
    ols = fit_ols_baseline(X_train, y_train)

    iterations, burn = (250, 50) if args.quick else (12000, 4000)
    steps = [.02] if args.quick else [.005, .01, .02, .05]
    n_chains = 2 if args.quick else 4
    output = Path(__file__).resolve().parent / "results"
    plots, tables = output / "plots", output / "tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    def sample_chain_group(beta_step, base_seed):
        start = time.perf_counter()
        chains = [
            run_mh_chain(X_train, y_train, n_iter=iterations, burn=burn,
                         beta_step=beta_step, log_sigma2_step=.05,
                         tau2=10., a0=2., b0=1., seed=base_seed + i)
            for i in range(n_chains)
        ]
        return chains, time.perf_counter() - start

    chains, elapsed = sample_chain_group(.02, args.seed)
    print(f"Main sampling: {elapsed:.2f} seconds")
    diagnostics = build_diagnostics_table(chains, param_names)
    diagnostics.to_csv(tables / "diagnostics.csv", index=False)
    plot_traces_for_all_params(chains, param_names, plots)
    plot_running_means_for_all_params(chains, param_names, plots)
    plot_acf_for_params(chains[0], param_names, [0, 1, 2], 60, plots / "acf_selected_params.png")

    sensitivity = []
    for step in steps:
        sens, duration = sample_chain_group(step, args.seed + 10)
        diag = build_diagnostics_table(sens, param_names)
        for _, row in diag.iterrows():
            sensitivity.append({
                "beta_step": step, "param": row["param"],
                "Rhat": row["Rhat"], "ESS_mean": row["ESS (mean over chains)"],
                "ESS_per_second": row["ESS (mean over chains)"] / duration,
            })
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(tables / "sensitivity.csv", index=False)

    fig, ax = plt.subplots()
    for param, group in sensitivity.groupby("param"):
        ax.plot(group["beta_step"], group["ESS_per_second"], marker="o", label=param)
    ax.set(xlabel="beta_step", ylabel="ESS per second", title="Sampler efficiency across proposal scales")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "ess_per_second_vs_beta_step.png")
    plt.close(fig)

    betas = np.vstack([c["betas_post"] for c in chains])
    log_vars = np.concatenate([c["log_sigma2_post"] for c in chains])
    summary = []
    for j, name in enumerate(param_names):
        samples = betas[:, j]
        summary.append({
            "param": name, "post_mean": samples.mean(),
            "post_median": np.median(samples),
            "CI_2.5%": np.quantile(samples, .025),
            "CI_97.5%": np.quantile(samples, .975),
            "P(beta>0)": np.mean(samples > 0)
        })
    pd.DataFrame(summary).to_csv(tables / "posterior_summary.csv", index=False)
    pd.DataFrame({"param": param_names, "OLS_estimate": ols.params}).to_csv(
        tables / "ols_estimates.csv", index=False)

    n_draws = min(2000, len(betas))
    draws = posterior_predictive_samples(
        betas, log_vars, X_test, n_draws=n_draws, seed=args.seed + 999
    )
    prediction = draws.mean(axis=0)
    pd.DataFrame({
        "y_true": y_test, "pred_mean": prediction,
        "pred_2.5": np.quantile(draws, .025, axis=0),
        "pred_97.5": np.quantile(draws, .975, axis=0),
    }).to_csv(tables / "posterior_predictive_test.csv", index=False)
    rmse = np.sqrt(np.mean((prediction - y_test) ** 2))
    print(f"Held-out predictive RMSE: {rmse:.4f}")
    if args.quick:
        print("QUICK MODE ONLY: convergence and predictive performance are not research-grade.")
    print("Output written to:", output)


if __name__ == "__main__":
    main()

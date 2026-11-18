"""Trace, running-mean and autocorrelation plots for MCMC chains."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .diagnostics import autocorr


def _series(ch, j, p):
    return ch["betas_post"][:, j] if j < p else ch["log_sigma2_post"]


def plot_traces_for_all_params(chains, param_names, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(list(param_names) + ["log_sigma2"]):
        fig, ax = plt.subplots(figsize=(8, 3))
        for i, chain in enumerate(chains):
            ax.plot(_series(chain, j, len(param_names)), alpha=.7, lw=.7, label=f"Chain {i+1}")
        ax.set(title=f"Trace — {name}", xlabel="Post-burn iteration", ylabel="Parameter")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"trace_{name}.png", dpi=150)
        plt.close(fig)


def plot_running_means_for_all_params(chains, param_names, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(list(param_names) + ["log_sigma2"]):
        fig, ax = plt.subplots(figsize=(8, 3))
        for i, ch in enumerate(chains):
            x = _series(ch, j, len(param_names))
            avg = np.cumsum(x) / np.arange(1, len(x)+1)
            ax.plot(avg, alpha=.8, label=f"Chain {i+1}")
        ax.set(title=f"Running mean — {name}", xlabel="Post-burn iteration")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"running_mean_{name}.png", dpi=150)
        plt.close(fig)


def plot_acf_for_params(chain, param_names, idx_list, max_lag, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j in idx_list:
        acf = autocorr(chain["betas_post"][:, j], max_lag)
        ax.plot(range(len(acf)), acf, label=param_names[j])
    ax.set(title="Autocorrelation of selected parameters", xlabel="Lag", ylabel="ACF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

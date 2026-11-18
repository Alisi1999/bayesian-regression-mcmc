"""MCMC convergence diagnostics. Refactored from collaborative coursework."""
import numpy as np
import pandas as pd


def autocorr(x, max_lag=100):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    max_lag = min(max_lag, len(x) - 1)
    if denom < 1e-12:
        return np.full(max_lag + 1, np.nan)
    acf = [1.]
    for lag in range(1, max_lag + 1):
        acf.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(acf)


def ess_from_acf(x, max_lag=200):
    x = np.asarray(x)
    if len(x) < 2 or np.var(x) < 1e-12:
        return np.nan
    acf = autocorr(x, max_lag=max_lag)
    acc = 0.0
    for rho in acf[1:]:
        if np.isnan(rho) or rho < 0:
            break
        acc += rho
    return len(x) / (1 + 2 * acc)


def split_rhat(chains_1d):
    parts = []
    for chain in chains_1d:
        x = np.asarray(chain)
        half = len(x) // 2
        parts.extend([x[:half], x[half:2*half]])
    n = min(map(len, parts))
    if n < 2:
        return np.nan
    parts = [part[:n] for part in parts]
    W = np.mean([part.var(ddof=1) for part in parts])
    B = n * np.var([part.mean() for part in parts], ddof=1)
    return float(np.sqrt(((n-1) / n * W + B / n) / W)) if W > 0 else np.nan


def geweke_z(x, first=0.1, last=0.5):
    x = np.asarray(x)
    n_first, n_last = int(first * len(x)), int(last * len(x))
    if n_first < 2 or n_last < 2:
        return np.nan
    a, b = x[:n_first], x[-n_last:]
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va < 1e-12 or vb < 1e-12:
        return np.nan
    return float((a.mean()-b.mean()) / np.sqrt(va / len(a) + vb / len(b)))


def build_diagnostics_table(chains, param_names):
    rows = []
    def add_row(param, samples):
        ess = [ess_from_acf(x, max_lag=300) for x in samples]
        gew = [geweke_z(x) for x in samples]
        rows.append({
            "param": param,
            "Rhat": split_rhat(samples),
            "ESS (min over chains)": np.nanmin(ess),
            "ESS (mean over chains)": np.nanmean(ess),
            "Geweke z (mean)": np.nanmean(gew),
            "Geweke z (max abs)": np.nanmax(np.abs(gew)),
        })
    for j, name in enumerate(param_names):
        add_row(name, [ch["betas_post"][:, j] for ch in chains])
    add_row("log(sigma^2)", [ch["log_sigma2_post"] for ch in chains])
    return pd.DataFrame(rows)

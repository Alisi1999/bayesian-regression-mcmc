# Bayesian Linear Regression with MCMC

A **collaborative Master's project** completed for *On the Theory and Practice of Monte Carlo Simulations* at TU Dortmund University.

We studied **convergence, mixing, and sampling efficiency** of a component-wise **Random-Walk Metropolis–Hastings (MH)** sampler for Bayesian linear regression on the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).

> **Project status:** This repository is being prepared from the original group submission. It is **private** while we review the shared code and publication permissions. Saved outcomes are historical project results; the complete experiment has not been independently rerun for this repository.

## Research questions

- Do independently initialized MCMC chains converge to consistent posterior distributions?
- How do proposal scales affect acceptance, autocorrelation, effective sample size (ESS), and efficiency?
- How well does Bayesian posterior prediction perform on held-out housing data?

## Method

- **Data:** California Housing, 20,640 observations; five predictors (median income, house age, average rooms, average bedrooms, population).
- **Preprocessing:** 80/20 train-test split and feature standardization fitted on the training set.
- **Model:** Bayesian linear regression, Gaussian coefficient priors, inverse-gamma noise variance prior.
- **Inference:** Custom component-wise Metropolis–Hastings sampler; regression coefficients and log-noise variance updated with random-walk proposals.
- **Evaluation:** Four MCMC chains, burn-in, trace plots, autocorrelation, ESS, split-R-hat, Geweke statistics, and proposal-scale sensitivity analysis.
- **Prediction:** Posterior predictive intervals and an ordinary least squares (OLS) comparison.

## Results from the original report

The original submitted report records **RMSE ≈ 0.8021** on the held-out test set (housing targets in units of $100,000). We assessed posterior convergence and tuning efficiency using multiple diagnostic measures.

**These results have not been regenerated using a new full run of the cleaned repository.** They are reported for project context, not as a fresh benchmark.

## Repository organization

- bayes_regression/ — Data preparation, probabilistic model, MCMC sampler, diagnostics, predictive routines, plotting.
- run_analysis.py — Refactored driver for the full study design (computationally demanding); not identical byte-for-byte to the original script.
- quickstart.py — Small synthetic-data smoke test without external downloads.
- requirements.txt — Required Python libraries.
- tests/ — Lightweight tests of the core routines.

The original report, student IDs, downloaded dataset CSV, and bulk generated plots have been intentionally omitted. The dataset is loaded using scikit-learn when running the full analysis.

## Getting started

Install Python 3.10+ and create a virtual environment:

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Try the small synthetic-data example:

~~~bash
python quickstart.py
~~~

For the full original analysis:

~~~bash
python run_analysis.py

# Smaller California Housing demonstration (not suitable for convergence claims)
python run_analysis.py --quick
~~~

**Warning:** The full analysis runs multiple chains for 12,000 iterations, plus several additional sensitivity runs. It can be slow; it also needs internet access on the first use of the California Housing dataset. The original script used a time-dependent random seed; the refactored driver now accepts --seed (default 123) for more repeatable experiments.

## Contributions and attribution

The **MCMC implementation and diagnostic code were developed jointly** by Seyed Ali Poorseyed and a project teammate. Seyed Ali Poorseyed independently carried out and interpreted the empirical analysis documented in his final report.

This is a portfolio presentation of collaborative academic work, not a claim of sole authorship. The repository source has been cleaned and partially refactored, including a short-chain autocorrelation fix. The original submission is preserved separately; full experiments and archived report numbers have not yet been reproduced with this version. Lightweight core checks passed on the original local copy, but the exact published files have not yet undergone a complete end-to-end rerun. No open-source license has been added pending publication review.

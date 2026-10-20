"""Estimate expected returns and covariance from monthly returns.

All outputs are ANNUALIZED: mean x 12, covariance x 12. Annualizing the
covariance by 12 assumes months are independent (no autocorrelation). That is
a standard simplification and is listed in DECISIONS.md honest limits.

See DECISIONS.md D-007 (covariance) and D-008 (expected returns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

PERIODS_PER_YEAR = 12


# ---------------------------------------------------------------- covariance


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Plain sample covariance (divides by T-1), annualized."""
    return returns.cov() * PERIODS_PER_YEAR


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """sklearn Ledoit-Wolf on RAW returns. Target is a scaled identity matrix.

    Kept for comparison only. With assets whose volatilities differ by 10x
    (short Treasuries vs emerging equity), shrinking toward one common
    variance inflates the low-vol assets' variance and deflates the high-vol
    ones. See DECISIONS.md P-003.
    """
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    return cov * PERIODS_PER_YEAR, float(lw.shrinkage_)


def ledoit_wolf_correlation_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf applied to the CORRELATION matrix, then rescaled. DEFAULT.

    Steps: standardize each asset to unit variance, run Ledoit-Wolf (which now
    shrinks correlations toward zero, since every variance is 1), then scale
    back by each asset's sample volatility. Each asset keeps its own sample
    volatility exactly; only the noisy correlations are pulled in.
    """
    vols = returns.std(ddof=1)
    z = (returns - returns.mean()) / vols
    lw = LedoitWolf(assume_centered=True).fit(z.values)
    corr = lw.covariance_
    # LedoitWolf divides by T, the sample std above by T-1. Re-normalize the
    # diagonal to exactly 1 so it is a true correlation matrix.
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    cov = corr * np.outer(vols.values, vols.values)
    cov_df = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    return cov_df * PERIODS_PER_YEAR, float(lw.shrinkage_)


def estimate_covariance(returns: pd.DataFrame, method: str = "lw_corr") -> pd.DataFrame:
    """method: "sample", "lw" (raw Ledoit-Wolf), or "lw_corr" (default)."""
    if method == "sample":
        cov = sample_covariance(returns)
    elif method == "lw":
        cov, _ = ledoit_wolf_covariance(returns)
    elif method == "lw_corr":
        cov, _ = ledoit_wolf_correlation_covariance(returns)
    else:
        raise ValueError(f"unknown covariance method {method!r}")
    check_covariance(cov)
    return cov


@dataclass
class CovarianceCheck:
    symmetric: bool
    min_eigenvalue: float
    psd: bool
    condition_number: float


def check_covariance(cov: pd.DataFrame | np.ndarray, tol: float = 1e-10) -> CovarianceCheck:
    """Raise if the matrix is not symmetric or not positive semi-definite.

    PSD means no portfolio can have negative variance. A non-PSD matrix lets
    an optimizer "find" a negative-risk portfolio, which is nonsense.
    """
    m = np.asarray(cov, dtype=float)
    symmetric = bool(np.allclose(m, m.T, atol=tol))
    eig = np.linalg.eigvalsh((m + m.T) / 2)
    min_eig = float(eig.min())
    psd = min_eig >= -tol * max(1.0, float(eig.max()))
    cond = float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")
    if not symmetric:
        raise ValueError("covariance matrix is not symmetric")
    if not psd:
        raise ValueError(f"covariance matrix is not PSD (min eigenvalue {min_eig:.3e})")
    return CovarianceCheck(symmetric, min_eig, psd, cond)


# ---------------------------------------------------------- expected returns


def historical_mean(returns: pd.DataFrame) -> pd.Series:
    """Arithmetic mean monthly return, annualized. Very noisy; comparison only."""
    return returns.mean() * PERIODS_PER_YEAR


@dataclass
class BayesSteinResult:
    mu: pd.Series  # annualized shrunk expected returns
    shrinkage: float  # phi in [0, 1]; 0 = pure history, 1 = all assets equal target
    target: float  # annualized target return (mean of the min-variance portfolio)
    historical: pd.Series  # annualized raw means, for comparison


def bayes_stein_mean(returns: pd.DataFrame) -> BayesSteinResult:
    """Jorion (1986) Bayes-Stein shrinkage of expected returns. DEFAULT.

    Plain language: take each asset's historical average, and pull it part of
    the way toward one common number. The common number is the return of the
    minimum-variance portfolio, because that is the portfolio whose return we
    can estimate most precisely. How far we pull (phi) is set by the data:
    the noisier the history and the more similar the assets look, the harder
    we pull.

        mu0 = (1' S^-1 mu) / (1' S^-1 1)                (min-variance mean)
        phi = (N + 2) / ((N + 2) + T (mu - mu0)' S^-1 (mu - mu0))
        mu_BS = (1 - phi) mu + phi mu0

    Computed on monthly data, then annualized. S uses Jorion's unbiased
    scaling (T - 1) / (T - N - 2) on the sample covariance.
    """
    x = returns.values
    t, n = x.shape
    if t <= n + 2:
        raise ValueError("need more observations than assets + 2 for Bayes-Stein")
    mu = x.mean(axis=0)
    s = np.cov(x, rowvar=False, ddof=1) * (t - 1) / (t - n - 2)
    s_inv = np.linalg.inv(s)
    ones = np.ones(n)
    mu0 = float(ones @ s_inv @ mu / (ones @ s_inv @ ones))
    diff = mu - mu0
    phi = float((n + 2) / ((n + 2) + t * diff @ s_inv @ diff))
    mu_bs = (1 - phi) * mu + phi * mu0
    cols = returns.columns
    return BayesSteinResult(
        mu=pd.Series(mu_bs * PERIODS_PER_YEAR, index=cols),
        shrinkage=phi,
        target=mu0 * PERIODS_PER_YEAR,
        historical=pd.Series(mu * PERIODS_PER_YEAR, index=cols),
    )


def estimate_expected_returns(returns: pd.DataFrame, method: str = "bayes_stein") -> pd.Series:
    """method: "historical" or "bayes_stein" (default)."""
    if method == "historical":
        return historical_mean(returns)
    if method == "bayes_stein":
        return bayes_stein_mean(returns).mu
    raise ValueError(f"unknown expected return method {method!r}")

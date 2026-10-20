"""Risk contribution: how much of portfolio volatility each position causes.

For volatility sigma = sqrt(w' Sigma w):

    marginal contribution  MCR_i = (Sigma w)_i / sigma
        "if I add a little more of asset i, how much does volatility rise"
    contribution           CR_i  = w_i * MCR_i
    percent contribution   PCR_i = CR_i / sigma

Because volatility is homogeneous of degree 1 in the weights (double every
weight and volatility doubles), Euler's theorem gives sum_i CR_i = sigma
exactly. The test suite checks this.

On a surplus basis the liability is treated as one more position with weight
-f, so its row shows how much surplus risk comes from the liability side.
See DECISIONS.md D-011.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.optimizer import LiabilityModel

LIABILITY_LABEL = "Liabilities"


def _augment(
    weights: pd.Series, cov: pd.DataFrame, liabilities: LiabilityModel | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    w = weights.reindex(cov.index).values.astype(float)
    s = cov.values.astype(float)
    names = list(cov.index)
    if liabilities is None or liabilities.liability_to_asset_ratio == 0:
        return w, s, names
    c = np.asarray(liabilities.cov_with_assets, dtype=float)
    n = len(w)
    big = np.zeros((n + 1, n + 1))
    big[:n, :n] = s
    big[:n, n] = big[n, :n] = c
    big[n, n] = liabilities.volatility**2
    return np.append(w, -liabilities.liability_to_asset_ratio), big, [*names, LIABILITY_LABEL]


def risk_contributions(
    weights: pd.Series, cov: pd.DataFrame, liabilities: LiabilityModel | None = None
) -> pd.DataFrame:
    """Table with weight, marginal, absolute and percent contribution per position."""
    w, s, names = _augment(weights, cov, liabilities)
    var = float(w @ s @ w)
    if var <= 0:
        raise ValueError("portfolio variance is zero; contributions undefined")
    vol = np.sqrt(var)
    mcr = s @ w / vol
    cr = w * mcr
    return pd.DataFrame(
        {"weight": w, "marginal": mcr, "contribution": cr, "pct_contribution": cr / vol},
        index=pd.Index(names, name="position"),
    )


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame, liabilities: LiabilityModel | None = None) -> float:
    w, s, _ = _augment(weights, cov, liabilities)
    return float(np.sqrt(w @ s @ w))

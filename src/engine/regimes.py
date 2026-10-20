"""Rules-based market regimes, using only information available at the time.

Two independent yes/no classifications (DECISIONS.md D-026, D-027):

- Rates regime: "rising" if the 10-year Treasury yield is higher than it was
  12 months earlier, else "falling".
- Volatility regime: "high" if SPY's realized volatility over the last 63
  trading days (about 3 months), annualized, is above 20%, else "low".

No look-ahead: the label for month t is computed from data up to the END of
month t-1, so it is known before month t starts. A strategy could actually
have used it. The test suite checks this by changing future data and
confirming past labels do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.estimation import PERIODS_PER_YEAR

RATE_LOOKBACK_MONTHS = 12
RATE_THRESHOLD_PCT_POINTS = 0.0
VOL_WINDOW_DAYS = 63
VOL_THRESHOLD = 0.20
TRADING_DAYS = 252


def _known_before(signal_at_month_end: pd.Series, months: pd.DatetimeIndex) -> pd.Series:
    """Shift a month-end signal forward one month: month t gets the value from t-1."""
    # Shift by one CALENDAR month, not one row: with data ending in month X,
    # the label for month X+1 must still exist (P-009).
    shifted = signal_at_month_end.shift(1, freq="ME")
    return shifted.reindex(months)


def rate_regime(signals: pd.DataFrame, months: pd.DatetimeIndex) -> pd.Series:
    y = signals["^TNX"].dropna().resample("ME").last()
    change = y - y.shift(RATE_LOOKBACK_MONTHS)
    label = pd.Series(np.where(change > RATE_THRESHOLD_PCT_POINTS, "rising", "falling"), index=y.index)
    label[change.isna()] = None
    out = _known_before(label, months)
    if out.isna().any():
        raise ValueError(f"not enough yield history to label {int(out.isna().sum())} months")
    return out.rename("rates")


def realized_vol(signals: pd.DataFrame) -> pd.Series:
    """Trailing 63-day SPY volatility, annualized, sampled at each month end."""
    r = signals["SPY"].dropna().pct_change()
    vol = r.rolling(VOL_WINDOW_DAYS, min_periods=VOL_WINDOW_DAYS).std() * np.sqrt(TRADING_DAYS)
    return vol.resample("ME").last()


def vol_regime(signals: pd.DataFrame, months: pd.DatetimeIndex) -> pd.Series:
    vol = realized_vol(signals)
    label = pd.Series(np.where(vol > VOL_THRESHOLD, "high", "low"), index=vol.index)
    label[vol.isna()] = None
    out = _known_before(label, months)
    if out.isna().any():
        raise ValueError(f"not enough SPY history to label {int(out.isna().sum())} months")
    return out.rename("volatility")


def label_regimes(signals: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.concat([rate_regime(signals, months), vol_regime(signals, months)], axis=1)


def _avg_correlation(r: pd.DataFrame) -> float:
    c = r.corr().values
    return float(c[~np.eye(len(c), dtype=bool)].mean())


def regime_summary(
    returns: pd.DataFrame, labels: pd.Series, portfolios: dict[str, pd.Series], rf: pd.Series
) -> dict[str, pd.DataFrame]:
    """Per regime: months, asset mean/vol, average correlation, portfolio stats.

    Annualized: mean x 12, vol x sqrt(12). Sharpe uses the T-bill (BIL) return
    earned in the same months.
    """
    assets, summary, ports = {}, [], []
    for reg in sorted(labels.unique()):
        idx = labels.index[labels == reg]
        r = returns.loc[idx]
        assets[reg] = pd.DataFrame(
            {
                "mean": r.mean() * PERIODS_PER_YEAR,
                "vol": r.std() * np.sqrt(PERIODS_PER_YEAR),
            }
        )
        summary.append({"regime": reg, "months": len(idx), "avg_correlation": _avg_correlation(r)})
        rf_reg = float(rf.loc[idx].mean() * PERIODS_PER_YEAR)
        for name, w in portfolios.items():
            pr = r[list(w.index)] @ w
            mean, vol = pr.mean() * PERIODS_PER_YEAR, pr.std() * np.sqrt(PERIODS_PER_YEAR)
            ports.append(
                {
                    "regime": reg,
                    "portfolio": name,
                    "ann_return": mean,
                    "ann_vol": vol,
                    "sharpe": (mean - rf_reg) / vol if vol > 0 else np.nan,
                    "worst_month": pr.min(),
                    "pct_months_negative": float((pr < 0).mean()),
                }
            )
    return {
        "summary": pd.DataFrame(summary).set_index("regime"),
        "assets": pd.concat(assets, axis=1),
        "portfolios": pd.DataFrame(ports),
    }

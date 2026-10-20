"""Stress tests: replay historical crises and apply simple hypothetical shocks.

Historical scenarios (DECISIONS.md D-024) use DAILY prices so that falls
inside a month are visible. The portfolio is rebalanced to target weights at
each month end and drifts with prices in between, which is exactly the
monthly-rebalanced portfolio used everywhere else: its month-end values
compound to the same numbers as the monthly portfolio returns (tested).

Hypothetical shocks (D-025) are SIMPLE FACTOR SHOCKS: a fixed, instant
percentage move per asset, written out below. They are NOT duration-based.
Stage 2 replaces the rate and spread shocks with bond math (duration times
yield change).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    start: str  # first month in the window, "YYYY-MM"
    end: str  # last month in the window, "YYYY-MM"
    why: str


HISTORICAL_SCENARIOS = [
    Scenario(
        "2008 GFC",
        "2007-11",
        "2009-02",
        "US equity peak (Oct 2007 month end) to the last month end before the March 2009 low. "
        "Credit and liquidity crisis; correlations jumped, Treasuries rallied.",
    ),
    Scenario(
        "2013 taper tantrum",
        "2013-05",
        "2013-08",
        "Fed signals slower bond buying (22 May 2013); 10y yield rose from about 1.6% to about 2.8%. "
        "Pure rate shock with equities roughly flat.",
    ),
    Scenario(
        "2020 COVID crash",
        "2020-02",
        "2020-03",
        "Fastest bear market on record (19 Feb to 23 Mar 2020); credit spreads blew out, then policy response.",
    ),
    Scenario(
        "2022 rate shock",
        "2022-01",
        "2022-12",
        "Calendar 2022: fastest Fed hiking cycle in decades; stocks and bonds fell together.",
    ),
]


# Simple factor shocks: instant % change per asset. Unlisted assets: 0%.
HYPOTHETICAL_SHOCKS: dict[str, dict[str, float]] = {
    # Equity-like assets fall 30%; EM falls more (it fell more in 2008 and
    # 2020). No flight-to-quality rally assumed for bonds (conservative,
    # 2022-style), so diversification is not given free credit.
    "Equity -30%": {"SPY": -0.30, "EFA": -0.30, "EEM": -0.35, "VNQ": -0.30},
    # Credit spreads widen. Sizes are about each fund's worst single month in
    # our data (HYG -11.5% Oct 2008, LQD -10.7% Sep 2008). Treasuries flat.
    "Credit spreads wider": {"HYG": -0.12, "LQD": -0.10},
    # Rates up about 1 percentage point. Sizes are rough, judgment-based
    # rate sensitivities per fund (longer bonds lose more), NOT computed from
    # duration. Stage 2 computes these properly.
    "Rates up": {"SHY": -0.02, "IEF": -0.07, "TLT": -0.15, "TIP": -0.06, "LQD": -0.07},
}


def combined_shock(*names: str) -> dict[str, float]:
    """Add several shocks together (simple sum per asset)."""
    out: dict[str, float] = {}
    for n in names:
        for a, x in HYPOTHETICAL_SHOCKS[n].items():
            out[a] = out.get(a, 0.0) + x
    return out


def apply_shock(weights: pd.Series, shock: dict[str, float], f: float = 0.0, liability_shock: float = 0.0) -> float:
    """Instant change in surplus per dollar of assets: w . shock - f * liability_shock.

    Liabilities default to none. In Stage 3 a rates-up shock will LOWER the
    liability value too (liability_shock < 0), which is the whole point of ALM.
    """
    s = pd.Series(shock, dtype=float).reindex(weights.index, fill_value=0.0)
    return float(weights @ s - f * liability_shock)


# ----------------------------------------------------------- historical


def _window(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Daily prices from the last trading day BEFORE `start` to the last day of `end`."""
    before = prices.loc[: pd.Period(start, "M").start_time - pd.Timedelta(days=1)]
    if before.empty:
        raise ValueError(f"no price before {start} to use as the starting point")
    base_day = before.index[-1]
    return prices.loc[base_day : pd.Period(end, "M").end_time]


def daily_portfolio_path(prices: pd.DataFrame, weights: pd.Series, start: str, end: str) -> pd.Series:
    """Daily value of a portfolio worth 1 at the start, rebalanced every month end."""
    w = weights[weights != 0]
    p = _window(prices[list(w.index)], start, end)
    month = p.index.to_period("M")
    values = pd.Series(np.nan, index=p.index)
    values.iloc[0] = 1.0
    base_val, base_px = 1.0, p.iloc[0]
    for m in month.unique()[1:]:  # row 0 is the base day in the prior month
        seg = p[month == m]
        seg_vals = base_val * (seg / base_px).mul(w, axis=1).sum(axis=1)
        values.loc[seg.index] = seg_vals.values
        base_val, base_px = float(seg_vals.iloc[-1]), seg.iloc[-1]  # rebalance at month end
    return values


def max_drawdown(values: pd.Series | np.ndarray) -> float:
    """Worst peak-to-trough fall, as a positive fraction."""
    v = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(v)
    return float(((peak - v) / peak).max())


def historical_scenarios(
    prices: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    portfolios: dict[str, pd.Series],
    scenarios: list[Scenario] = HISTORICAL_SCENARIOS,
) -> pd.DataFrame:
    """One row per (scenario, portfolio): total return and drawdowns.

    max_drawdown_daily uses daily marks; max_drawdown_monthly uses month ends
    only, to show how much monthly data hides.
    """
    rows = []
    for sc in scenarios:
        for name, w in portfolios.items():
            path = daily_portfolio_path(prices, w, sc.start, sc.end)
            m = monthly_returns.loc[sc.start : sc.end, list(w.index)] @ w
            monthly_path = np.concatenate([[1.0], np.cumprod(1 + m.values)])
            rows.append(
                {
                    "scenario": sc.name,
                    "portfolio": name,
                    "start": sc.start,
                    "end": sc.end,
                    "months": len(m),
                    "total_return": float(path.iloc[-1] - 1),
                    "max_drawdown_daily": max_drawdown(path),
                    "max_drawdown_monthly": max_drawdown(monthly_path),
                }
            )
    return pd.DataFrame(rows)


def asset_scenario_returns(
    monthly_returns: pd.DataFrame, scenarios: list[Scenario] = HISTORICAL_SCENARIOS
) -> pd.DataFrame:
    """Compounded return of each asset over each scenario window."""
    return pd.DataFrame({sc.name: (1 + monthly_returns.loc[sc.start : sc.end]).prod() - 1 for sc in scenarios}).T


def hypothetical_report(portfolios: dict[str, pd.Series]) -> pd.DataFrame:
    shocks = dict(HYPOTHETICAL_SHOCKS)
    shocks["All three at once"] = combined_shock(*HYPOTHETICAL_SHOCKS)
    return pd.DataFrame({name: {s: apply_shock(w, sh) for s, sh in shocks.items()} for name, w in portfolios.items()})


# ------------------------------------------- duration-based (Stage 2)
#
# These ARE duration-based: each bond ETF is represented by a par bond at its
# representative maturity (scenarios.RepBond) and the shock is applied to the
# Treasury par curve or to the credit spread. We report the duration +
# convexity estimate AND a full revaluation of the representative bond, so the
# approximation error is visible. Equities, real estate, gold and BIL are NOT
# shocked here (0%), which is stated in every report (D-041).

# Curve shocks in basis points, given at par-curve tenors (linear in between).
CURVE_SHOCKS_BP: dict[str, dict[float, float]] = {
    "Parallel +100bp": {0.25: 100, 30.0: 100},
    "Parallel -100bp": {0.25: -100, 30.0: -100},
    # Steepener: short end down 50bp, long end up 50bp, pivot at 10 years.
    "Steepener": {0.25: -50, 2.0: -50, 10.0: 0, 30.0: 50},
    "Flattener": {0.25: 50, 2.0: 50, 10.0: 0, 30.0: -50},
}
# Credit spread (OAS) shocks in bp: roughly a moderate recession. 2008 was
# 3 to 4 times larger for high yield.
SPREAD_SHOCKS_BP: dict[str, dict[str, float]] = {
    "IG spreads +100bp": {"LQD": 100},
    "HY spreads +400bp": {"HYG": 400},
}


def curve_shock_at(shock: dict[float, float], tenors) -> np.ndarray:
    k = sorted(shock)
    return np.interp(np.asarray(tenors, float), k, [shock[x] for x in k]) * 1e-4


def duration_stress(
    rep: dict, par_tenors, par_yields, curve_shocks: dict | None = None, spread_shocks: dict | None = None
) -> pd.DataFrame:
    """Per bond ETF and shock: duration+convexity estimate vs full revaluation.

    Curve shocks: first-order term = key rate durations x the shock at each
    key tenor (exact for any curve shape to first order), second-order term =
    effective convexity x (shock at the bond's maturity)^2 / 2. Full
    revaluation: rebuild the zero curve from the shocked par curve and
    reprice. The ETF's rate beta scales the shock (D-037).
    Spread shocks: shift the bond's yield by the spread change; estimate with
    modified duration and convexity, revalue at the new yield. Both sides use
    the same semiannual yield convention.
    """
    from engine import fixedincome as fi

    tenors = np.asarray(par_tenors, float)
    base_yields = np.asarray(par_yields, float)
    base_curve = fi.bootstrap_zero_curve(tenors, base_yields)
    rows = []
    for etf, rb in rep.items():
        spread = rb.bond.coupon - base_curve.par_yield(rb.bond.maturity)  # 0 for Treasuries
        krd = fi.key_rate_durations(rb.bond, tenors, base_yields, spread=spread).values
        conv = fi.effective_convexity(rb.bond, tenors, base_yields, spread=spread)
        p0 = fi.price_from_curve(rb.bond, base_curve, spread)
        for name, shock in (CURVE_SHOCKS_BP if curve_shocks is None else curve_shocks).items():
            bump = rb.rate_beta * curve_shock_at(shock, tenors)
            dy_mat = rb.rate_beta * float(curve_shock_at(shock, [rb.bond.maturity])[0])
            est = float(-(krd @ bump) + 0.5 * conv * dy_mat**2)
            shocked = fi.bootstrap_zero_curve(tenors, base_yields + bump)
            full = fi.price_from_curve(rb.bond, shocked, spread) / p0 - 1
            rows.append(
                {
                    "shock": name,
                    "type": "rates",
                    "etf": etf,
                    "dy_bp": dy_mat * 1e4,
                    "duration_convexity": est,
                    "full_revaluation": full,
                }
            )
    for name, shock in (SPREAD_SHOCKS_BP if spread_shocks is None else spread_shocks).items():
        for etf, bp in shock.items():
            rb = rep[etf]
            ds = bp * 1e-4
            est = float(fi.duration_convexity_change(rb.par_duration, rb.par_convexity, ds))
            full = fi.price_from_yield(rb.bond, rb.ytm + ds) / fi.price_from_yield(rb.bond, rb.ytm) - 1
            rows.append(
                {
                    "shock": name,
                    "type": "spread",
                    "etf": etf,
                    "dy_bp": bp,
                    "duration_convexity": est,
                    "full_revaluation": full,
                }
            )
    out = pd.DataFrame(rows)
    out["approx_error"] = out["duration_convexity"] - out["full_revaluation"]
    return out


def duration_stress_portfolios(
    table: pd.DataFrame, portfolios: dict[str, pd.Series], column: str = "full_revaluation"
) -> pd.DataFrame:
    """Portfolio impact = sum of weight x ETF change; unshocked assets count as 0."""
    out = {}
    for shock, g in table.groupby("shock", sort=False):
        move = g.set_index("etf")[column]
        out[shock] = {p: float((w * move.reindex(w.index).fillna(0.0)).sum()) for p, w in portfolios.items()}
    return pd.DataFrame(out).T

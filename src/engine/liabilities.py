"""Synthetic insurer liabilities. SYNTHETIC: not any real insurer's book.

Shape (DECISIONS.md D-045): a block of payout annuities in runoff. Annual
payments at the end of years 1 to 30, falling 3% a year as annuitants die,
scaled so the present value is $1,000 million on today's Treasury curve.
Nominal and fixed: no inflation indexation, no policyholder options except an
optional surrender (outflow) shock used only in the liquidity test.

Discounting: the SAME Treasury zero curve as the assets (D-046), so a rate
move hits both sides consistently. The block is a short position in the
surplus: surplus = assets - liabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import fixedincome as fi

PV_TARGET = 1_000.0  # $ millions
YEARS = 30
DECAY = 0.03
FUNDING_RATIO = 1.10  # assets / liabilities at the start (D-047)
SURRENDER_RATE = 0.10  # share of liability PV that can be surrendered in a year (liquidity test)


@dataclass(frozen=True)
class LiabilityBlock:
    stream: fi.CashFlowStream  # times in years, amounts in $ millions
    label: str = "Synthetic payout annuity block (illustrative)"

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self.stream.times, float)

    @property
    def amounts(self) -> np.ndarray:
        return np.asarray(self.stream.amounts, float)


def annuity_block(
    curve: fi.ZeroCurve, pv: float = PV_TARGET, years: int = YEARS, decay: float = DECAY
) -> LiabilityBlock:
    t = np.arange(1, years + 1, dtype=float)
    raw = (1 - decay) ** (t - 1)
    scale = pv / fi.price_from_curve(fi.CashFlowStream(t, raw), curve)
    return LiabilityBlock(fi.CashFlowStream(t, raw * scale))


def flat_curve_pv(block: LiabilityBlock, y: float) -> float:
    """Closed form on a flat semiannual curve for the geometric annuity.

    PV = C v sum_{k=0}^{N-1} (g v)^k = C v (1 - q^N) / (1 - q), with
    v = (1 + y/2)^-2 (one year of discounting), g = first-to-second payment
    ratio, q = g v.
    """
    a = block.amounts
    c, g, n = a[0], a[1] / a[0], len(a)
    v = (1 + y / 2) ** -2
    q = g * v
    return float(c * v * (1 - q**n) / (1 - q))


def measures(block: LiabilityBlock, par_tenors, par_yields, spread: float = 0.0) -> dict:
    """PV, yield, durations, convexity, DV01 and key rate durations.

    spread: continuously compounded add-on to the Treasury zero curve used to
    discount the liabilities (D-063). 0 = the base case (Treasury curve).
    """
    tenors = np.asarray(par_tenors, float)
    y = np.asarray(par_yields, float)
    curve = fi.bootstrap_zero_curve(tenors, y)
    s = block.stream
    pv = fi.price_from_curve(s, curve, spread)
    irr = fi.ytm(s, pv)
    eff = fi.effective_duration(s, tenors, y, spread=spread)
    h = fi.BP
    spread_dv01 = (fi.price_from_curve(s, curve, spread - h) - fi.price_from_curve(s, curve, spread + h)) / 2
    return {
        "pv": pv,
        "irr": irr,
        "macaulay": fi.macaulay_duration(s, irr),
        "modified": fi.modified_duration(s, irr),
        "convexity": fi.convexity(s, irr),
        "effective_duration": eff,
        "effective_convexity": fi.effective_convexity(s, tenors, y, spread=spread),
        "dv01": eff * pv * fi.BP,  # $m per 1bp, curve-based
        "spread_dv01": spread_dv01,  # $m per 1bp of discount spread
        "krd": fi.key_rate_durations(s, tenors, y, spread=spread),
    }


def value_on_curves(
    block: LiabilityBlock, par_tenors, par: np.ndarray, t_years: float, spread: float | np.ndarray = 0.0
) -> np.ndarray:
    """PV at time t of the payments still to come, on many curves at once.

    par: (..., n_tenors) par curves observed at time t. Payments at or before
    t are treated as already paid. spread: discount add-on (scalar, or one
    value per curve with shape par.shape[:-1]); 0 = Treasury discounting.
    """
    remaining = block.times - t_years
    keep = remaining > 1e-9
    if not keep.any():
        return np.zeros(par.shape[:-1])
    times, log_dfs = fi.bootstrap_log_dfs(par_tenors, par)
    dfs = fi.interp_log_dfs(times, log_dfs, remaining[keep])
    sp = np.asarray(spread, float)
    if np.any(sp != 0):
        dfs = dfs * np.exp(-sp[..., None] * remaining[keep])
    return dfs @ block.amounts[keep]


def payments_between(block: LiabilityBlock, t0: float, t1: float) -> float:
    """Total scheduled payments in (t0, t1]."""
    m = (block.times > t0 + 1e-9) & (block.times <= t1 + 1e-9)
    return float(block.amounts[m].sum())


def historical_returns(block: LiabilityBlock, par_tenors, par_hist: pd.DataFrame) -> pd.Series:
    """Monthly liability "returns" on history, for regime analysis.

    A constant profile is rolled each month: value the block's cash flows on
    last month's curve, then value the same flows one month closer on this
    month's curve. The change is the liability return (accretion plus the
    effect of the curve move). No payment falls inside the first month.
    """
    dt = 1 / 12
    times, ld = fi.bootstrap_log_dfs(par_tenors, par_hist.values)
    v0 = fi.interp_log_dfs(times, ld, block.times) @ block.amounts
    v1 = fi.interp_log_dfs(times, ld, block.times - dt) @ block.amounts
    r = v1[1:] / v0[:-1] - 1
    return pd.Series(r, index=par_hist.index[1:], name="liability_return")

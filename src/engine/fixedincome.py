"""Yield curve construction and bond math.

Conventions (DECISIONS.md D-032, D-033):
- Treasury constant-maturity yields are treated as semiannual PAR yields
  (Treasury builds the CMT curve as a par curve). The 3-month yield is
  treated as a zero-coupon rate with semiannual compounding.
- Bonds pay semiannual coupons, face 100, priced on a coupon date (no accrued
  interest, so clean price = dirty price). Times are in years.
- Zero curve: par yields are interpolated LINEARLY onto a semiannual grid,
  discount factors are bootstrapped node by node, and between nodes the
  discount factor is interpolated LOG-LINEARLY (constant forward rate between
  nodes). Beyond 30 years the last forward rate is held flat.

Plain language: a par yield says "a bond paying this coupon is worth exactly
100". Bootstrapping turns those into zero rates: "what is $1 paid at time t
worth today", one maturity at a time, starting from the shortest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

FREQ = 2  # semiannual
BP = 1e-4


# ------------------------------------------------------------------ curve


@dataclass(frozen=True)
class ZeroCurve:
    times: np.ndarray  # node times (years), increasing, > 0
    dfs: np.ndarray  # discount factors at the nodes

    def df(self, t: float | np.ndarray) -> np.ndarray:
        """Discount factor; log-linear between nodes, flat forward beyond the ends."""
        t = np.asarray(t, dtype=float)
        x = np.concatenate([[0.0], self.times])
        y = np.concatenate([[0.0], np.log(self.dfs)])
        out = np.interp(t, x, y)
        # flat forward extrapolation past the last node
        fwd_last = -(y[-1] - y[-2]) / (x[-1] - x[-2])
        out = np.where(t > x[-1], y[-1] - fwd_last * (t - x[-1]), out)
        return np.exp(out)

    def zero_rate(self, t: float | np.ndarray, compounding: str = "continuous") -> np.ndarray:
        t = np.asarray(t, dtype=float)
        d = self.df(t)
        if compounding == "continuous":
            return -np.log(d) / t
        if compounding == "semiannual":
            return FREQ * (d ** (-1.0 / (FREQ * t)) - 1.0)
        raise ValueError("compounding must be 'continuous' or 'semiannual'")

    def par_yield(self, maturity: float) -> float:
        times = coupon_times(maturity)
        d = self.df(times)
        return float(FREQ * (1 - d[-1]) / d.sum())


def interpolate_par(par_tenors: Sequence[float], par_yields: Sequence[float], t: float | np.ndarray) -> np.ndarray:
    """Linear interpolation of par yields in maturity (flat beyond the ends)."""
    return np.interp(t, np.asarray(par_tenors, float), np.asarray(par_yields, float))


def bootstrap_zero_curve(par_tenors: Sequence[float], par_yields: Sequence[float], max_t: float = 30.0) -> ZeroCurve:
    """Semiannual bootstrap from par yields (decimals).

    Node 0.25: the 3-month yield as a zero rate, DF = (1 + y/2)^(-0.5).
    Nodes 0.5, 1.0, ..., max_t: a par bond with coupon c = par yield at that
    maturity (linearly interpolated) satisfies
        100 = sum_i (100 c / 2) DF(t_i) + 100 DF(T)
    which we solve for DF(T) using the DFs already found for earlier nodes.
    """
    tenors = np.asarray(par_tenors, float)
    yields = np.asarray(par_yields, float)
    order = np.argsort(tenors)
    tenors, yields = tenors[order], yields[order]
    times, dfs = [], []
    if tenors[0] < 0.5:
        times.append(tenors[0])
        dfs.append((1 + yields[0] / FREQ) ** (-FREQ * tenors[0]))
    grid = np.arange(0.5, max_t + 1e-9, 0.5)
    running = 0.0  # sum of DFs at earlier semiannual nodes
    for t in grid:
        c = float(interpolate_par(tenors, yields, t))
        d = (1 - c / FREQ * running) / (1 + c / FREQ)
        if not 0 < d < 1.5:
            raise ValueError(f"bootstrap produced an invalid discount factor {d} at t={t}")
        times.append(t)
        dfs.append(d)
        running += d
    return ZeroCurve(np.array(times), np.array(dfs))


# ------------------------------------------------------------------ bonds


@dataclass(frozen=True)
class Bond:
    maturity: float  # years
    coupon: float  # annual coupon rate, decimal (0.05 = 5%)
    face: float = 100.0


def coupon_times(maturity: float) -> np.ndarray:
    """Payment times: maturity, maturity - 0.5, ... down to the first > 0."""
    n = int(np.ceil(maturity * FREQ - 1e-9))
    return np.sort(maturity - np.arange(n) / FREQ)


@dataclass(frozen=True)
class CashFlowStream:
    """Any fixed set of cash flows (e.g. a liability block). Every pricing and
    risk function in this module accepts it in place of a Bond."""

    times: np.ndarray  # years from today, > 0
    amounts: np.ndarray


def cash_flows(bond: Bond | CashFlowStream) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bond, CashFlowStream):
        return np.asarray(bond.times, float), np.asarray(bond.amounts, float)
    t = coupon_times(bond.maturity)
    cf = np.full(len(t), bond.face * bond.coupon / FREQ)
    cf[-1] += bond.face
    return t, cf


# ------------------------------------------------ vectorized bootstrap


def bootstrap_log_dfs(par_tenors: Sequence[float], par_yields: np.ndarray, max_t: float = 30.0):
    """Same bootstrap as bootstrap_zero_curve, for many curves at once.

    par_yields: array (..., n_tenors). Returns (node_times, log discount
    factors with shape (..., n_nodes)). Used to value liabilities on every
    simulated curve (hundreds of thousands of curves).
    """
    tenors = np.asarray(par_tenors, float)
    y = np.asarray(par_yields, float)
    grid = np.arange(0.5, max_t + 1e-9, 0.5)
    # par yield at each grid point: linear interpolation along the last axis
    idx = np.clip(np.searchsorted(tenors, grid) - 1, 0, len(tenors) - 2)
    w = np.clip((grid - tenors[idx]) / (tenors[idx + 1] - tenors[idx]), 0.0, 1.0)
    c = y[..., idx] * (1 - w) + y[..., idx + 1] * w
    first = (1 + y[..., 0] / FREQ) ** (-FREQ * tenors[0])
    dfs = np.empty(y.shape[:-1] + (len(grid),))
    running = np.zeros(y.shape[:-1])
    for k in range(len(grid)):
        d = (1 - c[..., k] / FREQ * running) / (1 + c[..., k] / FREQ)
        dfs[..., k] = d
        running = running + d
    times = np.concatenate([[tenors[0]], grid])
    log_dfs = np.log(np.concatenate([first[..., None], dfs], axis=-1))
    return times, log_dfs


def interp_log_dfs(times: np.ndarray, log_dfs: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Discount factors at times t (1-D, shared by all curves), log-linear
    between nodes and through zero at t = 0, flat forward past the last node."""
    t = np.asarray(t, float)
    x = np.concatenate([[0.0], times])
    ld = np.concatenate([np.zeros(log_dfs.shape[:-1] + (1,)), log_dfs], axis=-1)
    j = np.clip(np.searchsorted(x, t) - 1, 0, len(x) - 2)
    w = (t - x[j]) / (x[j + 1] - x[j])
    out = ld[..., j] * (1 - w) + ld[..., j + 1] * w
    beyond = t > x[-1]
    if beyond.any():
        fwd = (ld[..., -1] - ld[..., -2]) / (x[-1] - x[-2])
        out[..., beyond] = ld[..., -1:] + fwd[..., None] * (t[beyond] - x[-1])
    return np.exp(out)


def price_from_curve(bond: Bond | CashFlowStream, curve: ZeroCurve, spread: float = 0.0) -> float:
    """Present value off the zero curve. `spread` is a continuously compounded
    parallel add-on to the zero rates (used for credit spread shocks)."""
    t, cf = cash_flows(bond)
    return float(np.sum(cf * curve.df(t) * np.exp(-spread * t)))


def price_from_yield(bond: Bond, y: float) -> float:
    t, cf = cash_flows(bond)
    return float(np.sum(cf * (1 + y / FREQ) ** (-FREQ * t)))


def ytm(bond: Bond, price: float) -> float:
    """Yield to maturity (semiannual, decimal) that reproduces `price`."""
    return float(brentq(lambda y: price_from_yield(bond, y) - price, -0.5, 2.0, xtol=1e-14))


def macaulay_duration(bond: Bond, y: float) -> float:
    """PV-weighted average time to the cash flows, in years."""
    t, cf = cash_flows(bond)
    pv = cf * (1 + y / FREQ) ** (-FREQ * t)
    return float(np.sum(t * pv) / pv.sum())


def modified_duration(bond: Bond, y: float) -> float:
    """% price change per 1.00 (100%) change in yield: -(1/P) dP/dy."""
    return macaulay_duration(bond, y) / (1 + y / FREQ)


def convexity(bond: Bond, y: float) -> float:
    """(1/P) d2P/dy2 for semiannual compounding."""
    t, cf = cash_flows(bond)
    p = price_from_yield(bond, y)
    return float(np.sum(cf * t * (t + 1 / FREQ) * (1 + y / FREQ) ** (-FREQ * t - 2)) / p)


def dv01(bond: Bond, y: float) -> float:
    """Price change (per 100 face) for a 1bp fall in yield."""
    return modified_duration(bond, y) * price_from_yield(bond, y) * BP


def _reprice_bumped(bond, par_tenors, par_yields, bump, spread: float = 0.0) -> float:
    curve = bootstrap_zero_curve(par_tenors, np.asarray(par_yields, float) + bump)
    return price_from_curve(bond, curve, spread)


def effective_duration(bond: Bond, par_tenors, par_yields, h: float = BP, spread: float = 0.0) -> float:
    """Duration w.r.t. a PARALLEL shift of the par curve, by bump and rebuild."""
    p0 = _reprice_bumped(bond, par_tenors, par_yields, 0.0, spread)
    up = _reprice_bumped(bond, par_tenors, par_yields, h, spread)
    dn = _reprice_bumped(bond, par_tenors, par_yields, -h, spread)
    return (dn - up) / (2 * p0 * h)


def effective_convexity(bond: Bond, par_tenors, par_yields, h: float = 10 * BP, spread: float = 0.0) -> float:
    p0 = _reprice_bumped(bond, par_tenors, par_yields, 0.0, spread)
    up = _reprice_bumped(bond, par_tenors, par_yields, h, spread)
    dn = _reprice_bumped(bond, par_tenors, par_yields, -h, spread)
    return (up + dn - 2 * p0) / (p0 * h * h)


def key_rate_durations(bond: Bond, par_tenors, par_yields, h: float = BP, spread: float = 0.0) -> pd.Series:
    """Duration to each key par rate, bumping one tenor at a time.

    Because par yields are interpolated linearly, bumping one key tenor moves
    the curve in a triangle that peaks at that tenor and fades to zero at its
    neighbours. The triangles add up to a parallel shift, so the key rate
    durations add up to the effective duration (tested).
    """
    tenors = np.asarray(par_tenors, float)
    base = np.asarray(par_yields, float)
    p0 = _reprice_bumped(bond, tenors, base, 0.0, spread)
    out = {}
    for i, k in enumerate(tenors):
        e = np.zeros_like(base)
        e[i] = h
        up = _reprice_bumped(bond, tenors, base, e, spread)
        dn = _reprice_bumped(bond, tenors, base, -e, spread)
        out[k] = (dn - up) / (2 * p0 * h)
    return pd.Series(out, name="key_rate_duration")


def duration_convexity_change(duration: float, conv: float, dy: float | np.ndarray) -> np.ndarray:
    """Second-order estimate of the % price change: -D dy + 0.5 C dy^2."""
    dy = np.asarray(dy, dtype=float)
    return -duration * dy + 0.5 * conv * dy**2

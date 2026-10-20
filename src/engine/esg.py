"""Empirical economic scenario generator (Stage 3 default). Fixes P-013.

Plain language: instead of a formula for how rates move, replay real history.
1. PCA on monthly changes of the Treasury par curve (8 tenors) finds the
   three shapes that explain almost all curve moves: level (everything moves
   together), slope (short vs long), curvature (middle vs ends).
2. Each simulated month is a REAL historical month, drawn in 6-month blocks
   (as in Stage 1b). That month's curve factor moves, credit spread move and
   ETF returns are used together, so all the cross-asset links of that month
   come along for free.
3. The curve is rebuilt month by month: y_t = y_{t-1} + kappa dt (anchor - y_{t-1})
   + (level, slope, curvature moves). The pull toward the anchor (D-044)
   replaces history's drift with a forward-looking view.

Asset returns per simulated month (D-043):
- bond ETFs: curve part + residual part.
  Curve part: carry (yield / 12) - D dy + 0.5 C dy^2 at the representative
  maturity, using the SIMULATED par curve.
  Residual part: what the curve did NOT explain in that same historical month
  (actual ETF return minus the same formula on the actual historical curve),
  demeaned. For credit this carries the real spread, default and equity-like
  behavior (2008 included); for TIP the real-yield and inflation news.
  Credit then adds today's expected excess return: (OAS - expected loss) / 12.
- BIL: the 3-month yield / 12;
- SPY, EFA, EEM, VNQ, GLD: that historical month's return, shifted so the
  long-run mean equals Bayes-Stein (mean adjustment, as in Stage 1b D-022).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import data
from engine import fixedincome as fi
from engine import scenarios as sc

EQUITY_LIKE = ["SPY", "EFA", "EEM", "VNQ", "GLD"]


# -------------------------------------------------------------------- PCA


@dataclass
class CurvePCA:
    tenors: list[float]
    loadings: np.ndarray  # (n_tenors, n_tenors), columns = components, sorted by variance
    eigenvalues: np.ndarray
    mean_change: np.ndarray  # (n_tenors,)
    scores: pd.DataFrame  # (months, n_tenors) factor moves per historical month

    @property
    def explained(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum()


def curve_pca(par: pd.DataFrame, tenors: list[float]) -> CurvePCA:
    """PCA (eigen-decomposition of the covariance) of monthly par-yield changes.

    Signs are fixed so that a positive level move raises all yields and a
    positive slope move raises long yields relative to short ones.
    """
    d = par.diff().dropna()
    mean = d.mean().values
    cov = np.cov((d.values - mean), rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vecs[:, 0].sum() < 0:
        vecs[:, 0] *= -1
    if vecs[-1, 1] < vecs[0, 1]:
        vecs[:, 1] *= -1
    if vecs[len(tenors) // 2, 2] > 0:  # curvature: belly down relative to wings
        vecs[:, 2] *= -1
    scores = pd.DataFrame((d.values - mean) @ vecs, index=d.index)
    return CurvePCA(list(tenors), vecs, vals, mean, scores)


# ---------------------------------------------------------------- anchors


def forward_par_curve(curve: fi.ZeroCurve, tenors: list[float], horizon: float = 5.0) -> np.ndarray:
    """Par yields the market implies today for `horizon` years ahead.

    Forward discount factor for maturity T starting at h: DF(h + t) / DF(h).
    Note: forwards include any term premium, so they lean above the market's
    true expected rates (D-044).
    """
    out = []
    dh = float(curve.df(horizon))
    for T in tenors:
        if T < 0.5:
            fwd_df = float(curve.df(horizon + T)) / dh
            out.append(fi.FREQ * (fwd_df ** (-1 / (fi.FREQ * T)) - 1))
        else:
            t = fi.coupon_times(T)
            d = curve.df(horizon + t) / dh
            out.append(fi.FREQ * (1 - d[-1]) / d.sum())
    return np.array(out)


# ------------------------------------------------------------- simulation


@dataclass(frozen=True)
class ESGConfig:
    n_paths: int = 10_000
    n_steps: int = 60
    block_length: int = 6
    seed: int = 20261019
    n_factors: int = 3
    kappa: float = 0.15  # pull toward the anchor, per year (Stage 2 Vasicek a)


@dataclass
class ESGScenarios:
    months: np.ndarray  # (n_paths, n_steps) index into the historical months
    par: np.ndarray  # (n_paths, n_steps + 1, n_tenors) simulated par curves
    returns: dict[str, np.ndarray]  # ETF -> (n_paths, n_steps) monthly returns
    tenors: list[float]
    ig_spread: np.ndarray | None = None  # (n_paths, n_steps + 1) IG spread level, for liability discounting (D-064)


def block_indices(n_hist: int, n_paths: int, n_steps: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = -(-n_steps // block)
    starts = rng.integers(0, n_hist, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_hist
    return idx.reshape(n_paths, n_blocks * block)[:, :n_steps]


def simulate(
    fm: sc.FactorModel,
    pca: CurvePCA,
    returns: pd.DataFrame,
    fred_m: pd.DataFrame,
    mu: pd.Series,
    anchor: np.ndarray,
    cfg: ESGConfig = ESGConfig(),  # noqa: B008 (frozen dataclass, safe default)
) -> ESGScenarios:
    """Joint scenarios: the Treasury curve and all 12 ETF returns."""
    rng = np.random.default_rng(cfg.seed)
    hist_months = pca.scores.index
    idx = block_indices(len(hist_months), cfg.n_paths, cfg.n_steps, cfg.block_length, rng)
    dt = 1 / 12
    k = cfg.n_factors
    shocks = pca.scores.values[:, :k][idx] @ pca.loadings[:, :k].T  # (paths, steps, tenors)

    tenors, par_hist = sc.par_curve_history(fred_m)
    y0 = par_hist.iloc[-1].values
    par = np.empty((cfg.n_paths, cfg.n_steps + 1, len(tenors)))
    par[:, 0] = y0
    for t in range(cfg.n_steps):
        par[:, t + 1] = par[:, t] + cfg.kappa * dt * (anchor - par[:, t]) + shocks[:, t]

    hist = returns.reindex(hist_months)
    par_h = par_hist.reindex(par_hist.index)  # month ends incl. the base month
    out: dict[str, np.ndarray] = {}
    out["BIL"] = par[:, :-1, 0] * dt
    for etf, rb in fm.rep.items():
        mat = sc.REP_MATURITY[etf]
        # curve part on the simulated curve
        y_m = _interp_last_axis(tenors, par, mat)
        dy = np.diff(y_m, axis=1)
        curve_part = y_m[:, :-1] * dt - rb.duration * dy + 0.5 * rb.convexity * dy**2
        # residual part from the same historical months
        yh = _interp_last_axis(tenors, par_h.values, mat)
        yh = pd.Series(yh, index=par_h.index)
        dyh = yh.diff().reindex(hist_months).values
        model_h = yh.shift(1).reindex(hist_months).values * dt - rb.duration * dyh + 0.5 * rb.convexity * dyh**2
        resid = hist[etf].values - model_h
        resid = resid - resid.mean()
        ret = curve_part + resid[idx]
        if etf in sc.CREDIT_ETFS:
            ret = ret + (fm.oas0[etf] - sc.EXPECTED_LOSS[etf]) * dt
        out[etf] = ret
    for a in EQUITY_LIKE:
        x = hist[a].values
        out[a] = (x - x.mean() + mu[a] / 12)[idx]
    ig = None
    if hasattr(fm, "spread") and hasattr(fm, "s0"):
        # IG spread path from the SAME historical months: Baa-10y log changes
        # (demeaned) with a pull to its long-run median, scaled to today's ICE IG
        # level. No extra random draws, so all other outputs are unchanged.
        log_s = np.log(fred_m[data.CREDIT_FACTOR_SERIES])
        dls = log_s.diff().reindex(hist_months).values
        dls = dls - np.nanmean(dls)
        sp = fm.spread
        ls = np.empty((cfg.n_paths, cfg.n_steps + 1))
        ls[:, 0] = np.log(fm.s0)
        for t in range(cfg.n_steps):
            ls[:, t + 1] = ls[:, t] + sp["kappa"] * dt * (sp["theta_log"] - ls[:, t]) + dls[idx[:, t]]
        ig = fm.oas0["LQD"] * np.exp(ls) / fm.s0
    return ESGScenarios(idx, par, out, list(tenors), ig)


def _interp_last_axis(tenors, arr: np.ndarray, t: float) -> np.ndarray:
    ten = np.asarray(tenors, float)
    j = int(np.clip(np.searchsorted(ten, t) - 1, 0, len(ten) - 2))
    w = (t - ten[j]) / (ten[j + 1] - ten[j])
    return arr[..., j] * (1 - w) + arr[..., j + 1] * w


def annual_vol(x: np.ndarray) -> float:
    return float(x.std() * np.sqrt(12))

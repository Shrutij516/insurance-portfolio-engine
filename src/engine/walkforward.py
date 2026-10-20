"""Walk-forward (out-of-sample) test of the headline recommendation process.

Plain language: pretend it is December 2010. Using ONLY data available then,
run the full Stage 3 process (scenario generator, capital proxy, insurer
constraints, duration band) and pick a portfolio. Hold it through 2011 on
what actually happened, then repeat every December. Compare with simpler
alternatives on the same realized history (DECISIONS.md D-059 to D-062).

Point-in-time rules at each rebalance date t (end of December):
- returns, yields and spreads up to and including month t only;
- scenario generator, PCA, forward-curve anchor, measured durations,
  Bayes-Stein means: all re-estimated from that truncated history;
- credit spread levels from Moody's Baa-10y at t (ICE OAS exists only from
  Oct 2023): IG = Baa-10y, HY = HY_SPREAD_MULTIPLE x Baa-10y (assumption);
- fixed model settings (not re-estimated): curve pull kappa, expected default
  losses, capital factors, haircuts. These are assumptions, not fitted to the
  evaluation period, but kappa was calibrated on the full sample in Stage 2
  (stated as a mild look-ahead).

Realized evaluation (D-061): one synthetic liability block issued at the
first rebalance (PV 1,000 on that date's curve) runs off for real. Assets
start at 1,100, earn actual monthly ETF total returns, buy-and-hold within
each year, pay each December's liability payment pro rata, and are rebalanced
to the new target weights. Liabilities are revalued every month on the actual
historical Treasury curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from engine import alm, data, esg
from engine import estimation as est
from engine import fixedincome as fi
from engine import liabilities as lb
from engine import scenarios as sc
from engine.optimizer import (
    check_feasible,
    max_return_weights,
    min_variance,
    surplus_expected_return,
)

ASSETS = alm.ASSETS
HY_SPREAD_MULTIPLE = 2.2  # HY spread proxy = 2.2 x Baa-10y (assumption, D-060)


@dataclass(frozen=True)
class WFConfig:
    first_rebalance: str = "2010-12-31"
    n_paths: int = 5000
    seed: int = 20261019
    a0: float = 1100.0
    l0: float = 1000.0
    target_frac_surplus: float = 0.10  # expected surplus growth target: 10% of current surplus
    band_frac_liab: float = 0.01  # duration band: 1% of liabilities per 100bp ($10m on $1bn)
    coverage: float = alm.COVERAGE_TARGET


DEFAULT = WFConfig()


@dataclass
class Decision:
    date: pd.Timestamp
    weights: pd.Series
    matched: pd.Series
    status: str
    details: dict = field(default_factory=dict)


def remaining_block(block0: lb.LiabilityBlock, years_elapsed: float) -> lb.LiabilityBlock:
    """Cash flows still to come, with times measured from now."""
    keep = block0.times > years_elapsed + 1e-9
    return lb.LiabilityBlock(
        fi.CashFlowStream(block0.times[keep] - years_elapsed, block0.amounts[keep]), label=block0.label
    )


def pit_spreads(fred_t: pd.DataFrame) -> dict[str, float]:
    baa = float(fred_t[data.CREDIT_FACTOR_SERIES].iloc[-1])
    return {"LQD": baa, "HYG": HY_SPREAD_MULTIPLE * baa}


def decide(
    t: pd.Timestamp,
    rets: pd.DataFrame,
    fred_m: pd.DataFrame,
    block_t: lb.LiabilityBlock,
    a_t: float,
    cfg: WFConfig = DEFAULT,
) -> Decision:
    """Run the headline process with data up to t only. Returns target weights."""
    # .copy(): fresh contiguous arrays, so results never depend on how the caller built the frames (P-025)
    r_t, f_t = rets.loc[:t].copy(), fred_m.loc[:t].copy()
    a = r_t[ASSETS]
    mu_bs = est.bayes_stein_mean(a).mu
    tenors, par = sc.par_curve_history(f_t)
    y = par.iloc[-1].values
    curve = fi.bootstrap_zero_curve(tenors, y)
    oas = pit_spreads(f_t)
    emp = sc.empirical_durations(r_t, f_t)
    rep = sc.representative_bonds(curve, emp, oas)
    fm = SimpleNamespace(rep=rep, oas0=oas)
    pca = esg.curve_pca(par, tenors)
    anchor = esg.forward_par_curve(curve, tenors)
    scen = esg.simulate(
        fm, pca, r_t, f_t, mu_bs, anchor, esg.ESGConfig(n_paths=cfg.n_paths, n_steps=12, seed=cfg.seed + t.year)
    )
    L = alm.liability_paths(block_t, scen)
    l_t = float(L[0, 0])
    s_t = a_t - l_t
    R1, rL = alm.one_year_inputs(scen, L, block_t)
    mu, cov, liab = alm.scenario_moments(R1, rL, l_t / a_t)
    krd = alm.etf_key_rate_durations(rep, tenors, y)
    lm = lb.measures(block_t, tenors, y)
    rate_chg = alm.etf_curve_changes(rep, tenors, y, alm.RATE_SHOCK)
    liab_rate = alm.liability_curve_changes(block_t, tenors, y, alm.RATE_SHOCK)
    lcr = alm.lcr_coefficients(
        alm.liquidity_stress_moves(rep, tenors, y, severe=True), a_t, alm.stressed_outflow(block_t, l_t)
    )
    matched = alm.key_rate_matched(krd, lm["krd"], a_t, l_t)
    details = {"surplus": s_t, "liabilities": l_t, "assets": a_t, "liability_duration": lm["effective_duration"]}
    if s_t <= 0:
        return Decision(t, matched, matched, "surplus <= 0: fell back to key-rate matched", details)
    rules = alm.capital_rules(a_t, s_t / cfg.coverage, rate_chg, liab_rate)
    band = alm.duration_band_rule(krd, lm["krd"], a_t, l_t, cfg.band_frac_liab * l_t)
    p = alm.insurer_problem(mu, cov, liab, lcr, [*rules, band])
    target = cfg.target_frac_surplus * s_t / a_t
    try:
        w, status = alm.solve_for_target(p, target)
        mv_ret = min_variance(p).expected_return
        hi = surplus_expected_return(max_return_weights(p), p)
    except ValueError as exc:
        return Decision(t, matched, matched, f"infeasible ({exc}): fell back to key-rate matched", details)
    dS = alm.surplus_change(w.reindex(ASSETS).fillna(0.0), R1, rL, a_t, l_t)
    details.update(
        {
            "pred_mean": float(dS.mean()),
            "pred_vol": float(dS.std(ddof=1)),
            "pred_var95": float(-np.quantile(dS, 0.05)),
            "pred_cvar95": alm.cvar_worst_k(-dS, 0.95),
        }
    )
    details.update(
        {
            "target": target * a_t,
            "min_var_return": mv_ret * a_t,
            "max_return": hi * a_t,
            "violations": check_feasible(w.values, p, tol=1e-6),
            "capital": alm.capital_charge(w, a_t, rate_chg, liab_rate)["total"],
            "cap": s_t / cfg.coverage,
            "gap_per_100bp": alm.duration_gap(w, krd, lm["krd"], a_t, l_t)["dollar_duration_gap_per_100bp"],
            "band": cfg.band_frac_liab * l_t,
            "lcr": float(lcr @ w),
        }
    )
    return Decision(t, w.reindex(ASSETS).fillna(0.0), matched, status, details)


# ------------------------------------------------------------------ realize


@dataclass
class Path:
    name: str
    assets: pd.Series
    liabilities: pd.Series
    weights: dict  # rebalance date -> target weights
    turnover: dict  # rebalance date -> one-way turnover (0 at the first date)

    @property
    def surplus(self) -> pd.Series:
        return self.assets - self.liabilities

    @property
    def funding_ratio(self) -> pd.Series:
        return self.assets / self.liabilities


def rebalance_dates(rets: pd.DataFrame, first: str) -> list[pd.Timestamp]:
    idx = rets.index[rets.index >= pd.Timestamp(first)]
    return [d for d in idx if d.month == 12]


def realize(
    name: str,
    rets: pd.DataFrame,
    fred_m: pd.DataFrame,
    block0: lb.LiabilityBlock,
    cfg: WFConfig,
    policy,
    reset: bool = False,
) -> Path:
    """Run a strategy on realized history. `policy(t, block_t, a_t)` returns target weights.

    reset=False (PRIMARY, pre-specified, D-061): one block runs off from the
    first rebalance; assets and surplus carry over from year to year.
    reset=True (robustness view, D-062): every December the balance sheet is
    reset to assets a0 against a fresh block with PV l0 on that date's curve,
    so each year's decision is judged on identical starting terms. Year-end
    values are recorded on the OLD block before the reset.
    """
    tenors, par = sc.par_curve_history(fred_m)
    t0 = pd.Timestamp(cfg.first_rebalance)
    months = rets.index[rets.index >= t0]
    rebal = set(rebalance_dates(rets, cfg.first_rebalance))
    A, Lv, W, TO = {}, {}, {}, {}
    holdings = None
    block = block0
    k0 = 0  # month index at which the current block started
    for k, m in enumerate(months):
        years = (k - k0) / 12
        if k > 0:
            holdings = holdings * (1 + rets.loc[m, ASSETS].values)
            pay = lb.payments_between(block, (k - 1 - k0) / 12, years)
            if pay:
                holdings = holdings - pay * holdings / holdings.sum()
        a_now = cfg.a0 if holdings is None else float(holdings.sum())
        l_now = float(lb.value_on_curves(block, tenors, par.loc[m].values[None, :], years)[0])
        A[m], Lv[m] = a_now, l_now
        if m in rebal:
            if reset and k > 0:
                block = lb.annuity_block(fi.bootstrap_zero_curve(tenors, par.loc[m].values), pv=cfg.l0)
                k0, years = k, 0.0
                holdings = cfg.a0 * holdings / holdings.sum()
                a_now = cfg.a0
            target = policy(m, remaining_block(block, years), a_now).reindex(ASSETS).fillna(0.0)
            drifted = np.zeros(len(ASSETS)) if holdings is None else holdings / holdings.sum()
            TO[m] = 0.0 if holdings is None else 0.5 * float(np.abs(target.values - drifted).sum())
            W[m] = target
            holdings = a_now * target.values
    return Path(name, pd.Series(A), pd.Series(Lv), W, TO)


def reset_year_changes(path: Path, cfg: WFConfig) -> pd.Series:
    """For reset=True paths: surplus change over each holding year (each year starts at a0 - l0)."""
    s = path.surplus
    ye = s[s.index.month == 12].iloc[1:]
    out = pd.Series((ye - (cfg.a0 - cfg.l0)).values, index=[str(d.year) for d in ye.index])
    if s.index[-1].month != 12:
        out[f"{s.index[-1].year} (Jan-{s.index[-1]:%b})"] = float(s.iloc[-1] - (cfg.a0 - cfg.l0))
    return out


def summarize(path: Path) -> dict:
    s = path.surplus
    year_end = s[s.index.month == 12]
    annual = year_end.diff().dropna()
    last = s.iloc[-1] - year_end.iloc[-1]  # partial final year
    monthly = s.diff().dropna()
    to = list(path.turnover.values())[1:]  # the first rebalance invests cash; not turnover
    return {
        "strategy": path.name,
        "total_surplus_change": float(s.iloc[-1] - s.iloc[0]),
        "mean_annual_surplus_change": float(annual.mean()),
        "annual_surplus_change_vol": float(annual.std(ddof=1)),
        "monthly_surplus_vol_annualized": float(monthly.std(ddof=1) * np.sqrt(12)),
        "worst_year": int(annual.idxmin().year),
        "worst_year_change": float(annual.min()),
        "surplus_change_2022": float(annual.loc["2022"].iloc[0]),
        "partial_2026_change": float(last),
        "start_funding_ratio": float(path.funding_ratio.iloc[0]),
        "end_funding_ratio": float(path.funding_ratio.iloc[-1]),
        "min_funding_ratio": float(path.funding_ratio.min()),
        "min_funding_ratio_date": str(path.funding_ratio.idxmin().date()),
        "avg_annual_turnover": float(np.mean(to)) if to else 0.0,
    }


def annual_table(paths: list[Path]) -> pd.DataFrame:
    out = {}
    for p in paths:
        s = p.surplus
        ye = s[s.index.month == 12]
        ch = ye.diff().dropna()
        ch.index = ch.index.year.astype(str)
        out[p.name] = ch
        last = s.iloc[-1] - ye.iloc[-1]
        out[p.name].loc[f"{s.index[-1].year} (Jan-{s.index[-1]:%b})"] = last
    return pd.DataFrame(out)

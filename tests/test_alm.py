import numpy as np
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint, minimize

from engine import alm
from engine import liabilities as lb
from engine.optimizer import (
    LiabilityModel,
    PortfolioProblem,
    check_feasible,
    efficient_frontier,
    min_variance,
    surplus_variance,
    target_return,
)


def _ins(s, floor=True, extra=()):
    c = s["ctx"]
    return alm.insurer_problem(c.mu, c.cov, c.liab, s["lcr"], [*s["rules"], *extra], ig_floor=floor)


def _target(s):
    c = s["ctx"]
    return 0.10 * (s["A0"] - c.l0) / s["A0"]


def test_surplus_identity_in_every_scenario(stage3):
    c, a0 = stage3["ctx"], stage3["A0"]
    w = pd.Series(0.0, index=alm.ASSETS)
    w["TLT"] = 1.0  # single asset: monthly rebalancing equals buy-and-hold
    sp = alm.simulate_surplus(w, c.scen, c.L, c.block, a0)
    assert np.array_equal(sp.surplus, sp.assets - sp.liabilities)
    assert np.allclose(sp.funding_ratio, sp.assets / sp.liabilities)
    one_year = alm.surplus_change(w, c.R1, c.rL, a0, c.l0)
    assert np.allclose(sp.surplus[:, 12] - sp.surplus[:, 0], one_year, atol=1e-8)


def test_zero_liability_equals_asset_only(stage3):
    c = stage3["ctx"]
    zero = LiabilityModel(0.0, c.liab.expected_return, c.liab.volatility, c.liab.cov_with_assets)
    a = min_variance(PortfolioProblem(c.mu, c.cov))
    b = min_variance(PortfolioProblem(c.mu, c.cov, liabilities=zero))
    assert np.allclose(a.weights.values, b.weights.values, atol=1e-10)
    t = float(c.mu.median())
    assert np.allclose(
        target_return(PortfolioProblem(c.mu, c.cov), t).weights.values,
        target_return(PortfolioProblem(c.mu, c.cov, liabilities=zero), t).weights.values,
        atol=1e-8,
    )


def test_matched_portfolio_has_near_zero_surplus_change_for_small_shifts(stage3):
    c, s = stage3["ctx"], stage3
    matched = alm.key_rate_matched(s["krd"], s["m"]["krd"], s["A0"], c.l0)
    shocks = {"+10bp": {0.25: 10, 30.0: 10}, "-10bp": {0.25: -10, 30.0: -10}}
    chg = alm.etf_curve_changes(c.fm.rep, c.tenors, c.y0, shocks)
    lch = alm.liability_curve_changes(c.block, c.tenors, c.y0, shocks)
    ss = alm.surplus_shocks({"matched": matched, "short": pd.Series({"SHY": 1.0})}, chg, lch, s["A0"])
    m = ss[ss.portfolio == "matched"].surplus_change.abs()
    short = ss[ss.portfolio == "short"].surplus_change.abs()
    assert (m < 1.0).all()  # under $1m for 10bp on a $1bn book
    assert (m < 0.1 * short.values).all()


def test_constraints_hold_in_every_output(stage3):
    s = stage3
    p = _ins(s)
    outs = [target_return(p, _target(s)).weights, min_variance(p).weights]
    outs.append(alm.min_cvar(p, s["ctx"].R1, s["ctx"].rL, _target(s))[0])
    fr = efficient_frontier(p, n_points=6)
    outs += [row[alm.ASSETS].astype(float) for _, row in fr.iterrows()]
    for w in outs:
        assert check_feasible(w.values, p, tol=1e-6) == []
        assert w["LQD"] >= 0.40 - 1e-6
        assert w[["SPY", "EFA", "EEM"]].sum() <= 0.10 + 1e-6
        assert float(s["lcr"] @ w) >= 1.0 - 1e-6
        cap = alm.capital_charge(w, s["A0"], s["rate_chg"], s["liab_rate"])["total"]
        assert cap <= (s["A0"] - s["ctx"].l0) / alm.COVERAGE_TARGET + 1e-4


def test_capital_charge_is_monotonic_in_risk(stage3):
    s = stage3
    base = pd.Series(0.0, index=alm.ASSETS)
    base["SHY"] = 1.0
    riskier = base.copy()
    riskier["SHY"], riskier["SPY"] = 0.9, 0.1
    c0 = alm.capital_charge(base, s["A0"], s["rate_chg"], s["liab_rate"])
    c1 = alm.capital_charge(riskier, s["A0"], s["rate_chg"], s["liab_rate"])
    assert c1["asset_charge"] - c0["asset_charge"] == pytest.approx(s["A0"] * 0.1 * 0.30)
    matched = alm.key_rate_matched(s["krd"], s["m"]["krd"], s["A0"], s["ctx"].l0)
    gap_m = abs(alm.duration_gap(matched, s["krd"], s["m"]["krd"], s["A0"], s["ctx"].l0)["surplus_dv01"])
    gap_s = abs(alm.duration_gap(base, s["krd"], s["m"]["krd"], s["A0"], s["ctx"].l0)["surplus_dv01"])
    r_m = alm.capital_charge(matched, s["A0"], s["rate_chg"], s["liab_rate"])["rate_charge"]
    assert gap_s > gap_m and c0["rate_charge"] > r_m  # bigger duration gap, bigger rate charge
    fr = efficient_frontier(_ins(s), n_points=6)
    caps = [
        alm.capital_charge(row[alm.ASSETS].astype(float), s["A0"], s["rate_chg"], s["liab_rate"])["total"]
        for _, row in fr.iterrows()
    ]
    assert np.all(np.diff(caps) >= -1e-6)  # more surplus return on the frontier uses more capital


def test_cvar_lp_matches_scenario_cvar(stage3):
    s = stage3
    c = s["ctx"]
    w, cvar, alpha = alm.min_cvar(_ins(s), c.R1, c.rL, _target(s))
    losses = -alm.surplus_change(w, c.R1, c.rL, s["A0"], c.l0) / s["A0"]
    assert alm.cvar_worst_k(losses, 0.95) == pytest.approx(cvar, rel=1e-6)
    # at the optimum, alpha is the 95% VaR: between the two order statistics around the quantile
    srt = np.sort(losses)
    k = int(round(0.05 * len(losses)))
    assert srt[-k - 1] - 1e-9 <= alpha <= srt[-k] + 1e-9


def test_optimizer_cross_checked_by_trust_constr(stage3):
    s = stage3
    p = _ins(s)
    tgt = _target(s)
    res = target_return(p, tgt)
    rows = [np.ones(p.n), p.mu.values] + [r for _, r in p.group_rows()] + [r for _, r in p.rule_rows()]
    lb_ = [1.0, tgt + p.f * p.mu_l] + [g.lower for g, _ in p.group_rows()] + [r.lower for r, _ in p.rule_rows()]
    ub_ = [1.0, tgt + p.f * p.mu_l] + [g.upper for g, _ in p.group_rows()] + [r.upper for r, _ in p.rule_rows()]
    lo, hi = np.array(p.bounds_list()).T
    alt = minimize(
        lambda w: 1e4 * surplus_variance(w, p),
        np.clip(np.full(p.n, 1 / p.n), lo, hi),
        jac=lambda w: 1e4 * (2 * p.cov.values @ w - 2 * p.f * p.c),
        hess=lambda w: 1e4 * 2 * p.cov.values,
        method="trust-constr",
        bounds=list(zip(lo, hi, strict=True)),
        constraints=[LinearConstraint(np.array(rows), lb_, ub_)],
        options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 10000},
    )
    assert surplus_variance(res.weights.values, p) <= surplus_variance(alt.x, p) * (1 + 1e-6)
    assert np.abs(res.weights.values - alt.x).max() < 1e-3


def test_liquidity_coverage_definition(stage3):
    s, c = stage3, stage3["ctx"]
    out = alm.stressed_outflow(c.block, c.l0)
    assert out == pytest.approx(c.block.amounts[0] + lb.SURRENDER_RATE * c.l0)
    moves = alm.liquidity_stress_moves(c.fm.rep, c.tenors, c.y0, True)
    w = pd.Series(0.0, index=alm.ASSETS)
    w["LQD"] = 1.0
    manual = s["A0"] * (1 + moves["LQD"]) * (1 - 0.15) / out
    assert float(alm.lcr_coefficients(moves, s["A0"], out) @ w) == pytest.approx(manual)
    assert moves["SPY"] == pytest.approx(-0.30) and moves["TLT"] < -0.10


def test_duration_band_rule_binds(stage3):
    s, c = stage3, stage3["ctx"]
    band = alm.duration_band_rule(s["krd"], s["m"]["krd"], s["A0"], c.l0, 10.0)
    w = target_return(_ins(s, extra=[band]), _target(s)).weights
    gap = alm.duration_gap(w, s["krd"], s["m"]["krd"], s["A0"], c.l0)["dollar_duration_gap_per_100bp"]
    assert abs(gap) <= 10.0 + 1e-6


def test_stage3_context_is_reproducible(stage3):
    from engine.run_stage3 import build_context

    b = build_context("forward", n_paths=2000)
    assert np.array_equal(b.R1.values, stage3["ctx"].R1.values)
    assert np.array_equal(b.rL, stage3["ctx"].rL)

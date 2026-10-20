import numpy as np
import pandas as pd
import pytest

from engine import alm
from engine import fixedincome as fi
from engine import liabilities as lb
from engine import scenarios as sc
from engine import walkforward as wf

CFG = wf.WFConfig(n_paths=1000)  # fewer paths for speed; same code path as the report


@pytest.fixture(scope="module")
def setup(real_returns, real_fred):
    tenors, par = sc.par_curve_history(real_fred)
    return real_returns, real_fred, tenors, par


def _fresh_block(par, tenors, t):
    return lb.annuity_block(fi.bootstrap_zero_curve(tenors, par.loc[t].values), pv=CFG.l0)


@pytest.mark.parametrize("date", ["2010-12-31", "2016-12-31", "2021-12-31"])
def test_no_look_ahead_truncated_data_gives_identical_weights(setup, date):
    rets, fred, tenors, par = setup
    t = pd.Timestamp(date)
    block = _fresh_block(par, tenors, t)
    full = wf.decide(t, rets, fred, block, CFG.a0, CFG)
    cut = wf.decide(t, rets.loc[:t], fred.loc[:t], block, CFG.a0, CFG)
    assert np.array_equal(full.weights.values, cut.weights.values)
    assert np.array_equal(full.matched.values, cut.matched.values)


def test_future_data_cannot_move_a_decision(setup):
    """Scramble everything after the rebalance date; the decision must not change."""
    rets, fred, tenors, par = setup
    t = pd.Timestamp("2013-12-31")
    block = _fresh_block(par, tenors, t)
    r2, f2 = rets.copy(), fred.copy()
    r2.loc[r2.index > t] = r2.loc[r2.index > t].values[::-1] * 5
    f2.loc[f2.index > t] = f2.loc[f2.index > t].values * 3
    a = wf.decide(t, rets, fred, block, CFG.a0, CFG).weights
    b = wf.decide(t, r2, f2, block, CFG.a0, CFG).weights
    assert np.array_equal(a.values, b.values)


def test_constraints_hold_at_every_rebalance(setup):
    rets, fred, tenors, par = setup
    for t in wf.rebalance_dates(rets, CFG.first_rebalance):
        d = wf.decide(t, rets, fred, _fresh_block(par, tenors, t), CFG.a0, CFG)
        w, det = d.weights, d.details
        assert d.status.startswith(("target", "solver")), d.status
        assert det["violations"] == []
        assert abs(w.sum() - 1) < 1e-6 and (w >= -1e-9).all()
        assert w["LQD"] >= 0.40 - 1e-6
        assert w[["SPY", "EFA", "EEM"]].sum() <= 0.10 + 1e-6
        assert w["HYG"] <= 0.05 + 1e-6
        assert abs(det["gap_per_100bp"]) <= det["band"] + 1e-6
        assert det["capital"] <= det["cap"] + 1e-4
        assert det["lcr"] >= 1.0 - 1e-6


def test_walk_forward_decisions_are_reproducible(setup):
    rets, fred, tenors, par = setup
    t = pd.Timestamp("2019-12-31")
    block = _fresh_block(par, tenors, t)
    a = wf.decide(t, rets, fred, block, CFG.a0, CFG).weights
    b = wf.decide(t, rets, fred, block, CFG.a0, CFG).weights
    assert np.array_equal(a.values, b.values)


def test_remaining_block():
    block = lb.LiabilityBlock(fi.CashFlowStream(np.arange(1.0, 6.0), np.array([10.0, 9, 8, 7, 6])))
    r = wf.remaining_block(block, 2.0)
    assert np.allclose(r.times, [1, 2, 3]) and np.allclose(r.amounts, [8, 7, 6])


def test_realized_accounting(setup):
    """Runoff starts at the initial balance sheet; a single-asset policy has zero turnover;
    in reset mode every holding year starts at surplus a0 - l0."""
    rets, fred, tenors, par = setup
    t0 = pd.Timestamp(CFG.first_rebalance)
    block0 = _fresh_block(par, tenors, t0)
    w = pd.Series(0.0, index=alm.ASSETS)
    w["SHY"] = 1.0
    run = wf.realize("shy", rets, fred, block0, CFG, lambda t, b, a: w)
    assert run.assets.iloc[0] == pytest.approx(CFG.a0)
    assert run.liabilities.iloc[0] == pytest.approx(CFG.l0, rel=1e-12)
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in run.turnover.values())  # single asset never drifts
    rst = wf.realize("shy", rets, fred, block0, CFG, lambda t, b, a: w, reset=True)
    ch = wf.reset_year_changes(rst, CFG)
    # each holding year starts at surplus 100, so the year change is year-end surplus minus 100
    ye = rst.surplus[rst.surplus.index.month == 12].iloc[1:]
    assert np.allclose(ch.iloc[: len(ye)].values, ye.values - (CFG.a0 - CFG.l0))

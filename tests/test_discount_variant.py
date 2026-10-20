import numpy as np
import pytest

from engine import alm
from engine import fixedincome as fi
from engine import liabilities as lb

SPREADS = [0.0, 0.0021, 0.0042, 0.0063, 0.0084]


def test_zero_spread_equals_base_case(stage3):
    c = stage3["ctx"]
    base = lb.measures(c.block, c.tenors, c.y0)
    zero = lb.measures(c.block, c.tenors, c.y0, spread=0.0)
    for k in ("pv", "effective_duration", "effective_convexity", "dv01"):
        assert zero[k] == base[k]
    assert np.array_equal(zero["krd"].values, base["krd"].values)
    L0 = alm.liability_paths(c.block, c.scen, n_steps=12)
    Lz = alm.liability_paths(c.block, c.scen, spread_path=0.0 * c.scen.ig_spread, n_steps=12)
    assert np.array_equal(L0, Lz)
    assert np.array_equal(L0, c.L[:, :13])  # and equal to the Stage 3 liability paths


def test_pv_and_duration_fall_as_the_spread_rises(stage3):
    c = stage3["ctx"]
    ms = [lb.measures(c.block, c.tenors, c.y0, spread=s) for s in SPREADS]
    pv = [m["pv"] for m in ms]
    dur = [m["effective_duration"] for m in ms]
    assert all(np.diff(pv) < 0)
    assert all(np.diff(dur) < 0)
    assert pv[0] == pytest.approx(1000.0, rel=1e-12)


def test_same_curve_consistency(stage3):
    """Scenario valuation at t=0 with a spread equals the analytic value on the
    same Treasury curve plus the same spread; the spread path starts at today's
    ICE IG level and stays positive."""
    c = stage3["ctx"]
    ig0 = c.fm.oas0["LQD"]
    assert np.allclose(c.scen.ig_spread[:, 0], ig0, rtol=1e-12)
    assert (c.scen.ig_spread > 0).all()
    for frac in (0.25, 0.5, 1.0):
        L = alm.liability_paths(c.block, c.scen, spread_path=frac * c.scen.ig_spread, n_steps=0)
        analytic = fi.price_from_curve(c.block.stream, c.fm.curve0, frac * ig0)
        assert np.allclose(L[:, 0], analytic, rtol=1e-12)


def test_key_rate_durations_sum_to_total_with_a_spread(stage3):
    c = stage3["ctx"]
    m = lb.measures(c.block, c.tenors, c.y0, spread=0.0042)
    assert m["krd"].sum() == pytest.approx(m["effective_duration"], rel=1e-5)
    # a 1bp discount-spread move changes PV by about the curve DV01 (both are parallel shifts of the discount rate)
    assert m["spread_dv01"] == pytest.approx(m["dv01"], rel=0.02)


def test_variant_optimization_obeys_constraints(stage3):
    import engine.run_discount_variant as dv

    c = stage3["ctx"]
    out = dv.analyse(c, 0.5)
    for label in ("free", "headline"):
        assert out[f"{label}_status"].startswith("target")
    w = out["_weights"]["headline"]
    assert w["LQD"] >= 0.40 - 1e-6 and w[["SPY", "EFA", "EEM"]].sum() <= 0.10 + 1e-6
    assert abs(w.sum() - 1) < 1e-6 and (w >= -1e-9).all()
    for label in ("free", "headline"):
        assert out[f"{label}_capital"] <= out["surplus"] / alm.COVERAGE_TARGET + 1e-4

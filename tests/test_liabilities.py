import numpy as np
import pandas as pd
import pytest

from engine import fixedincome as fi
from engine import liabilities as lb

TENORS = [0.25, 1, 2, 5, 7, 10, 20, 30]


def test_pv_matches_closed_form_on_flat_curve():
    for y in (0.02, 0.05, 0.08):
        curve = fi.bootstrap_zero_curve(TENORS, [y] * 8)
        block = lb.annuity_block(curve)
        assert fi.price_from_curve(block.stream, curve) == pytest.approx(1000.0, rel=1e-12)
        assert lb.flat_curve_pv(block, y) == pytest.approx(1000.0, rel=1e-12)


def test_block_shape(stage3):
    b = stage3["ctx"].block
    assert len(b.amounts) == 30 and b.times[0] == 1 and b.times[-1] == 30
    assert np.allclose(b.amounts[1:] / b.amounts[:-1], 1 - lb.DECAY)


def test_duration_matches_finite_difference_bump(stage3):
    ctx, m = stage3["ctx"], stage3["m"]
    s = ctx.block.stream
    y, h = m["irr"], 1e-5
    p = fi.price_from_yield(s, y)
    fd = (fi.price_from_yield(s, y - h) - fi.price_from_yield(s, y + h)) / (2 * p * h)
    assert m["modified"] == pytest.approx(fd, rel=1e-7)
    # curve bump: PV moves by about -DV01 per +1bp
    bumped = fi.price_from_curve(s, fi.bootstrap_zero_curve(ctx.tenors, ctx.y0 + 1e-4))
    assert bumped - m["pv"] == pytest.approx(-m["dv01"], rel=2e-3)


def test_key_rate_durations_sum_to_total(stage3):
    m = stage3["m"]
    assert m["krd"].sum() == pytest.approx(m["effective_duration"], rel=1e-5)


def test_assets_and_liabilities_use_the_same_curve(stage3):
    ctx = stage3["ctx"]
    pv_scen = ctx.L[:, 0]
    assert np.allclose(pv_scen, fi.price_from_curve(ctx.block.stream, ctx.fm.curve0), rtol=1e-12)
    assert ctx.l0 == pytest.approx(1000.0, rel=1e-12)


def test_value_after_one_year_and_payments(stage3):
    ctx = stage3["ctx"]
    b = ctx.block
    v1 = lb.value_on_curves(b, ctx.tenors, ctx.y0[None, :], 1.0)[0]
    manual = float(ctx.fm.curve0.df(b.times[1:] - 1.0) @ b.amounts[1:])
    assert v1 == pytest.approx(manual, rel=1e-12)
    assert lb.payments_between(b, 0.0, 1.0) == pytest.approx(b.amounts[0])
    assert lb.payments_between(b, 0.0, 1.0 - 1e-6) == 0.0


def test_historical_return_on_a_flat_unchanging_curve_is_accretion():
    y = 0.05
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    par = pd.DataFrame([[y] * 8] * 4, index=idx, columns=TENORS)
    block = lb.annuity_block(fi.bootstrap_zero_curve(TENORS, [y] * 8))
    r = lb.historical_returns(block, TENORS, par)
    assert np.allclose(r.values, (1 + y / 2) ** (2 / 12) - 1, atol=1e-12)

import numpy as np
import pytest

from engine import fixedincome as fi

TENORS = [0.25, 1, 2, 5, 7, 10, 20, 30]
UPWARD = [0.040, 0.042, 0.044, 0.047, 0.048, 0.050, 0.054, 0.055]
INVERTED = [0.055, 0.052, 0.049, 0.045, 0.044, 0.043, 0.044, 0.043]


@pytest.fixture(params=[UPWARD, INVERTED], ids=["upward", "inverted"])
def par(request):
    return request.param


def test_zero_coupon_price_matches_closed_form():
    y, T = 0.05, 7.5
    assert fi.price_from_yield(fi.Bond(T, 0.0), y) == pytest.approx(100 / (1 + y / 2) ** (2 * T), rel=1e-14)
    assert fi.macaulay_duration(fi.Bond(T, 0.0), y) == pytest.approx(T, rel=1e-14)
    curve = fi.bootstrap_zero_curve(TENORS, UPWARD)
    for t in (0.25, 3.0, 12.5, 30.0):
        z = float(curve.zero_rate(t))
        assert fi.price_from_curve(fi.Bond(t, 0.0), curve) == pytest.approx(100 * np.exp(-z * t), rel=1e-12)


def test_flat_par_curve_gives_flat_zero_curve():
    curve = fi.bootstrap_zero_curve(TENORS, [0.05] * len(TENORS))
    t = np.arange(0.5, 30.01, 0.5)
    assert np.allclose(curve.zero_rate(t, "semiannual"), 0.05, atol=1e-12)


def test_par_bonds_reprice_to_100(par):
    curve = fi.bootstrap_zero_curve(TENORS, par)
    for T in [*TENORS[1:], 0.5, 3.5, 8.5, 12.0, 25.0]:
        c = float(fi.interpolate_par(TENORS, par, T))
        assert fi.price_from_curve(fi.Bond(T, c), curve) == pytest.approx(100.0, abs=1e-9), T


def test_discount_factors_positive_and_decreasing(par):
    curve = fi.bootstrap_zero_curve(TENORS, par)
    d = curve.df(np.linspace(0.01, 35, 500))
    assert np.all(d > 0) and np.all(np.diff(d) < 0)


def test_ytm_round_trip():
    b = fi.Bond(12.0, 0.035)
    for y in (0.01, 0.05, 0.09):
        assert fi.ytm(b, fi.price_from_yield(b, y)) == pytest.approx(y, abs=1e-12)


@pytest.mark.parametrize("T,c,y", [(2, 0.04, 0.045), (10, 0.05, 0.05), (30, 0.02, 0.055), (25, 0.0, 0.05)])
def test_duration_and_convexity_match_finite_differences(T, c, y):
    b = fi.Bond(T, c)
    p = fi.price_from_yield(b, y)
    h = 1e-5
    up, dn = fi.price_from_yield(b, y + h), fi.price_from_yield(b, y - h)
    assert fi.modified_duration(b, y) == pytest.approx((dn - up) / (2 * p * h), rel=1e-7)
    h2 = 1e-4
    up2, dn2 = fi.price_from_yield(b, y + h2), fi.price_from_yield(b, y - h2)
    assert fi.convexity(b, y) == pytest.approx((up2 + dn2 - 2 * p) / (p * h2 * h2), rel=1e-5)
    assert fi.dv01(b, y) == pytest.approx(
        (fi.price_from_yield(b, y - 1e-4) - fi.price_from_yield(b, y + 1e-4)) / 2, rel=1e-4
    )


def test_key_rate_durations_sum_to_effective_duration(par):
    for T in (5.0, 12.0, 25.0):
        b = fi.Bond(T, float(fi.interpolate_par(TENORS, par, T)))
        krd = fi.key_rate_durations(b, TENORS, par)
        eff = fi.effective_duration(b, TENORS, par)
        # central differences with a 1bp bump leave O(h^2) cross terms, about 1e-7 relative
        assert krd.sum() == pytest.approx(eff, rel=1e-5)
        assert (krd >= -1e-9).all()


@pytest.mark.parametrize("T,max_err", [(2, 0.0001), (10, 0.0002), (30, 0.002)])
def test_duration_convexity_vs_full_revaluation_100bp(T, max_err):
    curve = fi.bootstrap_zero_curve(TENORS, UPWARD)
    b = fi.Bond(T, curve.par_yield(T))
    y = fi.ytm(b, fi.price_from_curve(b, curve))
    d, c = fi.modified_duration(b, y), fi.convexity(b, y)
    p0 = fi.price_from_yield(b, y)
    for move in (0.01, -0.01):
        full = fi.price_from_yield(b, y + move) / p0 - 1
        err_dc = abs(float(fi.duration_convexity_change(d, c, move)) - full)
        err_d = abs(-d * move - full)
        assert err_dc < err_d  # convexity always helps
        assert err_dc < max_err  # and the second-order estimate is close

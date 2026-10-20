import numpy as np
import pandas as pd
import pytest

from engine import rates as rt

P = rt.VasicekParams(a=0.15, b=0.03, sigma=0.01, lam=0.3)


@pytest.mark.parametrize("years", [1, 5, 10])
def test_simulated_mean_and_variance_match_analytic(years):
    sim = rt.simulate(P, 0.045, n_paths=100_000, n_steps=years * 12, seed=11)
    x = sim.r[:, -1]
    m, v = float(rt.mean_r(P, 0.045, years)), float(rt.var_r(P, years))
    assert abs(x.mean() - m) < 4 * np.sqrt(v / len(x))
    assert abs(x.var() - v) < 4 * v * np.sqrt(2 / (len(x) - 1))


@pytest.mark.parametrize("years", [1, 5, 10])
def test_martingale_discounted_bond_matches_closed_form(years):
    q = P.risk_neutral()
    sim = rt.simulate(q, 0.045, n_paths=100_000, n_steps=years * 12, seed=12)
    df = sim.discount_factors()[:, -1]
    closed = float(rt.zero_price(q, 0.045, float(years)))
    assert abs(df.mean() - closed) < 4 * df.std() / np.sqrt(len(df))


def test_pricing_measure_uses_risk_premium():
    assert P.b_q == pytest.approx(0.03 + 0.3 * 0.01 / 0.15)
    assert rt.zero_price(P, 0.045, 10.0) == pytest.approx(rt.zero_price(P.risk_neutral(), 0.045, 10.0))
    assert float(rt.zero_yield(P, 0.03, 10.0)) > float(rt.zero_yield(rt.VasicekParams(0.15, 0.03, 0.01), 0.03, 10.0))


def test_zero_price_limits():
    assert float(rt.zero_price(P, 0.04, 1e-8)) == pytest.approx(1.0, abs=1e-9)
    long_yield = float(rt.zero_yield(P, 0.04, 2000.0))
    assert long_yield == pytest.approx(P.b_q - P.sigma**2 / (2 * P.a**2), abs=2e-4)


def test_seeded_simulation_is_reproducible():
    a = rt.simulate(P, 0.04, 500, 24, seed=5)
    b = rt.simulate(P, 0.04, 500, 24, seed=5)
    c = rt.simulate(P, 0.04, 500, 24, seed=6)
    assert np.array_equal(a.r, b.r) and np.array_equal(a.integral, b.integral)
    assert not np.array_equal(a.r, c.r)


def test_calibration_recovers_known_parameters():
    true = rt.VasicekParams(0.5, 0.03, 0.012)
    path = rt.simulate(true, 0.03, n_paths=1, n_steps=6000, seed=3).r[0]
    cal = rt.calibrate(pd.Series(path))
    for k in ("a", "b", "sigma"):
        assert abs(getattr(cal.params, k) - getattr(true, k)) < 3 * cal.se[k], k


def test_calibration_mask_skips_pairs_across_gaps():
    s = pd.Series(rt.simulate(P, 0.04, n_paths=1, n_steps=49, seed=8).r[0])  # 50 months
    mask = pd.Series(True, index=s.index)
    mask.iloc[20:30] = False
    assert rt.calibrate(s, mask).n_pairs == 49 - 11  # pairs touching a dropped month are removed


def test_market_price_of_risk_recovers_target():
    base = rt.VasicekParams(0.15, 0.02, 0.01)
    r = pd.Series([0.01, 0.03, 0.05])
    target_params = rt.VasicekParams(0.15, 0.02, 0.01, lam=0.4)
    z10 = pd.Series(rt.zero_yield(target_params, r.values, 10.0))
    assert rt.fit_market_price_of_risk(base, r, z10) == pytest.approx(0.4, abs=1e-10)

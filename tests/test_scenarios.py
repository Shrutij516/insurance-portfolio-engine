import numpy as np
import pandas as pd
import pytest

from engine import fixedincome as fi
from engine import scenarios, stress


def test_correlated_shocks_match_input_matrix():
    corr = np.array([[1.0, -0.4, 0.2], [-0.4, 1.0, -0.6], [0.2, -0.6, 1.0]])
    z = scenarios.correlated_shocks(corr, 20_000, 24, np.random.default_rng(1))
    est = np.corrcoef(z.reshape(-1, 3).T)
    assert np.allclose(est, corr, atol=0.01)
    assert np.allclose(z.reshape(-1, 3).std(axis=0), 1.0, atol=0.01)


def test_simulated_factor_correlations_and_nonnegative_spreads(factor_model):
    fp = scenarios.simulate_factors(factor_model, 10_000, 60, seed=4)
    est = np.corrcoef(fp.shocks.reshape(-1, 3).T)
    assert np.allclose(est, factor_model.corr.values, atol=0.01)
    assert (fp.spread > 0).all()


def test_spreads_stay_positive_under_extreme_volatility(factor_model):
    fm = factor_model
    wild = dict(fm.spread, sigma=3.0)  # absurdly high volatility
    fm2 = scenarios.FactorModel(**{**fm.__dict__, "spread": wild})
    fp = scenarios.simulate_factors(fm2, 2_000, 60, seed=9)
    assert (fp.spread > 0).all()


def test_factor_simulation_is_reproducible(factor_model):
    a = scenarios.simulate_factors(factor_model, 300, 24, seed=1)
    b = scenarios.simulate_factors(factor_model, 300, 24, seed=1)
    ra, rb = scenarios.asset_returns(factor_model, a, 1), scenarios.asset_returns(factor_model, b, 1)
    assert all(np.array_equal(ra[k], rb[k]) for k in ra)
    assert np.array_equal(a.spread, b.spread)


def test_simulated_curve_starts_on_todays_curve(factor_model):
    fp = scenarios.simulate_factors(factor_model, 50, 3, seed=2)
    taus = np.array([scenarios.REP_MATURITY[n] for n in fp.rep_names])
    assert np.allclose(fp.yields[:, 0, :], factor_model.curve0.zero_rate(taus), atol=1e-12)


def test_yield_based_expected_return_is_yield_minus_default_loss(factor_model):
    fm = factor_model
    er = scenarios.yield_based_expected_returns(fm.curve0, fm.oas0)
    assert er["SHY"] == pytest.approx(fm.curve0.par_yield(2.0), abs=1e-15)
    assert er["TLT"] == pytest.approx(fm.curve0.par_yield(25.0), abs=1e-15)
    for etf, el in scenarios.EXPECTED_LOSS.items():
        expected = fm.curve0.par_yield(scenarios.REP_MATURITY[etf]) + fm.oas0[etf] - el
        assert er[etf] == pytest.approx(expected, abs=1e-15)


def test_empirical_durations_fit_treasuries_and_flag_hyg(real_returns, real_fred):
    emp = scenarios.empirical_durations(real_returns, real_fred)
    r = emp[emp.model == "rates only"].set_index("etf")
    for etf in ("SHY", "IEF", "TLT"):
        assert r.loc[etf, "r2"] >= 0.9
        assert abs(r.loc[etf, "duration"] / scenarios.PUBLISHED_DURATION_APPROX[etf] - 1) < 0.15
    assert r.loc["HYG", "r2"] < 0.1  # rates alone explain almost nothing for high yield
    two = emp[emp.model == "rates + Baa spread"].set_index("etf")
    assert (two["r2"] > r.loc[two.index, "r2"]).all()


def test_rate_beta_makes_used_duration_equal_empirical(factor_model, real_returns, real_fred):
    emp = scenarios.empirical_durations(real_returns, real_fred)
    r = emp[emp.model == "rates only"].set_index("etf")
    for etf in ("SHY", "IEF", "TLT", "TIP"):
        assert factor_model.rep[etf].duration == pytest.approx(r.loc[etf, "duration"], rel=1e-12)
    for etf in ("LQD", "HYG"):
        assert factor_model.rep[etf].rate_beta == 1.0


def test_zero_bound_mask_counts():
    idx = pd.date_range("2007-05-31", "2026-09-30", freq="ME")
    m = scenarios.zero_bound_mask(idx)
    assert (~m).sum() == 84 + 23  # Dec 2008 to Nov 2015, Apr 2020 to Feb 2022


def test_duration_stress_estimates_close_to_full_revaluation(factor_model, real_fred):
    tenors, par = scenarios.par_curve_history(real_fred)
    t = stress.duration_stress(factor_model.rep, tenors, par.iloc[-1].values)
    assert (t.approx_error.abs() < 0.0025).all()  # within 0.25 percentage points
    up = t[(t.shock == "Parallel +100bp")].set_index("etf")
    assert (up.full_revaluation < 0).all()  # every bond fund loses when rates rise
    spread_rows = t[t.type == "spread"]
    assert set(spread_rows.etf) == {"LQD", "HYG"}


def test_par_curve_repricing_on_real_data(real_fred):
    tenors, par = scenarios.par_curve_history(real_fred)
    for row in par.iloc[[0, 50, 120, -1]].values:  # 2007, 2011, 2017, 2026 curves
        curve = fi.bootstrap_zero_curve(tenors, row)
        for T in (1.0, 5.0, 10.0, 30.0):
            c = float(fi.interpolate_par(tenors, row, T))
            assert fi.price_from_curve(fi.Bond(T, c), curve) == pytest.approx(100.0, abs=1e-9)

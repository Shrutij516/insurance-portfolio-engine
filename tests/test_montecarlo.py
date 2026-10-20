import numpy as np
import pandas as pd
import pytest

from engine import data
from engine import estimation as est
from engine.montecarlo import (
    METHODS,
    SimulationConfig,
    SimulationResult,
    max_drawdowns,
    risk_metrics,
    simulate_portfolio,
    simulate_returns,
    var_cvar,
)
from engine.optimizer import LiabilityModel


@pytest.fixture(scope="module")
def syn(synthetic_returns):
    return est.historical_mean(synthetic_returns), est.sample_covariance(synthetic_returns), synthetic_returns


def _cfg(method, **kw):
    base = {"method": method, "n_paths": 2000, "horizon_months": 24}
    base.update(kw)
    return SimulationConfig(**base)


# ------------------------------------------------------- reproducibility


@pytest.mark.parametrize("method", METHODS)
def test_seeded_runs_are_reproducible(syn, method):
    mu, cov, hist = syn
    a = simulate_returns(mu, cov, _cfg(method, seed=7), hist)
    b = simulate_returns(mu, cov, _cfg(method, seed=7), hist)
    c = simulate_returns(mu, cov, _cfg(method, seed=8), hist)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ------------------------------------------------------------ convergence


def _moment_errors(mu, cov, n_paths, method="normal"):
    x = simulate_returns(mu, cov, SimulationConfig(method=method, n_paths=n_paths, horizon_months=12, seed=1))
    x = x.reshape(-1, len(mu))
    mu_m, cov_m = mu.values / 12, cov.values / 12
    mean_err = np.abs(x.mean(axis=0) - mu_m)
    se = np.sqrt(np.diag(cov_m) / len(x))
    cov_err = np.linalg.norm(np.cov(x, rowvar=False) - cov_m) / np.linalg.norm(cov_m)
    return mean_err, se, cov_err


@pytest.mark.parametrize("method", ["normal", "student_t"])
def test_simulated_moments_converge_to_inputs(syn, method):
    mu, cov, _ = syn
    m_small, _, c_small = _moment_errors(mu, cov, 500, method)
    m_big, se_big, c_big = _moment_errors(mu, cov, 100_000, method)
    assert c_big < c_small
    assert m_big.max() < m_small.max()
    assert np.all(m_big < 5 * se_big)  # within 5 standard errors
    assert c_big < 0.02 if method == "normal" else c_big < 0.05


def test_bootstrap_mean_is_shifted_to_expected_returns(syn):
    mu, cov, hist = syn
    x = simulate_returns(mu, cov, SimulationConfig(method="bootstrap", n_paths=20_000, horizon_months=12), hist)
    assert np.allclose(x.reshape(-1, len(mu)).mean(axis=0), mu.values / 12, atol=2e-4)


# --------------------------------------------------------------- VaR/CVaR


def test_var_cvar_known_values():
    gains = -np.arange(100) / 100.0  # 0, -0.01, ..., -0.99
    v, c = var_cvar(gains, 0.95)
    assert v == pytest.approx(0.9405)  # numpy linear quantile at 5%
    assert c == pytest.approx(np.mean(np.arange(95, 100) / 100.0))


@pytest.mark.parametrize("method", METHODS)
def test_var_below_cvar_and_monotone_in_confidence(syn, method):
    mu, cov, hist = syn
    w = pd.Series(0.2, index=mu.index)
    res = simulate_portfolio(w, mu, cov, _cfg(method), hist)
    for h in (12, 24):
        m = risk_metrics(res, h)
        assert m["VaR_95"] <= m["CVaR_95"]
        assert m["VaR_99"] <= m["CVaR_99"]
        assert m["VaR_95"] <= m["VaR_99"]
        assert m["CVaR_95"] <= m["CVaR_99"]
        assert 0 <= m["prob_loss"] <= 1


# ---------------------------------------------- deterministic and surplus


def _with_riskless(mu, cov, rate=0.03):
    names = [*mu.index, "Z"]
    mu2 = pd.Series([*mu.values, rate], index=names)
    cov2 = pd.DataFrame(0.0, index=names, columns=names)
    cov2.loc[mu.index, mu.index] = cov.values
    return mu2, cov2


@pytest.mark.parametrize("method", METHODS)
def test_zero_volatility_asset_gives_deterministic_path(syn, method):
    mu, cov, hist = syn
    mu2, cov2 = _with_riskless(mu, cov)
    hist2 = hist.assign(Z=0.03 / 12)  # constant monthly return in history too
    w = pd.Series(0.0, index=mu2.index)
    w["Z"] = 1.0
    res = simulate_portfolio(w, mu2, cov2, _cfg(method, n_paths=500), hist2)
    expected = (1 + 0.03 / 12) ** np.arange(25)
    assert np.allclose(res.asset_wealth, expected, rtol=0, atol=1e-12)
    assert risk_metrics(res)["p99_max_drawdown"] == 0.0


def test_zero_liability_equals_no_liability(syn):
    mu, cov, hist = syn
    w = pd.Series(0.2, index=mu.index)
    zero = LiabilityModel(0.0, 0.05, 0.1, np.full(len(mu), 0.001))
    a = simulate_portfolio(w, mu, cov, _cfg("normal"), hist)
    b = simulate_portfolio(w, mu, cov, _cfg("normal"), hist, liabilities=zero)
    assert np.array_equal(a.surplus, b.surplus)


def test_liability_identical_to_asset_leaves_surplus_flat(syn):
    """f = 1 and the liability behaves exactly like asset B: holding 100% B
    means assets and liabilities move together, so surplus stays at zero."""
    mu, cov, _ = syn
    k = 1
    liab = LiabilityModel(1.0, float(mu.iloc[k]), float(np.sqrt(cov.values[k, k])), cov.values[:, k])
    w = pd.Series(0.0, index=mu.index)
    w.iloc[k] = 1.0
    res = simulate_portfolio(w, mu, cov, _cfg("normal"), liabilities=liab)
    assert np.abs(res.surplus).max() < 1e-10
    assert res.asset_wealth.std() > 0.01  # assets themselves really do move


def test_bootstrap_with_liabilities_needs_history(syn):
    mu, cov, hist = syn
    liab = LiabilityModel(0.9, 0.03, 0.08, np.zeros(len(mu)))
    with pytest.raises(ValueError, match="history"):
        simulate_portfolio(pd.Series(0.2, index=mu.index), mu, cov, _cfg("bootstrap"), hist, liabilities=liab)


def test_max_drawdown_on_known_path():
    wealth = np.array([[1.0, 1.2, 0.9, 1.1, 0.6]])
    res = SimulationResult(
        SimulationConfig(horizon_months=4, n_paths=1), wealth, np.zeros_like(wealth), np.zeros((1, 4))
    )
    assert max_drawdowns(res)[0] == pytest.approx(0.5)  # (1.2 - 0.6) / 1.2


# ------------------------------------------------ real data: tails matter


def test_normal_understates_tail_vs_bootstrap_for_equity(real_returns):
    a = real_returns[list(data.UNIVERSE)]
    mu, cov = est.bayes_stein_mean(a).mu, est.estimate_covariance(a)
    w = pd.Series(0.0, index=mu.index)
    w["SPY"] = 1.0
    cvar = {}
    for m in ("normal", "bootstrap"):
        res = simulate_portfolio(w, mu, cov, SimulationConfig(method=m, horizon_months=12), a)
        cvar[m] = risk_metrics(res, 12)["CVaR_99"]
    assert cvar["bootstrap"] > cvar["normal"]

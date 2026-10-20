import numpy as np
import pandas as pd
import pytest

from engine import estimation as est
from engine.optimizer import LiabilityModel, PortfolioProblem, max_sharpe, surplus_volatility
from engine.risk import LIABILITY_LABEL, portfolio_volatility, risk_contributions


def test_contributions_sum_to_volatility_synthetic(synthetic_returns):
    cov = est.estimate_covariance(synthetic_returns)
    w = pd.Series([0.4, 0.1, 0.2, 0.2, 0.1], index=cov.index)
    rc = risk_contributions(w, cov)
    vol = np.sqrt(w.values @ cov.values @ w.values)
    assert rc.contribution.sum() == pytest.approx(vol, rel=1e-12)
    assert rc.pct_contribution.sum() == pytest.approx(1.0, rel=1e-12)


def test_contributions_sum_to_volatility_real_optimal(real_inputs):
    mu, cov, rf = real_inputs
    res = max_sharpe(PortfolioProblem(mu, cov, risk_free_rate=rf))
    rc = risk_contributions(res.weights, cov)
    assert rc.contribution.sum() == pytest.approx(res.volatility, rel=1e-10)


def test_marginal_contribution_matches_finite_difference(synthetic_returns):
    cov = est.estimate_covariance(synthetic_returns)
    w = pd.Series([0.3, 0.2, 0.2, 0.2, 0.1], index=cov.index)
    rc = risk_contributions(w, cov)
    h = 1e-7
    for i, a in enumerate(cov.index):
        up = w.copy()
        up.iloc[i] += h
        fd = (portfolio_volatility(up, cov) - portfolio_volatility(w, cov)) / h
        assert rc.marginal[a] == pytest.approx(fd, rel=1e-5)


def test_equal_risk_for_identical_uncorrelated_assets():
    cov = pd.DataFrame(np.eye(4) * 0.04, index=list("ABCD"), columns=list("ABCD"))
    rc = risk_contributions(pd.Series(0.25, index=cov.index), cov)
    assert np.allclose(rc.pct_contribution, 0.25)


def test_liability_appears_as_short_position_and_sum_is_surplus_vol(synthetic_returns):
    cov = est.estimate_covariance(synthetic_returns)
    mu = est.historical_mean(synthetic_returns)
    liab = LiabilityModel(0.8, 0.02, 0.06, cov.values[:, 2] * 0.5)
    w = pd.Series([0.2] * 5, index=cov.index)
    rc = risk_contributions(w, cov, liab)
    assert rc.loc[LIABILITY_LABEL, "weight"] == pytest.approx(-0.8)
    p = PortfolioProblem(mu, cov, liabilities=liab)
    assert rc.contribution.sum() == pytest.approx(surplus_volatility(w.values, p), rel=1e-12)

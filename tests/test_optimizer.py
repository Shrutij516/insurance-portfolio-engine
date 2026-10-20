import numpy as np
import pandas as pd
import pytest

from engine import estimation as est
from engine.optimizer import (
    GroupConstraint,
    LiabilityModel,
    PortfolioProblem,
    check_feasible,
    efficient_frontier,
    max_sharpe,
    min_variance,
    surplus_expected_return,
    surplus_variance,
    target_return,
)
from engine.run_stage1 import ILLUSTRATIVE_BOUNDS, ILLUSTRATIVE_GROUPS

TOL = 1e-6


def _problems(synthetic_returns, real_inputs):
    mu_s = est.historical_mean(synthetic_returns)
    cov_s = est.estimate_covariance(synthetic_returns)
    mu_r, cov_r, rf = real_inputs
    return {
        "synthetic_long_only": PortfolioProblem(mu_s, cov_s),
        "synthetic_grouped": PortfolioProblem(
            mu_s, cov_s, bounds=(0.05, 0.5), groups=[GroupConstraint("D+E cap", ["D", "E"], 0.0, 0.3)]
        ),
        "real_long_only": PortfolioProblem(mu_r, cov_r, risk_free_rate=rf),
        "real_constrained": PortfolioProblem(
            mu_r, cov_r, bounds=ILLUSTRATIVE_BOUNDS, groups=ILLUSTRATIVE_GROUPS, risk_free_rate=rf
        ),
    }


@pytest.fixture(scope="module")
def problems(synthetic_returns, real_inputs):
    return _problems(synthetic_returns, real_inputs)


PROBLEM_NAMES = ["synthetic_long_only", "synthetic_grouped", "real_long_only", "real_constrained"]


# ------------------------------------------- 1. budget and bounds every output


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_every_output_sums_to_one_and_respects_constraints(problems, name):
    p = problems[name]
    outputs = [min_variance(p), max_sharpe(p)]
    frontier = efficient_frontier(p, n_points=12)
    mid = float(frontier.target.iloc[6])
    outputs.append(target_return(p, mid))
    for res in outputs:
        w = res.weights.values
        assert res.success, res.message
        assert abs(w.sum() - 1) < TOL
        assert check_feasible(w, p) == []
    for _, row in frontier.iterrows():
        w = row[p.assets].to_numpy(dtype=float)
        assert row.success
        assert abs(w.sum() - 1) < TOL
        assert check_feasible(w, p) == []


def test_group_constraint_binds_when_it_should(problems):
    p = problems["real_constrained"]
    w = max_sharpe(p).weights
    assert w["HYG"] <= 0.10 + TOL
    assert w[["SPY", "EFA", "EEM", "VNQ"]].sum() <= 0.40 + TOL


# ------------------------------------ 2. closed form: two uncorrelated assets


def test_min_variance_two_uncorrelated_assets_closed_form():
    s1, s2 = 0.10, 0.20
    cov = pd.DataFrame(np.diag([s1**2, s2**2]), index=["A", "B"], columns=["A", "B"])
    mu = pd.Series([0.03, 0.06], index=["A", "B"])
    res = min_variance(PortfolioProblem(mu, cov))
    w_a = s2**2 / (s1**2 + s2**2)  # = 0.8
    assert res.weights["A"] == pytest.approx(w_a, abs=1e-8)
    assert res.volatility**2 == pytest.approx(s1**2 * s2**2 / (s1**2 + s2**2), rel=1e-8)


def test_min_variance_matches_closed_form_when_shorting_allowed(synthetic_returns):
    cov = est.sample_covariance(synthetic_returns)
    mu = est.historical_mean(synthetic_returns)
    res = min_variance(PortfolioProblem(mu, cov, bounds=(-10.0, 10.0)))
    inv1 = np.linalg.solve(cov.values, np.ones(len(cov)))
    assert np.allclose(res.weights.values, inv1 / inv1.sum(), atol=1e-6)


# ----------------------------------------------------- 3. frontier monotonic


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_frontier_monotonic(problems, name):
    f = efficient_frontier(problems[name], n_points=15)
    assert np.all(np.diff(f.expected_return) > 0)
    assert np.all(np.diff(f.volatility) >= -1e-9)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_max_sharpe_beats_every_frontier_point(problems, name):
    p = problems[name]
    best = max_sharpe(p).sharpe
    f = efficient_frontier(p, n_points=30)
    assert best >= f.sharpe.max() - 1e-6


# ------------------------------------- 6. zero liabilities == asset-only case


def test_surplus_moments_reduce_to_asset_moments_with_zero_liabilities(problems):
    p = problems["real_constrained"]
    w = np.full(p.n, 1 / p.n)
    assert surplus_expected_return(w, p) == pytest.approx(w @ p.mu.values, abs=1e-15)
    assert surplus_variance(w, p) == pytest.approx(w @ p.cov.values @ w, abs=1e-15)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_zero_liability_equals_asset_only(problems, name):
    base = problems[name]
    # f = 0 but every other liability input deliberately nonzero: it must not matter.
    zero_liab = LiabilityModel(
        liability_to_asset_ratio=0.0,
        expected_return=0.05,
        volatility=0.12,
        cov_with_assets=np.linspace(0.001, 0.01, base.n),
    )
    with_l = PortfolioProblem(
        base.mu,
        base.cov,
        liabilities=zero_liab,
        bounds=base.bounds,
        groups=base.groups,
        risk_free_rate=base.risk_free_rate,
    )
    for fn in (min_variance, max_sharpe):
        a, b = fn(base), fn(with_l)
        assert np.allclose(a.weights.values, b.weights.values, atol=1e-10)
        assert a.expected_return == pytest.approx(b.expected_return, abs=1e-12)
        assert a.volatility == pytest.approx(b.volatility, abs=1e-12)


# ---------------------------------- nonzero liabilities: the plug-in works


def test_liability_replicated_by_one_asset_is_fully_hedged(synthetic_returns):
    """If the liability behaves exactly like asset B and f = 1, holding 100% B
    makes surplus risk zero, so min surplus variance must pick it."""
    cov = est.sample_covariance(synthetic_returns)
    mu = est.historical_mean(synthetic_returns)
    k = 1  # asset B
    liab = LiabilityModel(
        liability_to_asset_ratio=1.0,
        expected_return=float(mu.iloc[k]),
        volatility=float(np.sqrt(cov.values[k, k])),
        cov_with_assets=cov.values[:, k],
    )
    res = min_variance(PortfolioProblem(mu, cov, liabilities=liab))
    assert res.weights.iloc[k] == pytest.approx(1.0, abs=1e-6)
    assert res.volatility == pytest.approx(0.0, abs=1e-5)
    # The asset-only answer is different, which is the point of surplus optimization.
    assert min_variance(PortfolioProblem(mu, cov)).weights.iloc[k] < 0.9


# ----------------------------------------------------------- error handling


def test_infeasible_constraints_raise(problems):
    p = problems["synthetic_long_only"]
    bad = PortfolioProblem(p.mu, p.cov, bounds=(0.0, 0.1))  # 5 assets x 10% < 100%
    with pytest.raises(ValueError, match="infeasible"):
        min_variance(bad)


def test_max_sharpe_undefined_when_nothing_beats_cash(problems):
    p = problems["synthetic_long_only"]
    with pytest.raises(ValueError, match="risk-free"):
        max_sharpe(PortfolioProblem(p.mu, p.cov, risk_free_rate=1.0))

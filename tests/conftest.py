import numpy as np
import pandas as pd
import pytest

from engine import data
from engine import estimation as est


@pytest.fixture(scope="session")
def synthetic_returns() -> pd.DataFrame:
    """Deterministic fake monthly returns for 5 assets. Used only in tests."""
    rng = np.random.default_rng(42)
    n, t = 5, 240
    vols = np.array([0.01, 0.03, 0.04, 0.05, 0.06])
    a = rng.normal(size=(n, n))
    m = a @ a.T + n * np.eye(n)  # positive definite by construction
    corr = m / np.sqrt(np.outer(np.diag(m), np.diag(m)))
    cov = corr * np.outer(vols, vols)
    means = np.array([0.001, 0.003, 0.005, 0.006, 0.008])
    x = rng.multivariate_normal(means, cov, size=t)
    return pd.DataFrame(x, columns=list("ABCDE"))


@pytest.fixture(scope="session")
def real_returns() -> pd.DataFrame:
    """Monthly returns from the cached Yahoo data. Skips if the cache is absent."""
    if not data.PRICES_CSV.exists():
        pytest.skip("data/prices_daily.csv not cached; run python -m engine.run_stage1 first")
    prices, _ = data.clean_prices(data.load_prices(download_if_missing=False))
    return data.monthly_returns(prices)


@pytest.fixture(scope="session")
def real_inputs(real_returns):
    a = real_returns[list(data.UNIVERSE)]
    rf = float(real_returns[data.RISK_FREE_TICKER].mean() * 12)
    return est.bayes_stein_mean(a).mu, est.estimate_covariance(a), rf


@pytest.fixture(scope="session")
def real_fred(real_returns):
    if not data.FRED_CSV.exists():
        pytest.skip("FRED cache missing; run python -m engine.run_stage2 first")
    return data.monthly_fred(data.load_fred(download_if_missing=False), real_returns.index)


@pytest.fixture(scope="session")
def factor_model(real_returns, real_fred):
    from engine import scenarios

    a = real_returns[list(data.UNIVERSE)]
    return scenarios.build_factor_model(real_returns, real_fred, est.bayes_stein_mean(a).mu)


@pytest.fixture(scope="session")
def stage3(real_fred):
    """Full Stage 3 context with fewer paths for speed (same code path as the report)."""
    from engine import alm
    from engine.run_stage3 import A0, build_context

    ctx = build_context("forward", n_paths=2000)
    krd = alm.etf_key_rate_durations(ctx.fm.rep, ctx.tenors, ctx.y0)
    from engine import liabilities as lb

    m = lb.measures(ctx.block, ctx.tenors, ctx.y0)
    rate_chg = alm.etf_curve_changes(ctx.fm.rep, ctx.tenors, ctx.y0, alm.RATE_SHOCK)
    liab_rate = alm.liability_curve_changes(ctx.block, ctx.tenors, ctx.y0, alm.RATE_SHOCK)
    lcr = alm.lcr_coefficients(
        alm.liquidity_stress_moves(ctx.fm.rep, ctx.tenors, ctx.y0, True), A0, alm.stressed_outflow(ctx.block, ctx.l0)
    )
    rules = alm.capital_rules(A0, (A0 - ctx.l0) / alm.COVERAGE_TARGET, rate_chg, liab_rate)
    return {
        "ctx": ctx,
        "krd": krd,
        "m": m,
        "rate_chg": rate_chg,
        "liab_rate": liab_rate,
        "lcr": lcr,
        "rules": rules,
        "A0": A0,
    }

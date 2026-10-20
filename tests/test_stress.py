import numpy as np
import pandas as pd
import pytest

from engine import data
from engine.stress import (
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SHOCKS,
    apply_shock,
    asset_scenario_returns,
    combined_shock,
    daily_portfolio_path,
    max_drawdown,
)


@pytest.fixture(scope="module")
def real_prices():
    if not data.PRICES_CSV.exists():
        pytest.skip("price cache missing")
    prices, _ = data.clean_prices(data.load_prices(download_if_missing=False))
    return prices


@pytest.fixture(scope="module")
def synthetic_prices():
    idx = pd.bdate_range("2019-12-02", "2020-06-30")
    rng = np.random.default_rng(3)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(idx), 3)), axis=0))
    return pd.DataFrame(px, index=idx, columns=["A", "B", "C"])


@pytest.mark.parametrize("sc", HISTORICAL_SCENARIOS, ids=lambda s: s.name)
def test_one_asset_scenario_return_matches_manual_compounding(real_prices, real_returns, sc):
    w = pd.Series({"SPY": 1.0})
    path = daily_portfolio_path(real_prices, w, sc.start, sc.end)
    manual = float(np.prod(1 + real_returns.loc[sc.start : sc.end, "SPY"].values) - 1)
    # Third route: last price in the window over last price before it.
    before = real_prices.loc[: pd.Period(sc.start, "M").start_time - pd.Timedelta(days=1), "SPY"].iloc[-1]
    after = real_prices.loc[: pd.Period(sc.end, "M").end_time, "SPY"].iloc[-1]
    assert path.iloc[-1] - 1 == pytest.approx(manual, abs=1e-12)
    assert manual == pytest.approx(after / before - 1, abs=1e-12)
    assert float(asset_scenario_returns(real_returns).loc[sc.name, "SPY"]) == pytest.approx(manual, abs=1e-12)


def test_daily_path_month_ends_equal_monthly_rebalanced_portfolio(synthetic_prices):
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    path = daily_portfolio_path(synthetic_prices, w, "2020-01", "2020-06")
    monthly = synthetic_prices.resample("ME").last().pct_change().loc["2020-01":"2020-06"]
    expected = np.cumprod(1 + monthly.values @ w.values)
    got = path.groupby(path.index.to_period("M")).last().iloc[1:].values
    assert np.allclose(got, expected, atol=1e-12)


def test_max_drawdown_simple():
    assert max_drawdown([1.0, 1.5, 0.75, 2.0, 1.8]) == pytest.approx(0.5)
    assert max_drawdown([1.0, 1.1, 1.2]) == 0.0


def test_apply_shock_is_weighted_sum_and_handles_liabilities():
    w = pd.Series({"SPY": 0.6, "TLT": 0.4})
    assert apply_shock(w, {"SPY": -0.30}) == pytest.approx(-0.18)
    # rates-up hits TLT; a liability worth 0.9 that falls 10% offsets it
    assert apply_shock(w, {"TLT": -0.15}, f=0.9, liability_shock=-0.10) == pytest.approx(-0.06 + 0.09)


def test_hypothetical_shocks_are_downside_and_use_known_tickers():
    for name, shock in HYPOTHETICAL_SHOCKS.items():
        assert set(shock) <= set(data.UNIVERSE), name
        assert all(x < 0 for x in shock.values()), name
    assert combined_shock("Equity -30%", "Rates up")["SPY"] == pytest.approx(-0.30)
    assert combined_shock("Credit spreads wider", "Rates up")["LQD"] == pytest.approx(-0.17)

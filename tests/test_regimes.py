import numpy as np
import pandas as pd
import pytest

from engine import data
from engine.regimes import label_regimes, rate_regime, regime_summary


@pytest.fixture(scope="module")
def real_signals():
    if not data.SIGNALS_CSV.exists():
        pytest.skip("signals cache missing; run python -m engine.run_stage1b first")
    return data.load_signals(download_if_missing=False)


@pytest.fixture(scope="module")
def real_labels(real_signals, real_returns):
    return label_regimes(real_signals, real_returns.index)


def test_labels_cover_every_month_and_are_exclusive(real_labels, real_returns):
    assert real_labels.index.equals(real_returns.index)
    assert not real_labels.isna().any().any()
    assert set(real_labels["rates"]) == {"rising", "falling"}
    assert set(real_labels["volatility"]) == {"high", "low"}
    # one label per month per dimension, so the regimes partition the months
    for col in real_labels:
        assert real_labels[col].value_counts().sum() == len(real_returns)


@pytest.mark.parametrize("cutoff", ["2008-06", "2013-03", "2020-01", "2022-06"])
def test_no_look_ahead_truncated_sample_gives_same_labels(real_signals, real_labels, cutoff):
    end = pd.Period(cutoff, "M").end_time
    months = real_labels.index[real_labels.index <= end + pd.offsets.MonthEnd(1)]
    truncated = label_regimes(real_signals.loc[:end], months)
    pd.testing.assert_frame_equal(truncated, real_labels.loc[months])


def test_no_look_ahead_scrambling_future_does_not_move_past(real_signals, real_labels):
    end = pd.Timestamp("2015-06-30")
    scrambled = real_signals.copy()
    future = scrambled.index > end
    scrambled.loc[future] = scrambled.loc[future].values[::-1] * 3  # nonsense future
    past = real_labels.index[real_labels.index <= end + pd.offsets.MonthEnd(1)]
    pd.testing.assert_frame_equal(label_regimes(scrambled, real_labels.index).loc[past], real_labels.loc[past])


def test_label_appears_the_month_after_the_signal():
    days = pd.bdate_range("2019-01-01", "2021-12-31")
    y = pd.Series(2.0, index=days)
    y[days >= "2021-03-01"] = 3.0  # yield jumps during March 2021
    sig = pd.DataFrame({"^TNX": y, "SPY": 100.0})
    months = pd.date_range("2020-02-29", "2021-12-31", freq="ME")
    lab = rate_regime(sig, months)
    assert lab["2021-03-31"] == "falling"  # March's own jump is not known in March
    assert lab["2021-04-30"] == "rising"


def test_2022_is_never_a_falling_rate_regime(real_labels):
    assert (real_labels.loc["2022", "rates"] == "rising").all()


def test_2008_crisis_is_high_volatility(real_labels):
    vol = real_labels["volatility"]
    assert (vol.loc["2008-09":"2009-06"] == "high").all()
    assert (vol.loc["2008"] == "high").mean() >= 2 / 3


def test_regime_summary_months_add_up(real_labels, real_returns):
    a = real_returns[list(data.UNIVERSE)]
    w = pd.Series(1 / len(a.columns), index=a.columns)
    out = regime_summary(a, real_labels["rates"], {"EW": w}, real_returns[data.RISK_FREE_TICKER])
    assert out["summary"]["months"].sum() == len(a)
    assert np.all(out["summary"]["avg_correlation"].between(-1, 1))

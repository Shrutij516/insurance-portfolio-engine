import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from engine import data


def _frame():
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    return pd.DataFrame({"X": np.linspace(100, 110, len(idx)), "Y": np.linspace(50, 60, len(idx))}, index=idx)


def test_clean_trims_to_latest_start_date():
    df = _frame()
    df.loc[:"2020-02-14", "Y"] = np.nan  # Y "launches" later
    cleaned, rep = data.clean_prices(df)
    assert cleaned.index.min() == pd.Timestamp("2020-02-17")
    assert rep.common_start == "2020-02-17"
    assert rep.rows_dropped_before_common_start > 0
    assert not cleaned.isna().any().any()


def test_clean_fills_short_gap_but_rejects_long_gap():
    df = _frame()
    df.iloc[10:13, 0] = np.nan  # 3-day gap: filled
    cleaned, rep = data.clean_prices(df, max_ffill_days=5)
    assert rep.nans_forward_filled["X"] == 3
    assert cleaned.iloc[10:13, 0].eq(cleaned.iloc[9, 0]).all()

    df.iloc[20:40, 1] = np.nan  # 20-day gap: refuse to fake it
    with pytest.raises(ValueError, match="gaps longer"):
        data.clean_prices(df, max_ffill_days=5)


def test_monthly_returns_drop_partial_final_month():
    df = _frame().loc[:"2020-06-15"]  # ends mid-June
    rets = data.monthly_returns(df)
    assert rets.index.max() == pd.Timestamp("2020-05-31")
    assert rets.index.min() == pd.Timestamp("2020-02-29")  # Jan month-end is only the base


def test_cached_data_matches_metadata(real_returns):
    meta = json.loads(data.METADATA_JSON.read_text())
    assert hashlib.sha256(data.PRICES_CSV.read_bytes()).hexdigest() == meta["sha256_of_csv"]
    assert set(meta["tickers"]) == set(data.all_tickers())
    assert not real_returns.isna().any().any()
    assert len(real_returns) >= 170  # roughly 15 years of months

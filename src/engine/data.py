"""Market data: download, cache, and clean adjusted ETF prices.

Design rules (see DECISIONS.md, D-003 to D-006):
- Prices are adjusted closes (dividends and splits folded in), so a price
  change is a total return.
- Data is downloaded once and cached to data/ as CSV. Later runs read the cache.
- If the download fails we raise an error. We never fabricate data silently.
- Different start dates are handled by trimming to the window where every
  ticker has data, and the trim is reported, not hidden.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PRICES_CSV = DATA_DIR / "prices_daily.csv"
METADATA_JSON = DATA_DIR / "prices_metadata.json"
SIGNALS_CSV = DATA_DIR / "signals_daily.csv"
SIGNALS_METADATA_JSON = DATA_DIR / "signals_metadata.json"

# Investable universe: ticker -> (asset class, why it is in the universe).
UNIVERSE: dict[str, str] = {
    "SPY": "US equity (S&P 500)",
    "EFA": "International developed equity",
    "EEM": "Emerging markets equity",
    "SHY": "US Treasuries, short (1-3y)",
    "IEF": "US Treasuries, intermediate (7-10y)",
    "TLT": "US Treasuries, long (20y+)",
    "LQD": "Investment grade corporate credit",
    "HYG": "High yield corporate credit",
    "TIP": "US TIPS (inflation-linked Treasuries)",
    "VNQ": "US real estate (REITs)",
    "GLD": "Gold",
}

# Not investable here. Used only as the cash / risk-free rate proxy for Sharpe.
RISK_FREE_TICKER = "BIL"  # 1-3 month T-bills

# Requested start. HYG (Apr 2007) and BIL (May 2007) launch later, so
# clean_prices() trims to the latest first-valid date. See DECISIONS.md D-020.
DEFAULT_START = "2007-04-01"
DEFAULT_END = "2026-10-01"  # exclusive in yfinance, so last day is 2026-09-30

# Regime signals (not investable). ^TNX is the 10-year Treasury yield in percent.
# SPY is repeated here only to have daily history before the asset panel
# starts, so trailing volatility is fully formed in the first month (D-027).
SIGNAL_TICKERS = ["^TNX", "SPY"]
SIGNALS_START = "2006-01-01"


class DataDownloadError(RuntimeError):
    """Raised when market data cannot be downloaded. Never swallowed."""


@dataclass
class CleaningReport:
    """What the cleaning step did, so it can be logged instead of hidden."""

    first_valid_date: dict[str, str]
    common_start: str
    end: str
    rows_dropped_before_common_start: int
    nans_forward_filled: dict[str, int]
    max_ffill_days: int


def all_tickers() -> list[str]:
    return [*UNIVERSE, RISK_FREE_TICKER]


def download_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Download adjusted daily closes from Yahoo Finance via yfinance.

    Raises DataDownloadError if anything is missing. Does not fall back to
    synthetic data; that requires an explicit human decision.
    """
    import yfinance as yf

    tickers = tickers or all_tickers()
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    except Exception as exc:  # network errors, rate limits, API changes
        raise DataDownloadError(f"yfinance download raised: {exc!r}") from exc

    if raw is None or raw.empty:
        raise DataDownloadError("yfinance returned an empty frame")

    prices = raw["Close"].copy()
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        raise DataDownloadError(f"no data returned for: {missing}")

    prices = prices[tickers]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    return prices


def save_prices(prices: pd.DataFrame, path: Path = PRICES_CSV, meta_path: Path = METADATA_JSON) -> None:
    import yfinance as yf

    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, float_format="%.6f")
    meta = {
        "source": "Yahoo Finance via yfinance, auto_adjust=True (adjusted close)",
        "yfinance_version": yf.__version__,
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tickers": list(prices.columns),
        "first_row": str(prices.index.min().date()),
        "last_row": str(prices.index.max().date()),
        "rows": len(prices),
        "sha256_of_csv": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def load_prices(path: Path = PRICES_CSV, download_if_missing: bool = True) -> pd.DataFrame:
    """Read the cached CSV, downloading it first only if it does not exist."""
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"{path} not found and download_if_missing=False")
        save_prices(download_prices(), path)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def load_signals(path: Path = SIGNALS_CSV, download_if_missing: bool = True) -> pd.DataFrame:
    """Daily regime signals (10y yield, SPY), cached like the prices.

    Kept separate from the asset panel so they never enter returns or the
    optimizer, and so their earlier start date does not affect the trim.
    """
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"{path} not found and download_if_missing=False")
        sig = download_prices(SIGNAL_TICKERS, start=SIGNALS_START, end=DEFAULT_END)
        save_prices(sig, path, SIGNALS_METADATA_JSON)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


# ------------------------------------------------------------------ FRED

FRED_CSV = DATA_DIR / "fred_daily.csv"
FRED_METADATA_JSON = DATA_DIR / "fred_metadata.json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_START = "2006-01-01"

# Treasury constant-maturity par yields, percent. Tenor in years.
TREASURY_SERIES: dict[str, float] = {
    "DGS3MO": 0.25,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}
# Credit spreads, percent. BAA10Y has full history and drives the credit
# factor. The ICE OAS series exist on FRED only from 2023-10-20 (licensing),
# so they are used ONLY for today's spread level (D-034).
CREDIT_FACTOR_SERIES = "BAA10Y"
ICE_IG_OAS, ICE_HY_OAS = "BAMLC0A0CM", "BAMLH0A0HYM2"
FRED_SERIES = [*TREASURY_SERIES, CREDIT_FACTOR_SERIES, ICE_IG_OAS, ICE_HY_OAS]
PARTIAL_HISTORY_OK = {ICE_IG_OAS, ICE_HY_OAS}


def download_fred(series: list[str] | None = None, start: str = FRED_START) -> pd.DataFrame:
    """Daily FRED series via the public fredgraph.csv endpoint (no API key).

    Missing days ("." in FRED files) become NaN. Raises DataDownloadError on
    any failed or empty series; never substitutes data.
    """
    import io
    import urllib.request

    frames = []
    for sid in series or FRED_SERIES:
        try:
            with urllib.request.urlopen(FRED_URL.format(sid), timeout=60) as resp:
                text = resp.read().decode()
        except Exception as exc:
            raise DataDownloadError(f"FRED download failed for {sid}: {exc!r}") from exc
        df = pd.read_csv(io.StringIO(text), na_values=["."])
        if df.shape[1] != 2 or df.iloc[:, 1].dropna().empty:
            raise DataDownloadError(f"FRED returned no data for {sid}")
        df.columns = ["Date", sid]
        frames.append(df.set_index(pd.to_datetime(df["Date"]))[[sid]])
    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "Date"
    last_day = pd.Timestamp(DEFAULT_END) - pd.Timedelta(days=1)  # DEFAULT_END is exclusive
    return out.loc[start:last_day]


def load_fred(path: Path = FRED_CSV, download_if_missing: bool = True) -> pd.DataFrame:
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"{path} not found and download_if_missing=False")
        fred = download_fred()
        path.parent.mkdir(parents=True, exist_ok=True)
        fred.to_csv(path, float_format="%.4f")
        meta = {
            "source": "FRED fredgraph.csv endpoint (St. Louis Fed), daily, percent",
            "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "series": list(fred.columns),
            "first_valid": {c: str(fred[c].first_valid_index().date()) for c in fred},
            "last_row": str(fred.index.max().date()),
            "rows": len(fred),
            "note": "ICE BofA OAS series (BAMLC0A0CM, BAMLH0A0HYM2) only available on FRED from 2023-10-20",
            "sha256_of_csv": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        FRED_METADATA_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def monthly_fred(fred: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Month-end levels in DECIMALS (4.25% -> 0.0425), aligned to the ETF months.

    Uses the last valid daily value in each month (same convention as the ETF
    month-end prices), and includes the month BEFORE the first return month so
    that the first month's change exists. Raises if any full-history series
    has a gap in the window.
    """
    m = fred.resample("ME").last() / 100.0
    window = m.loc[months.min() - pd.offsets.MonthEnd(1) : months.max()]
    full = [c for c in window.columns if c not in PARTIAL_HISTORY_OK]
    gaps = window[full].isna().sum()
    if gaps.any():
        raise ValueError(f"FRED month-end gaps in window: {gaps[gaps > 0].to_dict()}")
    return window


def clean_prices(prices: pd.DataFrame, max_ffill_days: int = 5) -> tuple[pd.DataFrame, CleaningReport]:
    """Trim to the common window and fill only short gaps.

    1. Find each ticker's first valid price. Start the panel at the LATEST of
       these, so every asset has real data for the whole sample. We do not
       backfill or proxy earlier history.
    2. Inside that window, forward fill gaps of at most `max_ffill_days`
       trading days (holiday mismatches, a missing print). Longer gaps raise,
       because carrying a stale price for weeks would fake low volatility.
    """
    first_valid = {c: prices[c].first_valid_index() for c in prices.columns}
    common_start = max(first_valid.values())
    trimmed = prices.loc[common_start:]
    nans_after_trim = trimmed.isna().sum()

    filled = trimmed.ffill(limit=max_ffill_days)
    still_nan = filled.isna().sum()
    if still_nan.any():
        bad = still_nan[still_nan > 0].to_dict()
        raise ValueError(f"gaps longer than {max_ffill_days} days remain: {bad}")

    report = CleaningReport(
        first_valid_date={c: str(d.date()) for c, d in first_valid.items()},
        common_start=str(common_start.date()),
        end=str(filled.index.max().date()),
        rows_dropped_before_common_start=len(prices) - len(trimmed),
        nans_forward_filled={c: int(n) for c, n in nans_after_trim.items()},
        max_ffill_days=max_ffill_days,
    )
    return filled, report


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (not log) month-end to month-end returns.

    Simple returns are used because a portfolio's return is exactly the
    weighted sum of its assets' simple returns. That is not true for log
    returns, and the optimizer relies on it.

    The first month-end price is only the base for the first return, so a
    mid-month start creates no partial month. A mid-month END would, so if
    the last price is not on the last business day of its month, that final
    partial month is dropped.
    """
    month_end = prices.resample("ME").last()
    rets = month_end.pct_change().iloc[1:]
    last_day = prices.index.max()
    if last_day != last_day + pd.offsets.BMonthEnd(0):
        rets = rets.iloc[:-1]
    return rets

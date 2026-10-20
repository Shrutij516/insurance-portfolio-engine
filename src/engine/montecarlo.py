"""Monte Carlo simulation of portfolio and surplus paths.

What it does, in plain language: invent 10,000 possible futures for the
asset returns, month by month, run the portfolio through each one, and look
at the spread of outcomes (how bad the bad cases are), not just the average.

Three ways to generate the monthly returns (DECISIONS.md D-021):
- "normal":    multivariate normal. Simple, but thin tails: it says crashes
               like October 2008 are near impossible.
- "student_t": multivariate Student-t. Fatter tails, and because one shared
               random "stress level" scales all assets in a month, assets
               crash together. Needs a tail parameter (degrees of freedom).
- "bootstrap": DEFAULT. Resample real historical months in blocks of 6, so
               fat tails, crisis correlations and volatility clustering come
               from the data instead of an assumed distribution.

All three use the same expected returns. The bootstrap shifts each asset's
history so its mean equals the chosen expected return (D-022), so the
methods differ only in the SHAPE of outcomes, which is what we compare.

Portfolios are rebalanced to target weights every month (D-023).

Surplus: assets A_t and liabilities L_t are tracked separately, starting at
A_0 = 1 and L_0 = f (liabilities per dollar of assets). Surplus S_t = A_t - L_t.
With no liabilities, S_t = A_t and every metric is the usual wealth metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.estimation import PERIODS_PER_YEAR
from engine.optimizer import LiabilityModel

METHODS = ("normal", "student_t", "bootstrap")


@dataclass(frozen=True)
class SimulationConfig:
    horizon_months: int = 60  # 5 years
    n_paths: int = 10_000
    seed: int = 20261019
    method: str = "bootstrap"
    block_length: int = 6  # bootstrap only, months
    t_df: float = 5.0  # student_t only; lower = fatter tails, must be > 2


DEFAULT_CONFIG = SimulationConfig()


@dataclass
class SimulationResult:
    config: SimulationConfig
    asset_wealth: np.ndarray  # (n_paths, horizon + 1), starts at 1
    liability_value: np.ndarray  # (n_paths, horizon + 1), starts at f (zeros if none)
    portfolio_returns: np.ndarray  # (n_paths, horizon) monthly asset portfolio returns

    @property
    def surplus(self) -> np.ndarray:
        return self.asset_wealth - self.liability_value

    @property
    def f(self) -> float:
        return float(self.liability_value[0, 0])


# ------------------------------------------------------------ return engines


def _matrix_sqrt(cov: np.ndarray) -> np.ndarray:
    """A with A A' = cov, via eigenvalues. Unlike Cholesky, this works when an
    asset has zero variance (singular matrix), which the tests use."""
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    return vecs * np.sqrt(np.clip(vals, 0.0, None))


def _parametric(mu_m: np.ndarray, cov_m: np.ndarray, cfg: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    n = len(mu_m)
    z = rng.standard_normal((cfg.n_paths, cfg.horizon_months, n)) @ _matrix_sqrt(cov_m).T
    if cfg.method == "student_t":
        if cfg.t_df <= 2:
            raise ValueError("t_df must be > 2 for the variance to exist")
        # One chi-square draw per path-month, shared by all assets: a common
        # "stress level". Scaled so the covariance still equals cov_m.
        w = rng.chisquare(cfg.t_df, (cfg.n_paths, cfg.horizon_months, 1))
        z = z * np.sqrt((cfg.t_df - 2) / w)
    return mu_m + z


def _bootstrap(history: np.ndarray, mu_m: np.ndarray, cfg: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap, mean-adjusted to mu_m.

    Pick random starting months, take `block_length` consecutive months from
    each (wrapping from the end of the sample back to the start so every month
    is equally likely), and glue blocks together until the horizon is filled.
    """
    t_hist = history.shape[0]
    b = cfg.block_length
    if not 1 <= b <= t_hist:
        raise ValueError(f"block_length must be between 1 and {t_hist}")
    shifted = history - history.mean(axis=0) + mu_m
    n_blocks = -(-cfg.horizon_months // b)  # ceiling division
    starts = rng.integers(0, t_hist, size=(cfg.n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)) % t_hist
    idx = idx.reshape(cfg.n_paths, n_blocks * b)[:, : cfg.horizon_months]
    return shifted[idx]


def simulate_returns(
    mu: pd.Series,
    cov: pd.DataFrame,
    config: SimulationConfig = DEFAULT_CONFIG,
    history: pd.DataFrame | None = None,
) -> np.ndarray:
    """Simulated MONTHLY simple returns, shape (n_paths, horizon_months, n_assets).

    mu and cov are annualized (as everywhere else in the engine) and are
    converted to monthly here. `history` (monthly returns, same columns) is
    required for the bootstrap.
    """
    if config.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    rng = np.random.default_rng(config.seed)
    mu_m = mu.values / PERIODS_PER_YEAR
    if config.method == "bootstrap":
        if history is None:
            raise ValueError("bootstrap needs the historical monthly returns")
        return _bootstrap(history[list(mu.index)].values, mu_m, config, rng)
    return _parametric(mu_m, cov.values / PERIODS_PER_YEAR, config, rng)


def _augment_with_liability(
    mu: pd.Series, cov: pd.DataFrame, history: pd.DataFrame | None, liab: LiabilityModel
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame | None]:
    """Treat the liability as one more simulated series, so it is drawn jointly
    with the assets (same shocks, same historical months)."""
    name = "__liability__"
    c = np.asarray(liab.cov_with_assets, dtype=float)
    mu2 = pd.concat([mu, pd.Series({name: liab.expected_return})])
    cov2 = cov.copy()
    cov2[name] = c
    cov2.loc[name] = [*c, liab.volatility**2]
    hist2 = None
    if history is not None:
        hist2 = history[list(mu.index)].copy()
        if liab.history is not None:
            if len(liab.history) != len(history):
                raise ValueError("liability history must align with asset history")
            hist2[name] = np.asarray(liab.history, dtype=float)
    return mu2, cov2, hist2


def simulate_portfolio(
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    config: SimulationConfig = DEFAULT_CONFIG,
    history: pd.DataFrame | None = None,
    liabilities: LiabilityModel | None = None,
) -> SimulationResult:
    """Simulate asset, liability and surplus paths for fixed target weights."""
    w = weights.reindex(mu.index).values.astype(float)
    if liabilities is not None and liabilities.liability_to_asset_ratio != 0:
        if config.method == "bootstrap" and liabilities.history is None:
            raise ValueError("bootstrap with liabilities needs LiabilityModel.history (Stage 3)")
        mu2, cov2, hist2 = _augment_with_liability(mu, cov, history, liabilities)
        sims = simulate_returns(mu2, cov2, config, hist2)
        r_assets, r_liab = sims[:, :, :-1], sims[:, :, -1]
        f = float(liabilities.liability_to_asset_ratio)
    else:
        r_assets = simulate_returns(mu, cov, config, history)
        r_liab, f = None, 0.0

    port = r_assets @ w  # monthly rebalanced to target weights
    ones = np.ones((config.n_paths, 1))
    assets = np.hstack([ones, np.cumprod(1 + port, axis=1)])
    if r_liab is None:
        liab = np.zeros_like(assets)
    else:
        liab = f * np.hstack([ones, np.cumprod(1 + r_liab, axis=1)])
    return SimulationResult(config, assets, liab, port)


# ------------------------------------------------------------------ metrics


def max_drawdowns(result: SimulationResult, months: int | None = None) -> np.ndarray:
    """Worst peak-to-trough fall of surplus on each path, as a positive number.

    Measured as (peak surplus - surplus now) / asset value at the peak. With
    no liabilities this is the usual (peak - now) / peak drawdown of wealth.
    """
    h = result.config.horizon_months if months is None else months
    s, a = result.surplus[:, : h + 1], result.asset_wealth[:, : h + 1]
    peak_s, peak_a = s[:, 0].copy(), a[:, 0].copy()
    worst = np.zeros(s.shape[0])
    for t in range(1, h + 1):
        new_peak = s[:, t] > peak_s
        peak_s = np.where(new_peak, s[:, t], peak_s)
        peak_a = np.where(new_peak, a[:, t], peak_a)
        worst = np.maximum(worst, (peak_s - s[:, t]) / peak_a)
    return worst


def var_cvar(gains: np.ndarray, level: float) -> tuple[float, float]:
    """Value at Risk and Conditional VaR (expected shortfall), as positive losses.

    VaR 95%: the loss that is exceeded in only 5% of scenarios.
    CVaR 95%: the AVERAGE loss in that worst 5%. Always >= VaR, and it sees
    how bad the tail is, which VaR ignores.
    """
    q = np.quantile(gains, 1 - level)
    return float(-q), float(-gains[gains <= q].mean())


def risk_metrics(
    result: SimulationResult,
    months: int | None = None,
    target_annual_return: float = 0.04,
    levels: tuple[float, ...] = (0.95, 0.99),
) -> dict[str, float]:
    """Summary of the outcome distribution at `months` (default: full horizon).

    Gains are surplus change per dollar of starting assets, S_h - S_0. With no
    liabilities that is simply the portfolio's cumulative return.
    """
    h = result.config.horizon_months if months is None else months
    years = h / PERIODS_PER_YEAR
    s = result.surplus
    gains = s[:, h] - s[:, 0]
    cum_target = (1 + target_annual_return) ** years - 1
    out: dict[str, float] = {
        "horizon_months": h,
        "mean_terminal_surplus": float(s[:, h].mean()),
        "p01_terminal_surplus": float(np.quantile(s[:, h], 0.01)),
        "p05_terminal_surplus": float(np.quantile(s[:, h], 0.05)),
        "median_terminal_surplus": float(np.median(s[:, h])),
        "p95_terminal_surplus": float(np.quantile(s[:, h], 0.95)),
        "prob_loss": float((gains < 0).mean()),
        "target_annual_return": target_annual_return,
        "prob_miss_target": float((gains < cum_target).mean()),
    }
    for lvl in levels:
        v, c = var_cvar(gains, lvl)
        tag = f"{round(lvl * 100)}"
        out[f"VaR_{tag}"] = v
        out[f"CVaR_{tag}"] = c
    dd = max_drawdowns(result, h)
    out["median_max_drawdown"] = float(np.median(dd))
    out["p95_max_drawdown"] = float(np.quantile(dd, 0.95))
    out["p99_max_drawdown"] = float(np.quantile(dd, 0.99))
    return out

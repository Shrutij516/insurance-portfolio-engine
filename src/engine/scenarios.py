"""Economic scenario generator: rates, credit spreads, equity -> 12 ETF returns.

Factors (DECISIONS.md D-035 to D-039):
1. Short rate: Vasicek (rates.py), calibrated on months outside the two
   zero-bound spells. The yield curve at each date comes from the closed-form
   Vasicek bond prices, plus a constant per-maturity offset so that at time 0
   it equals today's actual curve (Vasicek alone cannot fit today's curve).
2. Credit spread factor: Moody's Baa minus 10y Treasury (the only free credit
   spread with history back to 2007), modeled as a mean-reverting process on
   the LOG of the spread (exact discretization), so it can never go negative.
3. Equity factor: SPY monthly return, normal, mean from Bayes-Stein.
The three shocks are correlated through the Cholesky factor of a correlation
matrix estimated from the data.

Asset returns (monthly), every simplification stated:
- BIL: earns the simulated short rate (exactly, via the integral of r).
- SHY, IEF, TLT, TIP: carry (starting yield / 12) - D dy + 0.5 C dy^2 at the
  fund's representative maturity. No roll-down, no fees, constant duration.
  TIP uses its empirical sensitivity to NOMINAL yields and earns the nominal
  yield (assumes realized inflation = breakeven inflation).
- LQD, HYG: carry (Treasury yield + today's ICE OAS scaled with the Baa
  factor) - expected default loss / 12 - D dy - beta_spread * d(Baa)
  + 0.5 C dy^2. Default loss is a constant drag, not random defaults.
  beta_spread is each fund's measured sensitivity to the Baa factor.
- SPY: the equity factor.
- EFA, EEM, VNQ, GLD: regression on the three factor shocks plus correlated
  residual noise, with means set to Bayes-Stein.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import data
from engine import fixedincome as fi
from engine import rates as rt

# Representative maturity per bond ETF (years), from each fund's mandate.
REP_MATURITY = {"SHY": 2.0, "IEF": 8.5, "TLT": 25.0, "TIP": 7.5, "LQD": 12.0, "HYG": 4.0}
TREASURY_ETFS = ["SHY", "IEF", "TLT"]
CREDIT_ETFS = {"LQD": data.ICE_IG_OAS, "HYG": data.ICE_HY_OAS}
# Rough published effective durations (iShares fact sheets, 2024-2025 range),
# quoted from memory for cross-checking only. NOT fetched; verify before use.
PUBLISHED_DURATION_APPROX = {"SHY": 1.9, "IEF": 7.1, "TLT": 16.3, "TIP": 6.6, "LQD": 8.2, "HYG": 3.0}
# Expected annual credit loss = default probability x loss given default,
# rough long-run rating agency averages (D-038). Judgment, not fetched.
EXPECTED_LOSS = {"LQD": 0.0020, "HYG": 0.0250}
OTHER_ASSETS = ["EFA", "EEM", "VNQ", "GLD"]
ZERO_BOUND_SPELLS = [("2008-12-31", "2015-11-30"), ("2020-04-30", "2022-02-28")]
R2_EMPIRICAL_MIN = 0.9


def zero_bound_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True for months OUTSIDE the zero-bound spells (usable for rate dynamics)."""
    inside = np.zeros(len(index), bool)
    for start, end in ZERO_BOUND_SPELLS:
        inside |= (index >= start) & (index <= end)
    return pd.Series(~inside, index=index)


def par_curve_history(fred_m: pd.DataFrame) -> tuple[list[float], pd.DataFrame]:
    tenors = list(data.TREASURY_SERIES.values())
    return tenors, fred_m[list(data.TREASURY_SERIES)]


def par_yield_at(par: pd.DataFrame, tenors: list[float], maturity: float) -> pd.Series:
    return pd.Series([float(fi.interpolate_par(tenors, row, maturity)) for row in par.values], index=par.index)


# --------------------------------------------------- empirical durations


def _ols(y: np.ndarray, X: np.ndarray):
    X = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return coef, se, r2, e


def empirical_durations(returns: pd.DataFrame, fred_m: pd.DataFrame) -> pd.DataFrame:
    """Regress monthly ETF returns on the change in the par yield at the fund's
    representative maturity (and, for credit, the change in the Baa spread).

    Duration = minus the slope. Two rows per credit fund: rates only, and rates
    plus spread.
    """
    tenors, par = par_curve_history(fred_m)
    d_baa = fred_m[data.CREDIT_FACTOR_SERIES].diff().reindex(returns.index)
    rows = []
    for etf, mat in REP_MATURITY.items():
        dy = par_yield_at(par, tenors, mat).diff().reindex(returns.index)
        y = returns[etf].values
        coef, se, r2, _ = _ols(y, dy.values[:, None])
        rows.append(
            {
                "etf": etf,
                "model": "rates only",
                "duration": -coef[1],
                "duration_se": se[1],
                "spread_beta": np.nan,
                "spread_beta_se": np.nan,
                "r2": r2,
            }
        )
        if etf in CREDIT_ETFS:
            coef, se, r2, _ = _ols(y, np.column_stack([dy.values, d_baa.values]))
            rows.append(
                {
                    "etf": etf,
                    "model": "rates + Baa spread",
                    "duration": -coef[1],
                    "duration_se": se[1],
                    "spread_beta": -coef[2],
                    "spread_beta_se": se[2],
                    "r2": r2,
                }
            )
    out = pd.DataFrame(rows)
    out["published_approx"] = out["etf"].map(PUBLISHED_DURATION_APPROX)
    return out


# ------------------------------------------------ representative bonds


@dataclass(frozen=True)
class RepBond:
    etf: str
    bond: fi.Bond  # par bond at the representative maturity (Treasury + OAS for credit)
    ytm: float
    par_duration: float  # modified duration of that par bond
    par_convexity: float
    rate_beta: float  # multiplier on the par bond's rate sensitivity (1 = none)
    source: str

    @property
    def duration(self) -> float:
        """Rate duration used in the mapping: rate_beta x par-bond duration."""
        return self.rate_beta * self.par_duration

    @property
    def convexity(self) -> float:
        """Approximation: convexity scales with the square of the rate beta."""
        return self.rate_beta**2 * self.par_convexity


def representative_bonds(curve: fi.ZeroCurve, emp: pd.DataFrame, oas_today: dict[str, float]) -> dict[str, RepBond]:
    """One par bond per bond ETF, plus a measured "rate beta" (D-037).

    rate_beta = empirical duration / par-bond duration, used when the data
    measure it well (rates-only R^2 >= 0.9: SHY, IEF, TLT) and always for TIP
    (its sensitivity to NOMINAL yields is what the model needs). TLT's beta
    above 1 reflects its low-coupon bonds trading below par; TIP's below 1
    says only part of a nominal yield move reaches real yields. Credit funds:
    beta 1, par bond at Treasury + today's OAS, because their rates-only fit is
    poor (spreads dominate).
    """
    out = {}
    rates_only = emp[emp.model == "rates only"].set_index("etf")
    for etf, mat in REP_MATURITY.items():
        spread = oas_today.get(etf, 0.0) if etf in CREDIT_ETFS else 0.0
        b = fi.Bond(mat, curve.par_yield(mat) + spread)
        y = fi.ytm(b, fi.price_from_curve(b, curve, spread))
        d, c = fi.modified_duration(b, y), fi.convexity(b, y)
        row = rates_only.loc[etf]
        if etf in CREDIT_ETFS:
            beta, src = 1.0, f"bond math (rates-only R2 {row.r2:.2f} too weak)"
        elif etf == "TIP" or row.r2 >= R2_EMPIRICAL_MIN:
            beta, src = float(row.duration) / d, f"empirical (R2 {row.r2:.2f})"
        else:
            beta, src = 1.0, f"bond math (R2 {row.r2:.2f} too weak)"
        out[etf] = RepBond(etf, b, y, d, c, beta, src)
    return out


# ------------------------------------------------------- factor model


def calibrate_log_ou(spread: pd.Series, dt: float = rt.DT) -> dict[str, float]:
    """Mean-reverting model for log(spread), fitted by AR(1) on monthly logs."""
    x = np.log(spread.values)
    coef, se, _, e = _ols(x[1:], x[:-1, None])
    alpha, beta = coef
    kappa = -np.log(beta) / dt
    theta = alpha / (1 - beta)
    sd = float(np.sqrt(e @ e / (len(e) - 2)))
    sigma = sd * np.sqrt(2 * kappa / (1 - beta**2))
    return {
        "kappa": kappa,
        "kappa_se": float(se[1] / (beta * dt)),
        "theta_log": theta,
        "long_run_median_spread": float(np.exp(theta)),
        "sigma": sigma,
        "ar_beta": beta,
        "resid_sd": sd,
    }


@dataclass
class FactorModel:
    rates: rt.VasicekParams
    rates_cal: rt.Calibration
    spread: dict[str, float]
    equity_mu: float  # annual
    equity_sigma: float  # annual
    corr: pd.DataFrame  # 3x3 over ["rate", "spread", "equity"] shocks
    r0: float
    s0: float  # today's Baa spread
    curve0: fi.ZeroCurve
    oas0: dict[str, float]
    rep: dict[str, RepBond]
    other_betas: pd.DataFrame  # OTHER_ASSETS x 3 shocks (monthly return per 1 sd shock)
    other_resid_cov: pd.DataFrame  # monthly residual covariance
    other_mu: pd.Series  # annual means for OTHER_ASSETS
    spread_betas: dict[str, float]  # credit ETF sensitivity to the Baa factor
    shock_history: pd.DataFrame = field(repr=False, default=None)


def _standardized_shocks(fred_m: pd.DataFrame, returns: pd.DataFrame, cal: rt.Calibration, sp: dict) -> pd.DataFrame:
    """Historical standardized shocks: residuals of the rate AR(1), the log
    spread AR(1), and the equity return, each divided by its model sd."""
    r = fred_m["DGS3MO"]
    rate_res = (r - (cal.ar_alpha + cal.ar_beta * r.shift(1))) / cal.resid_sd
    ls = np.log(fred_m[data.CREDIT_FACTOR_SERIES])
    alpha_s = sp["theta_log"] * (1 - sp["ar_beta"])
    spread_res = (ls - (alpha_s + sp["ar_beta"] * ls.shift(1))) / sp["resid_sd"]
    eq = returns["SPY"]
    eq_res = (eq - eq.mean()) / eq.std()
    return pd.concat({"rate": rate_res, "spread": spread_res, "equity": eq_res}, axis=1).reindex(returns.index)


def build_factor_model(returns: pd.DataFrame, fred_m: pd.DataFrame, mu_bs: pd.Series) -> FactorModel:
    tenors, par = par_curve_history(fred_m)
    mask = zero_bound_mask(fred_m.index)
    cal = rt.calibrate(fred_m["DGS3MO"], mask, label="outside zero-bound spells")
    # market price of risk: match the average 10y zero yield over the sample
    z10 = pd.Series(
        [float(fi.bootstrap_zero_curve(tenors, row).zero_rate(10.0)) for row in par.values], index=par.index
    )
    lam = rt.fit_market_price_of_risk(cal.params, fred_m["DGS3MO"], z10)
    params = rt.VasicekParams(cal.params.a, cal.params.b, cal.params.sigma, lam)
    sp = calibrate_log_ou(fred_m[data.CREDIT_FACTOR_SERIES])
    shocks = _standardized_shocks(fred_m, returns, cal, sp)
    corr = shocks.corr()
    curve0 = fi.bootstrap_zero_curve(tenors, par.iloc[-1].values)
    oas0 = {etf: float(fred_m[s].dropna().iloc[-1]) for etf, s in CREDIT_ETFS.items()}
    emp = empirical_durations(returns, fred_m)
    rep = representative_bonds(curve0, emp, oas0)
    # other assets: regress on the three standardized shocks
    betas, resid = {}, {}
    for a in OTHER_ASSETS:
        coef, _, _, e = _ols(returns[a].values, shocks.values)
        betas[a] = coef[1:]
        resid[a] = e
    other_resid_cov = pd.DataFrame(resid).cov()
    return FactorModel(
        rates=params,
        rates_cal=cal,
        spread=sp,
        equity_mu=float(mu_bs["SPY"]),
        equity_sigma=float(returns["SPY"].std() * np.sqrt(12)),
        corr=corr,
        r0=float(fred_m["DGS3MO"].iloc[-1]),
        s0=float(fred_m[data.CREDIT_FACTOR_SERIES].iloc[-1]),
        curve0=curve0,
        oas0=oas0,
        rep=rep,
        other_betas=pd.DataFrame(betas, index=["rate", "spread", "equity"]).T,
        other_resid_cov=other_resid_cov,
        other_mu=mu_bs[OTHER_ASSETS],
        spread_betas={
            etf: float(emp[(emp.etf == etf) & (emp.model == "rates + Baa spread")].spread_beta.iloc[0])
            for etf in CREDIT_ETFS
        },
        shock_history=shocks,
    )


# ------------------------------------------------------------ simulation


@dataclass
class FactorPaths:
    shocks: np.ndarray  # (n_paths, n_steps, 3) correlated standard normals
    short: rt.ShortRatePaths
    yields: np.ndarray  # (n_paths, n_steps + 1, n_rep) zero yields at rep maturities, shifted to today
    spread: np.ndarray  # (n_paths, n_steps + 1) Baa spread
    equity: np.ndarray  # (n_paths, n_steps) SPY monthly returns
    rep_names: list[str]


def correlated_shocks(corr: np.ndarray, n_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Standard normals with the given correlation, via the Cholesky factor L
    (corr = L L'): independent draws z become correlated draws z L'."""
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_paths, n_steps, corr.shape[0]))
    return z @ L.T


def simulate_factors(fm: FactorModel, n_paths: int = 10_000, n_steps: int = 60, seed: int = 20261019) -> FactorPaths:
    rng = np.random.default_rng(seed)
    z = correlated_shocks(fm.corr.values, n_paths, n_steps, rng)
    short = rt.simulate(fm.rates, fm.r0, n_paths, n_steps, rt.DT, seed=seed + 1, shocks=z[:, :, 0])
    names = list(REP_MATURITY)
    taus = np.array([REP_MATURITY[n] for n in names])
    model_y = rt.model_curve(fm.rates, short.r, taus)
    offset = fm.curve0.zero_rate(taus) - rt.model_curve(fm.rates, np.array(fm.r0), taus)
    yields = model_y + offset  # constant shift: matches today's curve at t = 0
    # log-OU spread, exact discretization
    sp = fm.spread
    beta = np.exp(-sp["kappa"] * rt.DT)
    sd = sp["sigma"] * np.sqrt((1 - beta**2) / (2 * sp["kappa"]))
    ls = np.empty((n_paths, n_steps + 1))
    ls[:, 0] = np.log(fm.s0)
    for t in range(n_steps):
        ls[:, t + 1] = sp["theta_log"] + (ls[:, t] - sp["theta_log"]) * beta + sd * z[:, t, 1]
    mu_m = fm.equity_mu / 12
    sig_m = fm.equity_sigma / np.sqrt(12)
    equity = mu_m + sig_m * z[:, :, 2]
    return FactorPaths(z, short, yields, np.exp(ls), equity, names)


def asset_returns(fm: FactorModel, fp: FactorPaths, seed: int = 20261019) -> dict[str, np.ndarray]:
    """Monthly simple returns per ETF, each (n_paths, n_steps)."""
    dt = rt.DT
    out: dict[str, np.ndarray] = {"BIL": np.exp(fp.short.integral) - 1}
    y_prev, dy = fp.yields[:, :-1, :], np.diff(fp.yields, axis=1)
    col = {n: i for i, n in enumerate(fp.rep_names)}
    d_spread = np.diff(fp.spread, axis=1)
    for etf in [*TREASURY_ETFS, "TIP", *CREDIT_ETFS]:
        rb = fm.rep[etf]
        i = col[etf]
        ret = y_prev[:, :, i] * dt - rb.duration * dy[:, :, i] + 0.5 * rb.convexity * dy[:, :, i] ** 2
        if etf in CREDIT_ETFS:
            oas_prev = fm.oas0[etf] * fp.spread[:, :-1] / fm.s0
            ret = ret + (oas_prev - EXPECTED_LOSS[etf]) * dt - fm.spread_betas[etf] * d_spread
        out[etf] = ret
    out["SPY"] = fp.equity
    rng = np.random.default_rng(seed + 2)
    n_paths, n_steps = fp.equity.shape
    resid = rng.multivariate_normal(np.zeros(len(OTHER_ASSETS)), fm.other_resid_cov.values, size=(n_paths, n_steps))
    for j, a in enumerate(OTHER_ASSETS):
        out[a] = fm.other_mu[a] / 12 + fp.shocks @ fm.other_betas.loc[a].values + resid[:, :, j]
    return out


# ------------------------------------------- yield-based expected returns


def yield_based_expected_returns(curve: fi.ZeroCurve, oas_today: dict[str, float]) -> pd.Series:
    """Expected annual return of each bond ETF = starting yield - expected loss.

    Treasuries and TIP: par yield at the representative maturity (TIP: nominal,
    assuming inflation equals breakeven). Credit: Treasury par yield + today's
    ICE OAS - expected default loss. No fees, no roll-down, yields held
    constant (D-040).
    """
    out = {}
    for etf, mat in REP_MATURITY.items():
        y = curve.par_yield(mat)
        if etf in CREDIT_ETFS:
            y = y + oas_today[etf] - EXPECTED_LOSS[etf]
        out[etf] = y
    return pd.Series(out)

"""One-factor Vasicek short-rate model: calibration, exact simulation, bond prices.

    dr = a (b - r) dt + sigma dW          (real-world, "P" measure)

Plain language: the short rate is pulled back toward a long-run level b at
speed a, and is pushed around by random shocks of size sigma. "Half-life"
ln(2)/a is how long it takes to close half the gap to b.

Pricing ("Q" measure) adds a constant market price of risk lam: the level
used for pricing is b_q = b + lam * sigma / a. Convention here: lam > 0 means
a positive term premium (long bonds yield more than the expected path of
short rates), because pricing then assumes higher future rates.

Known limits, stated plainly (DECISIONS.md D-035):
- Vasicek does NOT fit today's yield curve exactly; it produces one smooth
  shape. Hull-White would fit it.
- Rates are normally distributed, so they CAN go negative. CIR would not.
- One factor: every yield moves in lockstep with the short rate, so the model
  cannot produce independent steepening or flattening.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DT = 1 / 12


@dataclass(frozen=True)
class VasicekParams:
    a: float  # mean-reversion speed, per year
    b: float  # long-run mean (real world), decimal
    sigma: float  # volatility of the short rate, per sqrt(year)
    lam: float = 0.0  # market price of risk (pricing only); > 0 = positive term premium

    @property
    def b_q(self) -> float:
        return self.b + self.lam * self.sigma / self.a

    @property
    def half_life(self) -> float:
        return float(np.log(2) / self.a)

    def risk_neutral(self) -> VasicekParams:
        """Same model, with the drift used for pricing (lam folded into b)."""
        return VasicekParams(self.a, self.b_q, self.sigma, 0.0)


# ------------------------------------------------------- analytic results


def mean_r(p: VasicekParams, r0: float, t: float | np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    return p.b + (r0 - p.b) * np.exp(-p.a * t)


def var_r(p: VasicekParams, t: float | np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    return p.sigma**2 * (1 - np.exp(-2 * p.a * t)) / (2 * p.a)


def _B(a: float, tau: np.ndarray) -> np.ndarray:
    return (1 - np.exp(-a * tau)) / a


def zero_price(p: VasicekParams, r: float | np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    """Closed-form price of $1 paid in tau years (uses the pricing level b_q).

    P = A(tau) exp(-B(tau) r),  B = (1 - e^{-a tau}) / a,
    ln A = (b_q - sigma^2 / (2 a^2)) (B - tau) - sigma^2 B^2 / (4 a)
    """
    tau = np.asarray(tau, float)
    return np.exp(
        -np.asarray(tau) * model_curve(p, np.asarray(r, float), np.atleast_1d(tau)).reshape(np.shape(r) + np.shape(tau))
    )


def zero_yield(p: VasicekParams, r: float | np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    """Continuously compounded zero yield implied by the model (shape r + tau)."""
    tau = np.asarray(tau, float)
    return model_curve(p, np.asarray(r, float), np.atleast_1d(tau)).reshape(np.shape(r) + np.shape(tau))


# ------------------------------------------------------------ calibration


@dataclass
class Calibration:
    params: VasicekParams
    se: dict[str, float]  # standard errors (delta method from the AR(1) regression)
    n_pairs: int
    ar_beta: float
    ar_alpha: float
    resid_sd: float
    label: str

    def table(self) -> dict[str, float]:
        p = self.params
        return {
            "sample": self.label,
            "n_pairs": self.n_pairs,
            "a": p.a,
            "a_se": self.se["a"],
            "half_life_years": p.half_life,
            "b": p.b,
            "b_se": self.se["b"],
            "sigma": p.sigma,
            "sigma_se": self.se["sigma"],
        }


def calibrate(short_rate: pd.Series, mask: pd.Series | None = None, dt: float = DT, label: str = "") -> Calibration:
    """Fit Vasicek by OLS on the exact AR(1) form r_{t+1} = alpha + beta r_t + e.

    beta = e^{-a dt},  alpha = b (1 - beta),  sd(e) = sigma sqrt((1 - beta^2) / (2a)).
    `mask` (True = keep) drops months; only consecutive pairs where BOTH
    months are kept are used, so no fake jump is created across a gap.
    Standard errors: OLS standard errors on alpha, beta, pushed through the
    formulas with the delta method; sigma's uses the usual sd/sqrt(2n) rule.
    """
    x, y = short_rate.values[:-1], short_rate.values[1:]
    keep = np.ones(len(x), bool)
    if mask is not None:
        m = mask.reindex(short_rate.index).values
        keep = m[:-1] & m[1:]
    x, y = x[keep], y[keep]
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    if not 0 < beta < 1:
        raise ValueError(f"AR(1) beta {beta:.4f} outside (0, 1): no mean reversion to calibrate")
    resid = y - X @ coef
    s2 = resid @ resid / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    a = -np.log(beta) / dt
    b = alpha / (1 - beta)
    sd = float(np.sqrt(s2))
    sigma = sd * np.sqrt(2 * a / (1 - beta**2))
    # delta method
    g_a = np.array([0.0, -1 / (beta * dt)])
    g_b = np.array([1 / (1 - beta), alpha / (1 - beta) ** 2])
    se_a = float(np.sqrt(g_a @ cov @ g_a))
    se_b = float(np.sqrt(g_b @ cov @ g_b))
    se_sigma = float(sigma / np.sqrt(2 * n))
    return Calibration(VasicekParams(a, b, sigma), {"a": se_a, "b": se_b, "sigma": se_sigma}, n, beta, alpha, sd, label)


def fit_market_price_of_risk(p: VasicekParams, short_rate: pd.Series, zero_10y: pd.Series, tau: float = 10.0) -> float:
    """Choose lam so the model's AVERAGE 10-year yield (evaluated at each
    month's actual short rate) equals the actual average 10-year zero yield.

    The model yield is linear in b_q, so this has a closed-form answer.
    """
    a, s = p.a, p.sigma
    B = float(_B(a, np.array(tau)))
    # y = -(ln A - B r) / tau, with ln A = (b_q - s^2/2a^2)(B - tau) - s^2 B^2 / 4a
    target = float(zero_10y.mean())
    rbar = float(short_rate.mean())
    # solve target = -[(b_q - k)(B - tau) - s^2 B^2/(4a) - B rbar] / tau for b_q
    k = s**2 / (2 * a**2)
    b_q = k + (-target * tau + s**2 * B**2 / (4 * a) + B * rbar) / (B - tau)
    return float((b_q - p.b) * a / s)


# -------------------------------------------------------------- simulation


@dataclass
class ShortRatePaths:
    r: np.ndarray  # (n_paths, n_steps + 1)
    integral: np.ndarray  # (n_paths, n_steps): integral of r over each step
    dt: float

    def discount_factors(self) -> np.ndarray:
        """exp(-integral of r from 0 to t) at each step end, (n_paths, n_steps)."""
        return np.exp(-np.cumsum(self.integral, axis=1))


def simulate(
    p: VasicekParams,
    r0: float,
    n_paths: int = 10_000,
    n_steps: int = 60,
    dt: float = DT,
    seed: int = 20261019,
    shocks: np.ndarray | None = None,
) -> ShortRatePaths:
    """EXACT simulation of (r, integral of r) over each step (no Euler error).

    Given r_t, the pair (r_{t+dt}, I = integral of r over the step) is jointly
    normal with known mean, variances and covariance, so we draw it directly.
    `shocks` (n_paths, n_steps), standard normal, can be passed in so the rate
    shock is correlated with other factors; otherwise drawn here.
    """
    a, b, s = p.a, p.b, p.sigma
    e1 = np.exp(-a * dt)
    var_rr = s**2 * (1 - np.exp(-2 * a * dt)) / (2 * a)
    var_ii = s**2 / a**2 * (dt - 2 * (1 - e1) / a + (1 - np.exp(-2 * a * dt)) / (2 * a))
    cov_ri = s**2 / (2 * a**2) * (1 - e1) ** 2
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((n_paths, n_steps)) if shocks is None else shocks
    z2 = rng.standard_normal((n_paths, n_steps))
    rho = cov_ri / np.sqrt(var_rr * var_ii)
    r = np.empty((n_paths, n_steps + 1))
    integral = np.empty((n_paths, n_steps))
    r[:, 0] = r0
    for t in range(n_steps):
        rt = r[:, t]
        r[:, t + 1] = b + (rt - b) * e1 + np.sqrt(var_rr) * z1[:, t]
        mean_i = b * dt + (rt - b) * (1 - e1) / a
        integral[:, t] = mean_i + np.sqrt(var_ii) * (rho * z1[:, t] + np.sqrt(1 - rho**2) * z2[:, t])
    return ShortRatePaths(r, integral, dt)


def model_curve(p: VasicekParams, r: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Zero yields (continuous) for each simulated short rate: shape r.shape + (len(taus),)."""
    taus = np.asarray(taus, float)
    a, s = p.a, p.sigma
    B = _B(a, taus)
    ln_a = (p.b_q - s**2 / (2 * a**2)) * (B - taus) - s**2 * B**2 / (4 * a)
    return (-(ln_a) + B * np.asarray(r)[..., None]) / taus

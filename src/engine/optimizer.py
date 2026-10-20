"""Mean-variance optimizer written on a SURPLUS basis.

Surplus = assets - liabilities. Measured per dollar of starting assets, the
surplus return over one period is

    r_S = w' r_A  -  f * r_L

where w are asset weights, r_A asset returns, r_L the return (growth rate) of
the liability value, and f = L0 / A0 (liabilities per dollar of assets, i.e.
1 / funding ratio). This is the Sharpe and Tint (1990) surplus formulation.

    E[r_S]   = w' mu  -  f * mu_L
    Var[r_S] = w' Sigma w  -  2 f w' c  +  f^2 sigma_L^2

where c is the vector of covariances between each asset and the liability.

With no liabilities (f = 0) every formula collapses to the textbook
asset-only version. Stage 3 plugs in a LiabilityModel; nothing in this file
needs to change. See DECISIONS.md D-009 and D-010.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

FEAS_TOL = 1e-6  # max allowed violation of any constraint in a returned result


# ------------------------------------------------------------------ inputs


@dataclass(frozen=True)
class LiabilityModel:
    """Liability inputs, annualized, expressed as returns on liability value.

    Stage 3 will build this from liability cash flows and the rate model
    (liability value moves like a short bond position). For Stage 1 it is
    only a placeholder interface; the default is no liabilities.

    liability_to_asset_ratio: f = L0 / A0. 0.9 means 90 cents of liabilities
        per dollar of assets (funding ratio 111%).
    expected_return: mu_L, expected growth of the liability value.
    volatility: sigma_L, volatility of the liability value's growth.
    cov_with_assets: c_i = Cov(r_asset_i, r_L), one entry per asset, in the
        same order as the asset universe.
    history: optional monthly liability returns aligned with the asset
        return history. Only the bootstrap Monte Carlo needs it (it resamples
        assets and liabilities from the same historical months). Stage 3 will
        build it by repricing liability cash flows on historical yield curves.
    """

    liability_to_asset_ratio: float
    expected_return: float
    volatility: float
    cov_with_assets: Sequence[float]
    history: Sequence[float] | None = None


@dataclass(frozen=True)
class GroupConstraint:
    """Total weight in `assets` must lie in [lower, upper]. E.g. HYG <= 10%."""

    name: str
    assets: Sequence[str]
    lower: float = 0.0
    upper: float = 1.0


@dataclass(frozen=True)
class LinearRule:
    """General linear constraint lower <= sum_i coefs[i] * w_i <= upper.

    Used in Stage 3 for rules whose coefficients are not 0/1, such as the
    liquidity coverage ratio and the capital charge cap. Assets missing from
    `coefs` have coefficient 0.
    """

    name: str
    coefs: dict[str, float]
    lower: float = -np.inf
    upper: float = np.inf


@dataclass
class PortfolioProblem:
    """Everything the optimizer needs.

    bounds: one (low, high) pair applied to every asset, or a dict of
        per-asset pairs. Default (0, 1) means long-only, no leverage.
    Weights always sum to 1 (fully invested budget constraint).
    """

    mu: pd.Series
    cov: pd.DataFrame
    liabilities: LiabilityModel | None = None
    bounds: tuple[float, float] | dict[str, tuple[float, float]] = (0.0, 1.0)
    groups: list[GroupConstraint] = field(default_factory=list)
    rules: list[LinearRule] = field(default_factory=list)
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if list(self.mu.index) != list(self.cov.index) or list(self.cov.index) != list(self.cov.columns):
            raise ValueError("mu and cov must be indexed by the same assets in the same order")
        if self.liabilities is not None and len(self.liabilities.cov_with_assets) != len(self.mu):
            raise ValueError("liability cov_with_assets must have one entry per asset")

    @property
    def assets(self) -> list[str]:
        return list(self.mu.index)

    @property
    def n(self) -> int:
        return len(self.mu)

    # Liability terms; all zero when there are no liabilities.
    @property
    def f(self) -> float:
        return 0.0 if self.liabilities is None else float(self.liabilities.liability_to_asset_ratio)

    @property
    def mu_l(self) -> float:
        return 0.0 if self.liabilities is None else float(self.liabilities.expected_return)

    @property
    def var_l(self) -> float:
        return 0.0 if self.liabilities is None else float(self.liabilities.volatility) ** 2

    @property
    def c(self) -> np.ndarray:
        if self.liabilities is None:
            return np.zeros(self.n)
        return np.asarray(self.liabilities.cov_with_assets, dtype=float)

    def bounds_list(self) -> list[tuple[float, float]]:
        """Effective per-asset bounds: the user's bounds intersected with any
        single-asset group (e.g. "HYG <= 10%" becomes HYG's upper bound).

        Folding these in means the solver never sees a constraint row that
        duplicates a bound (P-008).
        """
        if isinstance(self.bounds, dict):
            raw = [self.bounds.get(a, (0.0, 1.0)) for a in self.assets]
        else:
            raw = [tuple(self.bounds)] * self.n
        eff = {a: (float(lo), float(hi)) for a, (lo, hi) in zip(self.assets, raw, strict=True)}
        for g, _ in self._all_group_rows():
            if len(g.assets) == 1:
                a = g.assets[0]
                lo, hi = eff[a]
                eff[a] = (max(lo, g.lower), min(hi, g.upper))
                if eff[a][0] > eff[a][1]:
                    raise ValueError(f"constraints are infeasible: group {g.name!r} conflicts with bounds for {a}")
        return [eff[a] for a in self.assets]

    def _all_group_rows(self) -> list[tuple[GroupConstraint, np.ndarray]]:
        rows = []
        for g in self.groups:
            unknown = set(g.assets) - set(self.assets)
            if unknown:
                raise ValueError(f"group {g.name!r} has unknown assets {sorted(unknown)}")
            rows.append((g, np.array([1.0 if a in g.assets else 0.0 for a in self.assets])))
        return rows

    def rule_rows(self) -> list[tuple[LinearRule, np.ndarray]]:
        out = []
        for rule in self.rules:
            unknown = set(rule.coefs) - set(self.assets)
            if unknown:
                raise ValueError(f"rule {rule.name!r} has unknown assets {sorted(unknown)}")
            out.append((rule, np.array([float(rule.coefs.get(a, 0.0)) for a in self.assets])))
        return out

    def group_rows(self) -> list[tuple[GroupConstraint, np.ndarray]]:
        """Multi-asset groups only; single-asset groups live in bounds_list()."""
        return [(g, row) for g, row in self._all_group_rows() if len(g.assets) > 1]


# --------------------------------------------------------- surplus moments


def surplus_expected_return(w: np.ndarray, p: PortfolioProblem) -> float:
    return float(w @ p.mu.values - p.f * p.mu_l)


def surplus_variance(w: np.ndarray, p: PortfolioProblem) -> float:
    s = p.cov.values
    return float(w @ s @ w - 2 * p.f * w @ p.c + p.f**2 * p.var_l)


def surplus_volatility(w: np.ndarray, p: PortfolioProblem) -> float:
    return float(np.sqrt(max(surplus_variance(w, p), 0.0)))


def _grad_variance(w: np.ndarray, p: PortfolioProblem) -> np.ndarray:
    return 2 * p.cov.values @ w - 2 * p.f * p.c


# ------------------------------------------------------------------ result


@dataclass
class OptimizationResult:
    weights: pd.Series
    expected_return: float  # surplus expected return (= asset return if no liabilities)
    volatility: float  # surplus volatility (= asset volatility if no liabilities)
    sharpe: float  # (expected_return - rf) / volatility
    objective: str
    success: bool
    message: str


def _result(w: np.ndarray, p: PortfolioProblem, objective: str, success: bool, msg: str) -> OptimizationResult:
    ret = surplus_expected_return(w, p)
    vol = surplus_volatility(w, p)
    sharpe = (ret - p.risk_free_rate) / vol if vol > 0 else float("nan")
    return OptimizationResult(pd.Series(w, index=p.assets), ret, vol, sharpe, objective, success, msg)


def check_feasible(w: np.ndarray, p: PortfolioProblem, tol: float = FEAS_TOL) -> list[str]:
    """Return a list of human-readable constraint violations (empty = feasible)."""
    problems = []
    if abs(w.sum() - 1.0) > tol:
        problems.append(f"weights sum to {w.sum():.8f}, not 1")
    for a, x, (lo, hi) in zip(p.assets, w, p.bounds_list(), strict=True):
        if x < lo - tol or x > hi + tol:
            problems.append(f"{a} weight {x:.6f} outside [{lo}, {hi}]")
    for g, row in p._all_group_rows():  # every group, including single-asset ones
        tot = float(row @ w)
        if tot < g.lower - tol or tot > g.upper + tol:
            problems.append(f"group {g.name} total {tot:.6f} outside [{g.lower}, {g.upper}]")
    for rule, row in p.rule_rows():
        val = float(row @ w)
        scale = max(1.0, float(np.abs(row).max()))
        if val < rule.lower - tol * scale or val > rule.upper + tol * scale:
            problems.append(f"rule {rule.name} value {val:.6f} outside [{rule.lower}, {rule.upper}]")
    return problems


# ------------------------------------------------------------- the solver


# Every constraint in this problem is linear: lb <= row @ w <= ub.
LinearRow = tuple[np.ndarray, float, float]


def _linear_rows(p: PortfolioProblem, extra: list[LinearRow] | None = None) -> list[LinearRow]:
    rows: list[LinearRow] = [(np.ones(p.n), 1.0, 1.0)]  # budget: weights sum to 1
    rows += [(row, g.lower, g.upper) for g, row in p.group_rows()]
    rows += [(row, rule.lower, rule.upper) for rule, row in p.rule_rows()]
    return rows + (extra or [])


def _as_slsqp(rows: list[LinearRow]) -> list[dict]:
    cons = []
    for r, lb, ub in rows:
        if lb == ub:
            cons.append({"type": "eq", "fun": lambda w, r=r, b=lb: r @ w - b, "jac": lambda w, r=r: r})
        else:
            if np.isfinite(ub):
                cons.append({"type": "ineq", "fun": lambda w, r=r, b=ub: b - r @ w, "jac": lambda w, r=r: -r})
            if np.isfinite(lb):
                cons.append({"type": "ineq", "fun": lambda w, r=r, b=lb: r @ w - b, "jac": lambda w, r=r: r})
    return cons


def _lp(p: PortfolioProblem, c: np.ndarray) -> np.ndarray:
    """Solve min c'w over the feasible set (bounds, budget, groups) with an LP."""
    a_ub, b_ub = [], []
    for g, row in p.group_rows():
        a_ub += [row, -row]
        b_ub += [g.upper, -g.lower]
    for rule, row in p.rule_rows():
        if np.isfinite(rule.upper):
            a_ub.append(row)
            b_ub.append(rule.upper)
        if np.isfinite(rule.lower):
            a_ub.append(-row)
            b_ub.append(-rule.lower)
    res = linprog(
        c,
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.ones((1, p.n)),
        b_eq=[1.0],
        bounds=p.bounds_list(),
        method="highs",
    )
    if not res.success:
        raise ValueError(f"constraints are infeasible: {res.message}")
    return res.x


def _feasible_start(p: PortfolioProblem) -> np.ndarray:
    """Equal weights if feasible, otherwise any feasible point from an LP."""
    w = np.full(p.n, 1.0 / p.n)
    return w if not check_feasible(w, p) else _lp(p, np.zeros(p.n))


_SLSQP_PRECISION_STOP = 8  # "Positive directional derivative for linesearch"


def _frank_wolfe_gap(p: PortfolioProblem, rows: list[LinearRow], g: np.ndarray, w: np.ndarray) -> float:
    """g'w minus the minimum of g'x over the feasible set (an LP). >= 0; 0 at an optimum."""
    a_eq, b_eq, a_ub, b_ub = [], [], [], []
    for r, lb, ub in rows:
        if lb == ub:
            a_eq.append(r)
            b_eq.append(lb)
            continue
        if np.isfinite(ub):
            a_ub.append(r)
            b_ub.append(ub)
        if np.isfinite(lb):
            a_ub.append(-r)
            b_ub.append(-lb)
    res = linprog(
        g,
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(a_eq) if a_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=p.bounds_list(),
        method="highs",
    )
    return float(g @ w - res.fun) if res.success else float("inf")


def _solve(
    p: PortfolioProblem, fun, jac, extra_rows: list[LinearRow] | None = None, x0=None
) -> tuple[np.ndarray, bool, str]:
    rows = _linear_rows(p, extra_rows)
    res = minimize(
        fun,
        _feasible_start(p) if x0 is None else x0,
        jac=jac,
        method="SLSQP",
        bounds=p.bounds_list(),
        constraints=_as_slsqp(rows),
        options={"ftol": 1e-12, "maxiter": 2000},  # see DECISIONS.md P-004
    )
    lo, hi = np.array(p.bounds_list()).T
    # Remove solver noise of order 1e-12 outside the bounds, then re-check.
    w = np.clip(res.x, lo, hi)
    ok, msg = bool(res.success), str(res.message)

    # SLSQP sometimes stops at the optimum with status 8 because it cannot
    # improve further in double precision (P-005). Do not just trust or just
    # reject it: prove optimality with the Frank-Wolfe gap (P-026). Linearize
    # the objective at w and let an LP find the best feasible point for that
    # linear objective. For a convex problem, w is optimal exactly when the LP
    # cannot beat it; for max Sharpe this is the first-order (KKT) condition.
    # Unlike the earlier trust-constr re-solve (P-006), this works at corner
    # solutions where many constraints bind at once.
    if not ok and res.status == _SLSQP_PRECISION_STOP and not check_feasible(w, p):
        f_w = float(fun(w))
        gap = _frank_wolfe_gap(p, rows, np.asarray(jac(w), float), w)
        if gap <= 1e-6 * max(1.0, abs(f_w)):
            ok, msg = True, f"SLSQP precision stop; optimality confirmed (Frank-Wolfe gap {gap:.1e})"
        else:
            msg = f"SLSQP precision stop; Frank-Wolfe gap {gap:.3e} too large: not confirmed optimal"

    violations = check_feasible(w, p)
    if violations:
        ok, msg = False, f"{msg}; violations: {violations}"
    return w, ok, msg


# Variance values are ~1e-3 to 1e-2. Scaling the objective up keeps SLSQP's
# convergence test meaningful; it does not change the solution.
_SCALE = 1e4


def min_variance(p: PortfolioProblem) -> OptimizationResult:
    """Minimum surplus variance portfolio (ignores expected returns)."""
    w, ok, msg = _solve(
        p,
        lambda w: _SCALE * surplus_variance(w, p),
        lambda w: _SCALE * _grad_variance(w, p),
    )
    return _result(w, p, "min_variance", ok, msg)


def max_return_weights(p: PortfolioProblem) -> np.ndarray:
    """Highest expected return achievable under the constraints (an LP)."""
    return _lp(p, -p.mu.values)


def target_return(p: PortfolioProblem, target: float) -> OptimizationResult:
    """Minimum surplus variance subject to surplus expected return == target."""
    # w' mu - f mu_L == target, written as a linear row on w.
    target_row = (p.mu.values, target + p.f * p.mu_l, target + p.f * p.mu_l)
    w, ok, msg = _solve(
        p,
        lambda w: _SCALE * surplus_variance(w, p),
        lambda w: _SCALE * _grad_variance(w, p),
        extra_rows=[target_row],
    )
    if abs(surplus_expected_return(w, p) - target) > FEAS_TOL:
        ok, msg = False, f"{msg}; target return {target:.4f} not met"
    return _result(w, p, f"target_return({target:.4f})", ok, msg)


def max_sharpe(p: PortfolioProblem) -> OptimizationResult:
    """Maximize (E[r_S] - rf) / vol(r_S).

    On a convex feasible set where some portfolio beats rf, the Sharpe ratio
    is quasi-concave, so a local maximum is the global one. We still verify
    against the efficient frontier in the tests.
    """
    mu = p.mu.values
    w_max = max_return_weights(p)
    if surplus_expected_return(w_max, p) <= p.risk_free_rate:
        raise ValueError("no feasible portfolio beats the risk-free rate; max Sharpe undefined")

    def neg_sharpe(w):
        return -(surplus_expected_return(w, p) - p.risk_free_rate) / surplus_volatility(w, p)

    def grad(w):
        vol = surplus_volatility(w, p)
        ex = surplus_expected_return(w, p) - p.risk_free_rate
        dvol = _grad_variance(w, p) / (2 * vol)
        return -(mu / vol - ex * dvol / vol**2)

    # Start from the max-return corner, which has positive excess return.
    w, ok, msg = _solve(p, neg_sharpe, grad, x0=w_max)
    return _result(w, p, "max_sharpe", ok, msg)


def efficient_frontier(p: PortfolioProblem, n_points: int = 25) -> pd.DataFrame:
    """Trace the frontier from the min-variance portfolio to the max-return one.

    Returns one row per point: target, expected_return, volatility, sharpe,
    success, then one column per asset weight.
    """
    gmv = min_variance(p)
    w_max = max_return_weights(p)
    # The max-return end is a corner (often 100% in one asset) found exactly by
    # the LP. Re-solving it as a QP is degenerate: more active constraints than
    # free dimensions, which made trust-constr warn (P-008). Use the LP answer.
    # Caveat: if two portfolios tie for max return, the LP picks one of them,
    # not necessarily the lower-risk one.
    top = _result(w_max, p, "max_return", not check_feasible(w_max, p), "exact LP solution")
    r_lo, r_hi = gmv.expected_return, top.expected_return
    rows = []
    for i, tgt in enumerate(np.linspace(r_lo, r_hi, n_points)):
        res = gmv if i == 0 else top if i == n_points - 1 else target_return(p, float(tgt))
        rows.append(
            {
                "target": float(tgt),
                "expected_return": res.expected_return,
                "volatility": res.volatility,
                "sharpe": res.sharpe,
                "success": res.success,
                **res.weights.to_dict(),
            }
        )
    return pd.DataFrame(rows)

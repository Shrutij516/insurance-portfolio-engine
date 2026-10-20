"""Asset-liability management: surplus scenarios, matching, liquidity, capital.

Everything is in $ millions. A0 = assets at the start, L0 = liabilities
(present value on today's curve), S0 = A0 - L0.

Key identity used by the optimizer (D-048): over one year with buy-and-hold
asset returns R_i and liability return r_L = (L_1 + payments) / L_0 - 1,
    surplus change / A0 = w'R - f r_L,   f = L0 / A0,
which is EXACTLY the Stage 1 surplus formula. So the Stage 1 optimizer is
reused unchanged, fed with scenario-based moments.

The capital charge here is a PROXY (D-050). It is NOT NAIC RBC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from engine import data, stress
from engine import fixedincome as fi
from engine import liabilities as lb
from engine.esg import ESGScenarios
from engine.optimizer import GroupConstraint, LiabilityModel, LinearRule, PortfolioProblem

ASSETS = list(data.UNIVERSE)
BOND_ETFS = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG"]
TREASURY_MATCH_ETFS = ["SHY", "IEF", "TLT"]

# ------------------------------------------------------- surplus paths


@dataclass
class SurplusPaths:
    assets: np.ndarray  # (n_paths, n_steps + 1)
    liabilities: np.ndarray  # (n_paths, n_steps + 1)

    @property
    def surplus(self) -> np.ndarray:
        return self.assets - self.liabilities

    @property
    def funding_ratio(self) -> np.ndarray:
        return self.assets / self.liabilities


def liability_paths(
    block: lb.LiabilityBlock, esg: ESGScenarios, spread_path: np.ndarray | None = None, n_steps: int | None = None
) -> np.ndarray:
    """Liability value at every simulated month end (after that month's payment).

    spread_path: optional (n_paths, n_steps + 1) discount add-on per path and
    month (D-063/D-064); None = Treasury discounting. n_steps: value only the
    first n_steps months (default: all).
    """
    n_paths = esg.par.shape[0]
    steps = esg.par.shape[1] - 1 if n_steps is None else n_steps
    out = np.empty((n_paths, steps + 1))
    for t in range(steps + 1):
        sp = 0.0 if spread_path is None else spread_path[:, t]
        out[:, t] = lb.value_on_curves(block, esg.tenors, esg.par[:, t, :], t / 12, spread=sp)
    return out


def simulate_surplus(
    weights: pd.Series, esg: ESGScenarios, L: np.ndarray, block: lb.LiabilityBlock, a0: float
) -> SurplusPaths:
    """Assets rebalanced monthly to the weights; liability payments are paid
    out of assets at each year end."""
    w = weights.reindex(ASSETS).fillna(0.0).values
    r = np.stack([esg.returns[a] for a in ASSETS], axis=-1) @ w  # (paths, steps)
    n_paths, n_steps = r.shape
    A = np.empty((n_paths, n_steps + 1))
    A[:, 0] = a0
    for t in range(1, n_steps + 1):
        pay = lb.payments_between(block, (t - 1) / 12, t / 12)
        A[:, t] = A[:, t - 1] * (1 + r[:, t - 1]) - pay
    return SurplusPaths(A, L)


def one_year_inputs(esg: ESGScenarios, L: np.ndarray, block: lb.LiabilityBlock) -> tuple[pd.DataFrame, np.ndarray]:
    """Buy-and-hold 1-year asset returns per path, and the liability return."""
    R1 = pd.DataFrame({a: np.prod(1 + esg.returns[a][:, :12], axis=1) - 1 for a in ASSETS})
    rL = (L[:, 12] + lb.payments_between(block, 0.0, 1.0)) / L[:, 0] - 1
    return R1, rL


def scenario_moments(R1: pd.DataFrame, rL: np.ndarray, f: float) -> tuple[pd.Series, pd.DataFrame, LiabilityModel]:
    mu = R1.mean()
    cov = R1.cov()
    c = np.array([np.cov(R1[a].values, rL)[0, 1] for a in ASSETS])
    liab = LiabilityModel(f, float(rL.mean()), float(rL.std(ddof=1)), c)
    return mu, cov, liab


def surplus_change(weights: pd.Series, R1: pd.DataFrame, rL: np.ndarray, a0: float, l0: float) -> np.ndarray:
    """1-year surplus change per path in $m (buy-and-hold): A0 w'R - L0 r_L."""
    return a0 * (R1[ASSETS].values @ weights.reindex(ASSETS).fillna(0.0).values) - l0 * rL


def cvar_worst_k(losses: np.ndarray, level: float) -> float:
    """Mean of the worst (1 - level) share of losses; with N(1 - level) an
    integer this equals the Rockafellar-Uryasev CVaR exactly."""
    k = int(round(len(losses) * (1 - level)))
    return float(np.sort(losses)[-k:].mean())


# --------------------------------------------------------------- matching


def etf_key_rate_durations(rep: dict, par_tenors, par_yields) -> pd.DataFrame:
    """KRDs per ETF (rate beta applied); non-bond assets have zero."""
    tenors = np.asarray(par_tenors, float)
    curve = fi.bootstrap_zero_curve(tenors, par_yields)
    rows = {}
    for a in ASSETS:
        if a in rep:
            rb = rep[a]
            spread = rb.bond.coupon - curve.par_yield(rb.bond.maturity)
            rows[a] = rb.rate_beta * fi.key_rate_durations(rb.bond, tenors, par_yields, spread=spread).values
        else:
            rows[a] = np.zeros(len(tenors))
    return pd.DataFrame(rows, index=tenors).T


def key_rate_matched(
    etf_krd: pd.DataFrame, liab_krd: pd.Series, a0: float, l0: float, universe: list[str] = TREASURY_MATCH_ETFS
) -> pd.Series:
    """Weights (summing to 1, long-only, within `universe`) that minimize the
    squared gap between asset and liability dollar key rate durations."""
    from scipy.optimize import minimize

    K = etf_krd.loc[universe].values.T  # (tenors, n)
    target = l0 * liab_krd.values / a0

    def obj(w):
        g = K @ w - target
        return float(g @ g)

    def jac(w):
        return 2 * K.T @ (K @ w - target)

    n = len(universe)
    res = minimize(
        obj,
        np.full(n, 1 / n),
        jac=jac,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    return pd.Series(np.clip(res.x, 0, None), index=universe).reindex(ASSETS).fillna(0.0)


def duration_gap(weights: pd.Series, etf_krd: pd.DataFrame, liab_krd: pd.Series, a0: float, l0: float) -> dict:
    d_a = float(weights.reindex(ASSETS).fillna(0.0) @ etf_krd.sum(axis=1))
    d_l = float(liab_krd.sum())
    dollar_gap = a0 * d_a - l0 * d_l  # $m per 1.00 rate change
    return {
        "asset_duration": d_a,
        "liability_duration": d_l,
        "dollar_duration_gap_per_100bp": dollar_gap / 100,
        "surplus_dv01": dollar_gap * fi.BP,
    }


def etf_curve_changes(rep: dict, par_tenors, par_yields, curve_shocks: dict) -> pd.DataFrame:
    """Full-revaluation % change of each bond ETF for each curve shock (others 0)."""
    t = stress.duration_stress(rep, par_tenors, par_yields, curve_shocks=curve_shocks, spread_shocks={})
    chg = t.pivot(index="shock", columns="etf", values="full_revaluation")
    return chg.reindex(columns=ASSETS).fillna(0.0).loc[list(curve_shocks)]


def liability_curve_changes(
    block: lb.LiabilityBlock, par_tenors, par_yields, curve_shocks: dict, spread: float = 0.0
) -> pd.Series:
    """Liability value change for each Treasury curve shock (discount spread held fixed)."""
    base = fi.price_from_curve(block.stream, fi.bootstrap_zero_curve(par_tenors, par_yields), spread)
    out = {}
    for name, shock in curve_shocks.items():
        bump = stress.curve_shock_at(shock, par_tenors)
        shocked = fi.bootstrap_zero_curve(par_tenors, np.asarray(par_yields) + bump)
        out[name] = fi.price_from_curve(block.stream, shocked, spread) - base
    return pd.Series(out)


def surplus_shocks(
    portfolios: dict[str, pd.Series], etf_chg: pd.DataFrame, liab_chg: pd.Series, a0: float
) -> pd.DataFrame:
    """Surplus change in $m: A0 sum w_i change_i - liability change."""
    rows = []
    for name, w in portfolios.items():
        for shock in etf_chg.index:
            d_a = a0 * float(etf_chg.loc[shock] @ w.reindex(ASSETS).fillna(0.0))
            d_l = float(liab_chg[shock])
            rows.append(
                {
                    "portfolio": name,
                    "shock": shock,
                    "asset_change": d_a,
                    "liability_change": d_l,
                    "surplus_change": d_a - d_l,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------- liquidity

# (bucket, haircut) per ETF: LABELED ASSUMPTIONS inspired by Basel III LCR
# haircuts (Level 1: 0%, Level 2A: 15%, Level 2B: 50%), applied AFTER the
# stress price move. Insurers have no single standard (D-049).
LIQUIDITY = {
    "SHY": ("Level 1: Treasuries", 0.0),
    "IEF": ("Level 1: Treasuries", 0.0),
    "TLT": ("Level 1: Treasuries", 0.0),
    "TIP": ("Level 1: Treasuries", 0.0),
    "LQD": ("Level 2A: investment grade credit", 0.15),
    "HYG": ("Level 2B: high yield", 0.50),
    "SPY": ("Level 2B: equities", 0.50),
    "EFA": ("Level 2B: equities", 0.50),
    "EEM": ("Level 2B: equities", 0.50),
    "VNQ": ("Level 2B: real estate", 0.50),
    "GLD": ("Level 2B: gold", 0.50),
}


def liquidity_stress_moves(rep: dict, par_tenors, par_yields, severe: bool) -> pd.Series:
    """Price move per ETF: rates +100bp plus IG +100bp and HY +400bp spreads;
    'severe' also applies the Stage 1b equity -30% factor shock."""
    t = stress.duration_stress(
        rep, par_tenors, par_yields, curve_shocks={"Parallel +100bp": stress.CURVE_SHOCKS_BP["Parallel +100bp"]}
    )
    moves = t.groupby("etf")["full_revaluation"].sum()  # rate move + spread move for credit
    out = moves.reindex(ASSETS).fillna(0.0)
    if severe:
        for a, x in stress.HYPOTHETICAL_SHOCKS["Equity -30%"].items():
            out[a] += x
    return out


def stressed_outflow(block: lb.LiabilityBlock, l0: float, surrender_rate: float = lb.SURRENDER_RATE) -> float:
    """One year of scheduled payments plus a surrender of `surrender_rate` x L0."""
    return lb.payments_between(block, 0.0, 1.0) + surrender_rate * l0


def lcr_coefficients(moves: pd.Series, a0: float, outflow: float) -> pd.Series:
    """LCR(w) = sum_i coef_i w_i: liquid value after stress and haircut / outflow."""
    hair = pd.Series({a: LIQUIDITY[a][1] for a in ASSETS})
    return a0 * (1 + moves) * (1 - hair) / outflow


# ----------------------------------------------------------------- capital

# PROXY capital factors per $ of market value. LABELED ASSUMPTIONS loosely
# inspired by the order of magnitude of NAIC RBC C-1 factors; NOT NAIC RBC.
CAPITAL_FACTORS = {
    "SHY": 0.0,
    "IEF": 0.0,
    "TLT": 0.0,
    "TIP": 0.0,
    "LQD": 0.010,  # average A/BBB corporate bonds
    "HYG": 0.060,  # mix of BB and B bonds
    "SPY": 0.30,
    "EFA": 0.30,
    "EEM": 0.30,  # common stock
    "VNQ": 0.30,  # REIT shares treated as common stock
    "GLD": 0.30,  # other invested assets
}
RATE_SHOCK = {
    "Parallel +100bp": stress.CURVE_SHOCKS_BP["Parallel +100bp"],
    "Parallel -100bp": stress.CURVE_SHOCKS_BP["Parallel -100bp"],
}
COVERAGE_TARGET = 2.0  # surplus / capital proxy >= 2


def capital_charge(weights: pd.Series, a0: float, etf_rate_chg: pd.DataFrame, liab_rate_chg: pd.Series) -> dict:
    """Asset charge (factor x market value) + interest-rate charge (the worse
    surplus loss under +/-100bp, floored at 0). Simple sum: no diversification
    credit (NAIC uses a square-root covariance formula; we do not)."""
    w = weights.reindex(ASSETS).fillna(0.0)
    asset = a0 * float(sum(CAPITAL_FACTORS[a] * w[a] for a in ASSETS))
    losses = [-(a0 * float(etf_rate_chg.loc[k] @ w) - float(liab_rate_chg[k])) for k in RATE_SHOCK]
    rate = max(0.0, *losses)
    return {"asset_charge": asset, "rate_charge": rate, "total": asset + rate}


def capital_rules(a0: float, cap: float, etf_rate_chg: pd.DataFrame, liab_rate_chg: pd.Series) -> list[LinearRule]:
    """capital <= cap as linear rules: asset part alone, and asset part plus the
    surplus loss in each rate shock."""
    f = pd.Series(CAPITAL_FACTORS)[ASSETS]
    rules = [LinearRule("capital: asset charge", (a0 * f).to_dict(), upper=cap)]
    for k in RATE_SHOCK:
        coefs = a0 * (f - etf_rate_chg.loc[k][ASSETS])
        rules.append(LinearRule(f"capital: asset + rate ({k})", coefs.to_dict(), upper=cap - float(liab_rate_chg[k])))
    return rules


# ------------------------------------------------------------- constraints

INSURER_GROUPS = [
    GroupConstraint("IG credit floor", ["LQD"], 0.40, 1.0),
    GroupConstraint("Equity cap", ["SPY", "EFA", "EEM"], 0.0, 0.10),
    GroupConstraint("High yield cap", ["HYG"], 0.0, 0.05),
    GroupConstraint("Real estate cap", ["VNQ"], 0.0, 0.05),
    GroupConstraint("Gold cap", ["GLD"], 0.0, 0.05),
]


def insurer_problem(
    mu,
    cov,
    liab,
    lcr_coef: pd.Series,
    cap_rules: list[LinearRule],
    ig_floor: bool = True,
    with_liabilities: bool = True,
) -> PortfolioProblem:
    groups = [g for g in INSURER_GROUPS if ig_floor or g.name != "IG credit floor"]
    rules = [LinearRule("liquidity coverage >= 1", lcr_coef.to_dict(), lower=1.0), *cap_rules]
    return PortfolioProblem(mu, cov, liabilities=liab if with_liabilities else None, groups=groups, rules=rules)


# --------------------------------------------------------------- CVaR (LP)


def min_cvar(p: PortfolioProblem, R1: pd.DataFrame, rL: np.ndarray, target: float, level: float = 0.95):
    """Rockafellar-Uryasev: minimize CVaR of the 1-year surplus loss per $ of
    assets, subject to the problem's constraints and E[surplus return] = target.

        min  alpha + 1/((1-level) N) sum_p u_p
        s.t. u_p >= -(R_p w - f rL_p) - alpha,  u_p >= 0
    Returns (weights, cvar, alpha).
    """
    R = R1[p.assets].values
    n_s, n = R.shape
    f = p.f
    c = np.concatenate([np.zeros(n), [1.0], np.full(n_s, 1.0 / ((1 - level) * n_s))])
    # u_p >= -R_p w + f rL_p - alpha  ->  -R_p w - alpha - u_p <= -f rL_p
    A_scen = sparse.hstack([sparse.csr_matrix(-R), sparse.csr_matrix(-np.ones((n_s, 1))), -sparse.identity(n_s)])
    b_scen = -f * rL
    rows, b = [A_scen], [b_scen]
    for g, row in p.group_rows():
        rows += [
            sparse.csr_matrix(np.concatenate([row, np.zeros(1 + n_s)])),
            sparse.csr_matrix(np.concatenate([-row, np.zeros(1 + n_s)])),
        ]
        b += [np.array([g.upper]), np.array([-g.lower])]
    for rule, row in p.rule_rows():
        if np.isfinite(rule.upper):
            rows.append(sparse.csr_matrix(np.concatenate([row, np.zeros(1 + n_s)])))
            b.append(np.array([rule.upper]))
        if np.isfinite(rule.lower):
            rows.append(sparse.csr_matrix(np.concatenate([-row, np.zeros(1 + n_s)])))
            b.append(np.array([-rule.lower]))
    A_ub = sparse.vstack(rows).tocsr()
    b_ub = np.concatenate(b)
    mu = p.mu.values
    A_eq = np.vstack([np.concatenate([np.ones(n), np.zeros(1 + n_s)]), np.concatenate([mu, np.zeros(1 + n_s)])])
    b_eq = np.array([1.0, target + f * p.mu_l])
    bounds = [*p.bounds_list(), (None, None), *[(0, None)] * n_s]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise ValueError(f"CVaR LP failed: {res.message}")
    w = pd.Series(res.x[:n], index=p.assets)
    return w, float(res.fun), float(res.x[n])


def duration_band_rule(
    etf_krd: pd.DataFrame, liab_krd: pd.Series, a0: float, l0: float, band_per_100bp: float
) -> LinearRule:
    """Dollar duration gap per 100bp, (A0 w'D - L0 D_L) / 100, within +/- band ($m)."""
    d = etf_krd.sum(axis=1)[ASSETS]
    center = l0 * float(liab_krd.sum()) / 100
    return LinearRule(
        f"duration gap within +/-{band_per_100bp:g}m per 100bp",
        (a0 * d / 100).to_dict(),
        lower=center - band_per_100bp,
        upper=center + band_per_100bp,
    )


def solve_for_target(p: PortfolioProblem, target: float) -> tuple[pd.Series, str]:
    """The recommendation rule with its edge cases (D-061, reused in D-065).

    Minimum surplus volatility for the target expected surplus return; if the
    target is below the lowest feasible expected return, take the min-variance
    portfolio; if above the highest, take the max-return portfolio. Raises
    ValueError if the constraints themselves are infeasible.
    """
    from engine.optimizer import max_return_weights, min_variance, surplus_expected_return, target_return

    mv = min_variance(p)
    w_hi = max_return_weights(p)
    hi = surplus_expected_return(w_hi, p)
    if target <= mv.expected_return:
        return mv.weights, "target below min-variance return: min-variance portfolio"
    if target >= hi:
        return pd.Series(w_hi, index=p.assets), "target above max feasible return: max-return portfolio"
    res = target_return(p, target)
    return res.weights, ("target met" if res.success else f"solver: {res.message}")

"""Stage 3: liabilities, surplus, ALM, liquidity, capital-aware optimization,
regimes, and the client report.

    python -m engine.run_stage3

Writes reports/stage3_results.md (technical), reports/client_report.md and
reports/client_report.pdf (client-facing), CSVs and charts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import alm, data, esg, regimes, scenarios
from engine import estimation as est
from engine import fixedincome as fi
from engine import liabilities as lb
from engine import rates as rt
from engine.optimizer import (
    PortfolioProblem,
    efficient_frontier,
    max_sharpe,
    min_variance,
    target_return,
)
from engine.run_stage1 import ILLUSTRATIVE_BOUNDS, ILLUSTRATIVE_GROUPS, REPORTS_DIR
from engine.run_stage1b import COLORS, INK, INK_2, SURFACE, _plt, _style

A0 = lb.PV_TARGET * lb.FUNDING_RATIO  # $m
SURPLUS_GROWTH_TARGET = 0.10  # target expected surplus growth per year, share of starting surplus (D-052)
SEED = 20261019
DURATION_BAND = 10.0  # $m of surplus per 100bp, for the duration-band variant (D-053)


@dataclass
class Context:
    rets: pd.DataFrame
    a: pd.DataFrame
    fred_m: pd.DataFrame
    tenors: list
    par: pd.DataFrame
    y0: np.ndarray
    mu_bs: pd.Series
    fm: scenarios.FactorModel
    pca: esg.CurvePCA
    anchor: np.ndarray
    scen: esg.ESGScenarios
    block: lb.LiabilityBlock
    L: np.ndarray
    R1: pd.DataFrame
    rL: np.ndarray
    mu: pd.Series
    cov: pd.DataFrame
    liab: object
    l0: float


def build_context(anchor_kind: str = "forward", seed: int = SEED, n_paths: int = 10_000) -> Context:
    prices, _ = data.clean_prices(data.load_prices())
    rets = data.monthly_returns(prices)
    a = rets[list(data.UNIVERSE)]
    fred_m = data.monthly_fred(data.load_fred(), rets.index)
    tenors, par = scenarios.par_curve_history(fred_m)
    y0 = par.iloc[-1].values
    mu_bs = est.bayes_stein_mean(a).mu
    fm = scenarios.build_factor_model(rets, fred_m, mu_bs)
    pca = esg.curve_pca(par, tenors)
    anchor = {"forward": esg.forward_par_curve(fm.curve0, tenors), "historical": par.mean().values, "today": y0}[
        anchor_kind
    ]
    scen = esg.simulate(fm, pca, rets, fred_m, mu_bs, anchor, esg.ESGConfig(n_paths=n_paths, seed=seed))
    block = lb.annuity_block(fm.curve0)
    L = alm.liability_paths(block, scen)
    l0 = float(L[0, 0])
    R1, rL = alm.one_year_inputs(scen, L, block)
    mu, cov, liab = alm.scenario_moments(R1, rL, l0 / A0)
    return Context(rets, a, fred_m, tenors, par, y0, mu_bs, fm, pca, anchor, scen, block, L, R1, rL, mu, cov, liab, l0)


def _pct(x, d=2):
    return (x * 100).round(d)


def portfolio_metrics(name: str, w: pd.Series, ctx: Context, rate_chg, liab_rate, lcr_coef) -> dict:
    dS = alm.surplus_change(w, ctx.R1, ctx.rL, A0, ctx.l0)
    sp = alm.simulate_surplus(w, ctx.scen, ctx.L, ctx.block, A0)
    fr5 = sp.funding_ratio[:, -1]
    cc = alm.capital_charge(w, A0, rate_chg, liab_rate)
    s0 = A0 - ctx.l0
    return {
        "portfolio": name,
        "exp_surplus_change_1y": dS.mean(),
        "surplus_vol_1y": dS.std(ddof=1),
        "surplus_CVaR95_1y": alm.cvar_worst_k(-dS, 0.95),
        "prob_surplus_loss_1y": float((dS < 0).mean()),
        "surplus_sharpe_number_only": dS.mean() / dS.std(ddof=1),
        "capital_proxy": cc["total"],
        "capital_rate_part": cc["rate_charge"],
        "coverage_surplus_over_capital": s0 / cc["total"] if cc["total"] > 0 else np.inf,
        "liquidity_coverage": float(lcr_coef @ w.reindex(alm.ASSETS).fillna(0.0)),
        "funding_ratio_5y_median": float(np.median(fr5)),
        "funding_ratio_5y_p05": float(np.quantile(fr5, 0.05)),
        "prob_funding_below_100_5y": float((fr5 < 1).mean()),
    }


# ------------------------------------------------------------------ charts


def _fan(ax, x, months, color, title):
    q = np.percentile(x, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(months, q[0], q[4], color=color, alpha=0.18, linewidth=0, label="5th to 95th pct")
    ax.fill_between(months, q[1], q[3], color=color, alpha=0.35, linewidth=0, label="25th to 75th pct")
    ax.plot(months, q[2], color=color, linewidth=2, label="Median")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Months ahead", color=INK_2, fontsize=9)


def chart_generator_compare(pca_10y, vas_10y, path):
    plt = _plt()
    months = np.arange(pca_10y.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, sharey=True)
    for ax, x, title in zip(
        axes,
        [vas_10y * 100, pca_10y * 100],
        ["Vasicek, one factor (Stage 2 benchmark)", "Empirical PCA generator (Stage 3 default)"],
        strict=True,
    ):
        _style(ax)
        _fan(ax, x, months, COLORS[0], title)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    axes[0].set_ylabel("10-year yield (%)", color=INK_2, fontsize=9)
    fig.suptitle(
        "10-year Treasury yield, 10,000 simulated paths, both anchored to the market forward curve",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_funding_fans(paths: dict, path):
    plt = _plt()
    names = list(paths)
    fig, axes = plt.subplots(1, len(names), figsize=(11, 4.4), facecolor=SURFACE, sharey=True)
    for ax, name in zip(axes, names, strict=True):
        _style(ax)
        fr = paths[name].funding_ratio * 100
        _fan(ax, fr, np.arange(fr.shape[1]), COLORS[0], name)
        ax.axhline(100, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            "100% = assets just cover liabilities",
            (1, 100),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    axes[0].set_ylabel("Funding ratio (%)", color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.suptitle(
        "Funding ratio (assets / liabilities) over 5 years. Steps: each annual payment lowers assets and "
        "liabilities by the same dollars, which lifts the ratio when it is above 100%",
        color=INK,
        fontsize=10.5,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_surplus_dist(changes: dict, s0: float, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=SURFACE)
    _style(ax)
    lo = min(np.quantile(v, 0.001) for v in changes.values())
    hi = max(np.quantile(v, 0.999) for v in changes.values())
    bins = np.linspace(lo, hi, 80)
    for (name, v), color in zip(changes.items(), COLORS, strict=False):
        ax.hist(
            v,
            bins=bins,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{name} (1-in-20 avg loss ${alm.cvar_worst_k(-v, 0.95):.0f}m)",
        )
    ax.axvline(-s0, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(
        f"loss of the whole ${s0:.0f}m surplus",
        (-s0, ax.get_ylim()[1] * 0.9),
        xytext=(-6, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_2,
        ha="right",
    )
    ax.set_xlabel("Change in surplus over 1 year ($ millions)", color=INK_2, fontsize=9)
    ax.set_ylabel("Number of scenarios", color=INK_2, fontsize=9)
    ax.set_title("One-year surplus change across 10,000 scenarios", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_allocation(weights: pd.DataFrame, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    keep = weights[(weights > 0.005).any(axis=1)]
    y = np.arange(len(keep))
    h = 0.8 / len(keep.columns)
    for i, (col, color) in enumerate(zip(keep.columns, COLORS, strict=False)):
        vals = keep[col].values * 100
        bars = ax.barh(y + (i - (len(keep.columns) - 1) / 2) * h, vals, h * 0.9, color=color, label=col)
        for b, v in zip(bars, vals, strict=True):
            if v > 0.5:
                ax.annotate(
                    f"{v:.0f}%",
                    (v, b.get_y() + b.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=INK_2,
                )
    ax.set_yticks(y, keep.index)
    ax.invert_yaxis()
    ax.set_xlabel("Weight (%)", color=INK_2, fontsize=9)
    ax.set_title("Asset allocation: recommended vs alternatives", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_frontier(fr: pd.DataFrame, chosen: dict, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, sharey=True)
    for ax, xcol, xlabel in zip(
        axes, ["surplus_vol", "capital"], ["1-year surplus volatility ($m)", "Capital proxy used ($m)"], strict=True
    ):
        _style(ax)
        ax.plot(
            fr[xcol],
            fr["exp_surplus_change"],
            color=COLORS[0],
            linewidth=2,
            marker="o",
            markersize=5,
            label="Capital-aware surplus frontier",
        )
        ax.plot(
            chosen[xcol],
            chosen["exp_surplus_change"],
            marker="*",
            markersize=16,
            color=COLORS[1],
            linestyle="none",
            label="Recommended",
        )
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    axes[0].set_ylabel("Expected 1-year surplus change ($m)", color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.suptitle(
        "Surplus frontier under insurer constraints: return vs surplus risk and vs capital",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------- main


def main() -> None:
    ctx = build_context("forward")
    s0 = A0 - ctx.l0
    f = ctx.l0 / A0
    rf_now = float(ctx.y0[0])
    T = []  # technical report lines

    # ---- 0. generator comparison and anchor
    tlt_hist = float(ctx.a["TLT"].std() * np.sqrt(12))
    fp = scenarios.simulate_factors(ctx.fm, 10_000, 60, SEED)
    vas_ret = scenarios.asset_returns(ctx.fm, fp, SEED)
    i10 = ctx.tenors.index(10.0)
    pca_10 = ctx.scen.par[:, :, i10]
    # Vasicek with b anchored to the market forward short rate, lam refit
    cal = ctx.fm.rates_cal.params
    b_anchor = float(ctx.anchor[0])
    z10 = pd.Series(
        [float(fi.bootstrap_zero_curve(ctx.tenors, row).zero_rate(10.0)) for row in ctx.par.values], index=ctx.par.index
    )
    p_anch = rt.VasicekParams(cal.a, b_anchor, cal.sigma)
    p_anch = rt.VasicekParams(
        cal.a, b_anchor, cal.sigma, rt.fit_market_price_of_risk(p_anch, ctx.fred_m["DGS3MO"], z10)
    )
    sim_v = rt.simulate(p_anch, ctx.fm.r0, 10_000, 60, rt.DT, seed=SEED)
    off10 = float(ctx.fm.curve0.zero_rate(10.0)) - float(rt.zero_yield(p_anch, ctx.fm.r0, 10.0))
    vas_10 = rt.model_curve(p_anch, sim_v.r, np.array([10.0]))[..., 0] + off10
    vol_rows = []
    for etf in ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG"]:
        vol_rows.append(
            {
                "etf": etf,
                "historical": ctx.a[etf].std() * np.sqrt(12),
                "Vasicek factor model (Stage 2)": esg.annual_vol(vas_ret[etf]),
                "PCA generator (Stage 3)": esg.annual_vol(ctx.scen.returns[etf]),
            }
        )
    d10_hist = float(ctx.par["DGS10"].diff().std() * np.sqrt(12))
    T += [
        "# Stage 3 results: liabilities, surplus, ALM, liquidity, capital-aware optimization",
        "",
        "Generated by `python -m engine.run_stage3`. $ figures in millions. SYNTHETIC insurer. Capital figures "
        "are a PROXY, not NAIC RBC.",
        "",
        "## 0. Fixing P-013: empirical PCA curve generator",
        "",
        "Explained variance of monthly par-curve changes by principal component:",
        "",
        pd.DataFrame(
            {"explained": ctx.pca.explained[:5], "cumulative": np.cumsum(ctx.pca.explained[:5])},
            index=["level", "slope", "curvature", "PC4", "PC5"],
        )
        .round(4)
        .to_markdown(),
        "",
        "Loadings (first three components):",
        "",
        pd.DataFrame(ctx.pca.loadings[:, :3], index=ctx.tenors, columns=["level", "slope", "curvature"])
        .round(3)
        .to_markdown(),
        "",
        "Annual volatility, historical vs both generators:",
        "",
        _pct(pd.DataFrame(vol_rows).set_index("etf")).to_markdown(),
        "",
        f"10-year yield change volatility: history {d10_hist * 1e4:.0f}bp/yr, PCA generator "
        f"{np.diff(pca_10, axis=1).std() * np.sqrt(12) * 1e4:.0f}bp/yr, Vasicek "
        f"{np.diff(vas_10, axis=1).std() * np.sqrt(12) * 1e4:.0f}bp/yr. TLT volatility: history {tlt_hist:.1%}, "
        f"Vasicek {esg.annual_vol(vas_ret['TLT']):.1%}, PCA {esg.annual_vol(ctx.scen.returns['TLT']):.1%}.",
        "",
    ]
    chart_generator_compare(pca_10, vas_10, REPORTS_DIR / "stage3_generator_compare.png")
    T += [
        "![Generator comparison](stage3_generator_compare.png)",
        "",
        "Anchor curves (par yields, %): the market 5-year forward curve is the default; history and today's "
        "curve are sensitivities.",
        "",
        _pct(
            pd.DataFrame(
                {
                    "today": ctx.y0,
                    "forward (5y ahead, DEFAULT)": ctx.anchor,
                    "historical average": ctx.par.mean().values,
                },
                index=ctx.tenors,
            )
        ).to_markdown(),
        "",
        f"Vasicek benchmark with the anchored level: b = {b_anchor:.2%} (historical estimate "
        f"{cal.b:.2%}), lam refit = {p_anch.lam:.3f}.",
        "",
    ]

    # ---- 1. liabilities
    m = lb.measures(ctx.block, ctx.tenors, ctx.y0)
    T += [
        "## 1. Synthetic liabilities",
        "",
        f"{ctx.block.label}: 30 annual payments falling {lb.DECAY:.0%} a year, first payment "
        f"${ctx.block.amounts[0]:.1f}m. Discounted on the Treasury zero curve (same as assets).",
        "",
        pd.Series(
            {
                "PV ($m)": m["pv"],
                "IRR (semiannual)": m["irr"],
                "Macaulay duration": m["macaulay"],
                "Modified duration": m["modified"],
                "Effective duration (curve)": m["effective_duration"],
                "Convexity (yield)": m["convexity"],
                "Effective convexity (curve)": m["effective_convexity"],
                "DV01 ($m per bp)": m["dv01"],
                "Sum of key rate durations": m["krd"].sum(),
            }
        )
        .round(4)
        .to_markdown(),
        "",
        "Key rate durations:",
        "",
        m["krd"].round(4).to_frame().T.to_markdown(),
        "",
        f"Closed-form check on a flat 5% curve: PV formula "
        f"{lb.flat_curve_pv(lb.annuity_block(fi.bootstrap_zero_curve(ctx.tenors, [0.05] * 8)), 0.05):.6f} vs "
        "curve 1000.000000 (block rescaled to PV 1000 on that curve).",
        "",
        f"Assets ${A0:.0f}m, liabilities ${ctx.l0:.0f}m, surplus ${s0:.0f}m, funding ratio {A0 / ctx.l0:.0%}.",
        "",
    ]

    # ---- ALM inputs
    krd = alm.etf_key_rate_durations(ctx.fm.rep, ctx.tenors, ctx.y0)
    rate_chg = alm.etf_curve_changes(ctx.fm.rep, ctx.tenors, ctx.y0, alm.RATE_SHOCK)
    liab_rate = alm.liability_curve_changes(ctx.block, ctx.tenors, ctx.y0, alm.RATE_SHOCK)
    moves_base = alm.liquidity_stress_moves(ctx.fm.rep, ctx.tenors, ctx.y0, severe=False)
    moves_sev = alm.liquidity_stress_moves(ctx.fm.rep, ctx.tenors, ctx.y0, severe=True)
    outflow = alm.stressed_outflow(ctx.block, ctx.l0)
    lcr_sev = alm.lcr_coefficients(moves_sev, A0, outflow)
    lcr_base = alm.lcr_coefficients(moves_base, A0, outflow)
    cap = s0 / alm.COVERAGE_TARGET
    rules = alm.capital_rules(A0, cap, rate_chg, liab_rate)

    # ---- portfolios
    ao = {
        "Asset-only min variance": min_variance(PortfolioProblem(ctx.mu, ctx.cov)).weights,
        "Asset-only max Sharpe": max_sharpe(PortfolioProblem(ctx.mu, ctx.cov, risk_free_rate=rf_now)).weights,
        "Asset-only constrained max Sharpe": max_sharpe(
            PortfolioProblem(
                ctx.mu, ctx.cov, bounds=ILLUSTRATIVE_BOUNDS, groups=ILLUSTRATIVE_GROUPS, risk_free_rate=rf_now
            )
        ).weights,
    }
    matched = alm.key_rate_matched(krd, m["krd"], A0, ctx.l0)
    p_free = alm.insurer_problem(ctx.mu, ctx.cov, ctx.liab, lcr_sev, rules, ig_floor=False)
    p_ins = alm.insurer_problem(ctx.mu, ctx.cov, ctx.liab, lcr_sev, rules, ig_floor=True)
    target = SURPLUS_GROWTH_TARGET * s0 / A0
    unbanded = target_return(p_ins, target)
    free = target_return(p_free, target)
    # Headline recommendation (D-057): insurer constraints PLUS the duration-gap band
    band_rule = alm.duration_band_rule(krd, m["krd"], A0, ctx.l0, DURATION_BAND)
    p_band = alm.insurer_problem(ctx.mu, ctx.cov, ctx.liab, lcr_sev, [*rules, band_rule])
    band = target_return(p_band, target)
    w_cvar, cvar_lp, _ = alm.min_cvar(p_band, ctx.R1, ctx.rL, target)
    ports = {
        **ao,
        "Key-rate matched (Treasuries)": matched,
        "Surplus-aware, no credit floor": free.weights,
        "RECOMMENDED: surplus + capital aware, duration band": band.weights,
        "CVaR variant (same constraints)": w_cvar,
        "Comparison: no duration band": unbanded.weights,
    }
    metrics = pd.DataFrame([portfolio_metrics(n, w, ctx, rate_chg, liab_rate, lcr_sev) for n, w in ports.items()])
    metrics = metrics.set_index("portfolio")
    weights = pd.DataFrame(ports)

    # ---- 2. surplus and ALM
    shocks = {
        "Parallel +10bp": {0.25: 10, 30.0: 10},
        "Parallel -10bp": {0.25: -10, 30.0: -10},
        **{k: v for k, v in alm.stress.CURVE_SHOCKS_BP.items()},
    }
    etf_chg = alm.etf_curve_changes(ctx.fm.rep, ctx.tenors, ctx.y0, shocks)
    liab_chg = alm.liability_curve_changes(ctx.block, ctx.tenors, ctx.y0, shocks)
    sel = {
        k: ports[k]
        for k in [
            "Asset-only min variance",
            "Asset-only max Sharpe",
            "Key-rate matched (Treasuries)",
            "RECOMMENDED: surplus + capital aware, duration band",
            "Comparison: no duration band",
        ]
    }
    ss = alm.surplus_shocks(sel, etf_chg, liab_chg, A0)
    gaps = pd.DataFrame({n: alm.duration_gap(w, krd, m["krd"], A0, ctx.l0) for n, w in sel.items()}).T
    T += [
        "## 2. Surplus and ALM",
        "",
        "Duration gap (asset duration uses each ETF's rate beta; dollar gap = A x D_A - L x D_L):",
        "",
        gaps.round(3).to_markdown(),
        "",
        "Surplus change for curve shocks ($m; assets via key rate durations and convexity of each ETF's "
        "representative bond, liabilities by full revaluation):",
        "",
        ss.pivot(index="shock", columns="portfolio", values="surplus_change").loc[list(shocks)].round(2).to_markdown(),
        "",
    ]

    # ---- 3. liquidity
    hair = pd.DataFrame(
        {
            "bucket": {a_: alm.LIQUIDITY[a_][0] for a_ in alm.ASSETS},
            "haircut": {a_: alm.LIQUIDITY[a_][1] for a_ in alm.ASSETS},
            "move: rates+spreads": moves_base,
            "move: severe (+ equity -30%)": moves_sev,
        }
    )
    lcr_tab = pd.DataFrame(
        {
            n: {
                "LCR rates+spreads": float(lcr_base @ w.reindex(alm.ASSETS).fillna(0)),
                "LCR severe": float(lcr_sev @ w.reindex(alm.ASSETS).fillna(0)),
            }
            for n, w in ports.items()
        }
    ).T
    T += [
        "## 3. Liquidity",
        "",
        f"Stressed 1-year outflow = scheduled payments ${lb.payments_between(ctx.block, 0, 1):.1f}m + "
        f"{lb.SURRENDER_RATE:.0%} surrender of liabilities ${lb.SURRENDER_RATE * ctx.l0:.1f}m = ${outflow:.1f}m. "
        "Liquidity coverage = liquid value after the stress move and haircut / outflow. Buckets and haircuts are "
        "labeled assumptions inspired by Basel III LCR.",
        "",
        hair.assign(
            **{c: _pct(hair[c]) for c in ["haircut", "move: rates+spreads", "move: severe (+ equity -30%)"]}
        ).to_markdown(),
        "",
        lcr_tab.round(2).to_markdown(),
        "",
    ]

    # ---- 4. capital-aware optimization
    fr = efficient_frontier(p_band, n_points=9)
    fr_rows = []
    for _, row in fr.iterrows():
        w = row[alm.ASSETS].astype(float)
        dS = alm.surplus_change(w, ctx.R1, ctx.rL, A0, ctx.l0)
        fr_rows.append(
            {
                "exp_surplus_change": dS.mean(),
                "surplus_vol": dS.std(ddof=1),
                "CVaR95": alm.cvar_worst_k(-dS, 0.95),
                "capital": alm.capital_charge(w, A0, rate_chg, liab_rate)["total"],
                "LQD": w["LQD"],
                "TLT": w["TLT"],
                "SHY": w["SHY"],
                "success": bool(row.success),
            }
        )
    fr_df = pd.DataFrame(fr_rows)
    fr_df.to_csv(REPORTS_DIR / "stage3_frontier.csv", index=False)
    chosen = {
        "exp_surplus_change": metrics.loc[
            "RECOMMENDED: surplus + capital aware, duration band", "exp_surplus_change_1y"
        ],
        "surplus_vol": metrics.loc["RECOMMENDED: surplus + capital aware, duration band", "surplus_vol_1y"],
        "capital": metrics.loc["RECOMMENDED: surplus + capital aware, duration band", "capital_proxy"],
    }
    scen_cvar = alm.cvar_worst_k(-alm.surplus_change(w_cvar, ctx.R1, ctx.rL, A0, ctx.l0), 0.95)
    T += [
        "## 4. Capital-aware optimization",
        "",
        "Capital proxy factors (PROXY, not NAIC RBC): "
        + ", ".join(f"{k} {v:.0%}" for k, v in alm.CAPITAL_FACTORS.items())
        + ". Plus an interest-rate charge = the worse surplus loss under +/-100bp. Cap: capital <= surplus / "
        f"{alm.COVERAGE_TARGET:g} = ${cap:.0f}m.",
        "",
        "Insurer constraints: "
        + "; ".join(f"{g.name} [{g.lower:.0%}, {g.upper:.0%}] on {', '.join(g.assets)}" for g in alm.INSURER_GROUPS)
        + "; liquidity coverage (severe) >= 1.0; capital cap.",
        "",
        f"Objective: minimum surplus volatility for a target expected surplus change of "
        f"{SURPLUS_GROWTH_TARGET:.0%} of starting surplus (${SURPLUS_GROWTH_TARGET * s0:.0f}m a year). "
        f"CVaR variant: Rockafellar-Uryasev LP, LP CVaR ${cvar_lp * A0:.2f}m vs scenario-computed "
        f"${scen_cvar:.2f}m.",
        "",
        "### Weights",
        "",
        _pct(weights, 1).to_markdown(),
        "",
        "### Metrics",
        "",
        metrics.round(3).to_markdown(),
        "",
        "### Capital-aware surplus frontier (insurer constraints + duration band, the recommended set)",
        "",
        fr_df.round(3).to_markdown(index=False),
        "",
    ]
    chart_frontier(fr_df, chosen, REPORTS_DIR / "stage3_frontier.png")
    T += ["![Frontier](stage3_frontier.png)", ""]

    # ---- 5. regimes and duration shocks on the recommended portfolio
    w_rec = band.weights
    hist_l = lb.historical_returns(ctx.block, ctx.tenors, ctx.par).reindex(ctx.a.index)
    s_ret = ctx.a[alm.ASSETS] @ w_rec - f * hist_l  # monthly surplus change per $ assets
    labels = regimes.label_regimes(data.load_signals(), ctx.a.index)
    reg_rows = []
    for dim in ("rates", "volatility"):
        for reg in sorted(labels[dim].unique()):
            mask = labels[dim] == reg
            x = s_ret[mask] * A0
            reg_rows.append(
                {
                    "dimension": dim,
                    "regime": reg,
                    "months": int(mask.sum()),
                    "mean surplus change $m/yr": x.mean() * 12,
                    "surplus vol $m/yr": x.std() * np.sqrt(12),
                    "worst month $m": x.min(),
                    "share of months losing": float((x < 0).mean()),
                }
            )
    reg_df = pd.DataFrame(reg_rows)
    all_shocks = {**alm.stress.CURVE_SHOCKS_BP}
    etf_all = alm.etf_curve_changes(ctx.fm.rep, ctx.tenors, ctx.y0, all_shocks)
    liab_all = alm.liability_curve_changes(ctx.block, ctx.tenors, ctx.y0, all_shocks)
    spread = alm.stress.duration_stress(ctx.fm.rep, ctx.tenors, ctx.y0, curve_shocks={})
    shock_rows = []
    for name, w in {
        "Recommended": w_rec,
        "Asset-only constrained max Sharpe": ao["Asset-only constrained max Sharpe"],
        "Key-rate matched": matched,
        "Comparison: no duration band": unbanded.weights,
    }.items():
        for sname in all_shocks:
            dS = A0 * float(etf_all.loc[sname] @ w.reindex(alm.ASSETS).fillna(0)) - float(liab_all[sname])
            shock_rows.append(
                {
                    "portfolio": name,
                    "shock": sname,
                    "surplus_change": dS,
                    "funding_ratio_after": (A0 * (1 + float(etf_all.loc[sname] @ w.reindex(alm.ASSETS).fillna(0))))
                    / (ctx.l0 + liab_all[sname]),
                }
            )
        for _, r_ in spread.iterrows():
            dS = A0 * float(w.get(r_.etf, 0.0)) * r_.full_revaluation
            shock_rows.append(
                {"portfolio": name, "shock": r_.shock, "surplus_change": dS, "funding_ratio_after": (A0 + dS) / ctx.l0}
            )
    shock_df = pd.DataFrame(shock_rows)
    worst = shock_df.loc[shock_df[shock_df.portfolio == "Recommended"].surplus_change.idxmin()]
    T += [
        "## 5. Recommended portfolio across regimes and duration-based shocks",
        "",
        "Historical monthly surplus change per regime (assets: actual ETF returns; liabilities: the block "
        "revalued on each month's actual curve). In-sample.",
        "",
        reg_df.round(2).to_markdown(index=False),
        "",
        shock_df.pivot(index="shock", columns="portfolio", values="surplus_change").round(2).to_markdown(),
        "",
        f"Worst duration-based shock for the recommended portfolio: {worst.shock}, surplus change "
        f"${worst.surplus_change:.1f}m, funding ratio after {worst.funding_ratio_after:.1%}.",
        "",
    ]

    # ---- anchor sensitivity
    sens = []
    for kind in ("forward", "today", "historical"):
        c2 = ctx if kind == "forward" else build_context(kind)
        dS = alm.surplus_change(w_rec, c2.R1, c2.rL, A0, c2.l0)
        sp = alm.simulate_surplus(w_rec, c2.scen, c2.L, c2.block, A0)
        sens.append(
            {
                "anchor": kind,
                "TLT mean return": c2.scen.returns["TLT"].mean() * 12,
                "3M yield median at 5y": float(np.median(c2.scen.par[:, -1, 0])),
                "10Y yield median at 5y": float(np.median(c2.scen.par[:, -1, i10])),
                "recommended: exp surplus change $m": dS.mean(),
                "recommended: surplus vol $m": dS.std(),
                "recommended: funding ratio 5y median": float(np.median(sp.funding_ratio[:, -1])),
            }
        )
    sens_df = pd.DataFrame(sens).set_index("anchor")
    T += ["## Anchor sensitivity (recommended portfolio held fixed)", "", sens_df.round(4).to_markdown(), ""]

    # ---- charts for the client report
    sp_paths = {
        "Recommended": alm.simulate_surplus(w_rec, ctx.scen, ctx.L, ctx.block, A0),
        "Asset-only alternative": alm.simulate_surplus(
            ao["Asset-only constrained max Sharpe"], ctx.scen, ctx.L, ctx.block, A0
        ),
    }
    chart_funding_fans(sp_paths, REPORTS_DIR / "stage3_funding_fan.png")
    changes = {
        "Recommended": alm.surplus_change(w_rec, ctx.R1, ctx.rL, A0, ctx.l0),
        "Asset-only alternative": alm.surplus_change(
            ao["Asset-only constrained max Sharpe"], ctx.R1, ctx.rL, A0, ctx.l0
        ),
        "Asset-only min variance": alm.surplus_change(ao["Asset-only min variance"], ctx.R1, ctx.rL, A0, ctx.l0),
    }
    chart_surplus_dist(changes, s0, REPORTS_DIR / "stage3_surplus_distribution.png")
    chart_allocation(
        pd.DataFrame(
            {
                "Recommended": w_rec,
                "Asset-only alternative": ao["Asset-only constrained max Sharpe"],
                "Key-rate matched": matched,
            }
        ),
        REPORTS_DIR / "stage3_allocation.png",
    )

    (REPORTS_DIR / "stage3_results.md").write_text("\n".join(T))
    metrics.to_csv(REPORTS_DIR / "stage3_metrics.csv")
    weights.to_csv(REPORTS_DIR / "stage3_weights.csv")

    from engine.client_report import write_client_report

    write_client_report(metrics, weights, reg_df, shock_df, sens_df, ctx, A0, s0, outflow, cap, gaps)
    print("\n".join(T))
    print("\nWrote reports/stage3_results.md, reports/client_report.md, reports/client_report.pdf")


if __name__ == "__main__":
    main()

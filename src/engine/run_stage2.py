"""Stage 2: fixed income, stochastic rates, yield-based expected returns.

    python -m engine.run_stage2

Writes reports/stage2_results.md, CSVs and three charts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import data, scenarios, stress
from engine import estimation as est
from engine import fixedincome as fi
from engine import rates as rt
from engine.montecarlo import SimulationConfig, simulate_returns
from engine.optimizer import PortfolioProblem, efficient_frontier, max_sharpe, min_variance
from engine.run_stage1 import ILLUSTRATIVE_BOUNDS, ILLUSTRATIVE_GROUPS, REPORTS_DIR
from engine.run_stage1b import COLORS, GRID, INK, INK_2, SURFACE, _plt, _style

FAN_PATHS = 10_000
SEED = 20261019


def _pct(x, d=2):
    return (x * 100).round(d)


def optimized(mu: pd.Series, cov: pd.DataFrame, rf: float) -> dict[str, pd.Series]:
    return {
        "Min variance": min_variance(PortfolioProblem(mu, cov)).weights,
        "Max Sharpe": max_sharpe(PortfolioProblem(mu, cov, risk_free_rate=rf)).weights,
        "Constrained max Sharpe": max_sharpe(
            PortfolioProblem(mu, cov, bounds=ILLUSTRATIVE_BOUNDS, groups=ILLUSTRATIVE_GROUPS, risk_free_rate=rf)
        ).weights,
    }


# ------------------------------------------------------------------ charts


def chart_yield_history(par: pd.DataFrame, spread: pd.Series, path):
    plt = _plt()
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6.4), facecolor=SURFACE, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax = axes[0]
    _style(ax)
    for color, (col, label) in zip(
        COLORS, [("DGS3MO", "3-month"), ("DGS2", "2-year"), ("DGS10", "10-year"), ("DGS30", "30-year")], strict=True
    ):
        ax.plot(par.index, par[col] * 100, color=color, linewidth=2, label=label)
    for start, end in scenarios.ZERO_BOUND_SPELLS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=GRID, alpha=0.6, linewidth=0)
    ax.set_ylabel("Treasury par yield (%)", color=INK_2, fontsize=9)
    ax.set_title(
        "US Treasury yields, 2007 to 2026 (shaded: zero-bound spells)", color=INK, fontsize=12, loc="left", pad=26
    )
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=INK)
    ax2 = axes[1]
    _style(ax2)
    ax2.plot(spread.index, spread * 100, color=INK_2, linewidth=2)
    ax2.set_ylabel("Baa minus 10y (%)", color=INK_2, fontsize=9)
    ax2.set_title("Credit spread factor (Moody's Baa minus 10-year Treasury)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_fan(fm: scenarios.FactorModel, fp: scenarios.FactorPaths, hist_r: pd.Series, path):
    plt = _plt()
    months = np.arange(fp.short.r.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    i10 = fp.rep_names.index("IEF")  # 8.5y, the closest rep maturity to 10y
    series = [
        ("Short rate (Vasicek)", fp.short.r * 100),
        ("8.5-year zero yield (model + today's offset)", fp.yields[:, :, i10] * 100),
    ]
    for ax, (title, x) in zip(axes, series, strict=True):
        _style(ax)
        q = np.percentile(x, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(months, q[0], q[4], color=COLORS[0], alpha=0.18, linewidth=0, label="5th to 95th pct")
        ax.fill_between(months, q[1], q[3], color=COLORS[0], alpha=0.35, linewidth=0, label="25th to 75th pct")
        ax.plot(months, q[2], color=COLORS[0], linewidth=2, label="Median")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Months ahead", color=INK_2, fontsize=9)
        ax.set_ylabel("Yield (%)", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    fig.suptitle(
        f"Simulated rates, {fp.short.r.shape[0]:,} paths, 5 years (real-world measure)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_duration(rb: scenarios.RepBond, path):
    plt = _plt()
    dy = np.linspace(-0.03, 0.03, 121)
    p0 = fi.price_from_yield(rb.bond, rb.ytm)
    full = np.array([fi.price_from_yield(rb.bond, rb.ytm + d) / p0 - 1 for d in dy])
    d_only = -rb.par_duration * dy
    d_c = fi.duration_convexity_change(rb.par_duration, rb.par_convexity, dy)
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.plot(dy * 1e4, full * 100, color=COLORS[0], linewidth=2, label="Full revaluation")
    ax.plot(dy * 1e4, d_only * 100, color=COLORS[1], linewidth=2, linestyle=(0, (4, 3)), label="Duration only")
    ax.plot(dy * 1e4, d_c * 100, color=COLORS[2], linewidth=2, linestyle=(0, (1, 2)), label="Duration + convexity")
    notes = []
    for x in (-100, 100):
        f = fi.price_from_yield(rb.bond, rb.ytm + x / 1e4) / p0 - 1
        e = float(fi.duration_convexity_change(rb.par_duration, rb.par_convexity, x / 1e4))
        notes.append(f"{x:+d}bp: full {f:+.2%}, duration only {-rb.par_duration * x / 1e4:+.2%}, D+C {e:+.2%}")
    ax.text(0.02, 0.05, "\n".join(notes), transform=ax.transAxes, fontsize=9, color=INK, va="bottom")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Change in yield (bp)", color=INK_2, fontsize=9)
    ax.set_ylabel("Price change (%)", color=INK_2, fontsize=9)
    ax.set_title(
        f"{rb.bond.maturity:g}-year par bond (TLT representative, coupon {rb.bond.coupon:.2%}): "
        "duration vs full revaluation",
        color=INK,
        fontsize=12,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------- main


def bond_math_validation(tenors, par_today) -> pd.DataFrame:
    curve = fi.bootstrap_zero_curve(tenors, par_today)
    rows = []
    for mat in (2.0, 5.0, 10.0, 30.0):
        b = fi.Bond(mat, curve.par_yield(mat))
        p = fi.price_from_curve(b, curve)
        y = fi.ytm(b, p)
        h = fi.BP
        fd_d = (fi.price_from_yield(b, y - h) - fi.price_from_yield(b, y + h)) / (2 * p * h)
        sd_c = (fi.price_from_yield(b, y - h) + fi.price_from_yield(b, y + h) - 2 * p) / (p * h * h)
        eff = fi.effective_duration(b, tenors, par_today)
        krd = fi.key_rate_durations(b, tenors, par_today).sum()
        md, cx = fi.modified_duration(b, y), fi.convexity(b, y)
        row = {
            "maturity": mat,
            "coupon": b.coupon,
            "price_off_curve": p,
            "macaulay": fi.macaulay_duration(b, y),
            "modified": md,
            "fd_duration": fd_d,
            "convexity": cx,
            "second_diff_convexity": sd_c,
            "dv01": fi.dv01(b, y),
            "effective_duration": eff,
            "sum_krd": krd,
        }
        for move in (0.01, -0.01):
            full = fi.price_from_yield(b, y + move) / p - 1
            row[f"err_D_{move:+.0%}"] = -md * move - full
            row[f"err_DC_{move:+.0%}"] = float(fi.duration_convexity_change(md, cx, move)) - full
        z = fi.Bond(mat, 0.0)
        row["zero_price_curve"] = fi.price_from_curve(z, curve)
        row["zero_price_closed_form"] = 100 * np.exp(-float(curve.zero_rate(mat)) * mat)
        rows.append(row)
    return pd.DataFrame(rows).set_index("maturity")


def main() -> None:
    prices, _ = data.clean_prices(data.load_prices())
    rets = data.monthly_returns(prices)
    assets = list(data.UNIVERSE)
    a = rets[assets]
    fred_m = data.monthly_fred(data.load_fred(), rets.index)
    tenors, par = scenarios.par_curve_history(fred_m)
    par_today = par.iloc[-1].values
    curve = fi.bootstrap_zero_curve(tenors, par_today)
    bs = est.bayes_stein_mean(a)
    cov = est.estimate_covariance(a)
    rf_hist = float(rets[data.RISK_FREE_TICKER].mean() * 12)
    rf_now = float(par_today[0])

    L = [
        "# Stage 2 results: fixed income, stochastic rates, yield-based expected returns",
        "",
        f"Generated by `python -m engine.run_stage2`. Data: FRED Treasury par yields and Moody's Baa-10y spread, "
        f"month ends {fred_m.index.min():%Y-%m} to {fred_m.index.max():%Y-%m}. ICE OAS used only for today's level. "
        "Percent unless stated.",
        "",
        "## Today's curve (" + f"{fred_m.index[-1]:%Y-%m-%d})",
        "",
    ]
    taus = np.array([0.25, 1, 2, 5, 7, 10, 20, 30])
    L += [
        pd.DataFrame(
            {
                "par": par_today * 100,
                "zero (semiannual)": curve.zero_rate(taus, "semiannual") * 100,
                "zero (continuous)": curve.zero_rate(taus) * 100,
            },
            index=taus,
        )
        .round(3)
        .to_markdown(),
        "",
        f"ICE OAS today: IG {fred_m[data.ICE_IG_OAS].dropna().iloc[-1]:.2%}, "
        f"HY {fred_m[data.ICE_HY_OAS].dropna().iloc[-1]:.2%}. Baa-10y today: "
        f"{fred_m[data.CREDIT_FACTOR_SERIES].iloc[-1]:.2%}.",
        "",
    ]
    chart_yield_history(par, fred_m[data.CREDIT_FACTOR_SERIES], REPORTS_DIR / "stage2_yield_history.png")
    L += ["![Yield history](stage2_yield_history.png)", ""]

    # bond math
    bmv = bond_math_validation(tenors, par_today)
    bmv.to_csv(REPORTS_DIR / "stage2_bond_math_validation.csv")
    L += [
        "## Bond math validation (par bonds on today's curve)",
        "",
        "Errors are (estimate minus full revaluation) for a 100bp yield move, in percentage points of price.",
        "",
        bmv.assign(**{c: bmv[c] * 100 for c in bmv.columns if c.startswith("err_")}).round(5).to_markdown(),
        "",
    ]

    # Vasicek calibration
    mask = scenarios.zero_bound_mask(fred_m.index)
    idx = fred_m.index
    z1 = (idx >= "2008-12-31") & (idx <= "2015-11-30")
    cases = {
        "full 2007-2026": None,
        "excluding 2008-12 to 2015-11": pd.Series(~z1, index=idx),
        "excluding both zero-bound spells (USED)": mask,
        "post 2015-12 only": pd.Series(idx >= "2015-12-31", index=idx),
    }
    cal_rows = [rt.calibrate(fred_m["DGS3MO"], m, label=k).table() for k, m in cases.items()]
    cal_df = pd.DataFrame(cal_rows).set_index("sample")
    fm = scenarios.build_factor_model(rets, fred_m, bs.mu)
    p = fm.rates
    L += [
        "## Vasicek calibration (3-month yield as the short rate, monthly AR(1), exact form)",
        "",
        cal_df.round(4).to_markdown(),
        "",
        f"Months with the 3M yield below 0.25%: {int((fred_m['DGS3MO'] < 0.0025).sum())} of {len(fred_m)}.",
        "",
        f"Used: a = {p.a:.4f}, b = {p.b:.4f}, sigma = {p.sigma:.4f}, market price of risk lam = {p.lam:.4f} "
        f"(fitted so the model's average 10y zero yield equals the actual average), pricing level b_q = {p.b_q:.4f}. "
        f"Starting short rate r0 = {fm.r0:.4f}.",
        "",
    ]
    mt = np.array([0.25, 2, 5, 10, 20, 30])
    L += [
        "Model curve today vs actual (continuous zero yields). Vasicek does NOT fit today's curve; the simulation "
        "adds a constant per-maturity offset so it starts on the actual curve:",
        "",
        pd.DataFrame({"model": rt.zero_yield(p, fm.r0, mt) * 100, "actual": curve.zero_rate(mt) * 100}, index=mt)
        .assign(offset=lambda d: d.actual - d.model)
        .round(3)
        .to_markdown(),
        "",
    ]

    # simulation validation (Q measure for the martingale check)
    q = p.risk_neutral()
    simq = rt.simulate(q, fm.r0, 100_000, 120, rt.DT, seed=SEED)
    dfs = simq.discount_factors()
    val_rows = []
    for yrs in (1, 5, 10):
        k = yrs * 12 - 1
        mc = dfs[:, k]
        val_rows.append(
            {
                "horizon_years": yrs,
                "MC E[exp(-int r)]": mc.mean(),
                "closed form P(0,T)": float(rt.zero_price(q, fm.r0, float(yrs))),
                "MC std error": mc.std() / np.sqrt(len(mc)),
                "sim mean r": simq.r[:, k + 1].mean(),
                "analytic mean r": float(rt.mean_r(q, fm.r0, yrs)),
                "sim var r": simq.r[:, k + 1].var(),
                "analytic var r": float(rt.var_r(q, yrs)),
            }
        )
    L += [
        "## Simulation checks (100,000 paths, pricing measure)",
        "",
        pd.DataFrame(val_rows).set_index("horizon_years").to_markdown(floatfmt=".6f"),
        "",
    ]

    # factor model
    sp = fm.spread
    L += [
        "## Credit spread and correlation model",
        "",
        f"Log-OU on Baa-10y: kappa = {sp['kappa']:.3f} (se {sp['kappa_se']:.3f}), long-run median spread "
        f"{sp['long_run_median_spread']:.2%}, sigma (of log spread) = {sp['sigma']:.3f}. "
        "Shock correlations (standardized residuals, all months):",
        "",
        fm.corr.round(3).to_markdown(),
        "",
    ]
    fp = scenarios.simulate_factors(fm, FAN_PATHS, 60, SEED)
    sim_corr = np.corrcoef(fp.shocks.reshape(-1, 3).T)
    L += [
        f"Simulated shock correlations: rate-spread {sim_corr[0, 1]:.3f}, rate-equity {sim_corr[0, 2]:.3f}, "
        f"spread-equity {sim_corr[1, 2]:.3f}. Minimum simulated spread: {fp.spread.min():.3%}. "
        f"Share of paths with a negative short rate at 5 years: {(fp.short.r[:, -1] < 0).mean():.1%}.",
        "",
    ]
    chart_fan(fm, fp, fred_m["DGS3MO"], REPORTS_DIR / "stage2_rate_fan.png")
    L += ["![Rate fan chart](stage2_rate_fan.png)", ""]

    # empirical durations and rep bonds
    emp = scenarios.empirical_durations(rets, fred_m)
    emp.to_csv(REPORTS_DIR / "stage2_empirical_durations.csv", index=False)
    L += [
        "## Empirical durations (monthly ETF return on yield change at representative maturity)",
        "",
        "`published_approx` are rough fact-sheet figures quoted from memory, NOT fetched.",
        "",
        emp.round(3).to_markdown(index=False),
        "",
    ]
    rb = pd.DataFrame(
        [
            {
                "etf": k,
                "maturity": v.bond.maturity,
                "coupon": v.bond.coupon,
                "ytm": v.ytm,
                "par_duration": v.par_duration,
                "rate_beta": v.rate_beta,
                "duration_used": v.duration,
                "convexity_used": v.convexity,
                "source": v.source,
            }
            for k, v in fm.rep.items()
        ]
    )
    L += [
        "Representative bonds used for the mapping and stress tests:",
        "",
        rb.round(4).to_markdown(index=False),
        "",
        f"Spread betas to the Baa factor: {', '.join(f'{k} {v:.2f}' for k, v in fm.spread_betas.items())}.",
        "",
    ]

    # factor model vs bootstrap
    R = scenarios.asset_returns(fm, fp, SEED)
    boot = simulate_returns(bs.mu, cov, SimulationConfig(method="bootstrap", n_paths=FAN_PATHS), history=a)
    comp = []
    for i, asset in enumerate(assets):
        f = R[asset]
        b_ = boot[:, :, i]
        f1, b1 = np.prod(1 + f[:, :12], axis=1) - 1, np.prod(1 + b_[:, :12], axis=1) - 1
        comp.append(
            {
                "asset": asset,
                "factor mean": f.mean() * 12,
                "bootstrap mean": b_.mean() * 12,
                "factor vol": f.std() * np.sqrt(12),
                "bootstrap vol": b_.std() * np.sqrt(12),
                "historical vol": a[asset].std() * np.sqrt(12),
                "factor 1y 5th pct": np.quantile(f1, 0.05),
                "bootstrap 1y 5th pct": np.quantile(b1, 0.05),
                "factor corr with SPY": np.corrcoef(f.ravel(), R["SPY"].ravel())[0, 1],
                "historical corr with SPY": a[asset].corr(a["SPY"]),
            }
        )
    comp_df = pd.DataFrame(comp).set_index("asset")
    pct_cols = [c for c in comp_df.columns if "corr" not in c]
    comp_show = comp_df.copy()
    comp_show[pct_cols] = _pct(comp_show[pct_cols], 1)
    L += [
        "## Factor model vs historical bootstrap (monthly simulated returns, 10,000 paths)",
        "",
        "Bootstrap means are Bayes-Stein (Stage 1b); factor-model bond means come from yields plus the simulated "
        "rate path. BIL in the factor model earns the short rate; it is not in this table.",
        "",
        comp_show.round(3).to_markdown(),
        "",
    ]

    # yield-based expected returns
    y_er = scenarios.yield_based_expected_returns(curve, fm.oas0)
    mu_new = bs.mu.copy()
    mu_new[y_er.index] = y_er
    er = pd.DataFrame({"Bayes-Stein (history)": bs.mu, "yield-based (Stage 2)": mu_new})
    er["changed"] = er.index.isin(y_er.index)
    L += [
        "## Yield-based expected returns",
        "",
        "Bond ETFs: par yield at the representative maturity (credit: plus today's ICE OAS minus expected loss "
        f"IG {scenarios.EXPECTED_LOSS['LQD']:.2%}, HY {scenarios.EXPECTED_LOSS['HYG']:.2%}). Equities, REITs, "
        f"gold keep Bayes-Stein. Risk-free: historical BIL mean {rf_hist:.2%} before, today's 3M yield "
        f"{rf_now:.2%} after.",
        "",
        _pct(er.drop(columns="changed")).assign(changed=er["changed"]).to_markdown(),
        "",
    ]
    tvc = pd.DataFrame(
        {
            "excess over cash before": bs.mu[["SHY", "IEF", "TLT"]] - rf_hist,
            "excess over cash after": mu_new[["SHY", "IEF", "TLT"]] - rf_now,
        }
    )
    L += ["Treasuries vs cash:", "", _pct(tvc).to_markdown(), ""]

    before = optimized(bs.mu, cov, rf_hist)
    after = optimized(mu_new, cov, rf_now)
    w = pd.concat(
        {**{f"{k} (before)": v for k, v in before.items()}, **{f"{k} (after)": v for k, v in after.items()}}, axis=1
    )
    L += ["### Optimized weights before vs after", "", _pct(w, 1).to_markdown(), ""]
    stats = []
    for label, mu_, rf_, ports in (("before", bs.mu, rf_hist, before), ("after", mu_new, rf_now, after)):
        for name, wt in ports.items():
            m_ = float(wt @ mu_)
            v_ = float(np.sqrt(wt @ cov.values @ wt))
            stats.append(
                {"set": label, "portfolio": name, "exp_return": m_, "volatility": v_, "sharpe": (m_ - rf_) / v_}
            )
    st = pd.DataFrame(stats).set_index(["set", "portfolio"])
    L += [
        st.assign(
            exp_return=_pct(st.exp_return), volatility=_pct(st.volatility), sharpe=st.sharpe.round(3)
        ).to_markdown(),
        "",
    ]
    fr = efficient_frontier(
        PortfolioProblem(mu_new, cov, bounds=ILLUSTRATIVE_BOUNDS, groups=ILLUSTRATIVE_GROUPS, risk_free_rate=rf_now),
        n_points=12,
    )
    fr.to_csv(REPORTS_DIR / "stage2_frontier_constrained.csv", index=False)
    L += [
        "Constrained efficient frontier, yield-based inputs:",
        "",
        _pct(fr[["expected_return", "volatility"]])
        .assign(sharpe=fr.sharpe.round(3), success=fr.success)
        .to_markdown(index=False),
        "",
    ]

    # duration-based stress
    ds = stress.duration_stress(fm.rep, tenors, par_today)
    ds.to_csv(REPORTS_DIR / "stage2_duration_stress.csv", index=False)
    show = ds.set_index(["shock", "etf"])[["dy_bp", "duration_convexity", "full_revaluation", "approx_error"]].copy()
    show[["duration_convexity", "full_revaluation", "approx_error"]] *= 100
    L += [
        "## Duration-based stress tests (Stage 2)",
        "",
        "Shocks to the Treasury par curve or credit OAS, applied through key rate durations and convexity of each "
        "ETF's representative bond (rate beta applied). `dy_bp` is the effective shock at the bond's maturity. "
        "Equities, REITs, gold and BIL are not shocked (0%).",
        "",
        show.round(3).to_markdown(),
        "",
    ]
    ports = {
        **{f"{k} (before)": v for k, v in before.items()},
        **{f"{k} (after)": v for k, v in after.items()},
        "Equal weight": pd.Series(1 / len(assets), index=assets),
    }
    L += [
        "Portfolio impact (full revaluation, %):",
        "",
        _pct(stress.duration_stress_portfolios(ds, ports)).to_markdown(),
        "",
    ]
    shocks = dict(stress.HYPOTHETICAL_SHOCKS)
    L += [
        "For comparison, the Stage 1b SIMPLE FACTOR SHOCKS (judgment-based, not duration-based), same portfolios:",
        "",
        _pct(
            pd.DataFrame({n: {s: stress.apply_shock(wt, sh) for s, sh in shocks.items()} for n, wt in ports.items()})
        ).to_markdown(),
        "",
    ]
    chart_duration(fm.rep["TLT"], REPORTS_DIR / "stage2_duration_vs_revaluation.png")
    L += ["![Duration vs revaluation](stage2_duration_vs_revaluation.png)", ""]

    out = REPORTS_DIR / "stage2_results.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

"""Stage 1b: Monte Carlo, stress tests and regimes. Writes reports/stage1b_*.

python -m engine.run_stage1b
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import data, stress
from engine import estimation as est
from engine import regimes as rg
from engine.montecarlo import METHODS, SimulationConfig, risk_metrics, simulate_portfolio
from engine.optimizer import PortfolioProblem, max_sharpe, min_variance
from engine.run_stage1 import ILLUSTRATIVE_BOUNDS, ILLUSTRATIVE_GROUPS, REPORTS_DIR


def _plt():
    """pyplot with a file-only backend, imported only when charts are drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


TARGET_ANNUAL_RETURN = 0.04  # illustrative hurdle for "probability of missing target"

# Validated categorical palette (dataviz reference, light mode), fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def build_inputs():
    prices, _ = data.clean_prices(data.load_prices())
    rets = data.monthly_returns(prices)
    a = rets[list(data.UNIVERSE)]
    rf = float(rets[data.RISK_FREE_TICKER].mean() * est.PERIODS_PER_YEAR)
    mu, cov = est.bayes_stein_mean(a).mu, est.estimate_covariance(a)
    portfolios = {
        "Min variance": min_variance(PortfolioProblem(mu, cov)).weights,
        "Max Sharpe": max_sharpe(PortfolioProblem(mu, cov, risk_free_rate=rf)).weights,
        "Constrained max Sharpe": max_sharpe(
            PortfolioProblem(mu, cov, bounds=ILLUSTRATIVE_BOUNDS, groups=ILLUSTRATIVE_GROUPS, risk_free_rate=rf)
        ).weights,
        # Benchmark, NOT optimized: gives the shocks something to bite on,
        # since the optimized portfolios hold no credit at all.
        "Equal weight (benchmark)": pd.Series(1 / len(mu), index=mu.index),
    }
    return prices, rets, a, rf, mu, cov, portfolios


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_tails(sims: dict, path):
    """1-year return distribution, normal vs bootstrap, two portfolios."""
    plt = _plt()
    names = ["Constrained max Sharpe", "Equal weight (benchmark)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, name in zip(axes, names, strict=True):
        _style(ax)
        lo = min(np.quantile(sims[(name, m)].surplus[:, 12] - 1, 0.0005) for m in ("normal", "bootstrap"))
        hi = max(np.quantile(sims[(name, m)].surplus[:, 12] - 1, 0.9995) for m in ("normal", "bootstrap"))
        bins = np.linspace(lo, hi, 70)
        for color, m in zip(COLORS[:2], ("normal", "bootstrap"), strict=True):
            g = sims[(name, m)].surplus[:, 12] - 1
            v = risk_metrics(sims[(name, m)], 12)["VaR_99"]
            label = f"{m.capitalize()} (99% VaR: {-v:.1%})"
            ax.hist(g * 100, bins=bins * 100, histtype="step", linewidth=2, color=color, label=label)
            ax.axvline(-v * 100, color=color, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.set_title(name, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("1-year return (%)", color=INK_2, fontsize=9)
        ax.set_ylabel("Number of simulated paths", color=INK_2, fontsize=9)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.4)  # headroom so the legend clears the data
        ax.legend(
            frameon=False,
            fontsize=9,
            labelcolor=INK,
            title="Dashed line = 99% VaR",
            title_fontsize=8.5,
            loc="upper right",
        )
    fig.suptitle(
        "Normal vs bootstrap: same mean and volatility, different left tail", color=INK, fontsize=12, x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_stress(hist: pd.DataFrame, path):
    """Total return in each historical scenario, grouped by portfolio."""
    plt = _plt()
    scen = [s.name for s in stress.HISTORICAL_SCENARIOS]
    ports = list(dict.fromkeys(hist.portfolio))
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    _style(ax)
    width = 0.8 / len(ports)
    x = np.arange(len(scen))
    for i, (p, color) in enumerate(zip(ports, COLORS, strict=True)):
        vals = [float(hist[(hist.scenario == s) & (hist.portfolio == p)].total_return.iloc[0]) * 100 for s in scen]
        bars = ax.bar(x + (i - (len(ports) - 1) / 2) * width, vals, width * 0.9, color=color, label=p)
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v:.1f}",
                (b.get_x() + b.get_width() / 2, v),
                xytext=(0, 3 if v >= 0 else -11),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK_2,
            )
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(x, scen)
    ax.set_ylabel("Total return over the window (%)", color=INK_2, fontsize=9)
    ax.set_title("Historical stress windows: total return by portfolio", color=INK, fontsize=12, loc="left", pad=28)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=INK)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _pct(df, d=1):
    return (df * 100).round(d)


def main() -> None:
    prices, rets, a, rf, mu, cov, portfolios = build_inputs()
    lines = [
        "# Stage 1b results: Monte Carlo, stress tests, regimes",
        "",
        "Generated by `python -m engine.run_stage1b`. Percent unless stated. Window "
        f"{a.index.min():%Y-%m} to {a.index.max():%Y-%m} ({len(a)} months). Expected returns: Bayes-Stein. "
        "Covariance: Ledoit-Wolf on correlations. Portfolio weights are from Stage 1, estimated on the FULL "
        "sample, so every result below is in-sample (see DECISIONS.md honest limits).",
        "",
        "## Portfolios",
        "",
        _pct(pd.DataFrame(portfolios)).to_markdown(),
        "",
    ]

    # ------------------------------------------------------------ Monte Carlo
    base = SimulationConfig()
    lines += [
        "## Monte Carlo",
        "",
        f"{base.n_paths:,} paths, {base.horizon_months} months, seed {base.seed}, monthly rebalancing. "
        f"Bootstrap: circular blocks of {base.block_length} months, mean-adjusted to the Bayes-Stein returns. "
        f"Student-t: {base.t_df:g} degrees of freedom. VaR and CVaR are losses (positive = loss; negative means "
        "even that bad case is a gain). Target for 'miss target': "
        f"{TARGET_ANNUAL_RETURN:.0%} a year, illustrative.",
        "",
    ]
    sims, rows = {}, []
    for pname, w in portfolios.items():
        for m in METHODS:
            res = simulate_portfolio(w, mu, cov, SimulationConfig(method=m), history=a)
            sims[(pname, m)] = res
            for h in (12, base.horizon_months):
                x = risk_metrics(res, h, TARGET_ANNUAL_RETURN)
                rows.append({"portfolio": pname, "method": m, "horizon": f"{h // 12}y", **x})
    mc = pd.DataFrame(rows)
    mc.to_csv(REPORTS_DIR / "stage1b_montecarlo.csv", index=False)
    show = [
        "VaR_95",
        "CVaR_95",
        "VaR_99",
        "CVaR_99",
        "prob_loss",
        "prob_miss_target",
        "median_max_drawdown",
        "p95_max_drawdown",
        "p99_max_drawdown",
    ]
    for h in ("1y", f"{base.horizon_months // 12}y"):
        t = mc[mc.horizon == h].set_index(["portfolio", "method"])[show]
        lines += [f"### Risk metrics, {h} horizon", "", _pct(t).to_markdown(), ""]
    tw = mc[mc.horizon == f"{base.horizon_months // 12}y"].set_index(["portfolio", "method"])[
        [
            "p01_terminal_surplus",
            "p05_terminal_surplus",
            "median_terminal_surplus",
            "p95_terminal_surplus",
            "mean_terminal_surplus",
        ]
    ]
    lines += ["### Terminal wealth after 5 years (starting at 1.00)", "", tw.round(3).to_markdown(), ""]

    # tail understatement
    tail = []
    for pname, w in portfolios.items():
        n1 = mc[(mc.portfolio == pname) & (mc.method == "normal") & (mc.horizon == "1y")].iloc[0]
        b1 = mc[(mc.portfolio == pname) & (mc.method == "bootstrap") & (mc.horizon == "1y")].iloc[0]
        t1 = mc[(mc.portfolio == pname) & (mc.method == "student_t") & (mc.horizon == "1y")].iloc[0]
        pr = a[list(w.index)] @ w
        worst12 = float(((1 + pr).rolling(12).apply(np.prod, raw=True) - 1).min())
        tail.append(
            {
                "portfolio": pname,
                "CVaR99 normal": n1.CVaR_99,
                "CVaR99 student_t": t1.CVaR_99,
                "CVaR99 bootstrap": b1.CVaR_99,
                "normal understates CVaR99 by": 1 - n1.CVaR_99 / b1.CVaR_99,
                "VaR99 normal": n1.VaR_99,
                "VaR99 bootstrap": b1.VaR_99,
                "normal understates VaR99 by": 1 - n1.VaR_99 / b1.VaR_99,
                "worst actual 12m return": worst12,
                "monthly skew": pr.skew(),
                "monthly excess kurtosis": pr.kurt(),
            }
        )
    tail_df = pd.DataFrame(tail).set_index("portfolio")
    pct_cols = [c for c in tail_df.columns if "skew" not in c and "kurt" not in c]
    tail_show = tail_df.copy()
    tail_show[pct_cols] = _pct(tail_show[pct_cols])
    tail_show[["monthly skew", "monthly excess kurtosis"]] = tail_show[
        ["monthly skew", "monthly excess kurtosis"]
    ].round(2)
    lines += [
        "### How much does the normal model understate tail risk? (1-year horizon)",
        "",
        tail_show.to_markdown(),
        "",
    ]

    blocks = []
    w_c = portfolios["Constrained max Sharpe"]
    for b in (1, 3, 6, 12):
        res = simulate_portfolio(w_c, mu, cov, SimulationConfig(method="bootstrap", block_length=b), history=a)
        x = risk_metrics(res, 12)
        blocks.append(
            {
                "block_months": b,
                "VaR_99": x["VaR_99"],
                "CVaR_99": x["CVaR_99"],
                "p99_max_drawdown_5y": risk_metrics(res)["p99_max_drawdown"],
            }
        )
    lines += [
        "### Bootstrap block-length sensitivity (constrained max Sharpe, 1-year VaR/CVaR)",
        "",
        _pct(pd.DataFrame(blocks).set_index("block_months"), 2).to_markdown(),
        "",
    ]
    chart_tails(sims, REPORTS_DIR / "stage1b_tails.png")
    lines += ["![Normal vs bootstrap tails](stage1b_tails.png)", ""]

    # ----------------------------------------------------------------- stress
    hist = stress.historical_scenarios(prices[list(data.UNIVERSE)], a, portfolios)
    hist.to_csv(REPORTS_DIR / "stage1b_stress_historical.csv", index=False)
    lines += [
        "## Historical stress windows",
        "",
        "Daily marks, monthly rebalancing. `max_drawdown_monthly` uses month ends only, to show what monthly "
        "data hides.",
        "",
    ]
    for sc in stress.HISTORICAL_SCENARIOS:
        lines.append(f"- **{sc.name}** ({sc.start} to {sc.end}): {sc.why}")
    h = hist.set_index(["scenario", "portfolio"])[["total_return", "max_drawdown_daily", "max_drawdown_monthly"]]
    lines += [
        "",
        _pct(h).to_markdown(),
        "",
        "### Asset returns in each window",
        "",
        _pct(stress.asset_scenario_returns(a)).to_markdown(),
        "",
    ]
    chart_stress(hist, REPORTS_DIR / "stage1b_stress.png")
    lines += ["![Historical stress](stage1b_stress.png)", ""]

    shocks = dict(stress.HYPOTHETICAL_SHOCKS)
    shocks["All three at once"] = stress.combined_shock(*stress.HYPOTHETICAL_SHOCKS)
    lines += [
        "## Hypothetical shocks (simple factor shocks, NOT duration-based)",
        "",
        "Instant moves, no rebalancing, unlisted assets unchanged. Assumptions are in `src/engine/stress.py` "
        "and DECISIONS.md D-025. Stage 2 replaces rate and spread shocks with duration-based bond math.",
        "",
        "Shock per asset:",
        "",
        _pct(pd.DataFrame(shocks).fillna(0.0)).to_markdown(),
        "",
        "Portfolio impact:",
        "",
        _pct(stress.hypothetical_report(portfolios), 2).to_markdown(),
        "",
    ]

    # ---------------------------------------------------------------- regimes
    sig = data.load_signals()
    labels = rg.label_regimes(sig, a.index)
    labels.to_csv(REPORTS_DIR / "stage1b_regime_labels.csv")
    lines += [
        "## Rules-based regimes",
        "",
        f"- Rates: rising if the 10y Treasury yield is above its level {rg.RATE_LOOKBACK_MONTHS} months earlier.",
        f"- Volatility: high if SPY's trailing {rg.VOL_WINDOW_DAYS}-day realized volatility (annualized) is above "
        f"{rg.VOL_THRESHOLD:.0%}.",
        "- Each month is labeled with data up to the end of the PREVIOUS month (no look-ahead).",
        "",
        "Months in each combination:",
        "",
        pd.crosstab(labels.rates, labels.volatility).to_markdown(),
        "",
        f"Checks: 2022 rate labels = {labels.loc['2022', 'rates'].value_counts().to_dict()}; "
        f"2008 volatility labels = {labels.loc['2008', 'volatility'].value_counts().to_dict()}; "
        f"Sep 2008 to Jun 2009 all high = {bool((labels.loc['2008-09':'2009-06', 'volatility'] == 'high').all())}.",
        "",
    ]
    full = []
    for name, w in portfolios.items():
        pr = a[list(w.index)] @ w
        m, v = pr.mean() * 12, pr.std() * np.sqrt(12)
        full.append({"portfolio": name, "ann_return": m, "ann_vol": v, "sharpe": (m - rf) / v})
    full = pd.DataFrame(full).set_index("portfolio")
    for dim in ("rates", "volatility"):
        out = rg.regime_summary(a, labels[dim], portfolios, rets[data.RISK_FREE_TICKER])
        s = out["summary"].copy()
        s["avg_correlation"] = s["avg_correlation"].round(3)
        p = out["portfolios"].set_index(["regime", "portfolio"])
        p_show = _pct(p[["ann_return", "ann_vol", "worst_month", "pct_months_negative"]])
        p_show["sharpe"] = p["sharpe"].round(2)
        lines += [
            f"### {dim.capitalize()} regime",
            "",
            s.to_markdown(),
            "",
            "Asset annualized mean and volatility:",
            "",
            _pct(out["assets"]).to_markdown(),
            "",
            "Portfolio performance:",
            "",
            p_show.to_markdown(),
            "",
        ]
    full_show = _pct(full[["ann_return", "ann_vol"]])
    full_show["sharpe"] = full["sharpe"].round(2)
    lines += ["### Full sample, for comparison", "", full_show.to_markdown(), ""]

    out_path = REPORTS_DIR / "stage1b_results.md"
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

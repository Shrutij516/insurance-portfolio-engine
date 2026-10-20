"""Liability discount-rate variant: Treasury curve plus a share of the IG spread.

    python -m engine.run_discount_variant

Writes reports/discount_variant_results.md, CSVs and a chart (D-063 to D-065).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import alm
from engine import liabilities as lb
from engine.run_stage1 import REPORTS_DIR
from engine.run_stage1b import COLORS, INK, INK_2, SURFACE, _plt, _style
from engine.run_stage3 import DURATION_BAND, SURPLUS_GROWTH_TARGET, build_context

FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]  # share of the IG spread used as the discount add-on
RECOMMENDED_FRACTION = 0.5  # illiquidity premium assumption (D-063)
FUNDING_RATIO = 1.10  # held constant across variants for the optimization comparison (D-065)


def analyse(ctx, frac: float) -> dict:
    ig0 = ctx.fm.oas0["LQD"]
    s0 = frac * ig0
    m = lb.measures(ctx.block, ctx.tenors, ctx.y0, spread=s0)
    pv = m["pv"]
    a0 = FUNDING_RATIO * pv
    spread_path = frac * ctx.scen.ig_spread
    L = alm.liability_paths(ctx.block, ctx.scen, spread_path=spread_path, n_steps=12)
    R1, rL = alm.one_year_inputs(ctx.scen, L, ctx.block)
    mu, cov, liab = alm.scenario_moments(R1, rL, pv / a0)
    krd = alm.etf_key_rate_durations(ctx.fm.rep, ctx.tenors, ctx.y0)
    rate_chg = alm.etf_curve_changes(ctx.fm.rep, ctx.tenors, ctx.y0, alm.RATE_SHOCK)
    liab_rate = alm.liability_curve_changes(ctx.block, ctx.tenors, ctx.y0, alm.RATE_SHOCK, spread=s0)
    lcr = alm.lcr_coefficients(
        alm.liquidity_stress_moves(ctx.fm.rep, ctx.tenors, ctx.y0, severe=True), a0, alm.stressed_outflow(ctx.block, pv)
    )
    s_tot = a0 - pv
    rules = alm.capital_rules(a0, s_tot / alm.COVERAGE_TARGET, rate_chg, liab_rate)
    band = alm.duration_band_rule(krd, m["krd"], a0, pv, DURATION_BAND * pv / 1000)
    target = SURPLUS_GROWTH_TARGET * s_tot / a0
    out = {
        "fraction_of_IG": frac,
        "discount_spread": s0,
        "liability_pv": pv,
        "liability_effective_duration": m["effective_duration"],
        "liability_macaulay": m["macaulay"],
        "liability_dv01": m["dv01"],
        "liability_spread_dv01": m["spread_dv01"],
        "funding_ratio_assets_1100": 1100.0 / pv,
        "corr_LQD_vs_liability": float(np.corrcoef(R1["LQD"], rL)[0, 1]),
        "corr_HYG_vs_liability": float(np.corrcoef(R1["HYG"], rL)[0, 1]),
        "corr_TLT_vs_liability": float(np.corrcoef(R1["TLT"], rL)[0, 1]),
    }
    ig_move = alm.stress.duration_stress(ctx.fm.rep, ctx.tenors, ctx.y0, curve_shocks={})
    lqd_ig = float(ig_move[(ig_move.etf == "LQD")].full_revaluation.iloc[0])
    weights = {}
    for label, floor in (("free", False), ("headline", True)):
        p = alm.insurer_problem(mu, cov, liab, lcr, [*rules, band], ig_floor=floor)
        w, status = alm.solve_for_target(p, target)
        dS = alm.surplus_change(w, R1, rL, a0, pv)
        # IG spreads +100bp: assets lose LQD x its spread move; liabilities fall by the
        # discount-spread move (frac x 100bp) times the spread DV01
        d_assets = a0 * w["LQD"] * lqd_ig
        d_liab = -m["spread_dv01"] * 100 * frac
        out.update(
            {
                f"{label}_LQD": w["LQD"],
                f"{label}_TLT": w["TLT"],
                f"{label}_SHY": w["SHY"],
                f"{label}_IEF": w["IEF"],
                f"{label}_GLD": w["GLD"],
                f"{label}_SPY": w["SPY"],
                f"{label}_surplus_vol": float(dS.std(ddof=1)),
                f"{label}_CVaR95": alm.cvar_worst_k(-dS, 0.95),
                f"{label}_capital": alm.capital_charge(w, a0, rate_chg, liab_rate)["total"],
                f"{label}_IG_shock_surplus_change": d_assets - d_liab,
                f"{label}_status": status,
            }
        )
        weights[label] = w
    out["_weights"] = weights
    out["surplus"] = s_tot
    out["assets"] = a0
    return out


def chart(df: pd.DataFrame, path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=SURFACE)
    x = df["fraction_of_IG"] * 100
    ax = axes[0]
    _style(ax)
    ax.plot(x, df["free_LQD"] * 100, color=COLORS[0], linewidth=2, marker="o", label="Credit chosen freely (no floor)")
    ax.plot(
        x,
        df["headline_LQD"] * 100,
        color=COLORS[1],
        linewidth=2,
        marker="o",
        linestyle=(0, (4, 3)),
        label="Credit with the 40% floor",
    )
    ax.set_xlabel("Liability discount spread (% of the IG spread)", color=INK_2, fontsize=9)
    ax.set_ylabel("LQD weight (%)", color=INK_2, fontsize=9)
    ax.set_title("Does the optimizer want credit?", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    _style(ax)
    ax.plot(x, df["corr_LQD_vs_liability"], color=COLORS[0], linewidth=2, marker="o", label="LQD")
    ax.plot(x, df["corr_TLT_vs_liability"], color=COLORS[2], linewidth=2, marker="o", label="TLT")
    ax.plot(x, df["corr_HYG_vs_liability"], color=COLORS[3], linewidth=2, marker="o", label="HYG")
    ax.set_xlabel("Liability discount spread (% of the IG spread)", color=INK_2, fontsize=9)
    ax.set_ylabel("Correlation with 1-year liability return", color=INK_2, fontsize=9)
    ax.set_title("How closely assets track the liabilities", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ctx = build_context("forward")
    rows = [analyse(ctx, f) for f in FRACTIONS]
    weights = {f"{r['fraction_of_IG']:.0%} {k}": v for r in rows for k, v in r.pop("_weights").items()}
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "discount_variant_summary.csv", index=False)
    pd.DataFrame(weights).to_csv(REPORTS_DIR / "discount_variant_weights.csv")
    chart(df, REPORTS_DIR / "discount_variant.png")
    ig0 = ctx.fm.oas0["LQD"]
    rec = df[df.fraction_of_IG == RECOMMENDED_FRACTION].iloc[0]
    base = df[df.fraction_of_IG == 0].iloc[0]
    val = df[
        [
            "fraction_of_IG",
            "discount_spread",
            "liability_pv",
            "liability_effective_duration",
            "liability_macaulay",
            "liability_dv01",
            "liability_spread_dv01",
            "funding_ratio_assets_1100",
            "corr_LQD_vs_liability",
            "corr_HYG_vs_liability",
            "corr_TLT_vs_liability",
        ]
    ].copy()
    val["fraction_of_IG"] = (val["fraction_of_IG"] * 100).round(0).astype(int).astype(str) + "%"
    val["discount_spread"] = (val["discount_spread"] * 1e4).round(1)
    opt_cols = [c for c in df.columns if c.startswith(("free_", "headline_"))]
    opt = df[["fraction_of_IG", *opt_cols]].copy()
    opt["fraction_of_IG"] = (opt["fraction_of_IG"] * 100).round(0).astype(int).astype(str) + "%"
    L = [
        "# Liability discount-rate variant",
        "",
        "Generated by `python -m engine.run_discount_variant`. $ millions. Liabilities discounted at the Treasury "
        f"zero curve plus a share of the investment grade spread (today's ICE IG OAS {ig0:.2%}), as an illiquidity "
        f"premium. Recommended assumption: {RECOMMENDED_FRACTION:.0%} of the IG spread "
        f"({RECOMMENDED_FRACTION * ig0 * 1e4:.0f}bp); 0% is the Stage 3 base case. In the scenarios the add-on moves "
        "with the simulated IG spread (same historical months as the asset returns). Same cash flows in every "
        "variant. For the optimization, assets are set to 110% of each variant's liability value so the risk budget "
        "is comparable; 'funding_ratio_assets_1100' shows the pure valuation effect with assets fixed at 1,100.",
        "",
        "## Liability value, duration and how assets track it",
        "",
        "`discount_spread` in bp; `liability_spread_dv01` = $m change for 1bp of discount spread.",
        "",
        val.round(4).to_markdown(index=False),
        "",
        "## Re-optimized portfolios under each variant (same constraints, capital proxy and duration band)",
        "",
        "`free` = surplus-aware optimizer WITHOUT the 40% credit floor (what it chooses on its own); `headline` = "
        "with the floor (the recommended rule). `IG_shock_surplus_change` = surplus change if IG spreads widen "
        "100bp (assets: LQD revaluation; liabilities: discount spread rises by the same share of 100bp).",
        "",
        opt.round(4).to_markdown(index=False),
        "",
        "![Discount variant](discount_variant.png)",
        "",
        "## Reading",
        "",
        f"- Liability PV falls from {base.liability_pv:.1f} (Treasury) to {rec.liability_pv:.1f} at "
        f"{RECOMMENDED_FRACTION:.0%} of IG and {df.iloc[-1].liability_pv:.1f} at 100%; funding ratio with assets "
        f"fixed at 1,100 rises from {base.funding_ratio_assets_1100:.1%} to {rec.funding_ratio_assets_1100:.1%} "
        f"and {df.iloc[-1].funding_ratio_assets_1100:.1%}.",
        f"- Correlation of LQD with the liability return: {base.corr_LQD_vs_liability:.2f} at 0% vs "
        f"{rec.corr_LQD_vs_liability:.2f} at {RECOMMENDED_FRACTION:.0%} and {df.iloc[-1].corr_LQD_vs_liability:.2f} "
        "at 100%.",
        f"- Credit chosen freely (no floor): {base.free_LQD:.1%} at 0%, {rec.free_LQD:.1%} at "
        f"{RECOMMENDED_FRACTION:.0%}, {df.iloc[-1].free_LQD:.1%} at 100%.",
        f"- Headline portfolio LQD / TLT / SHY: {base.headline_LQD:.1%} / {base.headline_TLT:.1%} / "
        f"{base.headline_SHY:.1%} at 0% vs {rec.headline_LQD:.1%} / {rec.headline_TLT:.1%} / "
        f"{rec.headline_SHY:.1%} at {RECOMMENDED_FRACTION:.0%}; surplus volatility {base.headline_surplus_vol:.1f} "
        f"vs {rec.headline_surplus_vol:.1f}.",
        f"- IG spreads +100bp, headline portfolio: surplus change {base.headline_IG_shock_surplus_change:.1f} at 0% "
        f"vs {rec.headline_IG_shock_surplus_change:.1f} at {RECOMMENDED_FRACTION:.0%}.",
        "",
    ]
    (REPORTS_DIR / "discount_variant_results.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

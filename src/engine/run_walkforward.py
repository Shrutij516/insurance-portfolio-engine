"""Walk-forward (out-of-sample) test. Writes reports/walkforward_results.md and charts.

python -m engine.run_walkforward
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import data
from engine import fixedincome as fi
from engine import liabilities as lb
from engine import scenarios as sc
from engine import walkforward as wf
from engine.run_stage1 import REPORTS_DIR
from engine.run_stage1b import COLORS, INK, INK_2, SURFACE, _plt, _style

HEADLINE = "RECOMMENDED: surplus + capital aware, duration band"
NAMES = ["Walk-forward optimized", "Static in-sample headline", "Equal weight", "Key-rate matched Treasuries"]


def load():
    p, _ = data.clean_prices(data.load_prices())
    rets = data.monthly_returns(p)
    fred = data.monthly_fred(data.load_fred(), rets.index)
    return rets, fred


def run_mode(rets, fred, cfg: wf.WFConfig, reset: bool):
    tenors, par = sc.par_curve_history(fred)
    t0 = pd.Timestamp(cfg.first_rebalance)
    block0 = lb.annuity_block(fi.bootstrap_zero_curve(tenors, par.loc[t0].values), pv=cfg.l0)
    static = pd.read_csv(REPORTS_DIR / "stage3_weights.csv", index_col=0)[HEADLINE]
    equal = pd.Series(1 / len(wf.ASSETS), index=wf.ASSETS)
    decisions: dict = {}

    def optimized(t, block_t, a_t):
        d = wf.decide(t, rets, fred, block_t, a_t, cfg)
        decisions[t] = d
        return d.weights

    def matched(t, block_t, a_t):
        return wf.decide(t, rets, fred, block_t, a_t, cfg).matched

    policies = [optimized, lambda t, b, a: static, lambda t, b, a: equal, matched]
    paths = [wf.realize(n, rets, fred, block0, cfg, pol, reset=reset) for n, pol in zip(NAMES, policies, strict=True)]
    return paths, decisions


def chart_funding(paths, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    _style(ax)
    for p, color in zip(paths, COLORS, strict=True):
        fr = p.funding_ratio * 100
        ax.plot(fr.index, fr.values, color=color, linewidth=2, label=p.name)
    ax.axhline(100, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="#e4e3df", alpha=0.6, linewidth=0)
    ax.annotate("2022", (pd.Timestamp("2022-02-01"), ax.get_ylim()[1] * 0.97), fontsize=9, color=INK_2)
    ax.set_ylabel("Funding ratio (%)", color=INK_2, fontsize=9)
    ax.set_title(
        "Walk-forward, realized history: funding ratio of one runoff block (start Dec 2010 at 110%)",
        color=INK,
        fontsize=12,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_calibration(cal: pd.DataFrame, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(cal))
    ax.bar(x, cal["realized"], 0.6, color=COLORS[0], label="Realized surplus change")
    ax.plot(
        x,
        -cal["pred_var95"],
        color=COLORS[1],
        linewidth=2,
        marker="o",
        markersize=5,
        label="Predicted 1-in-20 loss threshold (VaR 95%)",
    )
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(x, cal.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"\$ millions (each year starts at surplus \$100m)", color=INK_2, fontsize=9)
    ax.set_title(
        r"Out-of-sample risk check: walk-forward portfolio, fresh \$1.1bn / \$1bn balance sheet each year",
        color=INK,
        fontsize=12,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _r(df, d=1):
    return df.round(d)


def main() -> None:
    rets, fred = load()
    cfg = wf.WFConfig()
    prim, dec_p = run_mode(rets, fred, cfg, reset=False)
    rst, dec_r = run_mode(rets, fred, cfg, reset=True)

    summ = pd.DataFrame([wf.summarize(p) for p in prim]).set_index("strategy")
    annual = wf.annual_table(prim)
    reset_tab = pd.DataFrame({p.name: wf.reset_year_changes(p, cfg) for p in rst})
    full_years = reset_tab.loc[[i for i in reset_tab.index if len(i) == 4]]
    reset_summ = pd.DataFrame(
        {
            "mean surplus change": full_years.mean(),
            "vol of surplus change": full_years.std(ddof=1),
            "worst year change": full_years.min(),
            "worst year": full_years.idxmin(),
            "2022 change": full_years.loc["2022"],
            "years with a loss": (full_years < 0).sum(),
        }
    )
    rows = []
    for t, d in dec_r.items():
        nxt = str(t.year + 1)
        key = nxt if nxt in reset_tab.index else next((i for i in reset_tab.index if i.startswith(nxt)), None)
        if key is None:
            continue
        rows.append(
            {
                "year": key,
                "status": d.status,
                "pred_mean": d.details.get("pred_mean", np.nan),
                "pred_vol": d.details.get("pred_vol", np.nan),
                "pred_var95": d.details.get("pred_var95", np.nan),
                "pred_cvar95": d.details.get("pred_cvar95", np.nan),
                "realized": reset_tab.loc[key, "Walk-forward optimized"],
            }
        )
    cal = pd.DataFrame(rows).set_index("year")
    cal_full = cal.loc[[i for i in cal.index if len(i) == 4]]
    exceed = int((cal_full["realized"] < -cal_full["pred_var95"]).sum())

    def dec_table(decs):
        out = []
        for t, d in decs.items():
            w = d.weights[d.weights > 0.005].round(3)
            out.append(
                {
                    "rebalance": str(t.date()),
                    "status": d.status,
                    "surplus": d.details.get("surplus"),
                    "target": d.details.get("target"),
                    "gap_per_100bp": d.details.get("gap_per_100bp"),
                    "capital": d.details.get("capital"),
                    "violations": len(d.details.get("violations", [])),
                    "weights": ", ".join(f"{k} {v:.0%}" for k, v in w.items()),
                }
            )
        return pd.DataFrame(out)

    dp, dr = dec_table(dec_p), dec_table(dec_r)
    for name, frame in (
        ("walkforward_primary_summary", summ),
        ("walkforward_primary_annual", annual),
        ("walkforward_reset_annual", reset_tab),
        ("walkforward_reset_summary", reset_summ),
        ("walkforward_calibration", cal),
    ):
        frame.to_csv(REPORTS_DIR / f"{name}.csv")
    dp.to_csv(REPORTS_DIR / "walkforward_primary_decisions.csv", index=False)
    dr.to_csv(REPORTS_DIR / "walkforward_reset_decisions.csv", index=False)
    chart_funding(prim, REPORTS_DIR / "walkforward_funding_ratio.png")
    chart_calibration(cal, REPORTS_DIR / "walkforward_calibration.png")

    L = [
        "# Walk-forward (out-of-sample) results",
        "",
        "Generated by `python -m engine.run_walkforward`. $ millions. Rebalance every December from "
        f"{cfg.first_rebalance} using only data available at that date; each following 12 months evaluated on "
        "realized ETF returns and the actual Treasury curve. Synthetic liabilities; capital is a PROXY, not NAIC "
        "RBC. Nothing was tuned after seeing these results.",
        "",
        "Strategies: walk-forward optimized (full headline process re-run each December with point-in-time data); "
        "static in-sample headline (Stage 3b weights, estimated on 2007 to 2026, so it has seen the future); "
        "equal weight; key-rate matched Treasuries (re-matched each December, point-in-time).",
        "",
        "## Primary view (pre-specified): one liability block runs off from Dec 2010",
        "",
        "Assets start at 1,100 against a block worth 1,000 on the Dec 2010 curve; payments each December; no "
        "dividends, so surplus accumulates.",
        "",
        _r(summ.T, 3).to_markdown(),
        "",
        "Surplus change by calendar year:",
        "",
        _r(annual).to_markdown(),
        "",
        "![Funding ratio path](walkforward_funding_ratio.png)",
        "",
        "Walk-forward decisions (primary view):",
        "",
        dp.round(2).to_markdown(index=False),
        "",
        "## Robustness view: fresh balance sheet every December",
        "",
        "Each December the balance sheet is reset to assets 1,100 against a new block worth 1,000 on that date's "
        "curve, so every year's decision is judged on identical terms (surplus 100 at the start of each year).",
        "",
        _r(reset_summ.T).to_markdown(),
        "",
        "Surplus change in each holding year:",
        "",
        _r(reset_tab).to_markdown(),
        "",
        "## Out-of-sample risk calibration (robustness view, walk-forward portfolio)",
        "",
        f"Predicted at each December from that date's scenarios vs realized next year. Realized losses worse than "
        f"the predicted 1-in-20 threshold (VaR 95%): {exceed} of {len(cal_full)} full years (expected about "
        f"{0.05 * len(cal_full):.1f}).",
        "",
        _r(cal).to_markdown(),
        "",
        "![Calibration](walkforward_calibration.png)",
        "",
        "Walk-forward decisions (robustness view):",
        "",
        dr.round(2).to_markdown(index=False),
        "",
    ]
    (REPORTS_DIR / "walkforward_results.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

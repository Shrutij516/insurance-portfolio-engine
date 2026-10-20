"""Client-facing monthly-style report: markdown plus a PDF (D-055).

The PDF is built with matplotlib's PdfPages (already a dependency): text
pages and the PNG charts. No new packages. Every number comes from the
Stage 3 results passed in; nothing is typed by hand.
"""

from __future__ import annotations

import re
import textwrap

from engine.run_stage1 import REPORTS_DIR

REC = "RECOMMENDED: surplus + capital aware, duration band"
ALT = "Asset-only constrained max Sharpe"
MINV = "Asset-only min variance"
MATCH = "Key-rate matched (Treasuries)"
FREE = "Surplus-aware, no credit floor"
UNB = "Comparison: no duration band"


def _m(x: float) -> str:
    """Money in $ millions with the sign in front: -$16.8m, $0.0m."""
    x = 0.0 if abs(x) < 0.05 else float(x)
    return f"{'-' if x < 0 else ''}${abs(x):,.1f}m"


def _build_sections(
    metrics, weights, reg_df, shock_df, sens_df, ctx, a0, s0, outflow, cap, gaps
) -> list[tuple[str, str]]:
    r, alt, mv, mt, fr, ub = (metrics.loc[k] for k in (REC, ALT, MINV, MATCH, FREE, UNB))
    gap_rec = gaps.loc[REC, "dollar_duration_gap_per_100bp"]
    gap_unb = gaps.loc[UNB, "dollar_duration_gap_per_100bp"]
    d_needed = gaps.loc[REC, "liability_duration"] * ctx.l0 / a0
    unb_shocks = shock_df[shock_df.portfolio == UNB].set_index("shock")["surplus_change"]
    w_rec = weights[REC]
    top = w_rec[w_rec > 0.005].sort_values(ascending=False)
    alloc = ", ".join(f"{k} {v:.1%}" for k, v in top.items())
    rec_shocks = shock_df[shock_df.portfolio == "Recommended"].set_index("shock")["surplus_change"]
    worst_shock = rec_shocks.idxmin()
    reg_rec = reg_df.set_index(["dimension", "regime"])
    worst_reg = reg_df.loc[reg_df["mean surplus change $m/yr"].idxmin()]
    worst_month = reg_df.loc[reg_df["worst month $m"].idxmin()]
    sens_fr = sens_df["recommended: funding ratio 5y median"]
    s = []
    s.append(
        (
            "Illustrative portfolio review: synthetic annuity insurer",
            textwrap.dedent(f"""\
        This is an ILLUSTRATIVE report for a SYNTHETIC insurer built for a portfolio project. It is not
        investment advice, and the capital figures are a simplified PROXY, not NAIC risk-based capital.

        Starting position: assets {_m(a0)}, liabilities {_m(ctx.l0)} (a block of payout annuities paying
        for 30 years), surplus {_m(s0)}, funding ratio {a0 / ctx.l0:.0%}. Data through
        {ctx.par.index[-1]:%B %Y}."""),
        )
    )
    s.append(
        (
            "Summary",
            textwrap.dedent(f"""\
        - Recommended allocation: {alloc}.
        - Against an asset-only portfolio built to maximize return per unit of asset risk, it cuts the
          one-year surplus volatility from {_m(alt.surplus_vol_1y)} to {_m(r.surplus_vol_1y)}, and the average
          loss in the worst 1-in-20 year from {_m(alt.surplus_CVaR95_1y)} to {_m(r.surplus_CVaR95_1y)}.
        - Capital used (proxy) falls from {_m(alt.capital_proxy)} to {_m(r.capital_proxy)}; surplus covers
          the proxy {r.coverage_surplus_over_capital:.1f} times (minimum required here: 2.0).
        - Expected surplus growth: {_m(r.exp_surplus_change_1y)} a year (target {_m(0.10 * s0)}), versus
          {_m(alt.exp_surplus_change_1y)} for the asset-only portfolio, which takes far more surplus risk.
        - After 5 years the median funding ratio is {r.funding_ratio_5y_median:.0%}; in the worst 5% of
          scenarios it is {r.funding_ratio_5y_p05:.0%} or lower.
        - Liquidity: liquid assets after a severe stress cover a year of payments plus a 10% surrender wave
          {r.liquidity_coverage:.1f} times."""),
        )
    )
    s.append(
        (
            "Why surplus, not assets",
            textwrap.dedent(f"""\
        The insurer's real risk is the gap between assets and the promises they back. The lowest-risk
        portfolio judged on assets alone (almost all short-term Treasuries) looks safe, but its value barely
        moves when interest rates fall while the liabilities rise sharply. Its worst 1-in-20 year costs
        {_m(mv.surplus_CVaR95_1y)} of surplus on average, more than the whole {_m(s0)} surplus. Matching the
        liabilities' interest rate sensitivity with Treasuries alone cuts that to
        {_m(mt.surplus_CVaR95_1y)}."""),
        )
    )
    s.append(
        (
            "Credit: what the optimizer did",
            textwrap.dedent(f"""\
        Because the liabilities are valued on the Treasury curve, credit spreads add risk to surplus without a
        matching offset. Left free, the optimizer holds no corporate credit (surplus volatility
        {_m(fr.surplus_vol_1y)}). Under the insurer-style rule of at least 40% investment grade credit, it holds
        exactly the minimum 40% and no more; that floor raises surplus volatility to {_m(ub.surplus_vol_1y)}
        (before the duration band, which adds a little more, see below).
        Insurers hold credit for spread income, hold-to-maturity accounting and liability discounting rules
        that this simplified model does not capture, so the floor is a policy choice to review, not a result
        of the optimization."""),
        )
    )
    combo = -(rec_shocks["Parallel +100bp"] + rec_shocks["IG spreads +100bp"])
    combo_unb = -(unb_shocks["Parallel +100bp"] + unb_shocks["IG spreads +100bp"])
    s.append(
        (
            "Interest rate position: kept within a band, and why",
            textwrap.dedent(f"""\
        The recommended portfolio keeps the gap between asset and liability interest rate sensitivity within
        plus or minus $10m of surplus per 1 point move in rates. Its asset duration is
        {gaps.loc[REC, "asset_duration"]:.1f} years against the {d_needed:.1f} needed for an exact match, so
        surplus moves about {_m(gap_rec)} for each 1 point fall in rates.

        Without that band, the optimizer would run a much longer duration (surplus gap {_m(gap_unb)} per 1
        point), because in history Treasury prices tended to rise when credit sold off, so extra Treasuries
        offset losses on the 40% credit holding. That relationship broke in 2022, when rates rose and credit
        spreads widened together. The band trades a little historical efficiency for robustness:
        - Cost: surplus volatility {_m(r.surplus_vol_1y)} vs {_m(ub.surplus_vol_1y)} without the band, and a
          1-in-20 average loss of {_m(r.surplus_CVaR95_1y)} vs {_m(ub.surplus_CVaR95_1y)}.
        - Benefit: a +100bp rate rise costs {_m(-rec_shocks["Parallel +100bp"])} instead of
          {_m(-unb_shocks["Parallel +100bp"])}; a 2022-style rise in rates plus a 100bp investment grade spread
          widening costs about {_m(combo)} instead of {_m(combo_unb)}."""),
        )
    )
    s.append(
        (
            "Stress and regime view",
            textwrap.dedent(f"""\
        Interest rate shocks (surplus change): parallel +100bp {_m(rec_shocks["Parallel +100bp"])}, parallel
        -100bp {_m(rec_shocks["Parallel -100bp"])}, steepener {_m(rec_shocks["Steepener"])}, flattener
        {_m(rec_shocks["Flattener"])}. Credit spread shocks: IG +100bp {_m(rec_shocks["IG spreads +100bp"])},
        HY +400bp {_m(rec_shocks["HY spreads +400bp"])}. The largest single loss is from the {worst_shock} shock
        ({_m(rec_shocks[worst_shock])}).

        Replaying 2007 to 2026 history month by month: in rising-rate months surplus changed by
        {_m(reg_rec.loc[("rates", "rising"), "mean surplus change $m/yr"])} a year on average, in falling-rate
        months {_m(reg_rec.loc[("rates", "falling"), "mean surplus change $m/yr"])}; in high-volatility months
        {_m(reg_rec.loc[("volatility", "high"), "mean surplus change $m/yr"])}, in low-volatility months
        {_m(reg_rec.loc[("volatility", "low"), "mean surplus change $m/yr"])}. On average the weakest
        environment is {worst_reg["regime"]} {"volatility" if worst_reg["dimension"] == "volatility" else "rates"}
        months; the worst single month was {_m(worst_month["worst month $m"])}, in a {worst_month["regime"]}
        {"volatility" if worst_month["dimension"] == "volatility" else "rates"} month. These are in-sample
        replays: the portfolio was built with knowledge of this history."""),
        )
    )
    s.append(
        (
            "Key risks and caveats",
            textwrap.dedent(f"""\
        - Synthetic liabilities, fixed and nominal, with no policyholder behavior beyond the surrender test.
        - Capital is a simplified proxy, not NAIC RBC. Factors are assumptions.
        - Expected returns: bonds use today's yields, equities use long-run historical estimates.
        - Scenarios replay 2007 to 2026 history; nothing worse than that period can appear.
        - The rate outlook matters little for this portfolio because its interest rate gap is capped: the
          5-year median funding ratio is {sens_fr["forward"]:.1%} with rates drifting toward market forward
          rates, {sens_fr["today"]:.1%} if today's curve holds, and {sens_fr["historical"]:.1%} if rates return
          toward their 2007 to 2026 average.
        - ETFs stand in for the bonds and loans an insurer would hold directly."""),
        )
    )
    s.append(
        (
            "Recommended actions",
            textwrap.dedent("""\
        1. Move toward the recommended allocation, away from asset-only portfolios that leave surplus exposed
           to falling rates.
        2. Adopt the duration-gap band (plus or minus $10m of surplus per 100bp) in the investment policy and
           re-check the gap monthly.
        3. Review the 40% credit floor with the investment committee: it is the main source of remaining
           surplus volatility in this model.
        4. Hold at least the current level of Treasuries as the liquidity buffer for surrender spikes.
        5. Before relying on these figures, replace the capital proxy with the insurer's actual capital model
           and the synthetic liabilities with real cash flows."""),
        )
    )
    return s


def write_client_report(metrics, weights, reg_df, shock_df, sens_df, ctx, a0, s0, outflow, cap, gaps) -> None:
    sections = _build_sections(metrics, weights, reg_df, shock_df, sens_df, ctx, a0, s0, outflow, cap, gaps)
    cols = [REC, UNB, ALT, MATCH]
    table = metrics.loc[
        cols,
        [
            "exp_surplus_change_1y",
            "surplus_vol_1y",
            "surplus_CVaR95_1y",
            "capital_proxy",
            "coverage_surplus_over_capital",
            "liquidity_coverage",
            "funding_ratio_5y_median",
            "funding_ratio_5y_p05",
        ],
    ].copy()
    table.columns = [
        "Exp. surplus change $m",
        "Surplus vol $m",
        "1-in-20 avg loss $m",
        "Capital proxy $m",
        "Coverage x",
        "Liquidity coverage x",
        "Funding ratio 5y median",
        "Funding ratio 5y p05",
    ]
    table.index = ["Recommended (banded)", "Without band", "Asset-only alternative", "Key-rate matched"]
    md_tab = table.copy()
    for c in ["Funding ratio 5y median", "Funding ratio 5y p05"]:
        md_tab[c] = (md_tab[c] * 100).round(1).astype(str) + "%"
    charts = [
        ("stage3_allocation.png", "Asset allocation"),
        ("stage3_funding_fan.png", "Funding ratio outlook"),
        ("stage3_surplus_distribution.png", "One-year surplus outcomes"),
        ("stage3_frontier.png", "Surplus frontier and capital"),
    ]
    lines = []
    for i, (title, body) in enumerate(sections):
        lines += [f"{'#' if i == 0 else '##'} {title}", "", body, ""]
        if title == "Summary":
            lines += [md_tab.round(2).to_markdown(), ""]
            lines += [f"![{t}]({p})" for p, t in charts[:3]] + [""]
    lines += [f"![{charts[3][1]}]({charts[3][0]})", ""]
    (REPORTS_DIR / "client_report.md").write_text("\n".join(lines))
    _write_pdf(sections, table, charts)


def _reflow(body: str, width: int) -> list[str]:
    """Join hard-wrapped source lines into paragraphs and list items, then wrap.

    A new item starts at a line beginning with "- " or "N. "; a blank line
    separates paragraphs.
    """
    out: list[str] = []
    items: list[str] = []
    for line in body.split("\n"):
        s = line.strip()
        if not s:
            items.append("")
        elif re.match(r"^(- |\d+\. )", s) or not items or items[-1] == "":
            items.append(s)
        else:
            items[-1] += " " + s
    for item in items:
        indent = "   " if re.match(r"^(- |\d+\. )", item) else ""
        out += textwrap.wrap(item, width, subsequent_indent=indent) or [""]
    return out


def _write_pdf(sections, table, charts) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # no creation timestamp, so regenerating with the same numbers gives the same file
    with PdfPages(REPORTS_DIR / "client_report.pdf", metadata={"CreationDate": None}) as pdf:
        page_lines: list[tuple[str, str]] = []

        def flush():
            if not page_lines:
                return
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            y = 0.95
            for kind, txt in page_lines:
                size, weight = (14, "bold") if kind == "h" else (9.5, "normal")
                fig.text(0.07, y, txt, fontsize=size, weight=weight, va="top", family="DejaVu Sans", parse_math=False)
                y -= 0.03 if kind == "h" else 0.0175
            pdf.savefig(fig)
            plt.close(fig)
            page_lines.clear()

        used = 0
        for title, body in sections:
            wrapped = _reflow(body, 105)
            need = 2 + len(wrapped)
            if used + need > 48:
                flush()
                used = 0
            page_lines.append(("h", title))
            page_lines += [("p", w) for w in wrapped]
            page_lines.append(("p", ""))
            used += need
        flush()
        # metrics table page
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        cells = [
            [name] + [f"{v:.1%}" if "Funding" in c else f"{v:,.2f}" for c, v in row.items()]
            for name, row in table.iterrows()
        ]
        n_cols = len(table.columns)
        tb = ax.table(
            cellText=cells,
            colLabels=["Portfolio"] + [textwrap.fill(c, 11) for c in table.columns],
            colWidths=[0.2] + [0.8 / n_cols] * n_cols,
            cellLoc="center",
            loc="center",
        )
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 2.6)
        for j in range(n_cols + 1):  # taller header row for the wrapped labels
            tb[(0, j)].set_height(tb[(0, j)].get_height() * 1.7)
        ax.set_title("Key metrics ($ millions unless stated). Capital is a PROXY, not NAIC RBC.", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        for path, title in charts:
            img = mpimg.imread(REPORTS_DIR / path)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=13)
            pdf.savefig(fig)
            plt.close(fig)

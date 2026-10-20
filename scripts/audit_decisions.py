"""Audit the numbers quoted in the documentation against the generated reports.

Each check recomputes a number from reports/ and verifies that the exact text
quoting it is present in the document where it is used:
- public:  DECISIONS.md and README.md (committed);
- private: PRIVATE_PREP.md (local preparation notes, gitignored). If that file
  is absent, as in a public clone, its text checks are skipped, not failed;
  the numbers themselves are still checked.

Run from the repo root after regenerating reports:

    .venv/bin/python scripts/audit_decisions.py

Exits with status 1 if any number or quoted text no longer matches.
"""

import subprocess
from pathlib import Path

import pandas as pd

R = "reports/"


def row(md: str, section: str, key: str, nth_table: int = 0) -> list[str]:
    """Cells of the first table row whose first cell == key, inside `section` (heading text)."""
    part = md[md.index(section) :]
    tables, cur = [], []
    for line in part.splitlines()[1:]:
        if line.startswith("#") and tables:
            break
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    for line in tables[nth_table]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == key:
            return cells[1:]
    raise KeyError((section, key))


def f(x) -> float:
    return float(x)


m = pd.read_csv(R + "stage3_metrics.csv", index_col=0)
w = pd.read_csv(R + "stage3_weights.csv", index_col=0)
s1 = open(R + "stage1_results.md").read()
s1b = open(R + "stage1b_results.md").read()
s3 = open(R + "stage3_results.md").read()
# Original 15-year-window Stage 1 report, archived as a file (D-058).
old1 = open(R + "archive/stage1_results_2011_window.md").read()
emp = pd.read_csv(R + "stage2_empirical_durations.csv")
REC, UNB = "RECOMMENDED: surplus + capital aware, duration band", "Comparison: no duration band"
AOMV, AOC, FREE = "Asset-only min variance", "Asset-only constrained max Sharpe", "Surplus-aware, no credit floor"

vol = {e: [f(x) for x in row(s3, "Annual volatility, historical vs both generators", e)] for e in ("SHY", "TLT")}
under = [
    f(row(s1b, "### How much does the normal model", p)[3])
    for p in ("Min variance", "Max Sharpe", "Constrained max Sharpe", "Equal weight (benchmark)")
]
lw = [f(x) for x in row(s1, "## Volatility by covariance estimator", "SHY")]
spy_old = f(row(old1, "## Weights", "SPY")[1])
shy_new = f(row(s1, "## Weights", "SHY")[1])
pca3 = f(row(s3, "Explained variance of monthly", "curvature")[1])
gap_unb = f(row(s3, "Duration gap (asset duration", UNB)[2])
shock_cols = [c.strip() for c in s3[s3.index("| shock             |") :].splitlines()[0].strip("|").split("|")][1:]
p100 = dict(zip(shock_cols, [f(x) for x in row(s3, "## 5. Recommended portfolio", "Parallel +100bp", 1)], strict=True))
sens = [f(row(s3, "## Anchor sensitivity", k)[-1]) for k in ("forward", "today", "historical")]
dur = f(row(s3, "## 1. Synthetic liabilities", "Effective duration (curve)")[0])
tlt_r2 = emp[(emp.etf == "TLT") & (emp.model == "rates only")].r2.iloc[0]
months = len(pd.read_csv(R + "stage1b_regime_labels.csv"))
ntests = subprocess.run(
    [".venv/bin/python", "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True
).stdout
ntests = int([ln for ln in ntests.splitlines() if "tests collected" in ln][0].split()[0])

WFO, WFS = "Walk-forward optimized", "Static in-sample headline"
wfr = pd.read_csv(R + "walkforward_reset_summary.csv", index_col=0)
wfp = pd.read_csv(R + "walkforward_primary_summary.csv", index_col=0).T.drop(["min_funding_ratio_date"]).astype(float)
cal = pd.read_csv(R + "walkforward_calibration.csv", index_col=0)
calf = cal.loc[[i for i in cal.index if len(str(i)) == 4]]
exceed = int((calf["realized"] < -calf["pred_var95"]).sum())
wfd = pd.read_csv(R + "walkforward_reset_decisions.csv")

dvs = pd.read_csv(R + "discount_variant_summary.csv").set_index("fraction_of_IG")
dv0, dv5 = dvs.loc[0.0], dvs.loc[0.5]

DOCS = {
    "public": Path("DECISIONS.md").read_text() + "\n" + Path("README.md").read_text(),
    "private": Path("PRIVATE_PREP.md").read_text() if Path("PRIVATE_PREP.md").exists() else None,
}

# (name, value recomputed from reports, value quoted, [(document, exact quoted text), ...])
checks = [
    ("tests", ntests, 155, [("public", "**155 passed"), ("private", "155 automated tests")]),
    ("about 19 years of data", round(months / 12), 19, [("public", "about 19 years"), ("private", "about 19 years")]),
    (
        "asset-only min variance 1-in-20 loss",
        round(m.loc[AOMV, "surplus_CVaR95_1y"], 1),
        155.5,
        [("public", "$155.5m"), ("private", "$156 million")],
    ),
    (
        "recommended 1-in-20 loss",
        round(m.loc[REC, "surplus_CVaR95_1y"], 1),
        71.5,
        [("public", "$71.5m"), ("private", "$72 million")],
    ),
    (
        "recommended loss below half the asset-only loss",
        bool(m.loc[REC, "surplus_CVaR95_1y"] < 0.5 * m.loc[AOMV, "surplus_CVaR95_1y"]),
        True,
        [("public", "more than half"), ("private", "more than half")],
    ),
    (
        "capital about a fifth",
        round(m.loc[REC, "capital_proxy"] / m.loc[AOC, "capital_proxy"], 2),
        0.19,
        [("public", "about a fifth"), ("private", "about a fifth")],
    ),
    (
        "recommended capital",
        round(m.loc[REC, "capital_proxy"], 1),
        33.5,
        [("public", "$33.5m"), ("private", "$34 million")],
    ),
    ("asset-only constrained capital", round(m.loc[AOC, "capital_proxy"], 1), 175.3, [("public", "$175.3m")]),
    ("recommended surplus volatility", round(m.loc[REC, "surplus_vol_1y"], 1), 34.7, [("public", "$34.7m")]),
    ("asset-only constrained surplus volatility", round(m.loc[AOC, "surplus_vol_1y"], 1), 72.9, [("public", "$72.9m")]),
    (
        "recommended funding ratio 5y median",
        round(m.loc[REC, "funding_ratio_5y_median"] * 100, 1),
        120.3,
        [("public", "120.3%"), ("private", "120%")],
    ),
    ("asset-only min variance SHY weight", round(w.loc["SHY", AOMV] * 100, 1), 97.7, [("public", "97.7%")]),
    ("no-floor surplus volatility", round(m.loc[FREE, "surplus_vol_1y"], 1), 20.1, [("public", "$20.1m")]),
    ("unbanded surplus volatility", round(m.loc[UNB, "surplus_vol_1y"], 1), 33.1, [("public", "$33.1m")]),
    (
        "band cost in surplus volatility",
        round(m.loc[REC, "surplus_vol_1y"] - m.loc[UNB, "surplus_vol_1y"], 1),
        1.6,
        [("public", "(+$1.6m)"), ("private", "$1.6m")],
    ),
    ("unbanded duration gap", round(gap_unb, 1), 20.3, [("public", "+$20.3m per 100bp")]),
    ("+100bp loss, unbanded", round(-p100[UNB], 1), 16.8, [("public", "$16.8m")]),
    ("+100bp loss, banded", round(-p100["Recommended"], 1), 7.7, [("public", "$7.7m")]),
    ("PCA three factors", round(pca3 * 100, 1), 98.2, [("public", "98.2%")]),
    (
        "TLT vol history / Vasicek / PCA",
        tuple(round(x, 1) for x in vol["TLT"]),
        (14.1, 4.5, 14.0),
        [("public", "14.0% vs 14.1%"), ("public", "Vasicek 4.5%")],
    ),
    ("SHY vol history / PCA", (vol["SHY"][0], vol["SHY"][2]), (1.52, 1.47), [("private", "SHY 1.47% vs 1.52%")]),
    ("long bonds about 3 times too calm", round(vol["TLT"][0] / vol["TLT"][1]), 3, [("private", "3 times")]),
    ("Ledoit-Wolf doubles SHY volatility", (lw[0], lw[1]), (1.52, 3.32), [("public", "from 1.52% to 3.32%")]),
    (
        "normal understates CVaR99",
        (round(min(under), 1), round(max(under), 1)),
        (16.5, 38.2),
        [("public", "16.5% to 38.2%"), ("private", "16% to 38%")],
    ),
    (
        "max Sharpe flip 83% SPY to 74% SHY",
        (round(spy_old), round(shy_new)),
        (83, 74),
        [("public", "83% SPY to 74% SHY")],
    ),
    ("TLT empirical duration R-squared", round(tlt_r2, 2), 0.98, [("private", "R-squared 0.98")]),
    (
        "anchor sensitivity funding ratio range",
        (round(min(sens) * 100, 1), round(max(sens) * 100, 1)),
        (119.6, 120.3),
        [("public", "119.6% (historical)"), ("private", "119.6% and 120.3%")],
    ),
    ("liability effective duration", round(dur, 2), 9.48, [("public", "duration 9.48")]),
    # walk-forward
    (
        "walk-forward optimized mean",
        round(wfr.loc[WFO, "mean surplus change"], 1),
        7.6,
        [("public", "+$7.6m"), ("private", "$7.6 million")],
    ),
    (
        "walk-forward static mean",
        round(wfr.loc[WFS, "mean surplus change"], 1),
        11.6,
        [("public", "+$11.6m"), ("private", "$11.6 million")],
    ),
    (
        "walk-forward optimized volatility",
        round(wfr.loc[WFO, "vol of surplus change"], 1),
        23.2,
        [("public", "$23.2m")],
    ),
    (
        "walk-forward optimized 2022",
        round(wfr.loc[WFO, "2022 change"], 1),
        -44.8,
        [("public", "-$44.8m"), ("private", "$44.8 million")],
    ),
    ("walk-forward static 2022", round(wfr.loc[WFS, "2022 change"], 1), -14.4, [("public", "-$14.4m")]),
    ("walk-forward 1-in-20 breaches", (exceed, len(calf)), (0, 15), [("public", "0 of 15")]),
    (
        "walk-forward primary 2022",
        (round(wfp.loc["surplus_change_2022", WFO], 1), round(wfp.loc["surplus_change_2022", WFS], 1)),
        (-59.8, -75.5),
        [("public", "-$59.8m"), ("public", "-$75.5m")],
    ),
    ("walk-forward band edge", bool((wfd["gap_per_100bp"] > 9.8).all()), True, [("public", "within $0.2m")]),
    # discount-rate variant
    (
        "variant liability PV at 0% / 50%",
        (round(dv0.liability_pv, 1), round(dv5.liability_pv, 1)),
        (1000.0, 961.2),
        [("public", "961.2"), ("private", "$961m")],
    ),
    (
        "variant LQD correlation",
        (round(dv0.corr_LQD_vs_liability, 2), round(dv5.corr_LQD_vs_liability, 2)),
        (0.52, 0.63),
        [("public", "0.63 (50%)"), ("private", "0.52 to 0.63")],
    ),
    (
        "variant credit chosen freely",
        (round(dv0.free_LQD * 100, 1), round(dv5.free_LQD * 100, 1)),
        (0.0, 10.5),
        [("public", "10.5%"), ("private", "about 10% credit")],
    ),
    (
        "variant headline volatility",
        (round(dv0.headline_surplus_vol, 1), round(dv5.headline_surplus_vol, 1)),
        (34.7, 20.3),
        [("public", "$20.3m"), ("private", "$20.3m")],
    ),
    (
        "variant IG +100bp",
        (round(dv0.headline_IG_shock_surplus_change, 1), round(dv5.headline_IG_shock_surplus_change, 1)),
        (-34.9, 11.2),
        [("public", "+$11.2m"), ("private", "(+$11.2m)")],
    ),
]

n_fail = n_skip = n_text = 0
for name, got, want, quotes in checks:
    problems = [] if got == want else [f"report {got} != quoted {want}"]
    for doc, text in quotes:
        if DOCS[doc] is None:
            n_skip += 1
            continue
        n_text += 1
        if text not in DOCS[doc]:
            problems.append(f"text {text!r} not found in {doc} docs")
    n_fail += bool(problems)
    print(
        "PASS" if not problems else "FAIL",
        "|",
        name,
        "| report:",
        got,
        "| quoted:",
        want,
        "" if not problems else "| " + "; ".join(problems),
    )
print(
    f"{len(checks) - n_fail}/{len(checks)} checks pass ({n_text} quoted texts checked, {n_skip} private texts skipped)"
)
if n_fail:
    raise SystemExit(1)

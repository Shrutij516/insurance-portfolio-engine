# Insurance Portfolio Solutions Engine

A tested Python project that reproduces, at small scale, the analytics an insurance company's investment team uses: estimating risk and return, simulating markets, and choosing a portfolio that protects **surplus** (assets minus what the insurer owes policyholders) rather than just chasing return.

It is a **learning and portfolio project** built on public data. The insurer is **synthetic**, and the capital figure is a **simplified proxy, not NAIC risk-based capital**. Nothing here is investment advice.

- Engineering log (decisions, problems, validation results, honest limits, glossary): [DECISIONS.md](DECISIONS.md)
- Client-style report: [reports/client_report.pdf](reports/client_report.pdf) ([markdown version](reports/client_report.md))
- Technical results by stage: [reports/](reports/)

## Headline results

A synthetic $1 billion block of payout-annuity liabilities (duration 9.5), backed by $1.1 billion of assets (surplus $100 million, funding ratio 110%). One-year figures across 10,000 simulated scenarios; "1-in-20 loss" is the average surplus loss in the worst 5% of scenarios.

| Portfolio | Surplus volatility | 1-in-20 loss | Capital proxy | Funding ratio after 5 years (median) |
|---|---|---|---|---|
| Asset-only, lowest volatility (98% short Treasuries) | $63.8m | $155.5m | $90.9m | 114.1% |
| Asset-only, return-seeking | $72.9m | $173.6m | $175.3m | 126.1% |
| **Recommended: surplus-aware, capital-aware, duration gap capped** | **$34.7m** | **$71.5m** | **$33.5m** | **120.3%** |
| Same, without the duration cap (comparison) | $33.1m | $65.8m | $31.8m | 119.9% |

What this shows:
- The portfolio that looks safest on its own loses **more than the whole $100m surplus** in a bad year, because its value does not move with the liabilities when rates fall.
- The recommended portfolio (40% investment grade credit, 34% long Treasuries, 20% short Treasuries, 5% gold, 1.5% equities) cuts that loss by more than half and uses about a fifth of the capital of the return-seeking alternative.
- It caps the interest rate gap at plus or minus $10m of surplus per 1 point move in rates. That costs $1.6m of surplus volatility in historical scenarios, but a 1 point rate rise costs $7.7m instead of $16.8m, which matters in a 2022-style market where rates and credit spreads rise together.

## What it does

| Stage | What was built |
|---|---|
| 1 | Market data for 11 asset-class ETFs (2007 to 2026, including 2008 and 2022); covariance with Ledoit-Wolf shrinkage; expected returns with Bayes-Stein shrinkage; a mean-variance optimizer written on a surplus basis from day one; risk contributions |
| 1b | Monte Carlo (block bootstrap, normal, Student-t) with VaR, CVaR and drawdowns; crisis replays with daily data; hypothetical shocks; interest-rate and volatility regimes without look-ahead |
| 2 | Treasury yield curve bootstrap; bond pricing, duration, convexity, DV01, key rate durations; Vasicek short-rate model with exact simulation; credit spread and equity factors; bond expected returns from today's yields; duration-based stress tests |
| 3 | Scenario generator from principal components of historical curve moves; synthetic annuity liabilities; surplus and funding ratio across scenarios; key-rate matching; liquidity coverage under a surrender wave; capital proxy; capital-aware surplus optimizer with a CVaR cross-check; client report |

Validation: 155 automated tests (closed-form checks, finite-difference checks, reproducibility, constraint checks, look-ahead checks), run with every warning treated as an error. Each check and its measured value is listed in the DECISIONS.md validation log. `scripts/audit_decisions.py` re-checks the headline numbers in DECISIONS.md against the generated reports.

## How to run

Requires Python 3.11 or newer.

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```

```bash
.venv/bin/python -m engine.run_stage3
```

```bash
.venv/bin/pytest
```

The Stage 1, 1b and 2 pipelines run the same way (`engine.run_stage1`, `engine.run_stage1b`, `engine.run_stage2`). The first run downloads prices from Yahoo Finance (via `yfinance`) and rates and spreads from FRED's public CSV endpoint, and caches them in `data/`. Raw data files are not committed (Yahoo's terms restrict redistribution); `data/*_metadata.json` records each file's hash so a rerun can confirm it used the same data. Yahoo re-adjusts history over time, so a fresh download can differ slightly from the one behind the committed reports. If a download fails, the code stops with an error; it never substitutes made-up data.

## Honest limits

- **Synthetic liabilities**: fixed, nominal payout annuities with no mortality model and no policyholder behavior beyond a surrender test.
- **Capital is a proxy**: factor charges by asset class plus an interest-rate charge, simple sum. Not NAIC RBC.
- **ETFs stand in** for the bonds, private placements and mortgages an insurer would actually hold.
- **Out-of-sample check**: a walk-forward test (rebuild every December from 2010 with only point-in-time data) shows the process earned less out of sample than in sample (+$7.6m vs +$11.6m a year on a fresh $100m surplus) and lost $44.8m in 2022 by leaning long duration; its predicted 1-in-20 loss was never breached. See [reports/walkforward_results.md](reports/walkforward_results.md).
- **Liabilities are discounted on the Treasury curve** in the headline, which makes credit look like pure surplus risk. A variant with half the investment grade spread as an illiquidity premium (an assumption) shows credit then partly hedges the liabilities: the optimizer holds about 10% credit on its own and the headline's surplus volatility falls from $34.7m to $20.3m. See [reports/discount_variant_results.md](reports/discount_variant_results.md).
- **Scenarios replay 2007 to 2026**, so nothing worse than that period can appear. Equity expected returns are shrunk historical averages, the weakest input.
- **One-year optimization** with buy-and-hold returns; no transaction costs, taxes or insurance accounting.

## Layout

```
src/engine/
  data.py           download, cache, clean prices; FRED rates and spreads; signals
  estimation.py     covariance (sample, Ledoit-Wolf), Bayes-Stein expected returns, PSD checks
  optimizer.py      surplus-based mean-variance optimizer with bounds, groups and linear rules
  risk.py           risk contributions
  montecarlo.py     Monte Carlo, VaR, CVaR, drawdowns
  stress.py         crisis replays, factor shocks, duration-based shocks
  regimes.py        rate and volatility regimes
  fixedincome.py    curve bootstrap, bond math, key rate durations
  rates.py          Vasicek model
  scenarios.py      Stage 2 factor scenarios, empirical durations, yield-based returns
  esg.py            PCA scenario generator (Stage 3 default)
  liabilities.py    synthetic annuity block
  alm.py            surplus paths, matching, liquidity, capital proxy, CVaR optimizer
  client_report.py  client report (markdown and PDF)
  walkforward.py    point-in-time decisions and realized evaluation (walk-forward test)
  run_stage*.py     one pipeline per stage, writing to reports/ (plus run_walkforward.py, run_discount_variant.py)
tests/              pytest suite (155 tests)
scripts/            audit_decisions.py
data/               cached data (gitignored) and metadata
reports/            generated results, charts, client report (archive/: original 15-year-window Stage 1 report)
```

## Universe

SPY (US equity), EFA (developed ex-US equity), EEM (emerging equity), SHY / IEF / TLT (short, intermediate, long Treasuries), LQD (investment grade credit), HYG (high yield), TIP (TIPS), VNQ (REITs), GLD (gold); BIL (T-bills) as the cash rate. Reasons are in DECISIONS.md (D-004).

## How it was built

A portfolio project on how insurance investment teams work, built in stages to learn the domain hands-on. Code was written with the help of an AI coding assistant (commits are marked); every modeling decision, failure and check is documented in DECISIONS.md.

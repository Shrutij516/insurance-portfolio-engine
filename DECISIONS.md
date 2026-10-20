# DECISIONS.md: Insurance Portfolio Solutions Engine

Last updated: 2026-10-20 (Stage 3c commit 2: liability discount-rate variant, D-063 to D-066. Data window mid-2007 to 2026, see D-020)

This file is the engineering log of a portfolio project on how insurance investment teams work: every design decision, problem, and validation result. Numbers quoted here come from real runs. Nothing is logged as passed unless it actually ran.

---

## 1. Project summary and goal

This project builds, in Python, a small version of the analytics an insurance company's investment team uses to decide how to invest its portfolio. An insurer's money is not "free": it backs promises to policyholders (liabilities), so what matters is surplus, the assets left over after covering those promises. The engine estimates risk and return for 11 asset-class ETFs over about 19 years (mid-2007 to 2026, including the 2008 crisis), finds optimal portfolios with mean-variance optimization, and breaks portfolio risk down by asset. The optimizer is written on a surplus basis from day one, so when liabilities are added in Stage 3 the same code optimizes against them, and with zero liabilities it gives the ordinary asset-only answer. The goal is a working, tested, honestly documented tool I can explain end to end in plain language.

**Stage status:** Stages 1, 1b, 2 and 3, a walk-forward out-of-sample test and a liability discount-rate variant are built and tested: data and estimation; surplus-based optimizer; Monte Carlo, stress tests and regimes; yield curve, bond math and rate models; synthetic liabilities, ALM, liquidity, capital-aware optimization and a client report. 155 tests passing, zero warnings. Stage 4 (extras) is not started.

---

## 2. Glossary (plain language, with why it matters to an insurer)

| Term | Plain meaning | Why it matters to an insurer |
|---|---|---|
| Asset allocation | How a portfolio is split across asset classes (stocks, bonds, gold, ...). | Most of a portfolio's risk and return comes from this split, not from picking individual securities. |
| Strategic asset allocation (SAA) | The long-term target mix, reviewed perhaps yearly. | The SAA is the insurer's main investment decision and is what this engine is built to inform. |
| Liabilities | Money the insurer has promised to pay in future (claims, annuity payments). | Assets exist to pay these; investment risk is judged against them. |
| Surplus | Assets minus liabilities. | Surplus is the cushion that keeps the insurer solvent. Regulators and rating agencies watch it. |
| Funding ratio | Assets divided by liabilities. 110% means $1.10 of assets per $1 of promises. | A falling funding ratio means the cushion is shrinking. |
| f (liability-to-asset ratio) | Liabilities per dollar of assets, L/A = 1 / funding ratio. | The weight on the liability in the surplus formula. |
| Asset-liability management (ALM) | Managing assets and liabilities together so they move alike. | The core discipline of insurance investing: you hedge your promises first, then seek return. |
| Surplus return | Change in surplus per dollar of starting assets: w'r_A minus f times r_L. | The quantity our optimizer manages. |
| Surplus volatility | How much the surplus swings. | An insurer cares more about this than about asset volatility. |
| Liability hedging | Holding assets that move like the liabilities, so surplus stays stable. | A portfolio that looks safe on its own can be risky against liabilities (see V-021). |
| Expected return | The average return we think an asset will earn per year. | Drives how much the optimizer wants each asset. Hardest input to estimate. |
| Volatility | Standard deviation of returns: how much returns typically swing. | The standard measure of risk in this framework. |
| Variance | Volatility squared. | What the optimizer actually minimizes, because it has nicer math. |
| Covariance | How two assets' returns move together, in variance units. | Diversification only works when assets do not all move together. |
| Correlation | Covariance rescaled to between -1 and +1. | Easier to read than covariance. Bonds and stocks have often had low or negative correlation. |
| Covariance matrix | The table of every pair's covariance, variances on the diagonal. | The main risk input to the optimizer. |
| Positive semi-definite (PSD) | A covariance matrix property meaning no portfolio can have negative variance. | A non-PSD matrix lets an optimizer "find" fake negative-risk portfolios. We check it every time. |
| Condition number | Ratio of largest to smallest eigenvalue of the matrix. Large means nearly redundant assets. | High values make optimizers unstable and weights jumpy. |
| Estimation error | The gap between what we estimate from limited history and the true value. | Optimizers amplify estimation error, choosing assets whose estimates are luckily high. |
| Shrinkage | Pulling a noisy estimate toward a simpler, more stable target. | Reduces estimation error. Standard practice in industry. |
| Ledoit-Wolf | A method that picks how much to shrink a covariance matrix using the data itself. | Gives a more stable risk model without hand tuning. |
| Bayes-Stein (Jorion) | A shrinkage method for expected returns: pulls each asset's average toward one common value. | Stops the optimizer from betting heavily on whichever asset had the luckiest past. |
| Mean-variance optimization (MVO) | Choose weights that give the lowest risk for a given expected return (Markowitz, 1952). | The standard starting point for SAA at insurers and pensions. |
| Efficient frontier | The curve of best possible portfolios: for each return level, the lowest risk. | Lets a client see the risk/return trade-off and pick a point. |
| Minimum variance portfolio | The single lowest-risk portfolio possible under the constraints. | The left end of the frontier. Ignores expected returns entirely, so it is robust to their errors. |
| Sharpe ratio | (Return minus risk-free rate) divided by volatility. Return per unit of risk. | Common summary of portfolio efficiency. |
| Max Sharpe portfolio | The portfolio with the highest Sharpe ratio. | The "best risk-adjusted" point, but very sensitive to expected return inputs. |
| Risk-free rate | Return on a near riskless investment, here short-term US T-bills. | The hurdle any risky asset must beat. |
| T-bills | US government debt maturing in under a year. | Proxy for cash. |
| Long-only | No negative weights (no short selling). | Insurers are mostly long-only by regulation and policy. |
| Budget constraint | Weights add up to 100%. | All money is invested somewhere (cash would be its own asset). |
| Group constraint | A limit on the total weight of a set of assets, e.g. high yield at most 10%. | Insurers have investment policy limits and regulatory limits by asset type. |
| Marginal contribution to risk (MCR) | How much portfolio volatility rises if you add a little more of one asset. | Shows which positions are expensive in risk terms at the margin. |
| Risk contribution | Weight times MCR. Contributions add up exactly to total volatility. | Shows where risk really comes from; a 31% equity weight is 60% of the risk (section 6). |
| Percent risk contribution | Risk contribution divided by total volatility. Adds to 100%. | Easy way to explain risk concentration to a client. |
| Adjusted close | Price adjusted for dividends and splits. | Its percentage change is total return, including income. Bond ETFs earn most of their return as income. |
| ETF | Exchange-traded fund: a basket you can trade like a stock. | Here used as a clean, priced proxy for each asset class. |
| Expense ratio | Annual fund fee, e.g. 0.09%. | Slightly lowers ETF returns versus the raw asset class. |
| US Treasuries | Bonds issued by the US government. Short (1-3y), intermediate (7-10y), long (20y+). | The core of liability hedging; long Treasuries move most like long-dated liabilities. |
| TIPS | Treasuries whose principal rises with inflation. | Hedge for inflation-linked liabilities. |
| Investment grade (IG) credit | Bonds of financially strong companies (BBB or better). | The largest asset class in most life insurer portfolios. |
| High yield (HY) credit | Bonds of weaker companies; higher yield, more default risk. | Higher return but heavier capital charges, so usually capped. |
| REITs | Companies that own real estate, traded like stocks. | Proxy for real estate exposure. |
| Emerging vs developed markets | Less mature economies (China, India, Brazil...) vs mature ones (Europe, Japan...). | Different risk and return profiles; diversification. |
| Annualization | Converting monthly figures to yearly: mean x 12, variance x 12. | Everyone quotes annual numbers. Assumes months are independent. |
| Simple vs log returns | Simple: (P1 - P0)/P0. Log: ln(P1/P0). | Portfolio simple return is exactly the weighted sum of asset simple returns, so the optimizer uses simple returns. |
| NAIC RBC | The US regulator's risk-based capital formula: how much capital an insurer must hold for its risks. | Capital is the binding constraint for insurers. We will NOT model real RBC; Stage 3 uses a labeled proxy. |
| Capital charge | Capital required per dollar of an asset, higher for riskier assets. | Makes risky assets "expensive" for insurers beyond their volatility. |
| Constant-maturity Treasury (CMT) yield | The Treasury's daily estimate of the yield on a government bond of exactly 3 months, 2 years, 10 years, and so on. | The standard, free source for the US yield curve. |
| Yield curve | Yields plotted against maturity. Usually upward sloping; "inverted" when short rates exceed long rates. | Liabilities are discounted off a curve, so its level and shape drive liability values. |
| Par yield | The coupon at which a bond is worth exactly 100. CMT yields are par yields. | What the market quotes; needs converting before you can discount individual cash flows. |
| Zero-coupon bond / zero rate | A bond paying one amount at one date; its yield is the zero rate for that maturity. | The right rate for discounting a single future cash flow, such as one claim payment. |
| Discount factor | What $1 paid at time t is worth today, e.g. 0.59 for 10 years at about 5.3%. | Present value of liabilities = sum of cash flows times discount factors. |
| Bootstrapping (a curve) | Building zero rates one maturity at a time from par yields, shortest first. | Standard way to turn market quotes into a discounting curve. Unrelated to the statistical bootstrap in Stage 1b. |
| Interpolation | Filling in yields between quoted maturities. | Choice affects forward rates and the value of cash flows between quoted points. |
| Forward rate | The interest rate implied today for borrowing between two future dates. | Sawtooth or wild forwards are a sign of a bad curve method. |
| Yield to maturity (YTM) | The single rate that discounts all of a bond's cash flows to its price. | The usual "yield" quoted for a bond or bond fund. |
| Coupon | The fixed interest a bond pays, usually semiannually for US bonds. | Determines cash flow timing and therefore duration. |
| Macaulay duration | Present-value-weighted average time until the bond's cash flows arrive, in years. | Intuition for "how long" a bond is. |
| Modified duration | Percent price change for a 1 percentage point change in yield (about Macaulay / (1 + y/2)). | The main measure of interest rate risk. Matching asset and liability duration protects surplus. |
| Effective duration | Duration measured by shifting the whole yield curve and repricing. | Works for any instrument, including liabilities, not just plain bonds. |
| Convexity | How duration itself changes as yields move; the curvature of price vs yield. | Makes big rate moves less painful than duration alone suggests (for plain bonds). Matters for large shocks. |
| DV01 | Dollar price change for a 1 basis point (0.01%) move in yield. | How traders and ALM teams size hedges. |
| Key rate duration (KRD) | Sensitivity to one point of the curve (e.g. the 10-year) moving, others fixed. | Shows exposure to curve SHAPE changes; insurers match KRDs, not just total duration. |
| Steepener / flattener | Long rates rise relative to short rates (steepener), or the reverse (flattener). | Hurts portfolios whose duration sits at a different point of the curve from the liabilities. |
| Basis point (bp) | 0.01 percentage point. 100bp = 1%. | The unit for rate and spread moves. |
| Credit spread / OAS | Extra yield over Treasuries for credit risk; OAS (option-adjusted spread) adjusts for bond options like calls. | Insurers' biggest asset class is corporate credit; spreads drive both return and risk. |
| Moody's Baa-10y spread | Yield on Baa-rated (lowest investment grade) seasoned corporate bonds minus the 10-year Treasury. | Free credit spread history back to 1986; our credit factor (D-032). |
| ICE BofA OAS indices | Benchmark investment grade and high yield index spreads. | The industry-standard spreads; on FRED only since Oct 2023, so used only for today's level. |
| Spread duration | Percent price change for a 1 percentage point change in credit spread. | About equal to rate duration for fixed-rate bonds. |
| Expected default loss | Probability of default x loss given default, per year. | Spread is not all profit: part of it pays for defaults. |
| Carry | Return earned if nothing changes, roughly the yield. | The starting yield is the best simple forecast of a bond fund's return. |
| Short rate | The interest rate for the very shortest borrowing; here proxied by the 3-month yield. | The single driver in one-factor rate models. |
| Vasicek model | Short rate pulled toward a long-run level at a set speed, plus normal random shocks. | Simple, closed-form bond prices, easy to test; can go negative and does not fit today's curve. |
| Mean reversion / half-life | Tendency to drift back toward a long-run level; half-life = time to close half the gap. | Controls how long rate shocks persist. |
| Hull-White model | Vasicek with a time-varying level chosen to fit today's curve exactly. | Standard for valuing rate-sensitive liabilities consistently with the market. |
| CIR model | Short-rate model whose volatility shrinks near zero, so rates stay non-negative. | Avoids negative rates, but is harder to calibrate and test. |
| Real-world vs risk-neutral measure | Real-world: how rates are expected to actually move. Risk-neutral: the adjusted dynamics used for pricing. | Projections (P) and valuations (Q) need different settings; mixing them is a classic error. |
| Market price of risk / term premium | Extra return investors demand for holding long bonds instead of rolling short ones. | Explains why long yields usually exceed expected future short rates. |
| Affine bond price | A bond price formula of the form A(T) x exp(-B(T) x r). | Lets the model produce a whole yield curve from one simulated short rate. |
| Martingale check | Average discounted payoff across simulated paths should equal the model's own price. | Catches simulation and discounting bugs. |
| Exact discretization vs Euler | Exact: draw from the true distribution over each step. Euler: approximate with a straight-line step. | Exact has no time-step error, so tests can be tight. |
| Zero lower bound | Period when the Fed held short rates at about 0% (2008 to 2015, 2020 to 2022). | Rates stuck at zero break models that assume normal fluctuations; distorts calibration. |
| Cholesky factor | A matrix L with L L' = correlation matrix; multiplying independent shocks by L makes them correlated. | Standard way to generate correlated scenarios. |
| Empirical duration | Duration measured by regressing a fund's returns on yield changes. | Checks whether a fund behaves like its stated duration. |
| R-squared (R2) | Share of variation in returns explained by the regression (0 to 1). | High for Treasury funds (0.98), near zero for high yield on rates alone. |
| Rate beta | Our multiplier on a representative bond's rate sensitivity, from empirical duration. | Bridges simple bond math and how the actual fund behaved. |
| Monte Carlo simulation | Generating thousands of possible futures at random and running the portfolio through each. | Shows the range of outcomes, especially the bad ones, not just the average. |
| Path | One simulated future, month by month. | Lets us measure things that depend on the route, like drawdowns. |
| Terminal wealth | What $1 grows to at the end of the horizon. | The headline outcome of a simulation. |
| VaR (Value at Risk) | The loss exceeded in only a small share of cases. 99% VaR: only 1 in 100 outcomes is worse. | The standard regulatory risk number (Solvency II uses 99.5% over one year). |
| CVaR (expected shortfall) | The average loss in those worst cases. | Always at least VaR, and it sees how bad the tail is, which VaR ignores. Increasingly preferred by regulators. |
| Drawdown / max drawdown | Fall from a previous peak; max drawdown is the worst such fall on a path. | Boards feel drawdowns, not volatility. A 20% fall from peak triggers questions even if it recovers. |
| Fat tails | Extreme outcomes happen more often than a normal (bell curve) distribution says. | A normal model understates crash risk, so it understates capital needed. |
| Skew | Asymmetry: negative skew means big losses are more common than equally big gains. | Most risky assets have negative skew; volatility alone misses it. |
| Excess kurtosis | How fat the tails are compared with a normal distribution (normal = 0). | Positive values mean more extreme months than a bell curve predicts. |
| Multivariate normal | The bell curve extended to many assets moving together. | Simplest simulation model; thin tails. |
| Student-t distribution | A bell-like curve with fatter tails, controlled by "degrees of freedom" (lower = fatter). | A middle ground between normal and history. |
| Bootstrap | Simulating by resampling real past data instead of assuming a distribution. | Keeps real crash behavior without guessing its shape. |
| Block bootstrap | Resampling chunks of consecutive months, not single months. | Keeps runs of bad months together (volatility clustering), which single-month resampling breaks up. |
| Volatility clustering | Calm and turbulent periods each tend to last; a bad month is often followed by another. | Drives multi-month drawdowns that independent-month models understate. |
| Rebalancing | Trading back to target weights after prices move. | Keeps the risk profile stable; insurers rebalance to their SAA within ranges. |
| Stress test | Revaluing the portfolio under a specific bad scenario. | Regulators and boards ask "what happens in a 2008 or a 2022?" |
| Historical scenario | A stress test that replays a real past crisis window. | Credible because it really happened; limited to what has happened. |
| Hypothetical (factor) shock | A made-up instant move per asset, e.g. "equities down 30%". | Tests scenarios that have not happened yet. Ours are simple factor shocks, not duration-based. |
| Taper tantrum (2013) | Bond sell-off when the Fed hinted it would slow bond purchases. | A clean example of a pure rate shock. |
| 10-year Treasury yield | The interest rate on 10-year US government debt. | The benchmark long rate; liability values move with long rates. |
| Regime | A labeled market environment, e.g. rising rates or high volatility. | Portfolios behave differently in different environments; averages hide this. |
| Look-ahead bias | Using information that would not have been known at the time. | Makes backtests look better than reality. Our regime labels use only past data. |
| Realized volatility | Volatility measured from recent actual daily returns. | A simple, observable gauge of turbulence. |
| In-sample | Evaluated on the same data used to build the model. | Results look better than they would on new data. |
| Principal component analysis (PCA) | Finds the few independent shapes that explain most of the variation in many related series. | Turns 8 yield changes into 3 meaningful factors. |
| Level, slope, curvature | The three classic yield curve moves: all rates up or down, short vs long, middle vs ends. | Level is 78% of curve variation in our data; insurers hedge level first, then slope. |
| Explained variance | Share of total variation captured by a component. | Shows how many factors you need (3 here, 98.2%). |
| Economic scenario generator (ESG) | Software that produces joint future paths of rates, spreads and asset returns. | The engine behind every insurer's ALM, capital and pricing projections. |
| Anchor (long-run level) | The level rates are pulled toward over time. | Sets the rate outlook in simulations; drives bond and liability results. |
| Forward curve | Rates the market implies today for borrowing in the future. | A market-based, forward-looking anchor. |
| Payout annuity / runoff | Contracts paying policyholders regular income; a closed block shrinks as people die. | Classic long-duration insurer liability. |
| Dollar duration gap | Assets x asset duration minus liabilities x liability duration. | How much surplus moves for a rate change; zero means immunized. |
| Surplus DV01 | Change in surplus for a 1bp rate move. | The single number an ALM desk watches daily. |
| Key-rate matching / immunization | Choosing assets whose key rate durations match the liabilities'. | Protects surplus against curve shifts and twists. |
| Liquidity coverage ratio | Liquid assets after a stress and haircuts, divided by stressed cash outflows. | Insurers must pay claims and surrenders without fire sales. |
| High-quality liquid assets (Level 1, 2A, 2B) | Bank regulation buckets for how easily an asset turns into cash; Level 1 is government debt. | Borrowed here as labeled assumptions for haircuts. |
| Haircut | Percentage of value assumed lost when selling an asset in a hurry. | Makes liquidity tests conservative. |
| Surrender | A policyholder cashing out a policy early. | A spike in surrenders is the classic insurer liquidity risk. |
| C-1 and C-3 risk | NAIC RBC labels for asset default risk (C-1) and interest rate risk (C-3). | Our proxy mimics the idea, not the formulas. |
| Coverage ratio (proxy) | Surplus divided by the capital proxy. | Like an RBC ratio: above 2 here means twice the required cushion. |
| Rockafellar-Uryasev CVaR | A trick that turns "minimize the average of the worst 5% losses" into a linear program. | Lets you optimize tail risk directly over thousands of scenarios. |
| Surplus efficient frontier | Best achievable expected surplus growth for each level of surplus risk. | The ALM version of the efficient frontier. |
| Flight to quality | In a crisis, investors sell risky assets and buy Treasuries, so Treasuries rise as credit falls. | A hedge that works in 2008 but failed in 2022. |
| Investment policy constraints | Limits an insurer's board sets on asset classes, credit quality, duration and liquidity. | Real portfolios are constrained far more than textbook ones. |

---

## 3. Decision log

Format: ID, date, stage, Decision, Options considered, Why this one, Trade-offs and limitations.

### D-001: Python environment
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Python 3.12.14 (a uv-managed CPython already installed on the machine), in a project `.venv`.
- **Options considered:** (a) Homebrew Python 3.12 or 3.11; (b) uv-managed CPython; (c) reinstall Homebrew Python; (d) system Python 3.9.
- **Why this one:** Homebrew Python was broken on this machine (see P-001). The uv build was already installed, needed no download and no system changes. System 3.9 fails the 3.11+ requirement.
- **Trade-offs and limitations:** The venv depends on uv's interpreter path. Anyone cloning the repo just needs any working Python 3.11+.

### D-002: Repo layout and tooling
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** `src/engine` package (src layout), `pyproject.toml`, pinned `requirements.txt`, pytest, ruff (lint and format).
- **Options considered:** flat scripts; notebooks only; src layout package.
- **Why this one:** A package with tests is how production analytics code is written. Notebooks are kept for exploration only. The src layout stops tests from accidentally importing local files instead of the installed package.
- **Trade-offs:** Slightly more setup than a notebook.

### D-003: Data source
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Yahoo Finance via `yfinance`, adjusted daily closes (`auto_adjust=True`), cached to `data/prices_daily.csv`.
- **Options considered:** yfinance (free, easy); FRED (rates and some indices, no ETFs); paid vendors (Bloomberg, FactSet; not available); synthetic data.
- **Why this one:** Free, covers all the ETFs, and adjusted prices give total return including dividends and bond coupons, which is essential: most of a bond fund's return is income.
- **Trade-offs:** Unofficial API that can break or rate-limit; Yahoo's adjustment method is not fully documented; Yahoo re-adjusts history when new dividends arrive, so a re-download can change old prices slightly. Mitigation: cache once, record the SHA-256 hash and download time in `data/prices_metadata.json`, and a test checks the cache matches the hash.

### D-004: Asset universe (tickers)
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** 11 investable ETFs plus BIL as the cash proxy:

| Ticker | Asset class | Alternatives considered | Why this one |
|---|---|---|---|
| SPY | US equity (S&P 500) | VTI, IVV | Most liquid, longest history, the standard benchmark. |
| EFA | International developed equity | VEA | Long history (2001), standard MSCI EAFE benchmark. |
| EEM | Emerging equity | VWO | Long history (2003). Higher fee than VWO, accepted for history and benchmark recognition. |
| SHY | Treasuries 1-3y | VGSH, SCHO | Long history (2002), clean short duration. |
| IEF | Treasuries 7-10y | VGIT | Long history (2002), intermediate duration. |
| TLT | Treasuries 20y+ | EDV, VGLT | Long history (2002). Long duration is what hedges long insurance liabilities. |
| LQD | Investment grade credit | VCIT, VCLT | Long history (2002), most liquid IG fund. IG is the core insurer asset. |
| HYG | High yield credit | JNK | Most liquid HY fund. Starts 2007-04-11, the latest-starting investable ticker. |
| TIP | TIPS | SCHP | Long history (2003). |
| VNQ | US REITs | IYR, XLRE | Long history (2004), broad REIT exposure. |
| GLD | Gold | IAU; DBC or GSG for broad commodities | Gold is a clean diversifier. Broad commodity funds hold futures and lose money to "roll costs" in some markets, which muddies the return series. |
| BIL | 1-3 month T-bills (not investable here) | SGOV (too new, 2020), SHV | Risk-free rate proxy. Starts 2007-05-30, the latest of all 12, so it sets the sample start. |

- **Why this set:** Covers every requested sleeve with one liquid, long-history ETF each, all available from 2007-05-30, so no proxies or backfilling are needed even with the 2008 crisis in the sample.
- **Trade-offs:** ETFs are proxies. Real insurers hold individual bonds, private placements, mortgage loans and private credit, not ETFs. ETF returns are after fees (0.03% to 0.49% a year). Only one commodity sleeve (gold). No cash as an investable asset (SHY is close to it).

### D-005: Sample window (SUPERSEDED by D-020 on 2026-10-19)

> Kept for the record. The original 15-year window below was replaced by a mid-2007 start. Figures in this entry refer to the old window.

- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** 2011-10-01 to 2026-09-30 (15 years). 3,770 trading days, 179 monthly returns.
- **Options considered:** (a) 15 years as requested; (b) start mid-2007 (about 19 years), the earliest date where all 12 tickers exist, which would include the 2008 financial crisis; (c) longer history with proxies.
- **Why this one:** It is what was requested, and all tickers exist for the whole window. It includes the 2020 COVID crash and the 2022 rate shock (the worst bond year in decades).
- **Trade-offs:** It **misses the 2008 crisis**, the most important credit stress in modern history. It is also dominated by a falling-then-rising rate path, which distorts historical bond returns (see P-002). **Open question for Stage 1b:** extend to mid-2007 for stress testing.

### D-006: Missing values and different start dates
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Trim the panel to start at the latest first-valid date across tickers. Inside the window, forward fill gaps of at most 5 trading days. Raise an error on any longer gap. Never backfill.
- **Options considered:** drop rows with any NaN; forward fill without limit; backfill early history; proxy missing history with a similar index; trim to common window.
- **Why this one:** Every asset gets real data for the whole sample. Short gaps (holidays, a missing print) are harmless to fill. Long fills would carry a stale price, which fakes low volatility and low correlation, so they are refused loudly.
- **Trade-offs:** The latest-starting asset sets the window length; here a non-investable asset (BIL, the cash proxy) sets it, costing 2 months versus HYG. Accepted, because a consistent risk-free series matters more. **Actual result (D-020 window):** requested from 2007-04-01. HYG first trades 2007-04-11 (6 missing days before it), BIL 2007-05-30 (40 missing days). The panel was trimmed to start 2007-05-30, dropping 40 rows; zero gaps remain inside the window, so no forward fill was needed. (On the original 2011 window the trim did nothing, because all tickers already existed.)

### D-007: Monthly returns, not daily
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Month-end to month-end returns from the last daily price in each month. A partial final month is dropped.
- **Options considered:** daily returns annualized x 252; weekly; monthly.
- **Why this one:** (1) Insurer SAA is a long-horizon decision; monthly is closer to that horizon. (2) Daily data has noise from different market closing times (international ETFs trade in US hours, but their underlying markets close earlier) and short-term mean reversion that makes daily-to-annual scaling inaccurate. (3) 232 monthly observations for 11 assets is a reasonable ratio (21 per asset).
- **Trade-offs:** Fewer data points (232 vs 4,866 daily), so estimates are noisier, which shrinkage addresses.

### D-008: Simple returns, not log returns
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Simple returns.
- **Options considered:** simple; log.
- **Why this one:** A portfolio's simple return is exactly the weighted sum of its assets' simple returns. That is what the optimizer's formula w'mu assumes. Log returns add over time but not across assets.
- **Trade-offs:** Simple returns are not symmetric (a -50% then +50% is a loss). Log returns will be used where compounding over time matters (Monte Carlo, Stage 1b).

### D-009: Covariance estimator
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Default is **Ledoit-Wolf shrinkage applied to the correlation matrix** (`lw_corr`): standardize returns, shrink correlations toward zero with Ledoit-Wolf, then rescale by each asset's sample volatility. Sample covariance and raw sklearn Ledoit-Wolf are kept for comparison.
- **Options considered:** (a) sample covariance; (b) sklearn Ledoit-Wolf on raw returns (target: scaled identity, one common variance for all assets); (c) Ledoit-Wolf in correlation space; (d) Ledoit-Wolf constant-correlation target (2004 paper); (e) factor models.
- **Why this one:** Raw Ledoit-Wolf shrinks every asset's variance toward one common number. Our assets' volatilities differ by about 15x (SHY 1.52%, VNQ 22.25%), so it **more than doubled SHY's volatility, from 1.52% to 3.32%** (P-003). That would make short Treasuries look twice as risky as they are. The correlation-space version keeps each volatility exactly at its sample value (tested, V-008) and only shrinks the noisy correlations. It is simple to explain and uses a tested library.
- **Trade-offs:** Shrinking correlations toward zero (not toward the average correlation) can understate how much equity-like assets move together. Shrinkage intensity is small here (0.054) because 232 months is decent data for 11 assets. Option (d) is a reasonable upgrade later.

### D-010: Expected returns
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** **Bayes-Stein shrinkage (Jorion, 1986)** of historical monthly means. Raw historical means are kept for comparison.
- **Options considered:**
  1. Raw historical mean: very noisy. With 19 years of data, the standard error on a 15%-vol asset's mean is about 15%/sqrt(19.3) = 3.4% a year, about as big as the premium itself.
  2. James-Stein toward the grand mean: simple, but the target is arbitrary.
  3. **Bayes-Stein (Jorion):** shrink toward the minimum-variance portfolio's mean, with intensity set by the data.
  4. Equilibrium (reverse optimization, the Black-Litterman prior): back out the returns that make market-cap weights optimal. Needs asset-class market caps, which we would have to assume.
  5. External capital market assumptions (CMAs) published by large asset managers: what practitioners actually use, but not reproducible from code and not mine to justify.
  6. Yield-based for bonds: starting yield is the best predictor of a bond fund's next 5 to 10 year return. Needs yield data (Stage 2).
- **Why this one:** It is data-driven (no hand-picked intensity), has a published justification, and is easy to say: "pull each average toward a common value, more when data are noisy." **Actual result:** shrinkage phi = 0.335, target 1.59%. SPY moves from 11.4% to 8.1%; GLD from 10.7% to 7.6%; TLT from 3.6% to 2.9%.
- **Trade-offs and limitations:** (1) The target is basically short Treasuries, so everything is pulled toward a low number. (2) **Bond returns are still history-based.** On the original 2011 window, every Treasury fund's estimate was below the T-bill rate (P-002). With the 2007 start, they are above it (SHY 1.73%, IEF 2.67%, TLT 2.90% vs 1.38%), because 2008 was a great year for Treasuries. The fact that two extra years flip the sign shows how fragile history-based bond returns are. **Planned fix in Stage 2:** yield-based expected returns for bonds. (3) Gold's 10.7% historical return reflects two big gold rallies in the window; shrinkage cuts it to 7.6% but it remains generous.

### D-011: Risk-free rate
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Average annualized return of BIL (1-3 month T-bills) over the same window: **1.38%** (was 1.51% on the 2011 window).
- **Options considered:** zero; a current T-bill yield (about 4% recently); historical BIL average over the window.
- **Why this one:** Consistent with the expected returns, which are also averages over the same window. Mixing a current 4% rate with long-run average returns would penalize every asset unfairly.
- **Trade-offs:** BIL's return is after a small fee. It is backward-looking. When bond returns become yield-based in Stage 2, the risk-free rate should become the current T-bill yield too, so both are forward-looking.

### D-012: Surplus-based optimizer and liability interface
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Optimize surplus return per dollar of assets, r_S = w'r_A - f r_L (Sharpe and Tint, 1990). E[r_S] = w'mu - f mu_L. Var[r_S] = w'Sigma w - 2f w'c + f^2 sigma_L^2. Liabilities enter through a `LiabilityModel` (f, mu_L, sigma_L, and c = covariance of each asset with liability returns). Default is `None` (no liabilities).
- **Options considered:** (a) asset-only optimizer now, rewrite later; (b) treat the liability as an extra "asset" with a fixed negative weight; (c) surplus formulation with a liability inputs object.
- **Why this one:** It meets the design rule: with f = 0 every term collapses to the asset-only formula, which is tested (V-016, V-017). Stage 3 only has to compute the four liability numbers from cash flows and the rate model; the optimizer does not change. Option (b) is mathematically the same and is what the risk module uses internally, but (c) is clearer to read.
- **Trade-offs:** It is a one-period model: liabilities are summarized by a mean, a volatility and covariances, not by full cash flow timing. Multi-period effects (liquidity, cash flow matching) come in Stage 3.

### D-013: Solver
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** SciPy SLSQP with exact gradients for the main solve. SciPy `linprog` (HiGHS) for maximum achievable return and for finding a feasible starting point. When SLSQP stops with its "positive directional derivative" status, re-solve with SciPy `trust-constr` and accept only if it cannot find a better point.
- **Options considered:** SLSQP; trust-constr alone; a convex QP library (cvxpy or OSQP), which is the industry standard for this problem; closed-form formulas (only without inequality constraints).
- **Why this one:** SciPy was requested. SLSQP handles bounds, equality and inequality constraints, and is fast for 11 assets. Exact gradients make it accurate. The cross-check makes "success" honest (P-004 to P-006).
- **Trade-offs:** SLSQP is a general solver, not a dedicated quadratic programming solver, so it needs care with tolerances. For hundreds of assets or many constraints, cvxpy would be the better tool.

### D-014: Max Sharpe method
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Maximize the Sharpe ratio directly with SLSQP, starting from the maximum-return portfolio. Raise an error if no feasible portfolio beats the risk-free rate.
- **Options considered:** direct maximization; scan the frontier and take the best point; Charnes-Cooper transformation into a convex QP.
- **Why this one:** Simple and exact enough. With a convex feasible set and positive excess return, the Sharpe ratio has a single peak, so a local maximum is the global one. Verified against a 30-point frontier scan (V-014): max Sharpe 0.5671 vs best scanned 0.5670 (long-only); 0.5581 vs 0.5580 (constrained).
- **Trade-offs:** The Sharpe ratio with liabilities is not standard. Currently defined as (E[r_S] - rf) / vol(r_S). With f > 0 this is debatable; Stage 3 will likely use a surplus target-return or surplus-at-risk objective instead. Flagged as open.

### D-015: Efficient frontier construction
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** For evenly spaced target returns from the minimum-variance portfolio's return up to the highest achievable return (solved exactly as a linear program), minimize variance with return fixed at the target.
- **Options considered:** fixed range of targets; risk-aversion sweep (maximize return minus lambda times variance); equality-target method between exact endpoints.
- **Why this one:** Every target is feasible by construction, and points are evenly spaced in return, which makes the frontier easy to read.
- **Trade-offs:** Points cluster at the steep end in risk terms. Equality targets need more care from the solver than a risk-aversion sweep.

### D-016: Risk contribution method
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Euler decomposition of volatility. MCR_i = (Sigma w)_i / vol; contribution_i = w_i MCR_i; they sum exactly to total volatility. With liabilities, the liability appears as a position with weight -f.
- **Options considered:** Euler (volatility); variance decomposition; standalone volatility per asset; tail-risk contributions (CVaR).
- **Why this one:** Standard in industry, adds up exactly (tested, V-019 and V-020), and reflects diversification. Treating the liability as a short position means Stage 3 shows liability-driven risk with no new code.
- **Trade-offs:** Volatility-based, so it ignores fat tails. Contributions can be negative (a hedge), which is correct but needs explaining.

### D-017: Illustrative constraints
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** For the "constrained" demo: each asset 0% to 35%, high yield at most 10%, growth assets (SPY, EFA, EEM, VNQ) at most 40% in total. Labeled illustrative.
- **Options considered:** none (long-only only); invented insurer-like limits; real policy limits (not available).
- **Why this one:** Shows that group constraints work and produces more diversified portfolios than unconstrained MVO.
- **Trade-offs:** **Not calibrated to any insurer's investment policy or regulation.** Real life insurers often hold 70%+ in investment grade bonds and under 10% in equities, largely because of capital charges, which Stage 3 adds.

### D-018: Do not commit raw price data
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** `data/*.csv` is gitignored. `data/prices_metadata.json` (source, version, date range, row count, SHA-256) is committed.
- **Options considered:** commit the CSV; ignore it entirely; commit metadata only.
- **Why this one:** Yahoo's terms restrict redistributing their data, and this repo may be public. The hash lets anyone verify a re-download matches.
- **Trade-offs:** A fresh clone must re-download, and Yahoo's re-adjustments may change the hash (the test will then fail visibly, which is the intended warning).

### D-019: Annualization by x12
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Annual mean = monthly mean x 12; annual covariance = monthly covariance x 12.
- **Options considered:** x12; geometric compounding for means; autocorrelation-adjusted scaling.
- **Why this one:** Standard and consistent with the one-period mean-variance model.
- **Trade-offs:** Assumes months are independent. Assets with smoothed or autocorrelated returns (none here; matters for private assets) would look less risky than they are.

### D-020: Extend the sample window to mid-2007 (supersedes D-005)
- **Date / Stage:** 2026-10-19 / Stage 1
- **Decision:** Request data from 2007-04-01 and let the cleaning step trim to the latest first-valid date. Result: **2007-05-30 to 2026-09-30**, 4,866 trading days, **232 monthly returns (June 2007 to September 2026)**, about 19.3 years.
- **Options considered:** (a) keep 15 years and add 2008 later as a hand-built stress scenario; (b) extend to the earliest date all 12 tickers exist; (c) extend further with proxy indices for HYG and BIL.
- **Why this one:** 2008 is the defining credit and liquidity crisis of modern markets and exactly what an insurer's board asks about. Having it in the estimation data, not just as a stress scenario, makes the risk model honest about credit and real estate. Option (b) gets it with real ETF data only. Option (c) adds proxy risk for little gain. Decided by me after the trade-off was raised (open question 1 in the previous version).
- **What changed (15-year to 19-year window):** HYG volatility 7.0% to 10.2%; VNQ 16.7% to 22.3%; EEM 16.9% to 21.1%. October 2008 is the worst month for 7 of 11 assets (VNQ -31.7%, EEM -25.6%, SPY -16.5%, HYG -11.5%). Treasury expected returns moved from below cash to above it. Long-only max Sharpe flipped from 83% SPY to 74% SHY. Full comparison in section 6.
- **Trade-offs and limitations:** Still only one big credit crisis; no 2000 to 2002 equity bear market, no 1994 bond crash, no 1970s inflation. The 2007 to 2009 period now carries a lot of weight in a shorter-than-ideal history, so results are sensitive to it. The cash proxy (BIL), not an investable asset, sets the start date.

### D-021: Monte Carlo return generator
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** Default is a **block bootstrap of historical months**. Multivariate normal and multivariate Student-t (5 degrees of freedom) are also implemented for comparison.
- **Options considered:** (a) multivariate normal; (b) multivariate Student-t; (c) block bootstrap of history; (d) regime-switching or GARCH models.
- **Why this one:** The bootstrap uses real months, so fat tails, negative skew, crisis correlations and volatility clustering come from the data instead of an assumed shape. It is easy to explain ("we replay chunks of real history in random order"). The normal model is kept because it is the textbook baseline, and showing how much it understates tails is a key result: **normal understates 1-year 99% CVaR by 16.5% to 38.2% depending on the portfolio** (section 6). Student-t is the middle ground; with df = 5 it recovers only part of the gap.
- **Trade-offs and limitations:** The bootstrap can only replay what happened in our 19 years; nothing worse than 2008 can appear, and with only 232 months the far tail is thin. Normal and t need no history but impose a shape. (d) is more realistic but much harder to explain and calibrate.

### D-022: Mean-adjusted bootstrap
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** Before resampling, shift each asset's history so its mean equals the same Bayes-Stein expected return used by the normal model.
- **Options considered:** raw bootstrap (historical means); mean-adjusted bootstrap.
- **Why this one:** It makes the comparison fair: normal and bootstrap then share the same mean (and, nearly, the same volatility), so any difference is purely tail shape and clustering. It also keeps the simulation consistent with the expected returns used to build the portfolio. Tested: bootstrap mean matches the target to 1.1e-4 per month.
- **Trade-offs:** Shifting means keeps everything else about history, including the 2008 sequence. Volatilities in the bootstrap are sample volatilities, the same as the normal model's (the covariance keeps sample volatilities, D-009); only correlations differ slightly (shrunk vs sample).

### D-023: Bootstrap block design
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** Circular blocks of 6 consecutive months.
- **Options considered:** block length 1 (independent months), 3, 6, 12; fixed vs random-length (stationary) blocks; circular vs non-circular.
- **Why this one:** 6 months keeps crisis sequences (several bad months in a row) together while still giving about 38 distinct starting blocks per pass through 232 months. Circular wrapping makes every month equally likely to be drawn.
- **Trade-offs:** Sensitivity is reported, and it is **noisy, not monotonic**: constrained max Sharpe 1-year CVaR 99% is 12.3% (1-month blocks), 12.6% (3), 13.7% (6), 12.5% (12). The choice of block length matters less than the choice of normal vs bootstrap. Blocks join at random points, so crisis sequences longer than 6 months get cut.

### D-024: Monthly rebalancing in simulations and stress tests
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** Portfolios are rebalanced to target weights at every month end, in Monte Carlo and in historical scenarios. Hypothetical shocks are instant, so no rebalancing applies.
- **Options considered:** monthly rebalancing; buy-and-hold (weights drift); threshold rebalancing (trade only when weights move outside a band).
- **Why this one:** It matches the one-period optimizer, whose weights are targets, and keeps the risk profile constant through the horizon.
- **Trade-offs:** No transaction costs. Real insurers rebalance less often and within tolerance bands.

### D-025: Monte Carlo outputs and definitions
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** 10,000 paths, 5 years monthly, fixed seed (20261019). Report at **both 1-year and 5-year horizons**: VaR and CVaR at 95% and 99% (as positive losses of surplus per dollar of starting assets), probability of a loss, probability of missing an illustrative 4% a year target, terminal wealth percentiles, and the max drawdown distribution (median, 95th, 99th percentile).
- **Options considered:** horizon 1 year only (regulatory convention); 5 years only (strategic horizon); both. VaR only vs VaR and CVaR.
- **Why this one:** Capital regimes think in one-year losses; SAA thinks in multi-year outcomes; reporting both covers both audiences. CVaR is added because VaR ignores how bad the tail is.
- **Surplus definitions:** gains = S_h - S_0 with A_0 = 1 and L_0 = f. Drawdown = (peak surplus - surplus now) / asset value at that peak. With no liabilities these reduce to ordinary cumulative return and ordinary drawdown. Liabilities plug in through the same `LiabilityModel`; the bootstrap additionally needs a liability return history (Stage 3), and raises a clear error without it.
- **Trade-offs:** The 4% target is illustrative, not an insurer's actual hurdle. 10,000 paths gives about 100 paths in the 99% tail, so 99% figures carry noticeable simulation noise (roughly a few tenths of a percentage point).

### D-026: Historical stress windows, marked daily
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** Four windows defined by whole months: **2008 GFC** Nov 2007 to Feb 2009; **2013 taper tantrum** May to Aug 2013; **2020 COVID** Feb to Mar 2020; **2022 rate shock** calendar 2022. The portfolio is marked with **daily prices** and rebalanced monthly.
- **Options considered:** window dates (peak-to-trough by SPY, by bonds, or calendar); monthly vs daily marking.
- **Why this one:** Dates match how each event is commonly described, chosen in advance by economic reasoning rather than by picking each portfolio's worst stretch. Daily marking shows falls inside a month. Month-end values of the daily path equal the monthly portfolio returns exactly (tested, max difference 3.3e-16), so the two views are consistent.
- **Finding:** Monthly data hides intra-month falls. COVID: constrained max Sharpe daily drawdown 10.5% vs 4.6% from month ends; min variance 0.85% vs 0.0%.
- **Trade-offs:** Fixed windows do not capture each portfolio's own worst period. The 2022 window uses the calendar year; long Treasuries kept falling into October 2023, which is outside it.

### D-027: Hypothetical shocks (simple factor shocks)
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** Instant shocks, unlisted assets unchanged:
  - **Equity -30%:** SPY, EFA, VNQ -30%; EEM -35% (EM fell more in 2008 and 2020). No bond rally assumed (conservative, 2022-style).
  - **Credit spreads wider:** HYG -12%, LQD -10%, about each fund's worst single month in our data (HYG -11.5% Oct 2008, LQD -10.7% Sep 2008).
  - **Rates up (about +1 percentage point):** SHY -2%, IEF -7%, TLT -15%, TIP -6%, LQD -7%. Judgment-based rate sensitivities, longer bonds lose more.
  - **All three at once:** sum of the above.
- **Options considered:** hand-specified shocks per asset; shock one factor and spread it to other assets using historical betas from the covariance matrix; duration-based shocks (Stage 2).
- **Why this one:** Transparent and auditable: every number is written down and can be challenged. Beta propagation uses normal-times correlations, which break in crises, and can imply bonds rally in every equity crash (true in 2008, false in 2022).
- **Trade-offs:** **Labeled as simple factor shocks, NOT duration-based.** Sizes are judgment. Stage 2 will compute rate and spread shocks from duration and spread duration. The optimized portfolios hold no credit, so the credit shock barely touches them (P-011); the equal-weight benchmark shows it (-2.0%).

### D-028: Rates regime rule
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** Month t is "rising" if the 10-year Treasury yield (^TNX) at the end of month t-1 is higher than 12 months before that; otherwise "falling".
- **Options considered:** lookback 3, 6 or 12 months; threshold 0 or a neutral band (e.g. +/-25bp, making three regimes); signal from yields vs from bond ETF returns.
- **Why this one:** 12 months captures the rate cycle rather than noise (25 label changes in 232 months). A zero threshold gives the two regimes requested and needs no tuning. Actual yields are cleaner than inferring rates from bond fund returns, which include coupon income.
- **Checks:** **all 12 months of 2022 are "rising"** (V-038). 103 rising months, 129 falling.
- **Trade-offs:** Lags turning points by design. A 1bp rise counts the same as 300bp. ^TNX is a new data series (cached separately, D-029).

### D-029: Volatility regime rule and signal data
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** Month t is "high" volatility if SPY's realized volatility over the 63 trading days ending at month end t-1, annualized, exceeds **20%**; otherwise "low". Signals (^TNX and SPY from 2006-01-03) are cached in a separate file, `data/signals_daily.csv`, with their own hash, so they never enter returns or the optimizer.
- **Options considered:** VIX (implied volatility) vs realized volatility; window 21, 63 or 126 days; threshold fixed 20%, full-sample median (look-ahead), or expanding median.
- **Why this one:** Realized volatility measures what actually happened and comes from data we already use. 63 days (about 3 months) balances speed and noise. A fixed 20% is easy to explain (roughly the long-run average level of the VIX) and avoids look-ahead from a full-sample median; an expanding median has too little history at the 2007 start.
- **Checks:** **Sep 2008 to Jun 2009 all "high"**; 8 of 12 months of 2008 high (V-039). 52 high months (22%), 180 low.
- **Trade-offs and limitations:** The 20% level was picked knowing roughly what market volatility looks like, a mild form of hindsight. **The rule lags sudden crashes:** Feb and Mar 2020 are labeled low because volatility was low at the end of January; the high-vol label then catches the April to August 2020 rebound. That is why equities show *higher* average returns in high-vol months (SPY 14.8% vs 10.4%). A trailing rule says "we are in a turbulent period", not "a crash is coming".

### D-030: Equal-weight benchmark portfolio
- **Date / Stage:** 2026-10-19 / Stage 1b
- **Decision:** Add an equal-weight portfolio (9.1% in each of the 11 assets), clearly labeled as a benchmark, not optimized, to every Stage 1b report.
- **Options considered:** report only the three optimized portfolios; add 100% SPY; add equal weight; add a 60/40.
- **Why this one:** The optimized portfolios hold no credit, no TLT, no TIPS, so several shocks had nothing to hit (P-011). Equal weight holds everything, so every scenario shows its effect, and it is a standard naive benchmark.
- **Trade-offs:** Not representative of an insurer portfolio.

### D-031: Single-asset caps folded into bounds; exact frontier endpoint
- **Date / Stage:** 2026-10-19 / Stage 1b (cleanup of P-008)
- **Decision:** (1) A group constraint on one asset (e.g. "HYG at most 10%") is intersected with that asset's weight bound instead of being passed as a separate constraint row. All groups, including single-asset ones, are still checked in `check_feasible`. (2) The efficient frontier's max-return endpoint uses the exact linear program solution instead of re-solving it as a degenerate quadratic problem.
- **Options considered:** keep suppressing the warning; fold caps into bounds only; fold caps and fix the endpoint.
- **Why this one:** Folding was requested and removes a genuinely redundant row. But it did NOT remove the warning (P-008 update): the real cause was the degenerate endpoint. Using the LP answer there is exact and removes the last source. Result: **zero warnings with all warnings treated as errors**, in tests and both pipelines. Results unchanged: frontier weights moved by at most 1.9e-7, volatilities by 1.4e-11.
- **Trade-offs:** If two portfolios tie for the maximum return, the LP returns one of them, not necessarily the lower-risk one. Documented in the code.

### D-032: Rate and credit spread data (FRED)
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** From FRED's public CSV endpoint (no API key): Treasury CMT yields 3M, 1Y, 2Y, 5Y, 7Y, 10Y, 20Y, 30Y; Moody's Baa minus 10-year Treasury (BAA10Y) as the **credit spread factor**; ICE BofA IG OAS and HY OAS used **only for today's spread level**. Cached to `data/fred_daily.csv` with a hash. Month-end values (last valid day of each month) aligned to the ETF months, including the base month May 2007 so the first change exists.
- **Options considered:** (a) ICE OAS for everything, calibrated on 3 years; (b) Baa-10y factor + each credit ETF's measured sensitivity to it, ICE only for today's level; (c) build an HY spread history from HYG returns (circular); (d) pause for paid data.
- **Why this one:** The ICE series on FRED start 2023-10-20 (licensing limit, P-012), so (a) would calibrate spread risk on three calm years with no crisis. (b) uses only real data with full history including 2008. **Chosen by me after the trade-off was put to me.**
- **Trade-offs:** One credit factor: HY has no independent spread dynamics; its sensitivity to the Baa factor (9.68) is measured, not modeled. Baa-10y is a yield spread on long Baa bonds, not an OAS, and runs higher (1.47% vs ICE IG 0.84% today). Daily gaps (bond holidays) do not matter at month ends.

### D-033: Curve construction and interpolation
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** CMT yields are semiannual par yields; the 3M yield is treated as a zero rate. Par yields are interpolated **linearly** onto a semiannual grid, discount factors are bootstrapped node by node, and between nodes the discount factor is interpolated **log-linearly** (a constant forward rate between nodes). Flat forward beyond 30 years.
- **Options considered:** linear on zero rates; cubic spline on par or zero rates; log-linear discount factors; monotone-convex (Hagan-West); a fitted curve (Nelson-Siegel-Svensson).
- **Why this one:** Simple, stable, and exactly reprices every par bond (tested to 1e-9, 2.8e-14 on real curves). Log-linear discount factors cannot produce negative forward rates from a sensible curve and never overshoot. Splines give smoother forwards but can oscillate between quotes. Nelson-Siegel does not reprice the inputs exactly.
- **Trade-offs:** Forward rates jump at the nodes (a sawtooth), which would matter for pricing forward-starting instruments, not for discounting liability cash flows at this level of precision. Linear par interpolation is a modeling choice between quoted tenors.

### D-034: Bond conventions and risk measures
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** Semiannual coupons, face 100, priced on a coupon date (no accrued interest). Measures: Macaulay, modified duration, convexity and DV01 in yield terms; **effective duration** and **key rate durations** by bumping the par curve and rebuilding. Key rate bumps are triangles from the linear interpolation, so they sum to a parallel shift and the KRDs sum to effective duration (tested, relative error up to 6.4e-7).
- **Options considered:** bump par yields vs zero rates for KRDs; exact day counts and accrued interest.
- **Why this one:** Bumping par yields matches how the curve is quoted and makes the KRDs add up. Day counts and accrued interest change numbers by tiny amounts and add a lot of code.
- **Trade-offs:** No day counts, no settlement lags, no callable or amortizing bonds. Effective duration (par-curve shift) is about 1% higher than modified duration (yield shift) for long bonds, because a par-curve shift is not a parallel zero-curve shift; both are reported.

### D-035: Stochastic rate model: Vasicek, confirmed with caveats
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** **Confirm** the proposed one-factor Vasicek model with **exact discretization**, short rate = 3-month yield. The rate and its time integral are drawn jointly from their exact distribution each month, so discounting has no time-step error.
- **Options considered:** Vasicek; Hull-White (Vasicek with a time-varying level that fits today's curve exactly); CIR (no negative rates); two-factor models (G2++ or a short rate plus a long-rate factor).
- **Why this one:** Closed-form mean, variance and bond prices give exact tests (mean and variance within 2 standard errors, martingale within 1.2 SE on real parameters). It is the easiest to explain. Hull-White fits today's curve but its fitted level comes from the curve itself, so it says little about how rates move historically. CIR's square-root volatility rules out negative rates but has non-normal distributions, harder calibration and fewer closed-form checks.
- **Challenge (stated plainly):** **Vasicek does NOT fit today's curve** (model curve is nearly flat at 4.2% while the real curve rises to 5.65% at 30 years) and **it can go negative** (3.1% of paths below zero at 5 years). More important in practice: **one factor badly understates long-bond risk**. Every yield moves only as a fraction B(T)/T of the short rate's move, so long yields barely move: simulated TLT volatility is 4.5% vs 14.1% historically, and TLT and IEF end up with almost the same volatility despite durations of 17 and 7.5 (P-013). For Stage 3, where liabilities are long-dated, I recommend a second (long-rate) factor (open question 10).

### D-036: Vasicek calibration sample, uncertainty and term premium
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** Fit the exact AR(1) form by OLS on monthly 3M yields, **using only month pairs outside both zero-bound spells** (Dec 2008 to Nov 2015 and Apr 2020 to Feb 2022; 107 of 233 months excluded). Result: a = 0.150 (se 0.184), half-life 4.6 years, b = 2.11% (se 2.29%), sigma = 1.02% (se 0.06%). Then fit one market-price-of-risk parameter lam = 0.337 so that the model's average 10-year zero yield over the sample equals the actual average; pricing level b_q = 4.39%.
- **Options considered:** full sample; excluding only 2008 to 2015; excluding both spells; post-2015 only; fixing b from an external forward-looking anchor.
- **Why this one:** At the zero bound the rate was pinned, not calm: 110 of 233 months had the 3M yield below 0.25%. Those months pull b toward zero (full sample b = 1.27%) and shrink sigma (0.75% vs 1.02%). The user flagged 2009 to 2015; the 2020 to 2022 spell has the same problem, so both are excluded. Only consecutive months both outside the spells are used, so no fake jump is fitted across a gap (tested).
- **Uncertainty, stated plainly:** Mean reversion is barely identified: a's standard error is larger than the estimate, and in a 500-draw block bootstrap of the full sample 31% of resamples showed no mean reversion at all. b's 95% interval runs from about -2.4% to +6.6%. Only sigma is precise.
- **Trade-offs:** Excluding months cuts the sample to 123 pairs. The real-world drift pulls the short rate from 4.2% toward 2.1% (median 3.1% after 5 years), which builds a bullish bond view into simulated returns (TLT simulated mean 6.7% vs 5.7% yield). Open question 11 proposes a forward-looking anchor for b.

### D-037: Representative bonds with a measured rate beta
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** Each bond ETF is a **par bond at a representative maturity** (SHY 2y, IEF 8.5y, TLT 25y, TIP 7.5y, LQD 12y, HYG 4y) times a **rate beta** = empirical duration / par-bond duration, used where the regression fits well (rates-only R2 at least 0.9: SHY 0.96, IEF 1.11, TLT 1.28) and for TIP (0.72, its sensitivity to nominal yields). Credit funds use beta 1 and a par bond at Treasury + today's OAS.
- **Options considered:** published durations as given; par bond at the representative maturity; bond with the coupon solved to match the empirical duration; par bond plus rate beta.
- **Why this one:** It keeps realistic bonds (normal coupons) for convexity and full revaluation, while matching how each fund actually behaved. The coupon-matching version produced a 41% coupon for TIP and 12.5% for SHY (P-015). Empirical durations cross-check well against rough published figures: SHY 1.80 (about 1.9), IEF 7.53 (about 7.1), TLT 16.96 (about 16.3), all R2 at least 0.92. TLT's beta above 1 reflects its low-coupon bonds trading below par; TIP's 0.72 says only about 72% of a nominal yield move reaches real yields.
- **Where it fails:** HYG's rates-only R2 is 0.00 (duration 0.43, not significant): high yield moves with credit and equity, not rates. LQD's is 0.40. With the Baa factor added, R2 rises to 0.48 (HYG) and 0.73 (LQD), but the two-factor rate durations are inflated by collinearity (HYG 5.15 vs about 3 published), so credit funds use bond math for rates.
- **Trade-offs:** Published durations are quoted from memory, not fetched. Convexity is scaled by beta squared, an approximation. Durations are held constant over the horizon.

### D-038: Credit spread model and credit returns
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** The Baa spread follows a **mean-reverting process on its logarithm** (exact discretization): kappa = 0.46 (se 0.23), long-run median 2.36%, volatility of the log spread 0.28. Credit ETF monthly return = carry (Treasury yield + today's ICE OAS scaled by Baa_t / Baa_0) / 12 - expected default loss / 12 - rate duration x yield change - spread beta x Baa change + convexity term. Expected annual loss: IG 0.20%, HY 2.50%.
- **Options considered:** Vasicek on the spread level (can go negative); CIR (non-negative if a parameter condition holds); log mean-reverting (always positive, volatility scales with the level).
- **Why this one:** Positive by construction (minimum simulated spread 0.51% to 0.57%; still positive at an absurd 300% volatility, tested). Volatility proportional to the level matches reality: spreads jump most when already wide (2008). Easy to fit with the same AR(1) method as rates.
- **Simplifications, every one:** one credit factor for IG and HY; constant default loss (no random defaults or default clustering); no downgrade migration beyond the IG loss assumption; spread duration via measured beta in simulation and bond-math spread duration in stress tests; ICE spread level scaled proportionally with Baa; no fees. Default loss numbers are rough long-run rating agency averages (HY: about 4% default rate x 60% loss; IG: default plus downgrade drag), judgment, not fetched.

### D-039: Correlated factors and the equity side
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** Three standardized shocks (rate residual, log-spread residual, equity return), correlation estimated on all 232 months: rate-spread -0.38, rate-equity +0.18, spread-equity -0.58. Correlated draws via the Cholesky factor. SPY is a normal equity factor (Bayes-Stein mean 8.1%, historical volatility 15.5%). EFA, EEM, VNQ, GLD: regression on the three shocks plus correlated normal residuals, means set to Bayes-Stein.
- **Options considered:** correlations on all months vs only months outside the zero bound; equity factor normal vs bootstrapped.
- **Why this one:** Using all months keeps 2008 and 2020 in the spread-equity correlation, which is where it matters. During the zero bound the rate shocks are tiny, so they contribute little to the rate correlations anyway. Normal shocks keep the factor model simple and testable.
- **Trade-offs:** Normal equity shocks have thin tails (Stage 1b showed normal understates the 99% CVaR by 16% to 38%). The rate factor is the 3M rate, which is weakly linked to equity; long rates matter more (TIP's simulated correlation with SPY is -0.18 vs +0.34 historically).

### D-040: Matching today's curve with a constant offset
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** Simulated yields = Vasicek model yields + a constant offset per maturity, equal to (today's actual zero yield - the model's zero yield today). At time 0 the simulation sits exactly on today's curve (tested to 1e-12). Offsets: about 0 at 3 months, +0.6% at 2y, +1.0% at 10y, +1.6% at 20y.
- **Options considered:** no offset (start on the model's flat curve); constant offset; Hull-White time-varying level.
- **Why this one:** Carry (the starting yield) is the biggest part of bond returns; starting on the wrong curve would misstate every bond's return by up to 1.6% a year. The offset fixes the level without changing the dynamics.
- **Trade-offs:** Not arbitrage-free in the strict Hull-White sense (the offset is not derived from the model's forward rates). Model zero yields are mapped to funds' par-yield-based durations, a small mismatch.

### D-041: Yield-based expected returns for bond ETFs
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** Bond ETF expected return = today's par yield at its representative maturity; credit adds today's ICE OAS and subtracts expected default loss; TIP uses the nominal yield (assumes realized inflation equals breakeven). Equities, REITs and gold keep Bayes-Stein. The risk-free rate switches from the historical BIL mean (1.38%) to **today's 3M yield (4.20%)**, so bonds and cash are both forward-looking.
- **Options considered:** keep history (Bayes-Stein); starting yield; starting yield plus expected roll-down; yields from the factor model's simulated path.
- **Why this one:** Starting yield is the single best simple predictor of a bond fund's return over the next several years, and it is transparent. Result: SHY 4.88%, IEF 5.24%, TLT 5.66%, TIP 5.21%, LQD 6.01%, HYG 5.64%. **The Treasuries-below-cash problem is gone and now rests on today's yields** (excess over cash: SHY +0.68%, IEF +1.04%, TLT +1.46%), not on which years happen to be in the sample.
- **Trade-offs:** Mixes forward-looking bond returns with backward-looking equity returns (SPY 8.1% Bayes-Stein), so the equity premium over cash shrinks to 3.9%. Ignores fund fees (HYG's is about 0.5%), roll-down, BEY-vs-annual compounding (about 6bp), and assumes yields do not change. The covariance is still historical.

### D-042: Duration-based stress tests
- **Date / Stage:** 2026-10-20 / Stage 2
- **Decision:** Shocks: parallel +100bp and -100bp; steepener (short end -50bp, 10y 0, 30y +50bp, linear between); flattener (the reverse); IG OAS +100bp; HY OAS +400bp. For each bond ETF: first-order term = key rate durations x the shock at each key tenor, plus effective convexity x (shock at maturity)^2 / 2, with the rate beta applied; compared with **full revaluation** of the representative bond. Spread shocks shift the bond's yield. Equities, REITs, gold and BIL are not shocked (0%), stated in the report. The Stage 1b simple factor shocks are kept alongside, labeled.
- **Options considered:** shock size; using ETF-level duration only vs KRDs; revaluing off the curve vs off the yield.
- **Why this one:** KRDs handle non-parallel shocks correctly to first order. Showing full revaluation next to the estimate makes the approximation error visible: at most 0.22 percentage points (TLT, -100bp, effective shock -128bp), 0.07 for HY +400bp. Spread shock sizes approximate a moderate recession; 2008 was 3 to 4 times larger for high yield.
- **Trade-offs:** Rate shocks are instant and leave equities untouched (a real rate shock would move them). The optimized portfolios still hold no credit, so spread shocks hit only the equal-weight benchmark (-0.72% IG, -1.13% HY).

### D-043: Empirical PCA scenario generator as the Stage 3 default
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** PCA on monthly par-curve changes (8 tenors), keep 3 factors (level 78.4%, slope 15.0%, curvature 4.8%; 98.2% together). Each simulated month is a real historical month, drawn in 6-month blocks; that month's curve factor moves and all ETF returns are used together. Bond ETF return = curve part (carry - duration x simulated yield change + convexity) + that month's historical residual (the part the curve did not explain, demeaned); credit adds today's OAS minus expected loss. Equity-like assets: that month's return, mean-adjusted to Bayes-Stein. Vasicek stays as the analytic benchmark.
- **Options considered:** keep Vasicek; two-factor Gaussian (G2++); PCA with parametric (normal) factor shocks; PCA with bootstrapped factor moves (joint with asset returns).
- **Why this one:** It fixes P-013 with real data: TLT volatility 14.0% vs 14.1% historically (Vasicek 4.5%); 10-year yield volatility 87bp vs 88bp (Vasicek 52bp). Drawing whole months keeps every cross-asset link: simulated correlations with SPY match history (HYG 0.73 vs 0.73, TIP 0.35 vs 0.34). The first version used only the curve and a spread formula and still made HYG and TIP too calm (P-019); adding the historical residual fixed it.
- **Trade-offs:** Cannot produce anything worse than 2007 to 2026. Zero-bound months make short-end volatility too low (3-month: 74bp vs 100bp outside the zero bound). TLT's simulated mean (6.8%) exceeds its yield (5.7%) partly because of the convexity term, which markets partly price away.

### D-044: Forward-looking rate anchor
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** The curve is pulled toward the **market 5-year forward par curve** (5.40% at 3 months to 5.84% at 30 years, computed from today's cached curve) at kappa = 0.15 a year. Sensitivities: today's curve (no change) and the 2007 to 2026 historical average curve (1.54% to 3.44%). Vasicek benchmark uses b = 5.40% (historical estimate 2.11%) with the term premium refit.
- **Options considered:** historical average (Stage 2); today's curve; market forward curve; the Fed's published longer-run policy rate (FRED series FEDTARMDLR).
- **Why this one:** It is forward-looking, market-based and needs no new download. FRED was unreachable during this stage (P-018), so the Fed series could not be fetched and was not guessed.
- **Trade-offs:** Forward rates include a term premium, so they lean above true expected rates; the anchor therefore tilts toward rising rates. Results are robust for the recommended (banded, D-057) portfolio: 5-year median funding ratio 120.3% (forward), 120.1% (today), 119.6% (historical); expected 1-year surplus change $10.0m, $10.6m, $14.8m. (Before D-057, the unbanded portfolio showed 119.9%, 119.9%, 121.0%.)

### D-045: Liability shape
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** SYNTHETIC payout-annuity block: 30 annual payments at year ends, falling 3% a year (annuitants dying), scaled to PV $1,000m on today's curve. First payment $93.3m. Effective duration 9.48, convexity 147.5, DV01 $0.95m per bp; key rate durations concentrated at 10 to 30 years (10y 1.86, 20y 3.86, 30y 2.66). Optional 10% surrender shock for the liquidity test.
- **Options considered:** level 30-year annuity (duration 11.2); decay 3% (9.5); 5% (8.4); 8% (7.1); deferred annuities with a hump.
- **Why this one:** Duration about 9.5 is typical of an annuity book; the geometric shape has a closed-form value on a flat curve, which gives an exact test.
- **Trade-offs:** No mortality model, no inflation linkage, no policyholder options beyond the surrender test, annual rather than monthly payments.

### D-046: Liability discounting on the Treasury curve
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** Discount liabilities on the same Treasury zero curve as the assets.
- **Options considered:** Treasury curve; Treasury + an IG credit spread (as some accounting regimes use); a fixed statutory rate.
- **Why this one:** Consistent with the assets and required by the brief; a rate move hits both sides through one curve.
- **Trade-offs:** **This choice is why the optimizer avoids credit**: spreads change asset values but not liability values. With credit-spread discounting, credit would partially hedge the liabilities. Open question 18.

### D-047: Starting balance sheet
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** Assets $1,100m, liabilities $1,000m, surplus $100m, funding ratio 110%.
- **Why this one:** About the 10% surplus cushion life insurers carry; round numbers.
- **Trade-offs:** Illustrative. To match dollar duration, assets need duration 9.48 / 1.1 = 8.6.

### D-048: Reusing the Stage 1 optimizer for ALM
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** Over one year with buy-and-hold returns R_i and liability return r_L = (L_1 + payments) / L_0 - 1, surplus change / A0 = w'R - f r_L exactly (tested to 1e-12 against the path simulation). So scenario means and covariances feed the unchanged Stage 1 surplus optimizer via LiabilityModel. New: a general linear-rule constraint type for liquidity and capital.
- **Why this one:** It is the payoff of the day-one design rule: liabilities plugged in without rewriting the optimizer.
- **Trade-offs:** One-year buy-and-hold horizon for optimization; 5-year results come from simulation with monthly rebalancing.

### D-049: Liquidity buckets, haircuts and stressed outflow
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** Buckets and haircuts (labeled assumptions inspired by Basel III LCR): Level 1 Treasuries (SHY, IEF, TLT, TIP) 0%; Level 2A investment grade (LQD) 15%; Level 2B (HYG, equities, REITs, gold) 50%, applied after the stress price move. Stress: rates +100bp plus IG +100bp and HY +400bp; severe adds equities -30% (EEM -35%). Outflow = one year of payments ($93.3m) + 10% surrender ($100m) = $193.3m. Constraint: coverage (severe) at least 1.0.
- **Why this one:** Simple, conservative, explainable; Basel buckets are a recognized reference even though insurers have no single standard.
- **Trade-offs:** **The constraint never binds**: coverage is 4.0x to 5.5x for every portfolio because they hold mostly Treasuries. It would matter for portfolios heavy in private or illiquid assets, which we do not model.

### D-050: Capital proxy
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** PROXY capital = asset charges (Treasuries 0%, IG 1%, HY 6%, equities, REITs and gold 30% of market value) + an interest-rate charge (the worse surplus loss under +/-100bp, floored at 0). Simple sum, no diversification credit. Constraint: surplus / capital at least 2 (capital at most $50m).
- **Options considered:** factors only; factors plus rate charge; square-root covariance aggregation like NAIC.
- **Why this one:** Captures the two things that drive insurer capital in this universe (asset default risk and rate mismatch), stays linear so it fits the optimizer as constraints, and is easy to audit.
- **Trade-offs:** **NOT NAIC RBC.** Factors are orders of magnitude, not regulatory values; simple sum overstates capital vs a covariance formula; no tax effect.

### D-051: Insurer constraints and what the optimizer does about credit
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** Investment grade credit (LQD) at least 40%; equities (SPY, EFA, EEM) at most 10%; high yield at most 5% (recommended below the suggested 10%, closer to typical life insurer below-investment-grade shares); REITs at most 5%; gold at most 5%; liquidity coverage at least 1.0; capital proxy at most surplus / 2.
- **Finding, reported as is:** Without the credit floor, the optimizer holds **no credit**; the surplus-optimal portfolio at the same $10m target is Treasuries plus small equity and gold (surplus volatility $20.1m). With the floor it holds **exactly 40% and never more** except at the maximum-return corner. The floor raises surplus volatility to $33.1m and the 1-in-20 loss from $36.7m to $65.8m.
- **Why this one:** The brief's insurer-like constraints; reporting the unforced result shows the floor is a policy choice, not an optimization result (driven by Treasury discounting, D-046).
- **Trade-offs:** High yield, REIT and liquidity limits never bind here.

### D-052: Objective and the recommended portfolio
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** Minimum one-year surplus volatility for a target expected surplus change of 10% of starting surplus ($10m a year), under D-050 and D-051 constraints. CVaR variant via Rockafellar-Uryasev LP at the same target. Surplus Sharpe reported only as a number.
- **Options considered:** maximize surplus Sharpe; minimum CVaR; target equal to the asset-only portfolio's return (infeasible: $16.7m exceeds the $12.9m maximum under insurer constraints); a growth target like an insurer return on capital.
- **Why this one:** A business-anchored target (about 10% return on surplus) inside the feasible range ($7.7m to $12.9m). Result: LQD 40%, TLT 39.4%, SHY 17.4%, GLD 3.2%; surplus volatility $33.1m; 1-in-20 average loss $65.8m; capital $31.8m (coverage 3.1x); liquidity 4.6x; 5-year median funding ratio 119.9%, 5th percentile 104.0%. CVaR variant nearly identical (TLT 40.2%, SHY 17.1%, GLD 2.7%; 1-in-20 loss $65.8m).
- **Trade-offs:** The 10% target is a judgment. The frontier is reported so other targets can be read off.
- **Stage 3b update (2026-10-20):** no longer the headline. Superseded by D-057 (same objective plus the duration-gap band); this unbanded portfolio is kept as the comparison.

### D-053: Duration posture and the band variant
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** Report that the recommended portfolio is **long duration** (asset duration 10.46 vs 8.62 needed; dollar gap +$20.3m per 100bp). Add a variant with the duration gap limited to plus or minus $10m per 100bp, and leave the choice to the investment committee.
- **Why:** Minimum surplus variance uses extra TLT to hedge the forced credit position, because historically Treasuries rallied when spreads widened (Baa spread changes vs 10-year yield changes: correlation -0.41). That failed in 2022. Rates +100bp plus IG +100bp costs about $51.7m. The band variant loses $7.7m instead of $16.8m for +100bp, but its surplus volatility is $34.7m (vs $33.1m) and its 1-in-20 loss $71.5m (vs $65.8m) under historical correlations.
- **Stage 3b update (2026-10-20):** the band variant is now the headline recommendation (D-057).

### D-054: Key-rate matched benchmark
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** Treasury-only portfolio (SHY, IEF, TLT) minimizing the squared gap between asset and liability dollar key rate durations: 33.6% SHY, 31.3% IEF, 35.1% TLT; asset duration 8.99, gap +$4.1m per 100bp; surplus change under +/-10bp: -$0.39m / +$0.43m (vs +$7.4m / -$7.6m for 100% SHY).
- **Why this one:** Shows what pure rate immunization achieves: surplus volatility $10.6m, capital $2.3m, but only $4.5m expected surplus growth.

### D-055: Client report format
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** `reports/client_report.md` plus `reports/client_report.pdf` (7 pages) built with matplotlib's PdfPages. Every number comes from the computed results; plain language; labeled illustrative and proxy.
- **Options considered:** markdown only; HTML; PDF via a new dependency (reportlab, WeasyPrint); PDF via matplotlib.
- **Why this one:** No new dependency, renders anywhere, version-controllable markdown source.
- **Trade-offs:** Basic layout; text pages are typeset by hand (P-021).

### D-056: Regime tie-in with historical liability returns
- **Date / Stage:** 2026-10-20 / Stage 3
- **Decision:** Value the liability profile on each historical month's curve (rolled one month each time) to get monthly liability returns (annual volatility 9.0%, correlation with TLT 0.98), then surplus change = w'r - f r_L per month, summarized by Stage 1b regimes; plus Stage 2 duration-based shocks.
- **Finding:** Average surplus change: falling-rate months $7.8m/yr, rising-rate $4.2m/yr; high-volatility $27.3m/yr (includes rebounds, D-029), low-volatility $0.1m/yr. Worst month -$47.9m. Worst instant shock: IG +100bp, -$34.9m (funding ratio 106.5%).
- **Trade-offs:** In-sample; uses actual historical ETF returns rather than yield-based expectations.

### D-057: Duration-banded portfolio becomes the headline recommendation
- **Date / Stage:** 2026-10-20 / Stage 3b
- **Decision:** The recommended portfolio is now the D-052 objective plus the duration-gap band (dollar gap within plus or minus $10m of surplus per 100bp). The unbanded D-052 portfolio is kept as the comparison in every report.
- **Options considered:** keep the unbanded portfolio as headline (lowest surplus volatility under historical correlations); banded headline; a tighter band (for example plus or minus $5m).
- **Why this one:** The unbanded optimum is long duration (+$20.3m per 100bp) because it relies on Treasuries rallying when credit sells off. That relationship failed in 2022. A board would rather pay a little expected efficiency than depend on a correlation that can flip.
- **Result:** LQD 40.0%, TLT 33.7%, SHY 19.8%, GLD 5.0%, SPY 1.5%. Asset duration 9.53 vs 8.62 needed; gap +$10.0m per 100bp (the band binds).
- **Cost (vs unbanded):** surplus volatility $34.7m vs $33.1m (+$1.6m); 1-in-20 average loss $71.5m vs $65.8m (+$5.7m); capital proxy $33.5m vs $31.8m (+$1.7m); probability the 5-year funding ratio ends below 100%: 2.9% vs 2.3%. These costs are measured with historical correlations, which is exactly where the unbanded portfolio looks best.
- **Benefit:** a +100bp rate rise costs $7.7m instead of $16.8m; a 2022-style rise in rates plus a 100bp investment grade spread widening costs about $42.6m instead of $51.7m; the rate part of the capital proxy falls from $16.8m to $7.7m. Steepener -$3.5m vs -$7.7m.
- **Also observed:** with the band, the CVaR linear program returns exactly the same weights as the variance objective, because the portfolio sits at a corner fixed by five binding conditions (budget, return target, credit floor, gold cap, duration band).
- **Trade-offs:** The band width ($10m, 10% of surplus per 100bp) is a judgment, not optimized. Expected surplus change is unchanged at the $10.0m target.

### D-058: Public release
- **Date / Stage:** 2026-10-20 / Stage 3c housekeeping
- **Decision:** The development history was squashed into a single commit for the public repository; commits use the author's GitHub noreply address. The original 15-year-window Stage 1 report, which the audit script compares against, is kept as a file in `reports/archive/` instead of being read from an old commit.
- **Why:** keep the public repository to the engineering content.

### D-059: Walk-forward design
- **Date / Stage:** 2026-10-20 / Stage 3c
- **Decision:** Expanding window, rebalance every December, first rebalance **Dec 2010** (training Jun 2007 to Dec 2010, 43 months, all of 2008 included); evaluate each following 12 months on realized data through Sept 2026 (15 full years plus Jan to Sep 2026). At each date re-run the **exact headline process** with point-in-time data: PCA scenario generator, forward-curve anchor, measured durations, Bayes-Stein for equity-like assets, yield-based bond carry, the same insurer constraints, capital proxy and duration band; 5,000 scenarios and a 12-month horizon per decision; seed fixed per year.
- **Options considered:** first rebalance Dec 2009 (31 months, two-thirds crisis), Dec 2010, Dec 2011; Ledoit-Wolf covariance with Bayes-Stein means (the Stage 1 method) vs the Stage 3 scenario moments.
- **Why this one:** Dec 2010 balances "includes 2008" with enough data (about 4 months per asset) and still leaves 2013, 2020 and 2022 in the test. The prompt listed Ledoit-Wolf, but the headline optimizer uses scenario means and covariances, so testing Ledoit-Wolf would test a different method from the one recommended; this deviation is deliberate.
- **Trade-offs:** 16 decisions is a small sample. The scenario generator resamples only history up to each date, so early decisions are dominated by 2008.

### D-060: Point-in-time inputs and fixed settings
- **Date / Stage:** 2026-10-20 / Stage 3c
- **Decision:** All data truncated at the rebalance date (copied into fresh arrays, P-025). Credit spread levels: ICE OAS exists only from Oct 2023, so IG spread = Moody's Baa-10y at that date and HY spread = 2.2 x Baa-10y (assumption; today's ICE ratio is 2.1x). Fixed settings not re-estimated: curve pull kappa = 0.15 (calibrated on the full sample in Stage 2, a mild look-ahead), expected default losses, capital factors, haircuts.
- **Why this one:** Uses only information available at each date for everything that is estimated; assumptions stay assumptions. Baa unscaled overstates the IG index spread (1.47% vs 0.84% today), which flatters credit carry; it does not change the result because credit stays at its 40% floor in every decision of the robustness view.

### D-061: Realized evaluation (primary view, pre-specified)
- **Date / Stage:** 2026-10-20 / Stage 3c
- **Decision:** One synthetic block (PV 1,000 on the Dec 2010 curve, 30 years, -3% a year) runs off for real; assets start at 1,100, earn actual monthly ETF total returns, buy-and-hold within each year, pay each December's liability payment pro rata, and are rebalanced each December. Liabilities are revalued monthly on the actual historical Treasury curve. Rules scale with the book: surplus growth target 10% of current surplus, duration band 1% of liabilities per 100bp, capital cap half of current surplus. If the target is outside the feasible range the rule takes the min-variance or max-return portfolio; if infeasible, the key-rate matched portfolio.
- **Simplifications:** no transaction costs (turnover reported instead), no dividends or capital injections, ETFs priced at month-end, annual liability payments, Treasury discounting.
- **Finding:** surplus accumulates (no dividends), so the pre-set scaling rules degenerate in later years: from Dec 2021 the target exceeds the best feasible return (max-return corner) and in Dec 2025 the constraints become infeasible (P-024). Results: walk-forward total surplus change +$280.8m vs +$98.6m static in-sample, +$433.8m equal weight, +$7.3m key-rate matched; worst year 2022 for the optimized portfolio (-$59.8m) vs -$75.5m static; average turnover 18.5% a year vs 2% to 4%.

### D-062: Robustness view and risk calibration (added after seeing the primary view)
- **Date / Stage:** 2026-10-20 / Stage 3c
- **Decision:** Add a second view in which every December the balance sheet is reset to assets 1,100 against a fresh block worth 1,000 on that date's curve, so each year's decision is judged on identical terms; and compare each year's predicted 1-in-20 loss (VaR 95% from that date's scenarios) with the realized result. **Added after seeing the primary results**, to remove the balance-sheet drift artifacts (P-024), not to change performance; the primary view is reported unchanged and nothing was tuned.
- **Result (15 full years):** walk-forward optimized mean surplus change +$7.6m a year, volatility $23.2m, worst year 2022 (-$44.8m), 3 losing years; static in-sample headline (which has seen the future) +$11.6m, $23.9m, worst -$26.5m (2011), 2022 -$14.4m; equal weight +$34.5m, $76.4m, worst -$134.2m; key-rate matched -$1.0m, $8.0m, worst -$25.1m. **The optimizer does worse out of sample than in sample**, mainly in 2022: it sat at or within $0.2m of the long edge of the duration band (+$10m per 100bp) in all 16 decisions, and in 2022 TLT fell 31.2% and LQD 17.9% together.
- **Risk calibration:** realized losses beyond the predicted 1-in-20 threshold: 0 of 15 years (about 0.8 expected). Predicted volatility $31.8m to $50.9m vs $23.2m realized; predicted mean $10.7m vs $7.6m realized (standard error of a 15-year mean is about $6m). The risk model was conservative out of sample; the return expectation was somewhat optimistic but within noise.

### D-063: Liability discount-rate variant (Treasury plus an illiquidity premium)
- **Date / Stage:** 2026-10-20 / Stage 3c
- **Decision:** Add a variant discounting liabilities at the Treasury zero curve plus a share of the investment grade spread. Recommended assumption for the variant: **50% of the IG spread** (today 0.84% x 50% = 42bp), labeled as an assumption; sensitivity 0%, 25%, 50%, 75%, 100%. **The headline stays on Treasury discounting** (0%).
- **Options considered:** Treasury only (base); a fixed spread; a share of the IG spread; the full single-A or IG spread (as some accounting regimes use for long-duration insurance liabilities).
- **Why this one:** Annuitants cannot trade their contracts, so insurers argue the liabilities earn part of the extra yield of illiquid assets; half the IG spread is a middle-of-the-road illiquidity premium, and the range shows how much the answer depends on it. Keeping Treasury discounting as the headline avoids assuming a premium no regulator or accounting basis has granted in this model.
- **Results:** liability PV 1,000.0 / 980.3 / 961.2 / 942.7 / 924.7 and effective duration 9.48 / 9.37 / 9.26 / 9.15 / 9.04 at 0% / 25% / 50% / 75% / 100%; funding ratio with assets fixed at 1,100: 110.0% / 112.2% / 114.4% / 116.7% / 119.0%.

### D-064: The discount spread moves with simulated credit spreads
- **Date / Stage:** 2026-10-20 / Stage 3c
- **Decision:** In the scenarios the add-on is share x simulated IG spread. The IG path uses the SAME bootstrapped historical months as the asset returns (Baa-10y log changes, demeaned, pulled to the long-run median, scaled to today's ICE IG level). No extra random draws, so every other Stage 3 output is unchanged (verified: reports byte-identical).
- **Why this one:** The hedging question only exists if liability values react to credit spreads; using the same months keeps the real link between spread moves and LQD's returns. Correlation of LQD with the 1-year liability return: 0.52 (0%), 0.63 (50%), 0.74 (100%); TLT stays about 0.94 to 0.98.
- **Trade-offs:** One credit factor for the liability spread; proportional scaling from Baa to ICE IG (as in Stage 2).

### D-065: Re-optimizing under the variant
- **Date / Stage:** 2026-10-20 / Stage 3c
- **Decision:** For each variant, assets are set to 110% of that variant's liability value (so the risk budget is comparable) and the same rule is re-run: same insurer constraints, capital proxy, duration band (1% of liabilities per 100bp) and 10% surplus growth target, with the shared edge-case rule (`alm.solve_for_target`, also used by the walk-forward): if the target is below the lowest feasible expected return, take the min-variance portfolio. Two versions: with the 40% credit floor (headline rule) and without it (to see what the optimizer chooses on its own).
- **Results (credit not forced):** credit chosen freely 0% / 0% / 10.5% / 16.9% / 20.8%. Headline rule (floor binding at 40% in every variant): TLT 33.7% / 31.5% / 27.9% / 25.3% / 22.9%; surplus volatility $34.7m / $24.8m / $20.3m / $17.0m / $14.9m; capital $33.5m / $10.8m / $6.5m / $4.1m / $5.7m; IG spreads +100bp surplus change -$34.9m / -$11.1m / +$11.2m / +$32.2m / +$51.8m. From 50% upward the 10% target is below the lowest feasible return, so the rule takes the min-variance portfolio.
- **Answer to "does the recommended portfolio change?":** yes, materially: less TLT, more SHY, much lower surplus volatility and capital, and the credit floor turns from the main risk into a slight over-hedge. The 0% row reproduces the Stage 3b headline exactly ($34.67m, $71.53m, $33.52m).

### D-066: Frank-Wolfe gap replaces the trust-constr cross-check
- **Date / Stage:** 2026-10-20 / Stage 3c
- **Decision:** When SLSQP stops on its precision status, confirm optimality with the Frank-Wolfe gap: linearize the objective at the solution and let an LP (HiGHS) find the best feasible point for that linear objective; accept if it cannot beat the solution by more than 1e-6 relative. Replaces the trust-constr re-solve of D-013 / P-006.
- **Why this one:** It is an exact optimality test for convex problems (the first-order condition for max Sharpe), needs no curvature, and works at corner solutions where trust-constr's matrix goes singular (P-026). Verified: all earlier reports (Stages 1 to 3, walk-forward) regenerate byte-identical; on the Stage 1 frontier 17 of 56 precision stops are now confirmed this way.

---

## 4. Problems and failures log

### P-001: Homebrew Python broken
- **What broke:** Creating a virtual environment with Homebrew Python 3.12 and 3.11 failed. uv also refused to use them.
- **How detected:** `python3.12 -m venv` failed in `ensurepip`; verbose output showed an ImportError.
- **Root cause:** Homebrew's `pyexpat` module links to macOS's system `/usr/lib/libexpat.1.dylib`, which lacks a symbol (`_XML_SetAllocTrackerActivationThreshold`) the newer Python expects. pip's installer and macOS version detection (which reads a plist file using expat) both depend on it.
- **Fix:** Used the uv-managed CPython 3.12.14 already installed on the machine (D-001). No system changes made.
- **What I learned:** Read the real error before changing code. This was an environment problem, not a project problem.

### P-002: Long Treasuries look worse than cash (data reality, not a bug)
- **What broke:** Not a code failure. On the original 2011 window, raw historical means put TLT at 0.97% a year, below T-bills (BIL 1.51%). Even after shrinkage, SHY, IEF and TLT sit below the risk-free rate, so max Sharpe never holds Treasuries.
- **How detected:** Inspecting the estimated means against BIL after the first data run.
- **Root cause:** The window includes 2022, when rates rose sharply and long Treasuries lost about 30%. That wiped out years of returns. History does not reflect today's higher starting yields.
- **Fix:** Largely resolved by extending the window to 2007 (D-020): Treasury estimates are now above the risk-free rate (TLT 2.90% vs 1.38%). The root issue remains: bond estimates are still backward-looking and flip sign with two extra years of data. Planned fix: yield-based bond expected returns in Stage 2.
- **What I learned:** Backward-looking returns can tell the opposite story to forward-looking ones. Always sanity-check inputs against economic intuition before trusting optimizer outputs.

### P-003: Off-the-shelf Ledoit-Wolf distorted low-volatility assets
- **What broke:** sklearn's Ledoit-Wolf raised SHY volatility from 1.40% to 2.78% and TIP from 4.99% to 5.43% (2011 window; on the 2007 window SHY goes from 1.52% to 3.32%), while slightly lowering equity volatility.
- **How detected:** Side-by-side table of volatility under each estimator (now in `reports/stage1_results.md`).
- **Root cause:** sklearn shrinks toward a scaled identity matrix: every asset's variance is pulled toward the average variance. Fine when assets have similar volatility, wrong for a multi-asset universe where volatilities differ 12x to 15x.
- **Fix:** Apply Ledoit-Wolf to standardized returns (correlation matrix) and rescale by sample volatilities (D-009). Test V-008 confirms volatilities are preserved exactly.
- **What I learned:** Library defaults encode assumptions. Check that a method's assumptions fit the data, not just that it runs.

### P-004: Optimizer falsely reported failure (tolerance too tight)
- **What broke:** 2 of 4 portfolios and 14 of 15 frontier points reported `success=False`.
- **How detected:** The report prints the solver status for every output; I did not ignore it.
- **Root cause:** I set SLSQP's tolerance to 1e-15, below what double-precision arithmetic can resolve on an objective of about 2. The solver reached the answer and then reported it could not improve. With 1e-12 the same objective value (1.9691254199) was reached with success.
- **Fix:** Tolerance 1e-12, with the objective scaled by 1e4 so the tolerance is meaningful.
- **What I learned:** A solver's success flag is information. Too tight is as bad as too loose.

### P-005: Remaining precision stops at constraint corners
- **What broke:** After P-004, one frontier point (and 10 of 60 in a larger check) still stopped with SLSQP status 8. On the 2007 window, 11 of 29 target-return solves per problem hit it; all were confirmed.
- **How detected:** Solver status column in the report.
- **Root cause:** At points where many constraints bind at once, SLSQP's line search cannot make further progress in floating point, even though it is at the optimum. Verified independently: trust-constr's weights agreed to 8e-8, volatility to 1e-9, and the solution was feasible with the target met to 1e-11.
- **Fix:** Automatic cross-check: on status 8 with a feasible point, re-solve with trust-constr and accept only if it finds no better point (D-013).
- **What I learned:** Neither blindly trust nor blindly reject a solver flag; verify with an independent method.

### P-006: My verification rule rejected a correct answer
- **What broke:** After a refactor, the cross-check rejected one SLSQP result because trust-constr's objective differed by 1.9e-7 (55.55410704 vs 55.55411741), above my 1e-7 agreement threshold.
- **How detected:** A test failed (`test_every_output_sums_to_one_and_respects_constraints[real_long_only]`).
- **Root cause:** trust-constr had stopped slightly early, at a *worse* point. My rule demanded agreement, when the right question is "can the second solver find a better point?" A worse result from the checker does not disprove the answer.
- **Fix:** Accept if trust-constr's objective is not lower than SLSQP's by more than 1e-7 (relative).
- **What I learned:** Verification logic needs the same care as the thing it verifies.

### P-007: Test fixture produced a singular covariance
- **What broke:** Two tests failed: the PSD test on synthetic data (min eigenvalue -1.3e-18) and a finite-difference gradient test.
- **How detected:** pytest.
- **Root cause:** A bug in my test data generator, not the engine. I built a correlation matrix with `np.corrcoef` on a 5x5 matrix, which treats rows as variables with only 5 observations, giving a rank-4 (singular) matrix.
- **Fix:** Build the correlation from a positive definite matrix (A A' + nI) rescaled to unit diagonal.
- **What I learned:** The PSD check worked exactly as designed: it caught a singular matrix I did not know I had made.

### P-008: "Singular Jacobian" warnings from the cross-check solver (updated in Stage 1b: first diagnosis was incomplete)
- **What broke:** After extending the window, pytest showed 3 warnings and the pipeline printed `UserWarning: Singular Jacobian matrix` from trust-constr.
- **How detected:** Warning summary in pytest output and stderr of the pipeline run.
- **First diagnosis (Stage 1, incomplete):** The single-asset group "HYG at most 10%" duplicates the direction of HYG's own weight bound, giving linearly dependent rows. Fix at the time: suppress that exact warning inside the cross-check only.
- **Stage 1b cleanup:** Folded single-asset caps into weight bounds (D-031) and removed the suppression. Running pytest with warnings as errors then still failed on **real long-only**, a problem with no group constraints at all. So the HYG explanation could not be the whole story.
- **Real root cause:** Recording which solves warn showed only one: the frontier's max-return endpoint, a corner where the solution is 100% SPY. There, the budget row, the target-return row and ten "weight = 0" bounds over-determine the point (more active constraints than free dimensions), so the matrix trust-constr factorizes is singular.
- **Fix:** Use the exact LP solution for the max-return endpoint (D-031). Suppression removed. Verified: `pytest -W error` 78 passed; both pipelines run with `-W error` and produce no warnings; Stage 1 results unchanged (frontier volatilities within 1.4e-11).
- **What I learned:** A plausible root cause is not a confirmed one. Turning warnings into errors and recording exactly which call triggers them found what reasoning missed. Removing a suppression is a good test of whether you really understood the problem.

### P-009: Regime labels missing for the month after the data ends
- **What broke:** Four no-look-ahead tests failed: labels computed on a truncated sample raised "not enough history to label 1 month".
- **How detected:** `test_no_look_ahead_truncated_sample_gives_same_labels`.
- **Root cause:** To give month t the signal from month t-1, I shifted the month-end signal by one ROW. When the data ends at month X, there is no row for X+1, so the label for X+1 vanished, even though it only needs data up to the end of X. On the full sample every month had a row, so the bug was invisible there.
- **Fix:** Shift by one calendar month (`shift(1, freq="ME")`). All four truncation tests pass and full-sample labels are unchanged.
- **What I learned:** The truncation test was written to catch look-ahead; it also caught a "look-behind" bug that would have broken live use (labeling the current month).

### P-010: Chart labels collided
- **What broke:** In the first render, the VaR annotations overlapped each other and the lines, and the stress chart legend covered a bar; values like "-0" hid small numbers.
- **How detected:** Rendering the PNGs and looking at them.
- **Root cause:** Annotation positions chosen without checking the actual layout.
- **Fix:** VaR values moved into the legend, legend placed in empty space, stress legend moved above the plot, labels to one decimal.
- **What I learned:** Automated checks do not cover layout; look at every chart.

### P-011: A stress test with nothing to hit
- **What broke:** Not a code error. The credit-spread shock moved all three optimized portfolios by 0.0% to 0.02%.
- **How detected:** Reading the first hypothetical shock table.
- **Root cause:** The optimizer chose zero credit (LQD, HYG) for all three portfolios, so a credit shock cannot affect them. The test was valid but uninformative.
- **Fix:** Added an equal-weight benchmark (D-030); the credit shock costs it 2.0%. Logged as a finding: the optimized portfolios' immunity to credit shocks is a side effect of holding no credit, which an insurer could not do (credit is their core asset).
- **What I learned:** A stress test is only informative if the portfolio holds what the scenario hits.

### P-012: ICE OAS history on FRED is only three years
- **What broke:** The requested ICE BofA IG and HY OAS series download fine but start on 2023-10-20.
- **How detected:** Probed the FRED endpoint and printed each series' first date before building anything on it.
- **Root cause:** FRED limits licensed ICE data to a rolling three years.
- **Fix:** Did not fabricate or splice. Stopped and asked, with verified alternatives (Baa-10y has history to 1986; no free long HY spread exists). Decision: Baa-10y factor plus measured HY and IG sensitivities; ICE only for today's level (D-032).
- **What I learned:** Check that the data covers the window before designing around it.

### P-013: One-factor rates make long bonds far too calm
- **What broke:** Not a code bug; a model limitation, quantified. Simulated volatility: TLT 4.5% vs 14.1% historical; IEF 4.3% vs 6.7%; LQD 4.3% vs 8.3%; TIP 2.7% vs 5.8%. TLT and IEF have nearly equal simulated volatility despite durations of 17 and 7.5.
- **How detected:** The required comparison of factor-model returns against the historical bootstrap.
- **Root cause:** In one-factor Vasicek every zero yield moves by B(T)/T times the short-rate move. With a = 0.15, the 25-year yield moves only about 0.26 of the short rate's move, so long yields barely move. In reality long rates move a lot on their own (the 3M was stuck at zero for 7 years while the 10-year swung by 2 points).
- **Fix:** Not fixed in Stage 2 (one-factor was the agreed scope). Documented, and recommended a second, long-rate factor before Stage 3 (open question 10). Until then, use the historical bootstrap for asset risk and the factor model for rate-path and curve-shape questions.
- **What I learned:** Validation against the model's own formulas (martingale, moments) can pass perfectly while the model is still wrong about the world. Comparing to history is a separate, necessary check.
- **Stage 3 update (2026-10-20): FIXED for scenarios.** The empirical PCA generator (D-043) is now the default; TLT volatility 14.0% vs 14.1% history (Vasicek 4.5%), 10-year yield volatility 87bp vs 88bp (Vasicek 52bp). Vasicek is kept only as the analytic benchmark, now with the forward-looking anchor (D-044).

### P-014: Term premium sign convention
- **What broke:** The fitted market price of risk came out negative (-0.337) even though it implied a positive term premium, which would confuse any reader.
- **How detected:** Reading the calibration output: b_q (4.39%) above b (2.11%) with a negative lam.
- **Root cause:** I had defined b_q = b - lam sigma / a, under which a positive term premium needs a negative lam.
- **Fix:** Redefined b_q = b + lam sigma / a, so lam > 0 means a positive term premium (lam = +0.337). Docstrings and tests updated.
- **What I learned:** Pick sign conventions so the natural case is positive; it prevents misreading.

### P-015: Coupon-matched representative bonds were absurd
- **What broke:** Solving each fund's coupon to match its empirical duration gave a 41% coupon for TIP and 12.5% for SHY.
- **How detected:** Printing the representative bonds.
- **Root cause:** Matching a duration below a par bond's requires an unrealistically high coupon at a fixed maturity.
- **Fix:** Par bond at the representative maturity times a measured rate beta (D-037).
- **What I learned:** A mathematically valid answer can still fail the "would a practitioner believe this?" test.

### P-016: Stress-test approximation error was inflated by convention mismatches
- **What broke:** First version showed duration+convexity errors of 0.34pp (TLT +100bp) and 0.65pp (HY +400bp), larger than convexity truncation explains.
- **How detected:** Comparing those errors with the clean yield-space validation (0.13pp for a 30-year bond).
- **Root cause:** (1) The spread shock was added with continuous compounding while duration is in semiannual yield terms (400bp continuous is about 404bp semiannual). (2) The estimate used a single yield duration while the revaluation shifted the whole par curve.
- **Fix:** Spread shocks revalue at the shocked semiannual yield; curve shocks use key rate durations for the first-order term. Errors fell to 0.07pp (HY) and at most 0.22pp (TLT).
- **What I learned:** When an approximation error looks too big, check the conventions on both sides before blaming the approximation.

### P-017: Two test-design errors (and two chart layout fixes)
- **What broke:** (1) The KRD-sum test failed with a 1e-6 absolute tolerance; the actual gap was 1.7e-6 on 9.0 (2e-7 relative). (2) The calibration-mask test fed a trending line, which has no mean reversion, and the calibrator rightly refused it. (3) Chart labels overlapped lines.
- **How detected:** pytest; looking at the rendered charts.
- **Root cause:** (1) Central finite differences with a 1bp bump leave second-order cross terms around 1e-7 relative; my tolerance was tighter than the method's precision. (2) Wrong synthetic input. (3) Fixed label positions.
- **Fix:** (1) Relative tolerance 1e-5 with a comment explaining why. (2) Use a simulated Vasicek path. (3) Moved legends and labels.
- **What I learned:** A tolerance should come from the method's known precision, not from how small a number looks.

### P-018: FRED unreachable during Stage 3
- **What broke:** Every FRED request (including DGS10, which worked in Stage 2) failed with a connection reset right after the server responded.
- **How detected:** curl exit code 56 on three retries and two series.
- **Fix:** No fabrication. Stage 2 FRED data was already cached, so nothing broke; the forward-looking anchor was computed from the cached curve (D-044); the Fed long-run rate is an open question (16).
- **What I learned:** Cache everything you depend on; design so a missing optional input degrades gracefully.

### P-019: First PCA generator still made credit and TIPS too calm
- **What broke:** With only the curve and a spread formula, simulated volatility was HYG 5.9% vs 10.2%, LQD 6.9% vs 8.3%, TIP 4.0% vs 5.8%, and TIP's correlation with SPY was -0.08 vs +0.34.
- **How detected:** The volatility and correlation comparison against history.
- **Root cause:** Credit spread moves were scaled to today's very tight spread, and TIP's real-yield and inflation news was not modeled.
- **Fix:** Add each month's historical residual (actual ETF return minus the curve formula on the actual curve) to the simulated curve part. Now HYG 10.2%, LQD 8.2%, TIP 5.8%, correlations matched.
- **What I learned:** When you model only part of an asset's risk, put the unexplained part back from data rather than dropping it.

### P-020: Recommended portfolio is long duration, and my report text said otherwise
- **What broke:** The draft client report said the recommended portfolio "keeps the duration gap near zero". It does not: gap +$20.3m per 100bp.
- **How detected:** Checking the surplus shock table: the portfolio lost $16.8m for +100bp.
- **Root cause:** I wrote the narrative before reading the numbers. The optimizer deliberately ran long duration to hedge the credit floor (D-053).
- **Fix:** Rewrote the text from the numbers, explained why, added the duration-band variant.
- **What I learned:** Generate narrative from results, never from expectations.

### P-021: PDF rendering bugs
- **What broke:** In the PDF, pairs of dollar signs were treated as maths markup ("$72.9m to $33.1m" became italic run-together text); paragraphs wrapped raggedly; negative amounts showed as "$-16.8m"; the metrics table cut off row labels.
- **How detected:** Rendering each PDF page to an image and looking at it.
- **Fix:** Disabled maths parsing for text, reflowed paragraphs, put the sign before the dollar sign, rebuilt the table with names as a normal first column.
- **What I learned:** Look at the deliverable the client will see, not just the code that makes it.

### P-022: Two edits silently did not apply
- **What broke:** Twice, a scripted edit did nothing: once because the code formatter had re-wrapped the target text, once because a failed search stopped a command chain before the edit ran. The run still "succeeded".
- **How detected:** Re-reading the file after the edit, before trusting the result.
- **Fix:** Re-applied against the exact current text and re-ran the checks.
- **What I learned:** Verify that a change happened, not just that the command exited cleanly.

### P-023: A weak test assertion
- **What broke:** Nothing failed; all 142 tests passed first time. Reviewing them, one CVaR assertion had an "or" that made it almost always true.
- **Fix:** Replaced with the exact property: the LP's threshold lies between the two order statistics around the 95% quantile.
- **What I learned:** A first-time pass is a reason to read the tests again.

### P-024: Pre-set scaling rules degenerate as the runoff surplus grows
- **What broke:** In the primary walk-forward view, the target became unreachable from Dec 2021 (max-return corner chosen) and the problem was infeasible in Dec 2025 (fallback to key-rate matched).
- **How detected:** The decision status column.
- **Root cause:** With no dividends, surplus grew from $100m to $381m while liabilities ran off, so a 10%-of-surplus target outgrew the feasible return, and the required asset duration (liability dollar duration / assets) fell below what a 40% credit floor alone contributes; there is no cash asset to bring duration down.
- **Fix:** Not tuned. The primary view is reported as specified; a reset view (D-062) judges every year on identical terms.
- **What I learned:** A test design that lets the balance sheet drift tests the drift as much as the decisions.

### P-025: A no-look-ahead test failed on floating-point noise
- **What broke:** Scrambling data after the decision date changed weights of about 1e-12 (the main weights agreed to 9 digits).
- **How detected:** `test_future_data_cannot_move_a_decision`.
- **Root cause:** Editing the frames changed their memory layout, so the truncated slices summed in a different order; not information leakage (exact truncation tests passed at three dates).
- **Fix:** `decide` copies its truncated inputs into fresh arrays; the scrambled-future test now passes with exact equality. Results unchanged.
- **What I learned:** Prove a suspected false alarm instead of loosening the tolerance.

### P-026: The corner-solution warning came back under the discount variant
- **What broke:** `Singular Jacobian matrix` from trust-constr (the P-008 warning) while re-optimizing under one variant, failing the warnings-as-errors run.
- **Root cause:** Same family as P-008: at a corner fixed by many binding constraints (credit floor, duration band, caps), the cross-check solver's constraint matrix is singular. P-008 fixed one such corner; this showed the approach did not generalize.
- **Fix:** Replaced the cross-check with the Frank-Wolfe gap (D-066). No warnings anywhere; all earlier reports byte-identical.
- **What I learned:** When the same failure returns in a new place, fix the method, not the instance.

### P-027: A target below the feasible range at the 100% spread
- **What broke:** At the 100% discount spread the 10% surplus target was below the lowest expected return any feasible portfolio could have (5.48% vs 5.31% per $ of assets); SLSQP reported "target not met" and the Frank-Wolfe LP was infeasible (gap reported as infinite, so the result was correctly not confirmed).
- **Fix:** The walk-forward's edge-case rule (min-variance portfolio when the target is below the feasible range) moved into `alm.solve_for_target` and is now shared; walk-forward results unchanged.
- **What I learned:** Edge-case rules belong in one place, so every analysis applies the same rule.

---

## 5. Validation log

Current run: 2026-10-20, Stage 3c commit 2 (discount-rate variant). Full suite: **155 passed, 0 failed, 0 skipped, 0 warnings, run with `-W error`**. `ruff check`: all checks passed. All six pipelines (`run_stage1`, `run_stage1b`, `run_stage2`, `run_stage3`, `run_walkforward`, `run_discount_variant`) run with `-W error`; Stage 1 to 3 and walk-forward reports regenerate byte-identical after D-064 and D-066 (the client PDF changed once, from removing its timestamp, and is now deterministic). Earlier rows re-ran green. Pinned environment: Python 3.12.14, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, matplotlib 3.11.2.

The same 42 tests also all passed on the original 2011 window (archived as `reports/archive/stage1_results_2011_window.md`); values from that run are in the "2011 window" column where they differ.

| ID | Check | Test / source | Expected | Actual (2007 window) | 2011 window | Result |
|---|---|---|---|---|---|---|
| V-001 | Download returns all 12 tickers | `download_prices` run | 12 tickers | 12 tickers, 4,906 rows, 2007-04-02 to 2026-09-30; missing only before launch: HYG 6 days, BIL 40 days | 3,770 rows, 0 missing | PASS |
| V-002 | Cached CSV matches recorded SHA-256; 170+ monthly returns; no NaN | `test_cached_data_matches_metadata` | hash match, >=170 months | hash match (ff42f6f3...), 232 months, 0 NaN | hash 8d6ea5fa..., 179 months | PASS |
| V-003 | Trim to latest start date (synthetic) | `test_clean_trims_to_latest_start_date` | starts 2020-02-17 | as expected | same | PASS |
| V-003b | Trim to latest start date (real data) | `clean_prices` run | start = BIL launch | start 2007-05-30, 40 rows dropped, 0 gaps inside window | no trim needed | PASS |
| V-004 | Short gap filled, long gap refused | `test_clean_fills_short_gap_but_rejects_long_gap` | 3-day gap filled; 20-day gap raises | as expected | same | PASS |
| V-005 | Partial final month dropped; first month-end is base only | `test_monthly_returns_drop_partial_final_month` | last 2020-05-31, first 2020-02-29 | as expected | same | PASS |
| V-006 | Covariance symmetric and PSD (real data, 3 estimators) | `test_covariance_symmetric_and_psd_real` | asymmetry 0, min eigenvalue > 0 | asymmetry 0.0 for all; min eigenvalue: sample 6.04e-5, raw LW 9.40e-4, LW-corr 9.45e-5 | 4.16e-5, 6.24e-4, 6.85e-5 | PASS |
| V-007 | Covariance PSD (synthetic); bad matrices rejected | `test_covariance_symmetric_and_psd_synthetic`, `test_check_covariance_rejects_bad_matrices` | PSD passes; [[1,2],[2,1]] and an asymmetric matrix raise | as expected | same | PASS |
| V-008 | LW-corr keeps sample volatilities, shrinks correlations | `test_lw_corr_keeps_sample_volatilities` | diagonal equals sample (rel 1e-12); every abs correlation <= sample | as expected; intensity 0.0544 | 0.0472 | PASS |
| V-009 | Annualization x12 | `test_annualization` | exact | exact | same | PASS |
| V-010 | Bayes-Stein in [0,1], between history and target, reduces spread | `test_bayes_stein_shrinks_between_history_and_target` | 0 <= phi <= 1 | phi 0.3353, target 1.59% | phi 0.3315, target 1.17% | PASS |
| V-011 | Weights sum to 1 and respect bounds and groups in every output | `test_every_output_sums_to_one_and_respects_constraints` | abs(sum-1) < 1e-6; zero violations | 32 outputs per real problem: max abs(sum-1) 8.1e-10 (long-only), 1.9e-10 (constrained); max bound violation 0 | 3.0e-10, 1.3e-10 | PASS |
| V-012 | Group constraints hold | `test_group_constraint_binds_when_it_should` | HYG <= 10%, growth <= 40% | HYG 0.0%, growth 30.9% | HYG 10.0%, growth 35.0% | PASS |
| V-013 | Two uncorrelated assets: min variance matches closed form | `test_min_variance_two_uncorrelated_assets_closed_form` | w_A = 0.8; variance 0.008 | 0.8000000000; 0.008 | same | PASS |
| V-014 | Max Sharpe beats every point of a 30-point frontier | `test_max_sharpe_beats_every_frontier_point` | max Sharpe >= frontier best - 1e-6 | long-only 0.56707 vs 0.56705; constrained 0.55808 vs 0.55800 | 0.65454 vs 0.65454; 0.58089 vs 0.58049 | PASS |
| V-015 | Efficient frontier monotonic | `test_frontier_monotonic` | return strictly up, volatility never down | long-only: min step +0.22% return, +0.09% vol; constrained: +0.14%, +0.02% | +0.31%/+0.08%; +0.16%/+0.01% | PASS |
| V-016 | Zero liabilities: surplus moments equal asset moments | `test_surplus_moments_reduce_to_asset_moments_with_zero_liabilities` | equal to 1e-15 | equal | same | PASS |
| V-017 | Zero-liability problem equals asset-only, even with nonzero mu_L, sigma_L, c | `test_zero_liability_equals_asset_only` | weights identical to 1e-10 | max weight difference 0.0 (min var and max Sharpe) | 0.0 | PASS |
| V-018 | Shorting allowed: min variance matches closed form | `test_min_variance_matches_closed_form_when_shorting_allowed` | max weight diff < 1e-6 | synthetic pass; real spot check 4.8e-9 | 1.07e-8 | PASS |
| V-019 | Risk contributions sum to total volatility | `test_contributions_sum_to_volatility_*`, `test_equal_risk_for_identical_uncorrelated_assets` | sum = vol | constrained max Sharpe 6.2805% vs 6.2805% (diff 2.8e-17); long-only 3.4198% vs 3.4198% (diff 1.4e-17) | 6.3298%, 12.0721% | PASS |
| V-020 | Marginal contribution matches finite difference | `test_marginal_contribution_matches_finite_difference` | rel error < 1e-5 | pass, all 5 assets | same | PASS |
| V-021 | Liability plug-in: liability identical to one asset, f = 1, is fully hedged | `test_liability_replicated_by_one_asset_is_fully_hedged`; real spot check with liability = TLT | 100% of that asset, surplus vol about 0 | synthetic pass. Real: asset-only min var = 98.1% SHY (asset vol 1.49%); surplus min var = 100% TLT, surplus vol 0.0. Asset-only portfolio held against this liability: **13.38% surplus vol**, liability causes 105% of it | 99.6% SHY; 12.27% surplus vol | PASS |
| V-022 | Liability shows as -f; contributions sum to surplus vol | `test_liability_appears_as_short_position_and_sum_is_surplus_vol` | weight -0.8; sum = surplus vol | as expected | same | PASS |
| V-023 | Infeasible constraints raise | `test_infeasible_constraints_raise` | ValueError | raised | same | PASS |
| V-024 | Max Sharpe refuses when nothing beats cash | `test_max_sharpe_undefined_when_nothing_beats_cash` | ValueError | raised | same | PASS |
| V-025 | All solves on real data report success | `python -m engine.run_stage1`, plus 30-point frontier check | all succeed | 4 portfolios + 15 frontier points: 19 of 19. 30-point check: per problem 18 direct, 11 precision stops confirmed by trust-constr, 0 failures | 19 of 19; 50 direct + 10 confirmed of 60 | PASS |
| V-026 | New download agrees with old one on the overlapping period | compare 2011 cache vs 2007 cache, 2011-10-03 to 2026-09-30 | tiny differences only | max daily return difference 2.7e-6; max relative price difference 2.3e-6 (Yahoo re-adjustment and rounding) | n/a | PASS |
| V-027 | 2008 crisis is in the sample | worst monthly return per asset | Oct 2008 among worst months | worst month is 2008-10 for 7 of 11 assets (VNQ -31.7%, EEM -25.6%, EFA -20.8%, SPY -16.5%, GLD -16.1%, HYG -11.5%, TIP -8.1%); LQD 2008-09 (-10.7%); TLT 2009-01; SHY and IEF in 2022 | n/a | PASS |
| V-028 | Single-asset caps folded into bounds; no solver warnings | `pytest -W error`; `python -W error -m engine.run_stage1` and `run_stage1b` | 0 warnings; Stage 1 results unchanged | 0 warnings in all three; frontier weights changed at most 1.9e-7, volatility 1.4e-11 | n/a | PASS |
| V-029 | Seeded Monte Carlo runs reproducible (normal, t, bootstrap) | `test_seeded_runs_are_reproducible` | same seed identical, different seed differs | identical arrays; different seeds differ | n/a | PASS |
| V-030 | Simulated mean and covariance converge to inputs (normal) | `test_simulated_moments_converge_to_inputs[normal]` | error shrinks with paths; mean within 5 SE; cov rel. error < 2% | cov rel. error 3.54% (500 paths) to 0.14% (100,000); max mean error 1.11 SE | n/a | PASS |
| V-031 | Same convergence for Student-t | `test_simulated_moments_converge_to_inputs[student_t]` | cov rel. error < 5% | 5.87% to 0.27%; max mean error 1.61 SE | n/a | PASS |
| V-032 | Bootstrap mean shifted to expected returns | `test_bootstrap_mean_is_shifted_to_expected_returns` | within 2e-4 per month | max difference 1.1e-4 | n/a | PASS |
| V-033 | VaR <= CVaR, both increase with confidence (all 3 methods, 1y and 2y) | `test_var_below_cvar_and_monotone_in_confidence`, `test_var_cvar_known_values` | VaR95 <= CVaR95, VaR99 <= CVaR99, VaR95 <= VaR99, CVaR95 <= CVaR99; known array gives VaR 0.9405 | all hold; known values exact. Real data also consistent (e.g. constrained max Sharpe bootstrap 1y: VaR95 6.4 <= CVaR95 9.5 <= ... CVaR99 13.7) | n/a | PASS |
| V-034 | Zero-volatility asset gives a deterministic path (all 3 methods) | `test_zero_volatility_asset_gives_deterministic_path` | wealth = (1 + 0.03/12)^t to 1e-12; drawdown 0 | max deviation 4.4e-16 for each method | n/a | PASS |
| V-035 | Surplus Monte Carlo: zero liabilities = none; liability identical to held asset keeps surplus flat; bootstrap without liability history refuses | `test_zero_liability_equals_no_liability`, `test_liability_identical_to_asset_leaves_surplus_flat`, `test_bootstrap_with_liabilities_needs_history` | identical; surplus about 0; ValueError | identical; max abs surplus 2.0e-15 while asset wealth std is 0.119; error raised | n/a | PASS |
| V-036 | Max drawdown on a known path; normal understates equity tail vs bootstrap | `test_max_drawdown_on_known_path`, `test_normal_understates_tail_vs_bootstrap_for_equity` | 0.5; bootstrap CVaR99 > normal | 0.5; SPY 1y CVaR99 bootstrap > normal | n/a | PASS |
| V-037 | Historical scenario return matches manual compounding for one asset (SPY, all 4 windows) | `test_one_asset_scenario_return_matches_manual_compounding` | daily path = product of monthly returns = price ratio, to 1e-12 | GFC -50.7848%, taper +3.0285%, COVID -19.4152%, 2022 -18.1754%; difference 0.0 in all four | n/a | PASS |
| V-038 | Daily path month ends equal the monthly-rebalanced portfolio | `test_daily_path_month_ends_equal_monthly_rebalanced_portfolio` (synthetic) + real spot check | match to 1e-12 | synthetic pass; real 3-asset GFC check max difference 3.3e-16 | n/a | PASS |
| V-039 | Shock arithmetic and shock table sanity | `test_apply_shock_is_weighted_sum_and_handles_liabilities`, `test_hypothetical_shocks_are_downside_and_use_known_tickers`, `test_max_drawdown_simple` | weighted sums exact; all shocks negative; known tickers | as expected | n/a | PASS |
| V-040 | Regime labels cover every month and are mutually exclusive | `test_labels_cover_every_month_and_are_exclusive`, `test_regime_summary_months_add_up` | 232 months, no gaps, one label each; regime months sum to 232 | rates 103 rising + 129 falling; volatility 52 high + 180 low | n/a | PASS |
| V-041 | No look-ahead: truncated sample and scrambled future give identical past labels | `test_no_look_ahead_truncated_sample_gives_same_labels` (cutoffs 2008-06, 2013-03, 2020-01, 2022-06), `test_no_look_ahead_scrambling_future_does_not_move_past`, `test_label_appears_the_month_after_the_signal` | identical labels; a March jump shows in April | identical for 14, 71, 153, 182 months; scrambled future leaves past unchanged; label appears next month | n/a | PASS (after P-009 fix) |
| V-042 | 2022 is not a falling-rate regime; 2008 is high volatility | `test_2022_is_never_a_falling_rate_regime`, `test_2008_crisis_is_high_volatility` | 2022: 0 falling months; Sep 2008 to Jun 2009 all high; at least 2/3 of 2008 high | 2022: 12 of 12 rising; Sep 2008 to Jun 2009 all high; 2008: 8 of 12 high | n/a | PASS |
| V-043 | Signal data consistent with price panel | spot check, SPY in signals file vs SPY in price file, same dates | tiny differences only | max relative difference 9.3e-7 | n/a | PASS |
| V-044 | FRED download and month-end alignment | `load_fred`, `monthly_fred` | 11 series; no month-end gaps in full-history series | 11 series, 5,424 daily rows 2006-01-02 to 2026-09-30; 233 month ends May 2007 to Sep 2026 with no gaps in Treasuries and Baa; ICE OAS from 2023-10-20 (36 months), used only for today's level | n/a | PASS (ICE limit flagged, P-012) |
| V-045 | Zero-coupon price vs closed form; Macaulay of a zero = maturity | `test_zero_coupon_price_matches_closed_form` | 100/(1+y/2)^(2T) to 1e-14; 100 exp(-zT) on the curve to 1e-12 | exact to tolerance | n/a | PASS |
| V-046 | Flat par curve gives a flat zero curve | `test_flat_par_curve_gives_flat_zero_curve` | 5.00% everywhere to 1e-12 | as expected | n/a | PASS |
| V-047 | Par bonds reprice to 100 off the bootstrapped curve | `test_par_bonds_reprice_to_100` (upward and inverted curves, 12 maturities incl. off-grid), `test_par_curve_repricing_on_real_data` (2007, 2011, 2017, 2026 curves) | 100 to 1e-9 | synthetic pass; real worst error 2.8e-14; today's curve report: 100.000 at 2, 5, 10, 30 years | n/a | PASS |
| V-048 | Discount factors positive and decreasing; YTM round trip | `test_discount_factors_positive_and_decreasing`, `test_ytm_round_trip` | DF > 0, strictly falling to 35 years; yield recovered to 1e-12 | as expected | n/a | PASS |
| V-049 | Duration vs finite difference; convexity vs second difference; DV01 | `test_duration_and_convexity_match_finite_differences` (4 bonds incl. a zero) | rel 1e-7 (duration), 1e-5 (convexity), 1e-4 (DV01) | pass; today's curve: 30y modified 14.3880 vs finite difference 14.3881, convexity 315.168 vs 315.168 | n/a | PASS |
| V-050 | Key rate durations sum to effective duration | `test_key_rate_durations_sum_to_effective_duration` (upward and inverted, 3 maturities) | relative 1e-5 | relative gaps -3.7e-8 (5y), -1.9e-7 (12y), -6.4e-7 (25y); today's curve 2/5/10/30y equal to 5 decimals (e.g. 30y 14.47940 vs 14.47940) | n/a | PASS (after P-017 tolerance fix) |
| V-051 | Duration+convexity vs full revaluation for +/-100bp | `test_duration_convexity_vs_full_revaluation_100bp` | convexity reduces error; under 0.01/0.02/0.2pp for 2/10/30y | today's curve, 30y: duration-only error -1.450/-1.721pp, D+C +0.126/-0.145pp; 10y D+C +0.012/-0.013pp; 2y +0.0002/-0.0002pp | n/a | PASS |
| V-052 | Simulated short-rate mean and variance match Vasicek formulas | `test_simulated_mean_and_variance_match_analytic` (1, 5, 10y, 100,000 paths) | within 4 standard errors | mean +1.02, +1.82, +0.71 SE; variance +0.17, +0.15, -0.42 SE. Real-data parameters (report): 1y mean 4.232% vs 4.226%, 10y variance 3.29e-4 vs 3.28e-4 | n/a | PASS |
| V-053 | Martingale: mean of exp(-integral of r) equals closed-form zero price | `test_martingale_discounted_bond_matches_closed_form` (1, 5, 10y) + report with real parameters | within 4 SE | synthetic +0.73, +0.36, -0.00 SE; real parameters 0.958734 vs 0.958755 (-1.2 SE), 0.809257 vs 0.809341 (-0.65 SE), 0.655196 vs 0.655290 (-0.40 SE) | n/a | PASS |
| V-054 | Pricing measure, limits, market price of risk | `test_pricing_measure_uses_risk_premium`, `test_zero_price_limits`, `test_market_price_of_risk_recovers_target` | b_q formula; P(0)=1; long yield = b_q - sigma^2/2a^2; lam recovered to 1e-10 | as expected | n/a | PASS |
| V-055 | Seeded runs reproducible (rates and full factor simulation) | `test_seeded_simulation_is_reproducible`, `test_factor_simulation_is_reproducible` | identical for same seed, different otherwise | as expected | n/a | PASS |
| V-056 | Calibration recovers known parameters; mask drops pairs across gaps | `test_calibration_recovers_known_parameters`, `test_calibration_mask_skips_pairs_across_gaps` | within 3 SE; 38 pairs | a 0.532 vs 0.5 (se 0.047), b 0.0304 vs 0.03 (se 0.001), sigma 0.0121 vs 0.012 (se 0.0001); 38 pairs | n/a | PASS (after P-017 input fix) |
| V-057 | Simulated shock correlations match the input matrix | `test_correlated_shocks_match_input_matrix`, `test_simulated_factor_correlations_and_nonnegative_spreads` | within 0.01 | synthetic max difference 0.0018; real matrix 0.0014 (e.g. spread-equity -0.580 vs -0.582) | n/a | PASS |
| V-058 | Spreads stay non-negative | same, plus `test_spreads_stay_positive_under_extreme_volatility` | all > 0 | minimum 0.51% (seed 4), 0.57% (report seed); with volatility 3.0: minimum 1.4e-7, still > 0 | n/a | PASS |
| V-059 | Simulation starts on today's curve | `test_simulated_curve_starts_on_todays_curve` | to 1e-12 | as expected | n/a | PASS |
| V-060 | Yield-based expected return = starting yield - default loss | `test_yield_based_expected_return_is_yield_minus_default_loss` | to 1e-15 | exact (e.g. LQD 5.37% + 0.84% - 0.20% = 6.01%; HYG 5.02% + 3.12% - 2.50% = 5.64%) | n/a | PASS |
| V-061 | Empirical durations: Treasuries fit, HYG flagged; spread factor helps credit | `test_empirical_durations_fit_treasuries_and_flag_hyg` | Treasury R2 >= 0.9 and within 15% of published approx; HYG rates-only R2 < 0.1 | SHY 1.80 (R2 0.92), IEF 7.53 (0.99), TLT 16.96 (0.98); HYG 0.00; credit R2 with Baa: LQD 0.40 to 0.73, HYG 0.00 to 0.48 | n/a | PASS |
| V-062 | Duration used = empirical where specified | `test_rate_beta_makes_used_duration_equal_empirical` | exact for SHY, IEF, TLT, TIP; beta 1 for credit | as expected | n/a | PASS |
| V-063 | Zero-bound months | `test_zero_bound_mask_counts` | 84 + 23 = 107 | 107 of 233 | n/a | PASS |
| V-064 | Duration-based stress: estimate close to full revaluation; signs; spread shocks only hit credit | `test_duration_stress_estimates_close_to_full_revaluation` | all errors under 0.25pp | max 0.22pp (TLT, -100bp); every bond fund negative at +100bp; spread rows only LQD and HYG | n/a | PASS |
| V-065 | PCA is a valid decomposition | `test_pca_is_a_valid_decomposition` | orthonormal loadings; all components rebuild history; first 3 explain > 95% | orthonormality error 1.1e-15; rebuild error 6.9e-18; 3 factors 98.15% | n/a | PASS |
| V-066 | Factor shapes are level, slope, curvature | `test_pca_factor_shapes_are_level_slope_curvature` | level all positive; slope long > short; curvature wings vs belly | as expected (level loadings 0.17 to 0.43) | n/a | PASS |
| V-067 | PCA generator matches historical bond volatility | `test_pca_generator_matches_historical_bond_volatility` | IEF, TLT, 10y yield vol within 10% of history | IEF 6.67% vs 6.72%; TLT 14.03% vs 14.13%; 10y 87.6bp vs 88.0bp | n/a | PASS |
| V-068 | Vasicek benchmark understates TLT volatility (documents P-013) | `test_vasicek_benchmark_understates_tlt_vol` | < 50% of history | 4.46% vs 14.13% | n/a | PASS |
| V-069 | Seeded generator reproducible | `test_seeded_generator_is_reproducible`, `test_stage3_context_is_reproducible` | identical with same seed, different otherwise; full Stage 3 context identical | as expected | n/a | PASS |
| V-070 | Curve pulled toward the anchor at the stated speed | `test_curve_is_pulled_toward_the_anchor` | with shocks off, y_60 = anchor + (y0 - anchor)(1 - kappa/12)^60 | max error 6.9e-18 | n/a | PASS |
| V-071 | Forward curve of a flat curve is flat; equity-like means adjusted | `test_forward_curve_of_a_flat_curve_is_flat`, `test_equity_like_scenarios_are_mean_adjusted` | 5% to 1e-12; mean within 4 SE of Bayes-Stein | flat exact; mean errors -0.04 to 0.50 SE | n/a | PASS |
| V-072 | Liability PV vs closed form on a flat curve | `test_pv_matches_closed_form_on_flat_curve` (2%, 5%, 8%) | 1000 to 1e-12 relative | curve 1000.0000000000, closed form 999.99999999999 to 1000.0000000000015 | n/a | PASS |
| V-073 | Liability duration vs finite-difference bump | `test_duration_matches_finite_difference_bump` | modified rel 1e-7; +1bp curve bump changes PV by -DV01 (rel 2e-3) | 9.4717067 vs 9.4717068; -0.9475 vs DV01 0.9482 | n/a | PASS |
| V-074 | Liability key rate durations sum to total | `test_key_rate_durations_sum_to_total` | rel 1e-5 | 9.482143 vs 9.482147 (rel -4.4e-7) | n/a | PASS |
| V-075 | Assets and liabilities discounted on the same curve | `test_assets_and_liabilities_use_the_same_curve`, `test_value_after_one_year_and_payments` | scenario PV at t=0 equals PV on the asset curve | max difference 2.3e-13; year-1 value matches manual sum to 1e-12 | n/a | PASS |
| V-076 | Historical liability return on a flat unchanging curve = accretion | `test_historical_return_on_a_flat_unchanging_curve_is_accretion` | (1 + y/2)^(2/12) - 1 | exact to 1e-12 | n/a | PASS |
| V-077 | Surplus identity in every scenario; 1-year identity used by the optimizer | `test_surplus_identity_in_every_scenario` | S = A - L exactly; path surplus change = A0 w'R - L0 r_L | exact; max error 1.05e-12 | n/a | PASS |
| V-078 | Zero liabilities = asset-only (Stage 3 inputs) | `test_zero_liability_equals_asset_only` | identical weights | max difference below 1e-10 (min variance), 1e-8 (target) | n/a | PASS |
| V-079 | Matched portfolio: near-zero surplus change for small parallel shifts | `test_matched_portfolio_has_near_zero_surplus_change_for_small_shifts` | below $1m for +/-10bp and below 10% of a 100% SHY portfolio | -$0.39m / +$0.43m vs +$7.43m / -$7.57m | n/a | PASS |
| V-080 | Constraints satisfied in every output | `test_constraints_hold_in_every_output` (recommended, min variance, CVaR, 6 frontier points) | IG >= 40%, equity <= 10%, liquidity >= 1, capital <= $50m, all bounds | all satisfied | n/a | PASS |
| V-081 | Capital charge monotonic in risk | `test_capital_charge_is_monotonic_in_risk` | +10% SPY adds A0 x 10% x 30%; larger duration gap gives larger rate charge; frontier capital non-decreasing | +$33.0m exactly; SHY rate charge > matched; frontier $12.0m to $50.0m increasing | n/a | PASS |
| V-082 | CVaR LP equals scenario CVaR; threshold equals VaR | `test_cvar_lp_matches_scenario_cvar` | rel 1e-6; alpha between order statistics | 0.0631183258 vs 0.0631183258; alpha 0.047131 in [0.047131, 0.047707] | n/a | PASS (after P-023 fix) |
| V-083 | Optimizer cross-checked by a second method | `test_optimizer_cross_checked_by_trust_constr` | SLSQP variance not worse than trust-constr; weights within 1e-3 | variance 1.0053356404e-3 vs 1.0053356419e-3; max weight difference 4.3e-9 | n/a | PASS |
| V-084 | Liquidity coverage definition | `test_liquidity_coverage_definition` | outflow = payments + 10% L0; LCR matches manual for LQD | outflow $193.3m; manual match; SPY move -30%, TLT move below -10% | n/a | PASS |
| V-085 | Duration band rule binds | `test_duration_band_rule_binds` | abs(gap) <= $10m per 100bp | gap $10.000m | n/a | PASS |
| V-086 | Banded portfolio as headline; all reports regenerated | `python -W error -m engine.run_stage1`, `run_stage1b`, `run_stage2`, `run_stage3` | no warnings; Stage 1, 1b, 2 outputs unchanged; banded gap at the +$10m bound | all four ran clean; Stage 1, 1b and 2 report files byte-identical (no git diff); recommended gap $10.000m per 100bp; CVaR variant weights identical to the variance solution | n/a | PASS |
| V-087 | Every audited number quoted in the documentation matches the regenerated reports, and the quoted text is present where it is used | `scripts/audit_decisions.py` (recomputes each number from reports/ and the archived 15-year report; checks the quoting text in DECISIONS.md and README.md, and in local PRIVATE_PREP.md when present) | all checks pass | 41 of 41 checks pass; 62 quoted texts checked and present | n/a | PASS |
| V-088 | No look-ahead: truncated data gives identical decisions | `test_no_look_ahead_truncated_data_gives_identical_weights` (Dec 2010, 2016, 2021) | weights and matched weights bit-identical | identical | n/a | PASS |
| V-089 | Scrambled future cannot move a decision | `test_future_data_cannot_move_a_decision` (Dec 2013) | bit-identical | identical after the P-025 fix (before: 1e-12 differences) | n/a | PASS |
| V-090 | Constraints hold at every walk-forward rebalance | `test_constraints_hold_at_every_rebalance` (16 dates, fresh balance sheet, 1,000 scenarios) | no violations; LQD >= 40%, equity <= 10%, HY <= 5%, gap within band, capital <= cap, liquidity >= 1 | all 16 satisfied | n/a | PASS |
| V-091 | Walk-forward reproducible; accounting identities | `test_walk_forward_decisions_are_reproducible`, `test_realized_accounting`, `test_remaining_block` | same seed same weights; runoff starts at 1,100 / 1,000; single-asset turnover 0; reset years start at surplus 100 | as expected | n/a | PASS |
| V-092 | Walk-forward report runs clean; decisions feasible in the robustness view | `python -W error -m engine.run_walkforward` | no warnings; reset-view statuses all "target met" or min-variance | clean; 16 of 16 feasible; primary view has 4 max-return corners and 1 infeasible fallback (P-024) | n/a | PASS |
| V-093 | Out-of-sample risk calibration | `reports/walkforward_calibration.csv` | realized losses beyond predicted VaR 95% in about 5% of years | 0 of 15 | n/a | PASS (risk not understated; returns below prediction, within noise) |
| V-094 | Zero discount spread equals the base case | `test_zero_spread_equals_base_case` | measures and scenario liability paths identical to Treasury discounting and to Stage 3 | identical (exact) | n/a | PASS |
| V-095 | Liability PV and duration fall as the spread rises | `test_pv_and_duration_fall_as_the_spread_rises` (0 to 84bp) | strictly decreasing | PV 1,000.0 to 924.7; duration 9.48 to 9.04 (full report) | n/a | PASS |
| V-096 | Same-curve consistency | `test_same_curve_consistency` | scenario value at t=0 = analytic value on the asset Treasury curve plus the same spread; IG path starts at ICE IG and stays positive | equal to 1e-12 for 25%, 50%, 100% | n/a | PASS |
| V-097 | Key rate durations sum to total with a spread; spread DV01 close to curve DV01 | `test_key_rate_durations_sum_to_total_with_a_spread` | rel 1e-5; within 2% | as expected | n/a | PASS |
| V-098 | Variant optimization obeys constraints | `test_variant_optimization_obeys_constraints` (50%) | rule status "target..."; LQD >= 40%, equity <= 10%, capital <= cap | as expected | n/a | PASS |
| V-099 | Frank-Wolfe confirmation leaves every earlier result unchanged | regenerate Stages 1 to 3 and walk-forward with `-W error`; `git status` | no report differences | none (client PDF only, from the deliberate timestamp removal) | n/a | PASS |

---

## 6. Stage 1 results (from `reports/stage1_results.md`)

Window 2007-05-30 to 2026-09-30, 232 months. Annualized. Expected returns are Bayes-Stein. Risk-free 1.38%. Illustrative constraints: each asset at most 35%, HYG at most 10%, growth assets at most 40%.

| Portfolio | Exp. return | Volatility | Sharpe | Main holdings |
|---|---|---|---|---|
| Long-only min variance | 1.85% | 1.49% | 0.31 | 98.1% SHY, 1.8% SPY |
| Long-only max Sharpe | 3.32% | 3.42% | 0.57 | 74.4% SHY, 16.8% SPY, 8.9% GLD |
| Constrained min variance | 2.57% | 3.87% | 0.31 | 35% SHY, 34.6% TIP, 20.2% IEF, 8.8% HYG |
| Constrained max Sharpe | 4.89% | 6.28% | 0.56 | 35% SHY, 30.9% SPY, 17.6% GLD, 16.6% IEF |

Risk contribution, constrained max Sharpe: SPY 30.9% of weight but **59.7% of risk**; GLD 17.6% of weight, 31.5% of risk; IEF 6.3%; SHY 2.5%.

**What changed when 2007 to 2011 was added (same code, same constraints):**

| Measure | 2011 to 2026 (179 months) | 2007 to 2026 (232 months) |
|---|---|---|
| Risk-free (BIL mean) | 1.51% | 1.38% |
| SPY expected return (Bayes-Stein) | 10.34% | 8.10% |
| TLT expected return (Bayes-Stein) | 1.04% (below cash) | 2.90% (above cash) |
| HYG volatility | 7.02% | 10.21% |
| VNQ volatility | 16.68% | 22.25% |
| Long-only max Sharpe | 82.8% SPY, 17.2% GLD; return 9.41%, vol 12.07% | 74.4% SHY, 16.8% SPY, 8.9% GLD; return 3.32%, vol 3.42% |
| Constrained max Sharpe | 35% SPY, 35% SHY, 14.2% GLD, 10% HYG, 5.8% IEF | 35% SHY, 30.9% SPY, 17.6% GLD, 16.6% IEF, 0% HYG |
| Surplus vol of asset-only min var vs TLT-like liability | 12.27% | 13.38% |

How to read these: unconstrained MVO is extreme and unstable. The long-only max Sharpe portfolio swung from mostly stocks to mostly short Treasuries just by adding two years of data. Max Sharpe only cares about return per unit of risk, not the level of return, so once bonds beat cash it happily picks a low-risk, low-return mix. This is the known weakness of MVO, and the reason for shrinkage, constraints and, later, liabilities and capital. The constrained portfolio moved much less, which is part of why practitioners use constraints. None of these are recommendations.

## 6b. Stage 1b results (from `reports/stage1b_results.md`)

All in-sample: portfolio weights come from Stage 1 estimates on the same 2007 to 2026 data. Portfolios: min variance (98.1% SHY), max Sharpe (74.4% SHY, 16.8% SPY, 8.9% GLD), constrained max Sharpe (35% SHY, 30.9% SPY, 17.6% GLD, 16.6% IEF), and the equal-weight benchmark.

**Monte Carlo, 1-year horizon (10,000 paths, losses in %):**

| Portfolio | 99% VaR normal | 99% VaR bootstrap | 99% CVaR normal | 99% CVaR t | 99% CVaR bootstrap | Normal understates CVaR by | Worst actual 12 months |
|---|---|---|---|---|---|---|---|
| Min variance | 1.6 | 3.0 | 2.2 | 2.4 | 3.6 | 38.2% | -5.3% |
| Max Sharpe | 4.7 | 5.8 | 5.8 | 6.5 | 7.0 | 18.0% | -6.7% |
| Constrained max Sharpe | 9.5 | 11.4 | 11.5 | 12.7 | 13.7 | 16.5% | -13.5% |
| Equal weight (benchmark) | 15.2 | 20.3 | 18.0 | 19.4 | 23.6 | 23.7% | -22.5% |

5-year horizon, bootstrap: probability of any loss is 1.6% (min variance), 2.2% (max Sharpe), 4.9% (constrained), 15.1% (equal weight). Probability of missing 4% a year: 99%, 66%, 37%, 47%. 99th percentile max drawdown: 5.8%, 11.4%, 22.7%, 37.8%. Median terminal wealth per $1: 1.10, 1.18, 1.27, 1.24.

**Historical stress (daily marks):**

| Window | Min variance | Max Sharpe | Constrained max Sharpe | Equal weight |
|---|---|---|---|---|
| 2008 GFC (Nov 2007 to Feb 2009) | +6.7% (dd 2.1%) | -3.5% (dd 8.2%) | -11.3% (dd 17.2%) | -23.7% |
| 2013 taper tantrum | -0.1% | 0.0% | -1.2% (dd 4.5%) | -6.2% |
| 2020 COVID | +1.7% (dd 0.9%) | -1.9% (dd 5.1%) | -4.6% (dd 10.5% daily vs 4.6% monthly) | -7.6% |
| 2022 rate shock | -4.1% (dd 5.3%) | -5.8% (dd 8.6%) | -9.4% (dd 13.7%) | -15.6% |

**Hypothetical shocks (instant):** equity -30%: -0.5%, -5.0%, -9.3%, -11.4%. Credit wider: 0.0%, 0.0%, 0.0%, -2.0%. Rates up: -2.0%, -1.5%, -1.9%, -3.4%. All three: -2.5%, -6.5%, -11.1%, -16.7%.

**Regimes (Sharpe ratio using T-bill return in the same months):**

| Portfolio | Full sample | Rates falling | Rates rising | Vol low | Vol high |
|---|---|---|---|---|---|
| Min variance | 0.40 | 1.06 | **-0.23** | 0.47 | 0.19 |
| Max Sharpe | 0.82 | 0.99 | 0.61 | 0.95 | 0.64 |
| Constrained max Sharpe | 0.82 | 0.89 | 0.73 | 0.95 | 0.66 |
| Equal weight | 0.49 | 0.54 | 0.42 | 0.54 | 0.49 |

Average pairwise correlation: 0.31 in falling-rate months vs **0.52 in rising-rate months**; 0.35 low-vol vs 0.44 high-vol.

**Do the optimized portfolios hold up? Honest answer:**
- **Min variance does not hold up when rates rise.** Its Sharpe drops from 1.06 to -0.23: it earns 1.4% a year while T-bills earn 1.8%, because short Treasuries lag cash when rates are climbing. "Lowest volatility" is not "safest" in that regime.
- **Max Sharpe degrades in rising rates** (0.99 to 0.61) and in high volatility (0.95 to 0.64).
- **Constrained max Sharpe is the most stable** of the optimized portfolios (0.66 to 0.95), but has the largest losses in absolute terms among them: -11.3% in 2008, -9.4% in 2022, worst month -7.7%.
- **2022 hurt everything:** stocks and bonds fell together, so diversification failed. Rising-rate months have much higher correlations (0.52 vs 0.31), which is the core risk for a bond-heavy insurer.
- **The volatility regime lags sudden crashes** (D-029), so high-vol months include rebounds; high-vol results look better than the crash experience.
- **All of this is in-sample.** The portfolios were built knowing these returns, so real-world results would be worse.
- **None of these portfolios is realistic for an insurer:** no credit (an insurer's core holding) and no liability hedge. Stage 3 changes that.

## 6c. Stage 2 results (from `reports/stage2_results.md`)

**Today's curve (2026-09-30):** 3M 4.20%, 2Y 4.88%, 5Y 5.09%, 10Y 5.29%, 20Y 5.68%, 30Y 5.64%. ICE OAS: IG 0.84%, HY 3.12%. Baa-10y 1.47%.

**Rate model:** Vasicek a = 0.150, b = 2.11%, sigma = 1.02%, lam = 0.337 (b_q = 4.39%). Simulated short rate after 5 years: median about 3.1%, 5th to 95th percentile about 0.4% to 5.7%; 3.1% of paths negative.

**Factor model vs historical bootstrap (annual volatility):** equities match by construction (SPY 15.5% vs 15.4%). Bonds are far too calm in the factor model: TLT 4.5% vs 14.1%, IEF 4.3% vs 6.7%, LQD 4.3% vs 8.2%, HYG 5.4% vs 10.1%, TIP 2.7% vs 5.8%. 1-year 5th percentile: TLT +0.3% (factor) vs -19.3% (bootstrap). **Use the bootstrap for asset risk until a second rate factor is added.**

**Expected returns before vs after (bond ETFs):**

| ETF | Bayes-Stein (history) | Yield-based | Excess over cash, after (rf 4.20%) |
|---|---|---|---|
| SHY | 1.73% | 4.88% | +0.68% |
| IEF | 2.67% | 5.24% | +1.04% |
| TLT | 2.90% | 5.66% | +1.46% |
| TIP | 2.78% | 5.21% | +1.01% |
| LQD | 3.25% | 6.01% | +1.81% |
| HYG | 3.94% | 5.64% | +1.44% |

**Optimized portfolios, before (Bayes-Stein, rf 1.38%) vs after (yield-based, rf 4.20%):**

| Portfolio | Before | After |
|---|---|---|
| Min variance | 98.1% SHY (unchanged: it ignores expected returns) | same |
| Max Sharpe | 74.4% SHY, 16.8% SPY, 8.9% GLD | 93.8% SHY, 5.8% SPY |
| Constrained max Sharpe | 35% SHY, 30.9% SPY, 17.6% GLD, 16.6% IEF | 35% SHY, 29.4% IEF, 25% SPY, 10.6% GLD |

Still **no credit and no TLT**: LQD's 1.81% excess return comes with 8.3% historical volatility (2008 included), a Sharpe of about 0.22 vs 0.45 for SHY. An insurer would hold credit anyway, for liability matching and spread income; that needs liabilities and capital (Stage 3).

**Duration-based stress, portfolio impact (full revaluation):**

| Shock | Constrained max Sharpe before | Constrained max Sharpe after | Equal weight |
|---|---|---|---|
| Parallel +100bp | -1.82% | -2.75% | -3.63% |
| Parallel -100bp | +1.95% | +2.97% | +4.21% |
| Steepener | +0.44% | +0.53% | -0.27% |
| IG +100bp / HY +400bp | 0.00% / 0.00% | 0.00% / 0.00% | -0.72% / -1.13% |

## 6d. Stage 3 results (from `reports/stage3_results.md`, 10,000 scenarios)

Balance sheet: assets $1,100m, liabilities $1,000m (duration 9.48), surplus $100m.

| Portfolio | Exp. 1y surplus change | Surplus vol | 1-in-20 avg loss | Capital proxy | Coverage | Liquidity | FR 5y median | FR 5y p05 | P(FR < 100%) 5y |
|---|---|---|---|---|---|---|---|---|---|
| Asset-only min variance (97.7% SHY) | -$2.8m | $63.8m | $155.5m | $90.9m | 1.1x | 5.5x | 114.1% | 90.5% | 16.8% |
| Asset-only constrained max Sharpe | +$16.7m | $72.9m | $173.6m | $175.3m | 0.6x | 4.0x | 126.1% | 90.8% | 11.0% |
| Key-rate matched (Treasuries) | +$4.5m | $10.6m | $20.9m | $2.3m | 43.7x | 5.2x | 116.5% | 110.0% | 0.0% |
| Surplus-aware, no credit floor | +$10.0m | $20.1m | $36.7m | $36.1m | 2.8x | 4.9x | 120.1% | 109.3% | 0.1% |
| **Recommended: surplus + capital aware + duration band (D-057)** | **+$10.0m** | **$34.7m** | **$71.5m** | **$33.5m** | **3.0x** | **4.5x** | **120.3%** | **103.0%** | **2.9%** |
| CVaR variant (same constraints incl. band) | +$10.0m | $34.7m | $71.5m | $33.5m | 3.0x | 4.5x | 120.3% | 103.0% | 2.9% |
| Comparison: no duration band (D-052) | +$10.0m | $33.1m | $65.8m | $31.8m | 3.1x | 4.6x | 119.9% | 104.0% | 2.3% |

Recommended (banded) weights: LQD 40.0%, TLT 33.7%, SHY 19.8%, GLD 5.0%, SPY 1.5%. Comparison (no band): LQD 40.0%, TLT 39.4%, SHY 17.4%, GLD 3.2%. Asset-only constrained max Sharpe: SPY 31.0%, SHY 35.0%, TLT 24.8%, GLD 9.3%.

Surplus change under instant shocks ($m), recommended (banded) / comparison (no band) / key-rate matched: parallel +100bp -7.7 / -16.8 / -2.3; parallel -100bp +13.0 / +24.8 / +6.6; steepener -3.5 / -7.7 / +0.9; flattener +3.7 / +8.1 / -0.5; IG +100bp -34.9 / -34.9 / 0.0; HY +400bp 0.0 for all (no HY held).

Where the recommended portfolio breaks: an investment grade spread shock (-$34.9m, funding ratio 106.5%), and a 2022-style combination of rising rates and widening spreads (about -$42.6m; -$51.7m without the band). Historical replay (in-sample): average surplus change $8.8m/yr in falling-rate months, $7.9m/yr in rising-rate months, $30.7m/yr in high-volatility months, $2.0m/yr in low-volatility months; worst month -$50.0m.

## 6e. Walk-forward results (from `reports/walkforward_results.md`)

Rebalance every December 2010 to 2025, point-in-time data only, evaluated on realized ETF returns and the actual Treasury curve.

**Robustness view (fresh $1.1bn assets vs $1bn liabilities every December), 15 full years 2011 to 2025:**

| Strategy | Mean surplus change / yr | Volatility | Worst year | 2022 | Losing years |
|---|---|---|---|---|---|
| Walk-forward optimized (out of sample) | +$7.6m | $23.2m | -$44.8m (2022) | -$44.8m | 3 |
| Static in-sample headline (has seen the future) | +$11.6m | $23.9m | -$26.5m (2011) | -$14.4m | 5 |
| Equal weight | +$34.5m | $76.4m | -$134.2m (2011) | +$20.6m | 5 |
| Key-rate matched Treasuries | -$1.0m | $8.0m | -$25.1m (2022) | -$25.1m | 7 |

**Primary view (one block runs off from Dec 2010, pre-specified):** total surplus change +$280.8m (walk-forward), +$98.6m (static in-sample), +$433.8m (equal weight), +$7.3m (key-rate matched); 2022: -$59.8m, -$75.5m, -$58.5m, -$12.8m; lowest funding ratio 107.5%, 105.8%, 94.1% (equal weight, Sept 2011), 108.2%; average turnover 18.5%, 2.4%, 3.6%, 3.2% a year. Later decisions hit max-return corners and one infeasible year (P-024).

**Honest reading:** out of sample the optimized process earned less than its in-sample version and its worst year (2022) was worse, because it sat at or next to the long edge of the duration band every year. Its risk forecasts were conservative (0 of 15 breaches of the predicted 1-in-20 loss). Equal weight earned most but with three times the surplus volatility and a funding ratio below 100% in 2011. Pure matching was the steadiest and earned nothing.

## 6f. Liability discount-rate variant (from `reports/discount_variant_results.md`)

Same cash flows; discount curve = Treasury zero curve + share x IG spread (today 84bp); the share moves with simulated spreads. Assets set to 110% of each variant's liability value for the optimization.

| Share of IG spread | 0% (base) | 25% | 50% (recommended variant) | 75% | 100% |
|---|---|---|---|---|---|
| Liability PV | $1,000.0m | $980.3m | $961.2m | $942.7m | $924.7m |
| Liability duration | 9.48 | 9.37 | 9.26 | 9.15 | 9.04 |
| Funding ratio, assets fixed at $1.1bn | 110.0% | 112.2% | 114.4% | 116.7% | 119.0% |
| LQD correlation with liabilities | 0.52 | 0.58 | 0.63 | 0.69 | 0.74 |
| Credit chosen freely (no floor) | 0.0% | 0.0% | 10.5% | 16.9% | 20.8% |
| Headline rule: TLT / SHY (LQD 40%) | 33.7% / 19.8% | 31.5% / 28.5% | 27.9% / 32.1% | 25.3% / 34.7% | 22.9% / 37.1% |
| Headline surplus volatility | $34.7m | $24.8m | $20.3m | $17.0m | $14.9m |
| Headline IG +100bp surplus change | -$34.9m | -$11.1m | +$11.2m | +$32.2m | +$51.8m |

**Reading:** with Treasury discounting, credit only adds surplus risk; with an illiquidity premium tied to credit spreads, credit becomes a partial hedge, the optimizer picks it on its own, and the fixed 40% credit allocation turns from the main risk into an over-hedge above about 40% of the spread. The base case remains the headline because it assumes no premium.

---

## 7. Honest limits (what this project does NOT do or claim)

- **Not investment advice.** Outputs are illustrations of a method.
- **No liabilities yet.** The liability interface exists and is tested with toy liabilities only. Stage 3 liabilities will be **synthetic**, not from a real insurer.
- **Not NAIC RBC.** Any capital measure in Stage 3 will be a simplified proxy, clearly labeled.
- **ETFs are proxies.** Real insurers hold individual bonds, private placements, mortgages and private credit. ETF returns include fees and tracking error.
- **About 19 years of history.** Includes 2008, 2020 and 2022, but not the 2000 to 2002 equity bear market, the 1994 bond crash, or a 1970s-style inflation. One credit crisis is a small sample.
- **Results are sensitive to the window.** Adding 2007 to 2011 changed the unconstrained optimal portfolio completely (section 6).
- **Backward-looking expected returns.** Bond estimates flip sign depending on the window (P-002). Gold's estimate is generous. Expected returns are the weakest input.
- **One-period model.** No rebalancing, no multi-year path, no cash flow timing, no liquidity needs yet.
- **Volatility as the risk measure.** Ignores fat tails and skew. No tail measures (VaR, CVaR) yet.
- **Annualization assumes independent months.**
- **Constraints are illustrative**, not from any real investment policy.
- **No taxes, transaction costs, or accounting** (statutory vs GAAP) effects.
- **Data from an unofficial API** (yfinance); verified by hash and by agreement between two downloads (V-026), but not against a second vendor.
- **In-sample vs out-of-sample:** a walk-forward test now exists (D-059 to D-062); out of sample the process earned less than in sample. Still in-sample: the static headline weights, the regime tables, the Stage 1b simulations, and fixed settings such as the curve pull speed (kappa) calibrated on the full sample.
- **The bootstrap cannot imagine a crisis worse than 2008**; with 232 months the far tail rests on a handful of crisis months.
- **Monte Carlo has no parameter uncertainty:** it treats the expected returns and covariance as known. Real uncertainty in expected returns would widen every distribution.
- **Hypothetical shocks are simple factor shocks, NOT duration-based**, with judgment-based sizes.
- **Volatility regimes lag sudden crashes** (COVID was labeled low-vol). The 20% threshold involves mild hindsight.
- **Regime samples are small** (52 high-vol months), so regime averages are noisy.
- **No transaction costs** in rebalancing.
- **Credit spread history is a proxy.** Moody's Baa-10y, not ICE OAS, which is free only from Oct 2023. One credit factor for IG and HY.
- **One-factor rate model understates long-bond risk about 3x** (P-013). Factor-model bond volatilities should not be used for risk until a second rate factor is added.
- **Vasicek does not fit today's curve** (patched with a constant offset) **and allows negative rates** (3.1% of paths at 5 years).
- **Real-world rate drift is weakly identified** and pulls rates down toward 2.1%, a built-in bullish bond view.
- **Default losses are constant assumptions** (IG 0.20%, HY 2.50%), not modeled defaults; no downgrade or default clustering.
- **Published ETF durations are quoted from memory** for cross-checking, not fetched.
- **No fund fees, roll-down or accrued interest** in bond returns; durations held constant.
- **TIP modeled through nominal yields** with an empirical pass-through; real yields and inflation are not modeled.
- **Stage 3 liabilities are synthetic** (fixed nominal payout annuities, no mortality model, no policyholder options beyond a surrender test).
- **Liabilities discounted at Treasuries in the headline**, which makes credit pure surplus risk. A variant with an illiquidity premium (share of the IG spread, D-063 to D-065) shows credit then hedges the liabilities; the premium size is an assumption.
- **Capital is a PROXY, not NAIC RBC**: assumed factors, simple sum, no covariance or tax effects.
- **Liquidity buckets and haircuts are assumptions** borrowed from bank regulation; the constraint never binds here.
- **Scenarios replay 2007 to 2026**; the rate anchor is the market forward curve (includes term premium); the Fed's long-run rate could not be fetched (FRED down).
- **One-year optimization horizon** with buy-and-hold returns; 5-year results by simulation; no transaction costs, taxes or statutory/GAAP accounting.
- **The recommended portfolio relies on a historical rate-spread correlation** (flight to quality) that failed in 2022 (D-053).
- **Max Sharpe with liabilities** uses a non-standard definition; to be revisited in Stage 3.

---

## 8. Open questions (need a decision from me)

Each has my recommended answer.

1. ~~Extend the window to mid-2007?~~ **Resolved 2026-10-19: yes (D-020).**
2. ~~Bond expected returns.~~ **Resolved 2026-10-20: yield-based in Stage 2 (D-041).**
3. ~~Surplus objective for Stage 3.~~ **Resolved 2026-10-20: minimum surplus volatility for a target surplus change, CVaR variant alongside, surplus Sharpe reported only (D-052).**
4. ~~Illustrative constraints.~~ **Resolved 2026-10-20: insurer constraints in Stage 3 (D-051).**
5. **Even longer history?** **Recommended: keep ETF-only data for now.** If more stress history is wanted later, add 2000 to 2002 as an explicit hypothetical or index-based scenario rather than splicing proxies into the estimation data.
6. **New: default Monte Carlo method.** **Recommended: keep the block bootstrap as the default,** and always show normal alongside it so the tail gap stays visible.
7. ~~Walk-forward testing.~~ **Resolved 2026-10-20: done (D-059 to D-062). Out of sample the process earned less than in sample; risk forecasts were conservative.**
8. **New: volatility signal.** Keep trailing realized volatility, or add VIX as a faster signal? **Recommended: keep realized volatility as the documented default, and add VIX as a comparison only if regimes become an input to decisions** (they are descriptive today).
9. **New: monthly parameter uncertainty.** Should Monte Carlo also randomize expected returns (draw them from their estimation error)? **Recommended: yes, as a Stage 4 extra;** it is the honest way to show that we do not know the expected returns.
10. ~~Second rate factor.~~ **Resolved 2026-10-20: empirical PCA generator is the default (D-043); Vasicek kept as benchmark.**
11. ~~Long-run rate level b.~~ **Resolved 2026-10-20: anchored to the market forward curve with sensitivities (D-044).** See 16 for the Fed series.
12. **New: credit spread data.** **Recommended: keep the Baa factor unless a long ICE OAS history can be obtained;** if one becomes available, add a separate HY spread factor.
13. **New: default loss assumptions.** **Recommended: keep IG 0.20% and HY 2.50% as labeled assumptions, and show a sensitivity (HY 1.5% to 3.5%),** since they decide whether high yield earns anything over Treasuries.
14. **TIP and inflation.** **Recommended: download real yields (FRED DFII5, DFII10) only if the liabilities become inflation-linked;** the synthetic block is nominal, so the empirical pass-through is adequate for now.
15. **Fund fees. Not done in Stage 3.** **Recommended: subtract expense ratios from ETF expected returns in Stage 4** (small for most, about 0.5% for HYG), once the fee numbers are fetched rather than remembered.
16. **New: Fed long-run policy rate as a second anchor.** **Recommended: fetch FEDTARMDLR from FRED when it is reachable and add it as a sensitivity;** it removes the term-premium bias of forward rates.
17. ~~Duration-gap band.~~ **Resolved 2026-10-20: adopted as the headline recommendation, plus or minus $10m per 100bp (D-057).**
18. ~~Liability discount rate.~~ **Resolved 2026-10-20: variant built (D-063 to D-065); headline stays on Treasury discounting, 50% of IG spread shown as the recommended variant.**
19. **Recommended next:** a fat-tailed scenario overlay for losses beyond 2008, then real fee data (15), then a pre-registered test of a centered or narrower duration band (20). Also open: which discount basis the client's regulator or accounting regime actually uses (it decides the credit question, D-065).
20. **New: the optimizer sits at the long edge of the duration band every year.** **Recommended: in a later, separately reported test, compare a band centered on zero gap or a narrower band,** and pre-register the comparison so it is not tuned to 2022.

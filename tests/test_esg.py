import numpy as np
import pytest

from engine import esg, scenarios
from engine import fixedincome as fi


@pytest.fixture(scope="module")
def pca(real_fred):
    tenors, par = scenarios.par_curve_history(real_fred)
    return esg.curve_pca(par, tenors), par


def test_pca_is_a_valid_decomposition(pca):
    p, par = pca
    V = p.loadings
    assert np.allclose(V.T @ V, np.eye(len(V)), atol=1e-12)  # orthonormal
    assert p.explained.sum() == pytest.approx(1.0)
    assert np.all(np.diff(p.eigenvalues) <= 0)
    d = par.diff().dropna().values
    assert np.allclose(p.scores.values @ V.T + p.mean_change, d, atol=1e-12)  # all components rebuild history
    assert p.explained[:3].sum() > 0.95


def test_pca_factor_shapes_are_level_slope_curvature(pca):
    V = pca[0].loadings
    assert (V[:, 0] > 0).all()  # level: every yield moves the same way
    assert V[-1, 1] > 0 > V[0, 1]  # slope: long end vs short end
    assert V[0, 2] > 0 and V[-1, 2] > 0 and V[3, 2] < 0  # curvature: wings vs belly


def test_pca_generator_matches_historical_bond_volatility(stage3):
    ctx = stage3["ctx"]
    for etf in ("IEF", "TLT"):
        hist = ctx.a[etf].std() * np.sqrt(12)
        assert esg.annual_vol(ctx.scen.returns[etf]) == pytest.approx(hist, rel=0.10)
    d10 = np.diff(ctx.scen.par[:, :, ctx.tenors.index(10.0)], axis=1).std() * np.sqrt(12)
    assert d10 == pytest.approx(ctx.par["DGS10"].diff().std() * np.sqrt(12), rel=0.10)


def test_vasicek_benchmark_understates_tlt_vol(stage3):
    """Documents P-013: the one-factor model is far too calm, the PCA generator is not."""
    ctx = stage3["ctx"]
    fp = scenarios.simulate_factors(ctx.fm, 2000, 60, seed=1)
    vas = esg.annual_vol(scenarios.asset_returns(ctx.fm, fp, 1)["TLT"])
    assert vas < 0.5 * ctx.a["TLT"].std() * np.sqrt(12)


def test_seeded_generator_is_reproducible(stage3):
    ctx = stage3["ctx"]
    args = (ctx.fm, ctx.pca, ctx.rets, ctx.fred_m, ctx.mu_bs, ctx.anchor)
    a = esg.simulate(*args, esg.ESGConfig(n_paths=300, n_steps=24, seed=3))
    b = esg.simulate(*args, esg.ESGConfig(n_paths=300, n_steps=24, seed=3))
    c = esg.simulate(*args, esg.ESGConfig(n_paths=300, n_steps=24, seed=4))
    assert np.array_equal(a.par, b.par) and all(np.array_equal(a.returns[k], b.returns[k]) for k in a.returns)
    assert not np.array_equal(a.par, c.par)


def test_curve_is_pulled_toward_the_anchor(stage3):
    """With the random moves switched off, the curve decays to the anchor at
    exactly (1 - kappa dt) per month."""
    ctx = stage3["ctx"]
    cfg = esg.ESGConfig(n_paths=5, n_steps=60, seed=1, n_factors=0, kappa=0.5)
    s = esg.simulate(ctx.fm, ctx.pca, ctx.rets, ctx.fred_m, ctx.mu_bs, ctx.anchor, cfg)
    expected = ctx.anchor + (ctx.y0 - ctx.anchor) * (1 - 0.5 / 12) ** 60
    assert np.allclose(s.par[:, -1, :], expected, atol=1e-14)


def test_forward_curve_of_a_flat_curve_is_flat():
    tenors = [0.25, 1, 2, 5, 7, 10, 20, 30]
    curve = fi.bootstrap_zero_curve(tenors, [0.05] * 8)
    assert np.allclose(esg.forward_par_curve(curve, tenors), 0.05, atol=1e-12)


def test_equity_like_scenarios_are_mean_adjusted(stage3):
    ctx = stage3["ctx"]
    for a in esg.EQUITY_LIKE:
        x = ctx.scen.returns[a]
        se = x.std() / np.sqrt(x.size / 6)  # blocks of 6 months: rough effective sample size
        assert abs(x.mean() - ctx.mu_bs[a] / 12) < 4 * se

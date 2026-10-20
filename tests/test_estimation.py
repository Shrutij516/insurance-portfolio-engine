import numpy as np
import pytest

from engine import data
from engine import estimation as est


@pytest.mark.parametrize("method", ["sample", "lw", "lw_corr"])
def test_covariance_symmetric_and_psd_real(real_returns, method):
    cov = est.estimate_covariance(real_returns[list(data.UNIVERSE)], method)
    m = cov.values
    assert np.allclose(m, m.T, atol=1e-14)
    assert np.linalg.eigvalsh(m).min() > 0


@pytest.mark.parametrize("method", ["sample", "lw", "lw_corr"])
def test_covariance_symmetric_and_psd_synthetic(synthetic_returns, method):
    chk = est.check_covariance(est.estimate_covariance(synthetic_returns, method))
    assert chk.symmetric and chk.psd and chk.min_eigenvalue > 0


def test_check_covariance_rejects_bad_matrices():
    with pytest.raises(ValueError, match="not PSD"):
        est.check_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))  # eigenvalues 3 and -1
    with pytest.raises(ValueError, match="not symmetric"):
        est.check_covariance(np.array([[1.0, 0.5], [0.1, 1.0]]))


def test_lw_corr_keeps_sample_volatilities(real_returns):
    a = real_returns[list(data.UNIVERSE)]
    s = est.sample_covariance(a)
    c, delta = est.ledoit_wolf_correlation_covariance(a)
    assert np.allclose(np.diag(c), np.diag(s), rtol=1e-12)
    assert 0 < delta < 1
    # and the off-diagonals really were shrunk toward zero correlation
    corr_s = s.values / np.sqrt(np.outer(np.diag(s), np.diag(s)))
    corr_c = c.values / np.sqrt(np.outer(np.diag(c), np.diag(c)))
    off = ~np.eye(len(a.columns), dtype=bool)
    assert np.all(np.abs(corr_c[off]) <= np.abs(corr_s[off]) + 1e-12)


def test_annualization(synthetic_returns):
    assert np.allclose(est.historical_mean(synthetic_returns), synthetic_returns.mean() * 12)
    assert np.allclose(est.sample_covariance(synthetic_returns), synthetic_returns.cov() * 12)


def test_bayes_stein_shrinks_between_history_and_target(real_returns):
    bs = est.bayes_stein_mean(real_returns[list(data.UNIVERSE)])
    assert 0 <= bs.shrinkage <= 1
    lo = np.minimum(bs.historical, bs.target)
    hi = np.maximum(bs.historical, bs.target)
    assert ((bs.mu >= lo - 1e-12) & (bs.mu <= hi + 1e-12)).all()
    # shrinkage reduces the spread across assets
    assert bs.mu.std() < bs.historical.std()

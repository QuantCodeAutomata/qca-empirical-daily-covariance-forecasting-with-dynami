import pytest
import numpy as np
from src.realized_measures import (
    parzen_kernel,
    compute_multivariate_kernel_cov,
    compute_realized_covariance_simple,
    compute_quarticity_proxy,
    compute_pi_matrix,
    compute_pi_vector,
)

# Function signatures confirmed from src/realized_measures.py:
#   compute_quarticity_proxy(returns, n_assets) -> (N, N)
#   compute_pi_matrix(IQ, kappa=3.777)          -> (N, N)   [= kappa * IQ]
#   compute_pi_vector(Pi)                       -> (N,)     [= sqrt(diag(Pi))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intraday(N, M, true_cov, seed):
    """Simulate M intraday i.i.d. returns from true_cov / M."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(true_cov / M)
    return rng.standard_normal((M, N)) @ L.T


def _make_pd_cov(N, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    return A @ A.T / N + np.eye(N) * 0.5


# ---------------------------------------------------------------------------
# Parzen kernel:  k(0)=1, k(1)=0, k(x)>=0 on [0,1]
# ---------------------------------------------------------------------------

def test_parzen_kernel_at_zero():
    assert np.isclose(parzen_kernel(0.0), 1.0)


def test_parzen_kernel_at_one():
    assert np.isclose(parzen_kernel(1.0), 0.0)


def test_parzen_kernel_positive_domain():
    xs = np.linspace(0.0, 1.0, 101)
    vals = np.array([parzen_kernel(x) for x in xs])
    assert np.all(vals >= -1e-12), "Parzen kernel must be non-negative on [0,1]"


def test_parzen_kernel_monotone_decrease():
    """k(x) is non-increasing on [0,1] for the Parzen kernel."""
    xs = np.linspace(0.0, 1.0, 51)
    vals = np.array([parzen_kernel(x) for x in xs])
    diffs = np.diff(vals)
    assert np.all(diffs <= 1e-12), "Parzen kernel should be non-increasing on [0,1]"


def test_parzen_kernel_outside_unit_interval():
    """k(x) = 0 for x > 1."""
    assert parzen_kernel(1.5) == 0.0
    assert parzen_kernel(2.0) == 0.0


# ---------------------------------------------------------------------------
# compute_realized_covariance_simple:  S = sum_i r_i r_i'
# ---------------------------------------------------------------------------

def test_simple_rcov_shape():
    N = 4
    M = 78
    Sigma = _make_pd_cov(N, seed=0)
    returns = _make_intraday(N, M, Sigma, seed=1)
    S = compute_realized_covariance_simple(returns)
    assert S.shape == (N, N), f"Expected ({N},{N}), got {S.shape}"


def test_simple_rcov_symmetric():
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=2)
    returns = _make_intraday(N, M, Sigma, seed=3)
    S = compute_realized_covariance_simple(returns)
    assert np.allclose(S, S.T, atol=1e-14)


def test_simple_rcov_pd():
    """With many intraday observations the sample RCov should be PD."""
    N = 3
    M = 390
    Sigma = _make_pd_cov(N, seed=4)
    returns = _make_intraday(N, M, Sigma, seed=5)
    S = compute_realized_covariance_simple(returns)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals > 0), "Simple RCov must be PD for M > N"


def test_simple_rcov_consistent():
    """For large M the simple RCov converges to the true covariance."""
    N = 2
    M = 10_000
    Sigma = np.array([[1.0, 0.4], [0.4, 0.8]])
    returns = _make_intraday(N, M, Sigma, seed=6)
    S = compute_realized_covariance_simple(returns)
    assert np.allclose(S, Sigma, atol=0.05), \
        "For large M, simple RCov should approximate true covariance"


# ---------------------------------------------------------------------------
# compute_multivariate_kernel_cov
# ---------------------------------------------------------------------------

def test_mk_rcov_shape():
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=10)
    returns = _make_intraday(N, M, Sigma, seed=11)
    S = compute_multivariate_kernel_cov(returns)
    assert S.shape == (N, N)


def test_mk_rcov_symmetric():
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=12)
    returns = _make_intraday(N, M, Sigma, seed=13)
    S = compute_multivariate_kernel_cov(returns)
    assert np.allclose(S, S.T, atol=1e-12)


def test_mk_rcov_pd():
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=14)
    returns = _make_intraday(N, M, Sigma, seed=15)
    S = compute_multivariate_kernel_cov(returns)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals > 0), "Multivariate kernel RCov must be PD"


def test_mk_vs_simple_similar():
    """MK and simple RCov should have comparable Frobenius magnitudes.

    The Parzen kernel adds lag-autocovariance corrections that can shift
    individual elements, but the overall scale (Frobenius norm) should remain
    within an order of magnitude of the simple estimator.
    """
    N = 3
    M = 390
    Sigma = _make_pd_cov(N, seed=16)
    returns = _make_intraday(N, M, Sigma, seed=17)
    S_simple = compute_realized_covariance_simple(returns)
    S_mk = compute_multivariate_kernel_cov(returns)
    frob_ratio = np.linalg.norm(S_simple, "fro") / np.linalg.norm(S_mk, "fro")
    assert 0.1 < frob_ratio < 10.0, \
        f"Frobenius norm ratio {frob_ratio:.2f} out of expected range (0.1, 10)"


# ---------------------------------------------------------------------------
# compute_quarticity_proxy:  IQ_{ij} = (T/3) * sum_k r_{ki}^2 * r_{kj}^2
# Signature: compute_quarticity_proxy(returns, n_assets) -> (N, N)
# ---------------------------------------------------------------------------

def test_quarticity_shape():
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=20)
    returns = _make_intraday(N, M, Sigma, seed=21)
    Q = compute_quarticity_proxy(returns, n_assets=N)
    assert Q.shape == (N, N), f"Expected ({N},{N}), got {Q.shape}"


def test_quarticity_nonnegative():
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=22)
    returns = _make_intraday(N, M, Sigma, seed=23)
    Q = compute_quarticity_proxy(returns, n_assets=N)
    assert np.all(Q >= 0.0), "Quarticity proxy must be element-wise non-negative"


def test_quarticity_diagonal_positive():
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=24)
    returns = _make_intraday(N, M, Sigma, seed=25)
    Q = compute_quarticity_proxy(returns, n_assets=N)
    assert np.all(np.diag(Q) > 0.0), "Diagonal quarticity must be strictly positive"


# ---------------------------------------------------------------------------
# compute_pi_matrix(IQ, kappa=3.777) -> (N, N)
# compute_pi_vector(Pi)              -> (N,)  = sqrt(diag(Pi))
# ---------------------------------------------------------------------------

def test_pi_matrix_shape():
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=30)
    returns = _make_intraday(N, M, Sigma, seed=31)
    IQ = compute_quarticity_proxy(returns, n_assets=N)
    Pi = compute_pi_matrix(IQ)
    assert Pi.shape == (N, N)


def test_pi_matrix_scales_iq():
    """Pi_t = kappa * IQ_t (default kappa = 3.777)."""
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=36)
    returns = _make_intraday(N, M, Sigma, seed=37)
    IQ = compute_quarticity_proxy(returns, n_assets=N)
    Pi = compute_pi_matrix(IQ)
    assert np.allclose(Pi, 3.777 * IQ)


def test_pi_vector_length():
    """compute_pi_vector returns an N-vector (one entry per asset)."""
    N = 4
    M = 78
    Sigma = _make_pd_cov(N, seed=32)
    returns = _make_intraday(N, M, Sigma, seed=33)
    IQ = compute_quarticity_proxy(returns, n_assets=N)
    Pi = compute_pi_matrix(IQ)
    pi = compute_pi_vector(Pi)
    assert pi.shape == (N,), f"Expected ({N},), got {pi.shape}"


def test_pi_vector_positive():
    """All pi_t elements must be positive (sqrt of positive diagonal entries)."""
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=34)
    returns = _make_intraday(N, M, Sigma, seed=35)
    IQ = compute_quarticity_proxy(returns, n_assets=N)
    Pi = compute_pi_matrix(IQ)
    pi = compute_pi_vector(Pi)
    assert np.all(pi > 0), "pi_t elements must be strictly positive"


def test_pi_vector_equals_sqrt_diag():
    """pi_t_i = sqrt(Pi_{t,ii}) by definition."""
    N = 3
    M = 78
    Sigma = _make_pd_cov(N, seed=38)
    returns = _make_intraday(N, M, Sigma, seed=39)
    IQ = compute_quarticity_proxy(returns, n_assets=N)
    Pi = compute_pi_matrix(IQ)
    pi = compute_pi_vector(Pi)
    assert np.allclose(pi, np.sqrt(np.diag(Pi)))

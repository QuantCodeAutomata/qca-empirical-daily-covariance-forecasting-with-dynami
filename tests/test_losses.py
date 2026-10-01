import pytest
import numpy as np
from src.losses import frobenius_loss, qlike_loss, diebold_mariano_test


# ---------------------------------------------------------------------------
# Frobenius loss:  L(H, S) = ||H - S||_F^2 = sum_{ij}(H_ij - S_ij)^2
# ---------------------------------------------------------------------------

def test_frobenius_loss_identity():
    N = 3
    I = np.eye(N)
    assert np.isclose(frobenius_loss(I, I), 0.0)


def test_frobenius_loss_positive():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    H = A @ A.T + np.eye(4)
    B = rng.standard_normal((4, 4))
    S = B @ B.T + np.eye(4)
    assert frobenius_loss(H, S) >= 0.0


def test_frobenius_loss_known_value():
    # H = 3*I_2, S = 2*I_2  →  H-S = I_2  →  ||I_2||_F = sqrt(1+1) = sqrt(2)
    H = np.diag([3.0, 3.0])
    S = np.diag([2.0, 2.0])
    assert np.isclose(frobenius_loss(H, S), np.sqrt(2.0))


def test_frobenius_loss_symmetric():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((3, 3))
    H = A @ A.T + np.eye(3)
    B = rng.standard_normal((3, 3))
    S = B @ B.T + np.eye(3)
    assert np.isclose(frobenius_loss(H, S), frobenius_loss(S, H))


# ---------------------------------------------------------------------------
# QLIKE loss:  L(H, S) = log|H| + Tr(H^{-1} S)
# ---------------------------------------------------------------------------

def test_qlike_loss_identity():
    # log|I_N| + Tr(I_N^{-1} I_N) = 0 + N = N
    N = 4
    I = np.eye(N)
    assert np.isclose(qlike_loss(I, I), float(N))


def test_qlike_loss_positive():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((3, 3))
    H = A @ A.T + np.eye(3)
    B = rng.standard_normal((3, 3))
    S = B @ B.T + np.eye(3)
    val = qlike_loss(H, S)
    assert np.isfinite(val)


def test_qlike_loss_known_2x2():
    # H = 2*I_2, S = I_2
    # log|2*I| = 2*log(2), Tr((2I)^{-1} I) = Tr(0.5*I) = 1
    # QLIKE = 2*log(2) + 1
    H = 2.0 * np.eye(2)
    S = np.eye(2)
    expected = 2.0 * np.log(2.0) + 1.0
    assert np.isclose(qlike_loss(H, S), expected)


def test_qlike_singular_returns_nan():
    N = 3
    H = np.zeros((N, N))   # singular forecast → undefined
    S = np.eye(N)
    val = qlike_loss(H, S)
    assert np.isnan(val)


def test_qlike_loss_finite_pd_inputs():
    rng = np.random.default_rng(4)
    for _ in range(10):
        A = rng.standard_normal((5, 5))
        H = A @ A.T + np.eye(5)
        B = rng.standard_normal((5, 5))
        S = B @ B.T + np.eye(5)
        assert np.isfinite(qlike_loss(H, S))


# ---------------------------------------------------------------------------
# Diebold–Mariano test
# ---------------------------------------------------------------------------

def test_dm_test_zero_diff():
    # Zero-mean loss differential → DM stat should be small and insignificant.
    # Using a symmetric perturbation ensures d_t has mean ≈ 0 but S > 0.
    rng = np.random.default_rng(5)
    base = np.abs(rng.standard_normal(500))
    noise = rng.standard_normal(500) * 0.01
    stat, pval = diebold_mariano_test(base + noise, base - noise)
    assert np.isfinite(stat), "DM stat must be finite for well-defined series"
    assert abs(stat) < 3.0, f"DM stat {stat:.3f} too large for zero-mean differential"


def test_dm_test_significant():
    # Loss series 1 is systematically much larger → should reject H0
    rng = np.random.default_rng(6)
    e1 = rng.standard_normal(500) + 10.0   # large systematic gap
    e2 = rng.standard_normal(500)
    stat, pval = diebold_mariano_test(e1, e2)
    assert pval < 0.05, f"Expected significant DM test, got p={pval:.4f}"


def test_dm_test_returns_two_floats():
    rng = np.random.default_rng(7)
    e1 = rng.standard_normal(100)
    e2 = rng.standard_normal(100)
    result = diebold_mariano_test(e1, e2)
    assert len(result) == 2
    stat, pval = result
    assert np.isfinite(stat)
    assert 0.0 <= pval <= 1.0

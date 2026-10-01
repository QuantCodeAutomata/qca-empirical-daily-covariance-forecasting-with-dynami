import pytest
import numpy as np
from src.utils import vech, unvech, vech_indices, nearest_spd, is_spd, make_symmetric


# ---------------------------------------------------------------------------
# vech / unvech
# ---------------------------------------------------------------------------

def test_vech_unvech_roundtrip_2x2():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert np.allclose(unvech(vech(A), 2), A)


def test_vech_unvech_roundtrip_5x5():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((5, 5))
    A = B @ B.T + np.eye(5)
    assert np.allclose(unvech(vech(A), 5), A)


def test_vech_length():
    for N in [2, 3, 5, 7]:
        A = np.eye(N)
        assert vech(A).shape[0] == N * (N + 1) // 2


def test_vech_indices():
    for N in [2, 4, 6]:
        rows, cols = vech_indices(N)
        assert len(rows) == N * (N + 1) // 2
        assert all(r >= c for r, c in zip(rows, cols)), \
            "All vech indices must be lower-triangular (row >= col)"


# ---------------------------------------------------------------------------
# nearest_spd
# ---------------------------------------------------------------------------

def test_nearest_spd_already_spd():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    B = nearest_spd(A)
    assert np.allclose(A, B, atol=1e-10)


def test_nearest_spd_indefinite():
    # eigenvalues 3 and -1 → indefinite
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    B = nearest_spd(A)
    assert is_spd(B), "nearest_spd must project an indefinite matrix onto the PD cone"


def test_nearest_spd_3x3():
    # Near-singular: one eigenvalue slightly negative after small perturbation
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    lam = np.array([2.0, 0.5, -0.1])
    A = Q @ np.diag(lam) @ Q.T
    A = (A + A.T) / 2  # enforce symmetry
    B = nearest_spd(A)
    assert is_spd(B)


# ---------------------------------------------------------------------------
# is_spd
# ---------------------------------------------------------------------------

def test_is_spd_positive():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    assert is_spd(A) is True


def test_is_spd_negative_non_pd():
    # Diagonal with negative entry
    A = np.array([[-1.0, 0.0], [0.0, 2.0]])
    assert is_spd(A) is False


def test_is_spd_negative_indefinite():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert is_spd(A) is False


def test_is_spd_singular():
    A = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-1 → not PD
    assert is_spd(A) is False


# ---------------------------------------------------------------------------
# make_symmetric
# ---------------------------------------------------------------------------

def test_make_symmetric_basic():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    S = make_symmetric(A)
    assert np.allclose(S, S.T)


def test_make_symmetric_values():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    S = make_symmetric(A)
    # Off-diagonal should be average
    assert np.isclose(S[0, 1], 2.5)
    assert np.isclose(S[1, 0], 2.5)


def test_make_symmetric_already_symmetric():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    S = make_symmetric(A)
    assert np.allclose(S, A)

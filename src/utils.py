"""
Utility functions for matrix operations, covariance-matrix bookkeeping,
and general quantitative-finance helpers used throughout the HARQ/HARQ-DRD
covariance forecasting project.
"""

from __future__ import annotations

from typing import List

import numpy as np
import scipy.linalg


# ---------------------------------------------------------------------------
# vech / unvech helpers
# ---------------------------------------------------------------------------


def vech_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower-triangular index arrays for an n×n matrix.

    Parameters
    ----------
    n : int
        Dimension of the square matrix.

    Returns
    -------
    rows : np.ndarray of shape (n*(n+1)//2,)
        Row indices of lower-triangular elements (including diagonal).
    cols : np.ndarray of shape (n*(n+1)//2,)
        Corresponding column indices.

    Notes
    -----
    The ordering follows the column-major convention used in the HARQ
    literature: (0,0), (1,0), (2,0), …, (n-1,0), (1,1), (2,1), …
    which is the output of ``np.tril_indices(n)``.
    """
    rows, cols = np.tril_indices(n)
    return rows, cols


def vech(M: np.ndarray) -> np.ndarray:
    """Extract the lower-triangular elements of a symmetric matrix as a vector.

    Parameters
    ----------
    M : np.ndarray of shape (n, n)
        Symmetric (or square) matrix.  Only lower-triangular entries
        (including the diagonal) are read.

    Returns
    -------
    np.ndarray of shape (n*(n+1)//2,)
        Vector of stacked lower-triangular elements.

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[1., 2.], [2., 3.]])
    >>> vech(M)
    array([1., 2., 3.])
    """
    n = M.shape[0]
    rows, cols = vech_indices(n)
    return M[rows, cols]


def unvech(v: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct a symmetric n×n matrix from a vech vector.

    Parameters
    ----------
    v : np.ndarray of shape (n*(n+1)//2,)
        Vector of lower-triangular elements as produced by :func:`vech`.
    n : int
        Target matrix dimension.

    Returns
    -------
    np.ndarray of shape (n, n)
        Symmetric matrix whose lower (and upper) triangle matches ``v``.

    Examples
    --------
    >>> import numpy as np
    >>> unvech(np.array([1., 2., 3.]), 2)
    array([[1., 2.],
           [2., 3.]])
    """
    rows, cols = vech_indices(n)
    A = np.zeros((n, n), dtype=v.dtype)
    A[rows, cols] = v
    A[cols, rows] = v  # mirror to upper triangle
    return A


# ---------------------------------------------------------------------------
# SPD utilities
# ---------------------------------------------------------------------------


def make_symmetric(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix: (A + A^T) / 2.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Square matrix, possibly slightly non-symmetric due to floating-point
        rounding.

    Returns
    -------
    np.ndarray of shape (n, n)
        Symmetrized matrix.
    """
    return (A + A.T) * 0.5


def is_spd(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check whether a matrix is symmetric positive-definite (SPD).

    The test first verifies symmetry up to ``tol``, then attempts a
    Cholesky decomposition; a ``LinAlgError`` indicates the matrix is not
    positive-definite.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Matrix to test.
    tol : float, optional
        Absolute tolerance for the symmetry check.  Default is 1e-10.

    Returns
    -------
    bool
        ``True`` if A is SPD, ``False`` otherwise.
    """
    if not np.allclose(A, A.T, atol=tol):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_spd(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Project a matrix to the nearest symmetric positive-definite matrix.

    Uses eigenvalue clipping (Higham 1988 variant): symmetrize, compute the
    eigendecomposition via ``scipy.linalg.eigh``, clip all eigenvalues below
    ``tol`` up to ``tol``, and reconstruct.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Input matrix (need not be symmetric or positive-definite).
    tol : float, optional
        Minimum eigenvalue threshold.  Default is 1e-8.

    Returns
    -------
    np.ndarray of shape (n, n)
        Nearest SPD matrix in the Frobenius-norm sense.

    Notes
    -----
    ``scipy.linalg.eigh`` is used rather than ``np.linalg.eig`` because it
    exploits symmetry, guarantees real eigenvalues, and is numerically more
    stable for covariance matrices.
    """
    A_sym = make_symmetric(A)
    eigvals, eigvecs = scipy.linalg.eigh(A_sym)
    eigvals_clipped = np.maximum(eigvals, tol)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


# ---------------------------------------------------------------------------
# Volatility helpers
# ---------------------------------------------------------------------------


def annualize_vol(vol: float, freq: int = 252) -> float:
    """Annualize a daily volatility estimate.

    Parameters
    ----------
    vol : float
        Daily volatility (standard deviation).
    freq : int, optional
        Number of trading days per year.  Default is 252.

    Returns
    -------
    float
        Annualized volatility: ``vol * sqrt(freq)``.
    """
    return vol * np.sqrt(freq)


# ---------------------------------------------------------------------------
# Rolling matrix mean
# ---------------------------------------------------------------------------


def rolling_mean_matrix(matrices: List[np.ndarray], window: int) -> np.ndarray:
    """Compute the element-wise mean of the last ``window`` matrices in a list.

    Parameters
    ----------
    matrices : list of np.ndarray, each of shape (n, n)
        Ordered sequence of matrices (e.g. daily realized covariances).
    window : int
        Number of most-recent matrices to average.  If ``len(matrices) <
        window``, all available matrices are used.

    Returns
    -------
    np.ndarray of shape (n, n)
        Element-wise mean of the selected matrices.

    Notes
    -----
    Used to construct the weekly (window=5) and monthly (window=22) HAR
    components of the HARQ and HARQ-DRD models.
    """
    subset = matrices[-window:]
    return np.stack(subset, axis=0).mean(axis=0)

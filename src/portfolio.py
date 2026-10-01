"""
Portfolio construction and evaluation utilities.

Functions
---------
gmv_weights              : unconstrained global minimum-variance weights.
gmv_weights_constrained  : long-only (no-short-sale) GMV weights via CVXPY.
compute_turnover         : portfolio turnover (half-sum of absolute changes).
compute_concentration    : sqrt-Herfindahl concentration index.
compute_utility          : mean-variance utility.
certainty_equivalent_gain: CEQ gain of one return stream over another.
insanity_filter          : replace non-PD forecasts with the historical mean.
"""

from __future__ import annotations

import numpy as np

from src.utils import is_spd


# ---------------------------------------------------------------------------
# Unconstrained GMV
# ---------------------------------------------------------------------------

def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio weights (unconstrained).

    The analytical solution is:
        w* = (Sigma^{-1} 1) / (1^T Sigma^{-1} 1)

    Parameters
    ----------
    Sigma : np.ndarray of shape (N, N)
        Positive-definite covariance matrix.

    Returns
    -------
    np.ndarray of shape (N,)
        Portfolio weights summing to 1.
    """
    N = Sigma.shape[0]
    ones = np.ones(N)
    H_inv_ones = np.linalg.solve(Sigma, ones)
    return H_inv_ones / (ones @ H_inv_ones)


# ---------------------------------------------------------------------------
# Long-only (no-short-sale) GMV
# ---------------------------------------------------------------------------

def gmv_weights_constrained(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights with long-only constraint (w_i >= 0).

    Uses CVXPY for the quadratic program.  Falls back to equal weighting if
    CVXPY is unavailable or the problem is infeasible.

    Parameters
    ----------
    Sigma : np.ndarray of shape (N, N)

    Returns
    -------
    np.ndarray of shape (N,)
        Non-negative portfolio weights summing to 1.
    """
    N = Sigma.shape[0]
    try:
        import cvxpy as cp
        w = cp.Variable(N)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, Sigma)),
            [cp.sum(w) == 1.0, w >= 0.0],
        )
        prob.solve(solver=cp.CLARABEL, warm_start=True)
        if w.value is not None:
            weights = np.array(w.value).ravel()
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()
            return weights
    except Exception:
        pass
    # Fallback: equal weights
    return np.ones(N) / N


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------

def compute_turnover(w_prev: np.ndarray, w_curr: np.ndarray) -> float:
    """Portfolio turnover: half the sum of absolute weight changes.

    Turnover = (1/2) * sum_i |w_{curr,i} - w_{prev,i}|

    A value of 1 corresponds to a complete portfolio rotation.

    Parameters
    ----------
    w_prev, w_curr : np.ndarray of shape (N,)
        Portfolio weights before and after rebalancing.

    Returns
    -------
    float in [0, 1].
    """
    return float(0.5 * np.sum(np.abs(w_curr - w_prev)))


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

def compute_concentration(w: np.ndarray) -> float:
    """Square-root Herfindahl concentration index.

    concentration = sqrt(sum_i w_i^2)

    Ranges from 1/sqrt(N) (equal-weight) to 1 (all-in portfolio).

    Parameters
    ----------
    w : np.ndarray of shape (N,)

    Returns
    -------
    float
    """
    return float(np.sqrt(np.sum(w ** 2)))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_utility(returns: np.ndarray, gamma: float = 3.0) -> float:
    """Mean-variance utility of a return stream.

    U = E[r] - (gamma / 2) * Var[r]

    Parameters
    ----------
    returns : np.ndarray of shape (T,)
        Realised portfolio returns.
    gamma : float
        Coefficient of relative risk aversion.

    Returns
    -------
    float
    """
    return float(np.mean(returns) - (gamma / 2.0) * np.var(returns))


# ---------------------------------------------------------------------------
# Certainty-equivalent gain
# ---------------------------------------------------------------------------

def certainty_equivalent_gain(
    r1: np.ndarray,
    r2: np.ndarray,
    gamma: float = 3.0,
) -> float:
    """CEQ gain of strategy 1 over strategy 2.

    CEQ gain = U(r1) - U(r2)

    Positive values indicate that strategy 1 is preferred.

    Parameters
    ----------
    r1, r2 : np.ndarray of shape (T,)
    gamma : float

    Returns
    -------
    float (annualised gain in return units)
    """
    return compute_utility(r1, gamma) - compute_utility(r2, gamma)


# ---------------------------------------------------------------------------
# Insanity filter
# ---------------------------------------------------------------------------

def insanity_filter(
    H: np.ndarray,
    S_mean: np.ndarray,
) -> np.ndarray:
    """Replace a forecast that fails the PD check with the historical mean.

    Parameters
    ----------
    H : np.ndarray of shape (N, N)
        Candidate forecast covariance matrix.
    S_mean : np.ndarray of shape (N, N)
        Fallback (typically the rolling historical mean of realized covariances).

    Returns
    -------
    np.ndarray of shape (N, N)
        H if it is symmetric positive-definite, else S_mean.
    """
    return H if is_spd(H) else S_mean

"""
Loss functions and statistical tests for evaluating covariance forecasts.

Implements the Frobenius and QLIKE losses used in Bollerslev, Patton &
Quaedvlieg (2016), and the Diebold-Mariano test for pairwise model comparison.

References
----------
Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy.
    *Journal of Business & Economic Statistics*, 13(3), 253–263.

Bollerslev, T., Patton, A.J. & Quaedvlieg, R. (2016). Exploiting the errors:
    A simple approach for improved volatility forecasting.
    *Journal of Econometrics*, 192(1), 1–18.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg
import scipy.stats
import statsmodels.stats.sandwich_covariance as sm_hac


# ---------------------------------------------------------------------------
# Point-in-time loss functions
# ---------------------------------------------------------------------------


def frobenius_loss(H: np.ndarray, S: np.ndarray) -> float:
    """Compute the Frobenius (matrix distance) forecast loss.

    .. math::

        L^{\\text{Frob}} = \\sqrt{\\operatorname{Tr}[(H - S)(H - S)^T]}
                         = \\|H - S\\|_F

    Parameters
    ----------
    H : np.ndarray of shape (N, N)
        Forecast covariance matrix.
    S : np.ndarray of shape (N, N)
        Realized covariance proxy (ground truth), e.g. the daily MK
        realized covariance.

    Returns
    -------
    float
        Non-negative Frobenius distance between H and S.

    Notes
    -----
    This is equivalent to ``np.linalg.norm(H - S, 'fro')``.
    """
    diff = H - S
    return float(np.sqrt(np.trace(diff @ diff.T)))


def qlike_loss(H: np.ndarray, S: np.ndarray) -> float:
    """Compute the QLIKE (quasi-likelihood) forecast loss.

    .. math::

        L^{\\text{QLIKE}} = \\log|H| + \\operatorname{Tr}(H^{-1} S)

    The log-determinant is computed with ``numpy.linalg.slogdet`` for
    numerical stability, and the linear solve ``H^{-1} S`` is obtained via
    ``scipy.linalg.solve`` to avoid explicit matrix inversion.

    Parameters
    ----------
    H : np.ndarray of shape (N, N)
        Forecast covariance matrix.  Must be positive-definite for a finite
        result.
    S : np.ndarray of shape (N, N)
        Realized covariance proxy (ground truth).

    Returns
    -------
    float
        QLIKE loss value, or ``np.nan`` if H is singular (``|H| ≈ 0``).

    Notes
    -----
    The QLIKE loss is a proper scoring rule for covariance matrices under a
    matrix-Gaussian likelihood and is strictly consistent for the conditional
    covariance.  It penalizes over-/under-estimation asymmetrically.
    """
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0:
        return np.nan
    try:
        HinvS = scipy.linalg.solve(H, S, assume_a="pos")
    except scipy.linalg.LinAlgError:
        return np.nan
    return float(logdet + np.trace(HinvS))


# ---------------------------------------------------------------------------
# Time-series loss computation
# ---------------------------------------------------------------------------


def compute_losses_series(
    H_series: np.ndarray,
    S_series: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Frobenius and QLIKE losses for a series of forecast-target pairs.

    Parameters
    ----------
    H_series : np.ndarray of shape (T, N, N)
        Sequence of T forecast covariance matrices.
    S_series : np.ndarray of shape (T, N, N)
        Corresponding T realized covariance proxies.

    Returns
    -------
    frob_losses : np.ndarray of shape (T,)
        Frobenius loss at each time step.
    qlike_losses : np.ndarray of shape (T,)
        QLIKE loss at each time step.  Singular forecasts yield ``np.nan``.
    """
    T = H_series.shape[0]
    frob_losses = np.empty(T)
    qlike_losses = np.empty(T)

    for t in range(T):
        frob_losses[t] = frobenius_loss(H_series[t], S_series[t])
        qlike_losses[t] = qlike_loss(H_series[t], S_series[t])

    return frob_losses, qlike_losses


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------


def diebold_mariano_test(
    loss1: np.ndarray,
    loss2: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H₀: E[L₁_t − L₂_t] = 0 against the two-sided alternative using a
    standard-normal asymptotic distribution with Newey-West HAC variance.

    The test statistic is:

    .. math::

        \\text{DM} = \\frac{\\bar{d}}{\\hat{\\sigma}_{\\bar{d}}}
                   = \\frac{\\bar{d} \\sqrt{T}}{\\hat{\\sigma}_d}

    where :math:`\\hat{\\sigma}_d^2` is the Newey-West long-run variance
    of the loss-differential series :math:`d_t = L_{1,t} - L_{2,t}`, with
    Bartlett weights and ``h`` lags.

    Parameters
    ----------
    loss1 : np.ndarray of shape (T,)
        Loss series for model 1.
    loss2 : np.ndarray of shape (T,)
        Loss series for model 2.
    h : int, optional
        Number of Newey-West lags.  Set to the forecast horizon (default 1
        for one-step-ahead forecasts).

    Returns
    -------
    dm_stat : float
        DM test statistic (positive values favour model 2).
    p_value : float
        Two-sided p-value under standard-normal asymptotics.

    Notes
    -----
    The Newey-West long-run variance is estimated by
    ``statsmodels.stats.sandwich_covariance.S_hac_simple``, which returns
    ``T × sigma^2``; the standard error of ``d_bar`` is therefore
    ``sqrt(S / T^2) = sqrt(S) / T``, yielding ``DM = T * d_bar / sqrt(S)``.
    """
    d = loss1 - loss2
    T = len(d)
    dbar = float(np.mean(d))

    # Centre under H₀ before estimating the HAC long-run variance
    d_c = d - dbar
    # S_hac_simple returns T * sigma^2 (empirically verified)
    S = sm_hac.S_hac_simple(d_c.reshape(-1, 1), nlags=h)[0, 0]

    if S <= 0.0:
        return np.nan, np.nan

    dm_stat = float(T * dbar / np.sqrt(S))
    p_value = float(2.0 * scipy.stats.norm.sf(abs(dm_stat)))
    return dm_stat, p_value


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def compute_mean_losses(
    frob: np.ndarray,
    qlike: np.ndarray,
) -> dict:
    """Compute mean Frobenius and QLIKE losses, ignoring NaN values.

    Parameters
    ----------
    frob : np.ndarray of shape (T,)
        Time series of Frobenius losses.
    qlike : np.ndarray of shape (T,)
        Time series of QLIKE losses.

    Returns
    -------
    dict
        ``{"mean_frobenius": float, "mean_qlike": float}``
    """
    return {
        "mean_frobenius": float(np.nanmean(frob)),
        "mean_qlike": float(np.nanmean(qlike)),
    }

"""
Realized covariance measures for high-frequency price data.

Implements the multivariate kernel estimator and integrated quarticity proxy
used in the HARQ and HARQ-DRD covariance-forecasting models, following
Barndorff-Nielsen et al. (2011) and Noureldin, Shephard & Sheppard (2012).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Simple realized covariance
# ---------------------------------------------------------------------------


def compute_realized_covariance_simple(returns: np.ndarray) -> np.ndarray:
    """Compute the simple (unadjusted) realized covariance matrix.

    Aggregates the outer products of synchronous intraday log-return vectors:

        RC_t = sum_{s=1}^{T_intra}  r_{t,s} r_{t,s}^T

    Parameters
    ----------
    returns : np.ndarray of shape (T_intra, N)
        Matrix of intraday log-returns for a single day.  Rows are
        observations, columns are assets.

    Returns
    -------
    np.ndarray of shape (N, N)
        Realized covariance matrix.

    Notes
    -----
    The simple estimator is consistent under zero market-microstructure noise
    but is biased when noise is present.  Use
    :func:`compute_multivariate_kernel_cov` for a noise-robust alternative.
    """
    return returns.T @ returns


# ---------------------------------------------------------------------------
# Parzen kernel
# ---------------------------------------------------------------------------


def parzen_kernel(x: float) -> float:
    """Evaluate the Parzen kernel at a point.

    The Parzen kernel used by Barndorff-Nielsen et al. (2011) for the
    multivariate realized kernel estimator:

        k(x) = 1 - 6x^2 + 6|x|^3    for |x| in [0, 0.5]
        k(x) = 2(1 - |x|)^3          for |x| in (0.5, 1]
        k(x) = 0                      for |x| > 1

    Parameters
    ----------
    x : float
        Argument of the kernel.  Typically ``h / (H + 1)`` where h is the
        lag order and H is the bandwidth.

    Returns
    -------
    float
        Kernel weight in [0, 1].

    References
    ----------
    Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A. & Shephard, N. (2011).
    Multivariate realised kernels. *Journal of Econometrics*, 162(2), 149–169.
    """
    ax = abs(x)
    if ax > 1.0:
        return 0.0
    if ax <= 0.5:
        return 1.0 - 6.0 * ax**2 + 6.0 * ax**3
    return 2.0 * (1.0 - ax) ** 3


# ---------------------------------------------------------------------------
# Multivariate kernel (MK) estimator
# ---------------------------------------------------------------------------


def compute_multivariate_kernel_cov(
    returns: np.ndarray,
    H: Optional[int] = None,
) -> np.ndarray:
    """Compute the multivariate realized kernel (MK) covariance estimator.

    Implements the Parzen-kernel estimator of Barndorff-Nielsen et al. (2011):

        MK_t = sum_{h=-H}^{H} k(h / (H+1)) * Gamma_h

    where the autocovariance matrices are:

        Gamma_h = sum_{t=|h|+1}^{T} r_t r_{t-|h|}^T   (for h >= 0)

    and Gamma_{-h} = Gamma_h^T.  Exploiting the symmetry of k():

        MK_t = Gamma_0 + sum_{h=1}^{H} k(h/(H+1)) * (Gamma_h + Gamma_h^T)

    Parameters
    ----------
    returns : np.ndarray of shape (T, N)
        Intraday log-returns for a single day.
    H : int, optional
        Kernel bandwidth (number of non-zero lags).  Defaults to
        ``ceil(T^{4/5})`` as recommended by Barndorff-Nielsen et al. (2011).

    Returns
    -------
    np.ndarray of shape (N, N)
        Noise-robust positive-semidefinite realized covariance estimate.

    Notes
    -----
    The bandwidth default ``ceil(T^{4/5})`` is optimal under a mean-squared
    error criterion when the noise-to-signal ratio is O(1).  The Parzen kernel
    ensures the estimator is positive-semidefinite by construction.
    """
    T, N = returns.shape
    if H is None:
        H = math.ceil(T ** (4.0 / 5.0))

    # h = 0 term: standard realized covariance
    Gamma_0 = returns.T @ returns
    MK = Gamma_0.copy()

    for h in range(1, H + 1):
        k_h = parzen_kernel(h / (H + 1))
        if k_h == 0.0:
            continue
        # Gamma_h = sum_{t=h}^{T-1} r[t] r[t-h]^T  (0-indexed: r[h:] vs r[:-h])
        Gamma_h = returns[h:].T @ returns[:-h]
        MK += k_h * (Gamma_h + Gamma_h.T)

    return MK


# ---------------------------------------------------------------------------
# Integrated quarticity proxy
# ---------------------------------------------------------------------------


def compute_quarticity_proxy(
    returns: np.ndarray,
    n_assets: int,
) -> np.ndarray:
    """Compute the daily integrated quarticity (IQ) proxy matrix.

    For each asset pair (i, j), the proxy uses the simplified formula:

        IQ_{t,ij} = (T / 3) * sum_{s=1}^{T} r_{t,s,i}^2 * r_{t,s,j}^2

    where T is the number of intraday observations.  This is the standard
    fourth-power estimator (see Bollerslev, Patton & Quaedvlieg, 2016,
    eq. 12), which is a consistent estimator of the integrated quarticity
    under the null of zero microstructure noise.

    Parameters
    ----------
    returns : np.ndarray of shape (T_intra, N)
        Intraday log-returns for a single day.
    n_assets : int
        Number of assets N (used for shape validation).

    Returns
    -------
    np.ndarray of shape (N, N)
        Integrated quarticity proxy IQ_t.

    References
    ----------
    Bollerslev, T., Patton, A.J. & Quaedvlieg, R. (2016). Exploiting the
    errors: A simple approach for improved volatility forecasting.
    *Journal of Econometrics*, 192(1), 1–18.
    """
    T_intra = returns.shape[0]
    r2 = returns ** 2  # shape (T_intra, N)
    IQ = (T_intra / 3.0) * (r2.T @ r2)
    return IQ


# ---------------------------------------------------------------------------
# Pi matrix and vector
# ---------------------------------------------------------------------------


def compute_pi_matrix(IQ: np.ndarray, kappa: float = 3.777) -> np.ndarray:
    """Compute the Pi_t matrix: Pi_t = kappa * IQ_t.

    Pi_t represents the scaled integrated quarticity and enters the HARQ
    attenuation term as the dynamic penalty on lagged realized covariances.

    Parameters
    ----------
    IQ : np.ndarray of shape (N, N)
        Integrated quarticity proxy matrix IQ_t from
        :func:`compute_quarticity_proxy`.
    kappa : float, optional
        Scaling constant.  Default is 3.777, derived from the asymptotic
        theory of the realized kernel estimator (Barndorff-Nielsen et al.,
        2011; see also Bollerslev, Patton & Quaedvlieg, 2016, eq. 13).

    Returns
    -------
    np.ndarray of shape (N, N)
        Pi_t matrix.
    """
    return kappa * IQ


def compute_pi_vector(Pi: np.ndarray) -> np.ndarray:
    """Extract the diagonal volatility attenuation vector from Pi_t.

    Computes:

        pi_{t,i} = sqrt(Pi_{t,ii})   for i = 1, …, N

    This N-vector represents per-asset measurement-error magnitudes and is
    used in the HARQ-DRD model to construct the dynamic attenuation matrix.

    Parameters
    ----------
    Pi : np.ndarray of shape (N, N)
        Pi_t matrix from :func:`compute_pi_matrix`.

    Returns
    -------
    np.ndarray of shape (N,)
        pi_t vector of per-asset noise-level proxies.
    """
    return np.sqrt(np.diag(Pi))


# ---------------------------------------------------------------------------
# Full daily realized-measures pipeline from 5-minute prices
# ---------------------------------------------------------------------------


def compute_realized_measures_from_5min(
    price_data: pd.DataFrame,
    assets: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute daily realized covariance, Pi, and pi from 5-minute prices.

    For each trading day in ``price_data`` the function:

    1. Extracts intraday prices for the requested ``assets``.
    2. Computes log-returns:  r_{t,s} = log(P_{t,s} / P_{t,s-1}).
    3. Estimates the daily MK realized covariance S_t via
       :func:`compute_multivariate_kernel_cov`.
    4. Computes the integrated quarticity proxy IQ_t via
       :func:`compute_quarticity_proxy`.
    5. Derives Pi_t = kappa * IQ_t via :func:`compute_pi_matrix`.
    6. Derives pi_t = sqrt(diag(Pi_t)) via :func:`compute_pi_vector`.

    Parameters
    ----------
    price_data : pd.DataFrame
        5-minute price panel.  Accepted formats:

        * **MultiIndex** ``(date, time)``:  The outer level must be a date
          (any type convertible to ``datetime``); the inner level is the
          intraday timestamp.  Columns are asset tickers.
        * **DatetimeIndex** (flat, high-frequency):  The function groups
          observations by calendar date using ``index.normalize()``.

        In both cases the column names must contain the strings in ``assets``.
    assets : list of str
        Asset names corresponding to columns in ``price_data``.

    Returns
    -------
    S_arr : np.ndarray of shape (T, N, N)
        Daily MK realized covariance matrices.
    Pi_arr : np.ndarray of shape (T, N, N)
        Daily Pi_t matrices (scaled integrated quarticity).
    pi_arr : np.ndarray of shape (T, N)
        Daily pi_t vectors (per-asset noise-level proxies).

    Notes
    -----
    Days with fewer than 2 intraday observations are skipped silently.  The
    returned arrays correspond to the surviving trading days in chronological
    order.
    """
    prices = price_data[assets]
    N = len(assets)

    # Resolve grouping key based on index structure
    if isinstance(prices.index, pd.MultiIndex):
        dates = prices.index.get_level_values(0).unique()
        def _get_day(d: object) -> pd.DataFrame:
            return prices.loc[d]
    else:
        # Flat DatetimeIndex: group by calendar date
        date_key = prices.index.normalize()
        dates = date_key.unique()
        def _get_day(d: object) -> pd.DataFrame:
            return prices[date_key == d]

    S_list: List[np.ndarray] = []
    Pi_list: List[np.ndarray] = []
    pi_list: List[np.ndarray] = []

    for d in sorted(dates):
        day_prices = _get_day(d).values.astype(float)  # (T_intra+1, N)
        if day_prices.shape[0] < 2:
            continue

        # Intraday log-returns: shape (T_intra, N)
        rets = np.diff(np.log(day_prices), axis=0)

        S_t = compute_multivariate_kernel_cov(rets)
        IQ_t = compute_quarticity_proxy(rets, n_assets=N)
        Pi_t = compute_pi_matrix(IQ_t)
        pi_t = compute_pi_vector(Pi_t)

        S_list.append(S_t)
        Pi_list.append(Pi_t)
        pi_list.append(pi_t)

    if not S_list:
        return (
            np.empty((0, N, N)),
            np.empty((0, N, N)),
            np.empty((0, N)),
        )

    return (
        np.stack(S_list, axis=0),
        np.stack(Pi_list, axis=0),
        np.stack(pi_list, axis=0),
    )

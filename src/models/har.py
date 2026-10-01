"""
HAR (Heterogeneous Autoregressive) and HARQ (HAR with noise attenuation)
models for covariance matrix forecasting.

Both models operate on the vech representation of realized covariances and
run element-wise OLS regressions across the K = N(N+1)/2 unique entries.

Parameter layouts
-----------------
HAR  : (K, 4)  — [intercept, beta_d, beta_w, beta_m]
HARQ : (K, 5)  — [intercept, theta_1q, beta_d, beta_w, beta_m]
         col 1 (theta_1q) is the noise-attenuation coefficient on pi_t * s_{d,t}.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_har_features(
    s_d: np.ndarray,
    s_w: np.ndarray,
    s_m: np.ndarray,
) -> np.ndarray:
    """Stack HAR regressors into a (K, 4) feature matrix.

    Parameters
    ----------
    s_d, s_w, s_m : np.ndarray of shape (K,)
        Daily, weekly, and monthly vech components.

    Returns
    -------
    np.ndarray of shape (K, 4)
        Columns: [1, s_d_k, s_w_k, s_m_k] for each vech element k.
    """
    K = s_d.shape[0]
    return np.column_stack([np.ones(K), s_d, s_w, s_m])


# ---------------------------------------------------------------------------
# HAR — element-wise OLS
# ---------------------------------------------------------------------------

def _build_xy(S_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct the (n_obs, K, 4) design matrix and (n_obs, K) response.

    Uses a rolling window that starts once 22 lags are available.
    """
    T, K = S_array.shape
    X_list, y_list = [], []
    for t in range(22, T - 1):
        s_d = S_array[t]
        s_w = S_array[t - 4 : t + 1].mean(axis=0)
        s_m = S_array[t - 21 : t + 1].mean(axis=0)
        X_list.append(build_har_features(s_d, s_w, s_m))   # (K, 4)
        y_list.append(S_array[t + 1])                        # (K,)
    return np.array(X_list), np.array(y_list)                # (n, K, 4), (n, K)


def fit_har_ols(S_array: np.ndarray) -> np.ndarray:
    """Fit element-wise HAR model via OLS.

    Parameters
    ----------
    S_array : np.ndarray of shape (T, K)
        Time series of vech'd realized covariances.

    Returns
    -------
    params : np.ndarray of shape (K, 4)
        OLS coefficients [intercept, beta_d, beta_w, beta_m] per element.
    """
    T, K = S_array.shape
    X_all, y_all = _build_xy(S_array)   # (n, K, 4), (n, K)
    params = np.zeros((K, 4))
    for k in range(K):
        Xk, yk = X_all[:, k, :], y_all[:, k]
        # Drop rows with NaN (can arise from near-zero synthetic variances after log)
        valid = np.isfinite(Xk).all(axis=1) & np.isfinite(yk)
        if valid.sum() < 5:          # degenerate window — return zeros
            continue
        params[k] = np.linalg.lstsq(Xk[valid], yk[valid], rcond=None)[0]
    return params


def forecast_har(params: np.ndarray, features: np.ndarray) -> np.ndarray:
    """One-step-ahead HAR forecast.

    Parameters
    ----------
    params : np.ndarray of shape (K, 4)
    features : np.ndarray of shape (K, 4)
        From :func:`build_har_features`.

    Returns
    -------
    np.ndarray of shape (K,)
        Vech of the forecast covariance.
    """
    return (params * features).sum(axis=1)


# ---------------------------------------------------------------------------
# HARQ — adds noise-attenuation term  theta_1q * pi_t * s_{d,t}
# ---------------------------------------------------------------------------

def _build_xy_harq(
    S_array: np.ndarray,
    Pi_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct HARQ design matrix (n_obs, K, 5) and response (n_obs, K)."""
    T, K = S_array.shape
    X_list, y_list = [], []
    for t in range(22, T - 1):
        s_d = S_array[t]
        s_w = S_array[t - 4 : t + 1].mean(axis=0)
        s_m = S_array[t - 21 : t + 1].mean(axis=0)
        pi_d = Pi_array[t]
        # Cols: [1, pi_d * s_d, s_d, s_w, s_m]
        feats = np.column_stack([np.ones(K), pi_d * s_d, s_d, s_w, s_m])
        X_list.append(feats)
        y_list.append(S_array[t + 1])
    return np.array(X_list), np.array(y_list)


def fit_harq_ols(S_array: np.ndarray, Pi_array: np.ndarray) -> np.ndarray:
    """Fit element-wise HARQ model via OLS.

    The HARQ model extends HAR with a noise-attenuation term:
        s_{t+1,k} = c_k + theta_1q_k * pi_{t,k} * s_{d,t,k}
                    + beta_d_k * s_{d,t,k} + beta_w_k * s_{w,t,k}
                    + beta_m_k * s_{m,t,k} + eps_{t,k}

    Parameters
    ----------
    S_array : np.ndarray of shape (T, K)
    Pi_array : np.ndarray of shape (T, K)
        Noise-proxy vectors; each element is the realized quarticity proxy
        for the corresponding vech position.

    Returns
    -------
    params : np.ndarray of shape (K, 5)
        Columns: [intercept, theta_1q, beta_d, beta_w, beta_m].
    """
    T, K = S_array.shape
    X_all, y_all = _build_xy_harq(S_array, Pi_array)   # (n, K, 5), (n, K)
    params = np.zeros((K, 5))
    for k in range(K):
        Xk, yk = X_all[:, k, :], y_all[:, k]
        valid = np.isfinite(Xk).all(axis=1) & np.isfinite(yk)
        if valid.sum() < 6:
            continue
        params[k] = np.linalg.lstsq(Xk[valid], yk[valid], rcond=None)[0]
    return params


def forecast_harq(
    params: np.ndarray,
    features: np.ndarray,
    pi_d: np.ndarray,
) -> np.ndarray:
    """One-step-ahead HARQ forecast.

    Parameters
    ----------
    params : np.ndarray of shape (K, 5)
    features : np.ndarray of shape (K, 4)
        HAR features [1, s_d, s_w, s_m] from :func:`build_har_features`.
    pi_d : np.ndarray of shape (K,)
        Today's noise-proxy vector.

    Returns
    -------
    np.ndarray of shape (K,)
    """
    s_d = features[:, 1]
    s_w = features[:, 2]
    s_m = features[:, 3]
    K = s_d.shape[0]
    harq_feats = np.column_stack([np.ones(K), pi_d * s_d, s_d, s_w, s_m])
    return (params * harq_feats).sum(axis=1)

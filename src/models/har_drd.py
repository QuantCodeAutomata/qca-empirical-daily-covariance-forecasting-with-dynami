"""
HAR-DRD and HARQ-DRD models: HAR/HARQ applied to the DRD decomposition.

The DRD (Diagonal × Correlation × Diagonal) factorisation writes any
covariance matrix as S = D R D, where D = diag(sqrt(S_ii)) and R is the
correlation matrix.  In log-space / Fisher-space the components are closer
to Gaussian, which improves OLS performance.

The module exposes low-level decompose/compose helpers as well as
high-level fit/forecast functions that wrap the HAR machinery.
"""

from __future__ import annotations

import numpy as np

from src.models.har import fit_har_ols, forecast_har, fit_harq_ols, forecast_harq, build_har_features
from src.utils import unvech, nearest_spd


# ---------------------------------------------------------------------------
# DRD helpers
# ---------------------------------------------------------------------------

def drd_decompose(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a covariance matrix S = D R D.

    Parameters
    ----------
    S : np.ndarray of shape (N, N)
        Symmetric positive-definite covariance matrix.

    Returns
    -------
    D : np.ndarray of shape (N, N)
        Diagonal matrix of standard deviations: diag(sqrt(S_ii)).
    R : np.ndarray of shape (N, N)
        Correlation matrix with unit diagonal.
    """
    _MIN_VAR = 1e-12
    std = np.sqrt(np.clip(np.diag(S), _MIN_VAR, None))
    D = np.diag(std)
    D_inv = np.diag(1.0 / std)
    R = D_inv @ S @ D_inv
    # Clip diagonal to exactly 1 to fix rounding
    np.fill_diagonal(R, 1.0)
    return D, R


def drd_compose(D: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Reconstruct covariance S = D R D from D and R.

    Parameters
    ----------
    D : np.ndarray of shape (N, N)
        Diagonal matrix of standard deviations.
    R : np.ndarray of shape (N, N)
        Correlation matrix.

    Returns
    -------
    np.ndarray of shape (N, N)
    """
    return D @ R @ D


# ---------------------------------------------------------------------------
# High-level fit / forecast (wrap scalar HAR on DRD components)
# ---------------------------------------------------------------------------

def _log_vol_series(S_list: list[np.ndarray]) -> np.ndarray:
    """Extract log-volatility (log std-dev) series from a list of (N,N) matrices.

    Diagonal entries are clipped to a small positive floor before log to guard
    against numerical noise in synthetic covariance matrices.
    """
    _MIN_VAR = 1e-12
    return np.array([
        np.log(np.sqrt(np.clip(np.diag(S), _MIN_VAR, None)))
        for S in S_list
    ])   # (T, N)


def _fisher_corr_series(S_list: list[np.ndarray]) -> np.ndarray:
    """Extract Fisher-transformed off-diagonal correlations."""
    rows = []
    for S in S_list:
        D, R = drd_decompose(S)
        N = R.shape[0]
        idx_l, idx_c = np.tril_indices(N, k=-1)
        rows.append(np.arctanh(np.clip(R[idx_l, idx_c], -0.9999, 0.9999)))
    return np.array(rows)    # (T, N*(N-1)//2)


def _infer_n(K: int) -> int:
    """Infer asset dimension N from vech length K = N*(N+1)//2."""
    N = int((-1 + np.sqrt(1 + 8 * K)) / 2)
    assert N * (N + 1) // 2 == K, f"Invalid vech length {K}"
    return N


def fit_har_drd(S_array_vech: np.ndarray, N: int | None = None) -> dict:
    """Fit HAR-DRD on vech'd covariance series.

    Runs separate HAR models on log-volatilities and Fisher-transformed
    correlations.

    Parameters
    ----------
    S_array_vech : np.ndarray of shape (T, K)
    N : int, optional — asset dimension; inferred from K when omitted.

    Returns
    -------
    dict with keys 'params_vol' (N, 4) and 'params_corr' (M, 4)
    where M = N*(N-1)//2.
    """
    if N is None:
        N = _infer_n(S_array_vech.shape[1])
    S_list = [unvech(S_array_vech[t], N) for t in range(len(S_array_vech))]
    log_vols = _log_vol_series(S_list)
    fish_corr = _fisher_corr_series(S_list)
    params_vol = fit_har_ols(log_vols)
    params_corr = fit_har_ols(fish_corr) if fish_corr.shape[1] > 0 else np.zeros((0, 4))
    return {"params_vol": params_vol, "params_corr": params_corr, "N": N}


def forecast_har_drd(params: dict, S_array_vech: np.ndarray) -> np.ndarray:
    """Forecast one step ahead using HAR-DRD.

    Parameters
    ----------
    params : dict  — from :func:`fit_har_drd`.
    S_array_vech : np.ndarray of shape (T, K) — recent vech series (at least 22 rows).

    Returns
    -------
    np.ndarray of shape (N, N) — forecast covariance.
    """
    N = params["N"]
    S_list = [unvech(S_array_vech[t], N) for t in range(len(S_array_vech))]

    log_vols = _log_vol_series(S_list)
    fish_corr = _fisher_corr_series(S_list)

    def _har_fc(series):
        s_d = series[-1]
        s_w = series[-5:].mean(axis=0)
        s_m = series[-22:].mean(axis=0)
        feats = build_har_features(s_d, s_w, s_m)
        p_key = "params_vol" if series.shape[1] == N else "params_corr"
        return forecast_har(params[p_key], feats)

    lv_hat = _har_fc(log_vols)
    std_hat = np.exp(lv_hat)
    D_hat = np.diag(std_hat)

    if fish_corr.shape[1] > 0:
        fc_hat = _har_fc(fish_corr)
        rho_hat = np.tanh(fc_hat)
        R_hat = np.eye(N)
        idx_l, idx_c = np.tril_indices(N, k=-1)
        R_hat[idx_l, idx_c] = rho_hat
        R_hat[idx_c, idx_l] = rho_hat
    else:
        R_hat = np.eye(N)

    return nearest_spd(D_hat @ R_hat @ D_hat)


def fit_harq_drd(S_array_vech: np.ndarray, Pi_array: np.ndarray,
                  N: int | None = None) -> dict:
    """Fit HARQ-DRD on vech'd covariance series.

    Parameters
    ----------
    S_array_vech : np.ndarray of shape (T, K)
    Pi_array : np.ndarray of shape (T, K)
    N : int, optional — asset dimension; inferred from K when omitted.

    Returns
    -------
    dict with keys 'params_vol', 'params_corr', 'N'.
    """
    if N is None:
        N = _infer_n(S_array_vech.shape[1])
    S_list = [unvech(S_array_vech[t], N) for t in range(len(S_array_vech))]
    log_vols = _log_vol_series(S_list)
    fish_corr = _fisher_corr_series(S_list)

    # Map pi_array columns to vol / corr subsets
    # Diagonal vech positions: (i,i) is at index (i+1)*(i+2)//2 - 1 in row-major lower-triangular
    diag_idx = np.array([(i + 1) * (i + 2) // 2 - 1 for i in range(N)])
    pi_vol = Pi_array[:, diag_idx]

    off_count = N * (N - 1) // 2
    off_mask = np.ones(Pi_array.shape[1], dtype=bool)
    off_mask[diag_idx] = False
    pi_corr = Pi_array[:, off_mask]

    params_vol = fit_harq_ols(log_vols, pi_vol)
    # Correlations use plain HAR per paper: "model correlations with scalar HAR equations only, not HARQ"
    params_corr = (fit_har_ols(fish_corr)
                   if fish_corr.shape[1] > 0 else np.zeros((0, 4)))
    return {"params_vol": params_vol, "params_corr": params_corr, "N": N}


def forecast_harq_drd(
    params: dict,
    S_array_vech: np.ndarray,
    pi_d: np.ndarray,
) -> np.ndarray:
    """Forecast one step ahead using HARQ-DRD.

    Parameters
    ----------
    params : dict — from :func:`fit_harq_drd`.
    S_array_vech : np.ndarray of shape (T, K)
    pi_d : np.ndarray of shape (K,) — today's noise proxy.

    Returns
    -------
    np.ndarray of shape (N, N)
    """
    N = params["N"]
    S_list = [unvech(S_array_vech[t], N) for t in range(len(S_array_vech))]

    log_vols = _log_vol_series(S_list)
    fish_corr = _fisher_corr_series(S_list)

    s_d_v = log_vols[-1]; s_w_v = log_vols[-5:].mean(0); s_m_v = log_vols[-22:].mean(0)
    feats_v = build_har_features(s_d_v, s_w_v, s_m_v)
    # Diagonal vech positions: consistent with fit_harq_drd
    diag_idx = np.array([(i + 1) * (i + 2) // 2 - 1 for i in range(N)])
    lv_hat = forecast_harq(params["params_vol"], feats_v, pi_d[diag_idx])
    std_hat = np.exp(lv_hat)
    D_hat = np.diag(std_hat)

    if fish_corr.shape[1] > 0:
        s_d_c = fish_corr[-1]; s_w_c = fish_corr[-5:].mean(0); s_m_c = fish_corr[-22:].mean(0)
        feats_c = build_har_features(s_d_c, s_w_c, s_m_c)
        # Correlations use plain HAR (no attenuation) per HARQ-DRD specification
        fc_hat = forecast_har(params["params_corr"], feats_c)
        rho_hat = np.tanh(fc_hat)
        R_hat = np.eye(N)
        idx_l, idx_c = np.tril_indices(N, k=-1)
        R_hat[idx_l, idx_c] = rho_hat
        R_hat[idx_c, idx_l] = rho_hat
    else:
        R_hat = np.eye(N)

    return nearest_spd(D_hat @ R_hat @ D_hat)


# ---------------------------------------------------------------------------
# Pre-computation helpers for fast rolling OOS loops
# ---------------------------------------------------------------------------

def precompute_drd_series(
    S_array_vech: np.ndarray,
    N: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert entire vech series to log-vol and Fisher-corr arrays once.

    Parameters
    ----------
    S_array_vech : np.ndarray of shape (T, K)
    N            : asset dimension; inferred when None.

    Returns
    -------
    log_vols  : np.ndarray of shape (T, N)
    fish_corr : np.ndarray of shape (T, M)  where M = N*(N-1)//2
    """
    if N is None:
        N = _infer_n(S_array_vech.shape[1])
    S_list = [unvech(S_array_vech[t], N) for t in range(len(S_array_vech))]
    return _log_vol_series(S_list), _fisher_corr_series(S_list)


def fit_har_drd_precomputed(
    log_vols: np.ndarray,
    fish_corr: np.ndarray,
    N: int,
) -> dict:
    """Fit HAR-DRD from pre-computed log-vol / fisher-corr window slices.

    Parameters
    ----------
    log_vols  : np.ndarray (T, N)
    fish_corr : np.ndarray (T, M)
    N         : asset dimension
    """
    params_vol = fit_har_ols(log_vols)
    params_corr = fit_har_ols(fish_corr) if fish_corr.shape[1] > 0 else np.zeros((0, 4))
    return {"params_vol": params_vol, "params_corr": params_corr, "N": N}


def fit_harq_drd_precomputed(
    log_vols: np.ndarray,
    fish_corr: np.ndarray,
    Pi_window: np.ndarray,
    N: int,
) -> dict:
    """Fit HARQ-DRD from pre-computed log-vol / fisher-corr window slices.

    Parameters
    ----------
    log_vols  : np.ndarray (T, N)
    fish_corr : np.ndarray (T, M)
    Pi_window : np.ndarray (T, K) — raw vech quarticity proxy for window
    N         : asset dimension
    """
    diag_idx = np.array([(i + 1) * (i + 2) // 2 - 1 for i in range(N)])
    pi_vol = Pi_window[:, diag_idx]          # (T, N)
    pi_mean = pi_vol.mean(axis=0)
    pi_vol_dm = pi_vol - pi_mean             # demean within window

    params_vol = fit_harq_ols(log_vols, pi_vol_dm)
    params_corr = fit_har_ols(fish_corr) if fish_corr.shape[1] > 0 else np.zeros((0, 4))
    return {
        "params_vol": params_vol,
        "params_corr": params_corr,
        "N": N,
        "pi_mean": pi_mean,
    }


def forecast_har_drd_precomputed(
    params: dict,
    log_vols: np.ndarray,
    fish_corr: np.ndarray,
) -> np.ndarray:
    """One-step-ahead HAR-DRD forecast from pre-computed series tails.

    Parameters
    ----------
    params    : dict from :func:`fit_har_drd_precomputed`
    log_vols  : np.ndarray (T, N)   — at least 22 rows
    fish_corr : np.ndarray (T, M)

    Returns
    -------
    np.ndarray (N, N) — symmetric PD covariance forecast
    """
    N = params["N"]
    s_d = log_vols[-1];  s_w = log_vols[-5:].mean(0);  s_m = log_vols[-22:].mean(0)
    lv_hat = forecast_har(params["params_vol"], build_har_features(s_d, s_w, s_m))
    D_hat = np.diag(np.exp(lv_hat))

    if fish_corr.shape[1] > 0:
        s_d = fish_corr[-1]; s_w = fish_corr[-5:].mean(0); s_m = fish_corr[-22:].mean(0)
        fc_hat = forecast_har(params["params_corr"], build_har_features(s_d, s_w, s_m))
        rho_hat = np.tanh(fc_hat)
        R_hat = np.eye(N)
        idx_l, idx_c = np.tril_indices(N, k=-1)
        R_hat[idx_l, idx_c] = rho_hat
        R_hat[idx_c, idx_l] = rho_hat
    else:
        R_hat = np.eye(N)

    return nearest_spd(D_hat @ R_hat @ D_hat)


def forecast_harq_drd_precomputed(
    params: dict,
    log_vols: np.ndarray,
    fish_corr: np.ndarray,
    pi_d: np.ndarray,
) -> np.ndarray:
    """One-step-ahead HARQ-DRD forecast from pre-computed series tails.

    Parameters
    ----------
    params    : dict from :func:`fit_harq_drd_precomputed`
    log_vols  : np.ndarray (T, N)   — at least 22 rows
    fish_corr : np.ndarray (T, M)
    pi_d      : np.ndarray (K,)     — raw vech quarticity for the last day
    """
    N = params["N"]
    diag_idx = np.array([(i + 1) * (i + 2) // 2 - 1 for i in range(N)])
    pi_vol = pi_d[diag_idx] - params["pi_mean"]   # apply stored window mean

    s_d = log_vols[-1];  s_w = log_vols[-5:].mean(0);  s_m = log_vols[-22:].mean(0)
    lv_hat = forecast_harq(params["params_vol"], build_har_features(s_d, s_w, s_m), pi_vol)
    D_hat = np.diag(np.exp(lv_hat))

    if fish_corr.shape[1] > 0:
        s_d = fish_corr[-1]; s_w = fish_corr[-5:].mean(0); s_m = fish_corr[-22:].mean(0)
        fc_hat = forecast_har(params["params_corr"], build_har_features(s_d, s_w, s_m))
        rho_hat = np.tanh(fc_hat)
        R_hat = np.eye(N)
        idx_l, idx_c = np.tril_indices(N, k=-1)
        R_hat[idx_l, idx_c] = rho_hat
        R_hat[idx_c, idx_l] = rho_hat
    else:
        R_hat = np.eye(N)

    return nearest_spd(D_hat @ R_hat @ D_hat)

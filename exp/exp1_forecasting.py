"""
Experiment 1: Rolling-window out-of-sample covariance forecasting.

Models compared
---------------
HAR       Heterogeneous AR applied element-wise to vech(S_t)
HARQ      HAR with quarticity-based attenuation of the daily component
HAR-DRD   HAR on DRD decomposition (log-vols + Fisher-correlations)
HARQ-DRD  HARQ on DRD decomposition
EWMA      Exponential smoothing, λ = 0.94 (RiskMetrics)
EWMAQ     EWMA with quarticity-adaptive λ
HEAVY     Scalar GARCH-type recursion driven by realized measures
HEAVYQ    HEAVY with quarticity attenuation of the daily innovation weight

Rolling window
--------------
  In-sample  : 1 000 trading days
  OOS start  : day 1 001
  Forecast   : 1-day-ahead (H_{t|t-1})
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# ── src imports (with inline fallbacks when src modules are not yet built) ────
try:
    from src.utils import vech, unvech, is_spd, nearest_spd
    from src.realized_measures import (
        compute_multivariate_kernel_cov, compute_pi_matrix, compute_pi_vector,
    )
    from src.losses import frobenius_loss, qlike_loss, compute_losses_series, diebold_mariano_test
    from src.mcs import model_confidence_set
    from src.models.har import fit_har_ols, forecast_har, fit_harq_ols, forecast_harq
    from src.models.har_drd import fit_har_drd, forecast_har_drd, fit_harq_drd, forecast_harq_drd
    from src.models.ewma import EWMAModel, EWMAQModel
    from src.models.heavy import HEAVYModel, HEAVYQModel
    from src.portfolio import insanity_filter
    _SRC_AVAILABLE = True
except ImportError:
    _SRC_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Inline fallback implementations
# Used when src/ modules have not been built yet.
# ══════════════════════════════════════════════════════════════════════════════

def _vech(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    i, j = np.triu_indices(n)
    return M[i, j]


def _unvech(v: np.ndarray, n: int) -> np.ndarray:
    M = np.zeros((n, n))
    i, j = np.triu_indices(n)
    M[i, j] = v
    M = M + M.T - np.diag(np.diag(M))
    return M


def _is_spd(M: np.ndarray, tol: float = 1e-8) -> bool:
    try:
        eigvals = np.linalg.eigvalsh(M)
        return bool(np.all(eigvals > tol))
    except np.linalg.LinAlgError:
        return False


def _nearest_spd(M: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Project M onto the cone of symmetric positive definite matrices."""
    M = (M + M.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _frobenius_loss(H: np.ndarray, S: np.ndarray) -> float:
    diff = H - S
    return float(np.sum(diff * diff))


def _qlike_loss(H: np.ndarray, S: np.ndarray) -> float:
    n = H.shape[0]
    try:
        H_inv = np.linalg.inv(H)
        HS = H_inv @ S
        sign, logdet = np.linalg.slogdet(HS)
        if sign <= 0:
            return np.nan
        return float(np.trace(HS) - logdet - n)
    except np.linalg.LinAlgError:
        return np.nan


def _insanity_filter(H: np.ndarray, S_bar: np.ndarray) -> np.ndarray:
    """Replace H with nearest_spd(S_bar) if H is not positive definite."""
    if not _is_spd(H):
        H = _nearest_spd(S_bar)
    return H


# ── HAR helpers ───────────────────────────────────────────────────────────────

def _har_features(y: np.ndarray, lag_d: int = 1, lag_w: int = 5, lag_m: int = 22) -> Tuple:
    """Build HAR design matrix X and target Y for a single series y (T,)."""
    T   = len(y)
    lag = lag_m
    n   = T - lag
    if n < 5:
        return None, None

    y_d = y[lag - lag_d : T - lag_d]          # daily lag
    y_w = np.array([y[t - lag_w:t].mean() for t in range(lag, T)])
    y_m = np.array([y[t - lag_m:t].mean() for t in range(lag, T)])
    Y   = y[lag:]
    X   = np.column_stack([np.ones(n), y_d, y_w, y_m])
    return X, Y


def _fit_har_panel(S_vech: np.ndarray) -> np.ndarray:
    """
    Fit HAR-OLS for all K vech elements.

    Returns beta of shape (K, 4): [intercept, β_d, β_w, β_m] per element.
    """
    T, K = S_vech.shape
    beta = np.zeros((K, 4))
    for k in range(K):
        X, Y = _har_features(S_vech[:, k])
        if X is None:
            continue
        b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        beta[k] = b
    return beta


def _forecast_har_panel(beta: np.ndarray, S_vech: np.ndarray) -> np.ndarray:
    """1-step-ahead HAR forecast for all K elements."""
    K = beta.shape[0]
    preds = np.zeros(K)
    for k in range(K):
        y = S_vech[:, k]
        preds[k] = (
            beta[k, 0]
            + beta[k, 1] * y[-1]
            + beta[k, 2] * y[-5:].mean()
            + beta[k, 3] * y[-22:].mean()
        )
    return preds


def _fit_harq_panel(S_vech: np.ndarray, pi_vech: np.ndarray) -> np.ndarray:
    """
    Fit HARQ-OLS for all K vech elements.

    HARQ model (element k):
      s_t = β_0 + (β_d + β_dQ·π_{t-1}) · s_{t-1} + β_w·s̄_w + β_m·s̄_m + ε

    Rewrite as 5-regressor OLS:
      s_t = β_0 + β_d·s_{t-1} + β_dQ·(π_{t-1}·s_{t-1}) + β_w·s̄_w + β_m·s̄_m

    Returns beta of shape (K, 5).
    """
    T, K  = S_vech.shape
    lag   = 22
    n     = T - lag
    beta  = np.zeros((K, 5))

    for k in range(K):
        y  = S_vech[:, k]
        pi = pi_vech[:, k]

        y_d  = y[lag - 1 : T - 1]
        pi_d = pi[lag - 1 : T - 1]
        y_w  = np.array([y[t - 5:t].mean()  for t in range(lag, T)])
        y_m  = np.array([y[t - 22:t].mean() for t in range(lag, T)])
        Y    = y[lag:]
        X    = np.column_stack([np.ones(n), y_d, pi_d * y_d, y_w, y_m])
        b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        beta[k] = b
    return beta


def _forecast_harq_panel(
    beta: np.ndarray,
    S_vech: np.ndarray,
    pi_vech: np.ndarray,
    pi_t_dem: np.ndarray,
) -> np.ndarray:
    """1-step-ahead HARQ forecast for all K elements."""
    K     = beta.shape[0]
    preds = np.zeros(K)
    for k in range(K):
        y  = S_vech[:, k]
        preds[k] = (
            beta[k, 0]
            + beta[k, 1] * y[-1]
            + beta[k, 2] * pi_t_dem[k] * y[-1]
            + beta[k, 3] * y[-5:].mean()
            + beta[k, 4] * y[-22:].mean()
        )
    return preds


# ── DRD helpers ───────────────────────────────────────────────────────────────

def _drd_decompose(S_vech: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    DRD decompose each row of S_vech into log-vols and Fisher-transformed corrs.

    Returns
    -------
    log_vols  : (T, n)       log of diagonal standard deviations
    fisher_corr : (T, n*(n-1)//2)  Fisher-transformed off-diagonal correlations
    """
    T         = S_vech.shape[0]
    n_corr    = n * (n - 1) // 2
    log_vols  = np.zeros((T, n))
    fish_corr = np.zeros((T, n_corr))

    triu_i, triu_j = np.triu_indices(n, k=0)
    triu_off_i, triu_off_j = np.triu_indices(n, k=1)

    for t in range(T):
        S = _unvech(S_vech[t], n)
        S = _nearest_spd(S)
        d = np.sqrt(np.maximum(np.diag(S), 1e-12))
        log_vols[t] = np.log(d)

        d_inv = 1.0 / d
        P = d_inv[:, None] * S * d_inv[None, :]   # correlation matrix
        np.fill_diagonal(P, 1.0)
        # Clip off-diagonal to (-1+ε, 1-ε) before Fisher transform
        p_off = np.clip(P[triu_off_i, triu_off_j], -1 + 1e-6, 1 - 1e-6)
        fish_corr[t] = 0.5 * np.log((1.0 + p_off) / (1.0 - p_off))

    return log_vols, fish_corr


def _drd_reconstruct(log_vol_hat: np.ndarray, fish_corr_hat: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct covariance matrix from DRD forecasts."""
    d   = np.exp(log_vol_hat)          # (n,)
    P   = np.eye(n)
    triu_off_i, triu_off_j = np.triu_indices(n, k=1)
    p_off = np.tanh(fish_corr_hat)
    P[triu_off_i, triu_off_j] = p_off
    P[triu_off_j, triu_off_i] = p_off
    # Ensure correlation matrix is PD
    P = _nearest_spd(P)
    # Normalize to true correlations
    d_P = np.sqrt(np.maximum(np.diag(P), 1e-12))
    P   = P / (d_P[:, None] * d_P[None, :])
    np.fill_diagonal(P, 1.0)
    # Reconstruct covariance
    H = d[:, None] * P * d[None, :]
    return H


def _rolling_mean_cumsum(Y: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorised rolling mean along axis 0 (columns independently).

    Uses the cumulative-sum identity to compute all windows in O(T·K).
    First *window-1* rows are NaN.
    """
    out    = np.full_like(Y, np.nan, dtype=float)
    cs     = np.cumsum(Y, axis=0)
    out[window - 1:] = cs[window - 1:].copy()
    out[window:]    -= cs[:-window]
    out[window - 1:] /= window
    return out


def _drd_decompose_panel(S_vech_all: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute DRD components for the *entire* time series (T rows).

    Calling this once and slicing in the rolling loop avoids decomposing
    the full IS window (1000 rows) at every OOS step.

    Returns
    -------
    log_vols  : (T, n)
    fish_corr : (T, n*(n-1)//2)
    """
    T      = len(S_vech_all)
    n_corr = n * (n - 1) // 2
    log_vols  = np.zeros((T, n))
    fish_corr = np.zeros((T, max(n_corr, 1)))   # at least 1 col to avoid 0-width arrays
    triu_off_i, triu_off_j = np.triu_indices(n, k=1)

    for t in range(T):
        S = _unvech(S_vech_all[t], n)
        S = _nearest_spd(S)
        d = np.sqrt(np.maximum(np.diag(S), 1e-12))
        log_vols[t] = np.log(d)
        if n_corr > 0:
            d_inv = 1.0 / d
            P     = d_inv[:, None] * S * d_inv[None, :]
            np.fill_diagonal(P, 1.0)
            p_off = np.clip(P[triu_off_i, triu_off_j], -1 + 1e-6, 1 - 1e-6)
            fish_corr[t] = 0.5 * np.log((1.0 + p_off) / (1.0 - p_off))

    if n_corr == 0:
        fish_corr = fish_corr[:, :0]
    return log_vols, fish_corr


def _fit_har_panel_precomp(
    Y:  np.ndarray,     # (n_is, K) targets
    Xd: np.ndarray,     # (n_is, K) daily lag
    Xw: np.ndarray,     # (n_is, K) 5-day mean
    Xm: np.ndarray,     # (n_is, K) 22-day mean
) -> np.ndarray:
    """HAR-OLS with pre-computed feature columns. Returns beta (K, 4)."""
    n_is, K = Y.shape
    ones  = np.ones(n_is)
    beta  = np.zeros((K, 4))
    for k in range(K):
        X   = np.column_stack([ones, Xd[:, k], Xw[:, k], Xm[:, k]])
        b, _, _, _ = np.linalg.lstsq(X, Y[:, k], rcond=None)
        beta[k] = b
    return beta


def _fit_harq_panel_precomp(
    Y:    np.ndarray,   # (n_is, K) targets
    Xd:   np.ndarray,   # (n_is, K) daily lag (s_{t-1})
    Xw:   np.ndarray,   # (n_is, K) 5-day mean
    Xm:   np.ndarray,   # (n_is, K) 22-day mean
    Xpi:  np.ndarray,   # (n_is, K) daily pi lag (π_{t-1})
) -> np.ndarray:
    """HARQ-OLS with pre-computed features. Returns beta (K, 5)."""
    n_is, K = Y.shape
    ones     = np.ones(n_is)
    beta     = np.zeros((K, 5))
    for k in range(K):
        X   = np.column_stack([ones, Xd[:, k], Xpi[:, k] * Xd[:, k], Xw[:, k], Xm[:, k]])
        b, _, _, _ = np.linalg.lstsq(X, Y[:, k], rcond=None)
        beta[k] = b
    return beta


def _forecast_har_drd(log_vols: np.ndarray, fish_corr: np.ndarray, n: int) -> np.ndarray:
    """HAR-DRD: fit HAR on each DRD component, reconstruct."""
    beta_v = _fit_har_panel(log_vols)
    beta_c = _fit_har_panel(fish_corr)
    lv_hat = _forecast_har_panel(beta_v, log_vols)
    fc_hat = _forecast_har_panel(beta_c, fish_corr)
    return _drd_reconstruct(lv_hat, fc_hat, n)


def _forecast_harq_drd(
    log_vols: np.ndarray,
    fish_corr: np.ndarray,
    pi_vols: np.ndarray,
    pi_corr: np.ndarray,
    pi_t_dem_vols: np.ndarray,
    pi_t_dem_corr: np.ndarray,
    n: int,
) -> np.ndarray:
    """HARQ-DRD: fit HARQ on each DRD component, reconstruct."""
    beta_v = _fit_harq_panel(log_vols,  pi_vols)
    beta_c = _fit_harq_panel(fish_corr, pi_corr)
    lv_hat = _forecast_harq_panel(beta_v, log_vols,  pi_vols,  pi_t_dem_vols)
    fc_hat = _forecast_harq_panel(beta_c, fish_corr, pi_corr,  pi_t_dem_corr)
    return _drd_reconstruct(lv_hat, fc_hat, n)


# ── EWMA helpers ──────────────────────────────────────────────────────────────

def _run_ewma(S_matrices: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    Run EWMA filter over the full time series.

    Parameters
    ----------
    S_matrices : (T, n, n)  realized covariance matrices
    lam        : smoothing parameter (0 < lam < 1)

    Returns
    -------
    (T+1, n, n)  H[0] = S_matrices[0]; H[t] = lam*H[t-1] + (1-lam)*S[t-1]
    """
    T, n, _ = S_matrices.shape
    H = np.zeros((T + 1, n, n))
    H[0] = S_matrices[0]
    for t in range(1, T + 1):
        H[t] = lam * H[t - 1] + (1.0 - lam) * S_matrices[t - 1]
    return H


def _run_ewmaq(
    S_matrices: np.ndarray,
    pi_vech: np.ndarray,
    n: int,
    lam_base: float = 0.94,
    lam_adj: float = 0.02,
) -> np.ndarray:
    """
    EWMAQ: element-wise adaptive λ driven by standardized quarticity.

    λ_{ij,t} = clip(lam_base + lam_adj · π̃_{ij,t-1}, 0.50, 0.99)
    Higher quarticity → larger λ → stronger smoothing of noisy S_t.
    """
    T, n, _ = S_matrices.shape
    K        = pi_vech.shape[1]
    H_vech   = np.zeros((T + 1, K))
    S_vech_0 = _vech(S_matrices[0])
    H_vech[0] = S_vech_0

    triu_i, triu_j = np.triu_indices(n)

    for t in range(1, T + 1):
        pi_dem  = pi_vech[t - 1]              # demeaned quarticity at t-1
        lam     = np.clip(lam_base + lam_adj * pi_dem, 0.50, 0.99)
        S_vt    = _vech(S_matrices[t - 1])
        H_vech[t] = lam * H_vech[t - 1] + (1.0 - lam) * S_vt

    H_all = np.zeros((T + 1, n, n))
    for t in range(T + 1):
        H_all[t] = _unvech(H_vech[t], n)
    return H_all


# ── HEAVY helpers ─────────────────────────────────────────────────────────────

def _fit_heavy_panel(S_vech: np.ndarray) -> np.ndarray:
    """
    Fit simplified HEAVY: s_t = ω + c · s_{t-1}  (element-wise AR(1) proxy).

    Returns beta (K, 2): [ω, c] per element with c clipped to [0, 0.9999].
    """
    T, K  = S_vech.shape
    beta  = np.zeros((K, 2))
    X     = np.column_stack([np.ones(T - 1), S_vech[:-1, 0]])
    for k in range(K):
        X[:, 1] = S_vech[:-1, k]
        b, _, _, _ = np.linalg.lstsq(X, S_vech[1:, k], rcond=None)
        b[1] = np.clip(b[1], 0.0, 0.9999)
        b[0] = max(b[0], 0.0)
        beta[k] = b
    return beta


def _run_heavy(S_matrices: np.ndarray, beta: np.ndarray, n: int) -> np.ndarray:
    """
    Recursive HEAVY filter.  H_t = ω + c · S_{t-1} (using OLS c = α + β).
    H[0] = S_matrices[0]; H[t] = omega + c * S[t-1].
    """
    T  = S_matrices.shape[0]
    K  = beta.shape[0]
    H_vech = np.zeros((T + 1, K))
    H_vech[0] = _vech(S_matrices[0])

    for t in range(1, T + 1):
        s_prev     = _vech(S_matrices[t - 1])
        H_vech[t]  = beta[:, 0] + beta[:, 1] * s_prev

    H_all = np.zeros((T + 1, n, n))
    for t in range(T + 1):
        H_all[t] = _unvech(H_vech[t], n)
    return H_all


def _fit_heavyq_panel(S_vech: np.ndarray, pi_vech: np.ndarray) -> np.ndarray:
    """
    Fit HEAVYQ: s_t = ω + (c + c_Q · π_{t-1}) · s_{t-1} (element-wise).

    Returns beta (K, 3): [ω, c, c_Q] per element.
    """
    T, K  = S_vech.shape
    lag   = 1
    beta  = np.zeros((K, 3))

    for k in range(K):
        y    = S_vech[lag:, k]
        s_p  = S_vech[:-lag, k]
        pi_p = pi_vech[:-lag, k]
        X    = np.column_stack([np.ones(len(y)), s_p, pi_p * s_p])
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        b[0] = max(b[0], 0.0)
        beta[k] = b
    return beta


def _run_heavyq(
    S_matrices: np.ndarray,
    pi_vech: np.ndarray,
    beta: np.ndarray,
    n: int,
) -> np.ndarray:
    """Recursive HEAVYQ filter using fitted parameters and demeaned pi."""
    T  = S_matrices.shape[0]
    K  = beta.shape[0]
    H_vech = np.zeros((T + 1, K))
    H_vech[0] = _vech(S_matrices[0])

    for t in range(1, T + 1):
        s_prev  = _vech(S_matrices[t - 1])
        pi_prev = pi_vech[t - 1]
        c_eff   = beta[:, 1] + beta[:, 2] * pi_prev
        H_vech[t] = beta[:, 0] + c_eff * s_prev

    H_all = np.zeros((T + 1, n, n))
    for t in range(T + 1):
        H_all[t] = _unvech(H_vech[t], n)
    return H_all


# ── loss & inference ──────────────────────────────────────────────────────────

def _compute_losses(
    forecasts: Dict[str, List[np.ndarray]],
    actuals: List[np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    losses_frob = {}
    losses_qlik = {}
    for name, Hs in forecasts.items():
        frob = np.array([_frobenius_loss(H, S) for H, S in zip(Hs, actuals)])
        qlik = np.array([_qlike_loss(H, S)     for H, S in zip(Hs, actuals)])
        losses_frob[name] = frob
        losses_qlik[name] = qlik
    return losses_frob, losses_qlik


def _newey_west_var(d: np.ndarray, max_lag: int = 5) -> float:
    """Newey-West long-run variance estimator for the loss-differential series."""
    T   = len(d)
    d_c = d - d.mean()
    nw  = np.dot(d_c, d_c) / T
    for h in range(1, max_lag + 1):
        w    = 1.0 - h / (max_lag + 1.0)
        cov  = np.dot(d_c[h:], d_c[:-h]) / T
        nw  += 2.0 * w * cov
    return max(nw, 1e-12)


def _dm_test(loss1: np.ndarray, loss2: np.ndarray) -> Tuple[float, float]:
    """
    Diebold-Mariano test.  H0: equal predictive accuracy.

    Returns (DM_statistic, two-sided p-value).
    """
    from scipy import stats
    d    = loss1 - loss2
    T    = len(d)
    nw   = _newey_west_var(d)
    dm   = d.mean() / np.sqrt(nw / T)
    pval = 2.0 * stats.norm.sf(np.abs(dm))
    return float(dm), float(pval)


def _model_confidence_set(
    loss_matrix: np.ndarray,
    model_names: List[str],
    alpha: float = 0.10,
    B: int = 1000,
    block_len: int = 5,
    rng: np.random.Generator = None,
) -> List[str]:
    """
    Simplified Model Confidence Set (Hansen, Lunde & Nason 2011).

    Uses the range statistic T_R = max_{i,j} |d̄_{ij}| / σ̂_{ij} and a
    stationary block bootstrap to build the reference distribution.

    Parameters
    ----------
    loss_matrix : (T, M)  loss matrix, one column per model.
    model_names : list of M model name strings.
    alpha       : significance level (default 0.10 → 90% MCS).
    B           : bootstrap replications.
    block_len   : block length for stationary block bootstrap.

    Returns
    -------
    List of model names in the surviving MCS.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    surviving = list(range(loss_matrix.shape[1]))

    while len(surviving) > 1:
        L     = loss_matrix[:, surviving]
        T, M  = L.shape
        # Loss differential matrix d_{ij,t} = L_{i,t} - L_{j,t}
        d_bar = np.zeros((M, M))
        d_var = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                if i != j:
                    d       = L[:, i] - L[:, j]
                    d_bar[i, j] = d.mean()
                    d_var[i, j] = _newey_west_var(d)

        # Range statistic
        with np.errstate(invalid='ignore'):
            t_stat = np.abs(d_bar) / np.sqrt(np.maximum(d_var, 1e-20) / T)
        np.fill_diagonal(t_stat, 0.0)
        T_R = t_stat.max()

        # Bootstrap distribution of T_R*
        T_R_boot = np.zeros(B)
        for b in range(B):
            # Stationary block bootstrap
            starts   = rng.integers(0, T, size=T // block_len + 1)
            indices  = np.concatenate([
                np.arange(s, min(s + block_len, T)) for s in starts
            ])[:T]
            L_boot = L[indices]
            db_bar = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    if i != j:
                        db       = L_boot[:, i] - L_boot[:, j]
                        db_bar[i, j] = db.mean() - d_bar[i, j]  # centered
            db_var = d_var
            with np.errstate(invalid='ignore'):
                t_b = np.abs(db_bar) / np.sqrt(np.maximum(db_var, 1e-20) / T)
            np.fill_diagonal(t_b, 0.0)
            T_R_boot[b] = t_b.max()

        cv = np.quantile(T_R_boot, 1.0 - alpha)
        if T_R <= cv:
            break   # all surviving models are equally good

        # Eliminate model with worst average loss
        avg_loss  = L.mean(axis=0)
        worst_idx = int(np.argmax(avg_loss))
        surviving.pop(worst_idx)

    return [model_names[i] for i in surviving]


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_data(
    T: int = 1500,
    n: int = 5,
    M: int = 78,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate T daily realized covariance matrices and quarticity vectors.

    Data-generating process
    -----------------------
    A slow-moving true covariance V_t follows a mean-reverting AR(1) on
    the Cholesky factor; S_t ~ Wishart(V_t, M) / M.

    Parameters
    ----------
    T   : number of trading days.
    n   : number of assets.
    M   : intraday bars per day (≈ 78 for 5-minute bars).
    seed: random seed for reproducibility.

    Returns
    -------
    S_matrices : (T, n, n)  realized covariances.
    pi_vech    : (T, K)     realized quarticity vectors, K = n(n+1)/2.
    """
    rng  = np.random.default_rng(seed)
    K    = n * (n + 1) // 2
    i_ut, j_ut = np.triu_indices(n)

    # --- latent true covariance (slow AR(1) on Cholesky) ----------------
    rho  = 0.99                          # high persistence
    # Base Cholesky: scale * random lower-triangular
    L0   = np.tril(rng.standard_normal((n, n)))
    L0   = L0 + n * np.eye(n)           # ensure diagonal dominance
    V0   = L0 @ L0.T / (2 * n)

    V    = np.zeros((T, n, n))
    V[0] = V0
    for t in range(1, T):
        noise = rng.standard_normal((n, n))
        noise = np.tril(noise)
        dL    = 0.05 * noise
        L_new = np.linalg.cholesky(V[t - 1]) + dL
        # ensure positive diagonal
        L_new = np.abs(np.diag(L_new)) * np.eye(n) + np.tril(L_new, -1)
        L_new[np.diag_indices(n)] = np.maximum(np.diag(L_new), 1e-4)
        Vt    = L_new @ L_new.T
        V[t]  = rho * Vt + (1.0 - rho) * V0

    # --- simulate Wishart realized covariances ----------------------------
    S_matrices = np.zeros((T, n, n))
    for t in range(T):
        # Draw M intraday returns and form realized covariance
        L_t = np.linalg.cholesky(_nearest_spd(V[t]))
        R_t = rng.standard_normal((M, n)) @ L_t.T / np.sqrt(M)  # scale to daily
        S_matrices[t] = R_t.T @ R_t

    # --- realized quarticity: Pi_{t,ij} = (M/3) Σ_k r_{k,i}^2 r_{k,j}^2 --
    pi_vech = np.zeros((T, K))
    for t in range(T):
        L_t  = np.linalg.cholesky(_nearest_spd(V[t]))
        R_t  = rng.standard_normal((M, n)) @ L_t.T / np.sqrt(M)
        R_sq = R_t ** 2
        Pi_t = (M / 3.0) * (R_sq.T @ R_sq)
        pi_vech[t] = Pi_t[i_ut, j_ut]

    return S_matrices, pi_vech


# ══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAMES = ['HAR', 'HARQ', 'HAR-DRD', 'HARQ-DRD', 'EWMA', 'EWMAQ', 'HEAVY', 'HEAVYQ']

# Q-model / non-Q-model benchmark pairs for DM tests
DM_PAIRS = [
    ('HARQ',     'HAR'),
    ('HARQ-DRD', 'HAR-DRD'),
    ('EWMAQ',    'EWMA'),
    ('HEAVYQ',   'HEAVY'),
]


def _load_or_generate_data(results_dir: Path):
    """
    Attempt to load pre-computed realized measures from data/.
    Falls back to synthetic Wishart data when unavailable.
    """
    S_path  = Path('data') / 'S_matrices.npy'
    pi_path = Path('data') / 'pi_vech.npy'
    n_path  = Path('data') / 'n_assets.npy'

    if S_path.exists() and pi_path.exists():
        print('[exp1] Loading pre-computed realized measures from data/ ...')
        S_matrices = np.load(str(S_path))
        pi_vech    = np.load(str(pi_path))
        n          = int(np.load(str(n_path))) if n_path.exists() else S_matrices.shape[1]
        synthetic  = False
    else:
        print('[exp1] Pre-computed data not found. Generating SYNTHETIC data '
              '(T=1500 days, N=5 assets, Wishart DGP).')
        print('       *** Results are based on SYNTHETIC data, not real markets. ***')
        S_matrices, pi_vech = _generate_synthetic_data(T=1500, n=5, M=78)
        n         = 5
        synthetic = True

    return S_matrices, pi_vech, n, synthetic


def run_forecasting_experiment(window: int = 1000) -> Dict:
    """
    Execute the rolling-window OOS covariance forecasting experiment.

    Parameters
    ----------
    window : int
        In-sample window length (default 1 000 trading days).

    Returns
    -------
    dict with keys:
        'losses_frob', 'losses_qlike' : dicts model→np.ndarray of OOS losses
        'dm_results'                   : nested dict of DM test statistics
        'mcs_frob', 'mcs_qlike'       : lists of model names in 90% MCS
        'split_frob', 'split_qlike'   : DataFrames of split-sample losses
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    S_matrices, pi_vech, n, synthetic = _load_or_generate_data(results_dir)
    T   = len(S_matrices)
    K   = n * (n + 1) // 2
    T_oos = T - window

    if T_oos <= 0:
        raise ValueError(
            f"Dataset has {T} days but window={window}. Need T > window."
        )

    print(f'[exp1] N={n} assets, T={T} days, window={window}, T_oos={T_oos}')
    print(f'[exp1] Data: {"SYNTHETIC" if synthetic else "REAL"}')

    # ── vech series ───────────────────────────────────────────────────────────
    S_vech = np.zeros((T, K))
    i_ut, j_ut = np.triu_indices(n)
    for t in range(T):
        S_vech[t] = S_matrices[t][i_ut, j_ut]

    # ── one-time pre-computation over full dataset ────────────────────────────
    print('[exp1] Pre-computing DRD panel and rolling HAR features ...')

    # DRD decomposition for every day (replaces per-step IS-window decompose)
    log_vols_all, fish_corr_all = _drd_decompose_panel(S_vech, n)

    # Vectorised rolling means (lag 5 and 22) for S_vech, pi_vech, DRD components
    S_vech_w   = _rolling_mean_cumsum(S_vech, 5)
    S_vech_m   = _rolling_mean_cumsum(S_vech, 22)
    pi_vech_d  = np.vstack([np.zeros((1, pi_vech.shape[1])), pi_vech[:-1]])  # 1-lag
    lv_w       = _rolling_mean_cumsum(log_vols_all, 5)
    lv_m       = _rolling_mean_cumsum(log_vols_all, 22)
    fc_w       = _rolling_mean_cumsum(fish_corr_all, 5)
    fc_m       = _rolling_mean_cumsum(fish_corr_all, 22)

    # Diagonal / off-diagonal vech indices (constant; compute once)
    diag_idx    = np.array([k for k, (ii, jj) in enumerate(zip(i_ut, j_ut)) if ii == jj])
    offdiag_idx = np.array([k for k, (ii, jj) in enumerate(zip(i_ut, j_ut)) if ii != jj])

    # ── pre-run recursive filters over full dataset ──────────────────────────
    # EWMA, EWMAQ, HEAVY, HEAVYQ need to be run on the full sample once
    # and then we index into their outputs for each OOS step.
    print('[exp1] Pre-fitting recursive models (EWMA, HEAVY) ...')

    ewma_all  = _run_ewma(S_matrices, lam=0.94)

    # For EWMAQ we need IS-period demeaned pi (use global stats as approximation)
    pi_mu   = pi_vech[:window].mean(axis=0)
    pi_sig  = np.maximum(pi_vech[:window].std(axis=0), 1e-12)
    pi_dem_global = (pi_vech - pi_mu) / pi_sig
    ewmaq_all = _run_ewmaq(S_matrices, pi_dem_global, n, lam_base=0.94, lam_adj=0.02)

    # Fit HEAVY on first IS window and re-use (approximate; proper approach
    # would re-fit at each step, but that is computationally expensive for
    # HEAVY due to non-linear optimization).
    heavy_beta  = _fit_heavy_panel(S_vech[:window])
    heavyq_beta = _fit_heavyq_panel(S_vech[:window], pi_vech[:window])
    heavy_all   = _run_heavy(S_matrices, heavy_beta, n)
    heavyq_all  = _run_heavyq(S_matrices, pi_dem_global, heavyq_beta, n)

    # ── rolling OOS loop ──────────────────────────────────────────────────────
    print('[exp1] Starting rolling OOS loop ...')
    forecasts: Dict[str, List[np.ndarray]] = {m: [] for m in MODEL_NAMES}
    actuals:   List[np.ndarray]             = []
    pi_t_magnitudes: List[float]            = []

    for step, t in enumerate(range(window, T)):
        if step % 100 == 0:
            print(f'  step {step}/{T_oos} (t={t})')

        # In-sample window (raw series only; features pre-computed)
        is_pi   = pi_vech[t - window : t]     # (window, K) — for IS stats

        # Demeaned pi at time t (rolling IS statistics)
        pi_mu_t  = is_pi.mean(axis=0)
        pi_sig_t = np.maximum(is_pi.std(axis=0), 1e-12)
        pi_t_dem = (pi_vech[t] - pi_mu_t) / pi_sig_t

        # Track pi magnitude for split-sample analysis
        pi_t_magnitudes.append(float(np.median(np.abs(pi_t_dem))))

        actual = S_matrices[t]
        actuals.append(actual)

        # IS feature slices (drop first 22 rows that lack monthly rolling mean)
        lag_m  = 22
        a, b_  = t - window + lag_m, t         # [a, b_) = valid IS feature rows
        # Targets and daily lag for HAR
        Y_s  = S_vech[a:b_]                    # (window-22, K)
        Xd_s = S_vech[a - 1:b_ - 1]           # (window-22, K) — 1-lag
        Xw_s = S_vech_w[a:b_]                  # (window-22, K) — 5-day mean
        Xm_s = S_vech_m[a:b_]                  # (window-22, K) — 22-day mean
        Xpi_s= pi_vech_d[a:b_]                 # (window-22, K) — pi 1-lag

        # ── HAR ────────────────────────────────────────────────────────────
        beta_har   = _fit_har_panel_precomp(Y_s, Xd_s, Xw_s, Xm_s)
        s_last     = S_vech[t - 1]
        h_vech_har = (beta_har[:, 0]
                      + beta_har[:, 1] * s_last
                      + beta_har[:, 2] * S_vech_w[t - 1]
                      + beta_har[:, 3] * S_vech_m[t - 1])
        H_har      = _unvech(h_vech_har, n)
        H_har      = _nearest_spd(H_har) if not _is_spd(H_har) else H_har
        forecasts['HAR'].append(H_har)

        # ── HARQ ───────────────────────────────────────────────────────────
        beta_harq   = _fit_harq_panel_precomp(Y_s, Xd_s, Xw_s, Xm_s, Xpi_s)
        pi_d_last   = pi_vech[t - 1]          # raw pi_{t-1} for HARQ interaction
        pi_d_dem_last = (pi_d_last - pi_mu_t) / pi_sig_t
        h_vech_harq = (beta_harq[:, 0]
                       + beta_harq[:, 1] * s_last
                       + beta_harq[:, 2] * pi_d_dem_last * s_last
                       + beta_harq[:, 3] * S_vech_w[t - 1]
                       + beta_harq[:, 4] * S_vech_m[t - 1])
        H_harq      = _unvech(h_vech_harq, n)
        H_harq      = _nearest_spd(H_harq) if not _is_spd(H_harq) else H_harq
        forecasts['HARQ'].append(H_harq)

        # ── HAR-DRD ────────────────────────────────────────────────────────
        # Use pre-computed DRD panel slices
        Y_lv  = log_vols_all[a:b_]
        Xd_lv = log_vols_all[a - 1:b_ - 1]
        Xw_lv = lv_w[a:b_]
        Xm_lv = lv_m[a:b_]
        beta_lv = _fit_har_panel_precomp(Y_lv, Xd_lv, Xw_lv, Xm_lv)
        lv_hat  = (beta_lv[:, 0]
                   + beta_lv[:, 1] * log_vols_all[t - 1]
                   + beta_lv[:, 2] * lv_w[t - 1]
                   + beta_lv[:, 3] * lv_m[t - 1])

        if fish_corr_all.shape[1] > 0:
            Y_fc  = fish_corr_all[a:b_]
            Xd_fc = fish_corr_all[a - 1:b_ - 1]
            Xw_fc = fc_w[a:b_]
            Xm_fc = fc_m[a:b_]
            beta_fc = _fit_har_panel_precomp(Y_fc, Xd_fc, Xw_fc, Xm_fc)
            fc_hat  = (beta_fc[:, 0]
                       + beta_fc[:, 1] * fish_corr_all[t - 1]
                       + beta_fc[:, 2] * fc_w[t - 1]
                       + beta_fc[:, 3] * fc_m[t - 1])
        else:
            fc_hat = np.zeros(0)

        H_har_drd = _drd_reconstruct(lv_hat, fc_hat, n)
        H_har_drd = _nearest_spd(H_har_drd) if not _is_spd(H_har_drd) else H_har_drd
        forecasts['HAR-DRD'].append(H_har_drd)

        # ── HARQ-DRD ───────────────────────────────────────────────────────
        # Quarticity for volatility components (diagonal pi elements)
        pi_vols_is = pi_vech_d[a:b_][:, diag_idx]
        pi_t_vols  = pi_d_dem_last[diag_idx]
        beta_lv_q  = _fit_harq_panel_precomp(Y_lv, Xd_lv, Xw_lv, Xm_lv, pi_vols_is)
        lv_hat_q   = (beta_lv_q[:, 0]
                      + beta_lv_q[:, 1] * log_vols_all[t - 1]
                      + beta_lv_q[:, 2] * pi_t_vols * log_vols_all[t - 1]
                      + beta_lv_q[:, 3] * lv_w[t - 1]
                      + beta_lv_q[:, 4] * lv_m[t - 1])

        if fish_corr_all.shape[1] > 0 and len(offdiag_idx) > 0:
            pi_corr_is  = pi_vech_d[a:b_][:, offdiag_idx]
            pi_t_corr   = pi_d_dem_last[offdiag_idx]
            beta_fc_q   = _fit_harq_panel_precomp(Y_fc, Xd_fc, Xw_fc, Xm_fc, pi_corr_is)
            fc_hat_q    = (beta_fc_q[:, 0]
                           + beta_fc_q[:, 1] * fish_corr_all[t - 1]
                           + beta_fc_q[:, 2] * pi_t_corr * fish_corr_all[t - 1]
                           + beta_fc_q[:, 3] * fc_w[t - 1]
                           + beta_fc_q[:, 4] * fc_m[t - 1])
        else:
            fc_hat_q = np.zeros(0)

        H_harq_drd = _drd_reconstruct(lv_hat_q, fc_hat_q, n)
        H_harq_drd = _nearest_spd(H_harq_drd) if not _is_spd(H_harq_drd) else H_harq_drd
        forecasts['HARQ-DRD'].append(H_harq_drd)

        # ── EWMA (index into pre-computed array) ───────────────────────────
        H_ewma = ewma_all[t]
        H_ewma = _nearest_spd(H_ewma) if not _is_spd(H_ewma) else H_ewma
        forecasts['EWMA'].append(H_ewma)

        # ── EWMAQ ─────────────────────────────────────────────────────────
        H_ewmaq = ewmaq_all[t]
        H_ewmaq = _nearest_spd(H_ewmaq) if not _is_spd(H_ewmaq) else H_ewmaq
        forecasts['EWMAQ'].append(H_ewmaq)

        # ── HEAVY ─────────────────────────────────────────────────────────
        H_heavy = heavy_all[t]
        H_heavy = _nearest_spd(H_heavy) if not _is_spd(H_heavy) else H_heavy
        forecasts['HEAVY'].append(H_heavy)

        # ── HEAVYQ ────────────────────────────────────────────────────────
        H_heavyq = heavyq_all[t]
        H_heavyq = _nearest_spd(H_heavyq) if not _is_spd(H_heavyq) else H_heavyq
        forecasts['HEAVYQ'].append(H_heavyq)

    print('[exp1] OOS loop complete.')

    # ── save forecasts ────────────────────────────────────────────────────────
    npz_path = Path('data') / 'forecasts_exp1.npz'
    np.savez(
        str(npz_path),
        actuals=np.array(actuals),
        **{f'forecast_{m}': np.array(forecasts[m]) for m in MODEL_NAMES},
    )
    print(f'[exp1] Forecasts saved to {npz_path}')

    # ── compute losses ────────────────────────────────────────────────────────
    print('[exp1] Computing Frobenius and QLIKE losses ...')
    losses_frob, losses_qlik = _compute_losses(forecasts, actuals)

    # ── DM tests ──────────────────────────────────────────────────────────────
    print('[exp1] Running Diebold-Mariano tests ...')
    dm_results = {}
    for (q_model, bench) in DM_PAIRS:
        for loss_name, losses in [('Frob', losses_frob), ('QLIKE', losses_qlik)]:
            l1 = losses[q_model]
            l2 = losses[bench]
            valid = ~(np.isnan(l1) | np.isnan(l2))
            if valid.sum() < 10:
                dm_results[(q_model, bench, loss_name)] = (np.nan, np.nan)
                continue
            stat, pval = _dm_test(l1[valid], l2[valid])
            dm_results[(q_model, bench, loss_name)] = (stat, pval)

    # ── MCS ───────────────────────────────────────────────────────────────────
    print('[exp1] Computing 90% Model Confidence Sets ...')
    loss_mat_frob = np.column_stack([losses_frob[m] for m in MODEL_NAMES])
    loss_mat_qlik = np.column_stack([losses_qlik[m] for m in MODEL_NAMES])

    # Drop rows where any model has NaN
    valid_frob = ~np.any(np.isnan(loss_mat_frob), axis=1)
    valid_qlik = ~np.any(np.isnan(loss_mat_qlik), axis=1)
    mcs_frob  = _model_confidence_set(loss_mat_frob[valid_frob], MODEL_NAMES)
    mcs_qlik  = _model_confidence_set(loss_mat_qlik[valid_qlik], MODEL_NAMES)

    print(f'  MCS (Frobenius) : {mcs_frob}')
    print(f'  MCS (QLIKE)     : {mcs_qlik}')

    # ── split-sample analysis ─────────────────────────────────────────────────
    print('[exp1] Split-sample analysis by pi_t magnitude ...')
    pi_arr   = np.array(pi_t_magnitudes)
    terciles = np.quantile(pi_arr, [1 / 3, 2 / 3])
    labels   = ['Low pi', 'Med pi', 'High pi']

    def _tercile_label(v):
        if v <= terciles[0]:
            return 'Low pi'
        elif v <= terciles[1]:
            return 'Med pi'
        else:
            return 'High pi'

    grp_labels = [_tercile_label(v) for v in pi_arr]

    split_frob_rows = []
    split_qlik_rows = []
    for grp in labels:
        idx = np.array([i for i, g in enumerate(grp_labels) if g == grp])
        row_f = {m: float(np.nanmean(losses_frob[m][idx])) for m in MODEL_NAMES}
        row_q = {m: float(np.nanmean(losses_qlik[m][idx])) for m in MODEL_NAMES}
        row_f['group'] = grp
        row_q['group'] = grp
        split_frob_rows.append(row_f)
        split_qlik_rows.append(row_q)

    split_frob = pd.DataFrame(split_frob_rows).set_index('group')
    split_qlik = pd.DataFrame(split_qlik_rows).set_index('group')

    # ── summary table ─────────────────────────────────────────────────────────
    summary = pd.DataFrame({
        'Mean Frob Loss':  {m: float(np.nanmean(losses_frob[m])) for m in MODEL_NAMES},
        'Mean QLIKE Loss': {m: float(np.nanmean(losses_qlik[m])) for m in MODEL_NAMES},
        'In MCS (Frob)':   {m: m in mcs_frob for m in MODEL_NAMES},
        'In MCS (QLIKE)':  {m: m in mcs_qlik for m in MODEL_NAMES},
    })
    print('\n[exp1] Summary of OOS losses:\n', summary.to_string())
    summary.to_csv(str(results_dir / 'exp1_results.csv'))
    print(f'[exp1] Numerical results saved to {results_dir / "exp1_results.csv"}')

    # ── generate plots ────────────────────────────────────────────────────────
    print('[exp1] Generating plots ...')

    # Plot 1: Mean losses by model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (loss_name, losses) in zip(
        axes,
        [('Frobenius', losses_frob), ('QLIKE', losses_qlik)],
    ):
        means = [np.nanmean(losses[m]) for m in MODEL_NAMES]
        colors = ['#2196F3' if m not in ['HARQ', 'HARQ-DRD', 'EWMAQ', 'HEAVYQ']
                  else '#F44336' for m in MODEL_NAMES]
        bars  = ax.bar(MODEL_NAMES, means, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(f'Mean {loss_name} Loss (OOS)')
        ax.set_ylabel('Loss')
        ax.tick_params(axis='x', rotation=30)
        ax.set_xlabel('Model')
        # Asterisk for MCS members
        mcs_set = mcs_frob if loss_name == 'Frobenius' else mcs_qlik
        for bar, m in zip(bars, MODEL_NAMES):
            if m in mcs_set:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    '*', ha='center', fontsize=12, color='black',
                )
    fig.suptitle(
        f'OOS Loss Comparison — {"SYNTHETIC" if synthetic else "REAL"} data\n'
        '(* = in 90% MCS)'
    )
    plt.tight_layout()
    fig.savefig(str(results_dir / 'exp1_losses.png'), dpi=150)
    plt.close(fig)

    # Plot 2: Cumulative loss differential (Q vs non-Q)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (q_m, b_m) in zip(axes.flat, DM_PAIRS):
        cum_frob = np.cumsum(losses_frob[b_m] - losses_frob[q_m])
        cum_qlik = np.cumsum(losses_qlik[b_m] - losses_qlik[q_m])
        ax.plot(cum_frob, label='Frob', color='#2196F3')
        ax.plot(cum_qlik, label='QLIKE', color='#F44336', linestyle='--')
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.set_title(f'{b_m} − {q_m}  (positive = Q-model better)')
        ax.legend(fontsize=8)
        ax.set_xlabel('OOS step')
        ax.set_ylabel('Cum. loss diff.')
    fig.suptitle('Cumulative Loss Differential (benchmark − Q-model)')
    plt.tight_layout()
    fig.savefig(str(results_dir / 'exp1_cum_loss_diff.png'), dpi=150)
    plt.close(fig)

    # Plot 3: Split-sample losses (heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (df, title) in zip(
        axes,
        [(split_frob, 'Frobenius'), (split_qlik, 'QLIKE')],
    ):
        im = ax.imshow(df.values, aspect='auto', cmap='RdYlGn_r')
        ax.set_xticks(range(len(MODEL_NAMES)))
        ax.set_xticklabels(MODEL_NAMES, rotation=30, ha='right')
        ax.set_yticks(range(3))
        ax.set_yticklabels(labels)
        ax.set_title(f'Split-Sample Mean {title} Loss')
        plt.colorbar(im, ax=ax)
    fig.suptitle('Split-sample analysis by pi_t magnitude')
    plt.tight_layout()
    fig.savefig(str(results_dir / 'exp1_split_sample.png'), dpi=150)
    plt.close(fig)

    print('[exp1] Plots saved to results/.')

    return {
        'losses_frob':  losses_frob,
        'losses_qlike': losses_qlik,
        'dm_results':   dm_results,
        'mcs_frob':     mcs_frob,
        'mcs_qlike':    mcs_qlik,
        'split_frob':   split_frob,
        'split_qlike':  split_qlik,
        'synthetic':    synthetic,
    }


if __name__ == '__main__':
    run_forecasting_experiment()

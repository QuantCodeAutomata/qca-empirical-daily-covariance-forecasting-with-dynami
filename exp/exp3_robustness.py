"""
Experiment 3: Robustness Analyses
==================================
Four analyses that stress-test the core HARQ-DRD result:

  Part A – Reliability split: compare losses on low-pi vs high-pi days.
  Part B – No-short-sale GMV: long-only constrained portfolio metrics.
  Part C – Multi-horizon (h=4, h=21) direct-step forecasts.
  Part D – Ex-post shrinkage alternatives as HAR-DRD baselines.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils import vech, unvech, is_spd, nearest_spd
from src.losses import frobenius_loss, qlike_loss, diebold_mariano_test
from src.models.har_drd import (
    fit_har_drd, forecast_har_drd, fit_harq_drd, forecast_harq_drd,
    precompute_drd_series,
    fit_har_drd_precomputed, forecast_har_drd_precomputed,
    fit_harq_drd_precomputed, forecast_harq_drd_precomputed,
)
from src.portfolio import (
    gmv_weights,
    gmv_weights_constrained,
    compute_turnover,
    compute_utility,
    certainty_equivalent_gain,
    insanity_filter,
)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

T_WINDOW = 1000        # rolling estimation window (days)
T_OOS = 500            # out-of-sample period length (days)
GAMMA = 3.0            # risk-aversion for CEQ calculations
TRADING_DAYS = 252     # annualisation factor
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _vk_to_n(K: int) -> int:
    """Recover matrix dimension N from vech length K = N*(N+1)//2."""
    return int(round((-1 + np.sqrt(1 + 8 * K)) / 2))


# ---------------------------------------------------------------------------
# Data loading  (with synthetic fallback)
# ---------------------------------------------------------------------------

def _nearest_spd(A: np.ndarray) -> np.ndarray:
    """Project symmetric matrix A to the nearest symmetric positive definite."""
    A = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _generate_synthetic_data(T: int = 1500, n: int = 5, M: int = 78, seed: int = 42):
    """Generate T days of synthetic realized covariances, pi vectors, and returns.

    Uses a slow-moving Cholesky AR(1) for the latent covariance and
    Wishart(V_t, M)/M for the noisy realized measure.

    Returns
    -------
    S_array  : (T, K) vech of realized covariances
    Pi_array : (T, K) quarticity-based pi proxies
    ret_array: (T, n) zero-mean daily returns drawn from N(0, V_t)
    dates    : DatetimeIndex (business days starting 1993-02-01)
    """
    rng = np.random.default_rng(seed)
    K = n * (n + 1) // 2
    i_ut, j_ut = np.triu_indices(n)

    # --- latent covariance (slow AR(1) on Cholesky) -----------------------
    rho = 0.99
    L0 = np.tril(rng.standard_normal((n, n)))
    L0 = L0 + n * np.eye(n)
    V0 = L0 @ L0.T / (2 * n)

    V = np.zeros((T, n, n))
    V[0] = V0
    for t in range(1, T):
        noise = np.tril(rng.standard_normal((n, n)))
        dL = 0.05 * noise
        Lc = np.linalg.cholesky(_nearest_spd(V[t - 1])) + dL
        Lc[np.diag_indices(n)] = np.maximum(np.abs(np.diag(Lc)), 1e-4)
        Vt = Lc @ Lc.T
        V[t] = rho * Vt + (1.0 - rho) * V0

    # --- realized covariances (Wishart proxy) and pi vectors ---------------
    S_array = np.zeros((T, K))
    Pi_array = np.zeros((T, K))
    ret_array = np.zeros((T, n))

    for t in range(T):
        Lc = np.linalg.cholesky(_nearest_spd(V[t]))
        # Realized covariance (noisy)
        R_t = rng.standard_normal((M, n)) @ Lc.T / np.sqrt(M)
        S_t = R_t.T @ R_t
        S_array[t] = S_t[i_ut, j_ut]
        # Quarticity proxy
        R2 = R_t ** 2
        Pi_t = (M / 3.0) * (R2.T @ R2)
        Pi_array[t] = Pi_t[i_ut, j_ut]
        # Daily return ~ N(0, V_t)
        ret_array[t] = rng.standard_normal(n) @ Lc.T

    dates = pd.bdate_range("1993-02-01", periods=T)
    return S_array, Pi_array, ret_array, dates


def _load_data(data_dir: Path):
    """Load pre-computed daily realized covariances, pi matrices, and returns.

    Falls back to a synthetic Wishart DGP when the expected parquet files are
    not present on disk (e.g. when running in CI or without a data download).

    Returns
    -------
    S_array  : (T, K) float64 — vech of realized covariances
    Pi_array : (T, K) float64 — vech of noise proxy matrices
    ret_array: (T, N) float64 — daily log-returns
    dates    : DatetimeIndex
    """
    s_path   = data_dir / "realized_covariances.parquet"
    pi_path  = data_dir / "pi_matrices.parquet"
    ret_path = data_dir / "daily_returns.parquet"

    if s_path.exists() and pi_path.exists() and ret_path.exists():
        print("[exp3] Loading pre-computed realized measures …")
        S_df   = pd.read_parquet(s_path)
        Pi_df  = pd.read_parquet(pi_path)
        ret_df = pd.read_parquet(ret_path)
        return (
            S_df.values.astype(np.float64),
            Pi_df.values.astype(np.float64),
            ret_df.values.astype(np.float64),
            S_df.index,
        )

    print("[exp3] Pre-computed data not found — generating SYNTHETIC data "
          "(T=1500, N=5, Wishart DGP).")
    print("       *** Results are based on SYNTHETIC data, not real markets. ***")
    return _generate_synthetic_data(T=T_WINDOW + T_OOS, n=5, M=78)


# ---------------------------------------------------------------------------
# Rolling OOS forecasts (HAR-DRD and HARQ-DRD)
# ---------------------------------------------------------------------------

def _compute_oos_forecasts(S_array, Pi_array):
    """Run rolling-window 1-step OOS forecasts for HAR-DRD and HARQ-DRD.

    Pre-computes log-vol and Fisher-corr series for the full sample once,
    then slices rolling windows — avoids re-converting 1000 matrices per step.

    Returns
    -------
    H_har  : (T_OOS, N, N) — HAR-DRD covariance forecasts
    H_harq : (T_OOS, N, N) — HARQ-DRD covariance forecasts
    """
    T = len(S_array)
    first_oos = T - T_OOS
    N = _vk_to_n(S_array.shape[1])
    H_har = np.zeros((T_OOS, N, N))
    H_harq = np.zeros((T_OOS, N, N))

    print("  Pre-computing DRD series for full sample …")
    lv_full, fc_full = precompute_drd_series(S_array, N)

    for i in range(T_OOS):
        t = first_oos + i
        lv_win = lv_full[t - T_WINDOW: t]
        fc_win = fc_full[t - T_WINDOW: t]
        pi_win = Pi_array[t - T_WINDOW: t]
        S_mean = unvech(S_array[t - T_WINDOW: t].mean(axis=0), N)

        params_har = fit_har_drd_precomputed(lv_win, fc_win, N)
        H_har[i] = insanity_filter(
            forecast_har_drd_precomputed(params_har, lv_win, fc_win), S_mean
        )

        params_harq = fit_harq_drd_precomputed(lv_win, fc_win, pi_win, N)
        H_harq[i] = insanity_filter(
            forecast_harq_drd_precomputed(params_harq, lv_win, fc_win, Pi_array[t - 1]), S_mean
        )

        if (i + 1) % 100 == 0:
            print(f"  OOS step {i + 1}/{T_OOS}")

    return H_har, H_harq, lv_full, fc_full


def _oos_pi_norms(Pi_array):
    """Euclidean norm ||pi_t|| for each OOS day."""
    first_oos = len(Pi_array) - T_OOS
    return np.linalg.norm(Pi_array[first_oos:], axis=1)


# ---------------------------------------------------------------------------
# Part A: Reliability split
# ---------------------------------------------------------------------------

def run_part_a(S_oos, H_har, H_harq, pi_norms, results_dir: Path):
    """Split OOS by ||pi_t|| (95/5 percentile) and compare losses on each subset.

    Parameters
    ----------
    S_oos    : (T_OOS, N, N) — realised covariance matrices (targets)
    H_har    : (T_OOS, N, N)
    H_harq   : (T_OOS, N, N)
    pi_norms : (T_OOS,)     — ||pi_t|| per OOS day
    """
    threshold = np.percentile(pi_norms, 95)
    masks = {
        "low_pi  (bottom 95%)": pi_norms <= threshold,
        "high_pi (top 5%)":     pi_norms > threshold,
    }

    rows = []
    for label, mask in masks.items():
        idx = np.where(mask)[0]
        l_har_frob  = np.array([frobenius_loss(H_har[t], S_oos[t]) for t in idx])
        l_har_ql    = np.array([qlike_loss(H_har[t], S_oos[t]) for t in idx])
        l_harq_frob = np.array([frobenius_loss(H_harq[t], S_oos[t]) for t in idx])
        l_harq_ql   = np.array([qlike_loss(H_harq[t], S_oos[t]) for t in idx])

        dm_frob_stat, dm_frob_pval = diebold_mariano_test(l_har_frob, l_harq_frob)
        dm_ql_stat,   dm_ql_pval   = diebold_mariano_test(l_har_ql, l_harq_ql)

        rows.append({
            "subset":           label,
            "n_days":           int(mask.sum()),
            "HAR_DRD_Frob":     np.nanmean(l_har_frob),
            "HARQ_DRD_Frob":    np.nanmean(l_harq_frob),
            "DM_Frob_stat":     dm_frob_stat,
            "DM_Frob_pval":     dm_frob_pval,
            "HAR_DRD_QLIKE":    np.nanmean(l_har_ql),
            "HARQ_DRD_QLIKE":   np.nanmean(l_harq_ql),
            "DM_QLIKE_stat":    dm_ql_stat,
            "DM_QLIKE_pval":    dm_ql_pval,
        })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, title, col_har, col_harq in [
        (axes[0], "Frobenius Loss", "HAR_DRD_Frob",  "HARQ_DRD_Frob"),
        (axes[1], "QLIKE Loss",     "HAR_DRD_QLIKE", "HARQ_DRD_QLIKE"),
    ]:
        xlabels = ["Low pi", "High pi"]
        x = np.arange(len(xlabels))
        width = 0.35
        ax.bar(x - width / 2, df[col_har],  width, label="HAR-DRD",  color="steelblue")
        ax.bar(x + width / 2, df[col_harq], width, label="HARQ-DRD", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_title(f"Part A: {title}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "exp3_part_a_reliability.png", dpi=150)
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# Part B: Long-only (no-short-sale) GMV
# ---------------------------------------------------------------------------

def run_part_b(S_oos, H_har, H_harq, ret_oos, results_dir: Path):
    """Constrained (long-only) GMV performance comparison.

    Parameters
    ----------
    ret_oos : (T_OOS, N) — daily log-returns aligned with S_oos
    """
    T = len(S_oos)

    w_har  = np.array([gmv_weights_constrained(H_har[t])  for t in range(T)])
    w_harq = np.array([gmv_weights_constrained(H_harq[t]) for t in range(T)])

    # Portfolio returns: w_t is formed at close of day t, applied to day t+1
    r_har  = np.array([w_har[t]  @ ret_oos[t + 1] for t in range(T - 1)])
    r_harq = np.array([w_harq[t] @ ret_oos[t + 1] for t in range(T - 1)])

    to_har  = np.mean([compute_turnover(w_har[t + 1],  w_har[t])  for t in range(T - 2)])
    to_harq = np.mean([compute_turnover(w_harq[t + 1], w_harq[t]) for t in range(T - 2)])

    rows = []
    for label, r_arr, to in [
        ("HAR-DRD (LO)",  r_har,  to_har),
        ("HARQ-DRD (LO)", r_harq, to_harq),
    ]:
        rows.append({
            "model":        label,
            "Ann_Vol":      r_arr.std() * np.sqrt(TRADING_DAYS),
            "Sharpe":       (r_arr.mean() / r_arr.std()) * np.sqrt(TRADING_DAYS),
            "Avg_Turnover": to,
            "Utility":      compute_utility(r_arr, gamma=GAMMA),
        })

    ceq = certainty_equivalent_gain(r_harq, r_har, gamma=GAMMA)
    rows[1]["CEQ_gain_vs_HAR"] = ceq
    rows[0]["CEQ_gain_vs_HAR"] = 0.0

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Cumulative returns
    axes[0].plot(np.cumprod(1 + r_har)  - 1, label="HAR-DRD (LO)",  color="steelblue")
    axes[0].plot(np.cumprod(1 + r_harq) - 1, label="HARQ-DRD (LO)", color="darkorange")
    axes[0].set_title("Part B: Long-Only GMV — Cumulative Returns")
    axes[0].set_xlabel("OOS day")
    axes[0].set_ylabel("Cumulative return")
    axes[0].legend()

    # Bar chart of key metrics
    metrics = ["Ann_Vol", "Sharpe", "Avg_Turnover"]
    x = np.arange(len(metrics))
    width = 0.35
    axes[1].bar(x - width / 2, [rows[0][m] for m in metrics], width,
                label="HAR-DRD (LO)",  color="steelblue")
    axes[1].bar(x + width / 2, [rows[1][m] for m in metrics], width,
                label="HARQ-DRD (LO)", color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_title("Part B: Portfolio Metrics")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / "exp3_part_b_lononly.png", dpi=150)
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# Part C: Multi-horizon direct-step forecasts
# ---------------------------------------------------------------------------

def _build_direct_horizon_dataset(S_array, Pi_array, horizon):
    """Aggregate features and targets for direct h-step forecasting.

    At each time t the forecast target is the h-step average:
        y_t = (1/h) * sum_{k=1}^{h} s_{t+k}

    The feature set is unchanged (daily, weekly, monthly lags at time t).
    Returns the same-length vech series and a matching array of h-step targets.
    """
    T = S_array.shape[0]
    max_t = T - horizon
    # Features: s_0 through s_{T-h-1} (use the window starting from lag 22)
    S_feat = S_array[:max_t]
    Pi_feat = Pi_array[:max_t]
    # Targets: rolling h-step-ahead average
    S_tgt = np.stack(
        [S_array[t + 1: t + horizon + 1].mean(axis=0) for t in range(max_t)],
        axis=0,
    )
    return S_feat, Pi_feat, S_tgt


def run_part_c(S_array, Pi_array, results_dir=None,
               lv_full: np.ndarray | None = None,
               fc_full: np.ndarray | None = None):
    """Weekly (h=4) and monthly (h=21) direct-horizon QLIKE comparison.

    Compares:
      1. HAR-DRD with h-step direct training targets.
      2. HARQ-DRD with h-step direct targets and h-aggregated pi attenuation.
      3. Daily HARQ-DRD (h=1) applied to the same h-step target (baseline).

    Parameters
    ----------
    lv_full, fc_full : pre-computed log-vol / fisher-corr arrays for the full
        sample (shape T×N and T×M).  When provided, the function avoids
        recomputing them from scratch.
    """
    T = len(S_array)
    first_oos = T - T_OOS
    N = _vk_to_n(S_array.shape[1])
    rows = []

    # Pre-compute full DRD series once if not already supplied
    if lv_full is None or fc_full is None:
        print("  [Part C] Pre-computing DRD series …")
        lv_full, fc_full = precompute_drd_series(S_array, N)

    for h_label, h in [("weekly", 4), ("monthly", 21)]:
        har_losses, harq_h_losses, harq_d_losses = [], [], []

        for i in range(T_OOS - h):
            t = first_oos + i
            lv_win = lv_full[t - T_WINDOW: t]
            fc_win = fc_full[t - T_WINDOW: t]
            pi_win = Pi_array[t - T_WINDOW: t]
            S_mean = unvech(S_array[t - T_WINDOW: t].mean(axis=0), N)

            # h-step-ahead target for evaluation
            S_target = nearest_spd(unvech(S_array[t: t + h].mean(axis=0), N))

            # --- Direct-horizon HAR-DRD ---
            # Fit on log-vol/fish-corr of truncated window (shifted by h-1)
            lv_feat = lv_win[: -(h - 1)] if h > 1 else lv_win
            fc_feat = fc_win[: -(h - 1)] if h > 1 else fc_win
            params_har_h = fit_har_drd_precomputed(lv_feat, fc_feat, N)
            H_har_h = insanity_filter(
                forecast_har_drd_precomputed(params_har_h, lv_win, fc_win), S_mean
            )

            # --- Direct-horizon HARQ-DRD ---
            pi_feat = pi_win[: -(h - 1)] if h > 1 else pi_win
            params_harq_h = fit_harq_drd_precomputed(lv_feat, fc_feat, pi_feat, N)
            pi_h = pi_win[-h:].mean(axis=0)   # h-step aggregated pi for forecast
            H_harq_h = insanity_filter(
                forecast_harq_drd_precomputed(params_harq_h, lv_win, fc_win, pi_h), S_mean
            )

            # --- Daily HARQ-DRD baseline (uses full window) ---
            params_harq_d = fit_harq_drd_precomputed(lv_win, fc_win, pi_win, N)
            H_harq_d = insanity_filter(
                forecast_harq_drd_precomputed(params_harq_d, lv_win, fc_win, pi_win[-1]), S_mean
            )

            har_losses.append(qlike_loss(H_har_h, S_target))
            harq_h_losses.append(qlike_loss(H_harq_h, S_target))
            harq_d_losses.append(qlike_loss(H_harq_d, S_target))

        rows.append({
            "horizon":               h_label,
            "h":                     h,
            "HAR_DRD_direct_QLIKE":  np.nanmean(har_losses),
            "HARQ_DRD_direct_QLIKE": np.nanmean(harq_h_losses),
            "HARQ_DRD_daily_QLIKE":  np.nanmean(harq_d_losses),
        })

    df = pd.DataFrame(rows)

    if results_dir is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(rows))
        width = 0.25
        ax.bar(x - width, df["HAR_DRD_direct_QLIKE"],  width, label="HAR-DRD (direct)", color="steelblue")
        ax.bar(x,          df["HARQ_DRD_direct_QLIKE"], width, label="HARQ-DRD (direct)", color="darkorange")
        ax.bar(x + width,  df["HARQ_DRD_daily_QLIKE"],  width, label="HARQ-DRD (daily)",  color="green")
        ax.set_xticks(x)
        ax.set_xticklabels([r["horizon"] for r in rows])
        ax.set_ylabel("Mean QLIKE")
        ax.set_title("Part C: Multi-Horizon Direct Forecasts")
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "exp3_part_c_multihorizon.png", dpi=150)
        plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# Part D: Ex-post shrinkage alternatives
# ---------------------------------------------------------------------------

def _identity_target(S_bar):
    """Scaled-identity shrinkage target: (mean variance) * I_N."""
    N = S_bar.shape[0]
    mean_var = np.trace(S_bar) / N
    return mean_var * np.eye(N)


def _equicorr_target(S_bar):
    """Equicorrelation shrinkage target derived from S_bar.

    H_eq_{ij} = rho_bar * sigma_i * sigma_j   (i ≠ j)
    H_eq_{ii} = sigma_i^2
    where rho_bar is the average off-diagonal sample correlation.
    """
    N = S_bar.shape[0]
    std = np.sqrt(np.maximum(np.diag(S_bar), 1e-12))
    corr = S_bar / np.outer(std, std)
    np.fill_diagonal(corr, 0.0)
    rho_bar = corr.sum() / (N * (N - 1))
    H_eq = rho_bar * np.outer(std, std)
    np.fill_diagonal(H_eq, std ** 2)
    return H_eq


def _factor_target(H_hat, S_bar):
    """Single-factor (market) shrinkage target via PCA of S_bar.

    F = lambda_1 * v_1 v_1'  +  diag(max(diag(H_hat) - lambda_1 * v_1^2, eps))
    """
    eigvals, eigvecs = np.linalg.eigh(S_bar)
    v1 = eigvecs[:, -1]
    lam1 = eigvals[-1]
    rank1 = lam1 * np.outer(v1, v1)
    d_resid = np.maximum(np.diag(H_hat) - lam1 * v1 ** 2, 1e-8)
    return rank1 + np.diag(d_resid)


def _select_rho(H_har_val, H_tgt_val, S_val, rho_grid=None):
    """Cross-validate shrinkage intensity on a validation set (Frobenius criterion)."""
    if rho_grid is None:
        rho_grid = np.linspace(0.0, 1.0, 41)
    best_rho, best_loss = 0.0, np.inf
    for rho in rho_grid:
        shrunk = [(1.0 - rho) * H_har_val[t] + rho * H_tgt_val[t]
                  for t in range(len(H_har_val))]
        loss = np.nanmean([frobenius_loss(shrunk[t], S_val[t])
                           for t in range(len(shrunk))])
        if loss < best_loss:
            best_loss, best_rho = loss, rho
    return best_rho


def run_part_d(S_oos, H_har, H_harq, S_bar, results_dir: Path):
    """Compare ex-post shrinkage of HAR-DRD against HARQ-DRD benchmark.

    Three shrinkage targets:
      (a) Scaled identity:     (1 - rho) * H_HAR + rho * mean_var * I
      (b) Equicorrelation:     (1 - rho) * H_HAR + rho * H_equicorr
      (c) Factor (market PCA): (1 - rho) * H_HAR + rho * H_factor

    rho is selected by cross-validation on the first half of the OOS period.
    """
    T = len(S_oos)
    n_val = T // 2
    H_har_val, H_har_tst  = H_har[:n_val],  H_har[n_val:]
    H_harq_tst             = H_harq[n_val:]
    S_val, S_tst           = S_oos[:n_val],  S_oos[n_val:]

    target_fns = {
        "identity":  lambda t: _identity_target(S_bar),
        "equicorr":  lambda t: _equicorr_target(S_bar),
        "factor":    lambda t: _factor_target(H_har[t], S_bar),
    }

    rows = []

    # HARQ-DRD benchmark on test half
    rows.append({
        "model": "HARQ-DRD",
        "rho":   float("nan"),
        "Frob":  np.nanmean([frobenius_loss(H_harq_tst[t], S_tst[t]) for t in range(len(S_tst))]),
        "QLIKE": np.nanmean([qlike_loss(H_harq_tst[t], S_tst[t]) for t in range(len(S_tst))]),
    })

    for tname, tgt_fn in target_fns.items():
        # Compute targets for validation and test sets
        H_tgt_val = [tgt_fn(t) for t in range(n_val)]
        H_tgt_tst = [tgt_fn(n_val + t) for t in range(T - n_val)]

        rho_opt = _select_rho(H_har_val, H_tgt_val, S_val)

        shrunk_tst = [(1.0 - rho_opt) * H_har_tst[t] + rho_opt * H_tgt_tst[t]
                      for t in range(len(S_tst))]
        rows.append({
            "model": f"HAR-DRD + {tname}",
            "rho":   rho_opt,
            "Frob":  np.nanmean([frobenius_loss(shrunk_tst[t], S_tst[t]) for t in range(len(S_tst))]),
            "QLIKE": np.nanmean([qlike_loss(shrunk_tst[t], S_tst[t]) for t in range(len(S_tst))]),
        })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["darkorange", "steelblue", "green", "firebrick"]
    for ax, metric in [(axes[0], "Frob"), (axes[1], "QLIKE")]:
        ax.bar(df["model"], df[metric], color=colors[: len(df)])
        ax.set_title(f"Part D: {metric} — Shrinkage vs HARQ-DRD")
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(results_dir / "exp3_part_d_shrinkage.png", dpi=150)
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_robustness_experiment():
    """Run all four robustness analyses and save results."""
    data_dir = Path(__file__).parent.parent / "data"
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading data …")
    S_array, Pi_array, ret_array, dates = _load_data(data_dir)
    T = len(S_array)
    first_oos = T - T_OOS
    print(f"  T={T} days, first OOS day index={first_oos}")

    N           = _vk_to_n(S_array.shape[1])
    S_oos_vech  = S_array[first_oos:]
    S_oos       = np.array([unvech(v, N) for v in S_oos_vech])
    ret_oos     = ret_array[first_oos:]
    S_bar       = unvech(S_array[:first_oos].mean(axis=0), N)

    # ── OOS forecasts ─────────────────────────────────────────────────────
    print("Computing OOS forecasts (rolling window) …")
    H_har, H_harq, lv_full, fc_full = _compute_oos_forecasts(S_array, Pi_array)
    pi_norms = _oos_pi_norms(Pi_array)

    all_frames = {}

    # ── Part A ────────────────────────────────────────────────────────────
    print("\n── Part A: Reliability Split ──")
    df_a = run_part_a(S_oos, H_har, H_harq, pi_norms, RESULTS_DIR)
    print(df_a.to_string(index=False))
    all_frames["A"] = df_a

    # ── Part B ────────────────────────────────────────────────────────────
    print("\n── Part B: Long-Only GMV ──")
    df_b = run_part_b(S_oos, H_har, H_harq, ret_oos, RESULTS_DIR)
    print(df_b.to_string(index=False))
    all_frames["B"] = df_b

    # ── Part C ────────────────────────────────────────────────────────────
    print("\n── Part C: Multi-Horizon Forecasts ──")
    df_c = run_part_c(S_array, Pi_array, results_dir=RESULTS_DIR,
                      lv_full=lv_full, fc_full=fc_full)
    print(df_c.to_string(index=False))
    all_frames["C"] = df_c

    # ── Part D ────────────────────────────────────────────────────────────
    print("\n── Part D: Ex-Post Shrinkage ──")
    df_d = run_part_d(S_oos, H_har, H_harq, S_bar, RESULTS_DIR)
    print(df_d.to_string(index=False))
    all_frames["D"] = df_d

    # ── Save combined CSV ─────────────────────────────────────────────────
    combined = pd.concat(
        {f"part_{k}": v for k, v in all_frames.items()},
        names=["part"],
    ).reset_index(level=0)
    out_path = RESULTS_DIR / "exp3_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    run_robustness_experiment()

"""
Experiment 4: Monte Carlo Simulation Study
==========================================
Evaluates HAR vs. HARQ covariance forecasting in a controlled environment
where the true latent covariance process is known.

Design
------
* N = 5 assets (random selection from a 10-stock universe).
* T = 2000 days of synthetic latent covariance matrices drawn from Wishart(dof=20).
* Intraday returns simulated using a U-shaped diurnal variance pattern:
      g(u) = A + B·(exp(−a·u) + exp(−b·(1−u))),   A=0.75, B=0.25, a=b=10
  normalised to sum to 1 over M_base = 390 intraday intervals (1-minute bars).
* Sampling frequencies M ∈ {390, 78, 26}  (1-min, 5-min, 15-min).
* Rolling 1 000-day estimation window, 1 000 OOS days.
* Benchmark: HAR_∞ fit on the true (latent) Sigma_t series.

Output
------
results/exp4_results.csv  — mean metrics by model × frequency
results/exp4_turnover.png — turnover vs. sampling frequency
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
from src.realized_measures import (
    compute_multivariate_kernel_cov,
    compute_quarticity_proxy,
    compute_pi_matrix,
)
from src.models.har import fit_har_ols, forecast_har, fit_harq_ols, forecast_harq, build_har_features
from src.portfolio import gmv_weights

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

N_UNIVERSE  = 10      # full stock universe
N_ASSETS    = 5       # assets used in experiment (random selection)
T_TOTAL     = 1500    # total simulated days  (reduced from 2000 to match project T=1500)
T_EST       = 1000    # rolling estimation window
T_OOS       = 500     # out-of-sample period  (reduced from 1000 to match T_OOS=500)
DOF         = 20      # Wishart degrees of freedom (controls variation)
M_BASE      = 390     # base intraday frequency (1-minute bars)
M_VALUES    = [390, 78, 26]   # target sampling frequencies: 1-min, 5-min, 15-min
DIURNAL_A   = 0.75
DIURNAL_B   = 0.25
DIURNAL_a   = 10.0
DIURNAL_b   = 10.0
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _generate_wishart_covariances(N, T, dof, rng):
    """Draw T positive-definite matrices from Wishart(dof, I_N / dof).

    Each Sigma_t = (1/dof) * A_t A_t^T with A_t ~ N(0, I_{N×dof}).
    This ensures E[Sigma_t] = I_N.
    """
    Sigma = np.zeros((T, N, N))
    for t in range(T):
        A = rng.standard_normal((N, dof))
        Sigma[t] = A @ A.T / dof
    return Sigma


def _make_diurnal_weights(M_base, A=DIURNAL_A, B=DIURNAL_B, a=DIURNAL_a, b=DIURNAL_b):
    """Return the normalised U-shaped intraday variance weights g_i.

    g(u) = A + B·(exp(−a·u) + exp(−b·(1−u)))
    evaluated at u_i = (i + 0.5) / M_base for i = 0, …, M_base−1,
    then normalised so sum(g) = 1.
    """
    u = (np.arange(M_base) + 0.5) / M_base
    g = A + B * (np.exp(-a * u) + np.exp(-b * (1.0 - u)))
    return g / g.sum()


def _simulate_intraday(Sigma, g, M_base, rng):
    """Simulate daily intraday return paths for all T days.

    At interval i:  r_i ~ N(0, g_i * Sigma_t)
    So sum_i Cov(r_i) = Sigma_t · sum_i g_i = Sigma_t.

    Returns R_all of shape (T, M_base, N).
    """
    T, N, _ = Sigma.shape
    sqrt_g = np.sqrt(g)                         # (M_base,)
    # Pre-compute Cholesky factors for all days
    L = np.zeros_like(Sigma)
    for t in range(T):
        L[t] = np.linalg.cholesky(Sigma[t])    # lower-triangular, (N, N)
    R_all = np.zeros((T, M_base, N))
    for t in range(T):
        Z = rng.standard_normal((M_base, N))    # (M_base, N) i.i.d. N(0,I)
        R_all[t] = (Z * sqrt_g[:, None]) @ L[t].T
    return R_all


def _aggregate(R_base, group_size):
    """Sum successive groups of `group_size` intraday returns.

    R_base : (M_base, N)
    Returns  (M_base // group_size, N)
    """
    M_base, N = R_base.shape
    M = M_base // group_size
    return R_base[: M * group_size].reshape(M, group_size, N).sum(axis=1)


def _realized_measures_for_day(R_agg, M):
    """Compute realised covariance (kernel) and pi vector from M×N returns.

    Returns
    -------
    s_vech  : (K,)  — vech of the MK realized covariance
    pi_vech : (K,)  — vech of the N×N noise-proxy matrix Pi_t
                      (matches the K-element vech layout of S_t so that
                       fit_harq_ols receives aligned (T, K) arrays)
    """
    N = R_agg.shape[1]
    S  = compute_multivariate_kernel_cov(R_agg)
    S  = nearest_spd(S)
    IQ = compute_quarticity_proxy(R_agg, n_assets=N)
    Pi = compute_pi_matrix(IQ)
    return vech(S), vech(Pi)   # both (K,)


# ---------------------------------------------------------------------------
# Rolling OOS estimation
# ---------------------------------------------------------------------------

def _rolling_oos(S_arr, Pi_arr, Sigma_vech, Sigma, N):
    """Run rolling-window OOS and return portfolio metrics per model.

    Models:
      HAR       – HAR fit on noisy S_t
      HARQ      – HARQ fit on S_t with pi_t attenuation
      HAR_inf   – HAR fit on true latent Sigma_t  (oracle benchmark)

    Returns dict: model → dict of metric arrays (length T_OOS − 1 for turnover).
    """
    K = N * (N + 1) // 2
    model_names = ["HAR", "HARQ", "HAR_inf"]
    metrics = {m: {"turnover": [], "port_var": [], "weight_dist": []} for m in model_names}
    prev_w = {m: None for m in model_names}

    for i in range(T_OOS):
        t = T_EST + i

        s_win   = S_arr[t - T_EST: t]       # (T_EST, K)
        pi_win  = Pi_arr[t - T_EST: t]      # (T_EST, K)  — vech of noise-proxy matrix
        sig_win = Sigma_vech[t - T_EST: t]  # (T_EST, K)

        # ── HAR forecast ─────────────────────────────────────────────────
        params_har = fit_har_ols(s_win)
        s_d = s_win[-1];  s_w = s_win[-5:].mean(0);  s_m = s_win[-22:].mean(0)
        feats = build_har_features(s_d, s_w, s_m)
        H_har = nearest_spd(unvech(forecast_har(params_har, feats), N))

        # ── HARQ forecast ────────────────────────────────────────────────
        params_harq = fit_harq_ols(s_win, pi_win)
        pi_d = pi_win[-1]                    # (N,) last noise-proxy vector
        H_harq = nearest_spd(unvech(forecast_harq(params_harq, feats, pi_d), N))

        # ── HAR_∞ forecast (oracle on true Sigma) ────────────────────────
        params_inf = fit_har_ols(sig_win)
        sd_inf = sig_win[-1]; sw_inf = sig_win[-5:].mean(0); sm_inf = sig_win[-22:].mean(0)
        feats_inf = build_har_features(sd_inf, sw_inf, sm_inf)
        H_inf = nearest_spd(unvech(forecast_har(params_inf, feats_inf), N))

        # ── True Sigma_t and optimal GMV ─────────────────────────────────
        Sig_t = Sigma[t]                        # (N, N) true covariance
        w_opt = gmv_weights(Sig_t)

        # ── Portfolio weights and metrics ─────────────────────────────────
        for m_name, H_fc in [("HAR", H_har), ("HARQ", H_harq), ("HAR_inf", H_inf)]:
            w_t = gmv_weights(H_fc)
            metrics[m_name]["port_var"].append(float(w_t @ Sig_t @ w_t))
            metrics[m_name]["weight_dist"].append(float(np.linalg.norm(w_t - w_opt)))
            if prev_w[m_name] is not None:
                to = 0.5 * np.sum(np.abs(w_t - prev_w[m_name]))
                metrics[m_name]["turnover"].append(float(to))
            prev_w[m_name] = w_t

    return metrics


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_simulation_experiment(seed: int = 42):
    """Execute the full Monte Carlo simulation and save results."""
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── Asset selection ───────────────────────────────────────────────────
    all_assets = np.arange(N_UNIVERSE)
    selected   = rng.choice(all_assets, size=N_ASSETS, replace=False)
    print(f"Selected assets (indices): {selected}")

    # ── Latent covariances ───────────────────────────────────────────────
    print(f"Generating {T_TOTAL} Wishart covariance matrices (N={N_ASSETS}, dof={DOF}) …")
    Sigma = _generate_wishart_covariances(N_ASSETS, T_TOTAL, DOF, rng)

    Sigma_vech = np.array([vech(Sigma[t]) for t in range(T_TOTAL)])  # (T, K)

    # ── Diurnal weights ──────────────────────────────────────────────────
    g = _make_diurnal_weights(M_BASE)
    print(f"Diurnal weights: min={g.min():.4f}, max={g.max():.4f}, sum={g.sum():.6f}")

    # ── Simulate intraday returns at M_BASE ──────────────────────────────
    print(f"Simulating {T_TOTAL} days × {M_BASE} intraday intervals …")
    R_all = _simulate_intraday(Sigma, g, M_BASE, rng)   # (T, M_BASE, N)

    # ── Compute realized measures for each target frequency ───────────────
    K = N_ASSETS * (N_ASSETS + 1) // 2
    freq_data = {}   # M → (S_arr, Pi_arr)

    for M in M_VALUES:
        group_size = M_BASE // M
        label = f"M={M} ({M_BASE // M}-min agg)"
        print(f"Computing realized measures: {label} …")
        S_arr  = np.zeros((T_TOTAL, K))
        Pi_arr = np.zeros((T_TOTAL, K))   # pi_t is K-vector: vech of N×N Pi matrix
        for t in range(T_TOTAL):
            R_agg = _aggregate(R_all[t], group_size)    # (M, N)
            s_vech, pi_vech = _realized_measures_for_day(R_agg, M)
            S_arr[t]  = s_vech
            Pi_arr[t] = pi_vech
        freq_data[M] = (S_arr, Pi_arr)

    # ── Rolling OOS at each frequency ────────────────────────────────────
    all_results = []

    for M in M_VALUES:
        bar_label = {390: "1-min", 78: "5-min", 26: "15-min"}[M]
        print(f"Rolling OOS estimation at {bar_label} (M={M}) …")
        S_arr, Pi_arr = freq_data[M]
        metrics = _rolling_oos(S_arr, Pi_arr, Sigma_vech, Sigma, N_ASSETS)

        for m_name, m_metrics in metrics.items():
            row = {
                "model":            m_name,
                "M":                M,
                "freq_label":       bar_label,
                "mean_turnover":    float(np.mean(m_metrics["turnover"])),
                "mean_port_var":    float(np.mean(m_metrics["port_var"])),
                "mean_weight_dist": float(np.mean(m_metrics["weight_dist"])),
                "std_turnover":     float(np.std(m_metrics["turnover"])),
            }
            all_results.append(row)
            print(f"  {m_name:10s}  turnover={row['mean_turnover']:.4f}  "
                  f"port_var={row['mean_port_var']:.6f}  "
                  f"weight_dist={row['mean_weight_dist']:.4f}")

    df = pd.DataFrame(all_results)

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "exp4_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _make_plots(df)

    return df


def _make_plots(df):
    """Produce the three summary plots from the simulation results."""
    RESULTS_DIR.mkdir(exist_ok=True)

    model_styles = {
        "HAR":     {"color": "steelblue",  "marker": "o", "ls": "-"},
        "HARQ":    {"color": "darkorange", "marker": "s", "ls": "-"},
        "HAR_inf": {"color": "green",      "marker": "^", "ls": "--"},
    }
    freq_labels = {M: lbl for M, lbl in zip(M_VALUES, ["1-min", "5-min", "15-min"])}
    x_pos = {390: 0, 78: 1, 26: 2}

    # ── Figure 1: Turnover vs frequency ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for m_name, style in model_styles.items():
        sub = df[df["model"] == m_name].sort_values("M", ascending=False)
        xs  = [x_pos[M] for M in sub["M"]]
        ax.plot(xs, sub["mean_turnover"], label=m_name,
                color=style["color"], marker=style["marker"], ls=style["ls"],
                lw=2, ms=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["1-min (M=390)", "5-min (M=78)", "15-min (M=26)"])
    ax.set_ylabel("Mean Daily Turnover  (½ · Σ|Δw|)")
    ax.set_xlabel("Sampling Frequency")
    ax.set_title("Simulation: Portfolio Turnover vs. Sampling Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_turnover.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: True portfolio variance vs frequency ────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for m_name, style in model_styles.items():
        sub = df[df["model"] == m_name].sort_values("M", ascending=False)
        xs  = [x_pos[M] for M in sub["M"]]
        ax.plot(xs, sub["mean_port_var"], label=m_name,
                color=style["color"], marker=style["marker"], ls=style["ls"],
                lw=2, ms=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["1-min", "5-min", "15-min"])
    ax.set_ylabel("Mean True Portfolio Variance  (w' Σ w)")
    ax.set_xlabel("Sampling Frequency")
    ax.set_title("Simulation: Realized Portfolio Variance vs. Sampling Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_port_variance.png", dpi=150)
    plt.close(fig)

    # ── Figure 3: Weight distance to oracle vs frequency ─────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for m_name, style in model_styles.items():
        sub = df[df["model"] == m_name].sort_values("M", ascending=False)
        xs  = [x_pos[M] for M in sub["M"]]
        ax.plot(xs, sub["mean_weight_dist"], label=m_name,
                color=style["color"], marker=style["marker"], ls=style["ls"],
                lw=2, ms=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["1-min", "5-min", "15-min"])
    ax.set_ylabel("Mean ||w − w*||₂")
    ax.set_xlabel("Sampling Frequency")
    ax.set_title("Simulation: Distance to Oracle GMV vs. Sampling Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_weight_distance.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 4: Monte Carlo simulation study")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_simulation_experiment(seed=args.seed)

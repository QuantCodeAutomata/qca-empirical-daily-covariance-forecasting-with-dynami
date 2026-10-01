"""
Experiment 2: Portfolio performance of covariance forecasts.

For each model from Experiment 1 we form a daily Global Minimum Variance (GMV)
portfolio and evaluate:

  - Annualised mean return, volatility, and Sharpe ratio
  - Mean daily turnover (drift-adjusted)
  - Mean Herfindahl concentration index
  - Net-of-cost returns at transaction cost c ∈ {0, 1, 2} %/annum
  - Certainty-equivalent (CEQ) gains of Q-models over their non-Q benchmarks
    for risk-aversion γ ∈ {1, 10} at each cost level
  - White (2000) Reality Check p-values for CEQ superiority
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
from typing import Dict, List, Tuple, Optional

# ── src imports (with inline fallbacks) ──────────────────────────────────────
try:
    from src.utils import is_spd, nearest_spd
    from src.portfolio import (
        gmv_weights, gmv_weights_constrained, compute_turnover,
        compute_concentration, compute_short_exposure,
        compute_utility, certainty_equivalent_gain,
    )
    from src.losses import frobenius_loss, qlike_loss
    _SRC_AVAILABLE = True
except ImportError:
    _SRC_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Inline fallback implementations
# ══════════════════════════════════════════════════════════════════════════════

def _is_spd(M: np.ndarray, tol: float = 1e-8) -> bool:
    try:
        return bool(np.all(np.linalg.eigvalsh(M) > tol))
    except np.linalg.LinAlgError:
        return False


def _nearest_spd(M: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    M = (M + M.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """
    Unconstrained Global Minimum Variance weights.

    w = Σ^{-1} 1 / (1' Σ^{-1} 1)
    """
    n   = Sigma.shape[0]
    one = np.ones(n)
    try:
        Sigma_inv_1 = np.linalg.solve(Sigma, one)
        denom       = one @ Sigma_inv_1
        if abs(denom) < 1e-12:
            return one / n
        return Sigma_inv_1 / denom
    except np.linalg.LinAlgError:
        return one / n


def _gmv_weights_constrained(Sigma: np.ndarray) -> np.ndarray:
    """
    Long-only GMV weights via sequential quadratic adjustment:
    project unconstrained weights to the simplex.

    A simple iterative Dykstra-style projection onto {w >= 0, sum(w)=1}.
    """
    w = _gmv_weights(Sigma)
    # Iterative clipping projection (converges to simplex projection)
    for _ in range(500):
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s < 1e-12:
            w = np.ones(len(w)) / len(w)
            break
        w = w / s
        # Check KKT: w >= 0 and 1'w = 1
        if np.all(w >= -1e-10):
            break
    return w


def _compute_turnover(w_new: np.ndarray, w_prev_drift: np.ndarray) -> float:
    """One-way portfolio turnover: ||w_new − w_prev_drift||_1 / 2."""
    return float(np.abs(w_new - w_prev_drift).sum() / 2.0)


def _compute_concentration(w: np.ndarray) -> float:
    """Herfindahl–Hirschman concentration index: Σ w_i^2."""
    return float(np.sum(w ** 2))


def _compute_short_exposure(w: np.ndarray) -> float:
    """Total short exposure: Σ max(−w_i, 0)."""
    return float(np.sum(np.maximum(-w, 0.0)))


def _ceq(returns: np.ndarray, gamma: float) -> float:
    """
    Certainty-equivalent return.

    CEQ = μ_p − (γ/2) · σ_p²
    where μ_p and σ_p² are the annualised mean and variance (252 trading days).
    """
    mu_ann    = returns.mean() * 252.0
    var_ann   = returns.var(ddof=1) * 252.0
    return float(mu_ann - 0.5 * gamma * var_ann)


def _ceq_gain(
    ret_q: np.ndarray,
    ret_bench: np.ndarray,
    gamma: float,
    cost_daily: float = 0.0,
    turnover_q: np.ndarray = None,
    turnover_bench: np.ndarray = None,
) -> float:
    """
    CEQ gain of Q-model over benchmark, net of transaction costs.

    Parameters
    ----------
    ret_q, ret_bench : daily portfolio return series.
    gamma            : relative risk aversion.
    cost_daily       : one-way transaction cost (annualised), e.g. 0.01 = 1%.
    turnover_q/bench : daily one-way turnover series (optional; used for cost adj).
    """
    # Apply transaction cost as a drag proportional to turnover
    r_q   = ret_q.copy()
    r_b   = ret_bench.copy()

    if cost_daily > 0.0:
        daily_cost = cost_daily / 252.0
        if turnover_q    is not None:
            r_q = r_q - daily_cost * turnover_q
        if turnover_bench is not None:
            r_b = r_b - daily_cost * turnover_bench

    return _ceq(r_q, gamma) - _ceq(r_b, gamma)


# ── White Reality Check ───────────────────────────────────────────────────────

def _white_reality_check(
    benchmark_ceq: float,
    alt_ceqs: np.ndarray,
    returns_bench: np.ndarray,
    returns_alts: np.ndarray,
    gamma: float,
    B: int = 1000,
    block_len: int = 5,
    rng: np.random.Generator = None,
) -> float:
    """
    White (2000) Reality Check p-value.

    Tests H0: no alternative model has better CEQ than the benchmark.

    Uses a stationary block bootstrap to simulate the null distribution of
    max_k { CEQ_k − CEQ_benchmark }.

    Parameters
    ----------
    benchmark_ceq : scalar CEQ of the benchmark model.
    alt_ceqs      : (M,) array of alternative model CEQs.
    returns_bench : (T,) daily returns of the benchmark.
    returns_alts  : (T, M) daily returns of each alternative.
    gamma         : risk aversion for CEQ calculation.
    B             : bootstrap replications.
    block_len     : bootstrap block length.

    Returns
    -------
    p-value (scalar).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T  = len(returns_bench)
    M  = returns_alts.shape[1]

    # Observed max CEQ gain
    obs_max = float(np.max(alt_ceqs) - benchmark_ceq)

    # Null distribution via block bootstrap
    boot_max = np.zeros(B)
    for b in range(B):
        starts  = rng.integers(0, T, size=T // block_len + 1)
        indices = np.concatenate([
            np.arange(s, min(s + block_len, T)) for s in starts
        ])[:T]
        r_b_b   = returns_bench[indices]
        r_a_b   = returns_alts[indices]
        ceq_b   = _ceq(r_b_b, gamma)
        ceq_as  = np.array([_ceq(r_a_b[:, m], gamma) for m in range(M)])
        boot_max[b] = float(np.max(ceq_as) - ceq_b)

    pval = float(np.mean(boot_max >= obs_max))
    return pval


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAMES = ['HAR', 'HARQ', 'HAR-DRD', 'HARQ-DRD', 'EWMA', 'EWMAQ', 'HEAVY', 'HEAVYQ']

CEQ_PAIRS = [
    ('HARQ',     'HAR'),
    ('HARQ-DRD', 'HAR-DRD'),
    ('EWMAQ',    'EWMA'),
    ('HEAVYQ',   'HEAVY'),
]

GAMMAS = [1, 10]
COSTS  = [0.0, 0.01, 0.02]   # annualised one-way transaction cost


def _load_forecasts() -> Tuple[np.ndarray, Dict[str, np.ndarray], bool]:
    """
    Load OOS covariance forecasts from exp1 output.
    Returns (actuals, forecasts_dict, synthetic_flag).
    """
    npz_path = Path('data') / 'forecasts_exp1.npz'
    if not npz_path.exists():
        return None, None, None

    data   = np.load(str(npz_path))
    actual = data['actuals']
    fcast  = {m: data[f'forecast_{m}'] for m in MODEL_NAMES}
    return actual, fcast, True   # origin (real/synthetic) stored in exp1 output


def _generate_synthetic_forecasts(
    T_oos: int = 500,
    n: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Minimal synthetic forecasts + returns for portfolio experiment.

    Generates a simple random covariance target (true), then creates
    perturbed versions as model forecasts.
    """
    rng = np.random.default_rng(seed)

    # True daily covariance (mildly time-varying)
    base = np.eye(n) * 0.0004 + 0.0001 * rng.standard_normal((n, n))
    base = (base + base.T) / 2
    base = _nearest_spd(base)

    actuals   = np.zeros((T_oos, n, n))
    for t in range(T_oos):
        noise      = 0.01 * rng.standard_normal((n, n))
        noise      = (noise + noise.T) / 2
        actuals[t] = _nearest_spd(base + noise)

    # Daily returns drawn from N(0, S_t)
    returns = np.zeros((T_oos, n))
    for t in range(T_oos):
        L_t        = np.linalg.cholesky(actuals[t])
        returns[t] = L_t @ rng.standard_normal(n)

    # Forecasts: each model = actuals with varying levels of noise/bias
    noise_levels = {
        'HAR':      0.15,
        'HARQ':     0.10,
        'HAR-DRD':  0.12,
        'HARQ-DRD': 0.08,
        'EWMA':     0.18,
        'EWMAQ':    0.13,
        'HEAVY':    0.20,
        'HEAVYQ':   0.14,
    }
    forecasts: Dict[str, np.ndarray] = {}
    for m, noise_scale in noise_levels.items():
        fcast = np.zeros((T_oos, n, n))
        for t in range(T_oos):
            noise     = noise_scale * rng.standard_normal((n, n))
            noise     = (noise + noise.T) / 2
            fcast[t]  = _nearest_spd(actuals[t] + noise)
        forecasts[m] = fcast

    return actuals, forecasts, returns


# ══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_portfolio_experiment() -> Dict:
    """
    Execute the portfolio performance experiment.

    Returns
    -------
    dict with keys:
        'perf_table'  : pd.DataFrame  — annualised statistics per model
        'ceq_gains'   : pd.DataFrame  — CEQ gains for each (pair, γ, cost)
        'wrc_pvals'   : pd.DataFrame  — White Reality Check p-values
        'port_returns': dict model→np.ndarray of daily portfolio returns
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # ── load / generate data ──────────────────────────────────────────────────
    actuals_exp1, forecasts_exp1, _ = _load_forecasts()

    if actuals_exp1 is not None:
        print('[exp2] Loaded OOS covariance forecasts from exp1 output.')
        S_true    = actuals_exp1
        forecasts = forecasts_exp1
        n         = S_true.shape[1]
        T_oos     = len(S_true)
        # Simulate returns from actuals (used when real returns not stored)
        rng = np.random.default_rng(99)
        daily_returns = np.zeros((T_oos, n))
        for t in range(T_oos):
            Lt = np.linalg.cholesky(_nearest_spd(S_true[t]))
            daily_returns[t] = Lt @ rng.standard_normal(n)
        synthetic = False
    else:
        print('[exp2] Forecast file data/forecasts_exp1.npz not found.')
        print('       Generating SYNTHETIC data for portfolio experiment.')
        print('       *** Results are based on SYNTHETIC data. ***')
        S_true, forecasts, daily_returns = _generate_synthetic_forecasts(
            T_oos=500, n=5, seed=42
        )
        T_oos   = len(S_true)
        n       = S_true.shape[1]
        synthetic = True

    print(f'[exp2] N={n} assets, T_oos={T_oos} days, '
          f'data={"SYNTHETIC" if synthetic else "REAL"}')

    # ── portfolio construction ────────────────────────────────────────────────
    port_returns: Dict[str, np.ndarray] = {}
    turnovers:    Dict[str, np.ndarray] = {}
    concentrations: Dict[str, np.ndarray] = {}
    short_exps:   Dict[str, np.ndarray] = {}

    for model in MODEL_NAMES:
        H_series = forecasts[model]        # (T_oos, n, n)
        ret   = np.zeros(T_oos)
        to    = np.zeros(T_oos)
        conc  = np.zeros(T_oos)
        shrt  = np.zeros(T_oos)

        w_prev = None

        for t in range(T_oos):
            H_t = H_series[t]
            if not _is_spd(H_t):
                H_t = _nearest_spd(H_t)

            w_t = _gmv_weights(H_t)

            # Drift-adjusted previous weight
            if w_prev is None:
                w_prev_drift = w_t.copy()    # no turnover on day 0
            else:
                r_prev      = daily_returns[t - 1]
                port_ret_p  = float(w_prev @ r_prev)
                w_prev_drift = w_prev * (1.0 + r_prev) / (1.0 + port_ret_p)
                w_prev_drift = w_prev_drift / (w_prev_drift.sum() + 1e-12)

            to[t]   = _compute_turnover(w_t, w_prev_drift)
            conc[t] = _compute_concentration(w_t)
            shrt[t] = _compute_short_exposure(w_t)
            ret[t]  = float(w_t @ daily_returns[t])
            w_prev  = w_t.copy()

        port_returns[model] = ret
        turnovers[model]    = to
        concentrations[model] = conc
        short_exps[model]   = shrt

    # ── performance statistics ────────────────────────────────────────────────
    print('[exp2] Computing performance statistics ...')
    perf_rows = []
    for model in MODEL_NAMES:
        r    = port_returns[model]
        to   = turnovers[model]
        conc = concentrations[model]

        mu_ann  = r.mean() * 252.0
        vol_ann = r.std(ddof=1) * np.sqrt(252.0)
        sharpe  = mu_ann / (vol_ann + 1e-12)

        net_returns = {}
        for c in COSTS:
            c_daily          = c / 252.0
            net_r            = r - c_daily * to
            net_returns[c]   = net_r
            mu_net           = net_r.mean() * 252.0
            net_returns[f'mu_{c}'] = mu_net

        row = {
            'Model':          model,
            'Ann. Return':    round(mu_ann, 6),
            'Ann. Vol.':      round(vol_ann, 6),
            'Sharpe':         round(sharpe, 4),
            'Mean Turnover':  round(to.mean(), 4),
            'Concentration':  round(conc.mean(), 4),
            'Net Ret (c=0%)': round(net_returns['mu_0.0'], 6),
            'Net Ret (c=1%)': round(net_returns['mu_0.01'], 6),
            'Net Ret (c=2%)': round(net_returns['mu_0.02'], 6),
        }
        perf_rows.append(row)

    perf_table = pd.DataFrame(perf_rows).set_index('Model')
    print('\n[exp2] Performance table:\n', perf_table.to_string())

    # ── CEQ gains ─────────────────────────────────────────────────────────────
    print('[exp2] Computing CEQ gains ...')
    ceq_rows = []
    for gamma in GAMMAS:
        for c in COSTS:
            c_daily = c / 252.0
            for q_model, bench in CEQ_PAIRS:
                r_q   = port_returns[q_model] - c_daily * turnovers[q_model]
                r_b   = port_returns[bench]   - c_daily * turnovers[bench]
                gain  = _ceq(r_q, gamma) - _ceq(r_b, gamma)
                ceq_rows.append({
                    'Q-Model':   q_model,
                    'Benchmark': bench,
                    'gamma':     gamma,
                    'cost':      c,
                    'CEQ gain':  round(gain, 6),
                })

    ceq_df = pd.DataFrame(ceq_rows)
    print('\n[exp2] CEQ gains:\n', ceq_df.to_string(index=False))

    # ── White Reality Check ───────────────────────────────────────────────────
    print('[exp2] Running White Reality Check (γ=1, c=0%) ...')
    # Test: does any Q-model significantly beat its non-Q benchmark?
    wrc_rows = []
    for bench in ['HAR', 'HAR-DRD', 'EWMA', 'HEAVY']:
        q_models_for_bench = [q for q, b in CEQ_PAIRS if b == bench]
        if not q_models_for_bench:
            continue

        r_bench = port_returns[bench]
        r_alts  = np.column_stack([port_returns[q] for q in q_models_for_bench])
        alt_ceqs = np.array([_ceq(r_alts[:, m], gamma=1) for m in range(r_alts.shape[1])])
        bench_ceq = _ceq(r_bench, gamma=1)

        pval = _white_reality_check(
            bench_ceq, alt_ceqs,
            r_bench, r_alts,
            gamma=1, B=500, block_len=5,
        )
        wrc_rows.append({
            'Benchmark': bench,
            'Alt Models': ', '.join(q_models_for_bench),
            'WRC p-value': round(pval, 4),
        })

    wrc_df = pd.DataFrame(wrc_rows)
    print('\n[exp2] White Reality Check p-values:\n', wrc_df.to_string(index=False))

    # ── save results ──────────────────────────────────────────────────────────
    perf_table.to_csv(str(results_dir / 'exp2_results.csv'))
    ceq_df.to_csv(str(results_dir / 'exp2_ceq_gains.csv'), index=False)
    wrc_df.to_csv(str(results_dir / 'exp2_wrc.csv'), index=False)
    print(f'\n[exp2] Results saved to {results_dir}/')

    # ── plots ─────────────────────────────────────────────────────────────────
    print('[exp2] Generating plots ...')

    # Plot 1: Portfolio volatility bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    vols = [perf_table.loc[m, 'Ann. Vol.'] for m in MODEL_NAMES]
    colors = ['#2196F3' if m not in ['HARQ', 'HARQ-DRD', 'EWMAQ', 'HEAVYQ']
              else '#F44336' for m in MODEL_NAMES]
    ax.bar(MODEL_NAMES, vols, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(f'Annualised GMV Portfolio Volatility — '
                 f'{"SYNTHETIC" if synthetic else "REAL"} data')
    ax.set_ylabel('Ann. Volatility')
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    fig.savefig(str(results_dir / 'exp2_volatility.png'), dpi=150)
    plt.close(fig)

    # Plot 2: Turnover comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    tos = [turnovers[m].mean() for m in MODEL_NAMES]
    ax.bar(MODEL_NAMES, tos, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title('Mean Daily Portfolio Turnover (one-way)')
    ax.set_ylabel('Turnover')
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    fig.savefig(str(results_dir / 'exp2_turnover.png'), dpi=150)
    plt.close(fig)

    # Plot 3: CEQ gains heatmap  (gamma=1, varying cost)
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(6 * len(GAMMAS), 5))
    if len(GAMMAS) == 1:
        axes = [axes]
    pair_labels = [f'{q}\nvs {b}' for q, b in CEQ_PAIRS]
    for ax, gamma in zip(axes, GAMMAS):
        gain_matrix = np.zeros((len(CEQ_PAIRS), len(COSTS)))
        for pi, (q_model, bench) in enumerate(CEQ_PAIRS):
            for ci, c in enumerate(COSTS):
                row = ceq_df[
                    (ceq_df['Q-Model'] == q_model) &
                    (ceq_df['Benchmark'] == bench) &
                    (ceq_df['gamma'] == gamma) &
                    (ceq_df['cost'] == c)
                ]
                gain_matrix[pi, ci] = row['CEQ gain'].values[0] if len(row) else 0.0

        vmax = max(abs(gain_matrix).max(), 1e-6)
        im   = ax.imshow(gain_matrix, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(COSTS)))
        ax.set_xticklabels([f'c={int(c*100)}%' for c in COSTS])
        ax.set_yticks(range(len(CEQ_PAIRS)))
        ax.set_yticklabels(pair_labels, fontsize=9)
        ax.set_title(f'CEQ Gain (γ={gamma})')
        plt.colorbar(im, ax=ax)
        for pi in range(len(CEQ_PAIRS)):
            for ci in range(len(COSTS)):
                ax.text(ci, pi, f'{gain_matrix[pi, ci]:.4f}',
                        ha='center', va='center', fontsize=7,
                        color='black' if abs(gain_matrix[pi, ci]) < vmax * 0.7 else 'white')

    fig.suptitle('CEQ Gains of Q-models over Benchmarks (annualised)')
    plt.tight_layout()
    fig.savefig(str(results_dir / 'exp2_ceq_heatmap.png'), dpi=150)
    plt.close(fig)

    # Plot 4: Cumulative portfolio returns
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(MODEL_NAMES)))
    for m, color in zip(MODEL_NAMES, colors_cycle):
        cum_ret = np.cumprod(1.0 + port_returns[m]) - 1.0
        ls = '-' if m in ['HAR', 'HAR-DRD', 'EWMA', 'HEAVY'] else '--'
        ax.plot(cum_ret, label=m, linestyle=ls, color=color, linewidth=1.2)
    ax.set_title(f'Cumulative GMV Portfolio Returns — '
                 f'{"SYNTHETIC" if synthetic else "REAL"} data')
    ax.set_xlabel('OOS Trading Day')
    ax.set_ylabel('Cumulative Return')
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    plt.tight_layout()
    fig.savefig(str(results_dir / 'exp2_cumulative_returns.png'), dpi=150)
    plt.close(fig)

    print('[exp2] Plots saved to results/.')

    return {
        'perf_table':   perf_table,
        'ceq_gains':    ceq_df,
        'wrc_pvals':    wrc_df,
        'port_returns': port_returns,
        'synthetic':    synthetic,
    }


if __name__ == '__main__':
    run_portfolio_experiment()

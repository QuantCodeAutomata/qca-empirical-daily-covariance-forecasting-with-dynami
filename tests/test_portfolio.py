import pytest
import numpy as np
from src.portfolio import (
    gmv_weights,
    compute_turnover,
    compute_concentration,
    compute_utility,
    certainty_equivalent_gain,
    insanity_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cov(N, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    return A @ A.T + np.eye(N) * 0.5


# ---------------------------------------------------------------------------
# gmv_weights
# ---------------------------------------------------------------------------

def test_gmv_weights_sum_to_one():
    N = 5
    H = _make_cov(N, seed=0)
    w = gmv_weights(H)
    assert np.isclose(w.sum(), 1.0), f"Weights must sum to 1, got {w.sum():.6f}"


def test_gmv_weights_shape():
    N = 7
    H = _make_cov(N, seed=1)
    w = gmv_weights(H)
    assert w.shape == (N,)


def test_gmv_weights_minimize_variance():
    """No perturbation of the GMV portfolio (keeping budget constraint) reduces variance."""
    rng = np.random.default_rng(2)
    N = 4
    H = _make_cov(N, seed=2)
    w_gmv = gmv_weights(H)
    var_gmv = w_gmv @ H @ w_gmv

    for _ in range(20):
        delta = rng.standard_normal(N)
        delta -= delta.mean()          # zero-sum perturbation: keeps budget constraint
        w_alt = w_gmv + 0.05 * delta
        var_alt = w_alt @ H @ w_alt
        assert var_gmv <= var_alt + 1e-12, \
            "GMV portfolio must have minimum variance"


def test_gmv_weights_identity_cov_equal():
    """For H = I_N, the GMV portfolio is equal-weight."""
    N = 5
    w = gmv_weights(np.eye(N))
    assert np.allclose(w, np.ones(N) / N, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------

def test_turnover_no_change():
    w = np.array([0.3, 0.3, 0.4])
    assert np.isclose(compute_turnover(w, w), 0.0)


def test_turnover_full_rotation():
    # Moving entirely from asset 0 to asset 1 in a 2-asset portfolio
    w_prev = np.array([1.0, 0.0])
    w_curr = np.array([0.0, 1.0])
    # Turnover = sum(|w_curr - w_prev|) / 2 = (1+1)/2 = 1 (100% rebalancing)
    # or sum(|w_curr - w_prev|) = 2 depending on convention
    # Either way it must be positive and larger than partial rebalancing
    assert compute_turnover(w_curr, w_prev) > 0.0


def test_turnover_partial_rebalance():
    w_prev = np.array([0.5, 0.5])
    w_curr = np.array([0.6, 0.4])
    to_full = compute_turnover(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    to_partial = compute_turnover(w_curr, w_prev)
    assert to_partial < to_full


# ---------------------------------------------------------------------------
# compute_concentration
# ---------------------------------------------------------------------------

def test_concentration_all_equal():
    """Equal-weight portfolio: concentration = 1/sqrt(N) = sqrt(Herfindahl)."""
    N = 9
    w = np.ones(N) / N
    expected = 1.0 / np.sqrt(N)
    assert np.isclose(compute_concentration(w), expected, rtol=1e-6)


def test_concentration_degenerate():
    """All-in portfolio: concentration = 1."""
    N = 5
    w = np.zeros(N)
    w[0] = 1.0
    assert np.isclose(compute_concentration(w), 1.0)


def test_concentration_ordering():
    """More concentrated portfolio has higher concentration metric."""
    N = 4
    w_equal = np.ones(N) / N
    w_concentrated = np.array([0.7, 0.1, 0.1, 0.1])
    assert compute_concentration(w_concentrated) > compute_concentration(w_equal)


# ---------------------------------------------------------------------------
# compute_utility
# ---------------------------------------------------------------------------

def test_utility_gamma_effect():
    """Higher risk-aversion (gamma) penalises variance more."""
    rng = np.random.default_rng(10)
    T = 500
    returns = rng.standard_normal(T) * 0.02    # high-ish daily vol

    u_low = compute_utility(returns, gamma=1.0)
    u_high = compute_utility(returns, gamma=10.0)
    assert u_high < u_low, \
        "Higher gamma must reduce utility for a positive-variance portfolio"


def test_utility_zero_variance():
    """Constant return series: utility should equal the mean return (no variance penalty)."""
    returns = np.ones(100) * 0.001
    u1 = compute_utility(returns, gamma=1.0)
    u5 = compute_utility(returns, gamma=5.0)
    assert np.isclose(u1, u5, rtol=1e-6), \
        "With zero variance the gamma term vanishes"


# ---------------------------------------------------------------------------
# certainty_equivalent_gain
# ---------------------------------------------------------------------------

def test_ceq_gain_zero_same_returns():
    rng = np.random.default_rng(11)
    returns = rng.standard_normal(300) * 0.01
    gain = certainty_equivalent_gain(returns, returns, gamma=3.0)
    assert np.isclose(gain, 0.0, atol=1e-10)


def test_ceq_gain_positive_when_better():
    rng = np.random.default_rng(12)
    # Series 1: same mean but lower variance → should have positive CEQ gain over series 2
    r_low_var = rng.standard_normal(500) * 0.01
    r_high_var = rng.standard_normal(500) * 0.05
    # Adjust means to be equal
    r_low_var -= r_low_var.mean()
    r_high_var -= r_high_var.mean()
    gain = certainty_equivalent_gain(r_low_var, r_high_var, gamma=5.0)
    assert gain > 0.0, "Lower variance portfolio must have positive CEQ gain"


# ---------------------------------------------------------------------------
# insanity_filter
# ---------------------------------------------------------------------------

def test_insanity_filter_not_triggered():
    """PD input passes through unchanged."""
    N = 3
    H = _make_cov(N, seed=20)
    S_mean = _make_cov(N, seed=21)
    result = insanity_filter(H, S_mean)
    assert np.allclose(result, H)


def test_insanity_filter_triggered():
    """Non-PD forecast is replaced by S_mean."""
    N = 3
    H_bad = np.array([[1.0, 2.0, 0.0],
                       [2.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])  # indefinite (off-diag entry > diagonal)
    S_mean = _make_cov(N, seed=22)
    result = insanity_filter(H_bad, S_mean)
    assert np.allclose(result, S_mean), \
        "Non-PD input must be replaced by S_mean"


def test_insanity_filter_singular_triggered():
    """Singular matrix (zero determinant) is also replaced."""
    N = 3
    H_sing = np.zeros((N, N))
    S_mean = _make_cov(N, seed=23)
    result = insanity_filter(H_sing, S_mean)
    assert np.allclose(result, S_mean)

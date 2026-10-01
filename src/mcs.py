"""
Model Confidence Set (MCS) procedure for covariance forecast comparison.

Implements the MCS algorithm of Hansen, Lunde & Nason (2011) using the range
test statistic and stationary bootstrap p-values via arch.bootstrap.

References
----------
Hansen, P.R., Lunde, A. & Nason, J.M. (2011). The model confidence set.
    *Econometrica*, 79(2), 453–497.

Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
    *Journal of the American Statistical Association*, 89(428), 1303–1313.
"""

from __future__ import annotations

import numpy as np
from arch.bootstrap import StationaryBootstrap


# ---------------------------------------------------------------------------
# Stationary bootstrap p-value
# ---------------------------------------------------------------------------


def bootstrap_dm_pvalue(
    diff: np.ndarray,
    B: int = 1000,
    block_size: int = 22,
    seed: int = 42,
) -> float:
    """Stationary-bootstrap two-sided p-value for H₀: E[diff] = 0.

    Implements the recentering approach of Politis & Romano (1994):

    1. Centre the series under H₀:  diff_c = diff − mean(diff).
    2. Draw B stationary-bootstrap resamples of diff_c.
    3. The p-value is the proportion of bootstrap means whose absolute value
       exceeds |mean(diff)|.

    Parameters
    ----------
    diff : np.ndarray of shape (T,)
        Loss-differential series d_t = L_{1,t} − L_{2,t}.
    B : int, optional
        Number of bootstrap replications.  Default is 1000.
    block_size : int, optional
        Expected block length for the stationary bootstrap.  The actual block
        length is geometrically distributed with mean ``block_size``.
        A value of 22 corresponds to approximately one trading month.
    seed : int, optional
        Random seed passed to :class:`arch.bootstrap.StationaryBootstrap` for
        reproducibility.

    Returns
    -------
    float
        Two-sided bootstrap p-value in [0, 1].

    Notes
    -----
    ``arch.bootstrap.StationaryBootstrap`` is used so that serial dependence
    in the loss-differential series (expected under any reasonable DGP) is
    properly accounted for.
    """
    dbar = float(np.mean(diff))
    diff_c = diff - dbar  # centre under H0: forces E[diff^*] ≈ 0

    bs = StationaryBootstrap(block_size, diff_c, seed=seed)
    boot_means = np.array(
        [d_b.mean() for (d_b,), _ in bs.bootstrap(B)],
        dtype=float,
    )
    return float(np.mean(np.abs(boot_means) >= abs(dbar)))


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------


def model_confidence_set(
    losses: np.ndarray,
    alpha: float = 0.10,
    B: int = 1000,
    block_size: int = 22,
    seed: int = 42,
) -> np.ndarray:
    """Compute the (1 − α) Model Confidence Set (MCS).

    Applies the sequential elimination algorithm of Hansen, Lunde & Nason
    (2011) using the *range* test statistic T_R and stationary bootstrap
    p-values.

    **Algorithm**

    Let S be the current set of models with |S| = m.

    1. For each model i ∈ S compute the relative loss differential:

       .. math::

           d_{i,t} = L_{i,t} - \\frac{1}{m} \\sum_{j \\in S} L_{j,t}

    2. Estimate the bootstrap variance of ``mean(d_{i,.})`` by drawing B
       joint stationary-bootstrap resamples of the T × m matrix D = [d_{i,t}],
       centred under H₀.

    3. Compute t-statistics:

       .. math::

           t_i = \\bar{d}_{i,.} \\,/\\, \\sqrt{\\widehat{\\operatorname{Var}}(\\bar{d}_{i,.})}

    4. Range test statistic:

       .. math::

           T_R = \\max_{i \\in S} t_i

    5. Bootstrap null distribution of T_R → p-value.

    6. If p-value < α: eliminate ``argmax_i t_i`` from S and go to step 1.
       Otherwise: stop; all remaining models form the MCS.

    Parameters
    ----------
    losses : np.ndarray of shape (T, M)
        Loss matrix.  Rows are time periods, columns are models.
    alpha : float, optional
        Significance level.  The procedure returns the (1 − α) MCS.
        Default is 0.10 (90 % MCS as in the paper).
    B : int, optional
        Number of stationary-bootstrap replications.  Default is 1000.
    block_size : int, optional
        Expected block length for the stationary bootstrap (default 22).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    np.ndarray of shape (M,) with dtype bool
        ``True`` for models that belong to the MCS, ``False`` for those
        eliminated.

    Notes
    -----
    * The joint bootstrap is run once per elimination step on the current
      active set D.  The same bootstrap draws are used for variance estimation
      and for the null distribution of T_R (fixed-denominator bootstrap).
    * Variance estimates are floored at 1e-16 to guard against degenerate
      (e.g. identical) loss series.
    * When only one model remains it is trivially included in the MCS.
    """
    T, M = losses.shape
    in_mcs = np.ones(M, dtype=bool)

    while True:
        active = np.where(in_mcs)[0]
        m = len(active)
        if m <= 1:
            # Trivially, one model cannot be rejected
            break

        L = losses[:, active]  # T × m

        # Step 1 – loss differentials relative to cross-section mean
        d = L - L.mean(axis=1, keepdims=True)  # T × m
        dbar = d.mean(axis=0)                  # m-vector

        # Step 2 – joint stationary bootstrap for variance + null distribution
        #
        # We draw from the ORIGINAL d (not centred) and then subtract dbar
        # inside the loop so that the bootstrap distribution of mean(d^*) is
        # centred at zero (i.e. we impose H₀ by recentring).
        bs = StationaryBootstrap(block_size, d, seed=seed)

        boot_dbar = np.zeros((B, m), dtype=float)
        for b, ((d_b,), _) in enumerate(bs.bootstrap(B)):
            # d_b is shape (T, m) – a stationary-bootstrap resample of d
            # Subtract dbar to impose H₀: E[mean(d^*) − dbar] ≈ 0
            boot_dbar[b] = d_b.mean(axis=0) - dbar

        # Step 3 – bootstrap variance and t-statistics
        var_hat = boot_dbar.var(axis=0, ddof=1)          # m-vector
        var_hat = np.maximum(var_hat, 1e-16)              # numerical floor
        t_stats = dbar / np.sqrt(var_hat)                 # m-vector

        # Step 4 & 5 – observed T_R and bootstrap p-value
        T_R_obs = float(np.max(t_stats))
        boot_T_R = (boot_dbar / np.sqrt(var_hat)).max(axis=1)  # B-vector
        p_value = float(np.mean(boot_T_R >= T_R_obs))

        # Step 6 – decision
        if p_value >= alpha:
            break  # H₀ of equal predictive ability not rejected

        # Eliminate the model with the worst (largest) t-statistic
        worst_local = int(np.argmax(t_stats))
        in_mcs[active[worst_local]] = False

    return in_mcs

"""
Preprocessing utilities: daily realized covariance, quarticity, and vech panels.

Notation (following Barndorff-Nielsen, Hansen, Lunde & Shephard 2011 and
Bollerslev, Patton & Quaedvlieg 2016):

  r_{t,j}   N-vector of 5-min log-returns for asset j on day t
  M_t       number of intraday bars on day t  (≈ 78 for 5-min, 6.5h session)
  S_t       N×N realized covariance:   Σ_j r_{t,j} r_{t,j}'
  Pi_t      N×N realized quarticity:   (M_t/3) Σ_j (r_{t,j} ⊗ r_{t,j})²  [element-wise]
              Pi_{t,ij} = (M_t/3) Σ_j r_{t,j,i}² · r_{t,j,j}²
  pi_t      vech(Pi_t)  —  K-vector, K = N(N+1)/2

Panels are stored as DataFrames with shape (T, K) where columns are a
pd.MultiIndex of (asset_i, asset_j) tuples for the upper-triangle pairs
(i ≤ j), matching the vech ordering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ── helpers ───────────────────────────────────────────────────────────────────

def _upper_triangle_index(assets: List[str]) -> pd.MultiIndex:
    """Return a MultiIndex of (asset_i, asset_j) for i ≤ j (vech ordering)."""
    pairs = [(ai, aj) for i, ai in enumerate(assets) for aj in assets[i:]]
    return pd.MultiIndex.from_tuples(pairs, names=['asset_i', 'asset_j'])


def _vech(M: np.ndarray, assets: List[str]) -> pd.Series:
    """Vectorise upper triangle of square matrix M into a pd.Series."""
    n = len(assets)
    idx = np.triu_indices(n)
    col_idx = _upper_triangle_index(assets)
    return pd.Series(M[idx], index=col_idx)


# ── daily realized measures ───────────────────────────────────────────────────

def compute_daily_realized_measures(
    intraday_data: Dict[str, pd.DataFrame],
    assets: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute daily realized covariance (S_t), quarticity matrix (Pi_t), and
    quarticity vector (pi_t = vech(Pi_t)) from 5-minute log-return data.

    Parameters
    ----------
    intraday_data : dict
        Mapping ticker → DataFrame returned by ``loader.load_intraday_bars``.
        Each DataFrame has a UTC DatetimeIndex and at minimum a 'close' column.
    assets : list of str
        Ordered list of tickers; must be a subset of *intraday_data* keys.

    Returns
    -------
    S_panel : pd.DataFrame, shape (T, K)
        Realized covariance vech panels.  Index: trading dates.
        Columns: MultiIndex (asset_i, asset_j) for i ≤ j.
    Pi_panel : pd.DataFrame, shape (T, K)
        Realized quarticity matrix vech panels (same column structure).
    pi_panel : pd.DataFrame, shape (T, K)
        Alias for Pi_panel (pi_t = vech(Pi_t)).  Returned separately for
        convenience when downstream code uses the pi_t terminology.
    """
    n = len(assets)
    col_idx = _upper_triangle_index(assets)

    # ── build per-asset intraday log-return series (ET-aligned) ──────────────
    ret_frames: Dict[str, pd.DataFrame] = {}
    for ticker in assets:
        df = intraday_data[ticker].copy()
        df.index = df.index.tz_convert('America/New_York')
        log_ret = np.log(df['close']).diff()

        date_arr   = df.index.normalize().date           # array of datetime.date
        day_starts = pd.Series(
            np.concatenate([[True], date_arr[1:] != date_arr[:-1]]),
            index=df.index, dtype=bool,
        )
        log_ret[day_starts] = np.nan

        dates_idx        = df.index.normalize()
        df['log_return'] = log_ret
        df['date']       = dates_idx.date
        df['interval']   = df.groupby(dates_idx).cumcount()
        ret_frames[ticker] = df[['date', 'interval', 'log_return']].dropna()

    # ── find common trading days ──────────────────────────────────────────────
    day_sets = [set(ret_frames[t]['date'].unique()) for t in assets]
    common_days = sorted(set.intersection(*day_sets))

    S_rows:  List[pd.Series] = []
    Pi_rows: List[pd.Series] = []
    dates_out: List = []

    for day in common_days:
        # Stack intraday return matrix: (M_t, N)
        cols = []
        M_t  = None
        valid = True
        for ticker in assets:
            day_df = ret_frames[ticker][ret_frames[ticker]['date'] == day]
            day_df = day_df.sort_values('interval')
            if day_df.empty:
                valid = False
                break
            cols.append(day_df['log_return'].values)
            M_t = len(cols[-1])

        if not valid:
            continue

        # Align to shortest series (handles minor length mismatches)
        min_len = min(len(c) for c in cols)
        if min_len < 2:
            continue
        R = np.column_stack([c[:min_len] for c in cols])  # (M_t, N)
        M_t = min_len

        # ── S_t: realized covariance ──────────────────────────────────────────
        # S_t = R' R  (sum of outer products Σ_j r_j r_j')
        S_t = R.T @ R                                        # (N, N)

        # ── Pi_t: realized quarticity matrix ─────────────────────────────────
        # Pi_{t,ij} = (M_t/3) * Σ_k r_{k,i}² * r_{k,j}²
        R_sq = R ** 2                                        # (M_t, N)
        Pi_t = (M_t / 3.0) * (R_sq.T @ R_sq)               # (N, N)

        S_rows.append(_vech(S_t,  assets))
        Pi_rows.append(_vech(Pi_t, assets))
        dates_out.append(day)

    dates_idx = pd.DatetimeIndex(pd.to_datetime(dates_out))
    S_panel  = pd.DataFrame(S_rows,  index=dates_idx, columns=col_idx)
    Pi_panel = pd.DataFrame(Pi_rows, index=dates_idx, columns=col_idx)

    return S_panel, Pi_panel, Pi_panel.copy()


# ── rolling demeaned pi ───────────────────────────────────────────────────────

def compute_rolling_demeaned_pi(
    pi_df: pd.DataFrame,
    window: int = 1000,
) -> pd.DataFrame:
    """
    For each out-of-sample day t > *window*, standardise pi_t using the
    rolling in-sample statistics from [t−window, t−1].

    Standardisation: π̃_{t,k} = (π_{t,k} − μ_k) / σ_k
    where μ_k and σ_k are the rolling mean and standard deviation of the
    preceding *window* observations for element k.

    A positive π̃_{t,k} signals above-average measurement error and triggers
    downward attenuation of the HAR daily component in HARQ models.

    Parameters
    ----------
    pi_df  : pd.DataFrame, shape (T, K) — full-sample pi_t panel (IS + OOS).
    window : int — in-sample window length.

    Returns
    -------
    pd.DataFrame, shape (T − window, K)
        Standardised pi for OOS days only (index aligned with pi_df).
    """
    T = len(pi_df)
    if T <= window:
        raise ValueError(
            f"pi_df has only {T} rows but window={window}. "
            "Need at least window+1 rows."
        )

    values = pi_df.values.astype(float)          # (T, K)
    oos_rows = []

    for t in range(window, T):
        is_slice  = values[t - window : t]        # (window, K)
        mu        = is_slice.mean(axis=0)
        sigma     = is_slice.std(axis=0)
        sigma     = np.where(sigma < 1e-12, 1.0, sigma)
        demeaned  = (values[t] - mu) / sigma
        oos_rows.append(demeaned)

    oos_index = pi_df.index[window:]
    return pd.DataFrame(
        np.array(oos_rows),
        index=oos_index,
        columns=pi_df.columns,
    )


# ── vech series ───────────────────────────────────────────────────────────────

def prepare_vech_series(
    S_panel: pd.DataFrame,
    assets: List[str],
) -> pd.DataFrame:
    """
    Validate and return the vech panel in canonical order.

    Ensures columns are a pd.MultiIndex of (asset_i, asset_j) for i ≤ j,
    with K = N(N+1)/2 columns.  Rows with any NaN are dropped.

    Parameters
    ----------
    S_panel : pd.DataFrame, shape (T, K) — output of
              :func:`compute_daily_realized_measures` (S or Pi panel).
    assets  : ordered list of tickers.

    Returns
    -------
    pd.DataFrame, shape (T', K)
        NaN-free vech panel ready for model fitting.
    """
    expected_cols = _upper_triangle_index(assets)
    k_expected    = len(expected_cols)

    if S_panel.shape[1] != k_expected:
        raise ValueError(
            f"S_panel has {S_panel.shape[1]} columns; "
            f"expected {k_expected} for {len(assets)} assets."
        )

    # Ensure column MultiIndex is correctly typed
    if not isinstance(S_panel.columns, pd.MultiIndex):
        S_panel = S_panel.copy()
        S_panel.columns = expected_cols

    result = S_panel.dropna()
    return result

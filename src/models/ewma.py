"""
Exponentially Weighted Moving Average (EWMA) covariance models.

EWMAModel  : standard single-parameter EWMA on squared returns.
EWMAQModel : noise-adjusted variant that blends daily returns with the
             realized covariance when computing the EWMA update.

Both models estimate the smoothing parameter alpha in (0, 1) by
minimising the average QLIKE loss over the training sample.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from src.utils import vech, unvech, nearest_spd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _qlike(H: np.ndarray, S: np.ndarray) -> float:
    """QLIKE loss: tr(H^{-1} S) + log|H|. Returns nan if H is singular."""
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return np.nan
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    H_inv_S = np.linalg.solve(H, S)
    return np.trace(H_inv_S) + log_det


def _sample_cov(R: np.ndarray) -> np.ndarray:
    """Outer-product covariance (no mean subtraction) from return rows."""
    T = R.shape[0]
    return (R.T @ R) / T


# ---------------------------------------------------------------------------
# EWMAModel
# ---------------------------------------------------------------------------

class EWMAModel:
    """EWMA covariance model.

    Update rule:
        H_t = (1 - alpha) * H_{t-1} + alpha * r_{t-1} r_{t-1}^T

    Parameters
    ----------
    N : int  — number of assets.
    """

    def __init__(self, N: int) -> None:
        self.N = N
        self.alpha_: float | None = None
        self._H_last: np.ndarray | None = None

    # ── fitting ────────────────────────────────────────────────────────────

    def fit(self, returns: np.ndarray) -> "EWMAModel":
        """Fit alpha by minimising average QLIKE on the training sample.

        Parameters
        ----------
        returns : np.ndarray of shape (T, N)
        """
        T, N = returns.shape
        H0 = _sample_cov(returns)

        def neg_qlike(log_alpha):
            alpha = float(np.exp(log_alpha))
            H = H0.copy()
            losses = []
            for t in range(T):
                r = returns[t]
                H = (1.0 - alpha) * H + alpha * np.outer(r, r)
                H = nearest_spd(H)
                losses.append(_qlike(H, np.outer(r, r)))
            valid = [x for x in losses if np.isfinite(x)]
            return np.mean(valid) if valid else 1e10

        result = minimize_scalar(neg_qlike, bounds=(-5.0, -0.01), method="bounded")
        self.alpha_ = float(np.exp(result.x))

        # Re-run to obtain the final H estimate
        H = H0.copy()
        for t in range(T):
            r = returns[t]
            H = (1.0 - self.alpha_) * H + self.alpha_ * np.outer(r, r)
            H = nearest_spd(H)
        self._H_last = H
        return self

    # ── prediction ─────────────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Return the current EWMA covariance estimate (N × N)."""
        if self._H_last is None:
            raise RuntimeError("Call fit() before predict().")
        return self._H_last.copy()


# ---------------------------------------------------------------------------
# EWMAQModel
# ---------------------------------------------------------------------------

class EWMAQModel:
    """Noise-adjusted EWMA covariance model.

    Blends the outer-product of returns with the realized covariance matrix
    S_t so that the update targets a less noisy estimate:

        H_t = (1 - alpha) * H_{t-1} + alpha * S_{t-1}

    Parameters
    ----------
    N : int  — number of assets.
    """

    def __init__(self, N: int) -> None:
        self.N = N
        self.alpha_: float | None = None
        self._H_last: np.ndarray | None = None

    def fit(self, returns: np.ndarray, S_array: np.ndarray) -> "EWMAQModel":
        """Fit alpha using realized covariances as targets.

        Parameters
        ----------
        returns : np.ndarray of shape (T, N)  (used for QLIKE evaluation)
        S_array : np.ndarray of shape (T, K)  (vech'd realized covariances)
        """
        T, N = returns.shape
        K = S_array.shape[1]
        H0 = _sample_cov(returns)

        def neg_qlike(log_alpha):
            alpha = float(np.exp(log_alpha))
            H = H0.copy()
            losses = []
            for t in range(T):
                S_t = unvech(S_array[t], N)
                H = (1.0 - alpha) * H + alpha * S_t
                H = nearest_spd(H)
                r = returns[t]
                losses.append(_qlike(H, np.outer(r, r)))
            valid = [x for x in losses if np.isfinite(x)]
            return np.mean(valid) if valid else 1e10

        result = minimize_scalar(neg_qlike, bounds=(-5.0, -0.01), method="bounded")
        self.alpha_ = float(np.exp(result.x))

        H = H0.copy()
        for t in range(T):
            S_t = unvech(S_array[t], N)
            H = (1.0 - self.alpha_) * H + self.alpha_ * S_t
            H = nearest_spd(H)
        self._H_last = H
        return self

    def predict(self) -> np.ndarray:
        """Return the current EWMAQ covariance estimate (N × N)."""
        if self._H_last is None:
            raise RuntimeError("Call fit() before predict().")
        return self._H_last.copy()

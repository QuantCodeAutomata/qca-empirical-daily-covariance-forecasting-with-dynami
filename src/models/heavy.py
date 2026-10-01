"""
HEAVY (High-frEquency-bAsed VolatilitY) covariance models.

HEAVYModel  : scalar-parameter HEAVY model fitted by minimising QLIKE.
HEAVYQModel : noise-adjusted variant that attenuates the realized covariance
              update when the quarticity proxy is large.

Update rule (HEAVYModel):
    H_{t+1} = (1 - alpha - beta) * H_unc + alpha * S_t + beta * H_t

Stationarity requires alpha + beta < 1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.utils import vech, unvech, nearest_spd


# ---------------------------------------------------------------------------
# shared helper
# ---------------------------------------------------------------------------

def _qlike(H: np.ndarray, S: np.ndarray) -> float:
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return np.nan
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return np.trace(np.linalg.solve(H, S)) + log_det


def _softmax2(params: np.ndarray, cap: float = 0.99) -> tuple[float, float]:
    """Map unconstrained (p0, p1) → (alpha, beta) with alpha+beta < cap."""
    e0, e1 = np.exp(params[0]), np.exp(params[1])
    denom = 1.0 + e0 + e1
    return float(e0 / denom * cap), float(e1 / denom * cap)


# ---------------------------------------------------------------------------
# HEAVYModel
# ---------------------------------------------------------------------------

class HEAVYModel:
    """Scalar HEAVY covariance model.

    Parameters
    ----------
    N : int  — number of assets.
    """

    def __init__(self, N: int) -> None:
        self.N = N
        self.alpha_: float | None = None
        self.beta_: float | None = None
        self._H_last: np.ndarray | None = None
        self._H_unc: np.ndarray | None = None

    # ── fitting ────────────────────────────────────────────────────────────

    def fit(self, rcov_list: list[np.ndarray]) -> "HEAVYModel":
        """Fit (alpha, beta) by minimising average QLIKE.

        Parameters
        ----------
        rcov_list : list of (N, N) realized covariance matrices.
        """
        S_arr = np.array(rcov_list)    # (T, N, N)
        T = S_arr.shape[0]
        H_unc = S_arr.mean(axis=0)
        self._H_unc = H_unc

        def _run(alpha: float, beta: float) -> float:
            H = H_unc.copy()
            losses = []
            omega = (1.0 - alpha - beta) * H_unc
            for t in range(1, T):
                H = omega + alpha * S_arr[t - 1] + beta * H
                H = nearest_spd(H)
                v = _qlike(H, S_arr[t])
                if np.isfinite(v):
                    losses.append(v)
            return float(np.mean(losses)) if losses else 1e10

        def objective(raw: np.ndarray) -> float:
            a, b = _softmax2(raw)
            return _run(a, b)

        res = minimize(
            objective,
            x0=np.array([np.log(0.1), np.log(0.85)]),
            method="Nelder-Mead",
            options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-4},
        )
        self.alpha_, self.beta_ = _softmax2(res.x)

        # Re-compute final H with fitted params
        H = H_unc.copy()
        omega = (1.0 - self.alpha_ - self.beta_) * H_unc
        for t in range(1, T):
            H = omega + self.alpha_ * S_arr[t - 1] + self.beta_ * H
            H = nearest_spd(H)
        self._H_last = H
        return self

    # ── prediction ─────────────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Return the N × N HEAVY forecast (one step ahead)."""
        if self._H_last is None:
            raise RuntimeError("Call fit() before predict().")
        return self._H_last.copy()


# ---------------------------------------------------------------------------
# HEAVYQModel
# ---------------------------------------------------------------------------

class HEAVYQModel:
    """Noise-adjusted HEAVY model.

    Modulates the alpha term by a noise factor derived from the quarticity
    proxy: effective_alpha(t) = alpha / (1 + theta * pi_t), capped at 1.

    Parameters
    ----------
    N : int  — number of assets.
    """

    def __init__(self, N: int) -> None:
        self.N = N
        self.alpha_: float | None = None
        self.beta_: float | None = None
        self._H_last: np.ndarray | None = None
        self._H_unc: np.ndarray | None = None

    def fit(
        self,
        rcov_list: list[np.ndarray],
        Pi_array: np.ndarray,
    ) -> "HEAVYQModel":
        """Fit HEAVYQ parameters.

        Parameters
        ----------
        rcov_list : list of (N, N) realized covariance matrices.
        Pi_array  : np.ndarray of shape (T, K) — quarticity proxy vech.
        """
        S_arr = np.array(rcov_list)    # (T, N, N)
        T = S_arr.shape[0]
        N = self.N
        K = N * (N + 1) // 2
        H_unc = S_arr.mean(axis=0)
        self._H_unc = H_unc

        # Scalar pi: use mean over all vech elements and take sqrt (std scale)
        pi_scalar = np.sqrt(Pi_array.mean(axis=1))    # (T,)

        def _run(alpha: float, beta: float, theta: float) -> float:
            H = H_unc.copy()
            losses = []
            omega = (1.0 - alpha - beta) * H_unc
            for t in range(1, T):
                pi_t = float(pi_scalar[t - 1])
                eff_alpha = alpha / max(1.0 + theta * pi_t, 1e-4)
                eff_alpha = min(eff_alpha, 1.0 - beta - 1e-6)
                H = (1.0 - eff_alpha - beta) * H_unc + eff_alpha * S_arr[t - 1] + beta * H
                H = nearest_spd(H)
                v = _qlike(H, S_arr[t])
                if np.isfinite(v):
                    losses.append(v)
            return float(np.mean(losses)) if losses else 1e10

        def objective(raw: np.ndarray) -> float:
            a, b = _softmax2(raw[:2])
            theta = max(float(raw[2]), 0.0)
            return _run(a, b, theta)

        res = minimize(
            objective,
            x0=np.array([np.log(0.1), np.log(0.85), 0.0]),
            method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4},
        )
        self.alpha_, self.beta_ = _softmax2(res.x[:2])
        theta = max(float(res.x[2]), 0.0)

        H = H_unc.copy()
        for t in range(1, T):
            pi_t = float(pi_scalar[t - 1])
            eff_alpha = self.alpha_ / max(1.0 + theta * pi_t, 1e-4)
            eff_alpha = min(eff_alpha, 1.0 - self.beta_ - 1e-6)
            H = (1.0 - eff_alpha - self.beta_) * H_unc + eff_alpha * S_arr[t - 1] + self.beta_ * H
            H = nearest_spd(H)
        self._H_last = H
        return self

    def predict(self) -> np.ndarray:
        if self._H_last is None:
            raise RuntimeError("Call fit() before predict().")
        return self._H_last.copy()

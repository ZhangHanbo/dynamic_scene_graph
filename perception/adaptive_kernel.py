"""
Adaptive robust loss kernel (Chebrolu et al. 2021 — Paper 2).

Implements the generalized Barron loss with a truncated partition function
and alternating minimization for the shape parameter `α`.

The generalized loss is:
    ρ(r, α, c) = (|α - 2| / α) · (((r/c)² / |α - 2| + 1)^(α/2) - 1)

Special cases:
    α = 2   → L2 loss
    α = 1   → pseudo-Huber / L1-L2
    α = 0   → Cauchy
    α = -2  → Geman-McClure
    α = -∞  → Welsch (very aggressive outlier rejection)

Smaller α → stronger outlier downweighting.

Usage:
    kernel = AdaptiveKernel(c=1.0)
    # Given residuals from the current iteration:
    alpha = kernel.fit_alpha(residuals)
    # Get IRLS weights:
    weights = kernel.weights(residuals, alpha)

The truncated partition function allows α < 0, which the original Barron
formulation cannot. This is essential for robotics applications where
strong outlier rejection is often needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Default parameter ranges
DEFAULT_ALPHA_MIN = -10.0
DEFAULT_ALPHA_MAX = 2.0
DEFAULT_TAU_FACTOR = 10.0  # truncation at |r| < TAU_FACTOR * c


def _rho_barron(r: np.ndarray, alpha: float, c: float) -> np.ndarray:
    """Generalized Barron loss ρ(r, α, c). Handles α=0 and α=2 as limits."""
    r = np.asarray(r, dtype=np.float64)
    rc2 = (r / c) ** 2

    # Special cases via pointwise limits (Barron paper Eq. 8)
    if abs(alpha - 2.0) < 1e-6:
        # α = 2: squared loss
        return 0.5 * rc2
    if abs(alpha) < 1e-6:
        # α = 0: Cauchy
        return np.log(0.5 * rc2 + 1.0)
    if alpha < -1e6:
        # α → -∞: Welsch
        return 1.0 - np.exp(-0.5 * rc2)

    abs_am2 = abs(alpha - 2.0)
    base = rc2 / abs_am2 + 1.0
    return (abs_am2 / alpha) * (np.power(base, 0.5 * alpha) - 1.0)


def _rho_derivative(r: np.ndarray, alpha: float, c: float) -> np.ndarray:
    """dρ/dr for IRLS weighting: w(r) = (1/r) · ρ'(r)."""
    r = np.asarray(r, dtype=np.float64)

    if abs(alpha - 2.0) < 1e-6:
        return r / (c * c)
    if abs(alpha) < 1e-6:
        return (2.0 * r) / (r * r + 2.0 * c * c)
    if alpha < -1e6:
        return (r / (c * c)) * np.exp(-0.5 * (r / c) ** 2)

    abs_am2 = abs(alpha - 2.0)
    base = (r / c) ** 2 / abs_am2 + 1.0
    return (r / (c * c)) * np.power(base, 0.5 * alpha - 1.0)


def irls_weight(r: np.ndarray, alpha: float, c: float,
                epsilon: float = 1e-10) -> np.ndarray:
    """Iteratively-Reweighted Least Squares weight for each residual.

    w(r) = (1/r) · dρ/dr

    These are the weights to plug into a weighted least-squares step.
    Outliers get near-zero weight under aggressive α.

    Args:
        r:       residuals (any shape).
        alpha:   kernel shape parameter.
        c:       scale parameter (inlier noise scale).
        epsilon: added to |r| to avoid division by zero.

    Returns:
        Weight array with the same shape as r, in [0, 1/c²].
    """
    r = np.asarray(r, dtype=np.float64)
    if abs(alpha - 2.0) < 1e-6:
        # L2: constant weight 1/c²
        return np.ones_like(r) / (c * c)

    # General case: w(r) = ρ'(r) / r
    drho = _rho_derivative(r, alpha, c)
    safe_r = np.where(np.abs(r) < epsilon, epsilon, r)
    return drho / safe_r


# ─────────────────────────────────────────────────────────────────────
# Truncated partition function (extends Barron to α < 0)
# ─────────────────────────────────────────────────────────────────────

def _log_partition(alpha: float, c: float,
                   tau_factor: float = DEFAULT_TAU_FACTOR,
                   n_quad: int = 201) -> float:
    """Numerical approximation of log ∫_{-τ}^{τ} exp(-ρ(r, α, 1)) dr.

    The truncation enables α < 0 (where the full integral diverges).
    Uses the trapezoidal rule on a dense grid; fast enough for a 1-D
    integral and avoids lookup tables.

    The result is c-independent after the change of variable; the factor
    of c is added by the caller.
    """
    tau = tau_factor
    r_grid = np.linspace(-tau, tau, n_quad)
    rho_vals = _rho_barron(r_grid, alpha, 1.0)
    # log-sum-exp for numerical stability
    neg_rho = -rho_vals
    m = np.max(neg_rho)
    integrand = np.exp(neg_rho - m)
    integral = np.trapz(integrand, r_grid)
    return float(m + np.log(integral + 1e-300))


# ─────────────────────────────────────────────────────────────────────
# Adaptive kernel
# ─────────────────────────────────────────────────────────────────────

class AdaptiveKernel:
    """Adaptive robust kernel with truncated partition function.

    Workflow (alternating minimization):
        1. Given current poses, compute residuals.
        2. kernel.fit_alpha(residuals) → best α for this distribution.
        3. kernel.weights(residuals, α) → IRLS weights for next step.
        4. Update poses via weighted least squares.
        5. Repeat until convergence.

    Args:
        c:                  scale parameter (inlier noise scale).
        alpha_min:          lower bound on α during fitting.
        alpha_max:          upper bound on α during fitting.
        tau_factor:         truncation radius in units of c.
        precompute_grid:    if True, build a (α, logZ(α)) lookup at init.
    """

    def __init__(self,
                 c: float = 1.0,
                 alpha_min: float = DEFAULT_ALPHA_MIN,
                 alpha_max: float = DEFAULT_ALPHA_MAX,
                 tau_factor: float = DEFAULT_TAU_FACTOR,
                 precompute_grid: bool = True,
                 grid_resolution: float = 0.1):
        self.c = float(c)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.tau_factor = float(tau_factor)

        self._alpha_grid: Optional[np.ndarray] = None
        self._logZ_grid: Optional[np.ndarray] = None

        if precompute_grid:
            self._alpha_grid = np.arange(alpha_min,
                                         alpha_max + grid_resolution,
                                         grid_resolution)
            self._logZ_grid = np.array([
                _log_partition(a, 1.0, tau_factor)
                for a in self._alpha_grid
            ])

    def neg_log_likelihood(self, residuals: np.ndarray, alpha: float) -> float:
        """Average negative log-likelihood of residuals under ρ_a.

        L(α) = mean_i [ρ(r_i, α, c) + log(c · Z̃(α))]

        The log-partition penalizes very negative α (which would trivially
        downweight all residuals) — without it, fitting α would collapse
        every point into an outlier.
        """
        residuals = np.asarray(residuals, dtype=np.float64)
        if residuals.size == 0:
            return 0.0

        rho_vals = _rho_barron(residuals, alpha, self.c)
        logZ = self._interp_logZ(alpha)
        return float(np.mean(rho_vals) + np.log(self.c) + logZ)

    def fit_alpha(self, residuals: np.ndarray,
                  coarse_step: float = 0.2) -> float:
        """Find α that minimizes the average negative log-likelihood.

        1-D grid search over [alpha_min, alpha_max]. Fast because the
        log-partition is precomputed and residuals are a flat array.
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        if residuals.size == 0:
            return self.alpha_max  # default to L2 if no evidence

        # Use precomputed grid if available, else evaluate on-the-fly
        if self._alpha_grid is not None:
            alphas = self._alpha_grid
        else:
            alphas = np.arange(self.alpha_min,
                               self.alpha_max + coarse_step,
                               coarse_step)

        best_nll = np.inf
        best_alpha = self.alpha_max
        for a in alphas:
            nll = self.neg_log_likelihood(residuals, a)
            if nll < best_nll:
                best_nll = nll
                best_alpha = a
        return float(best_alpha)

    def weights(self, residuals: np.ndarray, alpha: float) -> np.ndarray:
        """IRLS weights for each residual under the current α."""
        return irls_weight(residuals, alpha, self.c)

    def loss(self, residuals: np.ndarray, alpha: float) -> np.ndarray:
        """Per-residual ρ values (unnormalized)."""
        return _rho_barron(residuals, alpha, self.c)

    def _interp_logZ(self, alpha: float) -> float:
        """Linear interpolation on the precomputed log-partition grid."""
        if self._alpha_grid is None or self._logZ_grid is None:
            return _log_partition(alpha, 1.0, self.tau_factor)
        # Clamp to grid range
        a = np.clip(alpha, self._alpha_grid[0], self._alpha_grid[-1])
        return float(np.interp(a, self._alpha_grid, self._logZ_grid))


# ─────────────────────────────────────────────────────────────────────
# Convenience: wrap a factor graph's residual noise with the adaptive kernel
# ─────────────────────────────────────────────────────────────────────

def adapt_noise(residuals: np.ndarray, base_noise: np.ndarray,
                kernel: AdaptiveKernel, alpha: Optional[float] = None
                ) -> np.ndarray:
    """Inflate per-factor noise by the inverse of the adaptive IRLS weight.

    Given a base noise covariance Σ_base and residual r, the effective
    noise used by the optimizer is approximately Σ_eff = Σ_base / w(r),
    so that outliers with w ≈ 0 become Σ_eff ≈ ∞ (factor effectively
    removed from the optimization).

    Args:
        residuals: per-factor scalar residual norms, shape (N,).
        base_noise: (N, d, d) per-factor base noise covariances, or a
                    single (d, d) shared across all factors.
        kernel: AdaptiveKernel instance.
        alpha: pre-computed α; if None, fitted from residuals.

    Returns:
        (N, d, d) per-factor effective noise covariances.
    """
    residuals = np.asarray(residuals).ravel()
    if alpha is None:
        alpha = kernel.fit_alpha(residuals)
    w = kernel.weights(residuals, alpha)
    w = np.clip(w, 1e-12, None)  # avoid division blow-up

    base_noise = np.asarray(base_noise, dtype=np.float64)
    if base_noise.ndim == 2:
        # Broadcast a shared (d, d) over N factors
        return (1.0 / w)[:, None, None] * base_noise[None, ...]
    return (1.0 / w)[:, None, None] * base_noise

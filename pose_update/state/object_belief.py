"""Frame-agnostic per-object EKF primitives on SE(3).

Shared between the Gaussian and RBPF backends. Operates on a plain
`(μ, Σ)` pair without knowing whether μ is in base or world frame; the
caller picks the frame by choosing which lift to apply before calling
the core predict / innovation / update helpers.

Layers:

    measurement lifts           camera ← base / camera ← world
    ────────────────────────────────────────────────────────
    lift_measurement_base(...)  T_bc @ T_co_meas, Ad(T_bc) R_icp Ad(T_bc)^T
    lift_measurement_world(...) (T_wb T_bc) @ T_co_meas,
                                Ad(T_wb T_bc) R_icp Ad(..)^T + Σ_wb

    per-object EKF primitives   (frame-agnostic)
    ────────────────────────────────────────────────────────
    predict_ad_conjugate        μ ← Δ μ,  Σ ← Ad(Δ) Σ Ad(Δ)^T + Q,
                                followed by Φ saturation + diagonal floor
    innovation_from_belief      (ν, S, d², log_lik) at the tangent of μ
    joseph_update               IEKF + Joseph + Φ + diagonal floor
    merge_info_sum              Bayesian information-fusion of two Gaussians
    floor_diag                  per-axis diagonal floor

The Joseph update, saturation projector Φ, and floor_diag together
implement the three numerical safeguards of \S\ref{sec:predict} in
`docs/latex/ekf_update_implementation.tex`.

These functions are pure (no global state) and deterministic; they
obey the math in `docs/latex/bernoulli_ekf.tex` verbatim.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pose_update.state.ekf_se3 import (
    se3_exp, se3_log, se3_adjoint, saturate_covariance,
)


# Numerical floor for a log-likelihood (avoids -inf / NaN downstream).
LOG_EPS = -1e18


# ─────────────────────────────────────────────────────────────────────
# Diagonal floor
# ─────────────────────────────────────────────────────────────────────

def floor_diag(P: np.ndarray,
               P_min_diag: Optional[np.ndarray]) -> np.ndarray:
    """Lift the diagonal of P so that P_ii >= P_min_diag_i for every axis.

    Implementation: P_floored = P + diag(max(0, P_min - diag(P))). Adds a
    small PSD correction to the diagonal only; preserves off-diagonal
    correlations unchanged. Idempotent: if every P_ii already exceeds
    P_min_diag_i, returns P unchanged. Symmetry-safe.
    """
    if P_min_diag is None:
        return P
    cur = np.diag(P)
    bump = np.maximum(0.0, P_min_diag - cur)
    if not np.any(bump > 0.0):
        return P
    return P + np.diag(bump)


# ─────────────────────────────────────────────────────────────────────
# Measurement lifts — camera -> base / world
# ─────────────────────────────────────────────────────────────────────

def lift_measurement_base(T_co_meas: np.ndarray,
                           R_icp: np.ndarray,
                           T_bc: np.ndarray,
                           Ad_bc: Optional[np.ndarray] = None,
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Lift a camera-frame measurement to the base-frame tangent.

        T_bo_meas = T_bc · T_co_meas
        R_bo      = Ad(T_bc) · R_icp · Ad(T_bc)^T

    Σ_wb does NOT enter — this is the clean base-frame storage lift
    (bernoulli_ekf.tex §5). `Ad_bc` can be passed in for efficiency if
    the caller already has it cached; otherwise we compute it here.
    """
    if Ad_bc is None:
        Ad_bc = se3_adjoint(T_bc)
    T_bo_meas = T_bc @ np.asarray(T_co_meas, dtype=np.float64)
    R_sym = 0.5 * (R_icp + R_icp.T)
    R_bo = Ad_bc @ R_sym @ Ad_bc.T
    R_bo = 0.5 * (R_bo + R_bo.T)
    return T_bo_meas, R_bo


def lift_measurement_world(T_co_meas: np.ndarray,
                            R_icp: np.ndarray,
                            T_bc: np.ndarray,
                            T_wb: np.ndarray,
                            Sigma_wb: Optional[np.ndarray] = None,
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Lift a camera-frame measurement to the world-frame tangent.

        T_wc      = T_wb · T_bc
        T_wo_meas = T_wc · T_co_meas
        R_wo      = Ad(T_wc) · R_icp · Ad(T_wc)^T + Σ_wb      (Σ_wb optional)

    When `Sigma_wb` is None, the lift omits the SLAM-uncertainty term:
    that is the RBPF regime where every per-particle EKF conditions on
    that particle's sampled T_wb^k (so the conditional Σ_wb = 0).
    """
    T_wc = np.asarray(T_wb, dtype=np.float64) @ np.asarray(T_bc,
                                                           dtype=np.float64)
    Ad_wc = se3_adjoint(T_wc)
    T_wo_meas = T_wc @ np.asarray(T_co_meas, dtype=np.float64)
    R_sym = 0.5 * (R_icp + R_icp.T)
    R_wo = Ad_wc @ R_sym @ Ad_wc.T
    if Sigma_wb is not None:
        R_wo = R_wo + np.asarray(Sigma_wb, dtype=np.float64)
    R_wo = 0.5 * (R_wo + R_wo.T)
    return T_wo_meas, R_wo


# ─────────────────────────────────────────────────────────────────────
# EKF primitives (frame-agnostic; caller decides the frame via the lift)
# ─────────────────────────────────────────────────────────────────────

def predict_ad_conjugate(mu: np.ndarray,
                          cov: np.ndarray,
                          delta: np.ndarray,
                          Q: np.ndarray,
                          P_max: Optional[np.ndarray] = None,
                          P_min_diag: Optional[np.ndarray] = None,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Forward predict with a left-multiplicative SE(3) motion `delta`.

    μ_new = delta @ μ_old
    Σ_new = Ad(delta) · Σ_old · Ad(delta)^T + Q,
    then symmetrised, optional Φ saturation, optional diagonal floor.

    `delta = I` degenerates to the pure-noise inflation.

    Used by:
      * Gaussian static predict with `delta = u_k = T_wb,k^-1 T_wb,k-1`.
      * Gaussian / RBPF rigid-attachment predict with
            Gaussian: `delta = ΔT_bg`
            RBPF:     `delta = T_wb(t) · ΔT_bg · T_wb(t-1)^{-1}` (world-form)
      * RBPF static predict with `delta = I` (world-frame: μ unchanged).
    """
    Ad_d = se3_adjoint(delta)
    mu_new = delta @ mu
    cov_new = Ad_d @ cov @ Ad_d.T + Q
    cov_new = 0.5 * (cov_new + cov_new.T)
    if P_max is not None:
        cov_new = saturate_covariance(cov_new, P_max)
    if P_min_diag is not None:
        cov_new = floor_diag(cov_new, P_min_diag)
    return mu_new, cov_new


def innovation_from_belief(mu: np.ndarray,
                            cov: np.ndarray,
                            T_meas: np.ndarray,
                            R_meas: np.ndarray,
                            ) -> Tuple[np.ndarray, np.ndarray,
                                        float, float]:
    """Innovation quantities for (track-belief, lifted-measurement) pair.

    Returns `(ν, S, d², log_lik)` where:
        ν       = Log(μ^{-1} T_meas)          — 6-vec in tangent at μ
        S       = Σ + R_meas                   — residual covariance
        d²      = ν^T S^{-1} ν                 — Mahalanobis², ~χ²_6
        log_lik = -½ d² - ½ log det(2π S)      — Gaussian log-likelihood

    Both μ and T_meas must live in the same frame (base or world); the
    caller is responsible for lifting R_icp via
    `lift_measurement_base` or `lift_measurement_world` to produce R_meas.
    """
    cov_sym = 0.5 * (cov + cov.T)
    R_sym = 0.5 * (R_meas + R_meas.T)

    nu = se3_log(np.linalg.inv(mu) @ np.asarray(T_meas, dtype=np.float64))
    S = cov_sym + R_sym
    S = 0.5 * (S + S.T)

    sign, logdet = np.linalg.slogdet(S)
    try:
        Sinv_nu = np.linalg.solve(S, nu)
        d2 = float(nu @ Sinv_nu)
    except np.linalg.LinAlgError:
        d2 = float("inf")
    if sign <= 0 or not np.isfinite(logdet) or not np.isfinite(d2):
        log_lik = LOG_EPS
    else:
        two_pi_log = 6.0 * np.log(2.0 * np.pi)
        log_lik = -0.5 * d2 - 0.5 * (float(logdet) + two_pi_log)
    return nu, S, d2, log_lik


def joseph_update(mu: np.ndarray,
                   cov: np.ndarray,
                   T_meas: np.ndarray,
                   R_meas: np.ndarray,
                   iekf_iters: int = 2,
                   huber_w: float = 1.0,
                   P_max: Optional[np.ndarray] = None,
                   P_min_diag: Optional[np.ndarray] = None,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """IEKF update in the tangent at μ with Joseph-form covariance.

    Steps:
        R_eff = R_meas / w_huber        (w_huber ∈ (0, 1] scales R up)
        S     = Σ + R_eff
        K     = Σ · S^{-1}
        for _ in iekf_iters:
            δ = Log(μ_lin^{-1} · T_meas)
            μ_lin = μ · Exp(K · δ)
        Σ_post = (I-K) Σ (I-K)^T + K R_eff K^T    (Joseph form)
        Σ_post ← symmetrise → Φ saturate → diagonal floor
        μ ← μ_lin

    Returns the posterior `(μ_post, Σ_post)`. If `huber_w <= 0` (outer
    gate reject), the input is returned unchanged.
    """
    if huber_w <= 0.0:
        return mu, cov

    R_sym = 0.5 * (R_meas + R_meas.T)
    if 0.0 < huber_w < 1.0:
        R_sym = R_sym / huber_w

    S = cov + R_sym
    K = np.linalg.solve(S.T, cov.T).T   # Σ · S^{-1}
    I6 = np.eye(6)

    mu_lin = mu.copy()
    T_meas_a = np.asarray(T_meas, dtype=np.float64)
    for _ in range(max(1, iekf_iters)):
        delta = se3_log(np.linalg.inv(mu_lin) @ T_meas_a)
        mu_lin = mu @ se3_exp(K @ delta)

    I_K = I6 - K
    cov_post = I_K @ cov @ I_K.T + K @ R_sym @ K.T
    cov_post = 0.5 * (cov_post + cov_post.T)
    if P_max is not None:
        cov_post = saturate_covariance(cov_post, P_max)
    if P_min_diag is not None:
        cov_post = floor_diag(cov_post, P_min_diag)
    return mu_lin, cov_post


# ─────────────────────────────────────────────────────────────────────
# Bayesian information-sum merge (self-merge pass)
# ─────────────────────────────────────────────────────────────────────

def merge_info_sum(mu_a: np.ndarray, cov_a: np.ndarray,
                    mu_b: np.ndarray, cov_b: np.ndarray,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Bayesian fusion of two Gaussians on SE(3) linearised at μ_a.

        P_new^{-1} = P_a^{-1} + P_b^{-1}
        μ_new      = μ_a · Exp( P_new · P_b^{-1} · Log(μ_a^{-1} μ_b) )

    Both Gaussians are treated symmetrically through the information
    sum; the linearisation point is taken at μ_a for numerical
    convenience (any consistent choice gives the same answer to first
    order). The returned cov is symmetrised.
    """
    info_a = np.linalg.inv(cov_a)
    info_b = np.linalg.inv(cov_b)
    info_new = info_a + info_b
    cov_new = np.linalg.inv(info_new)
    cov_new = 0.5 * (cov_new + cov_new.T)

    delta_ab = se3_log(np.linalg.inv(mu_a) @ mu_b)
    correction = cov_new @ info_b @ delta_ab
    mu_new = mu_a @ se3_exp(correction)
    return mu_new, cov_new

"""
Extended Kalman Filter on SE(3) for per-object pose tracking.

Represents an object's pose posterior as (T, Σ) where T is a 4×4 SE(3) matrix
(the mean) and Σ is a 6×6 covariance in the se(3) tangent space with ordering
[v, ω] (translation first, rotation second).

Key operations:
    se3_exp, se3_log   — exponential / logarithm maps between SE(3) and se(3)
    ekf_predict        — prior update with process noise Q
    ekf_update         — measurement update in world frame
    ekf_update_base_frame — measurement update where prior and observation
                            share the base-to-world transform (avoids
                            Kalman overconfidence during HOLDING)
    pose_entropy       — scalar uncertainty = log det Σ

Paper references:
  * Noise composition follows Paper 1 (Popović et al. 2019) — upstream
    localization uncertainty is forwarded into the observation noise via
    the adjoint Jacobian.
  * The optional robust term (passed by caller via R_robust) implements the
    residual-adaptive weighting from Paper 2 (Chebrolu et al. 2021).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# SE(3) exp / log (Lie group maps)
# ─────────────────────────────────────────────────────────────────────

def _hat(omega: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from a 3-vector."""
    return np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ], dtype=np.float64)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map from se(3) twist ξ = [v, ω] (6,) to SE(3) (4, 4).

    Uses the closed-form Rodrigues formula with careful small-angle handling.
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    v, omega = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))

    T = np.eye(4, dtype=np.float64)

    if theta < 1e-10:
        # First-order approximation for small rotations
        T[:3, :3] = np.eye(3) + _hat(omega)
        T[:3, 3] = v
        return T

    K = _hat(omega / theta)
    K2 = K @ K
    s, c = np.sin(theta), np.cos(theta)

    R = np.eye(3) + s * K + (1.0 - c) * K2
    V = np.eye(3) + ((1.0 - c) / theta) * K + ((theta - s) / theta) * K2

    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithm map from SE(3) (4, 4) to se(3) twist [v, ω] (6,)."""
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]

    # Clamp for numerical safety (R may drift off SO(3) slightly)
    cos_theta = 0.5 * (np.trace(R) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-10:
        # Small-angle: omega ≈ vee((R - R^T) / 2)
        skew = 0.5 * (R - R.T)
        omega = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        v = t.copy()
    else:
        skew = (R - R.T) * (theta / (2.0 * np.sin(theta)))
        omega = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

        K = _hat(omega / theta)
        K2 = K @ K
        half_theta = 0.5 * theta
        # V^{-1}
        V_inv = (np.eye(3)
                 - 0.5 * theta * K
                 + (1.0 - half_theta / np.tan(half_theta)) * K2)
        v = V_inv @ t

    return np.concatenate([v, omega])


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """Adjoint Ad(T) mapping se(3) tangent vectors under the transform T.

    For T = [R, t; 0, 1], Ad(T) = [[R, [t]_× R], [0, R]] in (v, ω) ordering.
    """
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6), dtype=np.float64)
    Ad[:3, :3] = R
    Ad[:3, 3:] = _hat(t) @ R
    Ad[3:, 3:] = R
    return Ad


# ─────────────────────────────────────────────────────────────────────
# EKF predict / update
# ─────────────────────────────────────────────────────────────────────

def ekf_predict(T: np.ndarray, cov: np.ndarray,
                Q: np.ndarray,
                P_max: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction step with process noise Q added to covariance.

    Args:
        T:   (4, 4) SE(3) prior mean.
        cov: (6, 6) prior covariance in se(3).
        Q:   (6, 6) process noise covariance.
        P_max: optional (6, 6) covariance-saturation cap. When provided, the
               projector Phi of bernoulli_ekf.tex eq. (eq:phi) is applied
               after Q is added: P_out <- min(1, tr(P_max)/tr(P)) * P.
               Preserves symmetry and positive-definiteness; caps the trace
               at tr(P_max). When None (default), no cap is applied
               — the recursion matches the pre-Bernoulli behaviour.

    Returns:
        (T_pred, cov_pred) — mean unchanged (constant-velocity zero prior),
        covariance inflated by Q and optionally saturated.
    """
    cov_pred = cov + Q
    if P_max is not None:
        cov_pred = saturate_covariance(cov_pred, P_max)
    return T.copy(), cov_pred


def saturate_covariance(P: np.ndarray, P_max: np.ndarray) -> np.ndarray:
    """Covariance-saturation projector Phi from bernoulli_ekf.tex eq. (eq:phi).

        Phi(P) = min(1, tr(P_max) / tr(P)) * P.

    Scales P down uniformly so that tr(P) <= tr(P_max), preserving symmetry
    and positive-(semi-)definiteness. Acts as an additive static prior with
    total spread bounded by tr(P_max): the recursion is self-bounded under
    arbitrarily long missing-observation streaks.

    Args:
        P:     (6, 6) symmetric positive-semidefinite covariance.
        P_max: (6, 6) symmetric positive-definite cap. Only its trace is
               consulted, but callers should keep it PD so the design
               intent matches the equation.

    Returns:
        (6, 6) saturated covariance.
    """
    tr_P = float(np.trace(P))
    tr_Pmax = float(np.trace(P_max))
    if tr_P <= tr_Pmax or tr_P <= 0.0:
        return P
    scale = tr_Pmax / tr_P
    return scale * P


def huber_weight(d2: float,
                 G_in: float = 12.59,
                 G_out: float = 25.0) -> float:
    """Huber redescending M-estimator weight on a Mahalanobis squared
    residual; bernoulli_ekf.tex eq. (eq:huber).

        w = 1                     if d2 <= G_in
        w = sqrt(G_in / d2)       if G_in < d2 <= G_out
        w = 0                     if d2 > G_out

    The inner gate G_in is the chi^2_6(0.95) quantile (~12.59), the outer
    gate G_out is chi^2_6(0.9997) (~25). A caller with an observation of
    weight w scales its measurement noise by 1/w so the Kalman gain shrinks
    smoothly with |d|. w = 0 signals an outer-gate reject: the update is
    skipped and the measurement is routed to the unassigned branch of the
    Bernoulli update.

    Args:
        d2:    Mahalanobis squared distance (scalar).
        G_in:  inner-gate chi^2 quantile.
        G_out: outer-gate chi^2 quantile.

    Returns:
        scalar weight in [0, 1].
    """
    if not np.isfinite(d2) or d2 < 0.0:
        return 0.0
    if d2 <= G_in:
        return 1.0
    if d2 <= G_out:
        return float(np.sqrt(G_in / d2))
    return 0.0


def ekf_update(T_prior: np.ndarray, cov_prior: np.ndarray,
               T_meas: np.ndarray, R: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Measurement update in the same frame as the prior.

    The observation model is identity: the measurement is a direct observation
    of the pose variable. The innovation is computed in the tangent space.

    Args:
        T_prior:   (4, 4) prior mean.
        cov_prior: (6, 6) prior covariance.
        T_meas:    (4, 4) measurement mean.
        R:         (6, 6) measurement noise covariance.

    Returns:
        (T_posterior, cov_posterior).
    """
    # Innovation in tangent space at prior
    delta = se3_log(np.linalg.inv(T_prior) @ T_meas)

    S = cov_prior + R
    K = np.linalg.solve(S.T, cov_prior.T).T  # K = cov_prior @ S^{-1}

    correction = K @ delta
    T_post = T_prior @ se3_exp(correction)
    cov_post = (np.eye(6) - K) @ cov_prior
    # Symmetrize for numerical stability
    cov_post = 0.5 * (cov_post + cov_post.T)
    return T_post, cov_post


def ekf_update_base_frame(T_bo_prior: np.ndarray, cov_bo_prior: np.ndarray,
                          T_bo_meas: np.ndarray, R_bo: np.ndarray,
                          T_wb: np.ndarray, cov_wb: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """Update in base frame, then project to world frame.

    Use this when the prior and the measurement share the base-to-world
    transform T_wb (e.g., during HOLDING, where the EE-propagated prior
    and the camera observation both flow through T_wb). Fusing in world
    frame would count Σ_wb twice and cause Kalman overconfidence.

    Args:
        T_bo_prior:   (4, 4) prior object pose in base frame.
        cov_bo_prior: (6, 6) prior covariance in base frame.
        T_bo_meas:    (4, 4) measurement object pose in base frame.
        R_bo:         (6, 6) measurement noise in base frame
                      (does NOT include Σ_wb).
        T_wb:         (4, 4) base-to-world transform (used only for projection).
        cov_wb:       (6, 6) base-to-world covariance.

    Returns:
        (T_bo_post, cov_bo_post, T_wo_post, cov_wo_post)
        The base-frame posterior is the "clean" Bayesian fusion.
        The world-frame projection has Σ_wb injected as a lower bound.
    """
    T_bo_post, cov_bo_post = ekf_update(T_bo_prior, cov_bo_prior,
                                        T_bo_meas, R_bo)

    # Project to world frame
    T_wo = T_wb @ T_bo_post
    # Covariance of T_wo under the composition T_wb ⊕ T_bo:
    # Σ_wo ≈ Σ_wb + Ad(T_wb) Σ_bo Ad(T_wb)^T   (world frame tangent)
    Ad_wb = se3_adjoint(T_wb)
    cov_wo = cov_wb + Ad_wb @ cov_bo_post @ Ad_wb.T
    cov_wo = 0.5 * (cov_wo + cov_wo.T)
    return T_bo_post, cov_bo_post, T_wo, cov_wo


def compose_observation_noise(R_local: np.ndarray,
                              Sigma_wb: np.ndarray,
                              J: Optional[np.ndarray] = None
                              ) -> np.ndarray:
    """Compose observation noise per Paper 1 (forward SLAM uncertainty).

    R_eff = R_local + J · Σ_wb · J^T

    DEPRECATED for the RBPF fast tier. Under Rao-Blackwellization each
    particle conditions on its own T_wb sample with zero conditional
    uncertainty, so there is nothing to forward — adding Σ_wb to R here
    would double-count (the particle spread already represents it).
    This helper is kept only for legacy Gaussian-only consumers (e.g.,
    the factor-graph slow tier that treats T_wb as a fixed parameter).

    Args:
        R_local:  (6, 6) intrinsic measurement noise (e.g., from ICP fitness).
        Sigma_wb: (6, 6) base-to-world covariance from Layer 1.
        J:        (6, 6) Jacobian of residual w.r.t. T_wb. If None, use identity
                  (appropriate when the measurement is a world-frame pose).

    Returns:
        (6, 6) effective observation noise.
    """
    if J is None:
        J = np.eye(6)
    R_eff = R_local + J @ Sigma_wb @ J.T
    return 0.5 * (R_eff + R_eff.T)


# ─────────────────────────────────────────────────────────────────────
# Uncertainty summaries
# ─────────────────────────────────────────────────────────────────────

def pose_entropy(cov: np.ndarray) -> float:
    """Differential entropy of an SE(3) pose (Gaussian in tangent space).

    H = 0.5 * log((2πe)^6 · det(Σ)). Drops constants for use as a scalar
    uncertainty indicator: higher = more uncertain.

    Uses slogdet for numerical stability.
    """
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        # Degenerate; treat as very uncertain
        return float("inf")
    return 0.5 * logdet


def pose_is_uncertain(cov: np.ndarray, threshold: float = -5.0) -> bool:
    """Boolean uncertainty check for legacy callers that used `pose_uncertain`.

    The threshold is on pose_entropy; tunable. Default ~-5 corresponds
    roughly to a covariance with 1cm translation / 0.1 rad rotation scale.
    """
    return pose_entropy(cov) > threshold


# ─────────────────────────────────────────────────────────────────────
# Process noise schedules (Paper 3 — manipulation-phase-aware)
# ─────────────────────────────────────────────────────────────────────

# Small-scale constants for reference process noise magnitudes.
# All values are variances (σ²) for translation (m²) and rotation (rad²).
# Callers can override by passing their own Q.

_Q_STATIC_STABLE      = np.diag([1e-8]*3  + [1e-8]*3)   # effectively frozen
_Q_STATIC_UNSTABLE    = np.diag([1e-5]*3  + [1e-5]*3)   # slow drift
_Q_IDLE_DEFAULT       = np.diag([1e-6]*3  + [1e-6]*3)   # idle re-observation
_Q_JUST_RELEASED      = np.diag([1e-3]*3  + [1e-3]*3)   # may settle/roll
# Held object during grip closing/opening: a transient larger Q than
# steady holding (the object can shift inside the closing jaws by a
# few cm before the grip stabilises) but bounded by the proprio
# anchor — we know the gripper pose to mm via T_bg, so the held
# object's pose can't be "essentially unknown". A 1.0 m std (the
# original placeholder) explodes the cov 100x in one frame and
# defeats the rigid-attachment predict.
_Q_GRASPING_RELEASING = np.diag([0.02**2]*3 + [0.10**2]*3)
                                                  # 2 cm trans, ~6° rot
# Held object, base-frame fusion. The rigid-attachment mean model
# (μ ← ΔT_bg · μ) is only *approximately* correct: proprioception has
# ~mm-scale noise, the object can slip in the grip by cm, and the
# per-frame centroid-from-mask is noisy. Treating this as "effectively
# frozen" (1e-8) collapses the gate and every new detection goes to
# birth. A per-frame growth of √(2.5e-4) = 1.6 cm (trans) /
# √(1e-3) = 1.8 deg (rot) reaches P_max in ~250 frames of occlusion,
# which matches the physical reality (after ~8 s of not seeing a held
# object, we should expect ≥25 cm uncertainty).
_Q_HOLDING_BASE_FRAME = np.diag([2.5e-4]*3 + [1e-3]*3)  # grip slack + proprio jitter
_Q_HELD_WORLD_FRAME   = np.diag([1e-4]*3  + [1e-4]*3)   # inherits base drift


def process_noise_for_phase(phase: str,
                            is_target: bool,
                            frames_since_observation: int = 0,
                            frame: str = "world") -> np.ndarray:
    """Return process noise Q for a given manipulation phase and role.

    Args:
        phase:     'idle' | 'grasping' | 'holding' | 'releasing'.
        is_target: True if this is the object being manipulated.
        frames_since_observation: for stability decay.
        frame:     'world' (default) or 'base'. The 'base' frame applies
                   to the held object tracked in base-frame fusion — its
                   process noise is tiny because it's rigidly attached to
                   the EE in that frame.

    Returns:
        (6, 6) process noise covariance for one frame of prediction.
    """
    if is_target:
        if phase in ("grasping", "releasing"):
            return _Q_GRASPING_RELEASING.copy()
        if phase == "holding":
            return (_Q_HOLDING_BASE_FRAME.copy() if frame == "base"
                    else _Q_HELD_WORLD_FRAME.copy())
    # Non-target objects, or target during idle
    if frames_since_observation < 5:
        return _Q_IDLE_DEFAULT.copy()
    if frames_since_observation < 50:
        return _Q_STATIC_UNSTABLE.copy()
    return _Q_STATIC_STABLE.copy()


# ─────────────────────────────────────────────────────────────────────
# Beta-Bernoulli label belief (Paper 2 / Paper 3 calibrated scores)
# ─────────────────────────────────────────────────────────────────────

def update_label_belief(belief: dict, label: str, score: float
                        ) -> Tuple[dict, str]:
    """Update Beta-Bernoulli posterior for a label given a detection score.

    The score is treated as a Bernoulli observation (probability that the
    label applies). Under a Beta(1,1) prior:
        α += score
        β += (1 - score)

    The MAP label is argmax_k α_k / (α_k + β_k).

    Args:
        belief: dict mapping label → (α, β). Mutated-and-returned.
        label:  detected label.
        score:  calibrated detection score in [0, 1].

    Returns:
        (updated_belief, map_label).
    """
    score = float(np.clip(score, 0.0, 1.0))
    alpha, beta = belief.get(label, (1.0, 1.0))
    belief[label] = (alpha + score, beta + (1.0 - score))

    map_label = max(belief.items(),
                    key=lambda kv: kv[1][0] / (kv[1][0] + kv[1][1]))[0]
    return belief, map_label

"""Two dataclasses: :class:`BernoulliConfig` (fast-tier parameters) and :class:`TriggerConfig` (slow-tier scheduling)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Trigger policy (slow-tier scheduling)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TriggerConfig:
    """Slow-tier scheduling: fire on grasp / release / new-object / every-N-frames."""
    on_grasp: bool
    on_release: bool
    on_new_object: bool
    periodic_every_n_frames: int


# ─────────────────────────────────────────────────────────────────────
# Bernoulli-EKF mode config (docs/ekf_tracker/latex/bernoulli_ekf.tex)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BernoulliConfig:
    """Fast-tier parameters: Bernoulli existence model, gates, weights, birth thresholds, covariance clamps."""

    # ── Bernoulli existence model ────────────────────────────────
    association_mode: str
    p_s: float
    p_d: float
    alpha: float
    lambda_c: float
    lambda_b: float
    r_min: float

    # ── Probabilistic gates ──────────────────────────────────────
    G_in: float
    G_out: float
    G_in_trans: float
    G_out_trans: float
    G_out_rot: float
    gate_mode: str             # 'full' | 'trans' | 'trans_and_rot'
    cost_d2_mode: str          # 'full' | 'trans' | 'sum'
    max_residual_m: Optional[float]

    # ── Saturation cap / floor on P_bo (None disables) ───────────
    P_max: Optional[np.ndarray]
    P_min_diag: Optional[np.ndarray]

    # ── Robust / label switches ──────────────────────────────────
    enable_huber: bool
    init_cov_from_R: bool
    enforce_label_match: bool

    # ── Soft-mode cost augmentation ──────────────────────────────
    hungarian_label_penalty: float
    hungarian_score_weight: float

    # ── Subpart suppression ──────────────────────────────────────
    dedup_voxel_size_m: float
    dedup_containment_thresh: float
    dedup_require_same_label: bool

    # ── Birth gates ──────────────────────────────────────────────
    birth_border_margin_px: int
    birth_confirm_k: int
    birth_score_min: float
    birth_fitness_min: float
    birth_rmse_max: float
    birth_pending_ttl_frames: int
    birth_min_dist_m: float

    # ── Held-track anchoring ─────────────────────────────────────
    held_birth_radius_m: float
    held_meas_radius_m: float
    held_meas_innov_max_m: float
    r_held_floor: float

    # ── Self-merge ───────────────────────────────────────────────
    self_merge_trans_m: float

    # ── Scenario-specific runtime values (caller supplies) ───────
    K: Optional[np.ndarray]
    image_shape: Optional[Tuple[int, int]]
    T_bc: Optional[np.ndarray]

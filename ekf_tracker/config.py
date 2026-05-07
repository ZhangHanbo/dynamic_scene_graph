"""Shared dataclasses for the EKF tracker.

`BernoulliConfig` configures the Bernoulli-EKF fast tier
(see ``docs/ekf_tracker/latex/bernoulli_ekf.tex``).
`TriggerConfig` configures when the slow-tier `PoseGraphOptimizer` fires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Trigger policy (slow-tier scheduling)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TriggerConfig:
    """Configuration for when the slow tier fires.

    Fire on any manipulation event, on residual surprises, and as a periodic
    safety net every ~3 seconds at 30 Hz.
    """
    on_grasp: bool = True
    on_release: bool = True
    on_new_object: bool = True
    residual_threshold: float = 0.1        # in world-frame tangent norm
    periodic_every_n_frames: int = 90      # ~3 s at 30 Hz


# ─────────────────────────────────────────────────────────────────────
# Bernoulli-EKF mode config (docs/ekf_tracker/latex/bernoulli_ekf.tex)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BernoulliConfig:
    """Opts the fast tier into the Bernoulli-EKF behaviour.

    Default values match the paper (§12 Calibrated parameters).

    See ``docs/ekf_tracker/latex/bernoulli_ekf.tex`` for the full derivation.
    """
    association_mode: str = "hungarian"
    p_s: float = 1.0
    p_d: float = 0.9
    alpha: float = 4.4
    lambda_c: float = 1.0
    lambda_b: float = 1.0
    r_conf: float = 0.5
    r_min: float = 1e-3
    G_in: float = 12.59
    G_out: float = 25.0
    P_max: Optional[np.ndarray] = None
    enable_visibility: bool = True
    enable_huber: bool = True
    init_cov_from_R: bool = True
    enforce_label_match: bool = True
    # Soft perception-style cost terms (mirrors sam2_client._pair_cost):
    #   cost = d^2 - alpha*1[tau match] + label_penalty*1[label miss]
    #          + score_weight*(1 - score)
    hungarian_label_penalty: float = 0.0
    hungarian_score_weight: float = 0.0
    # d^2 block decomposition for the outer gate / cost.
    gate_mode: str = "full"          # 'full' | 'trans' | 'trans_and_rot'
    G_out_trans: float = 21.108      # chi^2_3(0.9997)
    G_out_rot: float = 21.108        # chi^2_3(0.9997)
    G_in_trans: float = 7.815        # chi^2_3(0.95) -- Huber inner gate (trans)
    G_in_rot: float = 7.815          # chi^2_3(0.95) -- Huber inner gate (rot)
    cost_d2_mode: str = "full"       # 'full' | 'trans' | 'sum'
    # Hard absolute-distance gate on the world-frame translation residual.
    max_residual_m: Optional[float] = 0.30
    # Per-axis floor on P_bo (passed into GaussianState).
    P_min_diag: Optional[np.ndarray] = None
    # Post-update track-to-track self-merge.
    self_merge_trans_m: float = 0.05
    self_merge_d2_trans: float = 0.0    # legacy Mahalanobis knob
    K: Optional[np.ndarray] = None
    image_shape: Optional[tuple] = None
    T_bc: Optional[np.ndarray] = None
    # Internal ICP knobs.
    icp_method: str = "icp_chain"      # PoseEstimator method
    icp_min_fitness: float = 0.75      # below this → centroid-only fallback
    icp_max_rmse: float = 0.015        # above this → centroid-only fallback
    icp_centroid_fallback_rot_var: float = 1e3
    icp_centroid_fallback_trans_std: float = 0.02
    # Pre-association voxel dedup of sub-part detections.
    dedup_voxel_size_m: float = 0.02
    dedup_containment_thresh: float = 0.8
    dedup_require_same_label: bool = False
    # Birth admission gates.
    birth_border_margin_px: int = 2
    birth_confirm_k: int = 3
    birth_score_min: float = 0.20
    birth_fitness_min: float = 0.5
    birth_rmse_max: float = 0.02
    birth_pending_ttl_frames: int = 30
    birth_min_dist_m: float = 0.05
    # Held-track anchoring.
    held_birth_radius_m: float = 0.25
    held_meas_radius_m: float = 0.25
    held_meas_innov_max_m: float = 0.20
    r_held_floor: float = 0.5
    r_held_min_match_frames: int = 5
    # Online relation detector.
    relation_backend: Optional[str] = None
    relation_server_url: Optional[str] = None
    relation_llm_model: str = "gpt-5.1"
    relation_score_threshold: float = 0.5
    relation_every_n_frames: int = 90
    relation_on_grasp: bool = True
    relation_on_release: bool = True
    relation_on_new_object: bool = True
    # Gravity-aware one-shot predict at the release transition.
    gravity_predict: bool = True
    workspace_floor_z: float = -1.0

    @classmethod
    def degeneracy(cls, **overrides) -> "BernoulliConfig":
        """Build a config that reproduces the pre-Bernoulli legacy behaviour
        exactly (used by the degeneracy test)."""
        base = dict(
            association_mode="oracle",
            p_s=1.0,
            p_d=0.9,
            alpha=0.0,
            lambda_c=1.0,
            lambda_b=1.0,
            r_conf=0.0,
            r_min=0.0,
            G_in=float("inf"),
            G_out=float("inf"),
            P_max=None,
            enable_visibility=False,
            enable_huber=False,
            init_cov_from_R=False,
            enforce_label_match=False,
            birth_border_margin_px=0,
            birth_confirm_k=1,
            birth_score_min=0.0,
            birth_fitness_min=0.0,
            birth_rmse_max=float("inf"),
        )
        base.update(overrides)
        return cls(**base)

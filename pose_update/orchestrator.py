"""
Two-tier orchestrator — Rao-Blackwellized variant.

Wires Layer 1 (SLAM) to Layer 2 (movable-object tracking) via a proper
Rao-Blackwellized factorization instead of collapsing the SLAM posterior
to one Gaussian and forwarding its covariance into R (the "cascaded KF"
pattern we deliberately avoid).

Factorization:
    p(x_{1:t}, {o}_i | z_{1:t}) = p(x_{1:t} | z_{1:t}) · Π_i p(o_i | x_{1:t}, z_{1:t})

* `p(x_{1:t} | z_{1:t})` — particles.
* `p(o_i | x_{1:t}, z_{1:t})` — per-particle EKF on SE(3), world frame.

Each detection contributes to (a) the per-particle object EKF and
(b) the per-particle log-weight, via the same innovation likelihood.
That is the "dual role" of vision the design discussion called out.

Slow tier (factor graph over raw observations) still exists. It consumes
the collapsed-mixture summary as its prior, runs the smoother, and the
posterior is injected back into every particle (Option A from the plan —
loses mixture structure but is the minimal integration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
from PIL import Image

from pose_update.state.slam_interface import (
    PoseEstimate, ParticlePose,
    collect_movable_masks, mask_out_movable, SlamBackend,
)
from pose_update.state.ekf_se3 import process_noise_for_phase, huber_weight
from pose_update.factor_graph import (
    PoseGraphOptimizer, Observation, RelationEdge, OptimizationResult,
)
from pose_update.state.rbpf_state import RBPFState
from pose_update.perception.icp_pose import PoseEstimator, centroid_cam_from_mask
from pose_update.perception.det_dedup import suppress_subpart_detections
from pose_update.perception.association import (
    hungarian_associate, oracle_associate,
    AssociationResult,
)
from pose_update.state.bernoulli import (
    r_predict, r_assoc_update_loglik, r_miss_update, r_birth,
)
from pose_update.perception.visibility import visibility_p_v
from pose_update.relations.relation_client import RelationClient, build_relation_client
from pose_update.manipulation.gravity_predict import predict_landing_pose
from pose_update.manipulation.object_dynamics import lookup_dynamics


# ─────────────────────────────────────────────────────────────────────
# Trigger policy
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Relation temporal filter
# ─────────────────────────────────────────────────────────────────────

class RelationFilter:
    """Exponential moving-average filter over scene-graph edges.

    Raw edge scores flicker frame-to-frame because the geometric
    relation test is noisy (random mock point clouds, bbox jitter).
    This filter smooths the binary present/absent signal into a
    stable 0-or-1 output.

    Per-edge EMA:
        ema(t) = α · raw(t) + (1 − α) · ema(t − 1)

    Output: emit the edge (score=1) when ema ≥ threshold, suppress
    otherwise. An edge not detected this frame gets raw=0.
    """

    def __init__(self, alpha: float = 0.3, threshold: float = 0.5):
        self.alpha = alpha
        self.threshold = threshold
        self._ema: Dict[tuple, float] = {}

    def update(self, raw_edges: List[RelationEdge]) -> List[RelationEdge]:
        """Accept raw edges from one frame; return the filtered set."""
        detected: Dict[tuple, float] = {}
        raw_meta: Dict[tuple, RelationEdge] = {}
        for edge in raw_edges:
            key = (edge.parent, edge.child, edge.relation_type)
            detected[key] = edge.score
            raw_meta[key] = edge

        all_keys = set(self._ema.keys()) | set(detected.keys())
        filtered: List[RelationEdge] = []
        for key in all_keys:
            raw = detected.get(key, 0.0)
            prev = self._ema.get(key, raw)
            ema = self.alpha * raw + (1.0 - self.alpha) * prev
            self._ema[key] = ema
            if ema >= self.threshold:
                parent, child, rel_type = key
                ref = raw_meta.get(key)
                filtered.append(RelationEdge(
                    parent=parent, child=child,
                    relation_type=rel_type,
                    score=1.0,
                    parent_size=ref.parent_size if ref else None,
                    child_size=ref.child_size if ref else None,
                ))
        # Prune dead edges (EMA decayed to near zero).
        self._ema = {k: v for k, v in self._ema.items() if v > 0.01}
        return filtered


# ─────────────────────────────────────────────────────────────────────
# Trigger policy
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
# Bernoulli-EKF mode config (docs/latex/bernoulli_ekf.tex)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BernoulliConfig:
    """Opts the orchestrator into the Bernoulli-EKF fast tier.

    Default values match the paper (§12 Calibrated parameters). Setting
    ``association_mode='oracle'`` + ``P_max=None`` + ``r_min=0.0`` +
    ``enable_visibility=False`` + ``G_in=G_out=inf`` reproduces the
    pre-Bernoulli legacy behaviour (the five substitutions of the
    degeneracy analysis).

    Attributes:
        association_mode: 'hungarian' for the full Mahalanobis cost with
            chi^2 gating, 'oracle' to short-circuit via ``det['id']``.
        p_s, p_d, lambda_c, lambda_b: Bayesian rates (§5, §7, §8).
        alpha: SAM2 tracklet-ID matching bonus in chi^2_6 units
            (paper §6.1 eq:cost_sam2). Default 4.4 (mild tie-breaker
            toward ID continuity); 0 disables the bonus.
        r_conf, r_min: emission and pruning thresholds on existence r.
        G_in, G_out: chi^2 Huber gates (eq:huber). +inf disables Huber
            re-weighting (w=1 always) and the outer gate effectively
            admits every pair (matched by the inner-gate branch).
        P_max: covariance-saturation cap (eq:phi). None = no cap.
        enable_visibility: if False, p_v = 1 for all tracks (miss branch
            unchanged from pre-Bernoulli); otherwise compute eq:pv. Also
            auto-disabled when T_bc is None (see below), because the
            pinhole projection requires a valid T_cw = T_wb * T_bc.
        K, image_shape: camera intrinsics / image size for p_v.
        T_bc: (4,4) camera-in-base transform. Required for visibility in
            any deployment where the camera frame differs from the base
            frame (e.g. Fetch's head camera). If None, visibility is
            skipped (p_v = 1.0) regardless of `enable_visibility`, because
            passing T_wb as T_cw projects through the wrong axes and
            yields random in/out-of-frustum results.
        init_cov_from_R: if True, birth covariance = R_icp (paper §8);
            if False, use the orchestrator's _INIT_OBJ_COV constant.
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
    # Activate by setting `enforce_label_match=False` and choosing
    # positive penalties. label_penalty in chi^2_6 units; values around
    # 4-8 let a geometrically excellent (small d^2) pair override a
    # noisy label disagreement, but a same-label feasible pair is still
    # preferred. score_weight in chi^2_6 units; 1-3 mildly disfavours
    # low-confidence detections.
    hungarian_label_penalty: float = 0.0
    hungarian_score_weight: float = 0.0
    # d^2 block decomposition for the outer gate / cost. The full 6-D
    # d^2 (paper baseline) is dominated by the rotation block when
    # ICP rotation is unreliable -- per-oid ICP chains drift in
    # rotation even when translation matches at 1mm, and the rotation
    # mismatch lifts d^2 above G_out=25, sending the same-position
    # re-detection to birth (the "track explosion" we measured).
    # Switch to gate_mode='trans' (chi^2_3 quantile on the translation
    # block only) to decouple the gate from rotation noise; rotation
    # can still influence the cost via cost_d2_mode='full' or 'sum'.
    gate_mode: str = "full"          # 'full' | 'trans' | 'trans_and_rot'
    G_out_trans: float = 21.108      # chi^2_3(0.9997)
    G_out_rot: float = 21.108        # chi^2_3(0.9997)
    G_in_trans: float = 7.815        # chi^2_3(0.95) -- Huber inner gate (trans block)
    G_in_rot: float = 7.815          # chi^2_3(0.95) -- Huber inner gate (rot block)
    cost_d2_mode: str = "full"       # 'full' | 'trans' | 'sum'
    # Hard absolute-distance gate on the world-frame translation
    # residual ‖ν[:3]‖ (metres). Catches the inflated-covariance
    # pathology where the chi² gate becomes meaningless during long
    # miss runs (a 0.8 m residual against σ ≈ 0.4 m gives d² ≈ 17,
    # under any chi² gate). Default 0.30 m. Set to None to disable.
    max_residual_m: Optional[float] = 0.30
    # Per-axis floor on P_bo (passed into GaussianState). Prevents the
    # EKF posterior from shrinking below realistic per-frame perception
    # jitter. None = no floor (paper baseline).
    P_min_diag: Optional[np.ndarray] = None
    # Post-update track-to-track self-merge.
    # Two surviving same-label tracks merge iff the Euclidean distance
    # between their (world-frame) means is below `self_merge_trans_m`.
    # Default 0.05 m (5 cm) ≈ one object radius for apples / cups / cans.
    # Euclidean (not Mahalanobis) so the merge radius does NOT scale with
    # the tracks' uncertainty — prevents the "two just-born tracks at 22 cm
    # collapse because σ=5 cm" failure mode.
    # Set to 0.0 to disable the merge pass entirely.
    self_merge_trans_m: float = 0.05
    # Legacy Mahalanobis knob, kept for backward compat; not used when
    # `self_merge_trans_m > 0`.
    self_merge_d2_trans: float = 0.0
    K: Optional[np.ndarray] = None
    image_shape: Optional[tuple] = None
    T_bc: Optional[np.ndarray] = None
    # Internal ICP (step 4): when K is set, the orchestrator owns ICP.
    # Pipeline becomes: coarse centroid → Hungarian on 3-DoF → fine ICP with
    # track prior per matched pair → 6-DoF update (or centroid-only fallback
    # if fitness gate fails).
    # When K is None, the orchestrator falls back to consuming a pre-computed
    # det["T_co"] / det["R_icp"] (upstream-ICP contract; backward compat).
    icp_method: str = "icp_chain"      # PoseEstimator method
    icp_min_fitness: float = 0.75      # below this → centroid-only fallback
    icp_max_rmse: float = 0.015        # above this → centroid-only fallback
    # Centroid-only fallback for matched pairs whose fine ICP failed: build
    # T_co_fb from (predicted rotation, measured centroid) + inflated R_icp
    # on the rotation block so only translation information reaches the EKF.
    icp_centroid_fallback_rot_var: float = 1e3   # rad² on R_icp diag rows 3-5
    icp_centroid_fallback_trans_std: float = 0.02  # m (σ for trans rows)
    # Pre-association voxel dedup of sub-part detections (single frame).
    # Two detections whose back-projected voxel sets satisfy
    # |A ∩ B| / min(|A|, |B|) > dedup_containment_thresh are collapsed
    # into one (smaller absorbed into larger, label histories merged).
    # Label-agnostic by default; `_voxels` is cached on each survivor for
    # downstream reuse. Runs only when `K` is set.
    dedup_voxel_size_m: float = 0.02
    dedup_containment_thresh: float = 0.8
    dedup_require_same_label: bool = False
    # Birth admission gates (applied before r_birth; reject ⇒ detection is
    # silently dropped for this frame, not consumed as a birth).
    #   A) border reject: refuse if the bbox touches any image edge (most
    #      false positives from gripper-on-arm enter through the left edge).
    #   B) SAM2 confirmation: require the per-label n_obs reported by
    #      perception to reach `birth_confirm_k` before spawning — filters
    #      transient detections whose mask never stabilises.
    #   C) score / ICP quality: floors on detection score and ICP fitness,
    #      ceiling on ICP rmse. Tuned against the Fetch apple-in-tray data
    #      where the spurious "cola"=gripper class is pinned at s=0.195.
    # Disable each with margin_px=0 / confirm_k=1 / thresholds at
    # 0 / 0 / +inf respectively.
    birth_border_margin_px: int = 2
    birth_confirm_k: int = 3
    birth_score_min: float = 0.20
    birth_fitness_min: float = 0.5
    birth_rmse_max: float = 0.02
    # Tracker-side pending-birth buffer: a detection's perception id
    # that goes unmatched for `birth_confirm_k` frames (within a rolling
    # window of `birth_pending_ttl_frames`) is admissible. The TTL makes
    # a brief perception dropout non-fatal to the counter.
    birth_pending_ttl_frames: int = 30
    # Proximity gate against existing live tracks: refuse to birth if
    # the candidate's world-frame centroid is within this distance of
    # ANY same-label live track. Catches SAM2-id-reseed duplicates that
    # the policy gates would otherwise admit (the new perception id
    # passes border + n_obs + score, but the underlying physical object
    # is already tracked under another oid). 15 cm > typical mask-reseed
    # offset (~5-10 cm) and < apple/bottle physical spacing in a tray.
    birth_min_dist_m: float = 0.05
    # Held-track anchoring on T_we = T_wb @ T_bg (proprioception):
    #   - Births: when checking the proximity gate against the held oid,
    #     substitute T_we for the held track's drifted mu_w. Uses
    #     `held_birth_radius_m` (wider than `birth_min_dist_m` because
    #     mask centroids of a partially-occluded held object are noisy).
    #   - Held-track measurements: in the matched-update loop, reject a
    #     measurement whose centroid_w differs from T_we by more than
    #     `held_meas_radius_m` (gripper-occluded sliver-centroid bias
    #     would otherwise pull mu off-target).
    #   - Held existence floor: if the held track gets no successful
    #     measurement update for ≥ `r_held_min_match_frames` frames in
    #     a row, do NOT decay r below `r_held_floor` — the gripper is
    #     closed; we know the object exists.
    held_birth_radius_m: float = 0.25
    held_meas_radius_m: float = 0.25
    # Hard ceiling on the per-update world-frame innovation
    # ‖centroid_w − μ_w‖ for the held track. The chi² gate alone
    # doesn't bound metric jumps: when the predicted μ has drifted but
    # the measurement sits near the gripper, an in-gate update can
    # snap μ by 30+ cm in one frame. With this clamp, such updates
    # are skipped (treated as miss) and the rigid-attach predict
    # carries the track instead. 0.20 m is conservative — raise if
    # legitimate held-object motion gets rejected.
    held_meas_innov_max_m: float = 0.20
    r_held_floor: float = 0.5
    r_held_min_match_frames: int = 5
    # Online relation detector. None → mock geometric test (legacy default).
    #   "rest" → alpha_robot's SuppRelAfford /relation_det_with_bboxes
    #   "llm"  → GPTChatBot prompted with RGB + numbered bboxes
    # `relation_server_url` overrides the default from
    # `arobot.configs.IP_CONFIGS["SuppRelAfford"]` (REST backend only).
    # Per-edge temporal aggregation is handled by the existing
    # RelationFilter EMA; no extra smoothing is needed here.
    relation_backend: Optional[str] = None
    relation_server_url: Optional[str] = None
    relation_llm_model: str = "gpt-5.1"
    relation_score_threshold: float = 0.5
    # Relation detector is expensive (REST: ~100 ms, LLM: ~1-3 s). We don't
    # need the scene graph at per-frame rate — physical scene structure changes
    # only when something moves. Gate the detector on:
    #   - a periodic tick (every N frames; 90 ≈ 3 s at 30 Hz);
    #   - manipulation events (grasp / release) that imply a graph change;
    #   - a new track birth (a new object may gain / give parenthood).
    # Between firings, `_cached_relations` is reused as-is (EMA is driven only
    # by actual detector calls, so it doesn't decay during quiet stretches).
    # Set `relation_every_n_frames = 0` to disable the periodic tick.
    relation_every_n_frames: int = 90
    relation_on_grasp: bool = True
    relation_on_release: bool = True
    relation_on_new_object: bool = True
    # Gravity-aware one-shot predict at the release transition. When
    # True and `voxel_obs` is set on the orchestrator, the EKF mean +
    # covariance for the just-released object are replaced by the
    # post-fall prediction from `pose_update.manipulation.gravity_predict`. Set
    # `False` to fall back to the existing static-Q-only predict.
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
            alpha=0.0,         # no SAM2 bonus
            lambda_c=1.0,
            lambda_b=1.0,
            r_conf=0.0,        # emit everything
            r_min=0.0,         # never prune
            G_in=float("inf"),
            G_out=float("inf"),
            P_max=None,
            enable_visibility=False,
            enable_huber=False,
            init_cov_from_R=False,
            enforce_label_match=False,
            # Disable birth-admission gates so the legacy-reproducing path
            # births on every unmatched detection (matching the pre-gate
            # behaviour of the orchestrator).
            birth_border_margin_px=0,
            birth_confirm_k=1,
            birth_score_min=0.0,
            birth_fitness_min=0.0,
            birth_rmse_max=float("inf"),
        )
        base.update(overrides)
        return cls(**base)


def birth_admissible(det: Dict[str, Any],
                      cfg: "BernoulliConfig",
                      image_shape: Optional[tuple],
                      *,
                      tracker_n_obs: Optional[int] = None,
                      tracker_max_score: Optional[float] = None,
                      require_pose: bool = True,
                      ) -> tuple:
    """Decide if an unmatched detection is eligible to spawn a new track.

    Policy gates (all must pass), each individually disablable via the
    corresponding `BernoulliConfig` field:
      A) `birth_border_margin_px`: bbox must not touch the image border
         within `margin_px` pixels. Rejects gripper-on-arm detections that
         hang into a fixed edge of the head camera.
      B) `birth_confirm_k`: either the tracker-side unmatched streak
         (`tracker_n_obs`, preferred) or perception's per-label `n_obs`
         (legacy fallback) must be ≥ k. Rejects transient detections.
      C) `birth_score_min`: floor on detection score (or `tracker_max_score`
         if the caller is tracking a rolling max across unmatched frames).

    ICP-quality gates (only fire when `require_pose=True` AND T_co is
    present): `birth_fitness_min` / `birth_rmse_max`. These are kept only
    for backwards-compatible callers that still run ICP as a birth probe;
    the new pending-buffer path runs ICP only on admission, so it passes
    `require_pose=False` and these gates are skipped.

    Returns:
        (admit: bool, reason: str). `reason` is one of
        "ok"/"border"/"confirm"/"score"/"fitness"/"rmse"/"no_pose".
    """
    if require_pose and det.get("T_co") is None:
        return False, "no_pose"

    # A) image-border touch.
    margin = int(cfg.birth_border_margin_px)
    if margin > 0:
        box = det.get("box")
        if box is not None and image_shape is not None:
            try:
                x0, y0, x1, y1 = (float(b) for b in box)
                H_img, W_img = int(image_shape[0]), int(image_shape[1])
                if (x0 <= margin
                        or y0 <= margin
                        or x1 >= W_img - 1 - margin
                        or y1 >= H_img - 1 - margin):
                    return False, "border"
            except (TypeError, ValueError):
                pass  # malformed box -> don't reject on border alone

    # B) Confirmation count. Prefer the tracker-side counter when the
    # caller provides one (pending-buffer path); fall back to perception's
    # per-label n_obs for legacy callers that don't track their own.
    k = int(cfg.birth_confirm_k)
    if k > 1:
        if tracker_n_obs is not None:
            if int(tracker_n_obs) < k:
                return False, "confirm"
        else:
            label = det.get("label")
            labels = det.get("labels") or {}
            n_obs = 0
            if (isinstance(labels, dict)
                    and isinstance(labels.get(label), dict)):
                try:
                    n_obs = int(labels[label].get("n_obs", 0))
                except (TypeError, ValueError):
                    n_obs = 0
            if n_obs == 0 and "n_obs" in det:
                try:
                    n_obs = int(det["n_obs"])
                except (TypeError, ValueError):
                    n_obs = 0
            if n_obs < k:
                return False, "confirm"

    # C) score floor. Prefer the rolling max across unmatched frames
    # (tracker_max_score) when provided.
    if cfg.birth_score_min > 0.0:
        if tracker_max_score is not None:
            try:
                score = float(tracker_max_score)
            except (TypeError, ValueError):
                score = 0.0
        else:
            try:
                score = float(det.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
        if score < cfg.birth_score_min:
            return False, "score"

    # D) ICP-quality gates — legacy only (ICP runs as a birth probe).
    # The new pending-buffer path passes require_pose=False and skips these;
    # ICP runs once on admission and its quality is not used as a gate.
    if require_pose:
        if cfg.birth_fitness_min > 0.0:
            fit = det.get("fitness")
            if fit is not None:
                try:
                    if float(fit) < cfg.birth_fitness_min:
                        return False, "fitness"
                except (TypeError, ValueError):
                    pass
        if math.isfinite(cfg.birth_rmse_max):
            rmse = det.get("rmse")
            if rmse is not None:
                try:
                    if float(rmse) > cfg.birth_rmse_max:
                        return False, "rmse"
                except (TypeError, ValueError):
                    pass

    return True, "ok"


# ─────────────────────────────────────────────────────────────────────
# Pending-birth buffer (tracker-side, separate from perception state)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class _PendingBirth:
    """One tracker-side candidate birth, keyed by perception id.

    This is the buffer that accumulates counters across frames for a
    detection stream that keeps arriving unmatched. Lives entirely on
    the tracker side — perception's `n_obs` is used only as a seed for
    `max_score`. The tracker oid is NOT allocated until admission.
    """
    source_id: Any              # perception's det["id"]; metadata only
    first_seen_frame: int
    last_seen_frame: int
    n_obs_tracker: int = 0      # frames seen unmatched in this tracker
    max_score: float = 0.0
    last_label: Optional[str] = None

    @classmethod
    def from_det(cls, det: Dict[str, Any], frame: int) -> "_PendingBirth":
        try:
            score = float(det.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return cls(
            source_id=det.get("id"),
            first_seen_frame=frame,
            last_seen_frame=frame,
            n_obs_tracker=0,
            max_score=score,
            last_label=det.get("label"),
        )

    def bump(self, det: Dict[str, Any], frame: int) -> None:
        self.last_seen_frame = frame
        self.n_obs_tracker += 1
        try:
            score = float(det.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score > self.max_score:
            self.max_score = score
        if det.get("label"):
            self.last_label = det.get("label")


# ─────────────────────────────────────────────────────────────────────
# Two-tier orchestrator (RBPF)
# ─────────────────────────────────────────────────────────────────────

class TwoTierOrchestrator:
    """Coordinator for the hierarchical movable-object tracking pipeline.

    The SLAM backend may return a `PoseEstimate` (Gaussian) or a `ParticlePose`.
    Internally we always keep N particles; if the backend gave us a Gaussian,
    we sample fresh each frame (approximate RBPF — particle identity across
    frames is carried by the *slot* index, not by trajectory continuity).

    Attributes:
        state:             RBPFState — particles + per-particle per-object EKFs
        n_particles:       int
        object_labels:     Dict[int, str]
        object_first_seen: Dict[int, int] (frame idx of first detection)
        frames_since_obs:  Dict[int, int]
        T_oe:              Dict[int, Optional[np.ndarray]]
                           (locked at grasp onset, object-in-EE frame)
        frame_count, last_opt_frame, last_state — orchestrator bookkeeping.
    """

    # Phases during which a manipulation-set member rides the gripper
    # rigidly. `releasing` is excluded — the held object has been let
    # go even though the FSM still carries `held_obj_id` until it
    # reaches `idle`.
    _RIGID_PHASES: Set[str] = frozenset(("grasping", "holding"))

    @staticmethod
    def should_apply_rigid(
        T_bg: Optional[np.ndarray],
        prev_T_bg: Optional[np.ndarray],
        manipulation_set: Set[int],
        phase: str,
    ) -> bool:
        """Predicate for the rigid-attachment branch of the fast tier.

        Rigid attach is applied iff (a) we have two consecutive
        proprio samples to form ΔT_bg, (b) some object is in the
        manipulation set, AND (c) the gripper is actually closed
        around the object (phase in {grasping, holding}).
        """
        return (T_bg is not None
                and prev_T_bg is not None
                and bool(manipulation_set)
                and phase in TwoTierOrchestrator._RIGID_PHASES)

    # Default loose init covariance for a newly-detected object (same
    # magnitude the legacy orchestrator used).
    # Birth covariance: σ_trans = 0.05 m = 5 cm per axis, σ_rot = 0.05 rad ≈ 2.9°.
    # Note the stored values are variances (σ²), hence 0.05² = 2.5e-3.
    _INIT_OBJ_COV = np.diag([0.05**2] * 3 + [0.05**2] * 3)
    # Additional per-step noise for manipulated objects (grip slack /
    # deformation). Additive on top of Ad(ΔT)·Σ·Adᵀ.
    _Q_MANIP_PER_STEP = np.diag([1e-6] * 3 + [1e-6] * 3)
    # Uniform Q used by the TSDF++-style vision-only baseline. Chosen so
    # the Kalman gain lands near 0.5 under R_icp ~ 1e-4, which allows the
    # filter to chase vision for moving objects without runaway noise for
    # static ones. The baseline has no manipulation phase to branch on.
    _Q_BASELINE = np.diag([1e-4] * 3 + [1e-4] * 3)

    def __init__(self,
                 slam_backend: SlamBackend,
                 trigger: Optional[TriggerConfig] = None,
                 optimizer: Optional[PoseGraphOptimizer] = None,
                 n_particles: int = 32,
                 ess_resample_frac: float = 0.5,
                 iekf_iters: int = 2,
                 rng_seed: Optional[int] = None,
                 verbose: bool = False,
                 baseline_mode: bool = False,
                 bernoulli: Optional[BernoulliConfig] = None):
        """Args:
            baseline_mode: if True, ignore all proprioception (T_ec, T_bg,
                gripper_state) and use a uniform vision-friendly Q. This
                is the TSDF++-style baseline: vision-only, no manipulation
                awareness, no rigid-attachment predict.
            bernoulli: if not None, use the Bernoulli-EKF fast tier
                (bernoulli_ekf.tex). The default None preserves the legacy
                upstream-ID lookup behaviour for backward compatibility.
        """
        self.slam = slam_backend
        self.trigger = trigger or TriggerConfig()
        self.optimizer = optimizer or PoseGraphOptimizer()
        self.verbose = verbose
        self.baseline_mode = baseline_mode
        self.bernoulli = bernoulli  # None = legacy path

        self.n_particles = n_particles
        self.ess_resample_frac = ess_resample_frac
        self.iekf_iters = iekf_iters

        rng = np.random.default_rng(rng_seed) if rng_seed is not None \
            else np.random.default_rng()
        self.state = RBPFState(n_particles=n_particles, rng=rng)

        # Object-level metadata (particle-independent)
        self.object_labels: Dict[int, str] = {}
        self.object_first_seen: Dict[int, int] = {}
        self.frames_since_obs: Dict[int, int] = {}
        self.T_oe: Dict[int, Optional[np.ndarray]] = {}

        # Gravity-aware predict on release. The driver constructs and
        # populates `voxel_obs` (a `pose_update.perception.voxel_observability.
        # VoxelObservability`) per frame; when None, the gravity hook
        # short-circuits and behaviour is identical to the legacy path.
        self.voxel_obs = None
        self._gravity_predict_log: List[Dict[str, Any]] = []

        # Bernoulli-EKF per-track state (only populated when bernoulli != None):
        #   existence r^(i)_{k|k} ; last-matched SAM2 tracklet id tau^(i).
        self.existence: Dict[int, float] = {}
        self.sam2_tau: Dict[int, int] = {}
        # Soft per-track label history (mirrors perception's `labels` field
        # in detection_h JSON: label -> {n_obs, sum_score, mean_score}). Used
        # by the soft-label cost in `hungarian_associate` and reported back to
        # downstream consumers via `self.objects[oid]["label_scores"]`.
        self.label_scores: Dict[int, Dict[str, Dict[str, float]]] = {}

        self.frame_count = 0
        self.last_opt_frame = -1
        self.last_state: Dict[str, Any] = {
            "phase": "idle", "held_obj_id": None}

        self._cached_relations: List[RelationEdge] = []
        self._relation_filter = RelationFilter(alpha=0.3, threshold=0.5)
        # Previous gripper pose in base frame; used to compute ΔT for the
        # rigid-attachment predict of the manipulation set.
        self._prev_T_bg: Optional[np.ndarray] = None

        # Internal PoseEstimator for step-4 "ICP inside the tracker" pipeline.
        # Active only when BernoulliConfig.K is provided; otherwise the tracker
        # falls back to consuming upstream-supplied T_co / R_icp per detection.
        self.pose_est: Optional[PoseEstimator] = None
        if bernoulli is not None and bernoulli.K is not None:
            self.pose_est = PoseEstimator(
                np.asarray(bernoulli.K, dtype=np.float64),
                method=bernoulli.icp_method,
            )
            self.pose_est.MIN_FITNESS = float(bernoulli.icp_min_fitness)
            self.pose_est.MAX_RMSE = float(bernoulli.icp_max_rmse)

        # Online relation detector (REST or LLM). Lazy init — remote
        # construction happens on first detect() call.
        self.relation_client: Optional[RelationClient] = None
        if bernoulli is not None and bernoulli.relation_backend is not None:
            self.relation_client = build_relation_client(
                backend=bernoulli.relation_backend,
                server_url=bernoulli.relation_server_url,
                llm_model=bernoulli.relation_llm_model,
            )
        # Stashed each step() so _recompute_relations can reach the current
        # RGB + per-detection oid mapping.
        self._frame_rgb: Optional[np.ndarray] = None
        self._last_det_to_oid: Dict[int, int] = {}  # detection index → track oid
        # Relation-tick bookkeeping (see BernoulliConfig.relation_*).
        self._last_relation_frame: int = -10**9  # forces a fire on frame 0
        self._known_oids_before_step: Set[int] = set()

        # Tracker-side pending-birth buffer. Keyed by perception id
        # (det["id"]); NOT by tracker oid. A tracker oid is minted only
        # when an entry is admitted, so `pose_est._refs` and `object_labels`
        # stay in lockstep. Perception id is metadata, stored as
        # `source_id` inside each entry; the tracker never treats it as
        # authoritative identity.
        self._pending_births: Dict[Any, _PendingBirth] = {}

    # --------------------------------------------------------------- #
    #  Backward-compatibility view: dict-of-dicts like the old API
    # --------------------------------------------------------------- #

    @property
    def objects(self) -> Dict[int, Dict[str, Any]]:
        """Collapsed-mixture view for legacy consumers.

        Each entry has the same shape the old orchestrator exposed:
            {"T": (4,4), "cov": (6,6), "label": str,
             "frames_since_observation": int, "T_oe": Optional[(4,4)]}
        In Bernoulli mode, tracks whose r_{k|k} < r_conf are filtered out
        of this view (they remain predicted / associated / updated but are
        tentative; callers should use .tentative_objects to see them).
        """
        out: Dict[int, Dict[str, Any]] = {}
        r_conf = (self.bernoulli.r_conf
                  if self.bernoulli is not None else 0.0)
        for oid, pe in self.state.collapsed_objects().items():
            if self.bernoulli is not None:
                r = self.existence.get(oid, 0.0)
                if r < r_conf:
                    continue
            entry = {
                "T": pe.T,
                "cov": pe.cov,
                "label": self.object_labels.get(oid, "unknown"),
                "frames_since_observation":
                    self.frames_since_obs.get(oid, 0),
                "T_oe": self.T_oe.get(oid),
            }
            if self.bernoulli is not None:
                entry["r"] = float(self.existence.get(oid, 0.0))
                entry["sam2_id"] = int(self.sam2_tau.get(oid, -1))
                # Soft per-track label distribution (mirrors perception's
                # `labels` field: label -> {n_obs, mean_score}). Lets
                # downstream consumers see the full label history rather than
                # just the primary label string.
                entry["label_scores"] = {
                    lbl: {"n_obs": int(stats["n_obs"]),
                          "mean_score": float(stats["mean_score"])}
                    for lbl, stats in self.label_scores.get(oid, {}).items()
                }
            out[oid] = entry
        return out

    @property
    def tentative_objects(self) -> Dict[int, Dict[str, Any]]:
        """Bernoulli mode: tracks with r < r_conf (below emission).

        Empty in legacy mode.
        """
        if self.bernoulli is None:
            return {}
        out: Dict[int, Dict[str, Any]] = {}
        r_conf = self.bernoulli.r_conf
        for oid, pe in self.state.collapsed_objects().items():
            r = self.existence.get(oid, 0.0)
            if r >= r_conf:
                continue
            out[oid] = {
                "T": pe.T,
                "cov": pe.cov,
                "label": self.object_labels.get(oid, "unknown"),
                "r": float(r),
                "sam2_id": int(self.sam2_tau.get(oid, -1)),
            }
        return out

    # --------------------------------------------------------------- #
    #  Public entry point
    # --------------------------------------------------------------- #

    def step(self,
             rgb: np.ndarray,
             depth: np.ndarray,
             detections: List[Dict[str, Any]],
             gripper_state: Dict[str, Any],
             T_ec: Optional[np.ndarray] = None,
             T_bg: Optional[np.ndarray] = None,
             odom_prior: Optional[PoseEstimate] = None,
             ) -> Dict[str, Any]:
        """Process one frame end-to-end.

        Args:
            rgb, depth: current frame images.
            detections: list of dicts with at minimum 'id' (int), 'mask' (H,W),
                'label' (str), 'score' (float), 'T_co' (4,4 ICP), 'R_icp'
                (6,6), 'fitness', 'rmse'. Detections without an 'id' are
                ignored (no data association done here).
            gripper_state: {'phase': 'idle'|'grasping'|'holding'|'releasing',
                            'held_obj_id': int or None}.
            T_ec: end-effector-to-camera transform (needed at grasp onset to
                  lock T_oe).
            T_bg: gripper pose in base frame from proprioception. Used by the
                  rigid-attachment predict for the manipulation set.
            odom_prior: optional odometry prior for the SLAM backend.

        Returns:
            Report dict. Backward-compatible keys: 'slam_pose', 'triggered',
            'alpha', 'residuals', 'objects'. Added: 'slam_raw',
            'base_particles', 'ess', 'resampled'.
        """
        report: Dict[str, Any] = {}
        # Snapshot known objects BEFORE the fast tier can add newly-seen
        # ones; the new-object trigger compares against this set.
        self._known_before_this_step = set(self.object_labels.keys())

        # ── 1. Layer 1: SLAM on depth-with-movable-masked-out ──────────
        movable_mask = collect_movable_masks(detections, depth.shape)
        masked_depth = mask_out_movable(depth, movable_mask)
        slam_raw = self.slam.step(rgb, masked_depth, odom_prior)

        # Ingest into particles (initialize on first call, otherwise refresh
        # T_wb per slot). Does NOT collapse to a single Gaussian.
        self.state.ingest_slam(slam_raw)

        # Collapsed summaries for legacy callers.
        slam_pose = self.state.collapsed_base()
        report["slam_raw"] = slam_raw
        report["slam_pose"] = slam_pose
        report["base_particles"] = ParticlePose(
            particles=np.stack(
                [p.T_wb for p in self.state.particles], axis=0),
            weights=self.state.normalized_weights(),
        )

        # ── 2. Fast tier: per-particle per-object EKF + weight update ──
        if self.bernoulli is not None:
            self._fast_tier_bernoulli(
                detections, gripper_state, T_ec, T_bg,
                depth, depth.shape[:2])
        else:
            self._fast_tier(detections, gripper_state, T_ec, T_bg)

        # ── 2b. Gravity-aware one-shot predict at release transition ───
        # Detect oids transitioning out of {holding, releasing}; for each,
        # replace the EKF mean + covariance with the post-fall prediction
        # from pose_update.manipulation.gravity_predict. No-ops when voxel_obs is None
        # or when bernoulli.gravity_predict is False.
        self._maybe_gravity_predict(gripper_state)

        # ── 3. Scene graph relations (on the collapsed view) ───────────
        # Only fire the relation detector on a periodic tick or on key
        # events (grasp / release / new track). Between firings we keep the
        # cached edges; the EMA is only updated when the detector actually
        # ran, so it doesn't decay during the quiet stretches.
        self._frame_rgb = rgb
        if self._should_recompute_relations(gripper_state):
            self._cached_relations = self._recompute_relations(detections)
            self._last_relation_frame = self.frame_count
        # Snapshot the post-step oid set so the next step can detect births.
        self._known_oids_before_step = {
            oid for oid, r in self.existence.items()
            if r >= (self.bernoulli.r_conf if self.bernoulli is not None else 0.0)
        } if self.bernoulli is not None else set(self.state.collapsed_objects().keys())

        # ── 4. Slow-tier trigger ───────────────────────────────────────
        should_trigger = self._should_trigger(gripper_state, detections)
        report["triggered"] = should_trigger

        if should_trigger:
            opt_result = self._slow_tier(
                slam_pose, detections, gripper_state, T_ec)
            report["alpha"] = opt_result.alpha
            report["residuals"] = opt_result.residuals
            self.last_opt_frame = self.frame_count
        else:
            report["alpha"] = None
            report["residuals"] = {}

        # ── 5. ESS-triggered resampling ────────────────────────────────
        resampled = self.state.resample_if_needed(
            threshold_frac=self.ess_resample_frac)
        report["resampled"] = resampled
        report["ess"] = self.state.ess()

        # Update bookkeeping
        self.last_state = dict(gripper_state)
        self._prev_T_bg = None if T_bg is None else T_bg.copy()
        self.frame_count += 1

        # Final collapsed view for the report
        report["objects"] = {
            oid: {"T": entry["T"].copy(),
                  "cov": entry["cov"].copy(),
                  "label": entry["label"]}
            for oid, entry in self.objects.items()
        }
        return report

    # --------------------------------------------------------------- #
    #  Fast tier: per-particle per-object EKF + weight accumulation
    # --------------------------------------------------------------- #

    def _fast_tier(self,
                   detections: List[Dict[str, Any]],
                   gripper_state: Dict[str, Any],
                   T_ec: Optional[np.ndarray],
                   T_bg: Optional[np.ndarray]) -> None:
        # Baseline mode: strip all proprioception and use uniform Q. This
        # is the vision-only TSDF++-style comparison.
        if self.baseline_mode:
            gripper_state = {"phase": "idle", "held_obj_id": None}
            T_ec = None
            T_bg = None
        phase = gripper_state.get("phase", "idle")
        held_id = gripper_state.get("held_obj_id")

        # Objects considered "moving with the robot" — the held one plus any
        # scene-graph passengers ("in"/"on" relations).
        manipulation_set = self._get_manipulation_set(held_id)

        # Manipulation-set members get a rigid-attachment predict when we
        # have two proprioception samples to form ΔT_gripper_b. Otherwise
        # we fall back to the legacy phase-aware inflated-Q predict for
        # them. Either way, each object is predicted exactly once.
        apply_rigid = self.should_apply_rigid(
            T_bg, self._prev_T_bg, manipulation_set, phase)
        delta_T_grip_b = (T_bg @ np.linalg.inv(self._prev_T_bg)
                          if apply_rigid else None)

        def Q_fn(oid: int, _particle) -> np.ndarray:
            if self.baseline_mode:
                return self._Q_BASELINE
            rigid_here = apply_rigid and (oid in manipulation_set)
            # Under rigid predict, Q_manip is added by rigid_attachment_predict
            # below — we skip the generic inflation here to avoid double-Q.
            if rigid_here:
                return np.zeros((6, 6))
            return process_noise_for_phase(
                phase=phase,
                is_target=(oid in manipulation_set),
                frames_since_observation=self.frames_since_obs.get(oid, 0),
                frame="world",
            )

        self.state.predict_objects(Q_fn)

        # Per-frame tick of frames_since_obs.
        for oid in self.frames_since_obs:
            self.frames_since_obs[oid] += 1

        if apply_rigid:
            for oid in manipulation_set:
                if any(oid in p.objects for p in self.state.particles):
                    self.state.rigid_attachment_predict(
                        oid, delta_T_grip_b, self._Q_MANIP_PER_STEP)

        # ── Measurement update for each observed object ────────────────
        for det in detections:
            oid = det.get("id")
            if oid is None:
                continue

            # Initialize on first sight (per-particle world-frame).
            if oid not in self.object_labels:
                T_co_meas = np.asarray(det["T_co"], dtype=np.float64)
                self.state.ensure_object(
                    oid, T_co_meas, self._INIT_OBJ_COV)
                self.object_labels[oid] = det.get("label", "unknown")
                self.object_first_seen[oid] = self.frame_count
                self.frames_since_obs[oid] = 0
                self.T_oe[oid] = None
                continue

            # EKF update (+ log-weight accumulation).
            R_icp = det.get("R_icp", np.eye(6) * 1e-3)
            T_co_meas = np.asarray(det["T_co"], dtype=np.float64)
            self.state.update_observation(
                oid=oid,
                T_co_meas=T_co_meas,
                R_icp=R_icp,
                iekf_iters=self.iekf_iters,
            )
            self.frames_since_obs[oid] = 0

            # Lock T_oe at grasp onset (first frame entering 'grasping').
            if (phase == "grasping"
                    and self.last_state.get("phase") != "grasping"
                    and oid == held_id
                    and T_ec is not None):
                collapsed = self.state.collapsed_object(oid)
                if collapsed is not None:
                    slam_pose = self.state.collapsed_base()
                    T_ew = slam_pose.T @ T_ec
                    self.T_oe[oid] = np.linalg.inv(T_ew) @ collapsed.T

    # --------------------------------------------------------------- #
    #  Fast tier (Bernoulli-EKF mode; bernoulli_ekf.tex)
    # --------------------------------------------------------------- #

    def _fast_tier_bernoulli(self,
                             detections: List[Dict[str, Any]],
                             gripper_state: Dict[str, Any],
                             T_ec: Optional[np.ndarray],
                             T_bg: Optional[np.ndarray],
                             depth: np.ndarray,
                             image_shape: tuple) -> None:
        """Bernoulli-EKF fast tier following docs/latex/bernoulli_ekf.tex.

        Implements the 7-step frame-by-frame algorithm (§11):
          1. Predict with Phi saturation + eq:bern_pred_r.
          2. Measure: ICP outputs already baked into det['T_co'], det['R_icp'].
          3. Associate: Hungarian on Mahalanobis cost (or oracle).
          4. Update per track: matched -> eq:r_assoc + EKF; missed -> eq:r_miss
             with p_v from the visibility predicate.
          5. Birth: unmatched measurements -> eq:birth_r + eq:birth_state.
          6. Prune: drop r < r_min.
          7. Emit: report via `self.objects` (filters by r_conf).
        """
        cfg = self.bernoulli
        if self.baseline_mode:
            gripper_state = {"phase": "idle", "held_obj_id": None}
            T_ec = None
            T_bg = None
        phase = gripper_state.get("phase", "idle")
        held_id = gripper_state.get("held_obj_id")

        manipulation_set = self._get_manipulation_set(held_id)

        apply_rigid = self.should_apply_rigid(
            T_bg, self._prev_T_bg, manipulation_set, phase)
        delta_T_grip_b = (T_bg @ np.linalg.inv(self._prev_T_bg)
                          if apply_rigid else None)

        def Q_fn(oid: int, _particle) -> np.ndarray:
            if self.baseline_mode:
                return self._Q_BASELINE
            rigid_here = apply_rigid and (oid in manipulation_set)
            if rigid_here:
                return np.zeros((6, 6))
            return process_noise_for_phase(
                phase=phase,
                is_target=(oid in manipulation_set),
                frames_since_observation=self.frames_since_obs.get(oid, 0),
                frame="world",
            )

        # ── 1. Predict state and existence (eq:ekf_*_pred + eq:bern_pred_r) ─
        self.state.predict_objects(Q_fn, P_max=cfg.P_max)

        for oid in self.frames_since_obs:
            self.frames_since_obs[oid] += 1

        if apply_rigid:
            for oid in manipulation_set:
                if any(oid in p.objects for p in self.state.particles):
                    self.state.rigid_attachment_predict(
                        oid, delta_T_grip_b, self._Q_MANIP_PER_STEP)

        for oid in list(self.existence.keys()):
            self.existence[oid] = r_predict(self.existence[oid], cfg.p_s)

        # ── 2. Associate + internal ICP ─────────────────────────────────
        # Branches by whether `cfg.K` is set:
        #   * K set        → step-4 "ICP inside the tracker" pipeline:
        #                     voxel dedup → coarse centroid → Hungarian on
        #                     3-DoF → fine ICP with track prior per match.
        #   * K None       → upstream-ICP contract: `det["T_co"]` /
        #                     `det["R_icp"]` are already populated; Hungarian
        #                     runs on 6-DoF innovations directly.
        track_oids = list(self.object_labels.keys())
        self._last_subpart_absorbed: List[Dict[str, Any]] = []
        self._last_det_to_oid = {}
        if cfg.K is not None and self.pose_est is not None:
            # 2a. Pre-association voxel dedup (single frame, label-agnostic).
            detections, absorbed = suppress_subpart_detections(
                list(detections), depth,
                np.asarray(cfg.K, dtype=np.float64),
                voxel_size=cfg.dedup_voxel_size_m,
                containment_thresh=cfg.dedup_containment_thresh,
                require_same_label=cfg.dedup_require_same_label,
            )
            self._last_subpart_absorbed = absorbed
            detections = self._make_coarse_dets(
                detections, depth, np.asarray(cfg.K, dtype=np.float64))
            assoc = self._hungarian_on_centroids(
                detections, track_oids, cfg)
            self._fine_icp_for_matches(
                assoc, detections, depth, cfg)
        elif cfg.association_mode == "oracle":
            assoc = oracle_associate(track_oids, detections)
        else:
            assoc = hungarian_associate(
                track_oids=track_oids,
                detections=detections,
                innovation_fn=self.state.innovation_stats,
                track_labels=self.object_labels,
                track_tau=self.sam2_tau,
                alpha=cfg.alpha,
                G_out=cfg.G_out,
                enforce_label_match=cfg.enforce_label_match,
                track_label_histories=self.label_scores,
                label_penalty=cfg.hungarian_label_penalty,
                score_weight=cfg.hungarian_score_weight,
                gate_mode=cfg.gate_mode,
                G_out_trans=cfg.G_out_trans,
                G_out_rot=cfg.G_out_rot,
                cost_d2_mode=cfg.cost_d2_mode,
                max_residual_m=cfg.max_residual_m,
            )

        # ── Visibility for missed tracks (eq:pv) ────────────────────────
        # Auto-disable when T_bc is missing: passing T_wb as T_cw projects
        # through the wrong axes (base +z vs camera optical axis) and
        # gives pathological p_v=0 for tracks that are actually visible.
        if cfg.enable_visibility and cfg.T_bc is not None:
            p_v_map = self._compute_visibility(detections, image_shape)
        else:
            p_v_map = {oid: 1.0 for oid in track_oids}

        # Track which detections have been consumed (matched or rejected
        # into birth). The set grows as we process the match loop so that
        # outer-gate rejects can also spawn new tracks.
        consumed_dets: Set[int] = set()

        # ── 3. Matched tracks: Huber + EKF + eq:r_assoc ─────────────────
        for oid, l in list(assoc.match.items()):
            det = detections[l]
            raw_T = det.get("T_co")
            # Centroid-only fallback: when the detection has no 6-DoF pose
            # but does carry a camera-frame centroid, run a 3-DoF
            # translation-only update path. See _fast_tier_bernoulli_match_3d.
            if raw_T is None:
                centroid = det.get("_centroid_cam")
                if centroid is None:
                    continue
                stats3 = self.state.centroid_innovation_stats(
                    oid, centroid, R_cam=np.diag([(0.02) ** 2] * 3))
                if stats3 is None:
                    continue
                nu3, S3, d2_t, log_lik = stats3
                d2 = d2_t
                if cfg.enable_huber:
                    w = huber_weight(d2_t, cfg.G_in_trans, cfg.G_out_trans)
                else:
                    w = 1.0
                if w <= 0.0:
                    assoc.unmatched_tracks.append(oid)
                    del assoc.match[oid]
                    continue
                self.state.update_observation_centroid(
                    oid=oid, centroid_cam=centroid,
                    R_cam=np.diag([(0.02) ** 2] * 3),
                    huber_w=w, P_max=cfg.P_max,
                )
                self.frames_since_obs[oid] = 0
                consumed_dets.add(l)
                self._last_det_to_oid[l] = oid
                r_prev = self.existence.get(oid, 1.0)
                r_new = r_assoc_update_loglik(
                    r_prev, log_L=log_lik,
                    p_d=cfg.p_d, lambda_c=cfg.lambda_c)
                self.existence[oid] = r_new
                # SAM2 id + label history bookkeeping (same as 6-DoF path).
                d_tau = det.get("sam2_id", det.get("id"))
                if d_tau is not None:
                    try:
                        self.sam2_tau[oid] = int(d_tau)
                    except (TypeError, ValueError):
                        pass
                self._merge_label_scores(oid, det)
                continue

            T_co = np.asarray(raw_T, dtype=np.float64)
            R_icp = np.asarray(det.get("R_icp"), dtype=np.float64)

            stats = self.state.innovation_stats(oid, T_co, R_icp)
            if stats is None:
                # Track was pruned between association and update; treat
                # this detection as a birth candidate.
                continue
            nu, S, d2, log_lik = stats

            # Match Hungarian's gate_mode for the Huber check so that
            # an unreliable rotation block doesn't unwind a
            # translation-feasible match (see bernoulli_ekf.tex §6 d^2
            # block decomposition).
            if cfg.gate_mode in ("trans", "trans_and_rot"):
                try:
                    d2_huber = float(nu[:3] @ np.linalg.solve(S[:3, :3], nu[:3]))
                except np.linalg.LinAlgError:
                    d2_huber = d2
                if cfg.gate_mode == "trans_and_rot":
                    try:
                        d2_r = float(nu[3:] @ np.linalg.solve(S[3:, 3:], nu[3:]))
                        d2_huber = max(d2_huber, d2_r)
                    except np.linalg.LinAlgError:
                        pass
                G_in_h, G_out_h = cfg.G_in_trans, cfg.G_out_trans
            else:
                d2_huber = d2
                G_in_h, G_out_h = cfg.G_in, cfg.G_out

            if cfg.enable_huber:
                w = huber_weight(d2_huber, G_in_h, G_out_h)
            else:
                w = 1.0

            if w <= 0.0:
                # Outer-gate reject: route to miss branch for this track
                # and let the detection go to birth.
                assoc.unmatched_tracks.append(oid)
                del assoc.match[oid]
                continue

            self.state.update_observation(
                oid=oid,
                T_co_meas=T_co,
                R_icp=R_icp,
                iekf_iters=self.iekf_iters,
                huber_w=w,
                P_max=cfg.P_max,
            )
            self.frames_since_obs[oid] = 0
            consumed_dets.add(l)
            self._last_det_to_oid[l] = oid

            # Existence update (eq:r_assoc) in log-space so a very low
            # likelihood does not underflow.
            r_prev = self.existence.get(oid, 1.0)
            r_new = r_assoc_update_loglik(
                r_prev, log_L=log_lik,
                p_d=cfg.p_d, lambda_c=cfg.lambda_c)
            self.existence[oid] = r_new

            # SAM2-ID bookkeeping: update only on a successful match.
            # Fall back to `id` if the detector client hasn't populated
            # `sam2_id` explicitly (paper §6.1: the upstream tracklet
            # identifier is the same quantity either way).
            d_tau = det.get("sam2_id", det.get("id"))
            if d_tau is not None:
                try:
                    self.sam2_tau[oid] = int(d_tau)
                except (TypeError, ValueError):
                    pass

            # Soft label history maintenance (mirrors perception's
            # detection_h `labels` field). Merge the detection's full
            # `labels` dict when present, else just bump det['label']
            # with det['score'].
            self._merge_label_scores(oid, det)

            # Lock T_oe at grasp onset (mirrors legacy path).
            if (phase == "grasping"
                    and self.last_state.get("phase") != "grasping"
                    and oid == held_id
                    and T_ec is not None):
                collapsed = self.state.collapsed_object(oid)
                if collapsed is not None:
                    slam_pose = self.state.collapsed_base()
                    T_ew = slam_pose.T @ T_ec
                    self.T_oe[oid] = np.linalg.inv(T_ew) @ collapsed.T

        # ── 4. Missed tracks: eq:r_miss ─────────────────────────────────
        for oid in assoc.unmatched_tracks:
            if oid not in self.existence:
                continue
            p_v = p_v_map.get(oid, 1.0)
            p_d_tilde = cfg.p_d * p_v
            r_prev = self.existence[oid]
            self.existence[oid] = r_miss_update(r_prev, p_d_tilde)

        # ── 5. Birth (eq:birth_r + eq:birth_state) ──────────────────────
        # Policy first, measurement second. The pending-buffer table
        # accumulates per-perception-id counters across frames; ICP runs
        # only on admission, exactly once, so `pose_est._refs[oid]` is in
        # 1:1 correspondence with committed tracks.
        self._last_birth_rejects: List[Dict[str, Any]] = []
        internal_icp = cfg.K is not None and self.pose_est is not None
        seen_pids: Set[Any] = set()
        for l in range(len(detections)):
            if l in consumed_dets:
                continue
            det = detections[l]
            if internal_icp and not det.get("_centroid_ok"):
                continue
            pid = det.get("id")
            # A distinct entry per perception id. Unknown or collision-prone
            # sentinels (None / -1) get a per-call unique key so they still
            # track independently but never share a counter.
            if pid is None:
                pid_key: Any = ("none", l)
            else:
                pid_key = pid
            seen_pids.add(pid_key)
            pending = self._pending_births.get(pid_key)
            if pending is None:
                pending = _PendingBirth.from_det(det, self.frame_count)
                self._pending_births[pid_key] = pending
            pending.bump(det, self.frame_count)

            admit, reason = self._birth_admissible(
                det, cfg, image_shape,
                tracker_n_obs=pending.n_obs_tracker,
                tracker_max_score=pending.max_score,
                require_pose=(not internal_icp),
            )
            if not admit:
                self._last_birth_rejects.append({
                    "global_idx": det.get("global_idx"),
                    "label": det.get("label"),
                    "score": det.get("score"),
                    "reason": reason,
                    "n_obs_tracker": pending.n_obs_tracker,
                })
                continue

            # Proximity gate against existing same-label live tracks.
            # Catches SAM2-id-reseed duplicates whose underlying physical
            # object is already tracked under another oid. Uses world-
            # frame centroid so head pan / robot motion don't bias it.
            if self._candidate_near_live_track(det, cfg):
                self._last_birth_rejects.append({
                    "global_idx": det.get("global_idx"),
                    "label": det.get("label"),
                    "score": det.get("score"),
                    "reason": "near_live",
                    "n_obs_tracker": pending.n_obs_tracker,
                })
                # Drop the pending entry — its perception id is associated
                # with an existing track that just happens to lack a hard
                # SAM2-id link. Holding the entry warm would re-trigger
                # this check every frame for the rest of the run.
                self._pending_births.pop(pid_key, None)
                continue

            # ── Admitted. Mint a fresh oid and run ICP ONCE to seed the
            # track's belief. ICP is the measurement here, not a gate.
            if internal_icp:
                mask = det.get("mask")
                if mask is None:
                    continue
                new_oid = self._next_track_id()
                T_co, R_icp, fitness, rmse = self.pose_est.estimate(
                    oid=int(new_oid), mask=mask, depth=depth,
                    T_co_init=None,
                )
                if T_co is None:
                    # Deterministic first-observation ICP failed (should
                    # not happen on a healthy mask since _centroid_ok was
                    # already True, but defend against it).
                    self.pose_est._refs.pop(int(new_oid), None)
                    continue
                det["T_co"] = T_co
                det["R_icp"] = R_icp
                det["fitness"] = float(fitness)
                det["rmse"] = float(rmse)
                det["_icp_ok"] = True
                det["_measurement_kind"] = "icp_anchor"
                born_oid = self._birth_track(det, cfg, forced_oid=int(new_oid))
            else:
                born_oid = self._birth_track(det, cfg)

            if born_oid is not None:
                self._last_det_to_oid[l] = int(born_oid)
                self._pending_births.pop(pid_key, None)

        # TTL-expire pending entries that haven't been seen for a while —
        # a perception id that disappears must not hold its counter forever.
        self._expire_pending_births(cfg)

        # ── 6. Prune (r < r_min) ────────────────────────────────────────
        if cfg.r_min > 0.0:
            to_prune = [oid for oid, r in self.existence.items()
                        if r < cfg.r_min]
            for oid in to_prune:
                self._prune_track(oid)

        # ── 7. Self-merge same-label tracks closer than the Euclidean gate.
        # Catches duplicates that Hungarian's one-to-one constraint couldn't
        # absorb within a single frame (SAM2 mask splits, re-prompts, etc.).
        self._self_merge_pass(held_id)

    def _self_merge_pass(self,
                          held_id: Optional[int],
                          protected_pairs: Optional[Set[Tuple[int, int]]] = None,
                          ) -> List[Dict[str, Any]]:
        """Fuse same-label tracks whose world-frame means are within
        `self_merge_trans_m` metres of each other.

        Gate is Euclidean (meters), not Mahalanobis — the merge radius is
        invariant to how tight/loose the two tracks' covariances are, so
        two fresh births at 22 cm cannot collapse just because σ starts
        at 5 cm.

        Greedy by ascending distance. When two tracks merge:
          - belief state fused via per-particle Bayesian information sum
            (`RBPFState.merge_tracks`);
          - orchestrator bookkeeping (existence, frames_since_obs,
            sam2_tau, label_scores, T_oe, object_labels) is unioned into
            the kept track;
          - the absorbed track's oid is removed.

        Tiebreak for keep-vs-drop: higher r wins; tie → lower oid wins;
        the held object is never dropped (merges into it).

        Args:
            held_id: current upstream-reported held object id (never dropped).
            protected_pairs: unordered ``(min_oid, max_oid)`` tuples that
                the relation graph asserts are distinct physical objects
                (e.g. an apple resting on a tray). Such pairs are
                skipped — they should never collapse regardless of how
                close their centroids drift.

        Returns:
            List of {keep_oid, drop_oid, dist_m, d2_trans} records for
            diagnostics.
        """
        cfg = self.bernoulli
        gate_m = float(getattr(cfg, "self_merge_trans_m", 0.0))
        if gate_m <= 0.0:
            return []
        oids = list(self.object_labels.keys())
        if len(oids) < 2:
            return []

        # Gather world-frame means (collapsed across particles).
        beliefs = {oid: self.state.collapsed_object(oid) for oid in oids}

        candidates: List = []
        for i in range(len(oids)):
            oi = oids[i]
            pe_i = beliefs[oi]
            if pe_i is None:
                continue
            label_i = self.object_labels[oi]
            for j in range(i + 1, len(oids)):
                oj = oids[j]
                if self.object_labels[oj] != label_i:
                    continue
                # Scene-graph protection: relation-asserted pairs must
                # not collapse, even if their world means drift to
                # within `self_merge_trans_m`.
                if (protected_pairs is not None
                        and (min(oi, oj), max(oi, oj)) in protected_pairs):
                    continue
                pe_j = beliefs[oj]
                if pe_j is None:
                    continue
                nu_t = (np.asarray(pe_j.T)[:3, 3]
                         - np.asarray(pe_i.T)[:3, 3])
                d = float(np.linalg.norm(nu_t))
                if d > gate_m:
                    continue
                # Diagnostic d²_trans: nu^T (P_i + P_j)^-1 nu.
                S_tt = (np.asarray(pe_i.cov)[:3, :3]
                         + np.asarray(pe_j.cov)[:3, :3])
                try:
                    d2_t = float(nu_t @ np.linalg.solve(S_tt, nu_t))
                except np.linalg.LinAlgError:
                    d2_t = float("nan")
                candidates.append((d, d2_t, oi, oj))
        candidates.sort(key=lambda t: t[0])

        merges: List[Dict[str, Any]] = []
        absorbed: Set[int] = set()
        for d, d2_t, oi, oj in candidates:
            if oi in absorbed or oj in absorbed:
                continue
            # Keep/drop decision.
            if oi == held_id:
                keep, drop = oi, oj
            elif oj == held_id:
                keep, drop = oj, oi
            else:
                r_i = self.existence.get(oi, 0.0)
                r_j = self.existence.get(oj, 0.0)
                if r_i > r_j:
                    keep, drop = oi, oj
                elif r_j > r_i:
                    keep, drop = oj, oi
                else:
                    keep, drop = (oi, oj) if oi < oj else (oj, oi)

            if not self.state.merge_tracks(keep, drop):
                continue

            # Orchestrator-level bookkeeping merge.
            self.existence[keep] = max(self.existence.get(keep, 0.0),
                                         self.existence.get(drop, 0.0))
            self.frames_since_obs[keep] = min(
                self.frames_since_obs.get(keep, 0),
                self.frames_since_obs.get(drop, 0))
            # SAM2 id: prefer the kept track's; fall back to drop's if kept
            # was unset (-1).
            tau_keep = self.sam2_tau.get(keep, -1)
            if tau_keep == -1:
                tau_drop = self.sam2_tau.get(drop, -1)
                if tau_drop != -1:
                    self.sam2_tau[keep] = tau_drop
            # Label-history merge: sum n_obs and weighted-average mean_score
            # per label.
            for lbl, st in self.label_scores.get(drop, {}).items():
                cur = self.label_scores.setdefault(keep, {}).setdefault(
                    lbl, {"n_obs": 0, "mean_score": 0.0})
                n_old = int(cur["n_obs"])
                m_old = float(cur["mean_score"])
                n_new = int(st.get("n_obs", 0))
                m_new = float(st.get("mean_score", 0.0))
                n_tot = n_old + n_new
                if n_tot > 0:
                    cur["n_obs"] = n_tot
                    cur["mean_score"] = (
                        m_old * n_old + m_new * n_new) / n_tot
            # T_oe: keep the held side's T_oe; else prefer any non-None.
            if self.T_oe.get(keep) is None:
                self.T_oe[keep] = self.T_oe.get(drop)
            # ICP reference cloud merge (step-4 internal ICP): voxel-
            # concatenate the absorbed track's surface samples into the
            # keeper's so subsequent ICP benefits from the richer cloud.
            if self.pose_est is not None:
                self._merge_icp_refs(keep, drop)

            # Drop the absorbed track everywhere.
            self.object_labels.pop(drop, None)
            self.object_first_seen.pop(drop, None)
            self.frames_since_obs.pop(drop, None)
            self.existence.pop(drop, None)
            self.sam2_tau.pop(drop, None)
            self.label_scores.pop(drop, None)
            self.T_oe.pop(drop, None)
            absorbed.add(drop)

            merges.append({
                "keep_oid": int(keep),
                "drop_oid": int(drop),
                "dist_m": float(d),
                "d2_trans": float(d2_t),
            })
        return merges

    # ------------------------------------------------------------------ #
    #  Step-4 helpers — internal ICP pipeline (only when cfg.K is set)
    # ------------------------------------------------------------------ #

    def _make_coarse_dets(self,
                           detections: List[Dict[str, Any]],
                           depth: np.ndarray,
                           K: np.ndarray) -> List[Dict[str, Any]]:
        """Compute the camera-frame centroid for every detection via
        back-projection of its mask + depth. Attaches
        `det["_centroid_cam"]` and `det["_centroid_ok"]`, and zeroes out
        `T_co / R_icp / fitness / rmse / _icp_ok` placeholders so the
        subsequent fine-ICP pass is the single writer of those fields.

        Detections whose mask has no valid-depth pixel keep
        `_centroid_ok=False` — Hungarian excludes them from the cost
        matrix, and they cannot birth (no anchor) either.

        Respects a legacy-compat short-circuit: if a detection already
        carries a pre-computed `T_co`, we still need a centroid for the
        Hungarian step, so we use `T_co[:3,3]` as the centroid. Fine ICP
        will overwrite T_co regardless.
        """
        out: List[Dict[str, Any]] = []
        for det in detections:
            d = dict(det)
            centroid = None
            mask = det.get("mask")
            if mask is not None:
                try:
                    centroid = centroid_cam_from_mask(mask, depth, K)
                except Exception:
                    centroid = None
            if centroid is None and det.get("T_co") is not None:
                centroid = np.asarray(det["T_co"], dtype=np.float64)[:3, 3]
            d["_centroid_cam"] = (None if centroid is None
                                    else np.asarray(centroid, dtype=np.float64).copy())
            d["_centroid_ok"] = centroid is not None
            # Reset ICP-result placeholders so fine ICP is the writer.
            d["T_co"] = None
            d["R_icp"] = None
            d["fitness"] = 0.0
            d["rmse"] = 0.0
            d["_icp_ok"] = False
            d["_measurement_kind"] = "dropped"
            out.append(d)
        return out

    def _hungarian_on_centroids(self,
                                 dets_coarse: List[Dict[str, Any]],
                                 track_oids: List[int],
                                 cfg: "BernoulliConfig",
                                 ) -> AssociationResult:
        """Run association on the 3-DoF centroid Mahalanobis.

        Wraps `state.centroid_innovation_stats` into the 6D-shaped
        `(nu, S, d2, log_lik)` expected by `hungarian_associate`. The
        rotation block of `S` is padded to `1e6·I` so the Hungarian's
        `trans` / `trans_and_rot` block decomposition sees a purely
        translational d², identical to the instrumented tracker's
        `_centroid_innov` wrapper.

        Detections that failed to produce a centroid are excluded from
        association (neither matched nor birthed — no anchor). The
        returned `AssociationResult` carries the FULL detection list's
        indices so downstream code indexes `dets_coarse` directly.
        """
        _R_CENT_CAM = np.diag([(0.02) ** 2] * 3)
        _ROT_PAD = np.eye(3, dtype=np.float64) * 1e6

        def _centroid_innov(oid: int,
                             T_co: np.ndarray,
                             R_icp: np.ndarray):
            centroid_cam = np.asarray(T_co, dtype=np.float64)[:3, 3]
            stats3 = self.state.centroid_innovation_stats(
                oid, centroid_cam, R_cam=_R_CENT_CAM)
            if stats3 is None:
                return None
            nu3, S3, d2_3, logL3 = stats3
            nu6 = np.zeros(6, dtype=np.float64)
            nu6[:3] = nu3
            S6 = np.zeros((6, 6), dtype=np.float64)
            S6[:3, :3] = S3
            S6[3:, 3:] = _ROT_PAD
            return nu6, S6, d2_3, logL3

        # Build the reduced list of association-eligible detections.
        dets_for_assoc: List[Dict[str, Any]] = []
        local_to_global: Dict[int, int] = {}
        for gi, d in enumerate(dets_coarse):
            if not d.get("_centroid_ok"):
                continue
            fake_T_co = np.eye(4, dtype=np.float64)
            fake_T_co[:3, 3] = np.asarray(d["_centroid_cam"], dtype=np.float64)
            dets_for_assoc.append({**d,
                                    "T_co": fake_T_co,
                                    "R_icp": np.zeros((6, 6))})
            local_to_global[len(dets_for_assoc) - 1] = gi

        if cfg.association_mode == "oracle":
            assoc_local = oracle_associate(track_oids, dets_for_assoc)
        else:
            assoc_local = hungarian_associate(
                track_oids=track_oids,
                detections=dets_for_assoc,
                innovation_fn=_centroid_innov,
                track_labels=self.object_labels,
                track_tau=self.sam2_tau,
                alpha=cfg.alpha,
                G_out=cfg.G_out,
                enforce_label_match=cfg.enforce_label_match,
                track_label_histories=self.label_scores,
                label_penalty=cfg.hungarian_label_penalty,
                score_weight=cfg.hungarian_score_weight,
                gate_mode=cfg.gate_mode,
                G_out_trans=cfg.G_out_trans,
                G_out_rot=cfg.G_out_rot,
                cost_d2_mode=cfg.cost_d2_mode,
                max_residual_m=cfg.max_residual_m,
            )

        # Remap local detection indices back to the full dets_coarse space.
        match_global = {int(oid): int(local_to_global[l])
                        for oid, l in assoc_local.match.items()}
        unmatched_dets_global = [int(local_to_global[l])
                                  for l in assoc_local.unmatched_detections]
        # Dets excluded by the centroid filter are neither matched nor
        # unmatched here — they cannot birth either (see step 5 below).
        return AssociationResult(
            match=match_global,
            unmatched_tracks=list(assoc_local.unmatched_tracks),
            unmatched_detections=unmatched_dets_global,
            cost_matrix=assoc_local.cost_matrix,
            gated_pairs=assoc_local.gated_pairs,
            d2_full_matrix=assoc_local.d2_full_matrix,
            d2_trans_matrix=assoc_local.d2_trans_matrix,
            d2_rot_matrix=assoc_local.d2_rot_matrix,
        )

    def _fine_icp_for_matches(self,
                               assoc: AssociationResult,
                               dets_coarse: List[Dict[str, Any]],
                               depth: np.ndarray,
                               cfg: "BernoulliConfig",
                               ) -> None:
        """For every matched (track, det) pair, run ICP seeded with the
        track's predicted camera-frame pose `T_co^pred = inv(T_bc) · μ_bo`.
        Writes the fit back into `det["T_co"]`, `det["R_icp"]`, etc.

        Triage by ICP quality:
          * fitness ≥ min, rmse ≤ max  → 6-DoF measurement (T_co, R_icp).
          * fitness fails              → centroid-only fallback: T_co built
              from (predicted rotation, measured centroid) plus R_icp with
              inflated rotation variance so the EKF update is effectively
              translation-only.
          * centroid invalid           → detection is demoted (no
              measurement, track flows to miss).
        """
        if self.pose_est is None or cfg.T_bc is None:
            return

        # Cache camera-frame prior per live track.
        T_bc = np.asarray(cfg.T_bc, dtype=np.float64)
        T_cb = np.linalg.inv(T_bc)
        T_co_pred: Dict[int, np.ndarray] = {}
        for o in self.object_labels:
            pe = self.state.collapsed_object_base(o)
            if pe is None:
                continue
            T_co_pred[int(o)] = T_cb @ pe.T

        rot_var = float(cfg.icp_centroid_fallback_rot_var)
        trans_std = float(cfg.icp_centroid_fallback_trans_std)

        for oid, l in list(assoc.match.items()):
            det = dets_coarse[l]
            mask = det.get("mask")
            if mask is None:
                continue
            T_init = T_co_pred.get(int(oid))
            T_co, R_icp, fitness, rmse = self.pose_est.estimate(
                oid=int(oid), mask=mask, depth=depth,
                T_co_init=T_init,
            )
            det["fitness"] = float(fitness)
            det["rmse"] = float(rmse)
            det["_icp_prior_oid"] = int(oid)
            det["_icp_prior_used"] = T_init is not None
            if T_co is not None:
                det["T_co"] = T_co
                det["R_icp"] = R_icp
                det["_icp_ok"] = True
                det["_measurement_kind"] = "icp"
                continue

            # Centroid fallback: Hungarian already certified the pair via
            # the 3-DoF centroid Mahalanobis; don't drop the match just
            # because the surface alignment was poor.
            centroid = det.get("_centroid_cam")
            if centroid is None:
                det["T_co"] = None
                det["R_icp"] = None
                det["_icp_ok"] = False
                det["_measurement_kind"] = "dropped"
                continue
            T_fb = np.eye(4, dtype=np.float64)
            if T_init is not None:
                T_fb[:3, :3] = T_init[:3, :3]
            T_fb[:3, 3] = np.asarray(centroid, dtype=np.float64)
            R_fb = np.diag([trans_std ** 2] * 3 + [rot_var] * 3)
            det["T_co"] = T_fb
            det["R_icp"] = R_fb
            det["_icp_ok"] = True
            det["_measurement_kind"] = "centroid_fallback"

    def _compute_visibility(self,
                             detections: List[Dict[str, Any]],
                             image_shape: tuple) -> Dict[int, float]:
        """Collect per-track bboxes + mean depth from the CURRENT frame's
        matched detections (matched pairs only), then run eq:pv.

        Tracks without a matching detection this frame fall back to a
        bbox-less record -- so their p_v is driven solely by the frustum
        projection gate. This is a reasonable first cut; a more faithful
        implementation would project each track's TSDF to image space.
        """
        cfg = self.bernoulli
        if cfg.K is None:
            return {oid: 1.0 for oid in self.object_labels}

        # Base pose -> camera pose.
        slam_pose = self.state.collapsed_base()
        T_wb = slam_pose.T

        tracks_for_vis: List[Dict[str, Any]] = []
        for oid in self.object_labels:
            pe = self.state.collapsed_object(oid)
            if pe is None:
                continue
            bbox_im = None
            mean_depth = None
            # Look up the most recent detection that matched this oid's
            # label as a proxy for the image-space bbox.
            for det in detections:
                if det.get("label") == self.object_labels.get(oid):
                    if det.get("box") is not None:
                        bx = det["box"]
                        try:
                            bbox_im = tuple(float(v) for v in bx)
                        except (TypeError, ValueError):
                            bbox_im = None
                    if det.get("T_co") is not None:
                        T_co = np.asarray(det["T_co"], dtype=np.float64)
                        mean_depth = float(T_co[2, 3])
                    break
            tracks_for_vis.append({
                "oid": int(oid),
                "T": pe.T,
                "bbox_image": bbox_im,
                "mean_depth_camera": mean_depth,
            })

        # Compose T_cw = T_wb * T_bc when T_bc is supplied (Fetch head camera
        # has a fixed offset from base; for test harnesses where camera ==
        # base, caller sets T_bc = I). When T_bc is None the outer guard
        # above would have short-circuited to p_v = 1.0 already, but we
        # preserve the old behaviour (T_cw = T_wb) as a fallback here in
        # case this private helper is called directly.
        T_cw = T_wb @ cfg.T_bc if cfg.T_bc is not None else T_wb
        return visibility_p_v(tracks_for_vis, cfg.K, T_cw,
                               cfg.image_shape or image_shape)

    def _birth_admissible(self,
                           det: Dict[str, Any],
                           cfg: BernoulliConfig,
                           image_shape: tuple,
                           *,
                           tracker_n_obs: Optional[int] = None,
                           tracker_max_score: Optional[float] = None,
                           require_pose: bool = True,
                           ) -> tuple:
        """Thin instance-method wrapper around `birth_admissible` for the
        orchestrator's own birth loop; see the module-level helper for
        semantics."""
        return birth_admissible(
            det, cfg, image_shape,
            tracker_n_obs=tracker_n_obs,
            tracker_max_score=tracker_max_score,
            require_pose=require_pose,
        )

    def _birth_track(self, det: Dict[str, Any],
                     cfg: BernoulliConfig,
                     forced_oid: Optional[int] = None) -> Optional[int]:
        """Initialise a new Bernoulli track from an unmatched detection
        (eq:birth_r + eq:birth_state).

        In Hungarian mode, the internal track id is minted fresh by the
        tracker -- upstream ``det['id']`` is NOT reused, because an
        unmatched detection by definition does not correspond to any
        currently-known track. Reusing a colliding ``det['id']`` would
        silently overwrite an existing track's (pose, r, label) state.

        In oracle mode, identity is taken from upstream directly.

        `forced_oid` (step-4 internal ICP path): the caller already minted
        an oid and ran anchor ICP against `pose_est._refs[forced_oid]`,
        so we must use that same oid here.

        Returns the minted/forced oid on success, or None if birth failed
        (e.g. missing T_co).
        """
        T_co = det.get("T_co")
        if T_co is None:
            return None
        T_co = np.asarray(T_co, dtype=np.float64)
        R_icp = det.get("R_icp")
        if R_icp is None:
            R_icp = np.eye(6) * 1e-3
        R_icp = np.asarray(R_icp, dtype=np.float64)

        # Mint the internal track id. Oracle mode respects upstream; Hungarian
        # mode always assigns a fresh id to avoid colliding with existing
        # tracks that happen to share det['id'].
        if forced_oid is not None:
            d_id = int(forced_oid)
        elif cfg.association_mode == "oracle":
            raw_id = det.get("id")
            if raw_id is None:
                d_id = self._next_track_id()
            else:
                d_id = int(raw_id)
                if d_id in self.object_labels:
                    # Even in oracle mode, a colliding upstream id on an
                    # otherwise unmatched detection is a bug; mint a fresh
                    # one to keep the track state consistent.
                    d_id = self._next_track_id()
        else:
            d_id = self._next_track_id()

        if cfg.init_cov_from_R:
            init_cov = 0.5 * (R_icp + R_icp.T)
        else:
            init_cov = self._INIT_OBJ_COV.copy()

        self.state.ensure_object(d_id, T_co, init_cov)
        self.object_labels[d_id] = det.get("label", "unknown")
        self.object_first_seen[d_id] = self.frame_count
        self.frames_since_obs[d_id] = 0
        self.T_oe.setdefault(d_id, None)

        score = float(det.get("score", 1.0))
        self.existence[d_id] = r_birth(
            score, lambda_b=cfg.lambda_b, lambda_c=cfg.lambda_c)

        d_tau = det.get("sam2_id", det.get("id"))
        if d_tau is not None:
            try:
                self.sam2_tau[d_id] = int(d_tau)
            except (TypeError, ValueError):
                self.sam2_tau[d_id] = -1
        else:
            self.sam2_tau[d_id] = -1

        # Seed the soft label history from the detection itself.
        self.label_scores[d_id] = {}
        self._merge_label_scores(d_id, det)
        return d_id

    def _merge_label_scores(self,
                             oid: int,
                             det: Dict[str, Any]) -> None:
        """Merge a detection's label distribution into the track's soft
        label history.

        Accepts both schemas:
          (a) det['labels'] = {label: {n_obs, mean_score}}  (perception
              detection_h JSON: this is the rich form)
          (b) det['label'] = str + det['score'] = float       (lightweight)

        For (a) we mass-merge per-label `n_obs` and average `mean_score`
        weighted by sample count. For (b) we add a single observation of
        det['label'] with weight det['score'].
        """
        store = self.label_scores.setdefault(oid, {})

        labels_dict = det.get("labels")
        if isinstance(labels_dict, dict) and labels_dict:
            for lbl, stats in labels_dict.items():
                if not isinstance(stats, dict):
                    continue
                n_new = int(stats.get("n_obs", 0))
                m_new = float(stats.get("mean_score", 0.0))
                if n_new <= 0:
                    continue
                cur = store.setdefault(
                    lbl, {"n_obs": 0, "mean_score": 0.0})
                n_old = int(cur["n_obs"])
                m_old = float(cur["mean_score"])
                n_total = n_old + n_new
                cur["n_obs"] = n_total
                cur["mean_score"] = ((m_old * n_old + m_new * n_new)
                                       / max(1, n_total))
            return

        # Fallback: just the primary label + score.
        lbl = det.get("label")
        if lbl is None:
            return
        score = float(det.get("score", 1.0))
        cur = store.setdefault(lbl, {"n_obs": 0, "mean_score": 0.0})
        n_old = int(cur["n_obs"])
        m_old = float(cur["mean_score"])
        cur["n_obs"] = n_old + 1
        cur["mean_score"] = (m_old * n_old + score) / (n_old + 1)

    def _prune_track(self, oid: int) -> None:
        """Remove `oid` from every particle and every orchestrator dict."""
        self.state.delete_object(oid)
        self.object_labels.pop(oid, None)
        self.object_first_seen.pop(oid, None)
        self.frames_since_obs.pop(oid, None)
        self.T_oe.pop(oid, None)
        self.existence.pop(oid, None)
        self.sam2_tau.pop(oid, None)
        self.label_scores.pop(oid, None)
        if self.pose_est is not None:
            self.pose_est._refs.pop(oid, None)

    def _merge_icp_refs(self, keep: int, drop: int) -> None:
        """Fold the absorbed track's ICP reference cloud into the keeper's.

        Points are in each track's object-local frame; under the self-merge
        condition the two frames have collapsed (d²_t ≈ 0), so concatenation
        is a reasonable approximation. We voxel-downsample at the pose
        estimator's VOXEL_SIZE and cap at MAX_REF_POINTS.
        """
        if self.pose_est is None:
            return
        ref_drop = self.pose_est._refs.pop(drop, None)
        ref_keep = self.pose_est._refs.get(keep)
        if ref_drop is None or ref_keep is None:
            return
        if getattr(ref_drop, "ref_points", None) is None:
            return
        if getattr(ref_keep, "ref_points", None) is None:
            ref_keep.ref_points = ref_drop.ref_points
            return
        try:
            import open3d as _o3d  # type: ignore
            merged = np.vstack([ref_keep.ref_points, ref_drop.ref_points])
            pc = _o3d.geometry.PointCloud()
            pc.points = _o3d.utility.Vector3dVector(merged)
            pc = pc.voxel_down_sample(self.pose_est.VOXEL_SIZE)
            merged = np.asarray(pc.points, dtype=np.float64)
            if len(merged) > self.pose_est.MAX_REF_POINTS:
                stride = np.linspace(0, len(merged) - 1,
                                      self.pose_est.MAX_REF_POINTS
                                      ).astype(np.int64)
                merged = merged[stride]
            ref_keep.ref_points = merged
            if hasattr(ref_keep, "obj_radius"):
                ref_keep.obj_radius = max(float(np.linalg.norm(
                    merged, axis=1).max()), 0.03)
        except Exception:
            # Best-effort fold; worst case the keeper keeps its own cloud.
            pass

    def _next_track_id(self) -> int:
        """Next unused track id (for Hungarian-mode births without id)."""
        if not self.object_labels:
            return 1
        return max(self.object_labels.keys()) + 1

    def _candidate_near_live_track(self,
                                    det: Dict[str, Any],
                                    cfg: "BernoulliConfig") -> bool:
        """True if `det`'s world-frame centroid is within
        ``cfg.birth_min_dist_m`` of any same-label live track.

        World-frame centroid:
            c^w = T_wb · T_bc · [c^cam, 1]

        Compared per-axis Euclidean against the collapsed mean of every
        live track sharing the same label. If T_wb / T_bc / centroid is
        unavailable, returns False (don't block birth on missing data).
        """
        gate = float(getattr(cfg, "birth_min_dist_m", 0.0))
        if gate <= 0.0 or not self.object_labels:
            return False
        c_cam = det.get("_centroid_cam")
        if c_cam is None:
            return False
        T_wb = getattr(self.state, "T_wb", None)
        T_bc = getattr(self.state, "T_bc", None)
        if T_wb is None or T_bc is None:
            return False
        c_h = np.array([float(c_cam[0]), float(c_cam[1]),
                         float(c_cam[2]), 1.0], dtype=np.float64)
        c_world = (T_wb @ T_bc @ c_h)[:3]
        cand_label = det.get("label")
        for oid, lbl in self.object_labels.items():
            if cand_label is not None and lbl != cand_label:
                continue
            pe = self.state.collapsed_object(oid)
            if pe is None:
                continue
            mu_w = np.asarray(pe.T, dtype=np.float64)[:3, 3]
            if float(np.linalg.norm(c_world - mu_w)) <= gate:
                return True
        return False

    def _expire_pending_births(self, cfg: "BernoulliConfig") -> None:
        """Drop pending-birth entries whose last_seen_frame is too old.

        Keeps a brief perception dropout (≤ `birth_pending_ttl_frames`)
        from resetting the tracker-side confirmation counter while still
        bounding the buffer size.
        """
        ttl = int(getattr(cfg, "birth_pending_ttl_frames", 30))
        if ttl <= 0 or not self._pending_births:
            return
        cutoff = self.frame_count - ttl
        stale = [pid for pid, p in self._pending_births.items()
                 if p.last_seen_frame < cutoff]
        for pid in stale:
            self._pending_births.pop(pid, None)

    # --------------------------------------------------------------- #
    #  Slow tier: joint pose graph (Option A — operates on collapsed)
    # --------------------------------------------------------------- #

    def _slow_tier(self,
                   slam_pose: PoseEstimate,
                   detections: List[Dict[str, Any]],
                   gripper_state: Dict[str, Any],
                   T_ec: Optional[np.ndarray]) -> OptimizationResult:
        collapsed = self.state.collapsed_objects()
        priors: Dict[int, PoseEstimate] = dict(collapsed)

        observations: List[Observation] = []
        for det in detections:
            oid = det.get("id")
            if oid is None or oid not in priors:
                continue
            T_co = det.get("T_co")
            if T_co is None:
                # New-contract caller: no upstream ICP. Slow tier has no
                # observation to feed the pose graph this step; will run
                # again next trigger using the orchestrator's internal ICP
                # output (stashed on the fast-tier pass).
                continue
            observations.append(Observation(
                obj_id=oid,
                T_co=T_co,
                R_icp=det.get("R_icp", np.eye(6) * 1e-3),
                fitness=det.get("fitness", 0.9),
                rmse=det.get("rmse", 0.005),
            ))

        held_id = gripper_state.get("held_obj_id")
        T_ew = None
        T_oe = None
        if (held_id is not None
                and held_id in priors
                and T_ec is not None
                and self.T_oe.get(held_id) is not None):
            T_ew = slam_pose.T @ T_ec
            T_oe = self.T_oe[held_id]

        result = self.optimizer.run(
            slam_pose=slam_pose,
            priors=priors,
            observations=observations,
            relations=self._cached_relations,
            held_obj_id=held_id,
            T_ew=T_ew,
            T_oe=T_oe,
        )

        # Inject optimized posteriors back into every particle (Option A).
        for oid, pe in result.posteriors.items():
            self.state.inject_posterior(oid, pe)

        if self.verbose:
            print(f"[slow tier] α={result.alpha:.2f}, "
                  f"iters={result.num_iterations}, "
                  f"residuals={ {k: v.tolist() for k, v in result.residuals.items()} }")

        return result

    # --------------------------------------------------------------- #
    #  Manipulation-set propagation via scene-graph relations
    # --------------------------------------------------------------- #

    def _get_manipulation_set(self,
                              held_id: Optional[int]) -> Set[int]:
        """Transitive closure of the held object under "in"/"on" relations.

        If the robot holds a bowl and the scene graph says "apple in bowl",
        then the apple rides with the bowl under the rigid-attachment predict.

        Uses `self._cached_relations` from the previous step — spatial
        relations don't change instantaneously, and the trigger policy
        recomputes them frequently.
        """
        if held_id is None:
            return set()

        manipulated: Set[int] = {held_id}
        for _ in range(8):
            changed = False
            for edge in self._cached_relations:
                if edge.relation_type not in ("in", "on"):
                    continue
                # parent = the contained/on-top object, child = container/
                # base: if child is in the manipulation set, parent rides too.
                if edge.child in manipulated and edge.parent not in manipulated:
                    manipulated.add(edge.parent)
                    changed = True
            if not changed:
                break
        return manipulated

    # --------------------------------------------------------------- #
    #  Relation recomputation
    # --------------------------------------------------------------- #

    def _recompute_relations(
        self,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RelationEdge]:
        """Build scene graph edges and smooth through the RelationFilter EMA.

        When ``self.relation_client`` is available, queries it on the current
        frame: normalised bboxes from matched/born detections → ``(N, N)``
        parent-probability matrix → ``RelationEdge`` list. Temporal
        aggregation is handled by the existing per-edge EMA filter
        (α=0.3, threshold=0.5).

        Falls back to the legacy mock-point-cloud geometric test when no
        learned client is configured, or when the client is unavailable
        (server down / API key missing / <2 dets this frame).
        """
        learned = self._try_learned_relations(detections)
        if learned is not None:
            return self._relation_filter.update(learned)

        collapsed = self.state.collapsed_objects()
        if len(collapsed) < 2:
            return self._relation_filter.update([])

        # Legacy geometric mock (mock 5 cm point clouds).
        class _OrchObj:
            def __init__(self, oid, T, size=None):
                self.id = oid
                self.pose_init = T.copy()
                self.pose_cur = T.copy()
                extent = size if size is not None else np.array([0.05] * 3)
                pts = np.random.uniform(-extent, extent, size=(50, 3)) \
                    + T[:3, 3]
                self._points = pts.astype(np.float32)
                self.child_objs = {}
                self.parent_obj_id = None

        mock_objs = [_OrchObj(oid, pe.T) for oid, pe in collapsed.items()]
        try:
            from scene.object_relation_graph import (
                compute_spatial_relations_with_scores,
            )
            _, scores = compute_spatial_relations_with_scores(
                mock_objs, tolerance=0.02, overlap_threshold=0.2)
        except Exception:
            return self._relation_filter.update([])

        raw_edges: List[RelationEdge] = []
        for (parent_id, child_id, rel_type), score in scores.items():
            if rel_type not in ("on", "in"):
                continue
            raw_edges.append(RelationEdge(
                parent=parent_id, child=child_id,
                relation_type=rel_type, score=score,
            ))
        return self._relation_filter.update(raw_edges)

    def _try_learned_relations(
        self,
        detections: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[RelationEdge]]:
        """Call the relation detector on current-frame detections.

        Returns raw ``RelationEdge`` list (pre-EMA) on success, or ``None``
        to signal "use the fallback path" (no client / no inputs / query
        failed / fewer than two oids this frame).
        """
        if self.relation_client is None or self._frame_rgb is None:
            return None
        if not detections or not self._last_det_to_oid:
            return None

        # Build (oid, normalised bbox) pairs from this frame's matched/born
        # detections. One entry per oid (if a track matched twice in one
        # frame, first wins — shouldn't happen after voxel dedup).
        H, W = self._frame_rgb.shape[:2]
        if W <= 0 or H <= 0:
            return None
        oid_list: List[int] = []
        bboxes: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        seen_oids: Set[int] = set()
        for det_idx, oid in self._last_det_to_oid.items():
            if oid in seen_oids:
                continue
            if not (0 <= det_idx < len(detections)):
                continue
            det = detections[det_idx]
            box = det.get("box")
            if box is None:
                continue
            try:
                x0, y0, x1, y1 = (float(b) for b in box)
            except (TypeError, ValueError):
                continue
            bbox_n = np.array(
                [x0 / W, y0 / H, x1 / W, y1 / H],
                dtype=np.float32,
            )
            # Reuse the SAM mask already in the detection (perception emits
            # base64 PNG or ndarray). No re-segmentation here.
            mask = det.get("mask")
            if isinstance(mask, str):
                from pose_update.relations.relation_client import decode_mask_b64
                mask = decode_mask_b64(mask, size=(W, H))
            elif mask is not None:
                mask = np.asarray(mask) > 0
            oid_list.append(int(oid))
            bboxes.append(bbox_n)
            masks.append(mask)
            seen_oids.add(oid)

        if len(oid_list) < 2:
            return None

        # RelationClient wants a PIL image.
        rgb_pil = (self._frame_rgb if isinstance(self._frame_rgb, Image.Image)
                   else Image.fromarray(
                       np.asarray(self._frame_rgb, dtype=np.uint8)))
        usable_masks = masks if all(m is not None for m in masks) else None
        p_parent = self.relation_client.detect(
            rgb_pil, np.stack(bboxes, axis=0), masks=usable_masks,
        )
        if p_parent is None:
            return None

        thr = float(self.bernoulli.relation_score_threshold)
        raw_edges: List[RelationEdge] = []
        n = len(oid_list)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s = float(p_parent[i, j])
                if s < thr:
                    continue
                raw_edges.append(RelationEdge(
                    parent=oid_list[i],
                    child=oid_list[j],
                    relation_type="on",
                    score=s,
                ))
        return raw_edges

    # --------------------------------------------------------------- #
    #  Trigger policy
    # --------------------------------------------------------------- #

    def _maybe_gravity_predict(self,
                                gripper_state: Dict[str, Any]) -> None:
        """Replace the EKF mean + covariance for a just-released object
        with the post-fall prediction.

        Triggered when last_phase ∈ {holding, releasing} and cur_phase is
        not — i.e. the FSM exits the manipulation window. The just-released
        oid is taken from `last_state["held_obj_id"]`. No-ops when:
            * the bernoulli backend is not active,
            * `BernoulliConfig.gravity_predict` is False,
            * `self.voxel_obs` is None,
            * the oid no longer exists in the filter (e.g. pruned mid-release).
        """
        if self.bernoulli is None or not getattr(
                self.bernoulli, "gravity_predict", False):
            return
        if self.voxel_obs is None:
            return
        last_phase = self.last_state.get("phase", "idle")
        cur_phase = gripper_state.get("phase", "idle")
        manip = ("holding", "releasing")
        if not (last_phase in manip and cur_phase not in manip):
            return
        just_released = self.last_state.get("held_obj_id")
        if just_released is None:
            return
        pe = self.state.collapsed_object(just_released)
        if pe is None:
            return
        label = self.object_labels.get(just_released)
        dyn = lookup_dynamics(label)
        # Live-object overlay: every other tracked oid contributes
        # (x, y, z, radius) for the raycast collision check.
        other_voxels = []
        for oid, pe_o in self.state.collapsed_objects().items():
            if oid == just_released:
                continue
            T = pe_o.T
            other_dyn = lookup_dynamics(self.object_labels.get(oid))
            other_voxels.append(
                (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
                 float(other_dyn.radius_m)))
        T_land, P_land, info = predict_landing_pose(
            T_release=pe.T,
            P_release=pe.cov,
            voxel_obs=self.voxel_obs,
            dyn=dyn,
            workspace_floor_z=float(self.bernoulli.workspace_floor_z),
            live_object_voxels=other_voxels,
        )
        ok = self.state.overwrite_object_pose(just_released, T_land, P_land)
        log_entry = info.as_dict()
        log_entry["oid"] = int(just_released)
        log_entry["frame"] = int(self.frame_count)
        log_entry["written_back"] = bool(ok)
        log_entry["label"] = label
        self._gravity_predict_log.append(log_entry)

    def _should_recompute_relations(self,
                                    gripper_state: Dict[str, Any]) -> bool:
        """Gate the relation detector. Fires on key events and a periodic
        safety-net tick (config: ``BernoulliConfig.relation_*``).

        Key events:
          * the first step after the client comes up (``_last_relation_frame``
            is still the sentinel);
          * a grasp / release transition (scene graph almost always changes);
          * a new track joining the set of confirmed tracks.
        """
        if self.bernoulli is None or self.relation_client is None:
            # Legacy geometric path: the mock test is cheap, keep firing
            # every frame (matches the pre-throttle behaviour).
            return True

        cfg = self.bernoulli
        # First call after boot — always fire so _cached_relations populates.
        if self._last_relation_frame < 0:
            return True

        last_phase = self.last_state.get("phase", "idle")
        cur_phase = gripper_state.get("phase", "idle")

        if cfg.relation_on_grasp and \
                last_phase != "grasping" and cur_phase == "grasping":
            return True
        if cfg.relation_on_release and \
                last_phase == "releasing" and cur_phase != "releasing":
            return True

        # New confirmed track since last step? Compare current confirmed-oid
        # set against the snapshot taken at the end of the previous step.
        if cfg.relation_on_new_object:
            if self.bernoulli is not None:
                current = {
                    oid for oid, r in self.existence.items()
                    if r >= cfg.r_conf
                }
            else:
                current = set(self.state.collapsed_objects().keys())
            if current - self._known_oids_before_step:
                return True

        # Periodic tick (N=0 disables).
        if cfg.relation_every_n_frames > 0:
            if (self.frame_count - self._last_relation_frame
                    >= cfg.relation_every_n_frames):
                return True

        return False

    def _should_trigger(self,
                        gripper_state: Dict[str, Any],
                        detections: List[Dict[str, Any]]) -> bool:
        last_phase = self.last_state.get("phase", "idle")
        cur_phase = gripper_state.get("phase", "idle")

        # Manipulation events
        if self.trigger.on_grasp and \
                last_phase != "grasping" and cur_phase == "grasping":
            return True
        if self.trigger.on_release and \
                last_phase == "releasing" and cur_phase != "releasing":
            return True

        # New object appeared (compare against known set *before* this step)
        if self.trigger.on_new_object:
            seen_ids = {d.get("id") for d in detections
                        if d.get("id") is not None}
            known = getattr(self, "_known_before_this_step", set())
            if not seen_ids.issubset(known):
                return True

        # Periodic safety net
        if self.trigger.periodic_every_n_frames > 0:
            if (self.frame_count - self.last_opt_frame
                    >= self.trigger.periodic_every_n_frames):
                return True

        return False

#!/usr/bin/env python3
"""
Per-frame 5-panel visualization of the Bernoulli-EKF scene-graph tracker.

Panel layout (2 rows x 3 cols):
  [1] Perception overlay   [2] Top-down: state entering frame   [3] EKF step intermediates
  [4] Top-down: post-predict   [5] Top-down: post-update       [6] r / cov evolution

Drives a thin instrumented replica of pose_update.orchestrator._fast_tier_bernoulli
(so we can snapshot state between predict / associate / update / birth / prune
and dump per-track intermediates — d2, log_lik, Huber weight, existence r delta, etc.).

Data: expects apple_in_the_tray dataset layout:
  datasets/apple_in_the_tray/
    rgb/rgb_NNNNNN.png
    depth/depth_NNNNNN.npy
    pose_txt/amcl_pose.txt          (world <- base)
  tests/visualization_pipeline/apple_in_the_tray/perception/detection_h/
    detection_NNNNNN_final.json     (SAM2-tracked detections with masks + IDs)

Output: tests/visualization_pipeline/apple_in_the_tray/ekf_debug/
    frame_NNNNNN.png

Run:
    conda run -n ocmp_test python tests/visualize_ekf_tracking.py \
        --max-frame 700 --step 1
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from pose_update.association import hungarian_associate
from pose_update.bernoulli import (
    r_predict, r_assoc_update_loglik, r_miss_update, r_birth,
)
from pose_update.ekf_se3 import (
    huber_weight, process_noise_for_phase, saturate_covariance,
)
from pose_update.gaussian_state import GaussianState
from pose_update.det_dedup import suppress_subpart_detections
from pose_update.icp_pose import (
    PoseEstimator, centroid_cam_from_mask, _back_project,
)
from pose_update.obs_chain import ChainStore
from pose_update.orchestrator import (
    BernoulliConfig, birth_admissible, _PendingBirth, RelationFilter,
)
from pose_update.robot_models import create_gripper_geometry
from pose_update.grasp_owner_detector import (
    GraspOwnerDetector, InstrumentedTrackerState,
)
from pose_update.relation_utils import (
    expand_held_with_relations, should_recompute_relations,
    RelationTriggerState, RelationTriggerConfig,
)
from pose_update.factor_graph import RelationEdge
from pose_update.slam_interface import PoseEstimate
from pose_update.visibility import visibility_p_v


# ─── data paths ──────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(SCENEREP_ROOT, "datasets")
VIZ_BASE = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline")

# ─── Fetch head camera intrinsics (from configs/*.yaml) ──────────────────
K_DEFAULT = np.array([
    [554.3827, 0.0, 320.5],
    [0.0, 554.3827, 240.5],
    [0.0, 0.0,     1.0],
], dtype=np.float64)

# Centroid measurement noise (camera-frame). 2 cm std absorbs the
# perception-side mask boundary noise that free tracks (cup, bottle,
# free apples) produce; held tracks are anchored by rigid-attach
# predict so don't depend on Kalman gain magnitude. Used at three
# call sites (Hungarian d², matched-update innovation diagnostics,
# the actual EKF update); same value everywhere keeps gate / cost /
# update consistent.
_CENTROID_R_CAM_STD_M = 0.02
_R_CENT_CAM_3D = np.diag([_CENTROID_R_CAM_STD_M ** 2] * 3)
# Rotation-decoupling: centroid measurement carries no rotation info,
# so the 6D-shaped innovation pads the rotation block with ∞ so it
# falls out of every solve.
_ROTATION_DECOUPLE_VAR = 1e6

# Per-frame slack covariance for the rigid-attachment predict (grip
# slip / deformation). Tiny on top of Ad·P·Adᵀ.
_Q_MANIP_SLACK = np.diag([1e-6] * 3 + [1e-6] * 3)

# palette matches visualize_sam2_observations for inter-viz consistency
_PALETTE_RGB = [
    (  0, 200,  80), (220,  60,  40), ( 40, 140, 220), (245, 200,  20),
    (160,  80, 200), (240, 130,  30), ( 20, 180, 160), (230, 120, 110),
    (100, 160, 230), (250, 220,  60), ( 80, 200, 120), (200, 100, 160),
    (140, 200,  50), (100, 100, 240), (230, 160,  80), ( 40, 220, 200),
    (220,  80, 200), (120, 120, 120), (200, 220, 120), ( 60, 100, 180),
]


def _palette_color(oid: int) -> Tuple[int, int, int]:
    return _PALETTE_RGB[int(oid) % len(_PALETTE_RGB)]


def _palette_color_f(oid: int) -> Tuple[float, float, float]:
    r, g, b = _palette_color(oid)
    return (r / 255.0, g / 255.0, b / 255.0)


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def _load_amcl_poses(path: str) -> List[np.ndarray]:
    """Parse `amcl_pose.txt` lines: `idx x y z qx qy qz qw`."""
    out: List[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out.append(T)
    return out


def _load_T_bc_poses(path: str) -> Optional[Dict[int, np.ndarray]]:
    """Parse `T_bc.txt` lines: `idx x y z qx qy qz qw` -> {idx: T_bc}.

    T_bc is the per-frame base_link -> head_camera_rgb_optical_frame
    extrinsic, extracted from /tf at dataset-extraction time. Returns
    None if the file is missing (caller falls back to identity, i.e.
    camera == base, which is wrong on Fetch but back-compat with bags
    extracted before T_bc support was added).
    """
    if not os.path.exists(path):
        return None
    out: Dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            idx, tx, ty, tz, qx, qy, qz, qw = arr
            try:
                idx_i = int(idx)
                tx, ty, tz, qx, qy, qz, qw = (
                    float(tx), float(ty), float(tz),
                    float(qx), float(qy), float(qz), float(qw))
            except ValueError:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out[idx_i] = T
    return out if out else None


def _load_ee_poses(path: str) -> Optional[Dict[int, np.ndarray]]:
    """Parse `ee_pose.txt` lines: `idx x y z qx qy qz qw` -> {idx: T_bg}.

    Each line gives the end-effector-in-base pose. Used as T_bg for the
    rigid-attachment predict on held tracks.
    """
    if not os.path.exists(path):
        return None
    out: Dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            try:
                idx_i = int(arr[0])
                tx, ty, tz, qx, qy, qz, qw = (float(v) for v in arr[1:])
            except ValueError:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out[idx_i] = T
    return out if out else None


def _load_gripper_widths(path: str) -> Optional[Dict[int, float]]:
    """Parse `joints_pose.json` -> {idx: l+r finger joint width}.

    The Fetch gripper's two finger joints each report half the opening;
    their sum is the total jaw width in metres. Open ≈ 0.10 m, fully
    closed (around an object) ≈ 0.003 m.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    out: Dict[int, float] = {}
    for fr_key, joints in data.items():
        try:
            idx_i = int(fr_key)
        except (TypeError, ValueError):
            continue
        lg = joints.get("l_gripper_finger_joint")
        rg = joints.get("r_gripper_finger_joint")
        if lg is None or rg is None:
            continue
        out[idx_i] = float(lg) + float(rg)
    return out if out else None


from pose_update.gripper_state import GripperPhaseTracker as _GripperPhaseTracker  # noqa: E402

class _GripperStateInferrer:
    """Driver-side shim around :class:`pose_update.gripper_state.GripperPhaseTracker`.

    The full FSM lives in ``pose_update/gripper_state.py``. This shim
    wraps the production class with the test driver's tracker-coupling
    (it constructs an ``InstrumentedTrackerState`` adapter on each
    ``step``) and exposes the legacy ``_held_obj_id`` attribute name
    used by the rest of this driver.
    """
    def __init__(self, *args, **kwargs):
        # GraspOwnerDetector is the only kwarg the driver passes that
        # isn't part of GripperPhaseTracker's defaults; pass through
        # everything else by name.
        self._inner = _GripperPhaseTracker(*args, **kwargs)
        # Legacy attribute name used elsewhere in the driver.
        self._joints_now = None

    @property
    def _held_obj_id(self):
        return self._inner.held_obj_id

    def apply_merges(self, merges):
        self._inner.apply_merges(merges)

    def step(self, width, tracker, T_wb, T_bg, **kwargs):
        # Adapt the InstrumentedTracker → TrackerState protocol expected
        # by the production phase tracker.
        from pose_update.grasp_owner_detector import InstrumentedTrackerState
        ts = InstrumentedTrackerState(tracker)
        live_oids = set(int(o) for o in tracker.object_labels.keys())
        return self._inner.step(
            width=width, tracker_state=ts,
            T_wb=T_wb, T_bg=T_bg,
            live_oids=live_oids,
            **kwargs)


from pose_update.relation_orchestrator import RelationOrchestrator as _RelationPipeline  # noqa: E402,F401


def _load_detection_json(path: str) -> List[Dict[str, Any]]:
    """Decode one detection_h JSON into a list of dicts (mask decoded)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    out: List[Dict[str, Any]] = []
    for det in data.get("detections", []):
        mask_b64 = det.get("mask", "")
        if not mask_b64:
            continue
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask = np.array(Image.open(BytesIO(mask_bytes)).convert("L"))
            mask = (mask > 128).astype(np.uint8)
        except Exception:
            continue
        out.append({
            "id": int(det.get("object_id")),
            "label": det.get("label", "unknown"),
            # Carry the soft per-detection label distribution from
            # perception (used by the EKF cost as a label history seed
            # and by the JSON state dump for inspection).
            "labels": det.get("labels", {}),
            "mask": mask,
            "score": float(det.get("score", 0.0)),
            "mean_score": float(det.get("mean_score", 0.0)),
            "n_obs": int(det.get("n_obs", 0)),
            "box": det.get("box"),
        })
    return out


def _load_rgb(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_depth(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Instrumented tracker
# ─────────────────────────────────────────────────────────────────────────

class InstrumentedTracker:
    """Thin replica of `orchestrator._fast_tier_bernoulli` with snapshots,
    refactored to base-frame storage (bernoulli_ekf.tex §1.1, §3).

    Owns a `GaussianState` (object-in-base poses; Σ_wb never enters the
    recursion), a `PoseEstimator` (ICP chain mode) that converts
    (mask, depth) into camera-frame `(T_co, R_icp)` per detection,
    and Bernoulli bookkeeping in `self.existence / self.object_labels /
    self.sam2_tau / self.label_scores`.

    `step()` returns a debug dict containing:
      enter_tracks      — tracks at start of frame
      post_predict_tracks — after predict step
      assoc             — {match, unmatched_tracks, unmatched_dets, cost_matrix}
      matched           — per-pair {d2, w, log_lik, r_prev, r_new}
      missed            — per-track {p_v, p_d_tilde, r_prev, r_new}
      births            — per-birth {det_idx, new_oid, r_new, label, score}
      pruned            — per-prune {oid, r}
      post_update_tracks — final state
      slam_pose         — T_wb this frame  (used only for output composition)
    """

    def __init__(self,
                 K: np.ndarray,
                 bernoulli_cfg: BernoulliConfig,
                 pose_method: str = "icp_chain",
                 T_bc: Optional[np.ndarray] = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = bernoulli_cfg
        self.pose_est = PoseEstimator(K=self.K, method=pose_method)
        # Base-frame Bernoulli-EKF state. T_bc is the fixed camera-in-base
        # extrinsic; on the Fetch dataset we don't have it, so default to
        # identity (camera == base). This matches what the test harness
        # was effectively using all along; the difference is that now no
        # part of the recursion touches T_wb / Σ_wb.
        self.state = GaussianState(
            T_bc=T_bc,
            P_min_diag=getattr(bernoulli_cfg, "P_min_diag", None),
        )
        self.object_labels: Dict[int, str] = {}
        self.frames_since_obs: Dict[int, int] = {}
        self.existence: Dict[int, float] = {}
        self.sam2_tau: Dict[int, int] = {}
        # Soft per-track label distribution (mirrors perception's
        # detection_h `labels` field). Used by hungarian_associate's
        # soft-label cost and dumped into the JSON state for inspection.
        self.label_scores: Dict[int, Dict[str, Dict[str, float]]] = {}
        # Previous gripper-in-base, cached frame-to-frame for the
        # ΔT_bg control input on held tracks.
        self._prev_T_bg: Optional[np.ndarray] = None
        # Per-track observation chain (bernoulli_ekf.tex §"Per-track
        # observation chain"). Append-only; loop-closure-aware.
        self.chains = ChainStore()
        self._frame_count = 0
        # Tracker-side pending-birth buffer (see orchestrator._PendingBirth).
        # Keyed by perception id; counters are the tracker's own.
        self._pending_births: Dict[Any, Dict[str, Any]] = {}

    def _merge_label_scores(self,
                              oid: int,
                              det: Dict[str, Any]) -> None:
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
                cur = store.setdefault(lbl, {"n_obs": 0, "mean_score": 0.0})
                n_old = int(cur["n_obs"])
                m_old = float(cur["mean_score"])
                n_total = n_old + n_new
                cur["n_obs"] = n_total
                cur["mean_score"] = ((m_old * n_old + m_new * n_new)
                                       / max(1, n_total))
            return
        lbl = det.get("label")
        if lbl is None:
            return
        score = float(det.get("score", 1.0))
        cur = store.setdefault(lbl, {"n_obs": 0, "mean_score": 0.0})
        n_old = int(cur["n_obs"])
        m_old = float(cur["mean_score"])
        cur["n_obs"] = n_old + 1
        cur["mean_score"] = (m_old * n_old + score) / (n_old + 1)

    # ────────── state capture helper ──────────
    def _capture_tracks(self) -> Dict[int, Dict[str, Any]]:
        """Snapshot every track in BASE FRAME (the storage convention).

        Two world-frame views are derived:
          * `T_world`        -- the EKF current base-frame mean composed
                                with `T_wb` (one-step Markov; what the
                                top-down panels have been showing).
          * `T_world_chain`  -- the chain smoother's world-frame mean
                                from `obs_chain.world_frame_estimate`,
                                which is loop-closure-aware. Returns
                                None if the chain is empty.
        """
        T_wb = self.state.T_wb if self.state.T_wb is not None else np.eye(4)
        T_bc = self.state.T_bc
        out: Dict[int, Dict[str, Any]] = {}
        for oid in self.object_labels:
            pe = self.state.collapsed_object_base(oid)
            if pe is None:
                continue
            label_scores_snap = {
                lbl: {"n_obs": int(stats["n_obs"]),
                      "mean_score": float(stats["mean_score"])}
                for lbl, stats in self.label_scores.get(oid, {}).items()
            }
            T_world = T_wb @ pe.T

            # Chain-smoothed world-frame estimate (canonical).
            T_world_chain = None
            cov_world_chain = None
            chain_len = 0
            chain_n_used = 0
            try:
                est = self.chains.world_frame_estimate(oid, T_bc)
            except Exception:
                est = None
            ch_obj = self.chains.get(oid)
            if ch_obj is not None:
                chain_len = len(ch_obj)
            if est is not None:
                T_world_chain, cov_world_chain, chain_n_used = est

            out[int(oid)] = {
                "T": pe.T.copy(),                # BASE frame
                "cov": pe.cov.copy(),            # BASE frame tangent
                "T_world": T_world.copy(),       # filter composed (Markov)
                "T_world_chain": (T_world_chain.copy()
                                    if T_world_chain is not None else None),
                "cov_world_chain": (cov_world_chain.copy()
                                      if cov_world_chain is not None else None),
                "chain_len": chain_len,
                "chain_n_used": chain_n_used,
                "label": self.object_labels[oid],
                "label_scores": label_scores_snap,
                "r": float(self.existence.get(oid, 0.0)),
                "frames_since_obs": int(self.frames_since_obs.get(oid, 0)),
                "sam2_tau": int(self.sam2_tau.get(oid, -1)),
            }
        return out

    # ────────── birth / prune ──────────
    def _mint_tracker_oid(self, det: Dict[str, Any] = None) -> int:
        """Mint a fresh tracker oid (always a new consecutive int).

        Perception's detection id is data-side metadata and must not
        leak into the tracker's identity space; the tracker oid is
        assigned exclusively here and only at admission time. `det`
        is accepted for signature compatibility but unused.
        """
        return max(self.object_labels.keys(), default=0) + 1

    def _birth(self, det: Dict[str, Any],
               forced_oid: Optional[int] = None) -> Optional[int]:
        """Initialize a new Bernoulli track from an unmatched detection.

        When `forced_oid` is provided (e.g. the caller minted the oid
        first in order to run ICP against `_refs[oid]`), it is used
        instead of the SAM2-id-fallback logic in `_mint_tracker_oid`.
        """
        if forced_oid is not None:
            d_id = int(forced_oid)
        else:
            d_id = self._mint_tracker_oid(det)
        T_co = det.get("T_co")
        if T_co is None:
            return None
        T_co = np.asarray(T_co, dtype=np.float64)
        R_icp = np.asarray(det.get("R_icp", np.eye(6) * 1e-3),
                           dtype=np.float64)
        # Fallback birth covariance: σ = 0.05 m / 0.05 rad per axis
        # (stored as variances, so 0.05² = 2.5e-3 on the diagonal).
        init_cov = (0.5 * (R_icp + R_icp.T)
                    if self.cfg.init_cov_from_R
                    else np.diag([0.05**2] * 6))
        self.state.ensure_object(d_id, T_co, init_cov)
        self.object_labels[d_id] = det.get("label", "unknown")
        self.frames_since_obs[d_id] = 0
        score = float(det.get("score", 1.0))
        self.existence[d_id] = r_birth(score,
                                        lambda_b=self.cfg.lambda_b,
                                        lambda_c=self.cfg.lambda_c)
        tau_raw = det.get("sam2_id", det.get("id"))
        if tau_raw is not None:
            try:
                self.sam2_tau[d_id] = int(tau_raw)
            except (TypeError, ValueError):
                self.sam2_tau[d_id] = -1
        else:
            self.sam2_tau[d_id] = -1
        # Seed the soft label history from this detection.
        self.label_scores[d_id] = {}
        self._merge_label_scores(d_id, det)
        # Seed the observation chain with this birth detection.
        self.chains.append(
            d_id,
            frame=self._frame_count,
            T_co=T_co,
            R_co=R_icp,
            fitness=float(det.get("fitness", 1.0)),
            rmse=float(det.get("rmse", 0.0)),
        )
        return d_id

    def _prune(self, oid: int) -> None:
        self.state.delete_object(oid)
        self.object_labels.pop(oid, None)
        self.frames_since_obs.pop(oid, None)
        self.existence.pop(oid, None)
        self.sam2_tau.pop(oid, None)
        self.label_scores.pop(oid, None)
        self.chains.delete(oid)
        # Also drop the ICP reference cloud for this oid so the
        # memory footprint stays bounded across long sessions.
        self.pose_est._refs.pop(oid, None)

    def _candidate_near_live_track(self, det: Dict[str, Any]
                                    ) -> Optional[Dict[str, Any]]:
        """Thin wrapper around :func:`pose_update.birth_gating.is_near_live_track`.

        See that function for behaviour. This method simply provides
        the tracker context (T_wb, T_bc, held oid, T_we, cfg) so the
        production helper can be called.
        """
        from pose_update.birth_gating import (
            BirthGateConfig, is_near_live_track,
        )
        T_wb = getattr(self.state, "T_wb", None)
        T_bc = getattr(self.state, "T_bc", None)
        cfg = BirthGateConfig(
            birth_min_dist_m=float(
                getattr(self.cfg, "birth_min_dist_m", 0.05)),
            held_birth_radius_m=float(
                getattr(self.cfg, "held_birth_radius_m",
                         getattr(self.cfg, "birth_min_dist_m", 0.05))),
        )
        return is_near_live_track(
            det,
            tracker=self,
            T_wb=T_wb,
            T_bc=T_bc,
            held_oid_now=self._held_oid_now,
            held_T_we_now=self._held_T_we_now,
            cfg=cfg,
        )

    def _compute_visibility(self,
                            depth: np.ndarray,
                            image_shape: tuple) -> Dict[int, float]:
        """Per-track p_v via depth ray-tracing (see `pose_update.visibility`).

        Projects each track's object-local reference cloud (from
        `PoseEstimator._refs[oid].ref_points` — the accumulated surface
        model) through the current `T_bc(t)` + predicted base-frame
        mean, reads the depth image at every projected pixel, and z-
        tests against the predicted sample depth. Fully self-contained
        in the camera frame; no SLAM uncertainty enters.
        """
        T_bc = self.state.T_bc
        T_cb = np.linalg.inv(T_bc)
        tracks: List[Dict[str, Any]] = []
        for oid in self.object_labels:
            pe = self.state.collapsed_object_base(oid)
            if pe is None:
                continue
            T_co = T_cb @ pe.T  # base -> camera (camera-frame mean)
            ref = self.pose_est._refs.get(int(oid))
            entry: Dict[str, Any] = {
                "oid": int(oid),
                "T_co": T_co,
            }
            if ref is not None:
                entry["ref_points_obj"] = ref.ref_points
                entry["obj_radius"] = float(ref.obj_radius)
            tracks.append(entry)
        K = self.cfg.K if self.cfg.K is not None else self.K
        img_shape = self.cfg.image_shape or image_shape
        return visibility_p_v(tracks, K, depth, img_shape)

    # ────────── one step ──────────
    def step(self,
             rgb: np.ndarray,
             depth: np.ndarray,
             T_wb: np.ndarray,
             detections: List[Dict[str, Any]],
             phase: str = "idle",
             T_bg: Optional[np.ndarray] = None,
             held_oids: Optional[set] = None,
             held_seed: Optional[int] = None,
             relation_edges: Optional[Iterable] = None,
             T_bc: Optional[np.ndarray] = None
             ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """One frame of base-frame Bernoulli-EKF tracking.

        Args:
            T_wb: SLAM base-in-world; only ΔT_wb (frame-to-frame) is
                  used as the world-static control input u_k =
                  inv(ΔT_wb). Σ_wb does not enter the recursion.
            T_bg: optional gripper-in-base; ΔT_bg drives the held-object
                  control input u_k = ΔT_bg.
            held_oids: optional set of track oids currently held by the
                  gripper / in the manipulation set. Their predict uses
                  rigid_attachment_predict instead of the static path.
            T_bc: optional (4,4) base-to-camera-optical extrinsic for
                  THIS frame. When supplied, lets the tracker handle
                  head pan/tilt/torso-lift motion correctly: the
                  measurement lift `T_bo = T_bc(t) · T_co` and the ICP
                  prior `T_co_init = T_bc(t)^{-1} · T_bo` use the
                  per-frame value. When None, falls back to whatever
                  was set at construction (default identity, i.e.
                  camera == base).
        """
        dbg: Dict[str, Any] = {"frame": self._frame_count,
                                "slam_pose": T_wb.copy()}

        # 0. SLAM ingest (caches T_wb / prev_T_wb for the static u_k = inv ΔT_wb).
        # The covariance is irrelevant for the recursion; we set it tiny
        # only so as_gaussian/PoseEstimate construction doesn't complain.
        slam_pe = PoseEstimate(T=T_wb.astype(np.float64),
                               cov=np.diag([1e-6] * 6))
        self.state.ingest_slam(slam_pe)
        # Per-frame base-to-camera extrinsic. Must be installed BEFORE
        # any measurement-side call (ICP prior, innovation, update,
        # visibility) -- those all use self.state.T_bc internally.
        if T_bc is not None:
            self.state.set_camera_extrinsic(np.asarray(T_bc,
                                                         dtype=np.float64))
        dbg["T_bc"] = self.state.T_bc.copy()
        # Record this frame's pose for the observation chain's
        # post-hoc world-frame composition (loop-closure-ready).
        self.chains.record_pose(self._frame_count, T_wb)

        # Enter snapshot (BEFORE predict — i.e., last frame's posterior).
        dbg["enter_tracks"] = self._capture_tracks()

        # 1. Predict state + existence FIRST (standard KF order:
        #    predict → measure → update, not the other way around).
        #    Branch A: world-static tracks (NOT in held set) -- u_k =
        #    inv(ΔT_wb), supplied by the SLAM-derived base motion inside
        #    GaussianState.predict_static.  Held tracks get Q=0 here so we
        #    don't double-count Q_manip when the rigid step runs below.
        held_oids = held_oids or set()
        # Held-track anchoring on T_we = T_wb @ T_bg (proprioception).
        # Stashed for `_candidate_near_live_track` (births' proximity gate
        # uses T_we for the held label) and the matched-update loop
        # (Euclidean prefilter on held-track measurements).
        # Prefer the explicit `held_seed` (the directly-grasped oid)
        # when supplied — `held_oids` is the relation-expanded set and
        # `next(iter(set))` is non-deterministic (it picked an apple
        # instead of the tray at fr 519, breaking self-merge protection).
        if held_seed is not None and int(held_seed) in held_oids:
            self._held_oid_now: Optional[int] = int(held_seed)
        else:
            self._held_oid_now = (next(iter(held_oids))
                                  if held_oids else None)
        if self._held_oid_now is not None and T_bg is not None:
            self._held_T_we_now = T_wb @ np.asarray(T_bg, dtype=np.float64)
        else:
            self._held_T_we_now = None

        def Q_fn(oid: int) -> np.ndarray:
            return process_noise_for_phase(
                phase=phase,
                is_target=False,
                frames_since_observation=self.frames_since_obs.get(oid, 0),
                frame="base",  # <- base-frame Q now (legacy "world" param
                                #    was a misnomer for the RBPF path).
            )
        # Held oids are FULLY SKIPPED here; their motion is owned by the
        # rigid-attachment branch below (μ ← ΔT_bg · μ). Applying u_k to
        # held tracks first would compose to `ΔT_bg · u_k · μ`, which is
        # wrong for the held kinematics (`T_bo = T_bg · T_go`).
        self.state.predict_static(Q_fn, P_max=self.cfg.P_max,
                                   skip_oids=held_oids)

        # Branch B: held / manipulation-set tracks -- u_k = ΔT_bg.
        # Per-oid Q: the SEED (directly-grasped object) gets the
        # phase-aware Q (larger during grasping/releasing, smaller in
        # holding). TRANSITIVE held members (apples sitting on the
        # held tray) always get the smaller holding-Q regardless of
        # phase — the gripper isn't touching them, they don't deform
        # while it closes around the seed.
        delta_T_bg = None
        if T_bg is not None and self._prev_T_bg is not None and held_oids:
            delta_T_bg = T_bg @ np.linalg.inv(self._prev_T_bg)
            Q_seed = process_noise_for_phase(
                phase=phase, is_target=True, frame="base",
            )
            Q_transitive = process_noise_for_phase(
                phase="holding", is_target=True, frame="base",
            )
            for oid in held_oids:
                if oid in self.state.objects:
                    Q_oid = (Q_seed if oid == self._held_oid_now
                              else Q_transitive)
                    self.state.rigid_attachment_predict(
                        oid, delta_T_bg, Q_oid,
                        P_max=self.cfg.P_max)
        # Cache for next frame.
        self._prev_T_bg = None if T_bg is None else T_bg.copy()

        for oid in self.frames_since_obs:
            self.frames_since_obs[oid] += 1
        for oid in list(self.existence.keys()):
            self.existence[oid] = r_predict(self.existence[oid],
                                             self.cfg.p_s)

        dbg["post_predict_tracks"] = self._capture_tracks()

        # 2a. Pre-association voxel dedup: suppress detections whose
        # back-projected 3D voxel sets are mostly contained in another
        # detection in the SAME frame. Survivors are spatially disjoint.
        detections, subpart_absorbed = suppress_subpart_detections(
            list(detections), depth, self.K,
            voxel_size=getattr(self.cfg, "dedup_voxel_size_m", 0.02),
            containment_thresh=getattr(
                self.cfg, "dedup_containment_thresh", 0.8),
            require_same_label=getattr(
                self.cfg, "dedup_require_same_label", False),
        )
        dbg["subpart_absorbed"] = subpart_absorbed
        # Filled in below by the centroid-validity loop. Initialised
        # here so the JSON dump always carries the key.
        dbg["centroid_dropped"] = []

        # 2. COARSE measurement: back-project each mask's centroid to the
        #    camera frame. No ICP here -- ICP runs once per matched pair
        #    below, seeded with the correct tracker's filter prior.
        dets_with_pose: List[Dict[str, Any]] = []
        for det in detections:
            centroid = centroid_cam_from_mask(
                det["mask"], depth, self.K)
            d = dict(det)
            d["_centroid_cam"] = (None if centroid is None
                                    else centroid.copy())
            d["_centroid_ok"] = centroid is not None
            # Placeholders for the fine ICP result, filled in after
            # Hungarian. Kept so the JSON dump schema is stable.
            d["T_co"] = None
            d["R_icp"] = None
            d["fitness"] = 0.0
            d["rmse"] = 0.0
            d["_icp_ok"] = False
            d["_icp_prior_oid"] = None
            d["_icp_prior_used"] = False
            d["_icp_refined"] = False
            dets_with_pose.append(d)

        # Cached camera-frame prior per track (for post-Hungarian fine ICP).
        T_cb = np.linalg.inv(self.state.T_bc)
        T_co_pred: Dict[int, np.ndarray] = {}
        for o in self.object_labels:
            pe = self.state.collapsed_object_base(o)
            if pe is None:
                continue
            T_co_pred[int(o)] = T_cb @ pe.T

        # 3. Associate (Hungarian or oracle) on the CENTROID-ONLY d^2.
        # Feed ONLY detections with a valid centroid into association;
        # a detection with too few valid-depth pixels is dropped
        # silently (will not spawn a birth either).
        #
        # Custom innovation_fn: returns a 6D-shaped (nu, S, d2, logL)
        # whose translation block carries the true 3-DOF centroid
        # Mahalanobis and whose rotation block is a decoupled
        # `1e6 * I_3` so the Hungarian block decomposition gives
        # `d^2_trans = centroid d^2` exactly, `d^2_rot = 0`, and the
        # off-diagonal is 0 (critical: if we lifted a 6D R_icp through
        # Ad(T_bc) the huge rotation would contaminate the translation
        # block via `[t_bc]_x · R_rot · [t_bc]_x^T`, which dominates
        # for any non-trivial t_bc).  Gate / cost then work as usual
        # with `gate_mode='trans'`, `cost_d2_mode='trans'/'sum'`.
        # 2 cm centroid std absorbs the perception-side mask
        # boundary noise that free tracks (cup, bottle, free apples)
        # produce as the gripper passes overhead. Held-track motion
        # is owned by the rigid-attach predict, not Kalman gain
        # magnitude, so a looser R doesn't slow them down.
        _ROT_PAD = np.eye(3, dtype=np.float64) * 1e6
        def _centroid_innov(oid: int,
                              T_co: np.ndarray,
                              R_icp: np.ndarray):
            centroid_cam = np.asarray(T_co, dtype=np.float64)[:3, 3]
            stats3 = self.state.centroid_innovation_stats(
                oid, centroid_cam, R_cam=_R_CENT_CAM_3D)
            if stats3 is None:
                return None
            nu3, S3, d2_3, logL3 = stats3
            nu6 = np.zeros(6, dtype=np.float64)
            nu6[:3] = nu3
            S6 = np.zeros((6, 6), dtype=np.float64)
            S6[:3, :3] = S3
            S6[3:, 3:] = _ROT_PAD       # decoupled: no off-diagonals
            return nu6, S6, d2_3, logL3
        dets_for_assoc = []
        det_idx_in_assoc: List[int] = []
        for gi, d in enumerate(dets_with_pose):
            if not d.get("_centroid_ok"):
                dbg["centroid_dropped"].append({
                    "det_idx": int(gi),
                    "pid": (int(d["id"]) if d.get("id") is not None
                             else None),
                    "label": d.get("label"),
                    "score": float(d.get("score", 0.0)),
                    "reason": "no_valid_depth",
                })
                continue
            fake_T_co = np.eye(4, dtype=np.float64)
            fake_T_co[:3, 3] = d["_centroid_cam"]
            dets_for_assoc.append({**d,
                                    "T_co": fake_T_co,
                                    "R_icp": np.zeros((6, 6))})
            det_idx_in_assoc.append(gi)
        track_oids = list(self.object_labels.keys())
        if self.cfg.association_mode == "oracle":
            assoc = oracle_associate(track_oids, dets_for_assoc)
        else:
            assoc = hungarian_associate(
                track_oids=track_oids,
                detections=dets_for_assoc,
                innovation_fn=_centroid_innov,
                track_labels=self.object_labels,
                track_tau=self.sam2_tau,
                alpha=self.cfg.alpha,
                G_out=self.cfg.G_out,
                enforce_label_match=self.cfg.enforce_label_match,
                track_label_histories=self.label_scores,
                label_penalty=self.cfg.hungarian_label_penalty,
                score_weight=self.cfg.hungarian_score_weight,
                gate_mode=self.cfg.gate_mode,
                G_out_trans=self.cfg.G_out_trans,
                G_out_rot=self.cfg.G_out_rot,
                cost_d2_mode=self.cfg.cost_d2_mode,
            )

        # 3b. FINE measurement: for every matched (track, det) pair,
        # run ICP once with the tracker's filter prior as the seed and
        # the tracker-oid-keyed reference cloud (`PoseEstimator._refs`
        # is indexed by the tracker oid, not the SAM2 id).
        #
        # When ICP's fitness gate rejects the refinement, DON'T drop the
        # match. Hungarian already certified the pair via the 3-DOF
        # centroid Mahalanobis; the centroid measurement is still valid
        # even if the 6-DOF surface alignment is poor. Fall back to a
        # CENTROID-ONLY measurement with inflated rotation covariance
        # so the Kalman gain on rotation is ~0 (no rotation update, but
        # full translation update). Common-practice state-estimation
        # answer: "less confident measurement → larger R, not drop".
        _R_BIG = 1e6     # large rotation variance for centroid fallback
        _SIGMA_T_FB = 0.02
        for oid, l_local in list(assoc.match.items()):
            l_global = det_idx_in_assoc[l_local]
            det = dets_with_pose[l_global]
            T_co_init = T_co_pred.get(int(oid))
            T_co, R_icp, fitness, rmse = self.pose_est.estimate(
                oid=int(oid),
                mask=det["mask"], depth=depth,
                T_co_init=T_co_init,
            )
            det["fitness"] = fitness
            det["rmse"] = rmse
            det["_icp_prior_oid"] = int(oid)
            det["_icp_prior_used"] = T_co_init is not None
            if T_co is not None:
                # Full 6-DOF ICP measurement.
                det["T_co"] = T_co
                det["R_icp"] = R_icp
                det["_icp_ok"] = True
                det["_measurement_kind"] = "icp"
            else:
                # Centroid fallback: use the mask centroid as the
                # translation and the filter's predicted rotation, with
                # a HUGE rotation covariance so only translation
                # information flows into the update.
                centroid_cam = det.get("_centroid_cam")
                if centroid_cam is None:
                    det["T_co"] = None
                    det["R_icp"] = None
                    det["_icp_ok"] = False
                    det["_measurement_kind"] = "dropped"
                    continue
                T_co_fb = np.eye(4, dtype=np.float64)
                # Predicted rotation → zero rotation innovation.
                if T_co_init is not None:
                    T_co_fb[:3, :3] = T_co_init[:3, :3]
                T_co_fb[:3, 3] = np.asarray(centroid_cam,
                                               dtype=np.float64)
                R_fb = np.diag([_SIGMA_T_FB ** 2] * 3
                                 + [_R_BIG] * 3)
                det["T_co"] = T_co_fb
                det["R_icp"] = R_fb
                det["_icp_ok"] = True
                det["_measurement_kind"] = "centroid_fallback"
        # Map assoc local indices back to dets_with_pose indices
        local_to_global = {li: gi for li, gi in enumerate(det_idx_in_assoc)}
        match_global = {oid: local_to_global[l]
                        for oid, l in assoc.match.items()}
        # det_indices_in_assoc: list of GLOBAL det indices in column order of
        # the cost matrix (lets the renderer/diagnostic relate cost cells back
        # to specific detections).
        # det_meta_in_assoc: per-column (sam2_id, label) snapshots so the panel
        # can label columns without needing the raw detection list at render
        # time.
        det_meta = []
        for gi in det_idx_in_assoc:
            d = dets_with_pose[gi]
            det_meta.append({
                "global_idx": int(gi),
                "sam2_id": int(d["id"]) if d.get("id") is not None else None,
                "label": d.get("label", "?"),
                "score": float(d.get("score", 0.0)),
            })
        # Per-track label-history snapshots for the cost-matrix panel.
        track_label_hist_strs: List[str] = []
        for o in track_oids:
            hist = self.label_scores.get(int(o), {})
            if not hist:
                track_label_hist_strs.append("")
                continue
            ranked = sorted(hist.items(),
                            key=lambda kv: -kv[1].get("n_obs", 0))
            track_label_hist_strs.append(
                "/".join(f"{lbl[:4]}({s['n_obs']})" for lbl, s in ranked[:3]))
        dbg["assoc"] = {
            "track_oids": [int(o) for o in track_oids],
            "track_labels": [self.object_labels.get(int(o), "?")
                              for o in track_oids],
            "track_label_hists": track_label_hist_strs,
            "track_taus": [int(self.sam2_tau.get(int(o), -1))
                            for o in track_oids],
            "match": {int(o): int(l) for o, l in match_global.items()},
            "match_local": {int(o): int(l) for o, l in assoc.match.items()},
            "unmatched_tracks": [int(o) for o in assoc.unmatched_tracks],
            "unmatched_dets_local": [int(l) for l in assoc.unmatched_detections],
            "cost_matrix": assoc.cost_matrix.tolist()
                if assoc.cost_matrix.size else [],
            "det_indices_in_assoc": [int(g) for g in det_idx_in_assoc],
            "det_meta_in_assoc": det_meta,
            "n_dets_for_assoc": len(dets_for_assoc),
            "n_dets_total": len(dets_with_pose),
            "alpha": float(self.cfg.alpha),
            "G_out": float(self.cfg.G_out),
            "label_penalty": float(self.cfg.hungarian_label_penalty),
            "score_weight": float(self.cfg.hungarian_score_weight),
            "enforce_label_match": bool(self.cfg.enforce_label_match),
            "gate_mode": str(self.cfg.gate_mode),
            "G_out_trans": float(self.cfg.G_out_trans),
            "G_out_rot": float(self.cfg.G_out_rot),
            "cost_d2_mode": str(self.cfg.cost_d2_mode),
            "d2_full_matrix":
                (assoc.d2_full_matrix.tolist()
                 if assoc.d2_full_matrix is not None
                 and assoc.d2_full_matrix.size else []),
            "d2_trans_matrix":
                (assoc.d2_trans_matrix.tolist()
                 if assoc.d2_trans_matrix is not None
                 and assoc.d2_trans_matrix.size else []),
            "d2_rot_matrix":
                (assoc.d2_rot_matrix.tolist()
                 if assoc.d2_rot_matrix is not None
                 and assoc.d2_rot_matrix.size else []),
        }

        # 4. Visibility: depth ray-tracing against each track's
        #    accumulated reference cloud (see pose_update/visibility.py).
        #    Always computed; no config gate. p_v[oid] ∈ [0, 1]
        #    gates the miss-update penalty in Step 6.
        image_shape = rgb.shape[:2]
        p_v_map = self._compute_visibility(depth, image_shape)
        dbg["visibility"] = {int(o): float(v) for o, v in p_v_map.items()}

        # 5. Matched updates.
        # The fine ICP already ran in Step 3b above; this loop just
        # consumes the (T_co, R_icp) each matched detection now carries
        # and runs the EKF update. Pairs whose fine ICP failed the
        # fitness/rmse gate (T_co is None) are demoted to miss/birth.
        consumed_global: set = set()
        dbg["matched"] = []
        held_meas_radius = float(getattr(self.cfg, "held_meas_radius_m", 0.0))
        for oid, l_local in list(assoc.match.items()):
            l_global = det_idx_in_assoc[l_local]
            det = dets_with_pose[l_global]
            if not det.get("_icp_ok"):
                # Fine ICP failed its fitness/rmse gate. Demote to miss;
                # the detection won't birth here either because its
                # centroid wasn't unique enough to beat the gate.
                assoc.unmatched_tracks.append(oid)
                del assoc.match[oid]
                match_global.pop(oid, None)
                continue

            # Held-track measurement prefilter: when the matched track is
            # the held oid, the centroid_w must be within
            # `held_meas_radius_m` of T_we. Bad gripper-occluded sliver
            # centroids would otherwise pass Hungarian's gate (because
            # the EKF prior is wide during holding) and pull mu off-target,
            # causing the held-mu drift cascade. The detection IS still
            # consumed (it's the held apple's view; we don't want to
            # birth a duplicate); only the EKF update is skipped.
            if (oid == self._held_oid_now
                    and self._held_T_we_now is not None
                    and held_meas_radius > 0.0):
                c_cam = det.get("_centroid_cam")
                if c_cam is not None:
                    c_h = np.array([float(c_cam[0]), float(c_cam[1]),
                                     float(c_cam[2]), 1.0],
                                    dtype=np.float64)
                    c_world = (self.state.T_wb
                                @ self.state.T_bc @ c_h)[:3]
                    we = self._held_T_we_now[:3, 3]
                    err = float(np.linalg.norm(c_world - we))
                    if err > held_meas_radius:
                        # Skip EKF update; consume detection so it
                        # doesn't birth; mark as held-prefilter reject.
                        consumed_global.add(l_global)
                        del assoc.match[oid]
                        match_global.pop(oid, None)
                        dbg["matched"].append({
                            "oid": int(oid), "det_idx": int(l_global),
                            "pid": det.get("id"),
                            "d2": float("nan"),
                            "d2_trans": float("nan"),
                            "d2_rot": float("nan"),
                            "w": 0.0,
                            "reject_held_prefilter": True,
                            "held_meas_err_m": err,
                            "held_meas_radius_m": float(held_meas_radius),
                            "log_lik": 0.0,
                            "r_prev": float(self.existence.get(oid, 0.0)),
                            "r_new": float(self.existence.get(oid, 0.0)),
                            "fitness": float(det.get("fitness", 0.0)),
                            "rmse": float(det.get("rmse", 0.0)),
                            "icp_prior_used": False,
                            "icp_prior_oid": None,
                            "icp_refined": False,
                            "measurement_kind": "held_prefilter_skip",
                        })
                        continue
                    # Held-track innovation clamp: even if the measurement
                    # passed the T_we prefilter, reject any that would
                    # snap the *track's own* mu_w by more than the
                    # configured ceiling. Stops in-gate measurements
                    # from yanking μ when the track has drifted.
                    innov_max = float(getattr(
                        self.cfg, "held_meas_innov_max_m", 0.0))
                    if innov_max > 0.0:
                        pe = self.state.collapsed_object_base(int(oid))
                        if pe is not None:
                            mu_b = np.asarray(pe.T,
                                                dtype=np.float64)[:3, 3]
                            mu_w = (self.state.T_wb
                                     @ np.append(mu_b, 1.0))[:3]
                            innov_dist = float(np.linalg.norm(c_world - mu_w))
                            if innov_dist > innov_max:
                                consumed_global.add(l_global)
                                del assoc.match[oid]
                                match_global.pop(oid, None)
                                dbg["matched"].append({
                                    "oid": int(oid),
                                    "det_idx": int(l_global),
                                    "pid": det.get("id"),
                                    "d2": float("nan"),
                                    "d2_trans": float("nan"),
                                    "d2_rot": float("nan"),
                                    "w": 0.0,
                                    "reject_innov_clamp": True,
                                    "innov_dist_m": innov_dist,
                                    "innov_max_m": innov_max,
                                    "log_lik": 0.0,
                                    "r_prev": float(
                                        self.existence.get(oid, 0.0)),
                                    "r_new": float(
                                        self.existence.get(oid, 0.0)),
                                    "fitness": float(det.get("fitness", 0.0)),
                                    "rmse": float(det.get("rmse", 0.0)),
                                    "icp_prior_used": False,
                                    "icp_prior_oid": None,
                                    "icp_refined": False,
                                    "measurement_kind": "innov_clamp_skip",
                                })
                                continue
            kind = det.get("_measurement_kind", "icp")
            if kind == "centroid_fallback":
                # 3-DOF path: use a dedicated innovation + Joseph-update
                # on the translation partition only. Bypasses
                # `lift_measurement_base` (which Ad-couples the huge
                # R_rot into the translation block).
                centroid_cam = det["_centroid_cam"]
                stats3 = self.state.innovation_stats_centroid_3d(
                    oid, centroid_cam, R_cam=_R_CENT_CAM_3D)
                if stats3 is None:
                    continue
                nu3, S3, d2_t, log_lik = stats3
                d2 = d2_t        # 3-DOF d² is the only meaningful one here
                d2_r = 0.0
                # Build 6D nu/S for legacy logging only (diagnostic fields).
                nu = np.zeros(6); nu[:3] = nu3
                S = np.zeros((6, 6)); S[:3, :3] = S3
                S[3:, 3:] = np.eye(3) * 1e6
            else:
                # Full 6-DOF ICP measurement.
                T_co = np.asarray(det["T_co"], dtype=np.float64)
                R_icp = np.asarray(det["R_icp"], dtype=np.float64)
                stats = self.state.innovation_stats(oid, T_co, R_icp)
                if stats is None:
                    continue
                nu, S, d2, log_lik = stats
                try:
                    d2_t = float(nu[:3] @ np.linalg.solve(S[:3, :3], nu[:3]))
                    d2_r = float(nu[3:] @ np.linalg.solve(S[3:, 3:], nu[3:]))
                except np.linalg.LinAlgError:
                    d2_t = float("nan"); d2_r = float("nan")
            # Pick which d^2 component drives the Huber gate, matching the
            # gate_mode used in Hungarian. If we use the full 6-D d^2 here
            # while Hungarian gated on translation only, an unreliable
            # rotation block would silently flip a Hungarian-accepted
            # match into a Huber-reject (and the det then wrongly falls
            # to birth) -- the bug surfaced on apple_in_the_tray frame
            # 485 with d^2_full=112 vs d^2_trans=0.02.
            if self.cfg.gate_mode == "trans":
                d2_huber = d2_t
                G_in_h, G_out_h = self.cfg.G_in_trans, self.cfg.G_out_trans
            elif self.cfg.gate_mode == "trans_and_rot":
                d2_huber = max(d2_t, d2_r)
                G_in_h, G_out_h = self.cfg.G_in_trans, self.cfg.G_out_trans
            else:
                d2_huber = d2
                G_in_h, G_out_h = self.cfg.G_in, self.cfg.G_out
            w = (huber_weight(d2_huber, G_in_h, G_out_h)
                 if self.cfg.enable_huber else 1.0)
            r_prev = self.existence.get(oid, 1.0)

            if w <= 0.0:
                # Outer-gate reject: track goes to miss branch; det to birth.
                assoc.unmatched_tracks.append(oid)
                del assoc.match[oid]
                del match_global[oid]
                dbg["matched"].append({
                    "oid": int(oid), "det_idx": int(l_global),
                    "pid": det.get("id"),
                    "d2": float(d2),
                    "d2_trans": float(d2_t), "d2_rot": float(d2_r),
                    "w": 0.0,
                    "reject_outer_gate": True,
                    "log_lik": float(log_lik),
                    "r_prev": float(r_prev), "r_new": float(r_prev),
                    "fitness": float(det.get("fitness", 0.0)),
                    "rmse": float(det.get("rmse", 0.0)),
                    "icp_prior_used": bool(det.get("_icp_prior_used", False)),
                    "icp_prior_oid": (int(det["_icp_prior_oid"])
                                       if det.get("_icp_prior_oid") is not None
                                       else None),
                    "icp_refined": bool(det.get("_icp_refined", False)),
                    "measurement_kind": str(det.get("_measurement_kind",
                                                      "icp")),
                })
                continue

            if kind == "centroid_fallback":
                self.state.update_observation_centroid(
                    oid=oid,
                    centroid_cam=det["_centroid_cam"],
                    # 2 cm std (matched to assoc R_cam).
                    R_cam=_R_CENT_CAM_3D,
                    huber_w=w, P_max=self.cfg.P_max,
                )
            else:
                self.state.update_observation(
                    oid=oid, T_co_meas=T_co, R_icp=R_icp,
                    iekf_iters=2, huber_w=w, P_max=self.cfg.P_max,
                )
            self.frames_since_obs[oid] = 0
            consumed_global.add(l_global)
            r_new = r_assoc_update_loglik(
                r_prev, log_L=log_lik,
                p_d=self.cfg.p_d, lambda_c=self.cfg.lambda_c)
            self.existence[oid] = r_new
            dbg["matched"].append({
                "oid": int(oid), "det_idx": int(l_global),
                "pid": det.get("id"),
                "d2": float(d2),
                "d2_trans": float(d2_t), "d2_rot": float(d2_r),
                "w": float(w),
                "reject_outer_gate": False,
                "log_lik": float(log_lik),
                "r_prev": float(r_prev), "r_new": float(r_new),
                "fitness": float(det.get("fitness", 0.0)),
                "rmse": float(det.get("rmse", 0.0)),
                "icp_prior_used": bool(det.get("_icp_prior_used", False)),
                "icp_prior_oid": (int(det["_icp_prior_oid"])
                                   if det.get("_icp_prior_oid") is not None
                                   else None),
                "icp_refined": bool(det.get("_icp_refined", False)),
                "measurement_kind": str(det.get("_measurement_kind",
                                                  "icp")),
            })
            tau_raw = det.get("sam2_id", det.get("id"))
            if tau_raw is not None:
                try:
                    self.sam2_tau[oid] = int(tau_raw)
                except (TypeError, ValueError):
                    pass
            # Soft label history maintenance.
            self._merge_label_scores(oid, det)
            # Observation chain maintenance (canonical world-frame state).
            self.chains.append(
                oid,
                frame=self._frame_count,
                T_co=T_co,
                R_co=R_icp,
                fitness=float(det.get("fitness", 1.0)),
                rmse=float(det.get("rmse", 0.0)),
            )

        # 6. Missed updates.
        # Special case for the held oid: the gripper is closed; we know
        # the object exists. Don't decay r below `cfg.r_held_floor`.
        # This survives long stretches where the held apple is fully
        # occluded by the fingers (no visible mask → no match → would
        # otherwise prune after enough miss frames).
        dbg["missed"] = []
        r_held_floor = float(getattr(self.cfg, "r_held_floor", 0.0))
        # Pre-compute helpers for "why didn't this oid match?".
        gate_trans = float(self.cfg.G_out_trans)
        d2t_mat = (assoc.d2_trans_matrix
                    if (assoc.d2_trans_matrix is not None
                        and assoc.d2_trans_matrix.size) else None)
        track_oids_in_assoc = list(track_oids)  # row order in d2t_mat
        consumed_local = set(int(l) for l in match_global.values())
        for oid in assoc.unmatched_tracks:
            if oid not in self.existence:
                continue
            p_v = float(p_v_map.get(int(oid), 1.0))
            pdt = self.cfg.p_d * p_v
            r_prev = self.existence[oid]
            r_new = r_miss_update(r_prev, pdt)
            if oid == self._held_oid_now and r_held_floor > 0.0:
                r_new = max(r_new, r_held_floor)
            self.existence[oid] = r_new
            # Why no match? Scan the d² row for this oid.
            best_d2 = float("nan")
            best_local = -1
            best_pid = None
            miss_reason = "no_dets"
            if d2t_mat is not None and len(dets_for_assoc) > 0:
                try:
                    row = int(track_oids_in_assoc.index(int(oid)))
                except ValueError:
                    row = -1
                if row >= 0 and row < d2t_mat.shape[0]:
                    row_vals = d2t_mat[row]
                    if row_vals.size:
                        bi = int(np.argmin(row_vals))
                        best_d2 = float(row_vals[bi])
                        best_local = bi
                        if 0 <= bi < len(det_idx_in_assoc):
                            gi = int(det_idx_in_assoc[bi])
                            if 0 <= gi < len(dets_with_pose):
                                best_pid = dets_with_pose[gi].get("id")
                        if not np.isfinite(best_d2):
                            miss_reason = "all_gated"
                        elif best_d2 > gate_trans:
                            miss_reason = "all_gated"
                        elif bi in consumed_local:
                            miss_reason = "consumed"
                        else:
                            miss_reason = "other"
                else:
                    miss_reason = "not_in_assoc"
            dbg["missed"].append({
                "oid": int(oid),
                "pid": int(self.sam2_tau.get(oid, -1)),
                "p_v": p_v, "p_d_tilde": pdt,
                "r_prev": float(r_prev), "r_new": float(r_new),
                "best_d2_trans": best_d2,
                "best_local_idx": int(best_local),
                "best_pid": (int(best_pid)
                              if best_pid is not None else None),
                "gate": float(gate_trans),
                "reason": miss_reason,
            })

        # 7. Birth from unassigned detections.
        # Policy gate FIRST (border + tracker-side confirm counter + score),
        # ICP ONLY on admission. No oid is minted for a rejected candidate,
        # so `pose_est._refs` stays in 1:1 correspondence with committed
        # tracks — no leaked clouds poisoning subsequent frames.
        dbg["births"] = []
        dbg["birth_rejects"] = []
        for g_idx, det in enumerate(dets_with_pose):
            if g_idx in consumed_global:
                continue
            if not det.get("_centroid_ok"):
                continue

            pid = det.get("id")
            pid_key = pid if pid is not None else ("none", g_idx)
            pending = self._pending_births.get(pid_key)
            if pending is None:
                pending = _PendingBirth.from_det(det, self._frame_count)
                self._pending_births[pid_key] = pending
            pending.bump(det, self._frame_count)

            admit, reason = birth_admissible(
                det, self.cfg, image_shape,
                tracker_n_obs=pending.n_obs_tracker,
                tracker_max_score=pending.max_score,
                require_pose=False,
            )
            if not admit:
                rec: Dict[str, Any] = {
                    "det_idx": int(g_idx),
                    "pid": det.get("id"),
                    "label": str(det.get("label", "unknown")),
                    "score": float(det.get("score", 0.0)),
                    "reason": reason,
                    "n_obs_tracker": pending.n_obs_tracker,
                }
                if reason == "confirm":
                    rec["confirm_k"] = int(self.cfg.birth_confirm_k)
                elif reason == "score":
                    rec["score_min"] = float(self.cfg.birth_score_min)
                    rec["tracker_max_score"] = float(pending.max_score)
                elif reason == "border":
                    rec["box"] = det.get("box")
                    rec["margin_px"] = int(self.cfg.birth_border_margin_px)
                    rec["image_shape"] = list(image_shape) if image_shape else None
                dbg["birth_rejects"].append(rec)
                continue

            # Proximity gate against existing same-label live tracks.
            # Catches SAM2-id-reseed duplicates whose underlying object
            # is already tracked under another oid.
            near = self._candidate_near_live_track(det)
            if near is not None:
                dbg["birth_rejects"].append({
                    "det_idx": int(g_idx),
                    "pid": det.get("id"),
                    "label": str(det.get("label", "unknown")),
                    "score": float(det.get("score", 0.0)),
                    "reason": "near_live",
                    "n_obs_tracker": pending.n_obs_tracker,
                    "nearest_oid": int(near["nearest_oid"]),
                    "dist_m": float(near["dist_m"]),
                    "gate_m": float(near["gate_m"]),
                    "anchor": str(near["anchor"]),
                })
                self._pending_births.pop(pid_key, None)
                continue

            # Admitted → mint a fresh oid and run ICP ONCE to seed the
            # track's belief. Measurement, not a gate.
            new_oid = self._mint_tracker_oid()
            T_co, R_icp, fitness, rmse = self.pose_est.estimate(
                oid=int(new_oid),
                mask=det["mask"], depth=depth,
                T_co_init=None,
            )
            if T_co is None:
                self.pose_est._refs.pop(int(new_oid), None)
                continue
            det["T_co"] = T_co
            det["R_icp"] = R_icp
            det["fitness"] = fitness
            det["rmse"] = rmse
            det["_icp_ok"] = True
            born = self._birth(det, forced_oid=int(new_oid))
            if born is not None:
                dbg["births"].append({
                    "det_idx": int(g_idx), "new_oid": int(born),
                    "pid": det.get("id"),
                    "label": str(det.get("label", "unknown")),
                    "score": float(det.get("score", 0.0)),
                    "r_new": float(self.existence.get(born, 0.0)),
                    "n_obs_tracker": pending.n_obs_tracker,
                })
                self._pending_births.pop(pid_key, None)

        # TTL-expire stale pending entries (perception id disappeared).
        ttl = int(getattr(self.cfg, "birth_pending_ttl_frames", 30))
        if ttl > 0 and self._pending_births:
            cutoff = self._frame_count - ttl
            stale = [k for k, p in self._pending_births.items()
                     if p.last_seen_frame < cutoff]
            for k in stale:
                self._pending_births.pop(k, None)

        # 8. Prune.
        dbg["pruned"] = []
        if self.cfg.r_min > 0.0:
            to_prune = [o for o, r in self.existence.items()
                        if r < self.cfg.r_min]
            for oid in to_prune:
                dbg["pruned"].append({"oid": int(oid),
                                       "r": float(self.existence[oid])})
                self._prune(oid)

        # 9. Track-to-track self-merge (catches the case where one
        # physical object spawned multiple track ids because Hungarian's
        # one-to-one constraint couldn't absorb every same-frame
        # detection of it). Runs at every step, scales O(n^2) over
        # surviving tracks; cheap for n < 50.
        # The held oid (when set) is protected — it is always chosen as
        # the keeper in any pair it participates in, so the held
        # identity is preserved across self-merges.
        # Protect the directly-grasped seed, not an arbitrary member of
        # the relation-expanded held set. ``next(iter(set))`` is non-
        # deterministic — at fr 519 it picked an apple instead of the
        # tray, so the merge dropped the actual held seed and broke
        # rigid attachment for every transitive held member.
        _held_for_merge = self._held_oid_now
        # Build protected-pairs set from the current scene graph: any
        # two oids the LLM links via "in"/"on" are distinct physical
        # objects and must NOT collapse via self-merge, even if their
        # centroids drift to within `self_merge_trans_m`.
        protected_pairs: Set[Tuple[int, int]] = set()
        for e in (relation_edges or ()):
            rel = getattr(e, "relation_type", None)
            if rel not in ("in", "on"):
                continue
            try:
                a, b = int(e.parent), int(e.child)
            except (TypeError, ValueError, AttributeError):
                continue
            if a == b:
                continue
            protected_pairs.add((min(a, b), max(a, b)))
        dbg["self_merge_protected_pairs"] = sorted(protected_pairs)
        dbg["self_merges"] = self._self_merge_pass(
            held_id=_held_for_merge,
            protected_pairs=protected_pairs)

        dbg["post_update_tracks"] = self._capture_tracks()

        # Expose per-track reference clouds (object-local frame) for the
        # ICP 3D-point visualization. Keyed by tracker oid.
        dbg["track_refs"] = {
            int(oid): np.asarray(ref.ref_points, dtype=np.float64).copy()
            for oid, ref in self.pose_est._refs.items()
            if getattr(ref, "ref_points", None) is not None
            and np.asarray(ref.ref_points).size > 0
        }

        self._frame_count += 1
        return dbg, dets_with_pose

    # ────────── self-merge ──────────
    def _self_merge_pass(self,
                          held_id: Optional[int] = None,
                          protected_pairs: Optional[Set[Tuple[int, int]]] = None,
                          ) -> List[Dict[str, Any]]:
        """Find pairs of same-label tracks whose belief means are within
        `self_merge_trans_m` metres of each other and merge them via
        Bayesian information fusion.

        Gate metric is Euclidean (not Mahalanobis) so the merge radius is
        invariant to how tight/loose the two tracks' covariances are; two
        fresh births at 22 cm cannot collapse just because σ = 5 cm.

        Greedy: visit candidate pairs sorted by ascending distance; merge
        each pair only if BOTH tracks still exist (not yet absorbed into
        a previous merge).

        `held_id`: if provided, this oid is *always* chosen as the
        keeper in any pair it's in, so its identity (and rigid-
        attachment kinematics) survives the merge.

        `protected_pairs`: unordered ``(min_oid, max_oid)`` tuples that
        the scene graph asserts are distinct physical objects (e.g. an
        apple resting on a tray). Such pairs are skipped — they should
        never collapse regardless of how close the centroids drift.
        """
        cfg = self.cfg
        gate_m = float(getattr(cfg, "self_merge_trans_m", 0.0))
        if gate_m <= 0.0:
            return []
        merges: List[Dict[str, Any]] = []
        oids = list(self.object_labels.keys())
        if len(oids) < 2:
            return merges

        # Build candidate pairs with Euclidean distance below the gate.
        candidates: List[Tuple[float, int, int, float]] = []
        beliefs = {oid: self.state.collapsed_object_base(oid) for oid in oids}
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
                # Scene-graph protection: pairs the LLM asserts are
                # related (apple-on-tray etc.) are distinct physical
                # objects and must not collapse — drop the pair from
                # the candidate list before we even check distance.
                if (protected_pairs is not None
                        and (min(oi, oj), max(oi, oj)) in protected_pairs):
                    continue
                pe_j = beliefs[oj]
                if pe_j is None:
                    continue
                nu_t = (np.asarray(pe_j.T)[:3, 3]
                        - np.asarray(pe_i.T)[:3, 3])
                dist = float(np.linalg.norm(nu_t))
                if dist > gate_m:
                    continue
                # d²_trans is reported as a diagnostic only (not gated on).
                S_tt = pe_i.cov[:3, :3] + pe_j.cov[:3, :3]
                try:
                    d2_t = float(nu_t @ np.linalg.solve(S_tt, nu_t))
                except np.linalg.LinAlgError:
                    d2_t = float("nan")
                candidates.append((dist, oi, oj, d2_t))

        candidates.sort(key=lambda t: t[0])
        merged_set: set = set()
        for dist, oi, oj, d2_t in candidates:
            if oi in merged_set or oj in merged_set:
                continue
            # Held track is always the keeper — its rigid-attachment
            # kinematics differ from a static track's, so collapsing it
            # into a static one would lose the motion model.
            if held_id is not None and oi == held_id:
                keep, drop = oi, oj
            elif held_id is not None and oj == held_id:
                keep, drop = oj, oi
            else:
                # Decide which to keep: the one with more chain entries
                # (longer history is more reliable). Tie-break: lower oid.
                chain_i = (len(self.chains.get(oi)) if self.chains.get(oi)
                           else 0)
                chain_j = (len(self.chains.get(oj)) if self.chains.get(oj)
                           else 0)
                if chain_i >= chain_j:
                    keep, drop = oi, oj
                else:
                    keep, drop = oj, oi
            ok = self.state.merge_tracks(keep, drop)
            if not ok:
                continue
            # Merge the bookkeeping side (existence, label history,
            # frames_since_obs, sam2_tau, observation chain).
            r_keep = self.existence.get(keep, 0.0)
            r_drop = self.existence.get(drop, 0.0)
            self.existence[keep] = max(r_keep, r_drop)
            self.frames_since_obs[keep] = min(
                self.frames_since_obs.get(keep, 0),
                self.frames_since_obs.get(drop, 0))
            # Merge label_scores: sum n_obs and weighted-average mean_score.
            for lbl, st in self.label_scores.get(drop, {}).items():
                cur = self.label_scores.setdefault(keep, {}).setdefault(
                    lbl, {"n_obs": 0, "mean_score": 0.0})
                n0 = int(cur["n_obs"]); m0 = float(cur["mean_score"])
                n1 = int(st["n_obs"]); m1 = float(st["mean_score"])
                tot = n0 + n1
                if tot > 0:
                    cur["n_obs"] = tot
                    cur["mean_score"] = (m0 * n0 + m1 * n1) / tot
            # Merge chains: append all of drop's entries to keep's chain.
            ch_drop = self.chains.get(drop)
            if ch_drop is not None:
                for e in ch_drop.entries:
                    self.chains.append(keep, e.frame, e.T_co, e.R_co,
                                         e.fitness, e.rmse)
            # Fold the absorbed track's ICP reference cloud into the
            # keeper's so the richer surface sample benefits the next
            # ICP. Fallback to dropping if merging fails.
            ref_drop = self.pose_est._refs.pop(drop, None)
            ref_keep = self.pose_est._refs.get(keep)
            if ref_drop is not None and ref_keep is not None:
                try:
                    # `ref_drop.ref_points` are in oid_drop's object-local
                    # frame; transform to keep's via
                    # T_obj_keep_from_obj_drop  = μ_keep^{-1} μ_drop.
                    # Since both are SE(3) means we pull from belief_keep
                    # and belief_drop (but drop is already merged, so
                    # use the d²_t=0 approximation: assume the frames
                    # coincide. This is exactly what triggered the
                    # self-merge in the first place.).
                    import open3d as _o3d
                    merged = np.vstack([ref_keep.ref_points,
                                          ref_drop.ref_points])
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
                    ref_keep.obj_radius = max(float(np.linalg.norm(
                        merged, axis=1).max()), 0.03)
                except Exception:
                    pass
            # Drop the absorbed track's bookkeeping.
            self.object_labels.pop(drop, None)
            self.frames_since_obs.pop(drop, None)
            self.existence.pop(drop, None)
            self.sam2_tau.pop(drop, None)
            self.label_scores.pop(drop, None)
            self.chains.delete(drop)
            merged_set.add(drop)
            merges.append({
                "keep_oid": int(keep),
                "drop_oid": int(drop),
                "dist_m": float(dist),
                "d2_trans": float(d2_t),
            })
        return merges


# ─────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────

def _overlay_detections(rgb: np.ndarray,
                         detections: List[Dict[str, Any]],
                         alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    for det in detections:
        oid = det.get("id")
        if oid is None:
            continue
        color = _palette_color(oid)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        mask = det.get("mask")
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mb = mask.astype(bool)
            if mb.any():
                colored = np.zeros_like(out)
                colored[mb] = color
                out = np.where(mb[..., None],
                                (alpha * colored + (1 - alpha) * out).astype(np.uint8),
                                out)
        bb = det.get("box")
        if bb is not None and len(bb) == 4:
            x0, y0, x1, y1 = map(int, bb)
            cv2.rectangle(out, (x0, y0), (x1, y1), color_bgr[::-1], 2)
            tag = f"id:{oid} {det.get('label','?')} s={det.get('score',0):.2f}"
            ty = max(y0 - 4, 12)
            cv2.rectangle(out, (x0, ty - 10), (x0 + 10 + 8 * len(tag), ty + 3),
                          (255, 255, 255), -1)
            cv2.putText(out, tag, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color_bgr[::-1], 1, cv2.LINE_AA)
    return out


def _plot_topdown(ax,
                  tracks: Dict[int, Dict[str, Any]],
                  dets_with_pose: List[Dict[str, Any]],
                  T_wb: np.ndarray,
                  xlim: Tuple[float, float],
                  ylim: Tuple[float, float],
                  title: str,
                  show_obs: bool = False) -> None:
    """Top-down scatter in the world frame.

    Each tracked object: filled circle at (x, y) + dashed uncertainty ellipse
    from the xy block of cov. Radius / ellipse scaled for visibility.

    If `show_obs` True, also draws detection centroids (from T_wb · T_co)
    as unfilled black squares, so you can see measurement vs. state.
    """
    # Camera frustum proxy: robot base location as a small triangle.
    bx, by = float(T_wb[0, 3]), float(T_wb[1, 3])
    # Heading = x-axis of the base in world.
    hx, hy = float(T_wb[0, 0]), float(T_wb[1, 0])
    theta = np.arctan2(hy, hx)

    ax.plot([bx], [by], marker="^", markersize=10,
            color="black", zorder=3)
    # Draw heading arrow.
    ax.annotate("", xy=(bx + 0.15 * np.cos(theta), by + 0.15 * np.sin(theta)),
                xytext=(bx, by),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Track circles + uncertainty. Use the EKF-composed world-frame
    # mean (`T_world`) for visual continuity. If an observation chain
    # is present, ALSO mark the chain-smoothed world-frame mean as a
    # cross -- when the two diverge, the filter has drifted relative
    # to the loop-closure-aware chain.
    for oid, tr in tracks.items():
        T = tr.get("T_world", tr["T"])
        cov = tr["cov"]
        r_ex = tr["r"]
        x, y = float(T[0, 3]), float(T[1, 3])
        col = _palette_color_f(oid)
        # Scale circle by existence (more opaque when confident).
        alpha = 0.25 + 0.65 * max(0.0, min(1.0, r_ex))
        ax.scatter([x], [y], s=120, c=[col], alpha=alpha,
                   edgecolors="black", linewidths=0.8, zorder=4)
        # Uncertainty ellipse (3sigma in xy).
        cov_xy = cov[:2, :2]
        try:
            w_eig, v_eig = np.linalg.eigh(cov_xy)
            w_eig = np.clip(w_eig, 1e-10, None)
            width, height = 2 * 3 * np.sqrt(w_eig)
            angle = np.degrees(np.arctan2(v_eig[1, 1], v_eig[0, 1]))
            ell = mpatches.Ellipse(
                (x, y), width=float(width), height=float(height),
                angle=float(angle),
                fill=False, edgecolor=col, lw=1.0, linestyle="--",
                alpha=0.7, zorder=2,
            )
            ax.add_patch(ell)
        except Exception:
            pass
        ax.text(x + 0.015, y + 0.015,
                f"id:{oid}\nr={r_ex:.2f}",
                fontsize=6.5, color="black",
                bbox=dict(facecolor="white", alpha=0.7, pad=0.8,
                          edgecolor="none"), zorder=5)

        # Chain-smoothed world-frame mean (loop-closure-aware).
        # When divergent from the EKF mean it signals filter drift.
        T_chain = tr.get("T_world_chain")
        if T_chain is not None:
            cx, cy = float(T_chain[0, 3]), float(T_chain[1, 3])
            ax.scatter([cx], [cy], s=70, marker="x",
                        color=col, linewidths=1.6, alpha=0.9, zorder=6)
            # Connect filter mean and chain mean if they're far enough
            # apart to be visible.
            if (cx - x) ** 2 + (cy - y) ** 2 > 1e-6:
                ax.plot([x, cx], [y, cy], color=col, linewidth=0.6,
                        alpha=0.5, zorder=2)

    if show_obs:
        for det in dets_with_pose:
            if not det.get("_icp_ok"):
                continue
            T_co = det["T_co"]
            T_wo = T_wb @ T_co
            dx, dy = float(T_wo[0, 3]), float(T_wo[1, 3])
            oid = det.get("id")
            col = _palette_color_f(oid) if oid is not None else (0, 0, 0)
            ax.scatter([dx], [dy], s=60, facecolors="none",
                       edgecolors=col, marker="s", linewidths=1.4, zorder=6)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)


def _format_assoc_matrix(assoc: Dict[str, Any],
                          max_rows: int = 10,
                          max_cols: int = 8) -> List[str]:
    """Render the Hungarian cost matrix, with row/col annotations and a
    star next to the matched cells.

    Cost convention (`association.hungarian_associate`, perception-style):
       cost[i, l] =   d^2[i, l]
                    - alpha * 1[sam2_tau matches]
                    + label_penalty * 1[d_label NOT in track i's label history]
                    + score_weight  * (1 - score_l)
                  = +INF (printed as INF) otherwise (d^2 > G_out, etc.)
    Negative cost = SAM2 tracklet-id bonus fired (alpha overshot d^2);
    INF = either no innovation, no ICP, or d^2 > G_out (=25).
    """
    track_oids = assoc.get("track_oids", [])
    track_labels = assoc.get("track_labels", []) or ["?"] * len(track_oids)
    track_label_hists = assoc.get("track_label_hists",
                                    [""] * len(track_oids))
    track_taus = assoc.get("track_taus", []) or [-1] * len(track_oids)
    cm = assoc.get("cost_matrix", [])
    det_meta = assoc.get("det_meta_in_assoc", [])
    if not track_oids or not cm:
        return ["  (no tracks or no detections this frame)"]
    match_local = {int(o): int(l)
                   for o, l in assoc.get("match_local", {}).items()}

    n_rows = len(track_oids)
    n_cols = len(cm[0]) if cm else 0
    rows_used = min(n_rows, max_rows)
    cols_used = min(n_cols, max_cols)

    out: List[str] = []
    alpha = assoc.get("alpha", 0.0)
    G_out = assoc.get("G_out", 25.0)
    lbl_pen = assoc.get("label_penalty", 0.0)
    sc_w = assoc.get("score_weight", 0.0)
    gate_mode = assoc.get("gate_mode", "full")
    G_out_trans = assoc.get("G_out_trans", 21.108)
    G_out_rot = assoc.get("G_out_rot", 21.108)
    cost_d2_mode = assoc.get("cost_d2_mode", "full")
    out.append(f"COST = d^2[{cost_d2_mode}] - alpha*1[tau match]"
               f" + lbl_pen*1[label miss] + sc_w*(1-score)")
    if gate_mode == "trans":
        gate_str = f"gate=trans  G_out_trans={G_out_trans:.2f}"
    elif gate_mode == "trans_and_rot":
        gate_str = (f"gate=trans+rot  G_out_trans={G_out_trans:.2f}  "
                    f"G_out_rot={G_out_rot:.2f}")
    else:
        gate_str = f"gate=full  G_out={G_out:.1f}"
    out.append(f"  alpha={alpha:.1f}  lbl_pen={lbl_pen:.1f}  "
               f"sc_w={sc_w:.1f}  {gate_str}  "
               f"(INF=infeasible; * = matched)")

    # Column header: per-det metadata
    col_w = 11
    header = "  oid lbl tau hist        |"
    for li in range(cols_used):
        m = det_meta[li] if li < len(det_meta) else {
            "sam2_id": "?", "label": "?", "global_idx": li, "score": 0.0}
        header += (f" d{m['global_idx']:>3}({m['sam2_id']}/{m['label'][:4]:<4}"
                   f"|{m.get('score', 0.0):.2f})")
    if n_cols > max_cols:
        header += f" +{n_cols - max_cols}"
    out.append(header)

    for ri in range(rows_used):
        oid = int(track_oids[ri])
        lbl = str(track_labels[ri])[:4]
        tau = track_taus[ri]
        hist = (track_label_hists[ri] if ri < len(track_label_hists)
                else "")[:11]
        row_str = f"  {oid:>3} {lbl:<4} {tau:>3} {hist:<11} |"
        for li in range(cols_used):
            c = cm[ri][li]
            star = "*" if match_local.get(oid) == li else " "
            if c >= 1e10:
                row_str += f"          INF{star}"
            else:
                row_str += f"      {c:>+7.2f}{star}"
        out.append(row_str)
    if n_rows > max_rows:
        out.append(f"  ... +{n_rows - max_rows} more tracks ...")
    return out


def _format_intermediates_text(dbg: Dict[str, Any]) -> str:
    """Compose the multi-line text block for panel 3."""
    lines: List[str] = []

    enter = dbg["enter_tracks"]
    post_p = dbg["post_predict_tracks"]
    post_u = dbg["post_update_tracks"]
    lines.append(f"tracks before -> after predict -> after update: "
                 f"{len(enter)} / {len(post_p)} / {len(post_u)}")

    # Predict deltas per track: tr(P) grows, T unchanged.
    lines.append("")
    lines.append("PREDICT  (tr P before -> after; T unchanged):")
    pred_rows = []
    for oid in sorted(enter.keys()):
        if oid not in post_p:
            continue
        trP0 = float(np.trace(enter[oid]["cov"]))
        trP1 = float(np.trace(post_p[oid]["cov"]))
        chain_n = int(post_p[oid].get("chain_len", 0))
        pred_rows.append(f"  id:{oid:<3d} tr(P): {trP0:.2e} -> {trP1:.2e}  chain_len={chain_n}")
    lines.extend(pred_rows[:8] if pred_rows
                 else ["  (no tracks)"])
    if len(pred_rows) > 8:
        lines.append(f"  ... +{len(pred_rows) - 8} more ...")

    # Association results.
    assoc = dbg.get("assoc", {})
    lines.append("")
    lines.append(f"ASSOC  n_tracks={len(assoc.get('track_oids', []))}  "
                 f"n_dets_icp={assoc.get('n_dets_for_assoc', 0)}  "
                 f"(total {assoc.get('n_dets_total', 0)})")
    m = assoc.get("match", {})
    lines.append(f"  matched: {len(m)}  "
                 f"unmatched_tr: {len(assoc.get('unmatched_tracks', []))}  "
                 f"unmatched_det: {len(assoc.get('unmatched_dets_local', []))}")
    lines.append("")
    lines.extend(_format_assoc_matrix(assoc))

    # Matched pairs table.
    lines.append("")
    lines.append("MATCHED (d^2 gate = 25):")
    matched = dbg.get("matched", [])
    if matched:
        lines.append("  id   det   d^2  d2_t  d2_r   w    logL     r_prev -> r_new  fit/rmse")
        for m_row in matched[:12]:
            flag = "[REJ]" if m_row.get("reject_outer_gate") else "     "
            d2t = m_row.get("d2_trans", float("nan"))
            d2r = m_row.get("d2_rot", float("nan"))
            lines.append(
                f"  {m_row['oid']:<3d} {m_row['det_idx']:<3d}  "
                f"{m_row['d2']:5.2f} {d2t:5.2f} {d2r:6.2f} "
                f"{m_row['w']:4.2f} {m_row['log_lik']:7.1f}   "
                f"{m_row['r_prev']:.3f} -> {m_row['r_new']:.3f}  "
                f"{m_row['fitness']:.2f}/{m_row['rmse']*1e3:4.1f}mm {flag}"
            )
    else:
        lines.append("  (none)")

    # Missed branch.
    lines.append("")
    lines.append("MISSED (eq:r_miss):")
    missed = dbg.get("missed", [])
    if missed:
        lines.append("  id   p_v   p~_d   r_prev -> r_new")
        for m_row in missed[:10]:
            lines.append(
                f"  {m_row['oid']:<3d}  {m_row['p_v']:.2f}  "
                f"{m_row['p_d_tilde']:.2f}   "
                f"{m_row['r_prev']:.3f} -> {m_row['r_new']:.3f}"
            )
    else:
        lines.append("  (none)")

    # Births + prunes.
    births = dbg.get("births", [])
    prunes = dbg.get("pruned", [])
    lines.append("")
    lines.append(f"BIRTHS: {len(births)}    PRUNES: {len(prunes)}")
    for b in births[:8]:
        lines.append(f"  +id:{b['new_oid']} {b['label']} s={b['score']:.2f} "
                     f"r_new={b['r_new']:.3f}")
    for p in prunes[:8]:
        lines.append(f"  -id:{p['oid']} r={p['r']:.2e}")

    return "\n".join(lines)


def _plot_intermediates(ax, dbg: Dict[str, Any]) -> None:
    ax.axis("off")
    txt = _format_intermediates_text(dbg)
    ax.text(0.0, 1.0, txt, transform=ax.transAxes,
            fontsize=6.8, family="monospace",
            verticalalignment="top", horizontalalignment="left")
    ax.set_title("[3] EKF step intermediates", fontsize=10)


def _plot_r_evolution(ax,
                       r_history: Dict[int, List[Tuple[int, float]]],
                       xlim: Tuple[int, int]) -> None:
    ax.set_title("[6] existence r(t) per track", fontsize=10)
    any_plotted = False
    for oid, hist in r_history.items():
        if not hist:
            continue
        xs = [h[0] for h in hist]
        ys = [h[1] for h in hist]
        ax.plot(xs, ys, marker=".", markersize=3, linewidth=1.0,
                color=_palette_color_f(oid), label=f"id:{oid}")
        any_plotted = True
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(xlim)
    ax.set_xlabel("frame", fontsize=8)
    ax.set_ylabel("r", fontsize=8)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if any_plotted:
        # Unique handles (matplotlib legend picks duplicates up otherwise).
        handles, labels = ax.get_legend_handles_labels()
        seen_l = []
        seen_h = []
        for h, l in zip(handles, labels):
            if l not in seen_l:
                seen_l.append(l)
                seen_h.append(h)
        if seen_l:
            ax.legend(seen_h[:12], seen_l[:12], fontsize=6, ncol=2,
                      loc="lower left", framealpha=0.85)


def _compute_topdown_extent(tracks_snapshots: List[Dict[int, Dict[str, Any]]],
                             dets_with_pose: List[Dict[str, Any]],
                             T_wb: np.ndarray,
                             pad: float = 0.4) -> Tuple[Tuple[float, float],
                                                         Tuple[float, float]]:
    xs: List[float] = [float(T_wb[0, 3])]
    ys: List[float] = [float(T_wb[1, 3])]
    for snap in tracks_snapshots:
        for oid, tr in snap.items():
            T_use = tr.get("T_world", tr["T"])
            xs.append(float(T_use[0, 3]))
            ys.append(float(T_use[1, 3]))
    for det in dets_with_pose:
        if det.get("_icp_ok"):
            T_wo = T_wb @ det["T_co"]
            xs.append(float(T_wo[0, 3]))
            ys.append(float(T_wo[1, 3]))
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    # Enforce a minimum span so an almost-stationary camera doesn't look
    # infinitely zoomed.
    min_span = 0.6
    if x1 - x0 < min_span:
        c = 0.5 * (x0 + x1)
        x0, x1 = c - min_span / 2, c + min_span / 2
    if y1 - y0 < min_span:
        c = 0.5 * (y0 + y1)
        y0, y1 = c - min_span / 2, c + min_span / 2
    return (x0, x1), (y0, y1)


def _track_to_jsonable(tr: Dict[str, Any]) -> Dict[str, Any]:
    """One track snapshot → JSON-serialisable dict (matrices to lists).

    `T`, `cov`, `xyz` are BASE frame (the recursion's storage frame).
    `T_world`, `xyz_w` are derived for visualization only via composition
    with the current SLAM `T_wb` (no Σ_wb propagation -- not the EKF's
    cov anymore).
    """
    out = {
        "T": tr["T"].tolist(),
        "cov": tr["cov"].tolist(),
        "label": tr["label"],
        "label_scores": tr.get("label_scores", {}),
        "r": float(tr["r"]),
        "frames_since_obs": int(tr["frames_since_obs"]),
        "sam2_tau": int(tr["sam2_tau"]),
        "xyz": [float(tr["T"][0, 3]),
                float(tr["T"][1, 3]),
                float(tr["T"][2, 3])],
        "tr_cov": float(np.trace(tr["cov"])),
    }
    T_world = tr.get("T_world")
    if T_world is not None:
        out["T_world"] = T_world.tolist() if hasattr(T_world, "tolist") \
            else T_world
        out["xyz_w"] = [float(T_world[0, 3]),
                        float(T_world[1, 3]),
                        float(T_world[2, 3])]
    T_world_chain = tr.get("T_world_chain")
    if T_world_chain is not None:
        out["T_world_chain"] = T_world_chain.tolist() if hasattr(
            T_world_chain, "tolist") else T_world_chain
        out["xyz_w_chain"] = [float(T_world_chain[0, 3]),
                                float(T_world_chain[1, 3]),
                                float(T_world_chain[2, 3])]
    cov_world_chain = tr.get("cov_world_chain")
    if cov_world_chain is not None:
        out["cov_world_chain"] = cov_world_chain.tolist() if hasattr(
            cov_world_chain, "tolist") else cov_world_chain
    out["chain_len"] = int(tr.get("chain_len", 0))
    out["chain_n_used"] = int(tr.get("chain_n_used", 0))
    return out


def _dump_frame_json(out_path: str,
                      dbg: Dict[str, Any],
                      detections_raw: List[Dict[str, Any]],
                      dets_with_pose: List[Dict[str, Any]]) -> None:
    """Save the full per-frame EKF state for offline diagnosis.

    Schema:
      frame, slam_pose, detections (raw + ICP outputs), tracks at three
      snapshots (enter / post_predict / post_update), association
      (cost matrix, match dict, unmatched lists), visibility (per-oid p_v),
      matched_events, missed_events, births, prunes.

    Mask payloads from detections are NOT included (would balloon the file).
    """
    T_wb = dbg["slam_pose"]
    T_bc = np.asarray(dbg.get("T_bc", np.eye(4)), dtype=np.float64)

    # Detections: keep id/label/score/box + ICP outputs + world-frame T_wo
    # for easy diagnosis. Drop the binary mask. Iterate over
    # `dets_with_pose` (post-suppression) — each entry already carries
    # the raw fields (id/label/score/box/labels) plus the pose outputs,
    # so id and pose are guaranteed to come from the same record.
    # `detections_raw` is no longer joined here; it is still consumed
    # by the events panel for subpart-absorbed pid lookups.
    det_records: List[Dict[str, Any]] = []
    for i, dwp in enumerate(dets_with_pose):
        rec: Dict[str, Any] = {
            "global_idx": i,
            "id": int(dwp.get("id")) if dwp.get("id") is not None else None,
            "label": dwp.get("label"),
            "labels": dwp.get("labels", {}),
            "score": float(dwp.get("score", 0.0)),
            "mean_score": float(dwp.get("mean_score", 0.0)),
            "n_obs": int(dwp.get("n_obs", 0)),
            "box": dwp.get("box"),
            "_icp_ok": bool(dwp.get("_icp_ok", False)),
            "fitness": float(dwp.get("fitness", 0.0)),
            "rmse": float(dwp.get("rmse", 0.0)),
        }
        if dwp.get("_icp_ok"):
            T_co = np.asarray(dwp["T_co"], dtype=np.float64)
            # T_co is camera-frame; lift through T_bc to base, then T_wb
            # to world. Previously was T_wb @ T_co (missing T_bc) which
            # produced an xyz_w shifted by ~|t_bc| from the actual world
            # position the panels render via _centroid_cam.
            T_wo = T_wb @ T_bc @ T_co
            rec["T_co"] = T_co.tolist()
            rec["R_icp"] = np.asarray(dwp["R_icp"], dtype=np.float64).tolist()
            rec["T_wo"] = T_wo.tolist()
            rec["xyz_w"] = [float(T_wo[0, 3]),
                            float(T_wo[1, 3]),
                            float(T_wo[2, 3])]
        det_records.append(rec)

    payload = {
        "frame": int(dbg["frame"]),
        "slam_pose": T_wb.tolist(),
        "T_bc": np.asarray(dbg["T_bc"]).tolist() if "T_bc" in dbg else None,
        "gripper_state": dbg.get("gripper_state", {}),
        "held_oids_used": list(dbg.get("held_oids_used", [])),
        "relations": list(dbg.get("relations", [])),
        "relation_call": dict(dbg.get("relation_call", {})),
        "detections": det_records,
        "tracks_enter": {str(oid): _track_to_jsonable(tr)
                          for oid, tr in dbg["enter_tracks"].items()},
        "tracks_post_predict": {str(oid): _track_to_jsonable(tr)
                                 for oid, tr in dbg["post_predict_tracks"].items()},
        "tracks_post_update": {str(oid): _track_to_jsonable(tr)
                                for oid, tr in dbg["post_update_tracks"].items()},
        "association": dbg.get("assoc", {}),
        "matched_events": dbg.get("matched", []),
        "missed_events": dbg.get("missed", []),
        "births": dbg.get("births", []),
        "birth_rejects": dbg.get("birth_rejects", []),
        "subpart_absorbed": dbg.get("subpart_absorbed", []),
        "centroid_dropped": dbg.get("centroid_dropped", []),
        "prunes": dbg.get("pruned", []),
        "self_merges": dbg.get("self_merges", []),
        "self_merge_protected_pairs": dbg.get("self_merge_protected_pairs", []),
        "visibility": {int(o): float(v)
                        for o, v in dbg.get("visibility", {}).items()},
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────
# Nine per-step clean panels
# ─────────────────────────────────────────────────────────────────────────

# Fixed world extents for ALL top-down panels. 90° CW rotated: the plot's
# horizontal axis shows world_y (natural +y on the right, range [-2, +2])
# and the vertical axis shows world_x with the direction reversed so that
# world_x = 0 is at the BOTTOM and world_x = -4 is at the TOP. Objects
# ahead of the robot (more negative world_x) therefore sit higher on the
# page -- a driver's-view top-down.
_WORLD_X_RANGE = (-4.0, 0.0)
_WORLD_Y_RANGE = (-2.0, 3.0)


def _set_topdown_axes(ax, title):
    """Shared rotated top-down axis formatting. World y goes left-to-right,
    world x goes bottom-to-top (with 0 at the bottom = robot-side)."""
    ax.set_xlim(_WORLD_Y_RANGE)
    ax.set_ylim(0.0, _WORLD_X_RANGE[0])    # (0, -4): world_x=0 at bottom
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("world y [m]", fontsize=8)
    ax.set_ylabel("world x [m]", fontsize=8)
    ax.tick_params(labelsize=7)


def _draw_base_marker(ax, T_wb, color="k", size=90):
    """Robot base + heading arrow on the rotated top-down view."""
    x = float(T_wb[0, 3]); y = float(T_wb[1, 3])
    hx, hy = float(T_wb[0, 0]), float(T_wb[1, 0])   # base +x in world
    # Plot coords: (horizontal, vertical) = (world_y, world_x).
    ax.scatter([y], [x], marker="s", c=color, s=size,
               edgecolors="white", linewidths=1.2, zorder=10, label="base")
    ax.annotate("", xy=(y + 0.22 * hy, x + 0.22 * hx), xytext=(y, x),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
                zorder=10)


def _cov_ellipse_xy(ax, mean_xy, cov_xy, color, n_std=2.0, lw=1.0,
                      alpha=0.25):
    """Draw a 2-σ covariance ellipse on the rotated top-down.

    ``mean_xy = (world_x, world_y)`` and ``cov_xy`` is the 2x2 xy marginal
    in world frame. The plot uses (world_y, world_x) coordinates so we
    permute rows/columns of ``cov_xy`` before computing the ellipse.
    """
    cov = np.asarray(cov_xy, dtype=np.float64)
    # Permute to (y, x) basis: P = [[0,1],[1,0]]; cov_p = P cov P^T.
    cov_p = np.array([[cov[1, 1], cov[1, 0]],
                      [cov[0, 1], cov[0, 0]]], dtype=np.float64)
    vals, vecs = np.linalg.eigh(cov_p)
    vals = np.clip(vals, 0.0, None)
    width, height = 2.0 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    # Centre in plot coords = (world_y, world_x).
    ell = mpatches.Ellipse(xy=(float(mean_xy[1]), float(mean_xy[0])),
                           width=max(width, 1e-4),
                           height=max(height, 1e-4), angle=angle,
                           facecolor=color, edgecolor=color,
                           alpha=alpha, lw=lw, zorder=3)
    ax.add_patch(ell)


# ─── Hungarian cost-matrix helpers (≥5×5 padding, shared drawing) ──────

_HUNG_MIN = 5


def _pad_matrix_min(mat: np.ndarray,
                    track_oids: List[int],
                    det_meta: List[Dict[str, Any]],
                    min_size: int = _HUNG_MIN):
    """Pad a (n_tr, n_det) matrix to at least (min_size, min_size) with
    NaNs. Returns (padded_matrix, padded_track_labels, padded_det_labels)
    where labels for padding slots are ``None``.
    """
    mat = np.asarray(mat, dtype=np.float64) if mat is not None \
        else np.zeros((0, 0), dtype=np.float64)
    if mat.ndim != 2:
        mat = np.zeros((0, 0), dtype=np.float64)
    n_tr, n_det = mat.shape if mat.size else (0, 0)
    n_tr = max(n_tr, len(track_oids))
    n_det = max(n_det, len(det_meta))
    tgt_r = max(n_tr, min_size)
    tgt_c = max(n_det, min_size)
    padded = np.full((tgt_r, tgt_c), np.nan, dtype=np.float64)
    if mat.size:
        r, c = mat.shape
        padded[:r, :c] = mat
    track_labels = list(track_oids) + [None] * (tgt_r - len(track_oids))
    det_labels = list(det_meta) + [None] * (tgt_c - len(det_meta))
    return padded, track_labels, det_labels


def _draw_hungarian_matrix(ax, dbg, mat, title, *,
                           highlight_matches: bool = False,
                           value_fmt: str = ".2f"):
    """Render one Hungarian cost-component matrix, padded to ≥5×5.

    NaN cells render empty; ``>= 1e11`` cells (infeasible) render as "∞".
    """
    assoc = dbg.get("assoc", {})
    track_oids = [int(o) for o in assoc.get("track_oids", [])]
    det_meta = list(assoc.get("det_meta_in_assoc", []))
    match_local = assoc.get("match_local", {})

    mat_p, track_labels, det_labels = _pad_matrix_min(
        mat, track_oids, det_meta, min_size=_HUNG_MIN)

    # Clip infeasible cells for the colormap.
    mat_view = np.where(mat_p >= 1e11, np.nan, mat_p)
    ax.imshow(mat_view, cmap="viridis_r", aspect="auto")

    ax.set_xticks(range(len(det_labels)))
    ax.set_yticks(range(len(track_labels)))
    # Columns = detections in association order.  Label as ``pid:X``
    # using the per-column ``sam2_id`` snapshot (= the detection's
    # perception id), so the matrix's columns line up with the
    # ``pid:X`` labels in the [1A] Detected panel.
    ax.set_xticklabels([
        (f"pid:{m.get('sam2_id')}"
         if isinstance(m, dict) and m.get("sam2_id") is not None
         else "—")
        for m in det_labels
    ], fontsize=7)
    # Rows = live tracks. Label as ``oid:Y`` for parity with [1B] /
    # [1C].
    ax.set_yticklabels([
        (f"oid:{o}" if o is not None else "—") for o in track_labels
    ], fontsize=7)

    for i in range(mat_p.shape[0]):
        for j in range(mat_p.shape[1]):
            v = mat_p[i, j]
            if np.isnan(v):
                continue
            if v >= 1e11 or not np.isfinite(v):
                ax.text(j, i, "∞", ha="center", va="center",
                        fontsize=6, color="lightgray")
            else:
                ax.text(j, i, format(v, value_fmt), ha="center",
                        va="center", fontsize=6, color="white")

    if highlight_matches:
        for oid_str, l_local in match_local.items():
            oid = int(oid_str)
            if oid in track_oids:
                i = track_oids.index(oid)
                rect = mpatches.Rectangle(
                    (int(l_local) - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="red", lw=1.8, zorder=5)
                ax.add_patch(rect)
    ax.set_title(title, fontsize=10)


def _backproject_mask(mask: np.ndarray,
                      depth: np.ndarray,
                      K: np.ndarray,
                      *, max_samples: int = 800) -> np.ndarray:
    """(N, 3) camera-frame points from a binary mask + depth image."""
    if mask is None or depth is None:
        return np.zeros((0, 3), dtype=np.float64)
    H, W = depth.shape[:2]
    ys, xs = np.where((mask > 0) & np.isfinite(depth)
                       & (depth > 0.1) & (depth < 10.0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(xs) > max_samples:
        idx = np.linspace(0, len(xs) - 1, max_samples).astype(np.int64)
        xs = xs[idx]; ys = ys[idx]
    zs = depth[ys, xs].astype(np.float64)
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    Xs = (xs.astype(np.float64) - cx) * zs / fx
    Ys = (ys.astype(np.float64) - cy) * zs / fy
    return np.stack([Xs, Ys, zs], axis=1)


def _plot_step0_slam(ax, dbg, T_wb):
    """Step 0: SLAM ingest + per-frame T_bc. Base + camera on the rotated
    world top-down (fixed extent)."""
    _set_topdown_axes(ax, "[0] SLAM ingest  (base + camera)")
    T_bc = np.asarray(dbg.get("T_bc", np.eye(4)))
    T_wc = T_wb @ T_bc
    bx, by = float(T_wb[0, 3]), float(T_wb[1, 3])
    cx, cy, cz = float(T_wc[0, 3]), float(T_wc[1, 3]), float(T_wc[2, 3])

    _draw_base_marker(ax, T_wb, color="black", size=160)

    # Camera: dotted line to base + optical-axis arrow.
    ax.plot([by, cy], [bx, cx], linestyle=":", color="gray", lw=1.2,
            zorder=4)
    ax.scatter([cy], [cx], marker="^", c="tab:blue", s=140,
               edgecolors="white", linewidths=1.2, zorder=11)
    ox, oy = float(T_wc[0, 2]), float(T_wc[1, 2])   # optical-axis xy
    ax.annotate("", xy=(cy + 0.50 * oy, cx + 0.50 * ox),
                xytext=(cy, cx),
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.6),
                zorder=11)

    oz = float(T_wc[2, 2])
    tilt_deg = np.degrees(np.arccos(np.clip(-oz, -1.0, 1.0)))
    info = (f"base=({bx:+.2f},{by:+.2f})m  "
            f"cam_h={cz:.2f}m  tilt={tilt_deg:.0f}°")
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            fontsize=7, va="top", color="black",
            bbox=dict(facecolor="white", alpha=0.85, pad=2,
                       edgecolor="none"))


def _iter_tracks_xy(tracks_snap, T_wb):
    for oid, tr in tracks_snap.items():
        T_world = tr.get("T_world")
        if T_world is None:
            T_world = T_wb @ np.asarray(tr["T"])
        yield int(oid), (float(T_world[0, 3]), float(T_world[1, 3]))


def _plot_step1_predict(ax, dbg, T_wb):
    """Step 1: EKF predict. Predicted means + 2σ covariance ellipses."""
    _set_topdown_axes(ax, "[1] EKF predict  (μ + 2σ cov)")
    _draw_base_marker(ax, T_wb, color="lightgray")

    for oid, (x, y) in _iter_tracks_xy(
            dbg.get("post_predict_tracks", {}), T_wb):
        col = _palette_color_f(oid)
        # Plot coords: horizontal = world_y, vertical = world_x.
        ax.scatter([y], [x], s=50, color=col,
                   edgecolors="white", linewidths=1.0, zorder=5)
        tr = dbg["post_predict_tracks"][oid]
        cov = np.asarray(tr["cov"])[:2, :2]
        _cov_ellipse_xy(ax, (x, y), cov, color=col, n_std=2.0, alpha=0.18)
        ax.text(y, x, f"  {oid}", color=col, fontsize=7,
                va="center", ha="left", zorder=6)


def _build_pid_to_oid(dbg, dets_with_pose) -> Tuple[Dict[Any, int], set]:
    """Return (pid → oid map, held_set) from this frame's dbg + the
    POST-suppression detections (``dets_with_pose``).

    The map is built from `dbg["assoc"]["match"]` plus
    `det_indices_in_assoc`. Both index `dets_with_pose`, NOT the raw
    perception list — passing raw `detections` here used to mis-map
    pids whenever subpart-suppression dropped a det earlier in the
    list. Held set comes from `dbg["held_oids_used"]`
    (relation-expanded) or `dbg["gripper_state"]["held_obj_id"]`.
    """
    pid_to_oid: Dict[Any, int] = {}
    held_set: set = set()
    if dbg is None:
        return pid_to_oid, held_set
    assoc = dbg.get("assoc") or dbg.get("association") or {}
    det_idx_in_assoc = assoc.get("det_indices_in_assoc") or []
    match = assoc.get("match") or {}
    for oid_key, l_local in match.items():
        try:
            l_local = int(l_local)
            oid_int = int(oid_key)
        except (TypeError, ValueError):
            continue
        if 0 <= l_local < len(det_idx_in_assoc):
            l_global = int(det_idx_in_assoc[l_local])
            if 0 <= l_global < len(dets_with_pose):
                pid_to_oid[dets_with_pose[l_global].get("id")] = oid_int
    gs = dbg.get("gripper_state") or {}
    h = gs.get("held_obj_id")
    if h is not None:
        try:
            held_set = {int(h)}
        except (TypeError, ValueError):
            pass
    held_used = dbg.get("held_oids_used")
    if held_used:
        held_set = {int(o) for o in held_used}
    return pid_to_oid, held_set


def _plot_detected_pid(ax, rgb, detections, dbg=None):
    """[1A] Perception-side view: each mask labelled ``pid:X (label)``.

    Color is keyed on the perception id so the same SAM2 mask keeps
    its hue across frames (until SAM2 reseeds it). Pre-Hungarian
    rejects (subpart-absorbed, centroid-dropped) get a dimmed,
    cross-hatched style and a suffix in the label so the user can see
    *what* perception emitted vs. what reached the matcher.
    """
    ax.imshow(rgb)
    ax.set_title(f"[1A] Detected (pid)  ·  {len(detections)} dets",
                  fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Build pid → reject-reason maps from dbg.
    absorbed_pid_to_into: Dict[Any, Any] = {}
    if dbg is not None:
        for sa in dbg.get("subpart_absorbed", []) or []:
            try:
                fi = int(sa.get("from_idx"))
                ii = int(sa.get("into_idx"))
                pid_from = detections[fi].get("id")
                pid_into = detections[ii].get("id")
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if pid_from is not None:
                absorbed_pid_to_into[pid_from] = pid_into
    cent_dropped_pids = set()
    if dbg is not None:
        for cd in dbg.get("centroid_dropped", []) or []:
            pid = cd.get("pid")
            if pid is not None:
                cent_dropped_pids.add(pid)

    for det in detections:
        pid = det.get("id")
        mask = det.get("mask")
        if mask is None or not np.any(mask):
            continue
        ys, xs = np.where(mask > 0)
        cu, cv = float(xs.mean()), float(ys.mean())

        is_absorbed = pid in absorbed_pid_to_into
        is_cdrop = pid in cent_dropped_pids
        if is_absorbed:
            into_pid = absorbed_pid_to_into[pid]
            col = (0.55, 0.45, 0.30)  # muted brown
            ax.scatter(xs, ys, s=1, c=[col], alpha=0.08,
                        marker="x", linewidths=0)
            ax.scatter([cu], [cv], marker="x", c=[col], s=60,
                        linewidths=1.2, zorder=5)
            label = f"pid:{pid} (absorbed→{into_pid})"
            ax.text(cu + 6, cv - 6, label, color=col, fontsize=7,
                    weight="bold", zorder=5)
            continue
        if is_cdrop:
            col = (0.55, 0.55, 0.55)  # grey
            ax.scatter(xs, ys, s=1, c=[col], alpha=0.08,
                        marker="x", linewidths=0)
            ax.scatter([cu], [cv], marker="x", c=[col], s=60,
                        linewidths=1.2, zorder=5)
            label = f"pid:{pid} (no-depth)"
            ax.text(cu + 6, cv - 6, label, color=col, fontsize=7,
                    weight="bold", zorder=5)
            continue

        col = _palette_color_f(pid) if pid is not None else (0, 0, 0)
        ax.scatter(xs, ys, s=1, c=[col], alpha=0.18, marker=".",
                    linewidths=0)
        ax.scatter([cu], [cv], marker="o", c=[col], s=60,
                    edgecolors="white", linewidths=1.2, zorder=5)
        label = f"pid:{pid}  ({det.get('label', '?')})"
        ax.text(cu + 6, cv - 6, label, color=col, fontsize=7,
                weight="bold", zorder=5)


def _plot_tracked_oid(ax, rgb, dets_with_pose, dbg, K=None):
    """[1B] EKF-tracker view: visualise the d²_trans geometry directly.

    Each detection's perception centroid (back-projected from depth,
    re-projected to image) is drawn as a coloured circle. Each
    tracker oid's predicted μ_b (transformed to camera, projected)
    is drawn as a square. Matched (oid, pid) pairs share a colour
    and are joined by a dashed line annotated ``d²t=N.NN`` — the
    line length is the Euclidean piece of what Hungarian minimises.

    Unmatched detections: gray circle, ``[no-track]``.
    Unmatched tracker oids: open square in the oid's palette colour,
    ``[no-match]``. Held-set oids get a yellow square edge.

    Note: in trajectories where perception never produces the label
    "tray" (e.g. apple_in_the_tray), the tray is born as an
    ``apple`` track — so seeing oid 1 (apple) match apple detections
    is the correct behaviour given the labels coming in. Fixing it
    requires perception-side label correction, not a tracker change.
    """
    ax.imshow(rgb)
    ax.set_title(
        "[1B] Tracked  ·  ○ perception · □ EKF projected · — — d²_trans",
        fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    pid_to_oid, held_set = _build_pid_to_oid(dbg, dets_with_pose)
    H_img, W_img = rgb.shape[:2]

    if K is None:
        return

    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx_K = float(K[0, 2]); cy_K = float(K[1, 2])

    def _project_cam(p_c):
        """Camera-frame XYZ → pixel (u, v) or None if behind camera."""
        if p_c is None:
            return None
        Z = float(p_c[2])
        if Z <= 0.05:
            return None
        return (fx * float(p_c[0]) / Z + cx_K,
                fy * float(p_c[1]) / Z + cy_K)

    margin_px = 50.0
    u_min, u_max = margin_px, W_img - margin_px
    v_min, v_max = margin_px, H_img - margin_px
    cx_img, cy_img = W_img * 0.5, H_img * 0.5

    def _clamp_to_inset(p):
        """Off-image points get pushed onto the inset rectangle along
        the ray from image centre — preserves direction so multiple
        off-screen tracks spread along the boundary."""
        u, v = float(p[0]), float(p[1])
        if u_min <= u <= u_max and v_min <= v <= v_max:
            return (u, v)
        du = u - cx_img; dv = v - cy_img
        t_candidates = [1.0]
        if du > 1e-6:
            t_candidates.append((u_max - cx_img) / du)
        elif du < -1e-6:
            t_candidates.append((u_min - cx_img) / du)
        if dv > 1e-6:
            t_candidates.append((v_max - cy_img) / dv)
        elif dv < -1e-6:
            t_candidates.append((v_min - cy_img) / dv)
        t = min(t for t in t_candidates if t > 0.0)
        return (cx_img + t * du, cy_img + t * dv)

    assoc = dbg.get("assoc") or dbg.get("association") or {}
    track_oids = assoc.get("track_oids") or []
    track_labels = assoc.get("track_labels") or []
    label_by_oid = {int(o): (str(track_labels[i])
                              if i < len(track_labels) else "?")
                     for i, o in enumerate(track_oids)}
    label_by_pid = {d.get("id"): d.get("label", "?")
                     for d in dets_with_pose
                     if d.get("id") is not None}

    # ── d²t per oid for the dashed-line annotation ────────────────
    d2_by_oid: Dict[int, float] = {}
    for ev in dbg.get("matched", []) or []:
        if (ev.get("reject_outer_gate")
                or ev.get("reject_held_prefilter")
                or ev.get("reject_innov_clamp")):
            continue
        try:
            d2_by_oid[int(ev.get("oid", -1))] = float(
                ev.get("d2_trans", float("nan")))
        except (TypeError, ValueError):
            continue

    # ── Project all detection centroids (Hungarian's measurement) ─
    det_pix: Dict[Any, Optional[Tuple[float, float]]] = {}
    for det in dets_with_pose:
        pid = det.get("id")
        if pid is None or not det.get("_centroid_ok"):
            continue
        det_pix[pid] = _project_cam(det.get("_centroid_cam"))

    # ── Project all tracker μ_b (Hungarian's prior) ───────────────
    T_bc_arr = np.asarray(dbg.get("T_bc", np.eye(4)), dtype=np.float64)
    try:
        T_cb = np.linalg.inv(T_bc_arr)
    except np.linalg.LinAlgError:
        return
    post_predict = dbg.get("post_predict_tracks") or {}
    trk_pix: Dict[int, Optional[Tuple[float, float]]] = {}
    for oid_raw in track_oids:
        oid = int(oid_raw)
        tr = post_predict.get(oid) or post_predict.get(str(oid))
        if tr is None:
            continue
        T = np.asarray(tr.get("T", np.eye(4)), dtype=np.float64)
        if T.shape != (4, 4):
            continue
        mu_b = T[:3, 3]
        mu_c = T_cb @ np.array([mu_b[0], mu_b[1], mu_b[2], 1.0],
                                  dtype=np.float64)
        trk_pix[oid] = _project_cam(mu_c[:3])

    # ── Faint mask scatter for matched dets (background context) ──
    for det in dets_with_pose:
        pid = det.get("id")
        oid = pid_to_oid.get(pid)
        if oid is None:
            continue
        mask = det.get("mask")
        if mask is None or not np.any(mask):
            continue
        ys, xs = np.where(mask > 0)
        col = _palette_color_f(int(oid))
        ax.scatter(xs, ys, s=1, c=[col], alpha=0.10, marker=".",
                    linewidths=0)

    # ── Matched pairs: circle + square + dashed line + d²t ────────
    matched_oids: set = set()
    for pid, oid in pid_to_oid.items():
        oid_i = int(oid)
        col = _palette_color_f(oid_i)
        edge_col = "yellow" if oid_i in held_set else "white"
        edge_lw = 2.4 if edge_col == "yellow" else 1.2
        cir = det_pix.get(pid)
        sq = trk_pix.get(oid_i)
        if cir is not None:
            ax.scatter([cir[0]], [cir[1]], marker="o", c=[col], s=70,
                        edgecolors="white", linewidths=1.2, zorder=6)
            ax.text(cir[0] + 5, cir[1] - 5,
                    f"pid:{pid} ({label_by_pid.get(pid, '?')})",
                    color=col, fontsize=7, weight="bold", zorder=6)
        if sq is not None:
            sq_c = _clamp_to_inset(sq)
            ax.scatter([sq_c[0]], [sq_c[1]], marker="s", c=[col],
                        s=80, edgecolors=edge_col, linewidths=edge_lw,
                        zorder=6)
            ax.text(sq_c[0] + 5, sq_c[1] + 11,
                    f"oid:{oid_i} ({label_by_oid.get(oid_i, '?')})",
                    color=col, fontsize=7, weight="bold", zorder=6)
        if cir is not None and sq is not None:
            sq_c = _clamp_to_inset(sq)
            ax.plot([cir[0], sq_c[0]], [cir[1], sq_c[1]], "--",
                    color=col, lw=1.2, alpha=0.85, zorder=5)
            d2 = d2_by_oid.get(oid_i)
            if d2 is not None and d2 == d2:
                mx = (cir[0] + sq_c[0]) * 0.5
                my = (cir[1] + sq_c[1]) * 0.5
                ax.text(mx, my, f"d²t={d2:.1f}",
                        color=col, fontsize=7, weight="bold",
                        ha="center", va="center",
                        bbox=dict(facecolor="white",
                                   edgecolor="none", alpha=0.55,
                                   pad=1.0),
                        zorder=7)
        matched_oids.add(oid_i)

    # ── Unmatched detections — gray circle ────────────────────────
    for pid, p in det_pix.items():
        if pid in pid_to_oid:
            continue
        if p is None:
            continue
        ax.scatter([p[0]], [p[1]], marker="o",
                    c=[(0.55, 0.55, 0.55)], s=50,
                    edgecolors="white", linewidths=0.8,
                    alpha=0.75, zorder=5)
        ax.text(p[0] + 5, p[1] - 5,
                f"pid:{pid} ({label_by_pid.get(pid, '?')}) [no-track]",
                color=(0.55, 0.55, 0.55), fontsize=7, weight="bold",
                zorder=5)

    # ── Unmatched tracker oids — open square + label ──────────────
    for oid, p in trk_pix.items():
        if oid in matched_oids:
            continue
        if p is None:
            continue
        p_c = _clamp_to_inset(p)
        col = _palette_color_f(oid)
        edge_col = "yellow" if oid in held_set else col
        edge_lw = 2.4 if edge_col == "yellow" else 2.0
        ax.scatter([p_c[0]], [p_c[1]], marker="s",
                    facecolors="none", edgecolors=edge_col,
                    s=110, linewidths=edge_lw, zorder=6)
        ax.text(p_c[0] + 5, p_c[1] + 11,
                f"oid:{oid} ({label_by_oid.get(oid, '?')}) [no-match]",
                color=col, fontsize=7, weight="bold", zorder=6)


def _plot_step3_hungarian_cost(ax, dbg):
    """Step 3a: total Hungarian cost (cost_matrix). Matched cell in red."""
    assoc = dbg.get("assoc", {})
    C = np.asarray(assoc.get("cost_matrix", []), dtype=np.float64)
    _draw_hungarian_matrix(
        ax, dbg, C,
        "[3] Hungarian total cost  (red = matched)",
        highlight_matches=True, value_fmt=".1f",
    )


def _plot_step3_hungarian_d2_trans(ax, dbg):
    """Step 3b: translation-only Mahalanobis d²_trans (the gate metric)."""
    assoc = dbg.get("assoc", {})
    Dt = np.asarray(assoc.get("d2_trans_matrix", []), dtype=np.float64)
    _draw_hungarian_matrix(
        ax, dbg, Dt,
        "[3] d²_trans  (centroid Mahalanobis; gate)",
        highlight_matches=False, value_fmt=".2f",
    )


def _plot_step3_hungarian_adjust(ax, dbg):
    """Step 3c: non-Mahalanobis cost contributions = cost − d²_trans.
    Captures SAM2 continuity bonus (negative), label penalty (positive),
    and score penalty (positive)."""
    assoc = dbg.get("assoc", {})
    C = np.asarray(assoc.get("cost_matrix", []), dtype=np.float64)
    Dt = np.asarray(assoc.get("d2_trans_matrix", []), dtype=np.float64)
    if C.ndim == 2 and Dt.ndim == 2 and C.shape == Dt.shape and C.size:
        # Adjustment only defined on feasible cells.
        feasible = (C < 1e11) & np.isfinite(Dt)
        adj = np.where(feasible, C - Dt, np.nan)
    else:
        adj = np.zeros((0, 0), dtype=np.float64)
    _draw_hungarian_matrix(
        ax, dbg, adj,
        "[3] adjustments  (cost − d²_trans)",
        highlight_matches=False, value_fmt=".2f",
    )


def _plot_step3b_icp_points(ax, dbg, detections, dets_with_pose,
                             depth, T_wb, K):
    """Step 3b: ICP 3D point clouds per matched pair, colored by match.

    For each matched (track, detection) pair we show two clouds in the
    same track-palette color (no other symbols):

      * detection cloud -- back-projected from (mask, depth) into the
        camera frame, then into world via T_wc.
      * reference cloud -- the track's accumulated object-local surface,
        transformed by the ICP-aligned T_co and then into world via T_wc.

    When the two overlap (good alignment), you see a dense, tightly
    co-located blob of the same color; disagreement shows up as two
    same-color clusters that do not overlap.
    """
    _set_topdown_axes(ax, "[3b] ICP 3D clouds  (same color = matched)")
    _draw_base_marker(ax, T_wb, color="lightgray")

    T_bc = np.asarray(dbg.get("T_bc", np.eye(4)))
    T_wc = T_wb @ T_bc
    match_global = dbg.get("assoc", {}).get("match", {})
    track_refs = dbg.get("track_refs", {}) or {}

    def _to_world(pts_cam):
        if pts_cam.size == 0:
            return pts_cam
        return pts_cam @ T_wc[:3, :3].T + T_wc[:3, 3]

    for oid_str, det_gi in match_global.items():
        oid = int(oid_str)
        det_gi = int(det_gi)
        if det_gi >= len(dets_with_pose):
            continue
        dwp = dets_with_pose[det_gi]
        det_raw = detections[det_gi] if det_gi < len(detections) else None
        col = _palette_color_f(oid)

        # Detection cloud.
        if det_raw is not None and depth is not None:
            pts_cam = _backproject_mask(det_raw.get("mask"), depth, K)
            pts_w = _to_world(pts_cam)
            if len(pts_w) > 0:
                ax.scatter(pts_w[:, 1], pts_w[:, 0], s=1.5, c=[col],
                           alpha=0.35, marker=".", linewidths=0, zorder=4)

        # Reference cloud via T_co.
        ref = np.asarray(track_refs.get(oid, []), dtype=np.float64)
        if ref.ndim == 2 and ref.shape[0] > 0 and dwp.get("_icp_ok"):
            T_co = np.asarray(dwp["T_co"], dtype=np.float64)
            pts_cam2 = ref @ T_co[:3, :3].T + T_co[:3, 3]
            if len(pts_cam2) > 800:
                idx = np.linspace(0, len(pts_cam2) - 1, 800).astype(np.int64)
                pts_cam2 = pts_cam2[idx]
            pts_w2 = _to_world(pts_cam2)
            if len(pts_w2) > 0:
                ax.scatter(pts_w2[:, 1], pts_w2[:, 0], s=2.5, c=[col],
                           alpha=0.9, marker=".", linewidths=0, zorder=5)


def _plot_step4_visibility(ax, dbg, depth):
    """Step 4: per-track p_v (depth-raytrace) as a horizontal bar chart.

    Fixed x range [0, 1], padded to at least 5 rows so frame-to-frame
    scale never jumps.
    """
    p_v = dbg.get("visibility", {}) or {}
    oids_sorted = sorted(int(o) for o in p_v.keys())
    display_oids: List[Any] = list(oids_sorted)
    display_vals: List[float] = [float(p_v[o]) for o in oids_sorted]
    display_colors: List[Any] = [_palette_color_f(o) for o in oids_sorted]

    PAD_MIN = 5
    while len(display_oids) < PAD_MIN:
        display_oids.append(None)
        display_vals.append(0.0)
        display_colors.append((0.88, 0.88, 0.88))

    y_pos = np.arange(len(display_oids))
    ax.barh(y_pos, display_vals, color=display_colors,
            edgecolor="black", linewidth=0.5)
    # Label the y-ticks ``oid:Y (label)`` for parity with the row-1
    # tracking panel + events list. Track labels come from
    # ``post_update_tracks`` (best-known); falls back to "?" if missing.
    track_labels = {
        int(oid): tr.get("label", "?")
        for oid, tr in (dbg.get("post_update_tracks") or {}).items()
    }
    ax.set_yticks(y_pos)
    ax.set_yticklabels([
        (f"oid:{o} ({track_labels.get(int(o), '?')})"
         if o is not None else "—")
        for o in display_oids
    ], fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.axvline(0.5, linestyle=":", color="gray", alpha=0.6)
    ax.set_xlabel("p_v", fontsize=9)
    ax.set_title("[4] Visibility  (ray-traced depth)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    for y, v, o in zip(y_pos, display_vals, display_oids):
        if o is not None:
            ax.text(min(v + 0.02, 0.98), y, f"{v:.2f}",
                    va="center", fontsize=7)
    ax.invert_yaxis()


def _plot_events_list(ax, dbg, detections=None):
    """Steps 6/7/8: per-frame events as a text list ``oid ↔ pid: event``.

    The ``oid`` is the tracker's stable id; ``pid`` is the perception
    id of the detection involved (mirrors the ``p:X o:Y`` labels in
    the centroid panel so the two panels cross-reference directly).

    Each line is colored by the tracker's palette so it lines up with
    the same-track artifact in the spatial panels.
    """
    ax.axis("off")
    ax.set_title("[6/7/8] Events  (line 1: identity · line 2: detail)",
                 fontsize=10)

    detections = detections or []
    # events[i] = (sort_key, line1, line2, color)
    events: List[Tuple[int, str, str, Any]] = []

    assoc = dbg.get("assoc") or dbg.get("association") or {}
    track_oids = assoc.get("track_oids") or []
    track_labels = assoc.get("track_labels") or []
    label_by_oid: Dict[int, str] = {}
    for i, o in enumerate(track_oids):
        try:
            label_by_oid[int(o)] = (str(track_labels[i])
                                     if i < len(track_labels) else "?")
        except (TypeError, ValueError):
            continue

    def _pid_str(p):
        if p is None or (isinstance(p, int) and p < 0):
            return "  —"
        return f"{int(p):>3}"

    def _track_lbl(oid):
        return label_by_oid.get(int(oid), "?")

    # Matched events (per-record, already carry pid).
    for ev in dbg.get("matched", []) or []:
        oid = int(ev.get("oid", -1))
        pid = ev.get("pid")
        lbl = _track_lbl(oid)
        line1 = f"oid {oid:>3} ↔ pid {_pid_str(pid)}  {lbl}"
        if ev.get("reject_outer_gate"):
            line2 = "miss-gate"
        elif ev.get("reject_held_prefilter"):
            line2 = (f"held-skip  d="
                     f"{ev.get('held_meas_err_m', 0.0):.2f}m "
                     f"(>{ev.get('held_meas_radius_m', 0.0):.2f}m)")
        elif ev.get("reject_innov_clamp"):
            line2 = (f"held-innov  d="
                     f"{ev.get('innov_dist_m', 0.0):.2f}m "
                     f"(>{ev.get('innov_max_m', 0.0):.2f}m)")
        else:
            d2 = ev.get("d2_trans", float("nan"))
            line2 = f"hit  d²t={d2:.2f}"
        events.append((oid, line1, line2, _palette_color_f(oid)))

    # Missed-update events.
    for ev in dbg.get("missed", []) or []:
        oid = int(ev.get("oid", -1))
        pid = ev.get("pid")
        lbl = _track_lbl(oid)
        line1 = f"oid {oid:>3} ↔ pid {_pid_str(pid)}  {lbl}"
        pv = ev.get("p_v", 1.0)
        reason = ev.get("reason", "?")
        best_d2 = ev.get("best_d2_trans")
        gate = ev.get("gate")
        detail = ""
        if best_d2 is not None and gate is not None:
            try:
                bd = float(best_d2); gv = float(gate)
                if bd == bd:  # not NaN
                    bp = ev.get("best_pid")
                    bp_s = (str(int(bp)) if bp is not None else "—")
                    detail = (f" best d²t={bd:.1f} (G={gv:.1f}) "
                              f"pid={bp_s}")
            except (TypeError, ValueError):
                pass
        line2 = f"miss  p_v={pv:.2f}{detail}  [{reason}]"
        events.append((oid, line1, line2, (0.45, 0.45, 0.45)))

    # Births.
    for br in dbg.get("births", []) or []:
        new_oid = br.get("new_oid")
        if new_oid is None:
            continue
        oid = int(new_oid)
        pid = br.get("pid")
        lbl = br.get("label", _track_lbl(oid))
        line1 = f"oid {oid:>3} ↔ pid {_pid_str(pid)}  {lbl}"
        line2 = "birth (admit)"
        events.append((oid, line1, line2, _palette_color_f(oid)))

    # Birth rejects — no oid yet, sort by pid.
    for br in dbg.get("birth_rejects", []) or []:
        pid = br.get("pid")
        sort_key = (10**6 + (int(pid) if pid is not None else 999))
        lbl = br.get("label", "?")
        line1 = f"oid   — ↔ pid {_pid_str(pid)}  {lbl}"
        reason = br.get("reason", "?")
        if reason == "confirm":
            n_obs = br.get("n_obs_tracker", 0)
            k = br.get("confirm_k", "?")
            line2 = f"rej:confirm  n_obs={n_obs} (need={k})"
        elif reason == "score":
            s = br.get("tracker_max_score", br.get("score", 0.0))
            mn = br.get("score_min", 0.0)
            line2 = (f"rej:score  s={float(s):.2f} "
                     f"(need ≥{float(mn):.2f})")
        elif reason == "near_live":
            d = br.get("dist_m", 0.0)
            g = br.get("gate_m", 0.0)
            no = br.get("nearest_oid", "?")
            anc = br.get("anchor", "")
            anc_s = f" via {anc}" if anc else ""
            line2 = (f"rej:near_live  d={float(d):.2f}m "
                     f"(gate={float(g):.2f}, near oid {no}{anc_s})")
        elif reason == "border":
            box = br.get("box")
            mp = br.get("margin_px", "?")
            box_s = ""
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    box_s = (f" box=({int(box[0])},{int(box[1])},"
                             f"{int(box[2])},{int(box[3])})")
                except (TypeError, ValueError):
                    pass
            line2 = f"rej:border  margin={mp}{box_s}"
        else:
            line2 = f"rej:{reason}"
        events.append((sort_key, line1, line2, (0.55, 0.55, 0.55)))

    # Prunes.
    for pr in dbg.get("pruned", []) or []:
        oid = int(pr.get("oid", -1))
        r_val = float(pr.get("r", 0.0))
        lbl = _track_lbl(oid)
        line1 = f"oid {oid:>3}        {lbl}"
        line2 = f"prune (r={r_val:.2f})"
        events.append((oid, line1, line2, (0.85, 0.2, 0.2)))

    # Self-merges (drop → keep).
    for mg in dbg.get("self_merges", []) or []:
        keep, drop = mg.get("keep_oid"), mg.get("drop_oid")
        if keep is None or drop is None:
            continue
        d_oid = int(drop)
        lbl = _track_lbl(d_oid)
        line1 = f"oid {d_oid:>3}        {lbl}"
        line2 = f"merge → oid {int(keep)}"
        events.append((d_oid, line1, line2, (0.80, 0.20, 0.80)))

    # Subpart-absorbed (no oid; pid pair).
    for sa in dbg.get("subpart_absorbed", []) or []:
        try:
            from_idx = int(sa.get("from_idx"))
            into_idx = int(sa.get("into_idx"))
            pid_from = detections[from_idx].get("id")
            pid_into = detections[into_idx].get("id")
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if pid_from is None or pid_into is None:
            continue
        cont = float(sa.get("containment", 0.0))
        lbl_from = sa.get("from_label", "?")
        lbl_into = sa.get("into_label", "?")
        sort_key = 10**6 + int(pid_from)
        line1 = (f"pid {_pid_str(pid_from)} → "
                 f"pid {_pid_str(pid_into)}  {lbl_from}→{lbl_into}")
        line2 = f"absorb  c={cont:.2f}"
        events.append((sort_key, line1, line2, (0.55, 0.45, 0.30)))

    # Centroid drops.
    for cd in dbg.get("centroid_dropped", []) or []:
        pid = cd.get("pid")
        reason = cd.get("reason", "?")
        lbl = cd.get("label", "?")
        sort_key = 10**6 + (int(pid) if pid is not None else 999)
        line1 = f"oid   — ↔ pid {_pid_str(pid)}  {lbl}"
        line2 = f"cent-drop ({reason})"
        events.append((sort_key, line1, line2, (0.55, 0.55, 0.55)))

    events.sort(key=lambda x: x[0] if x[0] is not None else 10**9)

    if not events:
        ax.text(0.5, 0.5, "(no events)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        return

    y_start = 0.97
    y_step_event = 0.062
    line2_indent = 0.06
    line2_drop = 0.026
    max_events = 15
    n_drawn_events = 0
    for i, (_k, l1, l2, col) in enumerate(events[:max_events]):
        y1 = y_start - i * y_step_event
        if y1 < 0.04:
            break
        ax.text(0.03, y1, l1, transform=ax.transAxes,
                fontsize=8, color=col, family="monospace",
                va="top", weight="bold")
        ax.text(0.03 + line2_indent, y1 - line2_drop, l2,
                transform=ax.transAxes,
                fontsize=7, color=col, family="monospace", va="top")
        n_drawn_events += 1

    # ── Relations block (LLM scene graph) ─────────────────────────
    # Build per-oid "supporters" map from dbg["relations"].
    # Convention: parent = supported, child = supporter — so the
    # children of edges with our parent are the things this oid sits on.
    supporters: Dict[int, List[int]] = {}
    for rel in dbg.get("relations", []) or []:
        try:
            p_oid = int(rel["parent"])
            c_oid = int(rel["child"])
            rt = str(rel.get("type", "on"))
        except (KeyError, TypeError, ValueError):
            continue
        if rt not in ("on", "in"):
            continue
        supporters.setdefault(p_oid, []).append(c_oid)

    if supporters:
        rel_color = (0.10, 0.45, 0.50)   # dark teal
        # Start the block one event-row below the last event.
        y_rel = max(0.04 + 0.05,
                    y_start - n_drawn_events * y_step_event - 0.015)
        # Title divider.
        ax.text(0.03, y_rel,
                "── Relations (LLM) ──",
                transform=ax.transAxes,
                fontsize=8, color=rel_color, family="monospace",
                va="top", weight="bold")
        y_rel -= 0.034
        # One line per supported oid (sorted).
        max_rel_lines = 8
        rel_lines_drawn = 0
        for p_oid in sorted(supporters.keys()):
            if rel_lines_drawn >= max_rel_lines or y_rel < 0.02:
                break
            kids = sorted(set(supporters[p_oid]))
            kids_str = ", ".join(str(k) for k in kids)
            lbl = label_by_oid.get(p_oid, "?")
            ax.text(0.03, y_rel,
                    f"oid {p_oid:>3} ({lbl})  on: {kids_str}",
                    transform=ax.transAxes,
                    fontsize=8, color=rel_color, family="monospace",
                    va="top")
            y_rel -= 0.034
            rel_lines_drawn += 1


def render_frame(rgb: np.ndarray,
                 detections: List[Dict[str, Any]],
                 dbg: Dict[str, Any],
                 dets_with_pose: List[Dict[str, Any]],
                 r_history: Dict[int, List[Tuple[int, float]]],
                 frame_idx: int,
                 max_frame: int,
                 out_path: str,
                 traj: str,
                 depth: Optional[np.ndarray] = None,
                 K: Optional[np.ndarray] = None) -> None:
    """Nine-panel per-frame visualisation, one concept per panel.

    Layout (3×3):
        [0 SLAM]           [1 Predict]         [2 Coarse mask/centroid]
        [3 Hungarian cost] [3 d²_trans]        [3 adjustments]
        [3b ICP 3D points] [4 Visibility]      [6/7/8 Events id→event]
    """
    T_wb = dbg["slam_pose"]
    K_use = K if K is not None else K_DEFAULT

    # 4-row layout, row 4 spans all 3 cols and is taller for the
    # accumulated point-cloud panel.
    fig = plt.figure(figsize=(19, 22), dpi=100)
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[1.0, 1.0, 1.0, 1.6],
        hspace=0.32, wspace=0.26,
        left=0.05, right=0.98, top=0.96, bottom=0.04,
    )

    # Row 1 — Detected (pid)  |  Tracked (oid)  |  Events (oid ↔ pid)
    _plot_detected_pid(fig.add_subplot(gs[0, 0]), rgb, detections, dbg)
    _plot_tracked_oid (fig.add_subplot(gs[0, 1]), rgb, dets_with_pose, dbg,
                        K=K_use)
    _plot_events_list (fig.add_subplot(gs[0, 2]), dbg, detections=detections)

    # Row 2 — Hungarian matrices
    _plot_step3_hungarian_cost    (fig.add_subplot(gs[1, 0]), dbg)
    _plot_step3_hungarian_d2_trans(fig.add_subplot(gs[1, 1]), dbg)
    _plot_step3_hungarian_adjust  (fig.add_subplot(gs[1, 2]), dbg)

    # Row 3 — SLAM, EKF predict, visibility
    _plot_step0_slam      (fig.add_subplot(gs[2, 0]), dbg, T_wb)
    _plot_step1_predict   (fig.add_subplot(gs[2, 1]), dbg, T_wb)
    _plot_step4_visibility(fig.add_subplot(gs[2, 2]), dbg, depth)

    # Row 4 — wide accumulated point-cloud panel (spans all 3 cols)
    _plot_step3b_icp_points(fig.add_subplot(gs[3, :]),
                             dbg, detections, dets_with_pose,
                             depth, T_wb, K_use)

    fig.suptitle(
        f"EKF tracking pipeline — traj={traj}   frame={frame_idx:04d}   "
        f"base=({T_wb[0,3]:.2f},{T_wb[1,3]:.2f})",
        fontsize=13, y=0.985,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_in_the_tray")
    ap.add_argument("--max-frame", type=int, default=700,
                    help="exclusive upper bound on frame index")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--pose-method", default="icp_chain",
                    choices=("centroid", "icp_chain", "icp_anchor",
                             "icp_chain_strict", "icp_anchor_strict"))
    ap.add_argument("--out-subdir", default="ekf_debug")
    ap.add_argument("--state-subdir", default="ekf_state",
                    help="dir for per-frame JSON state dumps "
                         "(under tests/visualization_pipeline/{traj}/)")
    ap.add_argument("--no-png", action="store_true",
                    help="skip 5-panel PNG rendering (only dump JSON state)")
    ap.add_argument("--no-mp4", action="store_true",
                    help="skip composing the per-frame PNGs into an MP4")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="frame rate for the composed MP4 (default 10)")
    args = ap.parse_args()

    traj = args.trajectory
    ds_root = os.path.join(DATASET_DIR, traj)
    viz_root = os.path.join(VIZ_BASE, traj)
    rgb_dir = os.path.join(ds_root, "rgb")
    depth_dir = os.path.join(ds_root, "depth")
    det_dir = os.path.join(viz_root, "perception", "detection_h")
    pose_path = os.path.join(ds_root, "pose_txt", "amcl_pose.txt")
    out_dir = os.path.join(viz_root, args.out_subdir)
    state_dir = os.path.join(viz_root, args.state_subdir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    # ── load poses ──
    slam_poses = _load_amcl_poses(pose_path)
    if not slam_poses:
        raise SystemExit(f"no AMCL poses loaded from {pose_path}")
    # Per-frame base-to-camera-optical extrinsic (head pan/tilt/torso lift).
    # Falls back to None if the dataset wasn't extracted with T_bc support;
    # the tracker then keeps its construction-time T_bc (default identity).
    T_bc_path = os.path.join(ds_root, "pose_txt", "T_bc.txt")
    T_bc_map = _load_T_bc_poses(T_bc_path)
    if T_bc_map is None:
        print(f"[warn] no T_bc.txt at {T_bc_path}; using identity. "
              f"Re-extract the rosbag to enable head-motion-aware tracking.")
    else:
        print(f"[T_bc] loaded {len(T_bc_map)} per-frame extrinsics from "
              f"{T_bc_path}")

    # Per-frame gripper-in-base and gripper width. Without these, the
    # tracker cannot distinguish a moving held object from a world-static
    # one — the Hungarian gate fails and duplicates birth.
    T_bg_map = _load_ee_poses(os.path.join(ds_root, "pose_txt", "ee_pose.txt"))
    gripper_w_map = _load_gripper_widths(
        os.path.join(ds_root, "pose_txt", "joints_pose.json"))
    if T_bg_map is not None:
        print(f"[T_bg] loaded {len(T_bg_map)} per-frame EE poses")
    else:
        print(f"[warn] no ee_pose.txt; held-object tracking disabled")
    if gripper_w_map is not None:
        print(f"[grip] loaded {len(gripper_w_map)} per-frame finger widths")
    else:
        print(f"[warn] no joints_pose.json; phase will remain 'idle'")

    # Raw per-frame joints dict (needed by GraspOwnerDetector to call
    # the gripper geometry's `state_from_joints` with real l/r values).
    joints_path = os.path.join(ds_root, "pose_txt", "joints_pose.json")
    joints_map: Optional[Dict[int, Dict[str, float]]] = None
    if os.path.exists(joints_path):
        with open(joints_path) as f:
            raw = json.load(f)
        joints_map = {int(k): v for k, v in raw.items()}

    # Construct the robot-agnostic grasp-owner detector.
    gripper_geom = create_gripper_geometry(robot_type="fetch")
    grasp_detector = GraspOwnerDetector(gripper=gripper_geom)
    print(f"[grasp] {gripper_geom.describe()}")
    grip_inferrer = _GripperStateInferrer(detector=grasp_detector)

    # Construct the relation pipeline (LLM by default; opt out with
    # env var EKF_VIZ_RELATION_BACKEND=none for fast / offline runs).
    # The cache lives per-trajectory so re-runs on the same dataset
    # replay LLM responses from disk for free.
    relation_backend = os.environ.get("EKF_VIZ_RELATION_BACKEND", "llm")
    relation_cache_dir = os.environ.get(
        "EKF_VIZ_RELATION_CACHE_DIR",
        os.path.join(viz_root, "relation_cache"))
    relation_pipeline = _RelationPipeline(backend=relation_backend,
                                            cache_dir=relation_cache_dir)
    print(f"[relation] backend={relation_backend}  "
          f"trigger=on_grasp/on_release/on_new_object + every "
          f"{relation_pipeline._cfg.relation_every_n_frames} frames")

    # ── tracker setup ──
    cfg = BernoulliConfig(
        association_mode="hungarian",
        p_s=1.0,
        p_d=0.9,
        alpha=4.4,
        lambda_c=1.0,
        lambda_b=1.0,
        r_conf=0.5,
        r_min=1e-3,
        G_in=12.59,
        G_out=25.0,
        P_max=np.diag([0.25**2] * 3 + [(np.pi / 4) ** 2] * 3),
        enable_visibility=True,
        enable_huber=True,
        init_cov_from_R=False,        # fixed 5 cm / 0.05 rad at birth (vs ICP's R)
        # Perception-style soft cost on labels + score (mirrors
        # rosbag2dataset/sam2/sam2_client._pair_cost). The hard label
        # gate is OFF; a noisy label disagreement adds 6.0 chi^2_6 units
        # to the cost (less than the outer gate, so a geometrically
        # excellent pair can still match) and a per-detection score
        # bias of 2.0*(1-score) mildly disfavours low-confidence dets.
        enforce_label_match=False,
        hungarian_label_penalty=6.0,
        hungarian_score_weight=2.0,
        # Translation-only outer gate: per-oid ICP rotation chains
        # drift independently and can lift d^2_rot above the chi^2_6
        # gate even at sub-mm translation match (frame-485 case:
        # d^2_trans=0.02, d^2_rot=112). Trans-only gate at chi^2_3
        # 0.9997 = 21.108 fixes that. Cost still uses 'sum' so a
        # rotation match gets a tie-break.
        gate_mode="trans",
        G_out_trans=21.108,
        cost_d2_mode="sum",
        # Floor P_bo per axis -- prevents the EKF posterior from
        # shrinking below realistic per-frame perception jitter
        # (5 mm trans / 0.05 rad rot). Without this, a track with
        # 10k+ observations rejects 2-3 cm jitter on the next frame
        # via the chi^2_3 gate (frame-430 case).
        P_min_diag=np.array([0.005**2] * 3 + [0.05**2] * 3),
        # Track-to-track self-merge after each step: catches duplicate
        # tracks of the same physical object that survived a one-to-one
        # Hungarian round (e.g. SAM2 ids splitting). Euclidean gate in
        # metres — ≈ one object radius for apples / cups / cans. Does NOT
        # scale with covariance, so two fresh births can't collapse just
        # because σ starts at 5 cm.
        self_merge_trans_m=0.05,
        K=K_DEFAULT,
        image_shape=(480, 640),
    )
    tracker = InstrumentedTracker(K_DEFAULT, cfg, pose_method=args.pose_method)

    # ── per-track r(t) history ──
    r_history: Dict[int, List[Tuple[int, float]]] = {}

    max_frame = min(args.max_frame, len(slam_poses))
    frames_processed = 0
    frames_written = 0
    for idx in range(args.start, max_frame):
        if (idx - args.start) % args.step != 0:
            continue

        rgb = _load_rgb(os.path.join(rgb_dir, f"rgb_{idx:06d}.png"))
        depth = _load_depth(os.path.join(depth_dir, f"depth_{idx:06d}.npy"))
        if rgb is None or depth is None:
            continue

        detections = _load_detection_json(
            os.path.join(det_dir, f"detection_{idx:06d}_final.json"))

        T_wb = slam_poses[idx]
        T_bc_now = T_bc_map.get(idx) if T_bc_map is not None else None
        T_bg_now = T_bg_map.get(idx) if T_bg_map is not None else None
        w_now = gripper_w_map.get(idx) if gripper_w_map is not None else None
        joints_now = joints_map.get(idx) if joints_map is not None else None

        # Infer gripper state (phase + held oid) from proprioception.
        # Runs BEFORE predict so the predict step gets the correct
        # phase-dependent Q and rigid-attachment handling.
        # detections + depth + K + T_bc + joints are passed for the
        # geometric grasp-selector at grasp onset.
        gripper_state = grip_inferrer.step(
            width=w_now, tracker=tracker, T_wb=T_wb, T_bg=T_bg_now,
            detections=detections, depth=depth, K=K_DEFAULT,
            T_bc=T_bc_now, joints=joints_now)
        held_seed = gripper_state.get("held_obj_id")

        # Maybe re-run the relation backend (throttled by trigger gate).
        # Build per-detection-idx → tracker-oid map from the live tracks'
        # sam2_tau (each track stores the perception id it was last
        # matched to). This lets the relation client reason about
        # *tracker* identities rather than per-frame perception ids.
        det_to_oid: Dict[int, int] = {}
        tau_to_oid = {int(t): int(o) for o, t in tracker.sam2_tau.items()
                       if t is not None}
        for di, d in enumerate(detections):
            pid = d.get("id")
            if pid is None:
                continue
            oid = tau_to_oid.get(int(pid))
            if oid is not None:
                det_to_oid[di] = oid
        live_oids = {int(o) for o in tracker.object_labels.keys()}
        # Build per-track world-frame xyz + label snapshot for the
        # geometric-support fallback inside the relation pipeline.
        live_tracks: Dict[int, Dict[str, Any]] = {}
        for oid in live_oids:
            pe = tracker.state.collapsed_object_base(int(oid))
            if pe is None:
                continue
            mu_b = np.asarray(pe.T, dtype=np.float64)[:3, 3]
            mu_w = (T_wb @ np.append(mu_b, 1.0))[:3]
            live_tracks[int(oid)] = {
                "xyz_w": mu_w.tolist(),
                "label": tracker.object_labels.get(int(oid), "?"),
            }
        rel_summary = relation_pipeline.maybe_update(
            frame=idx, rgb=rgb, detections=detections,
            det_to_oid=det_to_oid,
            current_phase=gripper_state["phase"],
            current_oids=live_oids,
            held_oid=held_seed,
            live_tracks=live_tracks)

        # Expand held_oids using the latest filtered relation graph.
        held_oids = expand_held_with_relations(
            held_seed, relation_pipeline.edges)
        # Drop any oids the tracker doesn't actually have (relation
        # graph might still reference a pruned/merged track).
        held_oids = {o for o in held_oids if o in tracker.state.objects}

        dbg, dets_with_pose = tracker.step(
            rgb=rgb, depth=depth, T_wb=T_wb,
            detections=detections,
            phase=gripper_state["phase"],
            T_bc=T_bc_now,
            T_bg=T_bg_now,
            held_oids=held_oids,
            held_seed=held_seed,
            relation_edges=relation_pipeline.edges,
        )
        # Re-map held_obj_id if self-merge renamed it.
        grip_inferrer.apply_merges(dbg.get("self_merges", []))
        gripper_state["held_obj_id"] = grip_inferrer._held_obj_id
        # Also remap the relation EMA so next frame's held-set
        # expansion still finds the right edges.
        relation_pipeline.remap_after_merges(dbg.get("self_merges", []))
        dbg["gripper_state"] = dict(gripper_state)
        # Expose the expanded held set + the relation snapshot for the
        # state JSON / mask panel highlight.
        dbg["held_oids_used"] = sorted(int(o) for o in held_oids)
        dbg["relations"] = [
            {"parent": int(e.parent), "child": int(e.child),
             "type": str(e.relation_type), "score": float(e.score)}
            for e in relation_pipeline.edges
        ]
        if rel_summary.get("fired"):
            dbg["relation_call"] = rel_summary
        frames_processed += 1

        for oid, tr in dbg["post_update_tracks"].items():
            r_history.setdefault(oid, []).append((idx, float(tr["r"])))

        # Always dump the JSON state (cheap; useful for offline diagnosis).
        state_path = os.path.join(state_dir, f"frame_{idx:06d}.json")
        try:
            _dump_frame_json(state_path, dbg, detections, dets_with_pose)
        except Exception as e:
            print(f"[WARN] state dump failed at frame {idx}: {e}")

        if not args.no_png:
            out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
            try:
                render_frame(
                    rgb=rgb, detections=detections, dbg=dbg,
                    dets_with_pose=dets_with_pose,
                    r_history=r_history,
                    frame_idx=idx,
                    max_frame=max_frame,
                    out_path=out_path,
                    traj=traj,
                    depth=depth,
                )
                frames_written += 1
            except Exception as e:
                print(f"[WARN] render failed at frame {idx}: {e}")
        else:
            frames_written += 1

        if frames_processed % 20 == 0:
            print(f"[{traj}] frame {idx}: processed {frames_processed}, "
                  f"written {frames_written}, tracks={len(tracker.object_labels)}")

    print(f"[done] wrote {frames_written} frames under {out_dir}")

    if not args.no_png and not args.no_mp4 and frames_written > 0:
        mp4_path = out_dir.rstrip("/\\") + ".mp4"
        try:
            _compose_frames_to_mp4(out_dir, mp4_path, fps=args.fps)
            print(f"[mp4] {mp4_path}")
        except Exception as e:
            print(f"[mp4] composition failed: {e}")


def _compose_frames_to_mp4(png_dir: str, out_path: str,
                            fps: float = 10.0) -> None:
    """Stitch every ``frame_*.png`` in ``png_dir`` into a single MP4.

    Uses ffmpeg's concat demuxer with an explicit file list so that
    sparse PNG sequences (e.g. when ``--step`` skipped frames) compose
    without holes. Output is H.264 / yuv420p so QuickTime, browsers,
    and VLC all play it.
    """
    import glob
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH; install via `brew install ffmpeg`")

    pngs = sorted(glob.glob(os.path.join(png_dir, "frame_*.png")))
    if not pngs:
        raise RuntimeError(f"no frame_*.png files in {png_dir}")

    dur = 1.0 / float(fps)
    with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                       delete=False) as fh:
        list_path = fh.name
        for p in pngs:
            fh.write(f"file '{os.path.abspath(p)}'\n")
            fh.write(f"duration {dur:.6f}\n")
        # ffmpeg concat demuxer requires the last file repeated.
        fh.write(f"file '{os.path.abspath(pngs[-1])}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-vsync", "vfr",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()

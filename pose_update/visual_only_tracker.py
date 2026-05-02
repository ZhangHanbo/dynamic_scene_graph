"""
Visual-only object tracking baselines — NO filter, NO proprioception.

Each call to `update(oid, mask, depth, T_cw)` produces a single
deterministic world-frame pose `T_wo(t)` via direct geometric composition:

    T_wo(t) = T_cw(t) · T_co(t)
    T_co(t) = ICP(reference_cloud → current_cloud, init = (I, centroid_now))

This is intentionally separate from `PoseEstimator` (which feeds a
Bayesian filter) and from `TwoTierOrchestrator` (which wraps everything
in RBPF + per-particle EKFs + proprioception). Here there is no Σ_wb,
no R_icp, no Q schedule — just per-frame ICP and a one-line composition
to the world frame.

Two reference-update policies (the two baselines):

  mode = "first_frame"
    Reference cloud captured on the FIRST observation, never updated.
    Each later frame's ICP measures T_co relative to that fixed
    snapshot. For a static object viewed near the first viewpoint:
    drift-free. As the camera moves and the centroid drifts beyond
    ICP's correspondence threshold, fitness collapses and observations
    are rejected — world pose freezes at the last accepted T_wo.

  mode = "last_frame"
    Reference cloud REPLACED each successful frame with the current
    observation (centroid-shifted). Each ICP measures the relative
    motion between consecutive frames. Robust to long camera motions
    because every step is small, but errors compound across frames
    (drift). Tracks moved objects naturally because the reference
    follows them.

Both modes share the same fitness gate (≥ 0.90) and RMSE gate
(≤ 15 mm). On rejection the cached `T_wo` is returned unchanged
(equivalent to a constant-velocity-zero hold). No covariance is
produced — this is the load-bearing distinction from `PoseEstimator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d

# Reuse the back-projection / voxelization helpers from icp_pose.
from pose_update.perception.icp_pose import _back_project, _voxelize


MODES = ("first_frame", "last_frame")


@dataclass
class _TrackState:
    """Per-object running state. Strictly camera/world geometry,
    no probabilistic quantities."""
    T_wo: np.ndarray              # current world-frame pose estimate
    ref_points: np.ndarray        # (M, 3) reference cloud, object-local frame
    n_acc: int                    # accepted updates since first observation
    n_rej: int                    # rejected updates since first observation


class VisualOnlyTracker:
    """Direct visual tracking with NO filter and NO proprioception.

    Args:
        K:    (3, 3) camera intrinsics.
        mode: "first_frame" or "last_frame" reference update policy.
    """

    VOXEL = 0.005          # 5 mm
    ICP_THRESHOLD = 0.020  # 2 cm correspondence radius
    ICP_MAX_ITER = 30
    MIN_FITNESS = 0.90
    MAX_RMSE = 0.015       # 15 mm

    def __init__(self, K: np.ndarray, mode: str = "last_frame"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.K = np.asarray(K, dtype=np.float64)
        self.mode = mode
        self._state: Dict[int, _TrackState] = {}

    # --------------------------------------------------------------- #
    #  Public update
    # --------------------------------------------------------------- #

    def update(self,
               oid: int,
               mask: np.ndarray,
               depth: np.ndarray,
               T_cw: np.ndarray,
               ) -> Tuple[Optional[np.ndarray], bool, float, float]:
        """Update the world-frame pose estimate for one detection.

        Args:
            oid:   object id.
            mask:  (H, W) bool detection mask.
            depth: (H, W) float32 depth in metres.
            T_cw:  (4, 4) camera-to-world (Layer 1 SLAM output). Used
                   ONLY for the final composition T_wo = T_cw · T_co —
                   never inside ICP itself.

        Returns:
            T_wo:     (4, 4) current world-frame pose, or None if the
                      object has never been successfully tracked.
            accepted: whether this frame's ICP succeeded and updated T_wo.
            fitness:  ICP inlier fraction (1.0 on the first frame).
            rmse:     ICP inlier RMSE (0.0 on the first frame).
        """
        pts_cam = _back_project(mask, depth, self.K)
        if pts_cam is None:
            cached = self._state.get(oid)
            return (cached.T_wo if cached else None), False, 0.0, 0.0

        centroid_now = pts_cam.mean(axis=0)

        # ── First observation: anchor object-local frame at the centroid ─
        if oid not in self._state:
            ref_points = pts_cam - centroid_now
            T_co_init = np.eye(4, dtype=np.float64)
            T_co_init[:3, 3] = centroid_now
            T_wo = T_cw @ T_co_init
            self._state[oid] = _TrackState(
                T_wo=T_wo,
                ref_points=ref_points,
                n_acc=1,
                n_rej=0,
            )
            return T_wo, True, 1.0, 0.0

        state = self._state[oid]

        # ── ICP: align reference cloud to current observation ─────────
        # Init is purely camera-frame: identity rotation + current centroid.
        # No T_wb anywhere in ICP.
        init_T = np.eye(4, dtype=np.float64)
        init_T[:3, 3] = centroid_now

        src = _voxelize(state.ref_points, self.VOXEL)
        tgt = _voxelize(pts_cam, self.VOXEL)
        result = o3d.pipelines.registration.registration_icp(
            src, tgt,
            self.ICP_THRESHOLD,
            init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.ICP_MAX_ITER),
        )
        T_co = np.asarray(result.transformation, dtype=np.float64)
        fitness = float(result.fitness)
        rmse = float(result.inlier_rmse)

        # ── Fitness gate ─────────────────────────────────────────────
        if (fitness < self.MIN_FITNESS or rmse > self.MAX_RMSE
                or not np.isfinite(T_co).all()):
            state.n_rej += 1
            return state.T_wo, False, fitness, rmse

        # ── Accept: compose with localization to get world-frame pose ─
        T_wo_new = T_cw @ T_co
        state.T_wo = T_wo_new
        state.n_acc += 1

        # ── Reference update policy ──────────────────────────────────
        if self.mode == "last_frame":
            # Replace the reference with the current observation
            # (centroid-shifted to its own object-local frame).
            # Next frame will ICP-align THIS cloud to whatever comes next.
            state.ref_points = pts_cam - centroid_now
        # else mode == "first_frame": reference stays as captured at oid first
        # detection, never modified.

        return T_wo_new, True, fitness, rmse

    # --------------------------------------------------------------- #
    #  Inspection
    # --------------------------------------------------------------- #

    def world_pose(self, oid: int) -> Optional[np.ndarray]:
        """Return the current cached world-frame pose, or None if unseen."""
        st = self._state.get(oid)
        return st.T_wo.copy() if st else None

    def stats(self, oid: int) -> Tuple[int, int]:
        """Accepted, rejected counts since first observation of `oid`."""
        st = self._state.get(oid)
        if st is None:
            return 0, 0
        return st.n_acc, st.n_rej

    def reset(self, oid: Optional[int] = None) -> None:
        """Drop per-object state (e.g., on detection identity loss)."""
        if oid is None:
            self._state.clear()
        else:
            self._state.pop(oid, None)

"""Gripper phase FSM with adaptive transition windows.

Drives the per-frame ``{phase, held_obj_id, gripper_width_m,
is_moving, held_pid, held_grasp_count}`` summary the EKF tracker
consumes. The four phases form a cycle:

    idle ──(open→closed)──▶ grasping ──(stable+timer)──▶ holding
                                                             │
    idle ◀──(stable+timer)── releasing ◀──(closed→open)─────┘

`grasping` and `releasing` are the *transition windows* during which
the rigid-attachment motion model can't be trusted (the gripper is
physically moving and/or the held object is still settling). They begin
at the binary threshold-crossing of the finger width and persist until
BOTH:

  a) at least ``min_transition_frames`` frames have elapsed, AND
  b) the gripper width has stabilised — the last ``history_size``
     widths span less than ``motion_threshold_m``.

Held identity is assigned at the first frame of `grasping` by the
:class:`pose_update.grasp_owner_detector.GraspOwnerDetector` (geometric
containment + optional perception override + nearest-track fallback);
persists through `holding`; cleared at the first frame of `idle` after
`releasing` completes. Self-merges remap it via
:meth:`GripperPhaseTracker.apply_merges`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pose_update.grasp_owner_detector import (
    GraspOwnerDetector,
    TrackerState,
)


class GripperPhaseTracker:
    """Per-frame phase + held-oid inference from gripper proprioception.

    Robot-agnostic. Depends on
    :class:`pose_update.grasp_owner_detector.TrackerState` (a small
    abstract adapter exposing ``sam2_tau``, ``iter_world_centroids``,
    ``force_admit``) so it can plug into any tracker implementation.

    Public API:
        - ``step(width, tracker_state, T_wb, T_bg, ...)`` → summary dict.
        - ``apply_merges(merges)`` — remap ``held_obj_id`` after a
          self-merge pass.
        - ``held_obj_id`` (property): current held oid or ``None``.

    Dependencies:
        - ``GraspOwnerDetector``: for the geometric containment / fallback
          held-oid selection at grasp onset. Pass ``None`` to fall back to
          the legacy nearest-track-to-EE rule.
    """

    def __init__(
        self,
        closed_width_m: float = 0.025,
        open_width_m: float = 0.040,
        grasp_radius_m: float = 0.30,
        history_size: int = 5,
        motion_threshold_m: float = 0.01,
        min_transition_frames: int = 5,
        min_inside_count: int = 20,
        detector: Optional[GraspOwnerDetector] = None,
    ):
        self.closed_width = float(closed_width_m)
        self.open_width = float(open_width_m)
        self.grasp_radius = float(grasp_radius_m)
        self.history_size = int(history_size)
        self.motion_threshold = float(motion_threshold_m)
        self.min_transition_frames = int(min_transition_frames)
        self.min_inside_count = int(min_inside_count)
        self.detector = detector

        self._closed_prev: Optional[bool] = None
        self._held_obj_id: Optional[int] = None
        self._phase_prev: str = "idle"
        self._transition_remaining: int = 0
        self._width_history: List[float] = []
        self._joints_now: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def held_obj_id(self) -> Optional[int]:
        return self._held_obj_id

    def apply_merges(self, merges: List[Dict[str, Any]]) -> None:
        """Remap ``held_obj_id`` after self-merges. If the held oid was
        the drop in any merge, switch to the keeper.
        """
        if self._held_obj_id is None or not merges:
            return
        for m in merges:
            drop = m.get("drop_oid")
            keep = m.get("keep_oid")
            if drop is None or keep is None:
                continue
            if int(drop) == int(self._held_obj_id):
                self._held_obj_id = int(keep)

    def step(
        self,
        width: Optional[float],
        tracker_state: TrackerState,
        T_wb: np.ndarray,
        T_bg: Optional[np.ndarray],
        detections: Optional[List[Dict[str, Any]]] = None,
        depth: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
        T_bc: Optional[np.ndarray] = None,
        joints: Optional[Dict[str, Any]] = None,
        live_oids: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Return ``{phase, held_obj_id, gripper_width_m, is_moving,
        held_pid, held_grasp_count}``.

        ``detections / depth / K / T_bc`` are needed only at grasp onset
        to compute the geometric grasp-owner selection.

        ``live_oids``: optional iterable of oids the tracker considers
        live this frame; if the held oid is missing from it, the held
        identity is cleared (track was pruned).
        """
        self._joints_now = joints

        if width is None:
            return {"phase": self._phase_prev,
                    "held_obj_id": self._held_obj_id,
                    "gripper_width_m": None,
                    "is_moving": None}

        self._width_history.append(float(width))
        if len(self._width_history) > self.history_size:
            self._width_history.pop(0)

        # Seed on first observation.
        if self._closed_prev is None:
            self._closed_prev = width < self.closed_width
            self._phase_prev = "holding" if self._closed_prev else "idle"
            return {"phase": self._phase_prev,
                    "held_obj_id": None,
                    "gripper_width_m": width,
                    "is_moving": False}

        # Hysteresis on closed/open.
        if self._closed_prev:
            is_closed = width < self.open_width
        else:
            is_closed = width < self.closed_width

        if len(self._width_history) >= 2:
            spread = max(self._width_history) - min(self._width_history)
            is_moving = spread > self.motion_threshold
        else:
            is_moving = False

        if self._transition_remaining > 0:
            self._transition_remaining -= 1

        just_closed = (not self._closed_prev) and is_closed
        just_opened = self._closed_prev and (not is_closed)

        prev_phase = self._phase_prev
        new_phase = prev_phase

        held_pid_now: Optional[Any] = None
        held_count_now: int = 0

        if prev_phase == "idle":
            if just_closed:
                (self._held_obj_id, held_pid_now, held_count_now
                 ) = self._select_held_oid_at_grasp(
                    tracker_state, T_wb, T_bg, T_bc, detections, depth, K, width)
                new_phase = "grasping"
                self._transition_remaining = self.min_transition_frames
                self._width_history = [float(width)]
                is_moving = True

        elif prev_phase == "grasping":
            if just_opened:
                new_phase = "releasing"
                self._transition_remaining = self.min_transition_frames
                self._width_history = [float(width)]
                is_moving = True
            elif self._transition_remaining == 0 and not is_moving:
                new_phase = "holding"

        elif prev_phase == "holding":
            if just_opened:
                new_phase = "releasing"
                self._transition_remaining = self.min_transition_frames
                self._width_history = [float(width)]
                is_moving = True

        elif prev_phase == "releasing":
            if just_closed:
                (self._held_obj_id, held_pid_now, held_count_now
                 ) = self._select_held_oid_at_grasp(
                    tracker_state, T_wb, T_bg, T_bc, detections, depth, K, width)
                new_phase = "grasping"
                self._transition_remaining = self.min_transition_frames
                self._width_history = [float(width)]
                is_moving = True
            elif self._transition_remaining == 0 and not is_moving:
                new_phase = "idle"
                self._held_obj_id = None

        # Drop stale held identity when the track was pruned.
        if (self._held_obj_id is not None
                and live_oids is not None
                and int(self._held_obj_id) not in live_oids):
            self._held_obj_id = None

        self._closed_prev = is_closed
        self._phase_prev = new_phase
        return {"phase": new_phase,
                "held_obj_id": self._held_obj_id,
                "gripper_width_m": width,
                "is_moving": is_moving,
                "held_pid": held_pid_now,
                "held_grasp_count": held_count_now}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_held_oid_at_grasp(
        self,
        tracker_state: TrackerState,
        T_wb: np.ndarray,
        T_bg: Optional[np.ndarray],
        T_bc: Optional[np.ndarray],
        detections: Optional[List[Dict[str, Any]]],
        depth: Optional[np.ndarray],
        K: Optional[np.ndarray],
        width: float,
    ) -> Tuple[Optional[int], Optional[Any], int]:
        """Returns ``(held_oid, held_pid, inside_count)``."""
        if self.detector is None:
            return self._nearest_live_track(tracker_state, T_wb, T_bg), None, 0
        joints = self._joints_now
        if joints is None and width is not None:
            joints = {"l_gripper_finger_joint": 0.5 * float(width),
                      "r_gripper_finger_joint": 0.5 * float(width)}
        decision = self.detector.select(
            detections=detections, depth=depth, K=K,
            T_wb=T_wb, T_bg=T_bg, T_bc=T_bc,
            joints=joints,
            tracker_state=tracker_state,
        )
        return decision.held_oid, decision.held_pid, decision.inside_count

    def _nearest_live_track(
        self,
        tracker_state: TrackerState,
        T_wb: np.ndarray,
        T_bg: Optional[np.ndarray],
    ) -> Optional[int]:
        if T_bg is None:
            return None
        ee_world = (T_wb @ T_bg)[:3, 3]
        best_oid, best_d = None, float("inf")
        for oid, mu_w in tracker_state.iter_world_centroids():
            d = float(np.linalg.norm(np.asarray(mu_w) - ee_world))
            if d < best_d:
                best_d, best_oid = d, oid
        if best_oid is not None and best_d <= self.grasp_radius:
            return int(best_oid)
        return None

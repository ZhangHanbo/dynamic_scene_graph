"""
Two-tier orchestrator — Gaussian-low-level variant.

Used when the low-level SLAM returns a single-Gaussian posterior
`PoseEstimate(T_wb, Σ_wb)`. Tracks objects in robot base frame via
`GaussianState`; Σ_wb enters only at world-frame output composition.

Parallel to `TwoTierOrchestrator` (RBPF variant) — the two orchestrators
share the slow tier (`PoseGraphOptimizer`), the trigger policy, and the
scene-graph / manipulation-set logic. They differ only in the fast-tier
state representation:

    RBPF variant (orchestrator.py)
        - N particles, per-particle world-frame EKF
        - Vision dual role (weight + update)
        - Works with particle or Gaussian backends; handles multimodal x
    Gaussian variant (this file)
        - Single base-frame EKF per object
        - No particle reweighting (no multimodality)
        - Strictly cheaper; avoids Paper-1 multi-frame overconfidence
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Any

import numpy as np

from pose_update.state.slam_interface import (
    PoseEstimate,
    collect_movable_masks, mask_out_movable, SlamBackend,
)
from pose_update.state.ekf_se3 import process_noise_for_phase
from pose_update.factor_graph import (
    PoseGraphOptimizer, Observation, RelationEdge, OptimizationResult,
)
from pose_update.state.gaussian_state import GaussianState
from pose_update.manipulation.gravity_predict import predict_landing_pose
from pose_update.manipulation.object_dynamics import lookup_dynamics
from pose_update.orchestrator import TriggerConfig  # reuse
from pose_update.perception.voxel_observability import VoxelObservability


class TwoTierOrchestratorGaussian:
    """Coordinator for the Gaussian-low-level pipeline."""

    # Initial per-object covariance (base frame; same scale as RBPF).
    _INIT_OBJ_COV = np.diag([0.05] * 3 + [0.05] * 3)
    # Extra slack for the rigid-attachment predict.
    _Q_MANIP_PER_STEP = np.diag([1e-6] * 3 + [1e-6] * 3)

    def __init__(self,
                 slam_backend: SlamBackend,
                 trigger: Optional[TriggerConfig] = None,
                 optimizer: Optional[PoseGraphOptimizer] = None,
                 T_bc: Optional[np.ndarray] = None,
                 iekf_iters: int = 2,
                 verbose: bool = False,
                 voxel_obs: Optional[VoxelObservability] = None,
                 gravity_predict_enabled: bool = True,
                 workspace_floor_z: float = -1.0):
        """Args:
            slam_backend: backend whose `step()` returns a `PoseEstimate`
                          (or `ParticlePose` — will be collapsed to Gaussian).
            trigger:      slow-tier trigger configuration.
            optimizer:    factor-graph optimizer (default PoseGraphOptimizer).
            T_bc:         camera-in-base transform; identity if None.
            iekf_iters:   inner IEKF iteration count for measurement update.
            voxel_obs:    optional voxel observability grid for the
                          gravity-aware predict at release. None disables
                          gravity_predict.
            gravity_predict_enabled: master switch for the release-time
                          gravity_predict overwrite. Defaults to True; the
                          method is still no-op when voxel_obs is None.
            workspace_floor_z: world-z used as the gravity raycast floor
                          when the column is fully unobserved.
        """
        self.slam = slam_backend
        self.trigger = trigger or TriggerConfig()
        self.optimizer = optimizer or PoseGraphOptimizer()
        self.iekf_iters = iekf_iters
        self.verbose = verbose
        self.voxel_obs = voxel_obs
        self.gravity_predict_enabled = gravity_predict_enabled
        self.workspace_floor_z = float(workspace_floor_z)

        self.state = GaussianState(T_bc=T_bc)

        # Object-level metadata (same shape as RBPF orchestrator).
        self.object_labels: Dict[int, str] = {}
        self.object_first_seen: Dict[int, int] = {}
        self.frames_since_obs: Dict[int, int] = {}
        self.T_oe: Dict[int, Optional[np.ndarray]] = {}

        self.frame_count = 0
        self.last_opt_frame = -1
        self.last_state: Dict[str, Any] = {
            "phase": "idle", "held_obj_id": None}

        self._cached_relations: List[RelationEdge] = []
        self._prev_T_bg: Optional[np.ndarray] = None
        # Diagnostic log of every gravity_predict overwrite (plan §C.2).
        self._gravity_predict_log: List[Dict[str, Any]] = []

    # --------------------------------------------------------------- #
    #  Backward-compat view (world-frame poses)
    # --------------------------------------------------------------- #

    @property
    def objects(self) -> Dict[int, Dict[str, Any]]:
        """World-frame view for legacy consumers."""
        out: Dict[int, Dict[str, Any]] = {}
        for oid, pe in self.state.collapsed_objects_world().items():
            out[oid] = {
                "T": pe.T,
                "cov": pe.cov,
                "label": self.object_labels.get(oid, "unknown"),
                "frames_since_observation":
                    self.frames_since_obs.get(oid, 0),
                "T_oe": self.T_oe.get(oid),
            }
        return out

    # --------------------------------------------------------------- #
    #  Public step
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
        """Process one frame. See `TwoTierOrchestrator.step` for the arg
        contract — identical here.

        Report keys: 'slam_raw', 'slam_pose', 'triggered', 'alpha',
        'residuals', 'objects'. No 'base_particles' / 'ess' / 'resampled'
        — those are RBPF-specific.
        """
        report: Dict[str, Any] = {}
        self._known_before_this_step = set(self.object_labels.keys())

        # 1. SLAM on movable-masked depth
        movable_mask = collect_movable_masks(detections, depth.shape)
        masked_depth = mask_out_movable(depth, movable_mask)
        slam_raw = self.slam.step(rgb, masked_depth, odom_prior)

        # 2. Ingest (collapses particles if backend returns ParticlePose)
        self.state.ingest_slam(slam_raw)
        slam_pose = self.state.collapsed_base()
        report["slam_raw"] = slam_raw
        report["slam_pose"] = slam_pose

        # 3. Fast tier
        self._fast_tier(detections, gripper_state, T_ec, T_bg)

        # 3b. Gravity-aware one-shot predict at release transition
        # (plan §C.2). Replaces the just-released oid's mean+cov with the
        # post-fall prediction. No-op when voxel_obs is None or the FSM
        # didn't just exit {holding, releasing}.
        self._maybe_gravity_predict(gripper_state)

        # 4. Scene graph
        self._cached_relations = self._recompute_relations()

        # 5. Slow tier
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

        # Bookkeeping
        self.last_state = dict(gripper_state)
        self._prev_T_bg = None if T_bg is None else T_bg.copy()
        self.frame_count += 1

        report["objects"] = {
            oid: {"T": e["T"].copy(),
                  "cov": e["cov"].copy(),
                  "label": e["label"]}
            for oid, e in self.objects.items()
        }
        return report

    # --------------------------------------------------------------- #
    #  Fast tier
    # --------------------------------------------------------------- #

    def _fast_tier(self,
                   detections: List[Dict[str, Any]],
                   gripper_state: Dict[str, Any],
                   T_ec: Optional[np.ndarray],
                   T_bg: Optional[np.ndarray]) -> None:
        phase = gripper_state.get("phase", "idle")
        held_id = gripper_state.get("held_obj_id")
        manipulation_set = self._get_manipulation_set(held_id)

        apply_rigid = (T_bg is not None
                       and self._prev_T_bg is not None
                       and bool(manipulation_set))
        delta_T_bg = (T_bg @ np.linalg.inv(self._prev_T_bg)
                      if apply_rigid else None)

        def Q_fn(oid: int) -> np.ndarray:
            if apply_rigid and oid in manipulation_set:
                # Q_manip is added inside rigid_attachment_predict.
                return np.zeros((6, 6))
            return process_noise_for_phase(
                phase=phase,
                is_target=(oid in manipulation_set),
                frames_since_observation=self.frames_since_obs.get(oid, 0),
                frame="base",
            )

        # Static-motion predict (base motion propagation + phase-aware Q).
        # Tracks not observed last frame are kept at their keyframe cov
        # (plan §C.1): only the deterministic mean update applies, so
        # the world-frame uncertainty grows solely via the lever-arm of
        # the current Σ_wb in collapsed_object_world.
        unobserved_oids = {oid for oid, fso
                           in self.frames_since_obs.items()
                           if fso > 0 and oid in self.state.objects}
        self.state.predict_static(Q_fn, unobserved_oids=unobserved_oids)

        # Per-frame tick.
        for oid in self.frames_since_obs:
            self.frames_since_obs[oid] += 1

        # Rigid-attachment predict for manipulation set (local, base-frame).
        if apply_rigid:
            for oid in manipulation_set:
                if oid in self.state.objects:
                    self.state.rigid_attachment_predict(
                        oid, delta_T_bg, self._Q_MANIP_PER_STEP)

        # Measurement update per detection.
        for det in detections:
            oid = det.get("id")
            if oid is None:
                continue

            if oid not in self.object_labels:
                T_co_meas = np.asarray(det["T_co"], dtype=np.float64)
                self.state.ensure_object(oid, T_co_meas, self._INIT_OBJ_COV)
                self.object_labels[oid] = det.get("label", "unknown")
                self.object_first_seen[oid] = self.frame_count
                self.frames_since_obs[oid] = 0
                self.T_oe[oid] = None
                continue

            R_icp = det.get("R_icp", np.eye(6) * 1e-3)
            T_co_meas = np.asarray(det["T_co"], dtype=np.float64)
            self.state.update_observation(
                oid=oid, T_co_meas=T_co_meas,
                R_icp=R_icp, iekf_iters=self.iekf_iters,
            )
            self.frames_since_obs[oid] = 0

            # T_oe lock at grasp onset (world-frame quantity).
            if (phase == "grasping"
                    and self.last_state.get("phase") != "grasping"
                    and oid == held_id
                    and T_ec is not None):
                pe_world = self.state.collapsed_object_world(oid)
                if pe_world is not None and self.state.T_wb is not None:
                    T_ew = self.state.T_wb @ T_ec
                    self.T_oe[oid] = np.linalg.inv(T_ew) @ pe_world.T

    # --------------------------------------------------------------- #
    #  Gravity-aware predict at release (plan §C.2)
    # --------------------------------------------------------------- #

    def _maybe_gravity_predict(self,
                                gripper_state: Dict[str, Any]) -> None:
        """Replace the EKF mean+cov for a just-released oid with the
        post-fall prediction from `pose_update.manipulation.gravity_predict`.

        Mirrors `pose_update/orchestrator.py:_maybe_gravity_predict`
        (the RBPF orchestrator's hook). No-ops when:
          * `gravity_predict_enabled` is False,
          * `voxel_obs` is None,
          * the FSM didn't just exit {holding, releasing},
          * the just-released oid is no longer tracked.

        After §C.1, the post-overwrite cov_bo (containing
        `gravity_predict`'s σ_xy/σ_z/σ_yaw=π) is FROZEN by subsequent
        per-frame predict_static calls (the bottle is unobserved).
        """
        if not self.gravity_predict_enabled:
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
        pe_w = self.state.collapsed_object_world(just_released)
        if pe_w is None:
            return
        label = self.object_labels.get(just_released)
        dyn = lookup_dynamics(label)
        # Live-object overlay: every other tracked oid contributes
        # (x, y, z, radius) to the raycast collision check.
        other_voxels = []
        for oid, pe_o in self.state.collapsed_objects_world().items():
            if oid == just_released:
                continue
            T = pe_o.T
            other_dyn = lookup_dynamics(self.object_labels.get(oid))
            other_voxels.append(
                (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
                 float(other_dyn.radius_m)))
        T_land, P_land, info = predict_landing_pose(
            T_release=pe_w.T,
            P_release=pe_w.cov,
            voxel_obs=self.voxel_obs,
            dyn=dyn,
            workspace_floor_z=self.workspace_floor_z,
            live_object_voxels=other_voxels,
        )
        ok = self.state.overwrite_object_pose(
            just_released, T_land, P_land)
        log_entry = info.as_dict()
        log_entry["oid"] = int(just_released)
        log_entry["frame"] = int(self.frame_count)
        log_entry["written_back"] = bool(ok)
        log_entry["label"] = label
        self._gravity_predict_log.append(log_entry)

    # --------------------------------------------------------------- #
    #  Slow tier (Option A — same as RBPF, operates on world-frame collapse)
    # --------------------------------------------------------------- #

    def _slow_tier(self,
                   slam_pose: PoseEstimate,
                   detections: List[Dict[str, Any]],
                   gripper_state: Dict[str, Any],
                   T_ec: Optional[np.ndarray]) -> OptimizationResult:
        priors: Dict[int, PoseEstimate] = dict(
            self.state.collapsed_objects_world())

        observations: List[Observation] = []
        for det in detections:
            oid = det.get("id")
            if oid is None or oid not in priors:
                continue
            observations.append(Observation(
                obj_id=oid,
                T_co=det["T_co"],
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

        # Inject back (world-frame mean → base-frame mean; keep Σ_bo).
        for oid, pe in result.posteriors.items():
            self.state.inject_posterior_world(oid, pe)

        if self.verbose:
            print(f"[slow tier gauss] α={result.alpha:.2f}, "
                  f"iters={result.num_iterations}")

        return result

    # --------------------------------------------------------------- #
    #  Manipulation set + relations + triggers — identical logic to RBPF
    # --------------------------------------------------------------- #

    def _get_manipulation_set(self,
                              held_id: Optional[int]) -> Set[int]:
        if held_id is None:
            return set()
        manipulated: Set[int] = {held_id}
        for _ in range(8):
            changed = False
            for edge in self._cached_relations:
                if edge.relation_type not in ("in", "on"):
                    continue
                if edge.child in manipulated and edge.parent not in manipulated:
                    manipulated.add(edge.parent)
                    changed = True
            if not changed:
                break
        return manipulated

    def _recompute_relations(self) -> List[RelationEdge]:
        collapsed = self.state.collapsed_objects_world()
        if len(collapsed) < 2:
            return []

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
            return []

        edges: List[RelationEdge] = []
        for (parent_id, child_id, rel_type), score in scores.items():
            if rel_type not in ("on", "in"):
                continue
            edges.append(RelationEdge(
                parent=parent_id, child=child_id,
                relation_type=rel_type, score=score,
            ))
        return edges

    def _should_trigger(self,
                        gripper_state: Dict[str, Any],
                        detections: List[Dict[str, Any]]) -> bool:
        last_phase = self.last_state.get("phase", "idle")
        cur_phase = gripper_state.get("phase", "idle")

        if self.trigger.on_grasp and \
                last_phase != "grasping" and cur_phase == "grasping":
            return True
        if self.trigger.on_release and \
                last_phase == "releasing" and cur_phase != "releasing":
            return True

        if self.trigger.on_new_object:
            seen_ids = {d.get("id") for d in detections
                        if d.get("id") is not None}
            known = getattr(self, "_known_before_this_step", set())
            if not seen_ids.issubset(known):
                return True

        if self.trigger.periodic_every_n_frames > 0:
            if (self.frame_count - self.last_opt_frame
                    >= self.trigger.periodic_every_n_frames):
                return True

        return False

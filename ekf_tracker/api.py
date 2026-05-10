"""Public facade for the EKF tracker: :class:`EkfTracker` plus the :class:`EkfObject` and :class:`SceneView` snapshots."""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ekf_tracker.manipulation.grasp_owner_detector import GraspOwnerDetector
from ekf_tracker.manipulation.gravity_predict import predict_landing_pose
from ekf_tracker.manipulation.gripper_state_inferrer import _GripperStateInferrer
from ekf_tracker.config import BernoulliConfig, TriggerConfig
from ekf_tracker.orchestrator_gaussian import TwoTierOrchestratorGaussian
from ekf_tracker.relations.relation_orchestrator import RelationOrchestrator
from ekf_tracker.relations.relation_utils import expand_held_with_relations
from perception.voxel_observability import VoxelObservability
from utils.object_dynamics import lookup_dynamics
from utils.robot_models import create_gripper_geometry


# ─────────────────────────────────────────────────────────────────────
#  Public dataclasses
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EkfObject:
    """Read-only snapshot of one tracked object (world frame)."""
    id: int
    label: str
    pose: np.ndarray         # (4, 4) world-frame mean
    cov: np.ndarray          # (6, 6) world-frame tangent covariance
    r: float                 # Bernoulli existence


@dataclass
class SceneView:
    """Read-only snapshot of the scene + relation graph."""
    objects: Dict[int, EkfObject] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
#  Default configuration --- loaded bit-exactly from
#  ekf_tracker/configs/default.yaml. The two-hierarchy contract:
#  default + customization YAMLs are the only sources of defaults.
# ─────────────────────────────────────────────────────────────────────

def _default_bernoulli_cfg(K: np.ndarray,
                           image_shape: Tuple[int, int] = (480, 640),
                           ) -> BernoulliConfig:
    """Build a default :class:`BernoulliConfig` from ``configs/default.yaml``."""
    from ekf_tracker.configs import load_config, to_bernoulli_config
    return to_bernoulli_config(load_config(),
                                K=K, image_shape=image_shape)


# ─────────────────────────────────────────────────────────────────────
#  Mask decode helper (for detect())
# ─────────────────────────────────────────────────────────────────────

def _decode_mask_b64_to_uint8(mask_b64: str) -> np.ndarray:
    """Base64 PNG -> uint8 (H, W) {0, 1} mask."""
    import cv2
    data = base64.b64decode(mask_b64.encode("ascii"))
    arr = np.frombuffer(data, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return (im > 127).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
#  Public facade
# ─────────────────────────────────────────────────────────────────────

class EkfTracker:
    """Five-method facade over the fast tier, perception pipeline, manipulation, and relations."""

    def __init__(
        self,
        K: np.ndarray,
        *,
        T_bc: Optional[np.ndarray] = None,
        robot_type: str = "fetch",
        relation_backend: Optional[str] = None,
        relation_cache_dir: Optional[str] = None,
        pose_method: str = "icp_chain",
        owl_server: Optional[str] = None,
        sam2_server: Optional[str] = None,
        bernoulli_cfg: Optional[BernoulliConfig] = None,
        voxel_obs: Optional[VoxelObservability] = None,
        image_shape: Tuple[int, int] = (480, 640),
        trigger: Optional[TriggerConfig] = None,
        # Deprecated alias for owl_server; kept for back-compat.
        det_server: Optional[str] = None,
    ):
        self.K = np.asarray(K, dtype=np.float64)
        # Load defaults from the canonical YAML when not explicitly
        # supplied. The two-hierarchy contract: defaults live in
        # ekf_tracker/configs/default.yaml; nothing else.
        from ekf_tracker.configs import (
            load_config, to_bernoulli_config, to_trigger_config,
        )
        _ekf_cfg_dict = load_config()
        self._cfg = bernoulli_cfg or to_bernoulli_config(
            _ekf_cfg_dict, K=self.K, image_shape=image_shape)
        # The two-tier subclass is a strict superset of GaussianEkfTracker:
        # it inherits the entire fast tier and adds the slow tier used by
        # `smooth()`. The default YAML's `trigger:` section mirrors the
        # never-fires policy that reproduces `visualize_ekf_tracking.main()`
        # exactly; callers can opt into auto-triggering via a
        # customization YAML or an explicit `trigger=` kwarg.
        self._trigger = trigger or to_trigger_config(_ekf_cfg_dict)
        from ekf_tracker.configs import (
            build_visibility_kwargs, build_det_dedup_kwargs,
            build_process_noise_schedule, build_fast_tier_noise_config,
        )
        self._tracker = TwoTierOrchestratorGaussian(
            self.K, self._cfg,
            pose_method=pose_method,
            T_bc=T_bc,
            trigger=self._trigger,
            visibility_kwargs=build_visibility_kwargs(_ekf_cfg_dict),
            det_dedup_kwargs=build_det_dedup_kwargs(_ekf_cfg_dict),
            process_noise_schedule=build_process_noise_schedule(_ekf_cfg_dict),
            fast_tier_noise=build_fast_tier_noise_config(_ekf_cfg_dict),
        )
        from ekf_tracker.configs import (
            build_gripper_phase_tracker_kwargs,
            build_grasp_owner_detector_kwargs,
            build_voxel_observability_kwargs,
            build_voxel_integrate_kwargs,
            build_gravity_predict_kwargs,
            build_held_set_expansion_kwargs,
            build_object_dynamics_table,
        )
        self._voxel_integrate_kwargs = build_voxel_integrate_kwargs(_ekf_cfg_dict)
        self._gravity_predict_kwargs = build_gravity_predict_kwargs(_ekf_cfg_dict)
        self._held_set_expansion_kwargs = build_held_set_expansion_kwargs(_ekf_cfg_dict)
        (self._dynamics_default,
         self._dynamics_table,
         self._dynamics_footprint) = build_object_dynamics_table(_ekf_cfg_dict)
        self._gripper_geom = create_gripper_geometry(robot_type=robot_type)
        self._grasp_detector = GraspOwnerDetector(
            gripper=self._gripper_geom,
            **build_grasp_owner_detector_kwargs(_ekf_cfg_dict),
        )
        self._grip_inferrer = _GripperStateInferrer(
            detector=self._grasp_detector,
            **build_gripper_phase_tracker_kwargs(_ekf_cfg_dict),
        )
        from ekf_tracker.configs import build_relation_orchestrator_kwargs
        _rel_kwargs = build_relation_orchestrator_kwargs(_ekf_cfg_dict)
        # The `relation_backend` constructor kwarg overrides the YAML
        # for ad-hoc per-call backend selection; everything else comes
        # from the YAML.
        if relation_backend is not None:
            _rel_kwargs["backend"] = relation_backend
        self._relation_pipeline = RelationOrchestrator(
            cache_dir=relation_cache_dir,
            **_rel_kwargs,
        )
        self._voxel_obs = voxel_obs or VoxelObservability(
            **build_voxel_observability_kwargs(_ekf_cfg_dict),
        )
        # OWL + SAM2 streaming server URLs for live `detect()`. Defaults
        # come from `scripts/rosbag2dataset/server_configs.py`. Either
        # endpoint is hit only on the first call to `detect()`; if neither
        # the user nor the env is configured, we fall back to those defaults.
        from scripts.rosbag2dataset.server_configs import (
            OWL_SERVER_URL as _DEF_OWL_URL,
            SAM2_SERVER_URL as _DEF_SAM2_URL,
        )
        # Back-compat: legacy `det_server` arg used to mean the OWL+SAM
        # single-image endpoint. Treat it as an OWL-server alias.
        self._owl_url = owl_server or det_server or _DEF_OWL_URL
        self._sam2_url = sam2_server or _DEF_SAM2_URL

        # Phase tracking for the gravity-predict hook.
        self._prev_phase: str = "idle"
        self._prev_held: Optional[int] = None
        self._gravity_predict_log: List[Dict[str, Any]] = []
        self._r_history: Dict[int, List[Tuple[int, float]]] = {}
        self._frame_idx: int = 0

        # Most-recent debug + augmented detections for caller introspection.
        self._last_dbg: Optional[Dict[str, Any]] = None
        self._last_dets_with_pose: Optional[List[Dict[str, Any]]] = None

        # Most-recent slam / extrinsic / held-state context. Used by
        # `smooth()` to invoke the slow tier without the caller having
        # to re-supply everything that `step()` already saw.
        self._last_slam_pose: Optional[np.ndarray] = None
        self._last_T_bc: Optional[np.ndarray] = None
        self._last_T_bg: Optional[np.ndarray] = None
        self._last_held_seed: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────
    #  detect()
    # ─────────────────────────────────────────────────────────────────

    def detect(self,
               rgb: np.ndarray,
               vocabulary: List[str],
               history: Any = None,
               ) -> Tuple[List[Dict[str, Any]], Any]:
        """Run the OWLv2 + SAM2-streaming detection pipeline on one RGB frame."""
        from ekf_tracker.perception_pipeline import LiveDetectionPipeline
        if history is None:
            history = LiveDetectionPipeline(
                owl_url=self._owl_url,
                sam2_url=self._sam2_url,
            )
            history.start()
        elif not isinstance(history, LiveDetectionPipeline):
            raise TypeError(
                "EkfTracker.detect: `history` must be either None or a "
                "LiveDetectionPipeline returned by a previous detect() "
                f"call; got {type(history).__name__}.")
        detections = history.step(rgb, vocabulary)
        return detections, history

    # ─────────────────────────────────────────────────────────────────
    #  step() — exact mirror of main()'s per-frame pipeline (3492-3680)
    # ─────────────────────────────────────────────────────────────────

    def step(
        self,
        detections: List[Dict[str, Any]],
        rgb: np.ndarray,
        depth: np.ndarray,
        *,
        slam_pose: np.ndarray,
        T_bc: Optional[np.ndarray] = None,
        T_bg: Optional[np.ndarray] = None,
        gripper_width: Optional[float] = None,
        joints: Optional[Dict[str, float]] = None,
    ) -> SceneView:
        """One frame of the canonical pipeline (predict → associate → update → birth → prune); returns a :class:`SceneView`."""
        K = self.K
        T_bc_for_vox = T_bc if T_bc is not None else np.eye(4)

        # 1. Voxel observability integrate (3507-3519).
        try:
            self._voxel_obs.integrate_depth(
                depth=depth.astype(np.float32),
                K=K,
                T_cw=(np.asarray(slam_pose, dtype=np.float64)
                      @ np.asarray(T_bc_for_vox, dtype=np.float64)),
                **self._voxel_integrate_kwargs,
            )
        except Exception as e:
            print(f"[WARN] voxel_obs.integrate_depth failed at fr "
                  f"{self._frame_idx}: {e}")

        # 2. Gripper state inference (3529-3532).
        gripper_state = self._grip_inferrer.step(
            width=gripper_width,
            tracker=self._tracker,
            T_wb=slam_pose,
            T_bg=T_bg,
            detections=detections,
            depth=depth,
            K=K,
            T_bc=T_bc,
            joints=joints,
        )

        # 3. held_seed extraction (3533-3542).
        held_seed = gripper_state.get("held_obj_id")
        if gripper_state.get("phase") == "releasing":
            held_seed = None

        # 4. det_to_oid map from tracker.sam2_tau (3544-3558).
        tau_to_oid = {int(t): int(o)
                      for o, t in self._tracker.sam2_tau.items()
                      if t is not None}
        det_to_oid: Dict[int, int] = {}
        for di, d in enumerate(detections):
            pid = d.get("id")
            if pid is None:
                continue
            try:
                oid = tau_to_oid.get(int(pid))
            except (TypeError, ValueError):
                continue
            if oid is not None:
                det_to_oid[di] = oid

        # 5. Live tracks snapshot (3559-3572).
        live_oids: Set[int] = {int(o)
                                for o in self._tracker.object_labels.keys()}
        live_tracks: Dict[int, Dict[str, Any]] = {}
        T_wb_arr = np.asarray(slam_pose, dtype=np.float64)
        for oid in live_oids:
            pe = self._tracker.state.collapsed_object_base(int(oid))
            if pe is None:
                continue
            mu_b = np.asarray(pe.T, dtype=np.float64)[:3, 3]
            mu_w = (T_wb_arr @ np.append(mu_b, 1.0))[:3]
            live_tracks[int(oid)] = {
                "xyz_w": mu_w.tolist(),
                "label": self._tracker.object_labels.get(int(oid), "?"),
            }

        # 6. Relation pipeline maybe-update (3573-3579).
        rel_summary = self._relation_pipeline.maybe_update(
            frame=self._frame_idx,
            rgb=rgb,
            detections=detections,
            det_to_oid=det_to_oid,
            current_phase=gripper_state["phase"],
            current_oids=live_oids,
            held_oid=held_seed,
            live_tracks=live_tracks,
        )

        # 7. Held set expansion (3581-3586).
        held_oids = expand_held_with_relations(
            held_seed, self._relation_pipeline.edges,
            **self._held_set_expansion_kwargs)
        held_oids = {o for o in held_oids
                     if o in self._tracker.state.objects}

        # 8. Tracker step (3588-3597).
        dbg, dets_with_pose = self._tracker.step(
            rgb=rgb,
            depth=depth,
            T_wb=slam_pose,
            detections=detections,
            phase=gripper_state["phase"],
            T_bc=T_bc,
            T_bg=T_bg,
            held_oids=held_oids,
            held_seed=held_seed,
            relation_edges=self._relation_pipeline.edges,
        )

        # 9. Self-merge id reconciliation (3599-3603).
        self._grip_inferrer.apply_merges(dbg.get("self_merges", []))
        gripper_state["held_obj_id"] = self._grip_inferrer._held_obj_id
        self._relation_pipeline.remap_after_merges(
            dbg.get("self_merges", []))
        dbg["gripper_state"] = dict(gripper_state)

        # 10. Gravity-aware predict at release transition (3606-3656).
        cur_phase = gripper_state.get("phase", "idle")
        # Fire gravity-predict at the holding→{releasing,idle} edge --- the
        # FRAME the gripper opens --- not at releasing→idle. Waiting for
        # the FSM's release dwell (`min_transition_frames` frames) leaves
        # the apple hanging in mid-air at its in-gripper height during the
        # dwell, then teleporting on the dwell's last frame. Triggering
        # earlier collapses that mid-air hang to zero frames. A re-grasp
        # transition holding→grasping is excluded so a width-jitter flap
        # cannot install a landing prior on a still-held object.
        if (self._prev_phase == "holding"
                and cur_phase in ("releasing", "idle")
                and self._prev_held is not None
                and self._prev_held in self._tracker.state.objects):
            try:
                self._gravity_predict_at_release(
                    slam_pose, dbg,
                    depth=depth, T_bc=T_bc_for_vox)
            except Exception as e:
                print(f"[WARN] gravity_predict failed at fr "
                      f"{self._frame_idx}: {e}")

        # 11. Update phase tracking (3662-3665).
        self._prev_phase = cur_phase
        held_now = gripper_state.get("held_obj_id")
        if held_now is not None:
            self._prev_held = int(held_now)

        # Expose held + relation snapshot for downstream introspection
        # (matches main()'s lines 3669-3674).
        dbg["held_oids_used"] = sorted(int(o) for o in held_oids)
        dbg["relations"] = [
            {"parent": int(e.parent), "child": int(e.child),
             "type": str(e.relation_type), "score": float(e.score)}
            for e in self._relation_pipeline.edges
        ]
        if rel_summary.get("fired"):
            dbg["relation_call"] = rel_summary

        # 12. r_history append (3679-3680).
        for oid, tr in dbg.get("post_update_tracks", {}).items():
            self._r_history.setdefault(int(oid), []).append(
                (self._frame_idx, float(tr["r"])))

        self._last_dbg = dbg
        self._last_dets_with_pose = dets_with_pose
        # Cache slow-tier context so `smooth()` can re-invoke without
        # the caller re-supplying everything.
        self._last_slam_pose = np.asarray(slam_pose, dtype=np.float64).copy()
        self._last_T_bc = (np.asarray(T_bc, dtype=np.float64).copy()
                           if T_bc is not None else None)
        self._last_T_bg = (np.asarray(T_bg, dtype=np.float64).copy()
                           if T_bg is not None else None)
        self._last_held_seed = held_seed
        self._frame_idx += 1
        return self.get_scene()

    # ─────────────────────────────────────────────────────────────────
    #  Gravity-aware predict helper (3617-3656)
    # ─────────────────────────────────────────────────────────────────

    def _gravity_predict_at_release(self,
                                     slam_pose: np.ndarray,
                                     dbg: Dict[str, Any],
                                     *,
                                     depth: Optional[np.ndarray] = None,
                                     T_bc: Optional[np.ndarray] = None,
                                     ) -> None:
        prev_held = int(self._prev_held) if self._prev_held is not None else None
        if prev_held is None:
            return
        pe_w = self._tracker.state.collapsed_object_world(prev_held)
        if pe_w is None:
            return
        label = self._tracker.object_labels.get(prev_held)
        dyn = lookup_dynamics(
            label, default=self._dynamics_default, table=self._dynamics_table)
        # Live-object overlay: every other live oid contributes.
        other_voxels: List[Tuple[float, float, float, float]] = []
        for other_oid, other_pe in (
                self._tracker.state.collapsed_objects_world() or {}).items():
            if int(other_oid) == prev_held:
                continue
            T_o = other_pe.T
            other_dyn = lookup_dynamics(
                self._tracker.object_labels.get(int(other_oid)),
                default=self._dynamics_default, table=self._dynamics_table)
            other_voxels.append((
                float(T_o[0, 3]), float(T_o[1, 3]), float(T_o[2, 3]),
                float(other_dyn.radius_m)))
        # Fix B plumbing: per-frame depth + T_cw enable the visibility
        # override; if any input is missing, predict_landing_pose silently
        # falls through to Fix A / parametric branches.
        T_cw = None
        image_shape = None
        if depth is not None and T_bc is not None:
            T_cw = (np.asarray(slam_pose, dtype=np.float64)
                    @ np.asarray(T_bc, dtype=np.float64))
            image_shape = (int(depth.shape[0]), int(depth.shape[1]))
        T_land, P_land, info = predict_landing_pose(
            T_release=pe_w.T,
            P_release=pe_w.cov,
            voxel_obs=self._voxel_obs,
            dyn=dyn,
            shape_footprint_factors=self._dynamics_footprint,
            live_object_voxels=other_voxels,
            K=self.K,
            depth=depth,
            T_cw=T_cw,
            image_shape=image_shape,
            **self._gravity_predict_kwargs,
        )
        # Write back into base-frame state.
        obj = self._tracker.state.objects.get(prev_held)
        if obj is not None:
            T_wb_arr = np.asarray(slam_pose, dtype=np.float64)
            obj.mu_bo = np.linalg.inv(T_wb_arr) @ T_land
            obj.cov_bo = P_land.copy()
        log_entry = info.as_dict()
        log_entry["oid"] = prev_held
        log_entry["frame"] = int(self._frame_idx)
        log_entry["label"] = label
        self._gravity_predict_log.append(log_entry)
        dbg["gravity_predict"] = log_entry

    # ─────────────────────────────────────────────────────────────────
    #  get_scene()
    # ─────────────────────────────────────────────────────────────────

    def get_scene(self) -> SceneView:
        """Read-only snapshot of the current tracked objects and relations."""
        objects: Dict[int, EkfObject] = {}
        world = self._tracker.state.collapsed_objects_world() or {}
        for oid, pe in world.items():
            oid_i = int(oid)
            objects[oid_i] = EkfObject(
                id=oid_i,
                label=self._tracker.object_labels.get(oid_i, "?"),
                pose=np.asarray(pe.T, dtype=np.float64).copy(),
                cov=np.asarray(pe.cov, dtype=np.float64).copy(),
                r=float(self._tracker.existence.get(oid_i, 0.0)),
            )
        relations = [
            {"parent": int(e.parent),
             "child": int(e.child),
             "type": str(e.relation_type),
             "score": float(e.score)}
            for e in self._relation_pipeline.edges
        ]
        return SceneView(objects=objects, relations=relations)

    # ─────────────────────────────────────────────────────────────────
    #  get_points()
    # ─────────────────────────────────────────────────────────────────

    def get_points(self, object_id: int) -> np.ndarray:
        """Accumulated ICP point cloud for ``object_id`` in world coordinates."""
        ref = self._tracker.pose_est._refs.get(int(object_id))
        if ref is None or getattr(ref, "ref_points", None) is None:
            return np.empty((0, 3), dtype=np.float32)
        ref_pts_obj = np.asarray(ref.ref_points, dtype=np.float64)
        if ref_pts_obj.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        pe = self._tracker.state.collapsed_object_world(int(object_id))
        if pe is None:
            return np.empty((0, 3), dtype=np.float32)
        T_wo = np.asarray(pe.T, dtype=np.float64)
        pts_h = np.hstack([ref_pts_obj,
                           np.ones((ref_pts_obj.shape[0], 1))]).T
        pts_w = (T_wo @ pts_h).T[:, :3]
        return pts_w.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────
    #  smooth() — slow-tier pose-graph optimisation
    # ─────────────────────────────────────────────────────────────────

    def smooth(self) -> SceneView:
        """Run the slow-tier :class:`PoseGraphOptimizer` and return the refreshed scene."""
        if self._last_slam_pose is None:
            # Nothing to smooth over yet — just return the empty scene.
            return self.get_scene()
        self._tracker.smooth(
            T_wb=self._last_slam_pose,
            relation_edges=list(self._relation_pipeline.edges),
            held_oid=self._last_held_seed,
            T_bg=self._last_T_bg,
            T_bc=self._last_T_bc,
        )
        return self.get_scene()

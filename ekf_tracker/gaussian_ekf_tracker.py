"""Gaussian-EKF tracker (base-frame storage, Bernoulli existence).

This is the canonical fast-tier EKF used by the visualization driver
and the public ``EkfTracker`` facade.

Owns:
  * ``GaussianState`` (object-in-base-frame Gaussian belief)
  * ``PoseEstimator`` (per-detection ICP)
  * ``ChainStore`` (per-track loop-closure-aware observation chains)
  * Bernoulli existence ``r`` per track + soft label histories
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from perception.association import hungarian_associate, oracle_associate
from ekf_tracker.state.bernoulli import (
    r_predict, r_assoc_update_loglik, r_miss_update, r_birth,
)
from utils.ekf_se3 import (
    huber_weight, process_noise_for_phase, saturate_covariance,
)
from ekf_tracker.state.gaussian_state import GaussianState
from perception.det_dedup import suppress_subpart_detections
from perception.icp_pose import (
    PoseEstimator, centroid_cam_from_mask, _back_project,
)
from ekf_tracker.state.obs_chain import ChainStore
from ekf_tracker.config import BernoulliConfig
from ekf_tracker.birth_gate import _PendingBirth, birth_admissible
from utils.slam_interface import PoseEstimate
from perception.visibility import visibility_p_v


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


class GaussianEkfTracker:
    """Base-frame Gaussian-EKF tracker with Bernoulli existence.

    Refactored to base-frame storage (bernoulli_ekf.tex §1.1, §3).

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

    # Phases under which rigid-attachment predict is active. `releasing`
    # is intentionally excluded: even though `held_obj_id` may still be
    # set during the FSM's transition window, the gripper has opened and
    # the object must not ride the base any further (apple_drop fr 269-273
    # regression).
    _RIGID_PHASES = frozenset(("grasping", "holding"))

    @staticmethod
    def should_apply_rigid(T_bg, prev_T_bg, manipulation_set, phase) -> bool:
        """Predicate for the rigid-attachment branch of the fast tier.

        Rigid attach is applied iff (a) we have two consecutive proprio
        samples to form ΔT_bg, (b) some object is in the manipulation
        set, AND (c) the gripper is actually closed around the object
        (phase in {grasping, holding}).
        """
        return (T_bg is not None
                and prev_T_bg is not None
                and bool(manipulation_set)
                and phase in GaussianEkfTracker._RIGID_PHASES)

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

            # World-frame collapsed view: composes T_wb · T_bo and lifts
            # Σ_bo through Ad(T_bo^-1) Σ_wb Ad(T_bo^-1)^T + Σ_bo, so the
            # ellipse rendered at world-frame position uses world-frame
            # tangent. Falls back to None when SLAM hasn't been ingested.
            try:
                pe_world = self.state.collapsed_object_world(oid)
            except Exception:
                pe_world = None
            cov_world = (pe_world.cov.copy()
                         if pe_world is not None else None)

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
                "cov_world": cov_world,          # WORLD-frame tangent, or None
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
        """Thin wrapper around :func:`perception.birth_gating.is_near_live_track`.

        See that function for behaviour. This method simply provides
        the tracker context (T_wb, T_bc, held oid, T_we, cfg) so the
        production helper can be called.
        """
        from perception.birth_gating import (
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
        """Per-track p_v via depth ray-tracing (see `perception.visibility`).

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
        # Tracks not observed last frame keep their cov_bo at the
        # keyframe value (plan §C.1) -- only the deterministic mean
        # update applies, so world uncertainty grows solely via the
        # current Σ_wb-lever in collapsed_object_world.
        unobserved_oids = {oid for oid, fso
                           in self.frames_since_obs.items()
                           if fso > 0 and oid in self.state.objects}
        self.state.predict_static(Q_fn, P_max=self.cfg.P_max,
                                   skip_oids=held_oids,
                                   unobserved_oids=unobserved_oids)

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
                max_residual_m=getattr(
                    self.cfg, "max_residual_m", None),
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

"""
Gaussian-backend state for Layer 2.

When the low-level SLAM returns a single-Gaussian posterior
`PoseEstimate(T_wb, Σ_wb)`, this module tracks objects in the ROBOT BASE
FRAME, keeping the filter state independent of Σ_wb.

Why base frame (derivation)
───────────────────────────
The observation model in world frame is
    z = h(x, o) + v,   h(x, o) = T_wb⁻¹ · T_wo,   v ~ N(0, R_icp).
Marginalizing over x = T_wb ~ N(μ_wb, Σ_wb) and folding the linearized
contribution into R (the "Paper 1" recipe) gives
    R_eff = H_x Σ_wb H_xᵀ + R_icp.
That is correct for a SINGLE frame. Across frames it treats Σ_wb as
i.i.d., which is false — x_t and x_{t+1} are nearly the same random
variable. Recursive fusion under that fiction drives Σ_wo below Σ_wb,
which is physically impossible for a static object.

Base-frame storage removes Σ_wb from the recursion. With a known fixed
camera-to-base T_bc from kinematics, the observation becomes
    T_bo_meas = T_bc · T_co_meas,   noise Ad(T_bc) · R_icp · Ad(T_bc)ᵀ,
which is deterministic in x. The EKF state (μ_bo, Σ_bo) evolves without
ever touching Σ_wb. Σ_wb enters only once, at output time, via the
composition
    T_wo = T_wb · T_bo,   Σ_wo = Ad(T_bo⁻¹) · Σ_wb · Ad(·)ᵀ + Σ_bo,
which lower-bounds Σ_wo by the projected Σ_wb. Correct physics.

Regimes covered
───────────────
* Static object: base-frame pose changes as the base moves, but its
  world-frame pose is constant. We propagate μ_bo(t) = inv(ΔT_wb) · μ_bo(t−1)
  each frame (ΔT_wb from consecutive SLAM means). Σ_bo inflates by a
  phase-aware Q. Uncertainty in ΔT_wb is left for the output composition
  to absorb (it never enters the recursion).

* Manipulated object: rigidly attached to the gripper.
  T_bo(t) = ΔT_bg · T_bo(t−1), with ΔT_bg = T_bg(t)·inv(T_bg(t−1)) from
  proprioception. This predict is *local* — no T_wb coupling — which is
  cleaner than the particle version's T_wb(t)·ΔT_bg·inv(T_wb(t−1)) form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from pose_update.state.ekf_se3 import (
    se3_exp, se3_log, se3_adjoint, ekf_predict, saturate_covariance,
)
from pose_update.state.object_belief import (
    floor_diag as _floor_diag,
    lift_measurement_base,
    predict_ad_conjugate,
    innovation_from_belief,
    joseph_update,
    merge_info_sum,
    LOG_EPS,
)
from pose_update.state.slam_interface import (
    PoseEstimate, ParticlePose, as_gaussian,
)


# ─────────────────────────────────────────────────────────────────────
# Per-object belief (single Gaussian, base frame)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GaussianObjectBelief:
    """Gaussian SE(3) belief about one object, in ROBOT BASE frame.

    Attributes:
        mu_bo:  (4, 4) SE(3) mean  (object-in-base).
        cov_bo: (6, 6) covariance in se(3) tangent at mu_bo, [v, ω] ordering.
    """
    mu_bo: np.ndarray
    cov_bo: np.ndarray


# ─────────────────────────────────────────────────────────────────────
# GaussianState
# ─────────────────────────────────────────────────────────────────────

class GaussianState:
    """Per-object EKFs in base frame. Σ_wb enters only at output time.

    The state holds the *current* SLAM posterior (μ_wb, Σ_wb), the
    *previous* one (used by `predict_static` to propagate base motion),
    and a dict of per-object base-frame beliefs. Object-level metadata
    (labels, phase state) is an orchestrator concern — this class is
    the pure filter engine.

    If the low-level SLAM returns a ParticlePose, we collapse it to a
    Gaussian via `to_gaussian()` at ingest — this is the Gaussian
    pipeline, so multimodality is intentionally dropped. If you want
    multimodality preserved, use `RBPFState` instead.
    """

    _LOG_EPS = -1e18

    def __init__(self, T_bc: Optional[np.ndarray] = None,
                 P_min_diag: Optional[np.ndarray] = None):
        """Args:
            T_bc: (4, 4) camera-in-base transform at construction time.
                  If None, assumed to be identity (camera frame == base
                  frame). For a Fetch-style head-mounted camera this is
                  the kinematic chain
                      base_link -> torso_lift_link -> head_pan_link
                      -> head_tilt_link -> head_camera_link
                      -> head_camera_rgb_frame
                      -> head_camera_rgb_optical_frame
                  evaluated at the current head joints. May be updated
                  per-frame via `set_camera_extrinsic(T_bc_now)` --
                  required when the head pans or tilts during a
                  trajectory, since otherwise the predict-then-measure
                  pipeline would silently treat camera motion as object
                  motion.
            P_min_diag: (6,) per-axis lower bound on P_bo's diagonal,
                  ordering [v_x, v_y, v_z, w_x, w_y, w_z]. After every
                  predict and every update, the diagonal of P_bo is
                  lifted so that P_ii >= P_min_diag_i. Prevents the EKF
                  posterior from shrinking below realistic per-frame
                  perception jitter, which would otherwise make the
                  chi^2_3 outer gate spuriously reject 2-3 cm position
                  drift on the next frame. Default: None = no floor
                  (paper-baseline behaviour). Recommended:
                  diag([0.005**2]*3 + [0.05**2]*3) -- 5 mm trans / 0.05
                  rad rot. Off-diagonal correlations are preserved.
        """
        self.objects: Dict[int, GaussianObjectBelief] = {}

        # Current & previous SLAM posterior
        self.T_wb: Optional[np.ndarray] = None
        self.Sigma_wb: Optional[np.ndarray] = None
        self.prev_T_wb: Optional[np.ndarray] = None

        # Camera-in-base. Mutable per-frame via `set_camera_extrinsic`
        # so the pipeline tracks head pan/tilt + torso lift correctly.
        # Cached `_Ad_bc` is the SE(3) adjoint of the current `T_bc`,
        # used by every measurement/innovation/update path to convert
        # camera-frame ICP noise into base-frame tangent noise.
        self.T_bc: np.ndarray = (
            np.eye(4, dtype=np.float64) if T_bc is None
            else np.asarray(T_bc, dtype=np.float64).copy()
        )
        self._Ad_bc: np.ndarray = se3_adjoint(self.T_bc)

        # Per-axis covariance floor (None disables).
        self.P_min_diag: Optional[np.ndarray] = (
            np.asarray(P_min_diag, dtype=np.float64).reshape(6).copy()
            if P_min_diag is not None else None)

    # --------------------------------------------------------------- #
    #  Camera extrinsic (per-frame)
    # --------------------------------------------------------------- #

    def set_camera_extrinsic(self, T_bc: np.ndarray) -> None:
        """Update the base-to-camera transform for the current frame.

        Should be called by the orchestrator at the start of every
        frame, BEFORE any ICP / innovation / update / visibility call.
        T_bc is computed from the current torso/head joint positions
        (chain `base_link -> torso_lift_link -> head_pan_link ->
        head_tilt_link -> head_camera_link -> head_camera_rgb_frame ->
        head_camera_rgb_optical_frame`); see
        `pose_update.manipulation.fetch_kinematics.T_bc_from_joints` /
        `pose_update.manipulation.fetch_kinematics.T_bc_from_tf_dict`.

        Why per-frame: when the head pans or tilts while the base is
        stationary, the camera moves but the base does not. The
        per-frame `T_bc` is the only way the EKF can attribute the
        resulting `T_co` change to camera motion (kinematic, no
        innovation) instead of to object motion (innovation, gate
        rejection).

        State stored as `T_bo` is unaffected by `T_bc` changes (the
        derivation `T_bo = T_wb^{-1} T_wo` is `T_bc`-free); only the
        measurement lift `T_bo_meas = T_bc T_co_meas` and the
        camera-frame prior `T_co_pred = T_bc^{-1} T_bo` rely on
        `self.T_bc`.
        """
        T_bc = np.asarray(T_bc, dtype=np.float64)
        if T_bc.shape != (4, 4):
            raise ValueError(f"T_bc must be (4, 4), got {T_bc.shape}")
        self.T_bc = T_bc.copy()
        self._Ad_bc = se3_adjoint(self.T_bc)

    # --------------------------------------------------------------- #
    #  Inspection
    # --------------------------------------------------------------- #

    @property
    def initialized(self) -> bool:
        return self.T_wb is not None

    # --------------------------------------------------------------- #
    #  SLAM ingestion (collapses ParticlePose → PoseEstimate)
    # --------------------------------------------------------------- #

    def ingest_slam(self, slam_result) -> None:
        """Update (T_wb, Σ_wb) from the latest SLAM result.

        Cached `prev_T_wb` is used by `predict_static` to form ΔT_wb.
        If the backend returns a `ParticlePose`, we moment-match to a
        Gaussian here (the Gaussian pipeline cannot represent
        multimodality).
        """
        pe = as_gaussian(slam_result)
        self.prev_T_wb = None if self.T_wb is None else self.T_wb.copy()
        self.T_wb = pe.T.copy()
        self.Sigma_wb = pe.cov.copy()

    # --------------------------------------------------------------- #
    #  Object initialization (base-frame)
    # --------------------------------------------------------------- #

    def ensure_object(self,
                      oid: int,
                      T_co_meas: np.ndarray,
                      init_cov: np.ndarray) -> bool:
        """Create an entry for `oid` from a first observation.

        Mean lift:  T_bo_init = T_bc · T_co_meas    (kinematic, Σ_wb-free)
        Cov:        P_bo_init = init_cov                  (no frame lift)

        `init_cov` is treated as an \emph{already-in-storage-frame}
        prior on P_bo, not as a camera-frame measurement covariance.
        Rationale: at birth the "observation" is really just a mask
        centroid -- the rotation component of T_co is identity by
        convention, not a real rotation measurement -- so the
        rotation entries of `init_cov` encode a prior on the object's
        unknown rotation, not observation noise. Ad-lifting such a
        prior through `Ad(T_bc)` couples rotation variance into the
        translation block via `[t_bc]_x · R_rot · [t_bc]_x^T`, which
        on Fetch (|t_bc| ~ 1 m) inflates the birth position σ from
        `σ_t` to `sqrt(σ_t² + σ_r²·|t_bc|²)` -- too loose for the
        χ² gate. Matched updates on subsequent frames DO carry a real
        6-DOF ICP measurement, and `update_observation` correctly
        applies the Ad(T_bc) lift there.

        Callers that want a proper Ad-lift (e.g. when supplying an
        already-calibrated camera-frame R_cam with meaningful
        rotation-axis uncertainty) should do the lift themselves
        before calling.

        Returns True if this was a new object.
        """
        if oid in self.objects:
            return False
        T_co_meas = np.asarray(T_co_meas, dtype=np.float64)
        T_bo = self.T_bc @ T_co_meas
        cov_bo = np.asarray(init_cov, dtype=np.float64).copy()
        cov_bo = 0.5 * (cov_bo + cov_bo.T)
        self.objects[oid] = GaussianObjectBelief(
            mu_bo=T_bo,
            cov_bo=cov_bo,
        )
        return True

    # --------------------------------------------------------------- #
    #  Predict steps
    # --------------------------------------------------------------- #

    def predict_static(self, Q_fn: Callable[[int], np.ndarray],
                        P_max: Optional[np.ndarray] = None,
                        skip_oids: Optional[set] = None,
                        unobserved_oids: Optional[set] = None) -> None:
        """Propagate base-frame belief under BASE motion (world-static
        object). Derivation: a world-static object satisfies
            T_wb(t) · T_bo(t) = T_wb(t-1) · T_bo(t-1)         (T_wo const)
            => T_bo(t) = u_k · T_bo(t-1),
               u_k = T_wb(t)^{-1} · T_wb(t-1)
        u_k is the BODY-frame inverse of the base motion (NOT
        T_prev · T_now^{-1}, which would be the world-frame inverse;
        these differ by an adjoint conjugation when the rotation
        changes, and produce metre-scale drift on a real trajectory --
        bernoulli_ekf.tex §5).

        Two regimes per oid (keyframe-mechanism, plan §C.1):

        * `oid in unobserved_oids` -- no observation last frame. The
          stored cov_bo IS the implicit keyframe value (the post-update
          cov from the last observation). Apply ONLY the deterministic
          mean update `mu_bo = u_k @ mu_bo`. Skip the Ad-conjugate cov
          transport and skip Q. This is required to satisfy the
          intent that an unobserved static object's world uncertainty
          grows only via the current Σ_wb-lever in
          `collapsed_object_world`, NOT via per-frame predict drift.

        * Otherwise -- track was observed at frame k-1 (or just born).
          Run the full predict (Ad-conjugate + Q) so the upcoming
          joseph update has a properly inflated prior.

        `skip_oids`: tracks whose predict should be owned by a
        different motion model this frame (e.g. held objects whose
        world-pose dynamics are `T_bo = T_bg · T_go`, not
        `T_bo = u_k · T_bo`). The caller applies
        `rigid_attachment_predict` separately for those oids. Passing
        Q=0 for held oids in Q_fn is NOT sufficient: the mean motion
        `u_k · μ` would still compose (wrongly) with the rigid
        attachment update, giving `delta_T_bg · u_k · μ` instead of
        the correct `delta_T_bg · μ`.
        """
        if not self.objects:
            return
        skip = skip_oids if skip_oids is not None else set()
        unobs = unobserved_oids if unobserved_oids is not None else set()

        if self.prev_T_wb is None:
            # No base-motion baseline yet. delta = I -> μ unchanged.
            I4 = np.eye(4, dtype=np.float64)
            for oid, b in self.objects.items():
                if oid in skip:
                    continue
                if oid in unobs:
                    # Mean is unchanged (delta = I). cov_bo frozen.
                    continue
                b.mu_bo, b.cov_bo = predict_ad_conjugate(
                    b.mu_bo, b.cov_bo, I4, Q_fn(oid),
                    P_max=P_max, P_min_diag=self.P_min_diag)
            return

        # u_k = T_wb(t)^{-1} · T_wb(t-1). Body-frame relative inverse.
        u_k = np.linalg.inv(self.T_wb) @ self.prev_T_wb
        for oid, b in self.objects.items():
            if oid in skip:
                continue
            if oid in unobs:
                # Deterministic mean update only -- keep mu_bo
                # consistent with the new T_wb. cov_bo is FROZEN at
                # the keyframe value (no Ad transport, no Q).
                b.mu_bo = u_k @ b.mu_bo
                continue
            b.mu_bo, b.cov_bo = predict_ad_conjugate(
                b.mu_bo, b.cov_bo, u_k, Q_fn(oid),
                P_max=P_max, P_min_diag=self.P_min_diag)

    def rigid_attachment_predict(self,
                                 oid: int,
                                 delta_T_bg: np.ndarray,
                                 Q_manip: np.ndarray,
                                 P_max: Optional[np.ndarray] = None) -> None:
        """Apply a rigid-attachment predict in base frame
        (bernoulli_ekf.tex eq:proc with u_k = ΔT_bg).

        `delta_T_bg = T_bg(t) · inv(T_bg(t-1))` is the gripper change
        in the base frame (from proprioception).

        Base-frame kinematics of a gripper-attached object:
            T_bo(t) = T_bg(t) · T_go = ΔT_bg · T_bo(t-1)

        Thin wrapper around
        `object_belief.predict_ad_conjugate(μ, Σ, ΔT_bg, Q_manip, ...)`.
        """
        b = self.objects.get(oid)
        if b is None:
            return
        b.mu_bo, b.cov_bo = predict_ad_conjugate(
            b.mu_bo, b.cov_bo, delta_T_bg, Q_manip,
            P_max=P_max, P_min_diag=self.P_min_diag)

    # --------------------------------------------------------------- #
    #  Measurement update (base-frame IEKF)
    # --------------------------------------------------------------- #

    def update_observation(self,
                           oid: int,
                           T_co_meas: np.ndarray,
                           R_icp: np.ndarray,
                           iekf_iters: int = 2,
                           huber_w: float = 1.0,
                           P_max: Optional[np.ndarray] = None) -> None:
        """Base-frame IEKF update. Σ_wb does NOT enter here.

            T_bo_meas = T_bc · T_co_meas        (eq:meas_compose)
            R_bo      = Ad(T_bc) · R_icp · Ad(T_bc)ᵀ
            δ         = se3_log(μ_bo⁻¹ · T_bo_meas)
            S         = Σ_bo + R_bo / huber_w
            K         = Σ_bo · S⁻¹
            μ ← μ · Exp(K δ),    Σ ← Φ(Joseph)

        Args:
            huber_w: Huber redescending M-estimator weight in (0, 1] from
                     `ekf_se3.huber_weight()`. Scales R by 1/w so the gain
                     shrinks with d^2. Default 1 = no Huber.
                     A caller passing w=0 should NOT invoke this method
                     (route to the missed-detection branch); we treat <=0
                     as a no-op for safety.
            P_max:   optional (6,6) covariance saturation cap
                     (bernoulli_ekf.tex eq:phi).
        """
        b = self.objects.get(oid)
        if b is None or huber_w <= 0.0:
            return
        T_bo_meas, R_bo = lift_measurement_base(
            T_co_meas, R_icp, self.T_bc, Ad_bc=self._Ad_bc)
        b.mu_bo, b.cov_bo = joseph_update(
            b.mu_bo, b.cov_bo, T_bo_meas, R_bo,
            iekf_iters=iekf_iters, huber_w=huber_w,
            P_max=P_max, P_min_diag=self.P_min_diag)

    # --------------------------------------------------------------- #
    #  Innovation statistics for data association
    # --------------------------------------------------------------- #

    def innovation_stats(self,
                          oid: int,
                          T_co_meas: np.ndarray,
                          R_icp: np.ndarray) -> Optional[tuple]:
        """Return base-frame innovation quantities for the Hungarian
        cost matrix (bernoulli_ekf.tex §6).

        Returns (nu, S, d2, log_lik) where:
            nu      = innovation in se(3) tangent at the prior mean
            S       = residual covariance (Σ_bo + R_bo)
            d2      = nu^T S^{-1} nu  (chi^2_6 under H_0)
            log_lik = Gaussian log-likelihood of the innovation

        Returns None if `oid` is unknown.
        """
        b = self.objects.get(oid)
        if b is None:
            return None
        T_bo_meas, R_bo = lift_measurement_base(
            T_co_meas, R_icp, self.T_bc, Ad_bc=self._Ad_bc)
        return innovation_from_belief(b.mu_bo, b.cov_bo, T_bo_meas, R_bo)

    # Numerical floor for a log-likelihood (avoids -inf / NaN downstream).
    _LOG_EPS = LOG_EPS

    def delete_object(self, oid: int) -> bool:
        """Remove a track. Returns True if it was present."""
        if oid in self.objects:
            del self.objects[oid]
            return True
        return False

    # --------------------------------------------------------------- #
    #  Track-to-track merge (self-merge pass)
    # --------------------------------------------------------------- #

    def merge_tracks(self, oid_keep: int, oid_drop: int) -> bool:
        """Merge `oid_drop` into `oid_keep` by Bayesian information fusion
        of two same-frame Gaussians on SE(3).

            P_new^{-1}  = P_keep^{-1} + P_drop^{-1}                (info form)
            mu_new      = mu_keep · Exp(P_new · P_drop^{-1} · nu)
            with nu = Log(mu_keep^{-1} · mu_drop)

        This is the closed-form merge of two independent Gaussians that
        differ in mean by `nu` (linearised at mu_keep). Equivalent to
        running an EKF update on `oid_keep` with `oid_drop` as the
        observation -- here both Gaussians are equally trusted and we
        treat them symmetrically through the information sum.

        Returns True if both tracks existed and the merge happened, else
        False. After a successful merge, `oid_drop` is removed.

        The post-merge cov is symmetrised, P_min_diag-floored, and PSD-
        regularised.
        """
        a = self.objects.get(oid_keep)
        b = self.objects.get(oid_drop)
        if a is None or b is None or oid_keep == oid_drop:
            return False
        try:
            mu_new, P_new = merge_info_sum(a.mu_bo, a.cov_bo,
                                            b.mu_bo, b.cov_bo)
        except np.linalg.LinAlgError:
            return False
        a.mu_bo = mu_new
        a.cov_bo = _floor_diag(P_new, self.P_min_diag)
        del self.objects[oid_drop]
        return True

    # --------------------------------------------------------------- #
    #  Output composition (the only place Σ_wb enters)
    # --------------------------------------------------------------- #

    # ────────────── Centroid-only innovation (Phase C) ──────────────
    def centroid_innovation_stats(self,
                                   oid: int,
                                   centroid_cam: np.ndarray,
                                   R_cam: Optional[np.ndarray] = None,
                                   ) -> Optional[tuple]:
        """3-DOF centroid innovation in the base tangent (no ICP needed).

        Used by `SceneTracker`'s coarse association stage: the Hungarian
        cost for a (track, detection) pair is built from the
        translation-only Mahalanobis distance between the predicted
        base-frame position of the track and the back-projected
        centroid of the detection mask.

        Lift:
            t_bo_meas = T_bc @ [centroid_cam; 1]     (deterministic kinematic)
            R_bo_tt   = R_tt(T_bc) · R_cam · R_tt(T_bc)^T
                        where R_tt(T_bc) is the upper-left 3x3 block of
                        the adjoint Ad(T_bc). For a pure-translation
                        observation this is just the camera-to-base
                        rotation applied to R_cam.

        Returns `(ν3, S3, d²3, log_lik3)` or `None` if oid is unknown.

        Typical `R_cam`: a 3x3 diagonal, e.g.
        `diag(σ_centroid^2 × 3)` with σ_centroid ≈ 2-3 cm reflecting
        mask-edge + depth-sampling jitter. When `None`, a default of
        `(2 cm)^2` isotropic is used.
        """
        b = self.objects.get(oid)
        if b is None:
            return None
        centroid_cam = np.asarray(centroid_cam, dtype=np.float64).reshape(3)
        t_bo_meas = (self.T_bc @ np.array([centroid_cam[0],
                                             centroid_cam[1],
                                             centroid_cam[2], 1.0]))[:3]
        t_bo_prior = np.asarray(b.mu_bo[:3, 3], dtype=np.float64)
        nu = t_bo_meas - t_bo_prior

        if R_cam is None:
            R_cam = np.diag([(0.02) ** 2] * 3)
        else:
            R_cam = 0.5 * (np.asarray(R_cam, dtype=np.float64)
                            + np.asarray(R_cam, dtype=np.float64).T)

        R_bc = self.T_bc[:3, :3]
        R_bo_tt = R_bc @ R_cam @ R_bc.T
        P_tt = np.asarray(b.cov_bo[:3, :3], dtype=np.float64)
        S = P_tt + R_bo_tt
        S = 0.5 * (S + S.T)
        sign, logdet = np.linalg.slogdet(S)
        try:
            Sinv_nu = np.linalg.solve(S, nu)
            d2 = float(nu @ Sinv_nu)
        except np.linalg.LinAlgError:
            d2 = float("inf")
        if sign <= 0 or not np.isfinite(logdet) or not np.isfinite(d2):
            log_lik = LOG_EPS
        else:
            two_pi_log = 3.0 * np.log(2.0 * np.pi)
            log_lik = -0.5 * d2 - 0.5 * (float(logdet) + two_pi_log)
        return nu, S, d2, log_lik

    def update_observation_centroid(self,
                                     oid: int,
                                     centroid_cam: np.ndarray,
                                     R_cam: Optional[np.ndarray] = None,
                                     huber_w: float = 1.0,
                                     P_max: Optional[np.ndarray] = None
                                     ) -> None:
        """Translation-only Kalman update from a camera-frame centroid.

        Used by the fine-ICP fallback: when ICP refinement fails but
        Hungarian certified the pair via the 3-DOF centroid, we still
        have a valid POSITION observation. Updating translation only
        (not rotation) is the correct state-estimation answer to
        "the measurement's rotation is unobserved".

        Avoids the 6-DOF `update_observation`'s Ad(T_bc) lift, which
        would inflate the translation block by `|t_bc|^2 · σ_r^2` for
        any R with a large rotation block. Here we operate directly
        on the 3×3 translation partition:

            t_bo_meas = T_bc · [centroid; 1]                (lift position)
            R_bo_tt   = R_bc · R_cam · R_bc^T              (rotation-only lift)
            ν_t       = t_bo_meas - μ_bo[:3, 3]
            S_tt      = P_tt + R_bo_tt / huber_w
            K_tt      = P_{t, 1:6} @ S_tt^{-1}              (full P column block → 6×3 gain)
            μ_bo      ← μ_bo ⊞ [K_tt @ ν_t; 0_3]            (translation-only correction)
            P         ← (I - K_full H) P (I - K_full H)^T + K R K^T (Joseph)
              with  H = [I_3 | 0_3×3]          (3×6 picks translation slot)
                    K_full = P H^T S^{-1}      (6×3)

        i.e. the Kalman gain is restricted to the translation output,
        but the resulting update correctly decorrelates the covariance
        across all six axes per the cross-coupling in P.
        """
        b = self.objects.get(oid)
        if b is None or huber_w <= 0.0:
            return
        if R_cam is None:
            R_cam = np.diag([(0.02) ** 2] * 3)
        R_cam = 0.5 * (np.asarray(R_cam, dtype=np.float64)
                         + np.asarray(R_cam, dtype=np.float64).T)
        if 0.0 < huber_w < 1.0:
            R_cam = R_cam / huber_w

        centroid_cam = np.asarray(centroid_cam, dtype=np.float64).reshape(3)
        R_bc = self.T_bc[:3, :3]
        R_bo_tt = R_bc @ R_cam @ R_bc.T
        R_bo_tt = 0.5 * (R_bo_tt + R_bo_tt.T)

        # Measurement lift (translation in base frame).
        t_bo_meas = (self.T_bc @ np.array([centroid_cam[0],
                                             centroid_cam[1],
                                             centroid_cam[2], 1.0]))[:3]

        # Translation-only Kalman update — operate directly on the
        # base-frame translation block of mu_bo. The previous
        # implementation built a 6-vector tangent ``corr`` from a
        # base-frame innovation and then applied
        # ``mu_new = mu_bo @ se3_exp(corr)``. That right-multiply
        # interprets ``corr`` as a *body-local* tangent — but
        # ``corr[:3]`` came from a *base-frame* nu_t, so the effective
        # base-frame correction was R_bo · K · nu_t. For tracks with
        # any rotation in mu_bo (apples born from ICP routinely have
        # arbitrary attitudes), this rotates the correction direction
        # and can flip its sign — producing a slow systematic drift in
        # μ_w even when the measurement is perfectly stable. Verified
        # on apple_in_the_tray oid 3, frames 290-330: μ_w drifted +8 cm
        # in z while the apple was stationary, until d²t exceeded the
        # gate and the track lost correspondence. Fix: update only the
        # translation partition, additively in base frame. Rotation
        # block is left untouched (a translation-only observation
        # contributes nothing to rotation mean correction); mixing
        # frames between the body-frame state and the base-frame
        # measurement is what created the bug in the first place.
        nu_t = t_bo_meas - np.asarray(b.mu_bo[:3, 3], dtype=np.float64)
        cov = 0.5 * (b.cov_bo + b.cov_bo.T)
        P_tt = cov[:3, :3]            # (3, 3)
        S = P_tt + R_bo_tt
        S = 0.5 * (S + S.T)
        K_tt = np.linalg.solve(S.T, P_tt.T).T   # (3, 3): P_tt · S^{-1}
        delta_t_base = K_tt @ nu_t

        mu_new = b.mu_bo.copy()
        mu_new[:3, 3] = b.mu_bo[:3, 3] + delta_t_base

        # Joseph cov update on the translation block only. We zero the
        # cross-covariance with rotation: a translation-only
        # observation gives no information about rotation (R_rot was
        # taken as ∞ in the matched-update loop), so treating the two
        # blocks as decoupled after the update is the honest
        # expression of that. Keeping the cross-cov from before the
        # update — as the previous version did — let cov drift to
        # indefinite over many sequential updates because P_tt was
        # shrunk by Kalman while P_tr was not, breaking the Schur
        # complement P_rr − P_rt^T P_tt⁻¹ P_rt ⪰ 0.
        I3 = np.eye(3)
        IK = I3 - K_tt
        P_tt_post = IK @ P_tt @ IK.T + K_tt @ R_bo_tt @ K_tt.T
        P_tt_post = 0.5 * (P_tt_post + P_tt_post.T)
        cov_post = np.zeros((6, 6), dtype=np.float64)
        cov_post[:3, :3] = P_tt_post
        cov_post[3:, 3:] = cov[3:, 3:]
        cov_post = 0.5 * (cov_post + cov_post.T)

        # Belt-and-suspenders: clip any tiny negative eigenvalue
        # introduced by floating-point noise.
        try:
            evals, evecs = np.linalg.eigh(cov_post)
            if (evals < 0).any():
                evals = np.clip(evals, 0.0, None)
                cov_post = evecs @ np.diag(evals) @ evecs.T
                cov_post = 0.5 * (cov_post + cov_post.T)
        except np.linalg.LinAlgError:
            pass

        if P_max is not None:
            from pose_update.state.object_belief import saturate_covariance
            cov_post = saturate_covariance(cov_post, P_max)
        if self.P_min_diag is not None:
            from pose_update.state.object_belief import floor_diag as _fd
            cov_post = _fd(cov_post, self.P_min_diag)

        b.mu_bo = mu_new
        b.cov_bo = cov_post

    def innovation_stats_centroid_3d(self,
                                      oid: int,
                                      centroid_cam: np.ndarray,
                                      R_cam: Optional[np.ndarray] = None
                                      ) -> Optional[tuple]:
        """Thin alias around `centroid_innovation_stats`, returned under a
        clearer name for the matched-update loop. Returns
        `(nu_t, S_tt, d2_t, log_lik_3D)`.
        """
        return self.centroid_innovation_stats(oid, centroid_cam, R_cam=R_cam)

    # BasePoseBackend protocol methods (see `pose_update.state.base_pose_backend`)
    def collapsed_T_wb(self) -> np.ndarray:
        """Return the current SLAM mean T_wb. Part of BasePoseBackend."""
        assert self.T_wb is not None, "ingest_slam has not been called yet"
        return self.T_wb.copy()

    def known_oids(self):
        """List of currently-tracked oids. Part of BasePoseBackend."""
        return list(self.objects.keys())

    def camera_frame_prior(self, oid: int):
        """T_co^pred for this oid = T_bc^{-1} · μ_bo. Part of BasePoseBackend.

        Used by `SceneTracker.step()` as the ICP seed before Hungarian.
        Returns None if oid is unknown.
        """
        b = self.objects.get(oid)
        if b is None:
            return None
        return np.linalg.inv(self.T_bc) @ b.mu_bo

    def predict_static_all(self,
                            Q_fn: Callable[[int], np.ndarray],
                            skip_oids: Optional[set] = None,
                            P_max: Optional[np.ndarray] = None,
                            unobserved_oids: Optional[set] = None) -> None:
        """Static-object predict for every oid not in `skip_oids`.

        `skip_oids` is FULLY skipped (no u_k propagation, no Q) -- the
        caller is expected to apply a different motion model (e.g.
        `rigid_attachment_predict`) for those oids. See the
        `predict_static` docstring for why zeroing Q wasn't sufficient.

        `unobserved_oids` are tracks not observed last frame; they get
        only the deterministic mean update (cov_bo frozen, no Q) per
        the keyframe mechanism (plan §C.1).
        """
        self.predict_static(Q_fn, P_max=P_max, skip_oids=skip_oids,
                             unobserved_oids=unobserved_oids)

    def collapsed_base(self) -> PoseEstimate:
        """The current SLAM posterior, echoed for API symmetry with RBPF."""
        assert self.T_wb is not None, "ingest_slam has not been called yet"
        return PoseEstimate(T=self.T_wb.copy(), cov=self.Sigma_wb.copy())

    def overwrite_object_pose(self, oid: int,
                               T_wo: np.ndarray,
                               P_wo: np.ndarray) -> bool:
        """Force the EKF state for `oid` to a world-frame `(T_wo, P_wo)`.

        Used by the gravity-aware predict at release: the input is a
        world-frame mean+cov from `gravity_predict.predict_landing_pose`,
        and we install it into the body-frame storage so subsequent
        per-frame predict / collapsed_object_world calls reflect the
        post-fall belief. Returns True iff `oid` is currently tracked.

        The body-frame transport: under right-trivialisation, the
        right-tangent at the new `mu_bo = T_wb⁻¹ · T_wo` equals the
        right-tangent at `T_wo` numerically, so the cov transport is
        the identity. (Plan §C.2.)
        """
        T_wo = np.asarray(T_wo, dtype=np.float64)
        P_wo = np.asarray(P_wo, dtype=np.float64)
        if T_wo.shape != (4, 4) or P_wo.shape != (6, 6):
            raise ValueError("T_wo must be (4, 4) and P_wo (6, 6)")
        b = self.objects.get(oid)
        if b is None:
            return False
        assert self.T_wb is not None, "ingest_slam has not been called yet"
        b.mu_bo = np.linalg.inv(self.T_wb) @ T_wo
        b.cov_bo = 0.5 * (P_wo + P_wo.T)
        return True

    def collapsed_object_base(self, oid: int) -> Optional[PoseEstimate]:
        b = self.objects.get(oid)
        if b is None:
            return None
        return PoseEstimate(T=b.mu_bo.copy(), cov=b.cov_bo.copy())

    def collapsed_object_world(self, oid: int) -> Optional[PoseEstimate]:
        """Compose T_wo = T_wb · T_bo and propagate covariance.

        Under independence δ_wb ⊥ δ_bo (a reasonable approximation
        given base-frame storage keeps them decoupled through the
        recursion):
            δ_wo = Ad(T_bo⁻¹) · δ_wb + δ_bo
            Σ_wo = Ad(T_bo⁻¹) · Σ_wb · Ad(·)ᵀ + Σ_bo

        Σ_wo is lower-bounded by the projected Σ_wb term, which is
        exactly what we want for a static object observed many times
        from the same base.
        """
        if self.T_wb is None:
            return None
        b = self.objects.get(oid)
        if b is None:
            return None
        mu_wo = self.T_wb @ b.mu_bo
        Ad_bo_inv = se3_adjoint(np.linalg.inv(b.mu_bo))
        cov_wo = Ad_bo_inv @ self.Sigma_wb @ Ad_bo_inv.T + b.cov_bo
        cov_wo = 0.5 * (cov_wo + cov_wo.T)
        return PoseEstimate(T=mu_wo, cov=cov_wo)

    def collapsed_objects_world(self) -> Dict[int, PoseEstimate]:
        """World-frame collapsed view for every tracked object."""
        out: Dict[int, PoseEstimate] = {}
        for oid in self.objects:
            pe = self.collapsed_object_world(oid)
            if pe is not None:
                out[oid] = pe
        return out

    # --------------------------------------------------------------- #
    #  Slow-tier reconcile
    # --------------------------------------------------------------- #

    def inject_posterior_world(self,
                                oid: int,
                                posterior: PoseEstimate) -> None:
        """Re-center the base-frame mean from a world-frame slow-tier
        posterior. We deliberately do NOT overwrite Σ_bo — the
        world-frame posterior covariance has Σ_wb folded in, and
        injecting it back into base-frame covariance would double-count.
        The next observation will tighten Σ_bo naturally.
        """
        b = self.objects.get(oid)
        if b is None or self.T_wb is None:
            return
        b.mu_bo = np.linalg.inv(self.T_wb) @ posterior.T

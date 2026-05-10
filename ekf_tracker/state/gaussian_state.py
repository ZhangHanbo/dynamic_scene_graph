r"""Gaussian backend for Layer 2: per-object EKFs in the robot base frame.

The world covariance :math:`\Sigma_{wb}` never enters the per-frame recursion;
it is composed in only at output time:

.. math::
    T_{wo} = T_{wb}\, T_{bo}, \qquad
    \Sigma_{wo} = \Ad(T_{bo}^{-1})\, \Sigma_{wb}\, \Ad(T_{bo}^{-1})^{\top} + \Sigma_{bo}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from utils.ekf_se3 import (
    se3_exp, se3_log, se3_adjoint, ekf_predict, saturate_covariance,
)
from utils.object_belief import (
    floor_diag as _floor_diag,
    lift_measurement_base,
    predict_ad_conjugate,
    innovation_from_belief,
    joseph_update,
    merge_info_sum,
    LOG_EPS,
)
from utils.slam_interface import (
    PoseEstimate, ParticlePose, as_gaussian,
)


# ─────────────────────────────────────────────────────────────────────
# Per-object belief (single Gaussian, base frame)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GaussianObjectBelief:
    r"""Single-Gaussian SE(3) belief about one object in the base frame.

    Fields:
        mu_bo (np.ndarray): :math:`\mu_{bo} \in SE(3)`, shape (4, 4).
        cov_bo (np.ndarray): :math:`\Sigma_{bo}` in :math:`\se(3)` tangent at :math:`\mu_{bo}`,
            ordering :math:`[v, \omega]`, shape (6, 6).
    """
    mu_bo: np.ndarray
    cov_bo: np.ndarray


# ─────────────────────────────────────────────────────────────────────
# GaussianState
# ─────────────────────────────────────────────────────────────────────

class GaussianState:
    """Bag of per-object base-frame EKFs plus the latest SLAM posterior."""

    _LOG_EPS = -1e18

    def __init__(self, T_bc: Optional[np.ndarray] = None,
                 P_min_diag: Optional[np.ndarray] = None):
        r"""Construct an empty state with optional camera extrinsic and per-axis covariance floor.

        Args:
            T_bc: (4, 4) base-from-camera transform; defaults to identity.
            P_min_diag: (6,) per-axis lower bound on :math:`\diag(\Sigma_{bo})`,
                ordering :math:`[v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]`.
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
        """Update :math:`T_{bc}` for the current frame (head pan/tilt or torso lift)."""
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
        """True once :meth:`ingest_slam` has been called at least once."""
        return self.T_wb is not None

    # --------------------------------------------------------------- #
    #  SLAM ingestion (collapses ParticlePose → PoseEstimate)
    # --------------------------------------------------------------- #

    def ingest_slam(self, slam_result) -> None:
        r"""Take a new SLAM posterior :math:`(T_{wb}, \Sigma_{wb})`, caching the previous mean for :meth:`predict_static`."""
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
        r"""Create a new track from the first observation: :math:`\mu_{bo} = T_{bc}\, T_{co}^{\text{meas}}`, :math:`\Sigma_{bo} = \texttt{init\_cov}`. Returns True if the oid was new."""
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
        r"""World-static predict: :math:`\mu_{bo} \leftarrow u_k\, \mu_{bo}` with :math:`u_k = T_{wb}(t)^{-1} T_{wb}(t-1)`; covariance gets the :math:`\Ad`-conjugate transport plus :math:`Q`."""
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
        r"""Held-object predict: :math:`\mu_{bo} \leftarrow \Delta T_{bg}\, \mu_{bo}`; covariance via :math:`\Ad`-conjugate transport plus :math:`Q_{\text{manip}}`."""
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
        r"""Base-frame IEKF update from a 6-DOF camera-frame measurement.

        .. math::
            T_{bo}^{\text{meas}} &= T_{bc}\, T_{co}^{\text{meas}}, \\
            R_{bo} &= \Ad(T_{bc})\, R_{\text{icp}}\, \Ad(T_{bc})^{\top}, \\
            \delta &= \log(\mu_{bo}^{-1} T_{bo}^{\text{meas}}), \quad
            K = \Sigma_{bo} (\Sigma_{bo} + R_{bo}/w_{H})^{-1}.
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
        r"""Return :math:`(\nu, S, d^2, \log\!\mathcal{L})` for the 6-DOF Hungarian cost, or ``None`` if oid is unknown."""
        b = self.objects.get(oid)
        if b is None:
            return None
        T_bo_meas, R_bo = lift_measurement_base(
            T_co_meas, R_icp, self.T_bc, Ad_bc=self._Ad_bc)
        return innovation_from_belief(b.mu_bo, b.cov_bo, T_bo_meas, R_bo)

    # Numerical floor for a log-likelihood (avoids -inf / NaN downstream).
    _LOG_EPS = LOG_EPS

    def delete_object(self, oid: int) -> bool:
        """Remove a track; returns True if it existed."""
        if oid in self.objects:
            del self.objects[oid]
            return True
        return False

    # --------------------------------------------------------------- #
    #  Track-to-track merge (self-merge pass)
    # --------------------------------------------------------------- #

    def merge_tracks(self, oid_keep: int, oid_drop: int) -> bool:
        r"""Bayesian information-form merge of two same-frame Gaussians.

        .. math::
            P_{\text{new}}^{-1} = P_{\text{keep}}^{-1} + P_{\text{drop}}^{-1}, \quad
            \mu_{\text{new}} = \mu_{\text{keep}} \exp\!\bigl(P_{\text{new}}\, P_{\text{drop}}^{-1}\, \nu\bigr),
            \quad \nu = \log(\mu_{\text{keep}}^{-1} \mu_{\text{drop}}).
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
        r"""3-DOF centroid innovation stats :math:`(\nu_3, S_3, d^2_3, \log\!\mathcal{L}_3)` in the base tangent."""
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
        r"""Translation-only Kalman update on :math:`\mu_{bo}[:3, 3]` from a camera-frame centroid."""
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
            from utils.object_belief import saturate_covariance
            cov_post = saturate_covariance(cov_post, P_max)
        if self.P_min_diag is not None:
            from utils.object_belief import floor_diag as _fd
            cov_post = _fd(cov_post, self.P_min_diag)

        b.mu_bo = mu_new
        b.cov_bo = cov_post

    def innovation_stats_centroid_3d(self,
                                      oid: int,
                                      centroid_cam: np.ndarray,
                                      R_cam: Optional[np.ndarray] = None
                                      ) -> Optional[tuple]:
        """Alias of :meth:`centroid_innovation_stats`."""
        return self.centroid_innovation_stats(oid, centroid_cam, R_cam=R_cam)

    # BasePoseBackend protocol methods (see `utils.base_pose_backend`)
    def collapsed_T_wb(self) -> np.ndarray:
        """Current SLAM mean :math:`T_{wb}`."""
        assert self.T_wb is not None, "ingest_slam has not been called yet"
        return self.T_wb.copy()

    def known_oids(self):
        """List of currently-tracked oids."""
        return list(self.objects.keys())

    def camera_frame_prior(self, oid: int):
        r"""ICP seed :math:`T_{co}^{\text{pred}} = T_{bc}^{-1} \mu_{bo}` for ``oid``, or ``None``."""
        b = self.objects.get(oid)
        if b is None:
            return None
        return np.linalg.inv(self.T_bc) @ b.mu_bo

    def predict_static_all(self,
                            Q_fn: Callable[[int], np.ndarray],
                            skip_oids: Optional[set] = None,
                            P_max: Optional[np.ndarray] = None,
                            unobserved_oids: Optional[set] = None) -> None:
        """Apply :meth:`predict_static` to every oid except those in ``skip_oids``."""
        self.predict_static(Q_fn, P_max=P_max, skip_oids=skip_oids,
                             unobserved_oids=unobserved_oids)

    def collapsed_base(self) -> PoseEstimate:
        """Echo the current SLAM posterior as a :class:`PoseEstimate`."""
        assert self.T_wb is not None, "ingest_slam has not been called yet"
        return PoseEstimate(T=self.T_wb.copy(), cov=self.Sigma_wb.copy())

    def overwrite_object_pose(self, oid: int,
                               T_wo: np.ndarray,
                               P_wo: np.ndarray) -> bool:
        r"""Force :math:`(\mu_{bo}, \Sigma_{bo})` for ``oid`` from a world-frame :math:`(T_{wo}, P_{wo})` (used by gravity predict)."""
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
        """Base-frame mean+cov for ``oid``, or ``None``."""
        b = self.objects.get(oid)
        if b is None:
            return None
        return PoseEstimate(T=b.mu_bo.copy(), cov=b.cov_bo.copy())

    def collapsed_object_world(self, oid: int) -> Optional[PoseEstimate]:
        r"""Compose to world frame: :math:`T_{wo} = T_{wb} \mu_{bo}` and :math:`\Sigma_{wo} = \Ad(\mu_{bo}^{-1}) \Sigma_{wb} \Ad(\cdot)^{\top} + \Sigma_{bo}`."""
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
        r"""Re-center :math:`\mu_{bo}` from a slow-tier world-frame posterior; :math:`\Sigma_{bo}` is left untouched to avoid double-counting :math:`\Sigma_{wb}`."""
        b = self.objects.get(oid)
        if b is None or self.T_wb is None:
            return
        b.mu_bo = np.linalg.inv(self.T_wb) @ posterior.T

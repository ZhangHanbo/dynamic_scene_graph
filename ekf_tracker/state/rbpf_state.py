"""Rao-Blackwellized particle filter variant: weighted base particles, each with its own per-object EKF (research backend)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from utils.ekf_se3 import (
    se3_exp, se3_log, se3_adjoint, ekf_predict, saturate_covariance,
)
from utils.object_belief import (
    LOG_EPS,
    floor_diag as _floor_diag,
    lift_measurement_world,
    predict_ad_conjugate,
    innovation_from_belief,
    joseph_update,
    merge_info_sum,
)
from utils.slam_interface import (
    PoseEstimate, ParticlePose,
    sample_particles_from_gaussian, as_gaussian,
)


# ─────────────────────────────────────────────────────────────────────
# Per-particle per-object belief
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ParticleObjectBelief:
    """Per-particle per-object Gaussian belief in the world frame."""
    mu: np.ndarray
    cov: np.ndarray


@dataclass
class Particle:
    """One RBPF particle: SLAM pose :math:`T_{wb}`, log-weight, and per-object beliefs."""
    T_wb: np.ndarray
    log_weight: float
    objects: Dict[int, ParticleObjectBelief] = field(default_factory=dict)
    prev_T_wb: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────
# RBPF state
# ─────────────────────────────────────────────────────────────────────

class RBPFState:
    """Rao-Blackwellized particle-filter backend: weighted base particles, each carrying its own per-object EKF."""

    # Numerical floor for a log-weight (avoids -inf / NaN).
    _LOG_EPS = -1e18

    def __init__(self,
                 n_particles: int,
                 rng: Optional[np.random.Generator] = None,
                 T_bc: Optional[np.ndarray] = None,
                 P_min_diag: Optional[np.ndarray] = None):
        assert n_particles >= 1
        self.n_particles = n_particles
        self.particles: List[Particle] = []
        self._rng = rng if rng is not None else np.random.default_rng()

        # Per-frame camera extrinsic. Previously ignored by RBPFState
        # (implicit T_bc=I), which silently aliased head pan/tilt as
        # object motion. Now installed via `set_camera_extrinsic(...)`
        # at the top of every SceneTracker step, same semantics as the
        # Gaussian backend.
        self.T_bc: np.ndarray = (
            np.eye(4, dtype=np.float64) if T_bc is None
            else np.asarray(T_bc, dtype=np.float64).copy())
        self._Ad_bc: np.ndarray = se3_adjoint(self.T_bc)

        # Collapsed base-pose view (mean over particles). Updated in
        # `ingest_slam` so downstream code can treat the RBPF and the
        # Gaussian backends uniformly.
        self.T_wb: Optional[np.ndarray] = None
        self.prev_T_wb: Optional[np.ndarray] = None
        self.Sigma_wb: Optional[np.ndarray] = None

        # Per-axis covariance floor. Matches GaussianState semantics.
        self.P_min_diag: Optional[np.ndarray] = (
            np.asarray(P_min_diag, dtype=np.float64).reshape(6).copy()
            if P_min_diag is not None else None)

    # ------------------------------------------------------------------ #
    #  Camera extrinsic (per-frame)
    # ------------------------------------------------------------------ #

    def set_camera_extrinsic(self, T_bc: np.ndarray) -> None:
        """Update :math:`T_{bc}` for the current frame across every particle."""
        T_bc = np.asarray(T_bc, dtype=np.float64)
        if T_bc.shape != (4, 4):
            raise ValueError(f"T_bc must be (4, 4), got {T_bc.shape}")
        self.T_bc = T_bc.copy()
        self._Ad_bc = se3_adjoint(self.T_bc)

    # --------------------------------------------------------------- #
    #  Convenience / inspection
    # --------------------------------------------------------------- #

    @property
    def initialized(self) -> bool:
        return len(self.particles) == self.n_particles

    def normalized_weights(self) -> np.ndarray:
        """Return the normalized probability vector over particles."""
        logs = np.array([p.log_weight for p in self.particles])
        m = logs.max()
        if not np.isfinite(m):
            # All -inf: degenerate; return uniform.
            return np.full(self.n_particles, 1.0 / self.n_particles)
        ws = np.exp(logs - m)
        s = ws.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full(self.n_particles, 1.0 / self.n_particles)
        return ws / s

    def ess(self) -> float:
        """Effective sample size 1 / Σ w_k²."""
        ws = self.normalized_weights()
        return float(1.0 / np.sum(ws ** 2))

    # --------------------------------------------------------------- #
    #  Particle plumbing from a SLAM backend result
    # --------------------------------------------------------------- #

    def ingest_slam(self, slam_result) -> None:
        """Take a new SLAM particle posterior, replacing the particle set."""
        if isinstance(slam_result, ParticlePose):
            pp = slam_result
            if pp.n == self.n_particles:
                new_Ts = pp.particles
                new_log_increments = np.log(
                    np.clip(pp.weights, 1e-300, None))
                new_log_increments -= new_log_increments.max()
            else:
                idx = self._rng.choice(
                    pp.n, size=self.n_particles, p=pp.weights)
                new_Ts = pp.particles[idx]
                new_log_increments = np.zeros(self.n_particles)
        else:
            pe = as_gaussian(slam_result)
            sampled = sample_particles_from_gaussian(
                pe, self.n_particles, rng=self._rng)
            new_Ts = sampled.particles
            new_log_increments = np.zeros(self.n_particles)

        if not self.initialized:
            self.particles = [
                Particle(
                    T_wb=new_Ts[k].copy(),
                    log_weight=float(new_log_increments[k]),
                    objects={},
                    prev_T_wb=None,
                )
                for k in range(self.n_particles)
            ]
            # Initialise collapsed base view for BasePoseBackend protocol.
            self._refresh_collapsed_base()
            return

        for k, p in enumerate(self.particles):
            # Cache previous T_wb for the rigid-attachment predict before
            # overwriting. Across resamples slot-k continuity is preserved
            # (resample deep-copies whole particles including prev_T_wb).
            p.prev_T_wb = p.T_wb
            p.T_wb = new_Ts[k].copy()
            p.log_weight += float(new_log_increments[k])

        # Refresh the collapsed (T_wb, prev_T_wb) view that the
        # BasePoseBackend protocol exposes.
        self._refresh_collapsed_base()

    def _refresh_collapsed_base(self) -> None:
        """Cache the particle-mean :math:`T_{wb}` for downstream consumers."""
        if not self.particles:
            self.T_wb = None
            self.prev_T_wb = None
            self.Sigma_wb = None
            return
        prev = self.T_wb
        self.prev_T_wb = prev.copy() if prev is not None else None
        base_pe = self.collapsed_base()
        self.T_wb = base_pe.T.copy()
        self.Sigma_wb = base_pe.cov.copy()

    # --------------------------------------------------------------- #
    #  Object initialization
    # --------------------------------------------------------------- #

    def ensure_object(self,
                      oid: int,
                      T_co_meas: np.ndarray,
                      init_cov: np.ndarray) -> bool:
        """Add a new object to every particle from the first observation."""
        T_co_meas = np.asarray(T_co_meas, dtype=np.float64)
        T_bc = self.T_bc
        init_cov_sym = 0.5 * (init_cov + init_cov.T)
        if self.P_min_diag is not None:
            init_cov_sym = _floor_diag(init_cov_sym, self.P_min_diag)
        newly_added = False
        for p in self.particles:
            if oid in p.objects:
                continue
            T_wo = p.T_wb @ T_bc @ T_co_meas
            p.objects[oid] = ParticleObjectBelief(
                mu=T_wo.copy(),
                cov=init_cov_sym.copy(),
            )
            newly_added = True
        return newly_added

    # --------------------------------------------------------------- #
    #  Predict
    # --------------------------------------------------------------- #

    def predict_objects(self, Q_fn: Callable[[int, Particle], np.ndarray],
                        P_max: Optional[np.ndarray] = None) -> None:
        """Per-particle world-static predict for every tracked object."""
        for p in self.particles:
            for oid, belief in p.objects.items():
                Q = Q_fn(oid, p)
                belief.mu, belief.cov = ekf_predict(
                    belief.mu, belief.cov, Q, P_max=P_max)

    def rigid_attachment_predict(self,
                                 oid: int,
                                 delta_T_grip_b: np.ndarray,
                                 Q_manip: np.ndarray) -> None:
        r"""Per-particle rigid-attachment predict: :math:`\mu_{wo} \leftarrow \Delta T_{wg}\, \mu_{wo}`."""
        for p in self.particles:
            belief = p.objects.get(oid)
            if belief is None:
                continue
            T_wb_prev = p.prev_T_wb if p.prev_T_wb is not None else p.T_wb
            delta_T_grip_w = p.T_wb @ delta_T_grip_b @ np.linalg.inv(T_wb_prev)
            Ad = se3_adjoint(delta_T_grip_w)
            belief.mu = delta_T_grip_w @ belief.mu
            belief.cov = Ad @ belief.cov @ Ad.T + Q_manip
            belief.cov = 0.5 * (belief.cov + belief.cov.T)

    # --------------------------------------------------------------- #
    #  Measurement update + likelihood (vision's dual role)
    # --------------------------------------------------------------- #

    def update_observation(self,
                           oid: int,
                           T_co_meas: np.ndarray,
                           R_icp: np.ndarray,
                           iekf_iters: int = 2,
                           huber_w: float = 1.0,
                           P_max: Optional[np.ndarray] = None) -> None:
        """Per-particle 6-DOF IEKF update from a camera-frame measurement."""
        T_co_meas = np.asarray(T_co_meas, dtype=np.float64)
        T_bc = self.T_bc
        for p in self.particles:
            belief = p.objects.get(oid)
            if belief is None:
                continue
            # Per-particle world-frame lift: Σ_wb is zero conditional on
            # this particle's sampled T_wb^k (Rao-Blackwellization).
            T_wo_meas, R_wo = lift_measurement_world(
                T_co_meas, R_icp, T_bc, p.T_wb, Sigma_wb=None)

            # Innovation at the prior mean (for the particle-reweighting
            # likelihood) BEFORE the IEKF mutates μ.
            _, _, _, log_lik = innovation_from_belief(
                belief.mu, belief.cov, T_wo_meas, R_wo)

            # IEKF + Joseph posterior.
            mu_new, cov_new = joseph_update(
                belief.mu, belief.cov, T_wo_meas, R_wo,
                iekf_iters=iekf_iters, huber_w=huber_w,
                P_max=P_max, P_min_diag=self.P_min_diag)
            belief.mu = mu_new
            belief.cov = cov_new

            # Vision's dual role: the same log-likelihood that updated
            # the object EKF also reweights this particle's trajectory
            # posterior.
            p.log_weight += log_lik

    def update_observation_centroid(self,
                                     oid: int,
                                     centroid_cam: np.ndarray,
                                     R_cam: Optional[np.ndarray] = None,
                                     huber_w: float = 1.0,
                                     P_max: Optional[np.ndarray] = None
                                     ) -> None:
        """Per-particle translation-only Kalman update from a camera-frame centroid."""
        if huber_w <= 0.0:
            return
        centroid_cam = np.asarray(centroid_cam, dtype=np.float64).reshape(3)
        if R_cam is None:
            R_cam = np.diag([(0.02) ** 2] * 3)
        R_cam = 0.5 * (np.asarray(R_cam, dtype=np.float64)
                         + np.asarray(R_cam, dtype=np.float64).T)
        if 0.0 < huber_w < 1.0:
            R_cam_eff = R_cam / huber_w
        else:
            R_cam_eff = R_cam

        T_bc = self.T_bc
        for p in self.particles:
            belief = p.objects.get(oid)
            if belief is None:
                continue
            T_wc = p.T_wb @ T_bc
            R_wc = T_wc[:3, :3]
            t_wo_meas = (T_wc @ np.array([centroid_cam[0], centroid_cam[1],
                                             centroid_cam[2], 1.0]))[:3]
            R_wo_tt = R_wc @ R_cam_eff @ R_wc.T
            R_wo_tt = 0.5 * (R_wo_tt + R_wo_tt.T)

            cov = 0.5 * (belief.cov + belief.cov.T)
            P_col = cov[:, :3]                     # (6,3)
            P_tt = cov[:3, :3]                     # (3,3)
            S = P_tt + R_wo_tt
            S = 0.5 * (S + S.T)

            try:
                K_full = np.linalg.solve(S.T, P_col.T).T   # (6,3)
                sign, logdet = np.linalg.slogdet(S)
            except np.linalg.LinAlgError:
                continue

            nu_t = t_wo_meas - np.asarray(belief.mu[:3, 3], dtype=np.float64)
            corr = np.zeros(6, dtype=np.float64)
            corr[:3] = K_full[:3] @ nu_t
            corr[3:] = K_full[3:] @ nu_t
            mu_new = belief.mu @ se3_exp(corr)

            # Joseph covariance update; H selects translation partition.
            I6 = np.eye(6)
            IKH = I6.copy()
            IKH[:, :3] = IKH[:, :3] - K_full
            cov_post = IKH @ cov @ IKH.T + K_full @ R_wo_tt @ K_full.T
            cov_post = 0.5 * (cov_post + cov_post.T)
            if P_max is not None:
                cov_post = np.minimum(cov_post, P_max)  # simple cap
            if self.P_min_diag is not None:
                d = np.asarray(self.P_min_diag, dtype=np.float64)
                idx = np.arange(6)
                cov_post[idx, idx] = np.maximum(cov_post[idx, idx], d)

            belief.mu = mu_new
            belief.cov = cov_post

            # Particle log-likelihood from the 3-DoF innovation (against the
            # PRIOR state, computed just above via P_tt + R_wo_tt and the
            # same nu_t).
            if sign > 0 and np.isfinite(logdet):
                try:
                    d2 = float(nu_t @ np.linalg.solve(S, nu_t))
                    two_pi_log = 3.0 * np.log(2.0 * np.pi)
                    p.log_weight += -0.5 * d2 - 0.5 * (float(logdet) + two_pi_log)
                except np.linalg.LinAlgError:
                    pass

    # --------------------------------------------------------------- #
    #  Innovation statistics for data association
    # --------------------------------------------------------------- #

    def innovation_stats(self,
                         oid: int,
                         T_co_meas: np.ndarray,
                         R_icp: np.ndarray) -> Optional[tuple]:
        r"""Per-particle innovation :math:`(\nu, S, d^2, \log\!\mathcal{L})`, then particle-weight averaged."""
        collapsed = self.collapsed_object(oid)
        if collapsed is None:
            return None
        # Project camera-frame measurement to world frame via the
        # collapsed base pose + current camera extrinsic. This is the
        # single-Gaussian approximation used for the Hungarian cost
        # matrix; per-particle updates (in `update_observation`) remain
        # mixture-exact.
        base_pe = self.collapsed_base()
        T_wo_meas, R_wo = lift_measurement_world(
            T_co_meas, R_icp, self.T_bc, base_pe.T, Sigma_wb=None)
        return innovation_from_belief(
            collapsed.T, collapsed.cov, T_wo_meas, R_wo)

    # --------------------------------------------------------------- #
    #  Object deletion
    # --------------------------------------------------------------- #

    def delete_object(self, oid: int) -> bool:
        """Remove a track from every particle; returns True if it existed."""
        removed = False
        for p in self.particles:
            if oid in p.objects:
                del p.objects[oid]
                removed = True
        return removed

    # --------------------------------------------------------------- #
    #  Resampling
    # --------------------------------------------------------------- #

    def resample_if_needed(self, threshold_frac: float = 0.5) -> bool:
        """Systematic resampling when the effective sample size falls below threshold."""
        N = self.n_particles
        if self.ess() >= threshold_frac * N:
            return False

        ws = self.normalized_weights()
        cumsum = np.cumsum(ws)
        # Guard against float creep
        cumsum[-1] = 1.0
        u0 = float(self._rng.uniform(0.0, 1.0 / N))
        u = u0 + np.arange(N) / N
        idx = np.searchsorted(cumsum, u)
        idx = np.clip(idx, 0, N - 1)

        new_particles = []
        for i in idx:
            src = self.particles[int(i)]
            clone = Particle(
                T_wb=src.T_wb.copy(),
                log_weight=0.0,
                objects={
                    oid: ParticleObjectBelief(
                        mu=b.mu.copy(), cov=b.cov.copy())
                    for oid, b in src.objects.items()
                },
                prev_T_wb=(src.prev_T_wb.copy()
                           if src.prev_T_wb is not None else None),
            )
            new_particles.append(clone)
        self.particles = new_particles
        return True

    # --------------------------------------------------------------- #
    #  Centroid-only innovation (Phase C)
    # --------------------------------------------------------------- #

    def centroid_innovation_stats(self,
                                   oid: int,
                                   centroid_cam: np.ndarray,
                                   R_cam: Optional[np.ndarray] = None,
                                   ) -> Optional[tuple]:
        """Per-particle 3-DOF centroid innovation, particle-weight averaged."""
        collapsed = self.collapsed_object(oid)
        if collapsed is None:
            return None
        centroid_cam = np.asarray(centroid_cam, dtype=np.float64).reshape(3)
        T_wc = self.collapsed_T_wb() @ self.T_bc
        t_wo_meas = (T_wc @ np.array([centroid_cam[0],
                                        centroid_cam[1],
                                        centroid_cam[2], 1.0]))[:3]
        t_wo_prior = np.asarray(collapsed.T[:3, 3], dtype=np.float64)
        nu = t_wo_meas - t_wo_prior

        if R_cam is None:
            R_cam = np.diag([(0.02) ** 2] * 3)
        else:
            R_cam = 0.5 * (np.asarray(R_cam, dtype=np.float64)
                            + np.asarray(R_cam, dtype=np.float64).T)

        R_wc = T_wc[:3, :3]
        R_wo_tt = R_wc @ R_cam @ R_wc.T
        P_tt = np.asarray(collapsed.cov[:3, :3], dtype=np.float64)
        S = P_tt + R_wo_tt
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

    # --------------------------------------------------------------- #
    #  BasePoseBackend protocol (structural)
    # --------------------------------------------------------------- #

    def collapsed_T_wb(self) -> np.ndarray:
        """Current weighted-mean base pose (BasePoseBackend)."""
        return self.collapsed_base().T

    def known_oids(self) -> List[int]:
        """Union of oids across all particles (BasePoseBackend)."""
        oids: set = set()
        for p in self.particles:
            oids.update(p.objects.keys())
        return sorted(oids)

    def camera_frame_prior(self,
                            oid: int) -> Optional[np.ndarray]:
        r"""ICP seed :math:`T_{co}^{\text{pred}}` averaged across particles, or ``None``."""
        collapsed = self.collapsed_object(oid)
        if collapsed is None:
            return None
        T_wc = self.collapsed_T_wb() @ self.T_bc
        return np.linalg.inv(T_wc) @ collapsed.T

    def predict_static_all(self,
                            Q_fn: Callable[[int], np.ndarray],
                            skip_oids: Optional[set] = None,
                            P_max: Optional[np.ndarray] = None) -> None:
        """Run :meth:`predict_objects` for every oid not in ``skip_oids``."""
        skip = skip_oids or set()
        def _Q(oid: int, _p: Particle) -> np.ndarray:
            if oid in skip:
                return np.zeros((6, 6), dtype=np.float64)
            return Q_fn(oid)
        self.predict_objects(_Q, P_max=P_max)

    def collapsed_object_base(self, oid: int) -> Optional[PoseEstimate]:
        r"""Particle-mean base-frame :math:`(\mu, \Sigma)` for ``oid``, or ``None``."""
        pe = self.collapsed_object(oid)
        if pe is None:
            return None
        T_wb = self.collapsed_T_wb()
        T_wb_inv = np.linalg.inv(T_wb)
        mu_bo = T_wb_inv @ pe.T
        return PoseEstimate(T=mu_bo, cov=pe.cov.copy())

    def merge_tracks(self, oid_keep: int, oid_drop: int) -> bool:
        """Merge ``oid_drop`` into ``oid_keep`` per-particle."""
        if oid_keep == oid_drop:
            return False
        any_merged = False
        for p in self.particles:
            a = p.objects.get(oid_keep)
            b = p.objects.get(oid_drop)
            if a is None or b is None:
                continue
            try:
                mu_new, cov_new = merge_info_sum(a.mu, a.cov, b.mu, b.cov)
            except np.linalg.LinAlgError:
                continue
            a.mu = mu_new
            a.cov = _floor_diag(cov_new, self.P_min_diag)
            del p.objects[oid_drop]
            any_merged = True
        return any_merged

    def overwrite_object_pose(self, oid: int,
                              T_wo: np.ndarray,
                              P_wo: np.ndarray) -> bool:
        """Force every particle's belief for ``oid`` to a world-frame :math:`(T_{wo}, P_{wo})`."""
        T_wo = np.asarray(T_wo, dtype=np.float64)
        P_wo = np.asarray(P_wo, dtype=np.float64)
        if T_wo.shape != (4, 4) or P_wo.shape != (6, 6):
            raise ValueError("T_wo must be (4, 4) and P_wo (6, 6)")
        any_set = False
        for p in self.particles:
            if oid in p.objects:
                p.objects[oid].mu = T_wo.copy()
                p.objects[oid].cov = P_wo.copy()
                any_set = True
        return any_set

    # --------------------------------------------------------------- #
    #  Collapsing to single-Gaussian summaries (for legacy consumers)
    # --------------------------------------------------------------- #

    def collapsed_base(self) -> PoseEstimate:
        """Moment-match the base-pose particles to a single SE(3) Gaussian."""
        ws = self.normalized_weights()
        Ts = np.stack([p.T_wb for p in self.particles], axis=0)
        return ParticlePose(particles=Ts, weights=ws).to_gaussian()

    def collapsed_object(self, oid: int) -> Optional[PoseEstimate]:
        """Particle-mean world-frame :math:`(T_{wo}, P_{wo})` for ``oid``."""
        ws_all = self.normalized_weights()
        mus: List[np.ndarray] = []
        covs: List[np.ndarray] = []
        ws: List[float] = []
        for p, w in zip(self.particles, ws_all):
            b = p.objects.get(oid)
            if b is None:
                continue
            mus.append(b.mu)
            covs.append(b.cov)
            ws.append(w)
        if not mus:
            return None
        w_sum = float(np.sum(ws))
        if w_sum <= 0:
            return None
        ws_arr = np.asarray(ws) / w_sum

        pp_obj = ParticlePose(
            particles=np.stack(mus, axis=0),
            weights=ws_arr,
        )
        mean_pe = pp_obj.to_gaussian()  # .cov is the spread term

        expected_cov = sum(w * c for w, c in zip(ws_arr, covs))
        cov_total = mean_pe.cov + expected_cov
        cov_total = 0.5 * (cov_total + cov_total.T)
        return PoseEstimate(T=mean_pe.T, cov=cov_total)

    def collapsed_objects(self) -> Dict[int, PoseEstimate]:
        """Collapse every tracked object to a single-Gaussian summary."""
        oids = set()
        for p in self.particles:
            oids.update(p.objects.keys())
        out: Dict[int, PoseEstimate] = {}
        for oid in oids:
            pe = self.collapsed_object(oid)
            if pe is not None:
                out[oid] = pe
        return out

    # --------------------------------------------------------------- #
    #  Slow-tier reconcile (Option A: shift-and-inject collapsed posterior)
    # --------------------------------------------------------------- #

    def inject_posterior(self, oid: int, posterior: PoseEstimate) -> None:
        """Re-center every particle's belief from a slow-tier world-frame posterior."""
        for p in self.particles:
            if oid in p.objects:
                p.objects[oid] = ParticleObjectBelief(
                    mu=posterior.T.copy(),
                    cov=posterior.cov.copy(),
                )

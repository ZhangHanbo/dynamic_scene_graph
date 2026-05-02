"""Unit tests for the RBPF backend.

Parallel to `test_gaussian_state.py`; pins the RBPF-specific behaviour
(per-particle world-frame storage, per-frame T_bc lift, vision's
dual-role likelihood that reweights particles). A passing suite
confirms that `RBPFState` satisfies the `BasePoseBackend` contract and
handles head motion correctly -- the bug that was latent for the whole
production path before B.3.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R_

from pose_update.state.rbpf_state import RBPFState
from pose_update.state.ekf_se3 import se3_adjoint
from pose_update.state.slam_interface import PoseEstimate


def _T(t=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _is_psd(P: np.ndarray, tol: float = 1e-9) -> bool:
    sym = np.allclose(P, P.T, atol=1e-10)
    if not sym:
        return False
    eigs = np.linalg.eigvalsh(0.5 * (P + P.T))
    return bool(np.all(eigs >= -tol))


def _seed(n: int = 4, T_wb=None, T_bc=None,
           P_min_diag=None) -> RBPFState:
    """Build and initialise an RBPFState with N particles at `T_wb`."""
    s = RBPFState(n_particles=n, T_bc=T_bc, P_min_diag=P_min_diag,
                    rng=np.random.default_rng(0))
    pe = PoseEstimate(T=(T_wb if T_wb is not None else _T()),
                       cov=np.diag([1e-8]*6))
    s.ingest_slam(pe)
    return s


# ─────────────────────────────────────────────────────────────────────
# 1. Construction & camera extrinsic
# ─────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_defaults(self):
        s = RBPFState(n_particles=3)
        np.testing.assert_allclose(s.T_bc, np.eye(4))
        np.testing.assert_allclose(s._Ad_bc, np.eye(6))
        assert s.T_wb is None
        assert s.prev_T_wb is None
        assert s.particles == []

    def test_set_camera_extrinsic_updates_T_bc_and_Ad(self):
        s = RBPFState(n_particles=2)
        T_bc = _T(t=(0.15, 0.0, 1.1), rpy_deg=(0.0, 30.0, 0.0))
        s.set_camera_extrinsic(T_bc)
        np.testing.assert_allclose(s.T_bc, T_bc)
        np.testing.assert_allclose(s._Ad_bc, se3_adjoint(T_bc))

    def test_set_camera_extrinsic_rejects_bad_shape(self):
        s = RBPFState(n_particles=2)
        with pytest.raises(ValueError):
            s.set_camera_extrinsic(np.eye(3))


# ─────────────────────────────────────────────────────────────────────
# 2. SLAM ingest
# ─────────────────────────────────────────────────────────────────────

class TestSLAMIngest:
    def test_first_ingest_populates_particles(self):
        s = RBPFState(n_particles=5, rng=np.random.default_rng(0))
        s.ingest_slam(PoseEstimate(T=_T(t=(1.0, 0., 0.)),
                                     cov=np.diag([1e-6]*6)))
        assert len(s.particles) == 5
        for p in s.particles:
            # Particles sampled near the supplied T_wb; prev_T_wb is None.
            assert p.prev_T_wb is None

    def test_second_ingest_caches_prev_per_particle(self):
        s = _seed(n=4, T_wb=_T(t=(0.0, 0.0, 0.0)))
        for p in s.particles:
            assert p.prev_T_wb is None
        s.ingest_slam(PoseEstimate(T=_T(t=(0.3, 0., 0.)),
                                     cov=np.diag([1e-8]*6)))
        for p in s.particles:
            assert p.prev_T_wb is not None

    def test_ingest_refreshes_collapsed_view(self):
        # Gaussian SLAM sampling perturbs each particle's T_wb around
        # the provided mean, so with N=4 and cov=1e-8 the collapsed
        # mean has ~1e-4 sampling error. That's expected; use a
        # tolerance comfortably above the noise.
        s = _seed(n=4, T_wb=_T(t=(0.0, 0.0, 0.0)))
        assert s.T_wb is not None
        np.testing.assert_allclose(s.T_wb[:3, 3], (0.0, 0.0, 0.0), atol=1e-3)
        s.ingest_slam(PoseEstimate(T=_T(t=(0.5, 0., 0.)),
                                     cov=np.diag([1e-8]*6)))
        np.testing.assert_allclose(s.T_wb[:3, 3], (0.5, 0.0, 0.0), atol=1e-3)
        # `prev_T_wb` on the backend is the *previous collapsed mean*,
        # which was approximately (0, 0, 0) at frame 1.
        np.testing.assert_allclose(s.prev_T_wb[:3, 3], (0.0, 0.0, 0.0),
                                     atol=1e-3)


# ─────────────────────────────────────────────────────────────────────
# 3. Object lifecycle (per-particle)
# ─────────────────────────────────────────────────────────────────────

class TestEnsureObject:
    def test_ensure_lifts_mean_through_T_wb_T_bc_per_particle(self):
        """μ^k_o = T_wb^k · T_bc · T_co_meas (the regression for the
        long-standing implicit T_bc=I bug in RBPFState).

        The cov is stored verbatim (not Ad-lifted) -- same convention
        as `GaussianState.ensure_object`.
        """
        T_wb = _T(t=(2.0, 0.0, 0.0))
        T_bc = _T(t=(0.15, 0.0, 1.1), rpy_deg=(0.0, 30.0, 0.0))
        s = _seed(n=3, T_wb=T_wb, T_bc=T_bc)
        T_co = _T(t=(0.5, 0.0, 0.8))
        R = np.diag([1e-3] * 6)

        assert s.ensure_object(42, T_co, R) is True
        for p in s.particles:
            belief = p.objects[42]
            expected = p.T_wb @ T_bc @ T_co
            np.testing.assert_allclose(belief.mu, expected, atol=1e-10)
            np.testing.assert_allclose(belief.cov, R, atol=1e-12)

    def test_ensure_returns_false_when_oid_present_everywhere(self):
        s = _seed(n=2)
        T_co = _T(t=(0.5,)*3)
        R = np.diag([1e-3] * 6)
        s.ensure_object(1, T_co, R)
        assert s.ensure_object(1, T_co, R) is False


# ─────────────────────────────────────────────────────────────────────
# 4. Predict (world-frame storage -> identity-mean)
# ─────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_static_all_preserves_mean_inflates_cov(self):
        """World-frame storage: μ unchanged; cov += Q per particle."""
        s = _seed(n=3)
        s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-3]*6))
        mus = [p.objects[1].mu.copy() for p in s.particles]
        covs = [p.objects[1].cov.copy() for p in s.particles]
        Q = np.diag([1e-5]*6)
        s.predict_static_all(lambda oid: Q)
        for p, mu0, cov0 in zip(s.particles, mus, covs):
            np.testing.assert_allclose(p.objects[1].mu, mu0, atol=1e-12)
            np.testing.assert_allclose(p.objects[1].cov, cov0 + Q,
                                        atol=1e-12)

    def test_predict_static_all_skips_held_oids(self):
        s = _seed(n=2)
        s.ensure_object(1, _T(t=(0.5,)*3), np.diag([1e-3]*6))
        s.ensure_object(2, _T(t=(0.6,)*3), np.diag([1e-3]*6))
        covs_before = {oid: [p.objects[oid].cov.copy() for p in s.particles]
                        for oid in (1, 2)}
        Q = np.diag([1.0]*6)     # huge Q
        s.predict_static_all(lambda oid: Q, skip_oids={1})
        # oid=1 was skipped -> Q=0 -> cov unchanged.
        for p, cov0 in zip(s.particles, covs_before[1]):
            np.testing.assert_allclose(p.objects[1].cov, cov0, atol=1e-12)
        # oid=2 was NOT skipped -> cov inflates.
        for p, cov0 in zip(s.particles, covs_before[2]):
            np.testing.assert_allclose(p.objects[2].cov, cov0 + Q,
                                        atol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# 5. Measurement update (per-particle Joseph + log-weight dual role)
# ─────────────────────────────────────────────────────────────────────

class TestUpdateObservation:
    def test_update_applies_per_particle(self):
        s = _seed(n=3)
        s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-3]*6))
        mus0 = [p.objects[1].mu.copy() for p in s.particles]
        # Perturb the measurement — each particle's posterior should
        # shift toward it.
        T_co_meas = _T(t=(0.55, 0.0, 0.5))
        R = np.diag([1e-4]*6)
        s.update_observation(1, T_co_meas, R, iekf_iters=2)
        for p, mu0 in zip(s.particles, mus0):
            assert not np.allclose(p.objects[1].mu, mu0, atol=1e-6), \
                "posterior did not move; update had no effect"
            assert _is_psd(p.objects[1].cov)

    def test_update_reweights_particles(self):
        """Vision's dual role: same log-L that moves μ also adds to
        particle.log_weight."""
        s = _seed(n=3)
        s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-3]*6))
        lw0 = np.array([p.log_weight for p in s.particles])
        s.update_observation(1, _T(t=(0.5, 0., 0.5)),
                              np.diag([1e-4]*6), iekf_iters=1)
        lw1 = np.array([p.log_weight for p in s.particles])
        assert not np.allclose(lw0, lw1), \
            "particle log_weights did not change after update"

    def test_update_huber_zero_preserves_belief_mean(self):
        """The shared `joseph_update` helper returns early on huber_w<=0,
        so the belief mean and covariance are unchanged. (The particle
        log-weight still absorbs the innovation likelihood; RBPF's
        original contract and the orchestrator's convention is that
        the caller doesn't invoke `update_observation` with huber_w=0
        -- it routes to the miss branch instead -- but the early-return
        in joseph_update keeps this path safe even if a caller slips up.)
        """
        s = _seed(n=2)
        s.ensure_object(1, _T(t=(0.5, 0., 0.5)), np.diag([1e-3]*6))
        mus = [p.objects[1].mu.copy() for p in s.particles]
        covs = [p.objects[1].cov.copy() for p in s.particles]
        s.update_observation(1, _T(t=(0.6, 0., 0.5)),
                              np.diag([1e-4]*6), huber_w=0.0)
        for p, mu0, cov0 in zip(s.particles, mus, covs):
            np.testing.assert_allclose(p.objects[1].mu, mu0)
            np.testing.assert_allclose(p.objects[1].cov, cov0)


# ─────────────────────────────────────────────────────────────────────
# 6. Collapsed views (BasePoseBackend)
# ─────────────────────────────────────────────────────────────────────

class TestCollapsedViews:
    def test_camera_frame_prior_is_identity_when_T_wb_T_bc_identity(self):
        """With T_wb = I, T_bc = I: T_co_prior == collapsed μ_wo."""
        s = _seed(n=3, T_wb=_T(), T_bc=_T())
        T_co = _T(t=(0.5, 0.1, 0.8))
        s.ensure_object(42, T_co, np.diag([1e-3]*6))
        prior = s.camera_frame_prior(42)
        assert prior is not None
        # collapsed μ_wo should be close to T_co (μ^k_o = T_wb^k · I · T_co,
        # particles are near T_wb = I).
        np.testing.assert_allclose(prior[:3, 3], T_co[:3, 3], atol=1e-3)

    def test_camera_frame_prior_inverts_lift(self):
        """T_co^pred = (T_wb · T_bc)^{-1} · μ_wo — round-trip from ensure."""
        T_wb = _T(t=(1.0, 0.0, 0.0))
        T_bc = _T(t=(0.15, 0.0, 1.1))
        s = _seed(n=1, T_wb=T_wb, T_bc=T_bc)         # N=1 so collapsed == particle
        T_co_true = _T(t=(0.5, 0.0, 0.8))
        s.ensure_object(42, T_co_true, np.diag([1e-3]*6))
        prior = s.camera_frame_prior(42)
        np.testing.assert_allclose(prior, T_co_true, atol=1e-9)

    def test_known_oids_is_union_across_particles(self):
        s = _seed(n=3)
        s.ensure_object(1, _T(t=(0.5,)*3), np.diag([1e-3]*6))
        s.ensure_object(7, _T(t=(0.6,)*3), np.diag([1e-3]*6))
        assert set(s.known_oids()) == {1, 7}


# ─────────────────────────────────────────────────────────────────────
# 7. Merge (per-particle info fusion)
# ─────────────────────────────────────────────────────────────────────

class TestMergeTracks:
    def test_merge_deletes_drop_everywhere(self):
        s = _seed(n=2)
        s.ensure_object(1, _T(t=(0.5,)*3), np.diag([1e-3]*6))
        s.ensure_object(2, _T(t=(0.51, 0.0, 0.5)), np.diag([1e-3]*6))
        assert s.merge_tracks(1, 2) is True
        for p in s.particles:
            assert 1 in p.objects and 2 not in p.objects

    def test_merge_returns_false_when_keep_equals_drop(self):
        s = _seed(n=2)
        s.ensure_object(1, _T(), np.diag([1e-3]*6))
        assert s.merge_tracks(1, 1) is False

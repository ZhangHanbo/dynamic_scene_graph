"""Unit tests for pose_update/rbpf_state.py.

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python -m pytest tests/test_rbpf_state.py -v
"""
import os
import sys

import numpy as np
import pytest

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from ekf_tracker.state.rbpf_state import (
    ParticleObjectBelief, Particle, RBPFState,
)
from utils.slam_interface import PoseEstimate, ParticlePose
from utils.ekf_se3 import se3_exp, se3_log


def _T(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ─────────────────────────────────────────────────────────────────────
# SLAM ingestion
# ─────────────────────────────────────────────────────────────────────

class TestIngestSlam:
    def test_initializes_from_gaussian(self):
        st = RBPFState(n_particles=16, rng=_rng(0))
        assert not st.initialized
        st.ingest_slam(PoseEstimate(T=_T(0.5, 0, 0),
                                     cov=np.diag([1e-4]*6)))
        assert st.initialized
        assert len(st.particles) == 16
        # Each particle should start near the mean
        for p in st.particles:
            assert np.linalg.norm(p.T_wb[:3, 3] - np.array([0.5, 0, 0])) < 0.05
            assert p.log_weight == 0.0
            assert p.objects == {}

    def test_initializes_from_particles(self):
        st = RBPFState(n_particles=8, rng=_rng(1))
        Ts = np.stack([_T(x=0.1 * i) for i in range(8)], axis=0)
        w = np.ones(8) / 8.0
        st.ingest_slam(ParticlePose(particles=Ts, weights=w))
        assert st.initialized
        for k, p in enumerate(st.particles):
            np.testing.assert_allclose(p.T_wb[:3, 3], [0.1 * k, 0, 0])

    def test_subsequent_call_refreshes_Twb_keeps_objects(self):
        st = RBPFState(n_particles=4, rng=_rng(2))
        st.ingest_slam(PoseEstimate(T=_T(0, 0, 0), cov=np.diag([1e-6]*6)))
        # Inject a fake object per particle
        for p in st.particles:
            p.objects[42] = ParticleObjectBelief(
                mu=_T(1.0, 0, 0), cov=np.eye(6) * 0.01)
            p.log_weight = 7.0

        st.ingest_slam(PoseEstimate(T=_T(2.0, 0, 0), cov=np.diag([1e-6]*6)))
        # T_wb updated, objects and log_weight preserved
        for p in st.particles:
            assert np.linalg.norm(p.T_wb[:3, 3] - np.array([2.0, 0, 0])) < 0.01
            assert 42 in p.objects
            assert p.log_weight == 7.0


# ─────────────────────────────────────────────────────────────────────
# Ensure object & world-frame init
# ─────────────────────────────────────────────────────────────────────

class TestEnsureObject:
    def test_per_particle_world_frame_init(self):
        st = RBPFState(n_particles=4, rng=_rng(3))
        # Make particles have different T_wb
        st.particles = [
            Particle(T_wb=_T(x=0.0), log_weight=0.0),
            Particle(T_wb=_T(x=1.0), log_weight=0.0),
            Particle(T_wb=_T(x=2.0), log_weight=0.0),
            Particle(T_wb=_T(x=3.0), log_weight=0.0),
        ]
        added = st.ensure_object(
            oid=5, T_co_meas=_T(x=0.5), init_cov=np.eye(6) * 0.01)
        assert added
        # Each particle's μ_wo = T_wb · T_co; so x varies from 0.5 to 3.5
        xs = [p.objects[5].mu[0, 3] for p in st.particles]
        assert xs == [0.5, 1.5, 2.5, 3.5]

    def test_idempotent(self):
        st = RBPFState(n_particles=2, rng=_rng(3))
        st.particles = [
            Particle(T_wb=_T(), log_weight=0.0),
            Particle(T_wb=_T(), log_weight=0.0),
        ]
        assert st.ensure_object(9, _T(), np.eye(6))
        # Second call: no-op (already exists)
        assert not st.ensure_object(9, _T(x=99), np.eye(6))
        # Original means retained (not overwritten by the second call)
        for p in st.particles:
            assert p.objects[9].mu[0, 3] == 0.0


# ─────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_inflates_cov(self):
        st = RBPFState(n_particles=3, rng=_rng(4))
        st.particles = [
            Particle(T_wb=_T(), log_weight=0.0) for _ in range(3)
        ]
        for p in st.particles:
            p.objects[1] = ParticleObjectBelief(
                mu=_T(), cov=np.eye(6) * 0.01)
        Q = np.eye(6) * 0.001
        st.predict_objects(lambda oid, p: Q)
        for p in st.particles:
            # Covariance grew by exactly Q
            np.testing.assert_allclose(
                p.objects[1].cov, np.eye(6) * 0.011)

    def test_rigid_attachment_moves_mean_and_inflates(self):
        st = RBPFState(n_particles=2, rng=_rng(5))
        st.particles = [
            Particle(T_wb=_T(x=1.0), log_weight=0.0),
            Particle(T_wb=_T(x=2.0), log_weight=0.0),
        ]
        for p in st.particles:
            p.objects[7] = ParticleObjectBelief(
                mu=p.T_wb @ _T(x=0.3),  # object 30cm in front of base
                cov=np.eye(6) * 0.001)

        # Translation-only ΔT in base frame: gripper moves +5cm in x
        delta = _T(x=0.05)
        Q_manip = np.eye(6) * 1e-6
        st.rigid_attachment_predict(7, delta, Q_manip)

        # In base frame the move is +5cm; world-frame T_wb · Δ · T_wb⁻¹ for
        # pure translation just rotates the delta; with identity base rotation
        # the world-frame delta is still +5cm in x. So each particle's
        # object mean should move from (1.3, 2.3) to (1.35, 2.35).
        np.testing.assert_allclose(st.particles[0].objects[7].mu[0, 3], 1.35)
        np.testing.assert_allclose(st.particles[1].objects[7].mu[0, 3], 2.35)
        # Covariance inflated
        for p in st.particles:
            assert np.trace(p.objects[7].cov) > np.trace(np.eye(6) * 0.001)


# ─────────────────────────────────────────────────────────────────────
# Observation update / weighting (vision's dual role)
# ─────────────────────────────────────────────────────────────────────

class TestUpdateObservation:
    def test_covariance_shrinks_on_consistent_obs(self):
        st = RBPFState(n_particles=1, rng=_rng(6))
        st.particles = [Particle(T_wb=_T(), log_weight=0.0)]
        st.particles[0].objects[0] = ParticleObjectBelief(
            mu=_T(x=0.5), cov=np.diag([0.05]*6))
        R = np.eye(6) * 1e-4
        trace_before = np.trace(st.particles[0].objects[0].cov)
        for _ in range(10):
            st.update_observation(0, _T(x=0.5), R, iekf_iters=1)
        trace_after = np.trace(st.particles[0].objects[0].cov)
        assert trace_after < trace_before

    def test_weight_higher_for_better_fit(self):
        """Particle whose T_wb makes μ_wo closer to the observation gets a
        higher per-frame log-likelihood."""
        st = RBPFState(n_particles=2, rng=_rng(7))
        # Particle A: T_wb at origin; object mu at (1,0,0).
        # Particle B: T_wb at (0.5, 0, 0); object mu at (1,0,0) as well.
        # Observation T_co = (1, 0, 0) — so μ_wo_meas_A = (1,0,0),
        # μ_wo_meas_B = (1.5, 0, 0). A's belief matches, B's doesn't.
        st.particles = [
            Particle(T_wb=_T(), log_weight=0.0),
            Particle(T_wb=_T(x=0.5), log_weight=0.0),
        ]
        for p in st.particles:
            p.objects[0] = ParticleObjectBelief(
                mu=_T(x=1.0), cov=np.diag([0.01]*6))
        st.update_observation(0, _T(x=1.0), np.eye(6) * 1e-4, iekf_iters=1)
        # A should have higher log-likelihood than B
        assert st.particles[0].log_weight > st.particles[1].log_weight

    def test_dual_role_means_reweighting_shifts_ess(self):
        """When the object belief (μ_wo) is the SAME across particles but
        T_wb^k differs, the per-particle innovation differs — particles
        whose base pose is consistent with the observation get higher
        weight. ESS should drop.
        """
        st = RBPFState(n_particles=64, rng=_rng(8))
        st.ingest_slam(PoseEstimate(T=_T(), cov=np.diag([1e-2]*6)))
        # Seed a SHARED world-frame belief (as if from a landmark map).
        # Particles with T_wb close to origin will be consistent; those
        # further will not — vision differentiates them.
        for p in st.particles:
            p.objects[0] = ParticleObjectBelief(
                mu=_T(x=1.0), cov=np.diag([0.01]*6))
        ess0 = st.ess()
        for _ in range(5):
            st.update_observation(0, _T(x=1.0), np.eye(6) * 1e-4, iekf_iters=1)
        ess1 = st.ess()
        assert ess1 < ess0


# ─────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────

class TestResampling:
    def test_no_resample_when_ess_high(self):
        st = RBPFState(n_particles=16, rng=_rng(9))
        st.ingest_slam(PoseEstimate(T=_T(), cov=np.diag([1e-4]*6)))
        # All weights equal → ESS = N → no resample
        assert not st.resample_if_needed(threshold_frac=0.5)

    def test_resamples_when_ess_low(self):
        st = RBPFState(n_particles=16, rng=_rng(10))
        st.ingest_slam(PoseEstimate(T=_T(), cov=np.diag([1e-4]*6)))
        # Make one particle dominate
        for k, p in enumerate(st.particles):
            p.log_weight = 100.0 if k == 0 else 0.0
        assert st.ess() < 2.0
        did = st.resample_if_needed(threshold_frac=0.5)
        assert did
        # All particles should now be (near) copies of the dominant slot.
        t0 = st.particles[0].T_wb
        for p in st.particles:
            # Same underlying T_wb value; copies are independent arrays
            assert np.allclose(p.T_wb, t0)
            assert p.log_weight == 0.0

    def test_resample_deep_copies_objects(self):
        st = RBPFState(n_particles=4, rng=_rng(11))
        st.ingest_slam(PoseEstimate(T=_T(), cov=np.diag([1e-6]*6)))
        for p in st.particles:
            p.objects[3] = ParticleObjectBelief(
                mu=_T(x=0.7), cov=np.eye(6) * 0.001)
        # Force dominance
        st.particles[0].log_weight = 50.0
        st.resample_if_needed(threshold_frac=0.99)
        # Mutating one particle's belief should not affect the others
        st.particles[0].objects[3].mu[0, 3] = 999.0
        for p in st.particles[1:]:
            assert p.objects[3].mu[0, 3] != 999.0


# ─────────────────────────────────────────────────────────────────────
# Collapsed summaries
# ─────────────────────────────────────────────────────────────────────

class TestCollapse:
    def test_collapsed_base_agrees_with_particle_spread(self):
        st = RBPFState(n_particles=100, rng=_rng(12))
        st.ingest_slam(PoseEstimate(T=_T(x=0.3), cov=np.diag([1e-2]*6)))
        pe = st.collapsed_base()
        np.testing.assert_allclose(pe.T[:3, 3], [0.3, 0, 0], atol=0.03)
        assert np.trace(pe.cov) > 0

    def test_collapsed_object_mixture(self):
        """When particles disagree on T_wb, the collapsed object covariance
        reflects both the per-particle EKF cov and the spread across
        particles."""
        st = RBPFState(n_particles=16, rng=_rng(13))
        st.particles = [
            Particle(T_wb=_T(x=0.1 * k), log_weight=0.0)
            for k in range(16)
        ]
        for p in st.particles:
            p.objects[0] = ParticleObjectBelief(
                mu=p.T_wb @ _T(x=0.5),
                cov=np.diag([1e-4]*6),
            )
        pe = st.collapsed_object(0)
        assert pe is not None
        # Mean should be around x = 0.5 + mean(0 to 1.5) = 0.5 + 0.75 = 1.25
        assert 1.0 < pe.T[0, 3] < 1.5
        # Translational x-variance should reflect the spread (≈ var(0..1.5) ≈ 0.21),
        # much larger than per-particle cov (1e-4).
        assert pe.cov[0, 0] > 0.05


# ─────────────────────────────────────────────────────────────────────
# Slow-tier reconcile hook
# ─────────────────────────────────────────────────────────────────────

class TestInjectPosterior:
    def test_overwrites_all_particles(self):
        st = RBPFState(n_particles=3, rng=_rng(14))
        st.particles = [
            Particle(T_wb=_T(), log_weight=0.0) for _ in range(3)
        ]
        for k, p in enumerate(st.particles):
            p.objects[0] = ParticleObjectBelief(
                mu=_T(x=float(k)), cov=np.eye(6) * 0.1)
        target = PoseEstimate(T=_T(x=42.0), cov=np.eye(6) * 1e-6)
        st.inject_posterior(0, target)
        for p in st.particles:
            np.testing.assert_allclose(p.objects[0].mu[0, 3], 42.0)
            np.testing.assert_allclose(p.objects[0].cov, target.cov)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

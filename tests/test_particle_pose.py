"""
Unit tests for ParticlePose and particle-filter backend support.

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python -m pytest tests/test_particle_pose.py -v
"""

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from pose_update.state.slam_interface import (
    PoseEstimate, ParticlePose, as_gaussian,
    sample_particles_from_gaussian,
    PassThroughSlam, ParticlePassThroughSlam,
)
from pose_update.state.ekf_se3 import se3_exp, se3_log
from pose_update.orchestrator import TwoTierOrchestrator, TriggerConfig


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_pose(tx=0.0, ty=0.0, tz=0.0, yaw=0.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


# ─────────────────────────────────────────────────────────────────────
# ParticlePose construction and normalization
# ─────────────────────────────────────────────────────────────────────

class TestParticlePoseBasics:
    def test_construction_normalizes_weights(self):
        particles = np.array([_make_pose(), _make_pose(0.1), _make_pose(0.2)])
        weights = np.array([2.0, 3.0, 5.0])  # sum = 10
        pp = ParticlePose(particles=particles, weights=weights)
        assert abs(pp.weights.sum() - 1.0) < 1e-9
        np.testing.assert_array_almost_equal(
            pp.weights, [0.2, 0.3, 0.5])

    def test_zero_weights_become_uniform(self):
        particles = np.array([_make_pose(), _make_pose(0.1)])
        weights = np.zeros(2)
        pp = ParticlePose(particles=particles, weights=weights)
        np.testing.assert_array_almost_equal(pp.weights, [0.5, 0.5])

    def test_rejects_wrong_particle_shape(self):
        with pytest.raises(AssertionError):
            ParticlePose(particles=np.eye(4), weights=np.array([1.0]))

    def test_rejects_mismatched_weights(self):
        particles = np.stack([np.eye(4)] * 3)
        with pytest.raises(AssertionError):
            ParticlePose(particles=particles, weights=np.array([0.5, 0.5]))

    def test_effective_sample_size(self):
        particles = np.stack([np.eye(4)] * 4)
        # Uniform: ESS = N
        pp_uniform = ParticlePose(particles=particles, weights=np.ones(4))
        assert abs(pp_uniform.effective_sample_size() - 4.0) < 1e-6
        # Degenerate: one particle has all weight
        pp_deg = ParticlePose(
            particles=particles, weights=np.array([1.0, 0.0, 0.0, 0.0]))
        assert abs(pp_deg.effective_sample_size() - 1.0) < 1e-6

    def test_map_pose_picks_max_weight(self):
        particles = np.stack([_make_pose(), _make_pose(1.0), _make_pose(2.0)])
        pp = ParticlePose(particles=particles,
                          weights=np.array([0.1, 0.7, 0.2]))
        map_T = pp.map_pose()
        assert abs(map_T[0, 3] - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────
# Moment matching: ParticlePose -> PoseEstimate
# ─────────────────────────────────────────────────────────────────────

class TestToGaussian:
    def test_delta_distribution_gives_zero_covariance(self):
        T = _make_pose(1.0, 2.0, 0.5)
        particles = np.stack([T] * 10)
        pp = ParticlePose(particles=particles, weights=np.ones(10))
        pe = pp.to_gaussian()
        np.testing.assert_array_almost_equal(pe.T, T, decimal=6)
        # Covariance should be ≈ 0 (plus the tiny regularizer)
        assert np.trace(pe.cov) < 1e-8

    def test_mean_recovers_ground_truth(self):
        """Sample particles from a known Gaussian, check that to_gaussian()
        recovers the mean within sampling noise."""
        rng = np.random.default_rng(42)
        true_T = _make_pose(1.0, 2.0, 0.5, yaw=0.3)
        true_cov = np.diag([0.01, 0.01, 0.005, 0.001, 0.001, 0.01])
        pe = PoseEstimate(T=true_T, cov=true_cov)
        # Draw many particles
        pp = sample_particles_from_gaussian(pe, n_samples=2000, rng=rng)
        pe_back = pp.to_gaussian()

        # Mean recovery: within 2σ of each axis
        sigmas = np.sqrt(np.diag(true_cov))
        translation_error = np.linalg.norm(pe_back.T[:3, 3] - true_T[:3, 3])
        assert translation_error < 3 * sigmas[:3].max() / np.sqrt(100), \
            f"Mean off by {translation_error}"

    def test_covariance_recovers_ground_truth_isotropic(self):
        rng = np.random.default_rng(7)
        true_T = np.eye(4)
        true_cov = np.eye(6) * 0.01
        pe = PoseEstimate(T=true_T, cov=true_cov)
        pp = sample_particles_from_gaussian(pe, n_samples=3000, rng=rng)
        pe_back = pp.to_gaussian()
        # Covariance should be close to truth (relative error)
        rel_err = np.linalg.norm(pe_back.cov - true_cov) / np.linalg.norm(true_cov)
        assert rel_err < 0.15, \
            f"Covariance relative error {rel_err:.3f} too large:\n{pe_back.cov}"

    def test_weighted_mean_biases_toward_heavy_particles(self):
        """A bimodal distribution with most weight on one mode should return
        a mean near that mode."""
        particles = np.stack([_make_pose(0.0), _make_pose(10.0)])
        pp = ParticlePose(particles=particles,
                          weights=np.array([0.95, 0.05]))
        pe = pp.to_gaussian()
        # Mean should be much closer to 0 than 10
        assert pe.T[0, 3] < 2.0


class TestAsGaussian:
    def test_gaussian_input_passthrough(self):
        pe = PoseEstimate(T=np.eye(4), cov=np.eye(6) * 0.1)
        result = as_gaussian(pe)
        assert result is pe

    def test_particle_input_converts(self):
        particles = np.stack([np.eye(4)] * 5)
        pp = ParticlePose(particles=particles, weights=np.ones(5))
        pe = as_gaussian(pp)
        assert isinstance(pe, PoseEstimate)
        np.testing.assert_array_almost_equal(pe.T, np.eye(4), decimal=6)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            as_gaussian("not a pose")


# ─────────────────────────────────────────────────────────────────────
# Round-trip sampling
# ─────────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_gaussian_to_particles_to_gaussian(self):
        rng = np.random.default_rng(0)
        true_T = _make_pose(0.5, -0.3, 1.0, yaw=0.5)
        true_cov = np.diag([0.02, 0.02, 0.005, 0.01, 0.01, 0.02])
        pe = PoseEstimate(T=true_T, cov=true_cov)

        pp = sample_particles_from_gaussian(pe, n_samples=2000, rng=rng)
        assert pp.n == 2000

        pe_back = pp.to_gaussian()
        # Pose recovered close to original
        xi_err = se3_log(np.linalg.inv(pe.T) @ pe_back.T)
        assert np.linalg.norm(xi_err) < 0.05


# ─────────────────────────────────────────────────────────────────────
# ParticlePassThroughSlam backend
# ─────────────────────────────────────────────────────────────────────

class TestParticleBackend:
    def test_returns_prescribed_particles(self):
        p1 = ParticlePose(
            particles=np.stack([_make_pose(), _make_pose(0.1)]),
            weights=np.ones(2))
        p2 = ParticlePose(
            particles=np.stack([_make_pose(1.0), _make_pose(1.1)]),
            weights=np.ones(2))
        slam = ParticlePassThroughSlam([p1, p2])

        r1 = slam.step(None, None)
        r2 = slam.step(None, None)
        assert isinstance(r1, ParticlePose) and isinstance(r2, ParticlePose)
        assert r1.n == 2
        # Second frame has shifted mean
        g2 = r2.to_gaussian()
        assert g2.T[0, 3] > 0.9

    def test_reset(self):
        pp = ParticlePose(
            particles=np.stack([np.eye(4)] * 3),
            weights=np.ones(3))
        slam = ParticlePassThroughSlam([pp])
        _ = slam.step(None, None)
        slam.reset()
        # Should succeed again without IndexError
        _ = slam.step(None, None)


# ─────────────────────────────────────────────────────────────────────
# Orchestrator accepts either backend transparently
# ─────────────────────────────────────────────────────────────────────

class TestOrchestratorWithParticleBackend:
    def test_orchestrator_step_accepts_particle_backend(self):
        rng = np.random.default_rng(0)
        # Make a particle pose cloud around identity
        def _make_pp(base_T, sigma=0.01, n=100):
            xis = rng.standard_normal(size=(n, 6)) * sigma
            particles = np.stack([base_T @ se3_exp(xi) for xi in xis], axis=0)
            return ParticlePose(particles=particles, weights=np.ones(n))

        poses = [_make_pp(np.eye(4)) for _ in range(5)]
        slam = ParticlePassThroughSlam(poses)

        orch = TwoTierOrchestrator(
            slam, trigger=TriggerConfig(periodic_every_n_frames=-1))

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        det = {
            "id": 0, "label": "cup",
            "mask": np.zeros((100, 100), dtype=np.uint8),
            "score": 0.8,
            "T_co": _make_pose(0.3, 0.0, 0.5),
            "R_icp": np.eye(6) * 1e-4,
            "fitness": 0.9, "rmse": 0.002,
        }
        det["mask"][20:40, 20:40] = 1

        report = orch.step(rgb, depth, [det],
                           {"phase": "idle", "held_obj_id": None})
        # Orchestrator exposed both the Gaussianized form and the raw particles
        assert isinstance(report["slam_pose"], PoseEstimate)
        assert isinstance(report["slam_raw"], ParticlePose)
        # Object created and has a finite, PSD covariance
        assert 0 in orch.objects
        cov = orch.objects[0]["cov"]
        assert np.all(np.linalg.eigvalsh(cov) > -1e-9)

    def test_orchestrator_gaussian_backend_still_works(self):
        """Regression: existing Gaussian backends keep working."""
        slam = PassThroughSlam([np.eye(4)] * 5)
        orch = TwoTierOrchestrator(
            slam, trigger=TriggerConfig(periodic_every_n_frames=-1))

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        det = {
            "id": 0, "label": "cup",
            "mask": np.zeros((100, 100), dtype=np.uint8),
            "score": 0.8,
            "T_co": _make_pose(0.3, 0.0, 0.5),
            "R_icp": np.eye(6) * 1e-4,
            "fitness": 0.9, "rmse": 0.002,
        }
        det["mask"][20:40, 20:40] = 1

        report = orch.step(rgb, depth, [det],
                           {"phase": "idle", "held_obj_id": None})
        assert isinstance(report["slam_pose"], PoseEstimate)
        # slam_raw is the same as slam_pose for a Gaussian backend
        assert isinstance(report["slam_raw"], PoseEstimate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for pose_update/ekf_se3.py (Task 2).

Purely synthetic — no trajectory data required.

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python -m pytest tests/test_ekf_se3.py -v
"""

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from utils.ekf_se3 import (
    se3_exp, se3_log, se3_adjoint,
    ekf_predict, ekf_update, ekf_update_base_frame,
    compose_observation_noise,
    pose_entropy, pose_is_uncertain,
    process_noise_for_phase, update_label_belief,
)


def _random_pose(seed=None):
    if seed is not None:
        np.random.seed(seed)
    T = np.eye(4)
    T[:3, :3] = Rotation.random().as_matrix()
    T[:3, 3] = np.random.uniform(-1, 1, size=3)
    return T


class TestSE3Maps:
    def test_exp_log_roundtrip_small(self):
        xi = np.array([0.01, 0.02, -0.03, 0.001, -0.002, 0.003])
        T = se3_exp(xi)
        xi_back = se3_log(T)
        np.testing.assert_array_almost_equal(xi_back, xi, decimal=6)

    def test_exp_log_roundtrip_large(self):
        xi = np.array([0.5, -0.3, 0.7, 0.8, -0.4, 0.2])
        T = se3_exp(xi)
        xi_back = se3_log(T)
        np.testing.assert_array_almost_equal(xi_back, xi, decimal=5)

    def test_exp_zero_is_identity(self):
        T = se3_exp(np.zeros(6))
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_log_identity_is_zero(self):
        xi = se3_log(np.eye(4))
        np.testing.assert_array_almost_equal(xi, np.zeros(6), decimal=6)

    def test_adjoint_shape_and_structure(self):
        T = _random_pose(seed=0)
        Ad = se3_adjoint(T)
        assert Ad.shape == (6, 6)
        # Bottom-left 3x3 should be zero
        np.testing.assert_array_almost_equal(Ad[3:, :3], np.zeros((3, 3)))


class TestEKFPredict:
    def test_predict_grows_cov(self):
        T = np.eye(4)
        cov = np.eye(6) * 0.01
        Q = np.eye(6) * 0.001
        T2, cov2 = ekf_predict(T, cov, Q)
        # Mean unchanged
        np.testing.assert_array_almost_equal(T2, T)
        # Covariance inflated
        assert np.trace(cov2) > np.trace(cov)
        np.testing.assert_array_almost_equal(cov2, cov + Q)

    def test_predict_monotonic_growth(self):
        T = np.eye(4)
        cov = np.eye(6) * 0.01
        Q = np.eye(6) * 0.001
        traces = [np.trace(cov)]
        for _ in range(10):
            T, cov = ekf_predict(T, cov, Q)
            traces.append(np.trace(cov))
        for a, b in zip(traces[:-1], traces[1:]):
            assert b > a


class TestEKFUpdate:
    def test_update_with_perfect_observation_shrinks_cov(self):
        T_prior = np.eye(4)
        cov_prior = np.eye(6) * 0.1
        T_meas = se3_exp(np.array([0.01, 0, 0, 0, 0, 0]))
        R = np.eye(6) * 1e-6  # very confident observation
        T_post, cov_post = ekf_update(T_prior, cov_prior, T_meas, R)

        # Posterior should move toward measurement
        delta_to_meas = np.linalg.norm(se3_log(np.linalg.inv(T_post) @ T_meas))
        delta_prior_to_meas = np.linalg.norm(se3_log(np.linalg.inv(T_prior) @ T_meas))
        assert delta_to_meas < delta_prior_to_meas

        # Covariance should shrink
        assert np.trace(cov_post) < np.trace(cov_prior)

    def test_update_with_large_R_is_noop(self):
        T_prior = np.eye(4)
        cov_prior = np.eye(6) * 0.01
        T_meas = se3_exp(np.array([1.0, 0, 0, 0, 0, 0]))
        R = np.eye(6) * 1e8  # essentially no information
        T_post, cov_post = ekf_update(T_prior, cov_prior, T_meas, R)

        np.testing.assert_array_almost_equal(T_post, T_prior, decimal=4)
        np.testing.assert_array_almost_equal(cov_post, cov_prior, decimal=4)

    def test_covariance_stays_positive_definite(self):
        T_prior = _random_pose(seed=1)
        cov_prior = np.eye(6) * 0.05
        for i in range(20):
            T_meas = se3_exp(np.random.randn(6) * 0.01)
            T_meas = T_prior @ T_meas
            R = np.eye(6) * 0.001
            T_prior, cov_prior = ekf_update(T_prior, cov_prior, T_meas, R)
            eigvals = np.linalg.eigvalsh(cov_prior)
            assert np.all(eigvals > -1e-6), f"Non-PSD at step {i}: {eigvals}"


class TestSharedErrorFusion:
    def test_world_frame_fusion_would_double_count_σwb(self):
        """Illustrates the trap: naive world-frame fusion when both sources
        share T_wb would shrink covariance too aggressively."""
        Sigma_wb = np.eye(6) * 0.02
        # Both the prior and the observation include Σ_wb
        cov_prior_world = Sigma_wb.copy()
        cov_meas_world = Sigma_wb.copy()

        T_prior = np.eye(4)
        T_meas = se3_exp(np.array([0.01, 0, 0, 0, 0, 0]))
        _, cov_post = ekf_update(T_prior, cov_prior_world,
                                  T_meas, cov_meas_world)
        # Naive fusion: cov_post ≈ Sigma_wb / 2 — but this is WRONG because
        # the two sources were not independent. We detect this by noting
        # the posterior is below Σ_wb, which cannot be correct.
        assert np.trace(cov_post) < np.trace(Sigma_wb)

    def test_base_frame_fusion_preserves_σwb_lower_bound(self):
        """Base-frame fusion then world projection: world cov >= Σ_wb."""
        Sigma_wb = np.eye(6) * 0.02
        T_wb = _random_pose(seed=2)

        # Base-frame prior and observation with small independent noise
        T_bo_prior = _random_pose(seed=3)
        cov_bo_prior = np.eye(6) * 1e-6  # kinematic precision
        T_bo_meas = T_bo_prior @ se3_exp(np.array([0.001, 0, 0, 0, 0, 0]))
        R_bo = np.eye(6) * 1e-4           # ICP-level noise, no Σ_wb

        T_bo_post, cov_bo_post, T_wo, cov_wo = ekf_update_base_frame(
            T_bo_prior, cov_bo_prior, T_bo_meas, R_bo, T_wb, Sigma_wb)

        # World-frame cov must be at least as large as Σ_wb (projected)
        eigvals_wo = np.linalg.eigvalsh(cov_wo)
        eigvals_wb = np.linalg.eigvalsh(Sigma_wb)
        assert eigvals_wo.min() >= 0.9 * eigvals_wb.min(), \
            f"World cov eigenvalues {eigvals_wo} vs Σ_wb eigenvalues {eigvals_wb}"


class TestObservationNoiseComposition:
    def test_zero_slam_uncertainty_passthrough(self):
        R_local = np.eye(6) * 0.01
        Sigma_wb = np.zeros((6, 6))
        R_eff = compose_observation_noise(R_local, Sigma_wb)
        np.testing.assert_array_almost_equal(R_eff, R_local)

    def test_slam_uncertainty_inflates_noise(self):
        R_local = np.eye(6) * 0.001
        Sigma_wb_small = np.eye(6) * 0.0001
        Sigma_wb_large = np.eye(6) * 0.1
        R_eff_small = compose_observation_noise(R_local, Sigma_wb_small)
        R_eff_large = compose_observation_noise(R_local, Sigma_wb_large)
        assert np.trace(R_eff_large) > np.trace(R_eff_small)
        assert np.trace(R_eff_small) > np.trace(R_local)

    def test_noise_remains_symmetric(self):
        R_local = np.eye(6) * 0.01
        Sigma_wb = np.random.RandomState(0).randn(6, 6) * 0.001
        Sigma_wb = Sigma_wb @ Sigma_wb.T  # make PSD
        R_eff = compose_observation_noise(R_local, Sigma_wb)
        np.testing.assert_array_almost_equal(R_eff, R_eff.T)


class TestEntropy:
    def test_entropy_monotonic_in_scale(self):
        h_small = pose_entropy(np.eye(6) * 0.001)
        h_large = pose_entropy(np.eye(6) * 0.1)
        assert h_large > h_small

    def test_pose_is_uncertain_threshold(self):
        tight = np.eye(6) * 1e-6
        loose = np.eye(6) * 1.0
        assert not pose_is_uncertain(tight)
        assert pose_is_uncertain(loose)


class TestProcessNoiseSchedule:
    def test_grasping_noise_huge(self):
        Q_idle = process_noise_for_phase("idle", is_target=False)
        Q_grasp = process_noise_for_phase("grasping", is_target=True)
        assert np.trace(Q_grasp) > 1000 * np.trace(Q_idle)

    def test_holding_base_frame_is_tight(self):
        Q_holding_base = process_noise_for_phase(
            "holding", is_target=True, frame="base")
        Q_holding_world = process_noise_for_phase(
            "holding", is_target=True, frame="world")
        assert np.trace(Q_holding_base) < np.trace(Q_holding_world)

    def test_long_static_objects_have_near_zero_noise(self):
        Q_long = process_noise_for_phase(
            "idle", is_target=False, frames_since_observation=200)
        Q_recent = process_noise_for_phase(
            "idle", is_target=False, frames_since_observation=0)
        assert np.trace(Q_long) < np.trace(Q_recent)


class TestLabelBelief:
    def test_beta_bernoulli_converges_to_high_score_label(self):
        belief = {}
        for _ in range(20):
            belief, _ = update_label_belief(belief, "cup", 0.1)
        for _ in range(5):
            belief, _ = update_label_belief(belief, "bowl", 0.9)
        # bowl: α ≈ 1 + 5*0.9 = 5.5, β ≈ 1 + 5*0.1 = 1.5 → 0.79
        # cup:  α ≈ 1 + 20*0.1 = 3, β ≈ 1 + 20*0.9 = 19 → 0.14
        _, map_label = update_label_belief(belief, "bowl", 0.9)
        assert map_label == "bowl"

    def test_low_confidence_sum_cant_beat_high_confidence(self):
        belief = {}
        for _ in range(3):
            belief, _ = update_label_belief(belief, "bowl", 0.8)
        for _ in range(10):
            belief, map_label = update_label_belief(belief, "cup", 0.1)
        assert map_label == "bowl"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

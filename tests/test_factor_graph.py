"""
Unit tests for pose_update/factor_graph.py (Tasks 4 + 6b).

Most tests are synthetic. Validates:
  * Single-object case behaves like the EKF
  * Scene graph relation pulls inconsistent poses into consistency
  * Manipulation factor anchors held object to EE
  * Camera pose is never optimized (fixed parameter)
  * Outlier observations get downweighted
  * SLAM uncertainty inflates observation factor noise

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python -m pytest tests/test_factor_graph.py -v
"""

import os
import sys

import numpy as np
import pytest

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from pose_update.factor_graph import (
    PoseGraphOptimizer, Observation, RelationEdge,
    relation_residual,
)
from pose_update.state.slam_interface import PoseEstimate
from pose_update.state.ekf_se3 import se3_log


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_pose(x, y, z, yaw=0.0):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def _tiny_cov():
    return np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])


def _slam_pose():
    return PoseEstimate(T=np.eye(4), cov=np.diag([1e-5] * 6))


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

class TestRelationResidual:
    def test_on_zero_when_stacked(self):
        T_parent = _make_pose(0, 0, 0.5)
        T_child = _make_pose(0, 0, 0.5 + 0.1)  # 0.5 apart → half (0.1+0.1)
        r = relation_residual(T_parent, T_child, "on",
                              parent_size=np.array([0.1, 0.1, 0.1]),
                              child_size=np.array([0.1, 0.1, 0.1]))
        assert r < 1e-6

    def test_on_positive_when_floating(self):
        T_parent = _make_pose(0, 0, 0.5)
        T_child = _make_pose(0, 0, 1.0)
        r = relation_residual(T_parent, T_child, "on",
                              parent_size=np.array([0.1, 0.1, 0.1]),
                              child_size=np.array([0.1, 0.1, 0.1]))
        assert r > 0.1

    def test_in_zero_when_centered(self):
        T_parent = _make_pose(1.0, 2.0, 0.5)
        T_child = _make_pose(1.0, 2.0, 0.5)
        r = relation_residual(T_parent, T_child, "in",
                              parent_size=np.array([0.2, 0.2, 0.2]),
                              child_size=np.array([0.05, 0.05, 0.05]))
        assert r == 0.0

    def test_in_positive_when_outside(self):
        T_parent = _make_pose(0, 0, 0)
        T_child = _make_pose(1.0, 0, 0)
        r = relation_residual(T_parent, T_child, "in",
                              parent_size=np.array([0.2, 0.2, 0.2]))
        assert r > 0.5


class TestSingleObjectOptimization:
    def test_single_object_with_prior_only(self):
        prior = PoseEstimate(T=_make_pose(1.0, 2.0, 0.3), cov=_tiny_cov())
        opt = PoseGraphOptimizer(verbose=False)
        result = opt.run(
            slam_pose=_slam_pose(),
            priors={0: prior},
            observations=[],
        )
        np.testing.assert_array_almost_equal(
            result.posteriors[0].T, prior.T, decimal=4)

    def test_single_object_with_consistent_observation(self):
        T_true = _make_pose(1.0, 2.0, 0.5)
        prior = PoseEstimate(T=T_true, cov=np.diag([1e-3] * 6))
        # Observation agrees with prior (camera is identity, so T_co = T_wo)
        obs = Observation(obj_id=0, T_co=T_true,
                          R_icp=np.diag([1e-5] * 6),
                          fitness=0.95, rmse=0.003)
        opt = PoseGraphOptimizer(verbose=False)
        result = opt.run(
            slam_pose=_slam_pose(),
            priors={0: prior},
            observations=[obs],
        )
        np.testing.assert_array_almost_equal(
            result.posteriors[0].T, T_true, decimal=3)


class TestFixedCamera:
    def test_camera_pose_not_in_values(self):
        """The camera pose must never become a variable."""
        prior = PoseEstimate(T=_make_pose(1.0, 0, 0), cov=_tiny_cov())
        slam = _slam_pose()
        opt = PoseGraphOptimizer(verbose=False)
        result = opt.run(
            slam_pose=slam,
            priors={0: prior},
            observations=[Observation(obj_id=0, T_co=_make_pose(1.0, 0, 0))],
        )
        # Only object pose(s) should be in posteriors; no camera key
        assert set(result.posteriors.keys()) == {0}


class TestRelationConstraints:
    def test_containment_pulls_inconsistent_poses(self):
        """Apple placed outside bowl should be pulled inside by 'in' factor."""
        T_bowl = _make_pose(1.0, 2.0, 0.1)   # bowl at table height
        T_apple_bad = _make_pose(1.3, 2.0, 0.1)  # 30cm off to the side
        priors = {
            0: PoseEstimate(T=T_bowl, cov=_tiny_cov()),     # bowl: confident
            1: PoseEstimate(T=T_apple_bad,                   # apple: uncertain
                            cov=np.diag([0.1] * 3 + [0.01] * 3)),
        }
        relation = RelationEdge(
            parent=0, child=1, relation_type="in", score=0.8,
            parent_size=np.array([0.2, 0.2, 0.2]),
            child_size=np.array([0.04, 0.04, 0.04]),
        )

        opt = PoseGraphOptimizer(
            relation_base_sigma=0.005,   # tight for a clear test
            verbose=False,
        )
        result = opt.run(
            slam_pose=_slam_pose(),
            priors=priors,
            observations=[],
            relations=[relation],
        )

        # Apple's optimized x should move closer to bowl's x (=1.0)
        apple_x_prior = T_apple_bad[0, 3]
        apple_x_post = result.posteriors[1].T[0, 3]
        bowl_x = T_bowl[0, 3]
        assert abs(apple_x_post - bowl_x) < abs(apple_x_prior - bowl_x), \
            f"Apple not pulled toward bowl: prior x={apple_x_prior}, " \
            f"post x={apple_x_post}, bowl x={bowl_x}"


class TestManipulationFactor:
    def test_held_object_snapped_to_ee(self):
        T_ew = _make_pose(2.0, 1.0, 0.8)
        T_oe = _make_pose(0.0, 0.0, -0.05)  # object 5cm below EE
        expected_T_wo = T_ew @ T_oe

        # Start held object far from where the EE says it should be
        wrong_prior = _make_pose(-1.0, -1.0, 0.0)
        priors = {
            5: PoseEstimate(T=wrong_prior,
                            cov=np.diag([0.5] * 3 + [0.5] * 3)),  # loose
        }

        opt = PoseGraphOptimizer(
            manip_noise_sigma=0.001,  # tighter than the loose prior
            verbose=False,
        )
        result = opt.run(
            slam_pose=_slam_pose(),
            priors=priors,
            observations=[],
            held_obj_id=5, T_ew=T_ew, T_oe=T_oe,
        )

        # The optimized pose should be very close to T_ew @ T_oe
        T_post = result.posteriors[5].T
        diff = np.linalg.norm(T_post[:3, 3] - expected_T_wo[:3, 3])
        assert diff < 0.01, \
            f"Held object not snapped: distance={diff}, " \
            f"expected={expected_T_wo[:3, 3]}, got={T_post[:3, 3]}"


class TestOutlierRobustness:
    def test_outlier_observation_downweighted(self):
        """An observation wildly disagreeing with the prior should be rejected."""
        T_true = _make_pose(1.0, 2.0, 0.5)
        prior = PoseEstimate(T=T_true, cov=np.diag([1e-3] * 6))  # confident prior

        # Outlier: object appears to be 5 meters away
        T_outlier = _make_pose(6.0, 2.0, 0.5)
        obs_outlier = Observation(obj_id=0, T_co=T_outlier,
                                   R_icp=np.diag([1e-4] * 6),
                                   fitness=0.7, rmse=0.02)

        opt = PoseGraphOptimizer(adaptive_c=0.1, verbose=False)
        result = opt.run(
            slam_pose=_slam_pose(),
            priors={0: prior},
            observations=[obs_outlier],
        )

        # With adaptive kernel downweighting, posterior should stay near prior
        post_x = result.posteriors[0].T[0, 3]
        assert abs(post_x - T_true[0, 3]) < 1.0, \
            f"Outlier dragged posterior too far: {post_x} vs truth {T_true[0, 3]}"


class TestSlamUncertaintyPropagation:
    def test_high_slam_uncertainty_weakens_observation(self):
        """When Σ_wb is huge, observation factors have inflated noise and
        contribute less — the posterior stays near the prior."""
        T_prior = _make_pose(1.0, 0, 0)
        T_obs = _make_pose(1.1, 0, 0)  # 10cm away

        obs = Observation(obj_id=0, T_co=T_obs,
                          R_icp=np.diag([1e-6] * 6),
                          fitness=0.95, rmse=0.001)
        prior = PoseEstimate(T=T_prior, cov=np.diag([1e-3] * 6))

        # Case A: low SLAM uncertainty
        slam_confident = PoseEstimate(T=np.eye(4), cov=np.diag([1e-8] * 6))
        # Case B: high SLAM uncertainty
        slam_uncertain = PoseEstimate(T=np.eye(4), cov=np.diag([1e-1] * 6))

        opt = PoseGraphOptimizer(verbose=False)
        res_confident = opt.run(slam_confident, {0: prior}, [obs])
        res_uncertain = opt.run(slam_uncertain, {0: prior}, [obs])

        # Confident SLAM → posterior pulled toward observation
        # Uncertain SLAM → posterior stays closer to prior
        delta_confident = abs(res_confident.posteriors[0].T[0, 3] - T_prior[0, 3])
        delta_uncertain = abs(res_uncertain.posteriors[0].T[0, 3] - T_prior[0, 3])
        assert delta_confident > delta_uncertain, \
            f"Confident SLAM δ={delta_confident}, " \
            f"uncertain SLAM δ={delta_uncertain} — expected > "


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

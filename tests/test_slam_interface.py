"""
Unit tests for the Layer 1 interface (Task 1).

Tests the pose/cov dataclass, movable-mask collection/application, and the
PassThroughSlam reference backend.

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python -m pytest tests/test_slam_interface.py -v
"""

import os
import sys

import numpy as np
import pytest

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from pose_update.state.slam_interface import (
    PoseEstimate,
    collect_movable_masks,
    mask_out_movable,
    PassThroughSlam,
)


class TestPoseEstimate:
    def test_construction(self):
        p = PoseEstimate(T=np.eye(4))
        assert p.T.shape == (4, 4)
        assert p.cov.shape == (6, 6)

    def test_rejects_wrong_shape(self):
        with pytest.raises(AssertionError):
            PoseEstimate(T=np.eye(3))

    def test_custom_cov(self):
        cov = np.eye(6) * 0.5
        p = PoseEstimate(T=np.eye(4), cov=cov)
        np.testing.assert_array_equal(p.cov, cov)


class TestMovableMask:
    def _make_det(self, mask):
        return {"mask": mask.astype(np.uint8)}

    def test_empty_detections_returns_false_mask(self):
        mask = collect_movable_masks([], (480, 640))
        assert mask.shape == (480, 640)
        assert not mask.any()

    def test_single_detection_covers_its_region(self):
        m = np.zeros((480, 640), dtype=np.uint8)
        m[100:200, 200:300] = 1
        union = collect_movable_masks([self._make_det(m)], (480, 640),
                                      dilate_px=0)
        assert union[150, 250]
        assert not union[0, 0]

    def test_union_of_two_masks(self):
        m1 = np.zeros((480, 640), dtype=np.uint8)
        m1[0:100, 0:100] = 1
        m2 = np.zeros((480, 640), dtype=np.uint8)
        m2[200:300, 400:500] = 1
        union = collect_movable_masks(
            [self._make_det(m1), self._make_det(m2)],
            (480, 640), dilate_px=0)
        assert union[50, 50] and union[250, 450]
        assert not union[150, 200]

    def test_dilation_grows_mask(self):
        m = np.zeros((100, 100), dtype=np.uint8)
        m[50, 50] = 1
        no_dilate = collect_movable_masks([self._make_det(m)], (100, 100),
                                           dilate_px=0)
        dilated = collect_movable_masks([self._make_det(m)], (100, 100),
                                         dilate_px=3)
        assert no_dilate.sum() < dilated.sum()

    def test_mask_out_zeroes_correct_pixels(self):
        depth = np.ones((100, 100), dtype=np.float32) * 1.5
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 30:40] = True
        masked = mask_out_movable(depth, mask)
        assert (masked[10:20, 30:40] == 0).all()
        assert (masked[50:60, 50:60] == 1.5).all()

    def test_mask_out_does_not_modify_input(self):
        depth = np.ones((10, 10), dtype=np.float32) * 0.7
        mask = np.ones((10, 10), dtype=bool)
        _ = mask_out_movable(depth, mask)
        assert (depth == 0.7).all()


class TestPassThroughSlam:
    def test_returns_prescribed_poses_in_order(self):
        poses = [np.eye(4), np.eye(4) * 1.0]
        poses[1][0, 3] = 2.0  # translated in x
        slam = PassThroughSlam(poses)

        fake_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        fake_depth = np.zeros((2, 2), dtype=np.float32)

        p0 = slam.step(fake_rgb, fake_depth)
        p1 = slam.step(fake_rgb, fake_depth)

        np.testing.assert_array_equal(p0.T, poses[0])
        np.testing.assert_array_equal(p1.T, poses[1])

    def test_default_covariance_applied_consistently(self):
        poses = [np.eye(4)] * 3
        slam = PassThroughSlam(poses)
        covs = [slam.step(None, None).cov for _ in range(3)]
        for c in covs[1:]:
            np.testing.assert_array_equal(c, covs[0])

    def test_per_frame_covariance(self):
        poses = [np.eye(4)] * 2
        pfcov = [np.eye(6) * 0.1, np.eye(6) * 0.001]
        slam = PassThroughSlam(poses, per_frame_cov=pfcov)

        p0 = slam.step(None, None)
        p1 = slam.step(None, None)
        np.testing.assert_array_equal(p0.cov, pfcov[0])
        np.testing.assert_array_equal(p1.cov, pfcov[1])

    def test_reset_allows_replay(self):
        poses = [np.eye(4)]
        poses[0][0, 3] = 7.0
        slam = PassThroughSlam(poses)
        p0 = slam.step(None, None)
        slam.reset()
        p0b = slam.step(None, None)
        np.testing.assert_array_equal(p0.T, p0b.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

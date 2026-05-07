"""Unit tests for the depth ray-traced visibility predicate."""

from __future__ import annotations

import numpy as np
import pytest

from perception.visibility import visibility_p_v, _fibonacci_sphere


# Fetch-like intrinsics (640 x 480 head camera).
K = np.array([
    [554.3827, 0.0, 320.5],
    [0.0, 554.3827, 240.5],
    [0.0, 0.0,     1.0],
], dtype=np.float64)
H, W = 480, 640
IMG = (H, W)


def _T_co(xyz=(0.0, 0.0, 1.0)):
    """Build a camera-frame identity-rotation pose at translation `xyz`."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def _flat_depth(val: float) -> np.ndarray:
    """Full image at a single depth value."""
    return np.full((H, W), val, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# 1. Core frustum / occlusion semantics
# ─────────────────────────────────────────────────────────────────────

class TestCoreSemantics:
    def test_fully_visible_when_depth_matches_track(self):
        """Depth = track z + noise → p_v ≈ 1."""
        depth = _flat_depth(1.0)
        tracks = [{"oid": 7, "T_co": _T_co((0.0, 0.0, 1.0)),
                    "obj_radius": 0.05}]
        pv = visibility_p_v(tracks, K, depth, IMG)
        assert pv[7] == pytest.approx(1.0, abs=1e-9)

    def test_fully_occluded_when_depth_much_closer(self):
        """A flat depth of 0.3 m in front of a track at 1 m → p_v = 0."""
        depth = _flat_depth(0.3)
        tracks = [{"oid": 7, "T_co": _T_co((0.0, 0.0, 1.0)),
                    "obj_radius": 0.05}]
        pv = visibility_p_v(tracks, K, depth, IMG)
        assert pv[7] == 0.0

    def test_visible_when_depth_much_farther(self):
        """Depth farther than track → nothing closer to block it → p_v = 1."""
        depth = _flat_depth(3.0)
        tracks = [{"oid": 7, "T_co": _T_co((0.0, 0.0, 1.0)),
                    "obj_radius": 0.05}]
        pv = visibility_p_v(tracks, K, depth, IMG)
        assert pv[7] == pytest.approx(1.0, abs=1e-9)

    def test_out_of_frustum_gives_zero(self):
        """Track at x = +5 m is outside the image; p_v = 0."""
        depth = _flat_depth(1.0)
        tracks = [{"oid": 7, "T_co": _T_co((5.0, 0.0, 1.0)),
                    "obj_radius": 0.05}]
        pv = visibility_p_v(tracks, K, depth, IMG)
        assert pv[7] == 0.0

    def test_behind_camera_gives_zero(self):
        """Track with z < 0 is behind the camera → p_v = 0."""
        depth = _flat_depth(1.0)
        tracks = [{"oid": 7, "T_co": _T_co((0.0, 0.0, -0.5)),
                    "obj_radius": 0.05}]
        pv = visibility_p_v(tracks, K, depth, IMG)
        assert pv[7] == 0.0


# ─────────────────────────────────────────────────────────────────────
# 2. Invalid depth handling
# ─────────────────────────────────────────────────────────────────────

class TestInvalidDepth:
    def test_all_zeros_conservative_one(self):
        """Zero depth = no sensor data → p_v = 1 (no evidence of occlusion)."""
        depth = _flat_depth(0.0)
        tracks = [{"oid": 7, "T_co": _T_co((0.0, 0.0, 1.0)),
                    "obj_radius": 0.05}]
        pv = visibility_p_v(tracks, K, depth, IMG)
        assert pv[7] == pytest.approx(1.0, abs=1e-9)

    def test_nan_depth_conservative_one(self):
        depth = np.full((H, W), np.nan, dtype=np.float32)
        tracks = [{"oid": 7, "T_co": _T_co((0.0, 0.0, 1.0)),
                    "obj_radius": 0.05}]
        pv = visibility_p_v(tracks, K, depth, IMG)
        assert pv[7] == pytest.approx(1.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────
# 3. Partial occlusion — realistic scenario
# ─────────────────────────────────────────────────────────────────────

class TestPartialOcclusion:
    def test_half_image_occluded_gives_roughly_half(self):
        """Left half of image at 0.3 m (occluding), right half at 3 m
        (free). A sphere track centered at image centre should get
        p_v ≈ 0.5 (give or take tolerance effects near the boundary).
        """
        depth = np.full((H, W), 3.0, dtype=np.float32)
        depth[:, :W // 2] = 0.3     # left half is occluding
        tracks = [{"oid": 7, "T_co": _T_co((0.0, 0.0, 1.0)),
                    "obj_radius": 0.05}]
        pv = visibility_p_v(tracks, K, depth, IMG)
        # Expect roughly 0.5. The Fibonacci sphere has slight bias
        # depending on how many samples land left vs right, so allow
        # a 0.15-wide window.
        assert 0.35 < pv[7] < 0.65


# ─────────────────────────────────────────────────────────────────────
# 4. Ref-cloud vs fallback sphere
# ─────────────────────────────────────────────────────────────────────

class TestRefCloudSampling:
    def test_ref_cloud_overrides_fallback(self):
        """When ref_points_obj is provided, it's used for sampling."""
        depth = _flat_depth(1.0)
        # A "shape" — three points clustered at the object center.
        ref_points = np.array([[0, 0, 0],
                                 [0.01, 0, 0],
                                 [-0.01, 0, 0]], dtype=np.float64)
        tracks = [{
            "oid": 7,
            "T_co": _T_co((0.0, 0.0, 1.0)),
            "ref_points_obj": ref_points,
            "obj_radius": 0.02,
        }]
        pv = visibility_p_v(tracks, K, depth, IMG)
        # All 3 points project into the image, all depths match → p_v = 1.
        assert pv[7] == pytest.approx(1.0, abs=1e-9)

    def test_empty_ref_points_falls_back_to_sphere(self):
        """Empty ref cloud triggers sphere fallback; doesn't crash."""
        depth = _flat_depth(1.0)
        tracks = [{
            "oid": 7,
            "T_co": _T_co((0.0, 0.0, 1.0)),
            "ref_points_obj": np.zeros((0, 3)),
            "obj_radius": 0.05,
        }]
        pv = visibility_p_v(tracks, K, depth, IMG)
        assert pv[7] == pytest.approx(1.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────
# 5. Batched / multi-track vectorisation
# ─────────────────────────────────────────────────────────────────────

class TestBatch:
    def test_independent_tracks(self):
        """Mix of visible, occluded, and out-of-FOV tracks in one call."""
        depth = np.full((H, W), 2.0, dtype=np.float32)
        tracks = [
            {"oid": 1, "T_co": _T_co((0.0, 0.0, 1.0)), "obj_radius": 0.05},   # visible
            {"oid": 2, "T_co": _T_co((0.0, 0.0, 3.0)), "obj_radius": 0.05},   # behind depth
            {"oid": 3, "T_co": _T_co((10.0, 0.0, 1.0)), "obj_radius": 0.05},  # OOV
        ]
        pv = visibility_p_v(tracks, K, depth, IMG)
        assert pv[1] == pytest.approx(1.0, abs=1e-9)
        assert pv[2] == 0.0
        assert pv[3] == 0.0

    def test_empty_input_returns_empty_dict(self):
        assert visibility_p_v([], K, _flat_depth(1.0), IMG) == {}


# ─────────────────────────────────────────────────────────────────────
# 6. Determinism
# ─────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_two_runs_exact_match(self):
        """Fibonacci sampling is deterministic → same result twice."""
        depth = _flat_depth(1.0)
        tracks = [{"oid": 7, "T_co": _T_co((0.0, 0.0, 1.0)),
                    "obj_radius": 0.05}]
        pv1 = visibility_p_v(tracks, K, depth, IMG)
        pv2 = visibility_p_v(tracks, K, depth, IMG)
        assert pv1 == pv2


# ─────────────────────────────────────────────────────────────────────
# 7. Sphere sampler sanity
# ─────────────────────────────────────────────────────────────────────

class TestFibonacciSphere:
    def test_n_points_on_sphere_of_correct_radius(self):
        pts = _fibonacci_sphere(100, radius=0.1)
        assert pts.shape == (100, 3)
        radii = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(radii, 0.1, atol=1e-9)

    def test_n_equals_1_returns_single_point(self):
        pts = _fibonacci_sphere(1, radius=0.1)
        assert pts.shape == (1, 3)

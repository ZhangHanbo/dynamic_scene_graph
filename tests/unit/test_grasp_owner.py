"""Unit tests for ``pose_update.grasp_owner_detector.GraspOwnerDetector``.

Pins:
- Fix 1: Tier 2 (geometric containment) is invariant under T_wb
  perturbations because the cam→gripper transform now skips the
  world frame and uses ``inv(T_bg) @ T_bc`` directly.
- Fix 2: Tier 3 (nearest-track fallback) uses NEAREST POINT of the
  per-track surface cloud (not centroid) and respects the new 5 cm
  default radius. Falls back to centroid distance when a track has
  no stored ref cloud.
"""
from __future__ import annotations

import numpy as np
import pytest

from pose_update.grasp_owner_detector import (
    GraspOwnerDetector, HeldDecision, TrackerState,
)


class _StubTrackerState(TrackerState):
    """Minimal TrackerState driving Tier 3 tests."""

    def __init__(self, *,
                 centroids: dict | None = None,
                 pointclouds: dict | None = None):
        self._centroids = dict(centroids or {})
        self._pointclouds = dict(pointclouds or {})

    def sam2_tau(self):
        return {}

    def iter_world_centroids(self):
        for oid, mu_w in self._centroids.items():
            yield int(oid), np.asarray(mu_w)

    def iter_world_pointclouds(self):
        for oid, pts in self._pointclouds.items():
            yield int(oid), np.asarray(pts)

    def force_admit(self, det, depth):
        return None


def _make_det():
    """A GraspOwnerDetector wired up enough for Tier 3 tests
    (Tier 1/2 short-circuited because we pass empty inputs)."""
    class _DummyGripper:
        robot_name = "test"
        link_name = "gripper_link"
        slide_axis = "y"
        approach_axis = "x"
        height_axis = "z"
        def state_from_joints(self, joints):
            return None
        def inside_volume_g(self, state):
            class _Box:
                def count_inside(self, pts): return 0
                def corners(self): return np.zeros((8, 3))
            return _Box()
    return GraspOwnerDetector(gripper=_DummyGripper(),
                               fallback_radius_m=0.05)


# ─── Tier 3: nearest-point + 5 cm ──────────────────────────────────

def test_tier3_nearest_point_within_5cm():
    """Track has a surface point 4 cm from the EE → match."""
    ee_world = np.array([0.5, 0.0, 0.5])
    # Apple cloud centred at (0.55, 0.0, 0.5), radius ~3 cm.
    apple_pts = np.array([[0.5 + 0.04, 0.0, 0.5],     # 4 cm from EE
                           [0.5 + 0.06, 0.0, 0.5],     # 6 cm
                           [0.5 + 0.08, 0.0, 0.5]])    # 8 cm
    ts = _StubTrackerState(
        centroids={5: [0.55, 0.0, 0.5]},
        pointclouds={5: apple_pts},
    )
    det = _make_det()
    T_wb = np.eye(4)
    T_bg = np.eye(4); T_bg[:3, 3] = ee_world
    oid = det._nearest_live_track(ts, T_wb, T_bg)
    assert oid == 5


def test_tier3_centroid_within_radius_but_nearest_point_far():
    """Track centroid is 4 cm but nearest surface point is 7 cm —
    rejected by the 5 cm radius (we use nearest point)."""
    ee_world = np.array([0.0, 0.0, 0.0])
    # Centroid at (0, 0, 0.04) but cloud points all at radius 0.07.
    pts = np.array([[0.0, 0.0, 0.07],
                     [0.0, 0.0, 0.08],
                     [0.0, 0.0, 0.09]])
    ts = _StubTrackerState(
        centroids={3: [0.0, 0.0, 0.04]},
        pointclouds={3: pts},
    )
    det = _make_det()
    T_wb = np.eye(4)
    T_bg = np.eye(4)  # T_wg = identity → ee at origin
    oid = det._nearest_live_track(ts, T_wb, T_bg)
    assert oid is None


def test_tier3_falls_back_to_centroid_when_no_cloud():
    """A track without a stored point cloud still gets considered
    via centroid distance."""
    ee_world = np.array([0.0, 0.0, 0.0])
    ts = _StubTrackerState(
        centroids={7: [0.0, 0.0, 0.04]},   # 4 cm centroid
        pointclouds={},                      # no cloud at all
    )
    det = _make_det()
    T_wb = np.eye(4)
    T_bg = np.eye(4)
    oid = det._nearest_live_track(ts, T_wb, T_bg)
    assert oid == 7


def test_tier3_picks_closer_point_among_two_tracks():
    """Two tracks with surface points at 3 cm and 6 cm; closer wins."""
    ts = _StubTrackerState(
        centroids={1: [0.04, 0, 0], 2: [0.07, 0, 0]},
        pointclouds={
            1: np.array([[0.06, 0, 0]]),     # nearest 6 cm — outside 5 cm
            2: np.array([[0.03, 0, 0]]),     # nearest 3 cm
        },
    )
    det = _make_det()
    T_wb = np.eye(4); T_bg = np.eye(4)
    oid = det._nearest_live_track(ts, T_wb, T_bg)
    assert oid == 2


def test_tier3_default_radius_is_5cm():
    """Sanity: default radius dropped from 30 cm to 5 cm."""
    class _G:
        robot_name="t"; link_name="l"; slide_axis="y"; approach_axis="x"; height_axis="z"
        def state_from_joints(self, j): return None
        def inside_volume_g(self, s):
            class _B:
                def count_inside(self, p): return 0
            return _B()
    d = GraspOwnerDetector(gripper=_G())
    assert d.fallback_radius_m == 0.05


# ─── Fix 1: Tier 2 cam→gripper invariance under T_wb perturbation ──

def test_tier2_invariant_under_T_wb():
    """The geometric pick now uses inv(T_bg) @ T_bc directly, so
    perturbing T_wb (the SLAM pose) must NOT change the count of
    points inside the gripper volume.
    """
    # Build a real GripperGeometry — use Fetch since it ships with
    # the repo and the URDF lookup is deterministic.
    try:
        from pose_update.robot_models import create_gripper_geometry
    except Exception:
        pytest.skip("Fetch gripper geometry not available")
    gripper = create_gripper_geometry("fetch")
    state = gripper.state_from_joints({"l_gripper_finger_joint": 0.025,
                                         "r_gripper_finger_joint": 0.025})
    if state is None:
        pytest.skip("gripper.state_from_joints returned None")
    # Build a 4x4 EE pose in base frame and a base→camera extrinsic.
    T_bg = np.eye(4); T_bg[:3, 3] = [0.5, 0.1, 0.4]
    T_bc = np.eye(4); T_bc[:3, 3] = [0.05, 0.02, 1.0]
    T_bc[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                              dtype=np.float64)

    # A small bag of points inside the gripper volume in CAMERA frame.
    # We place them inside the inside_volume_g(state) AABB in gripper
    # frame, then back-transform through T_bc^-1 @ T_bg to camera.
    box = gripper.inside_volume_g(state)
    centre_g = box.center
    # Cluster 5 points around the centre.
    pts_g = centre_g + 0.001 * np.random.default_rng(0).standard_normal((5, 3))
    pts_g_h = np.hstack([pts_g, np.ones((5, 1))])
    T_cb = np.linalg.inv(T_bc)
    T_bb_g = T_bg
    pts_cam = (T_cb @ T_bb_g @ pts_g_h.T).T[:, :3]

    # Synthesise a depth image + mask that yields these camera points.
    H, W = 480, 640
    fx, fy, cx, cy = 554.38, 554.38, 320.5, 240.5
    depth = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    for p in pts_cam:
        z = float(p[2])
        if z <= 0:
            continue
        u = int(round(fx * p[0] / z + cx))
        v = int(round(fy * p[1] / z + cy))
        if 0 <= u < W and 0 <= v < H:
            depth[v, u] = z
            mask[v, u] = True
    if mask.sum() < 3:
        pytest.skip("synthesised mask too small for the test")

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    det_in = [{"id": 42, "label": "apple",
                "mask": mask}]

    detector = GraspOwnerDetector(gripper=gripper, min_inside_count=1)

    # Two arbitrary T_wb's — they should not affect the count_inside
    # because the cam→gripper transform doesn't depend on T_wb.
    T_wb_a = np.eye(4); T_wb_a[:3, 3] = [10.0, 2.0, 0.0]
    T_wb_b = np.eye(4); T_wb_b[:3, 3] = [-3.0, -7.0, 0.0]
    T_wb_b[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
                                dtype=np.float64)

    chosen_a, count_a = detector._geometric_pick(
        det_in, depth, K, T_bg, T_bc, state)
    chosen_b, count_b = detector._geometric_pick(
        det_in, depth, K, T_bg, T_bc, state)
    # Same call signature both times (T_bg, T_bc constant) — sanity that
    # geometry is deterministic.
    assert count_a == count_b
    # Now actually run select() with two different T_wb values; the
    # tier-2 chosen pid must be identical because geometric_pick no
    # longer reads T_wb.
    ts = _StubTrackerState()
    decision_a = detector.select(
        detections=det_in, depth=depth, K=K,
        T_wb=T_wb_a, T_bg=T_bg, T_bc=T_bc,
        joints={"l_gripper_finger_joint": 0.025,
                 "r_gripper_finger_joint": 0.025},
        tracker_state=ts)
    decision_b = detector.select(
        detections=det_in, depth=depth, K=K,
        T_wb=T_wb_b, T_bg=T_bg, T_bc=T_bc,
        joints={"l_gripper_finger_joint": 0.025,
                 "r_gripper_finger_joint": 0.025},
        tracker_state=ts)
    assert decision_a.held_pid == decision_b.held_pid
    assert decision_a.source == decision_b.source

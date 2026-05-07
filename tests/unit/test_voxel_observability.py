"""Unit tests for `pose_update/voxel_observability.py`.

Synthetic scenes: a single horizontal occupier plane, with a camera
looking down at it; verifies the 3-state classification, hysteresis,
in-bounds checks, and the raycast_down column-state branches.
"""
from __future__ import annotations

import numpy as np
import pytest

from perception.voxel_observability import (
    EMPTY,
    OCCUPIED,
    UNSEEN,
    RaycastDownResult,
    VoxelObservability,
)


def _make_grid():
    return VoxelObservability(
        voxel_size_m=0.05,
        workspace_aabb=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
        n_min_hit=2,
        n_min_pass=3,
    )


def _flat_floor_depth_frame(K, T_cw, plane_z, h=64, w=64):
    """Generate a synthetic depth frame of a horizontal plane at world z=plane_z,
    seen from camera at T_cw with intrinsics K.

    Each pixel back-projects through z=z_cam to the world plane; we solve for
    the depth that lands on the world z=plane_z plane along that ray.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    # camera-frame ray direction: (x/z, y/z, 1)
    rx_cam = (us - cx) / fx
    ry_cam = (vs - cy) / fy
    rz_cam = np.ones_like(rx_cam)
    dirs_cam = np.stack([rx_cam, ry_cam, rz_cam], axis=-1)  # (h, w, 3)
    # transform to world (rotate; translation is camera origin offset)
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    dirs_world = dirs_cam @ R.T  # (h, w, 3)
    # find s along (origin + s * dir) that lands at world_z = plane_z
    s = (plane_z - t[2]) / dirs_world[..., 2]
    # camera-frame depth = s * |dir_z_cam| = s (since rz_cam = 1)
    depth = s.astype(np.float32)
    depth[depth < 0] = 0.0
    return depth


# ---------------------------------------------------------------------
# Construction / basic state
# ---------------------------------------------------------------------


class TestConstruction:
    def test_default_shape_matches_aabb(self):
        g = VoxelObservability(
            voxel_size_m=0.1,
            workspace_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 0.5)),
        )
        assert g.shape == (10, 10, 5)

    def test_invalid_voxel_size_raises(self):
        with pytest.raises(ValueError):
            VoxelObservability(voxel_size_m=0.0)

    def test_invalid_aabb_raises(self):
        with pytest.raises(ValueError):
            VoxelObservability(workspace_aabb=((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)))

    def test_initial_state_is_all_unseen(self):
        g = _make_grid()
        assert g.state_at([0.0, 0.0, 1.0]) == UNSEEN

    def test_state_at_out_of_bounds_is_unseen(self):
        g = _make_grid()
        assert g.state_at([10.0, 10.0, 10.0]) == UNSEEN
        assert g.state_at([-10.0, 0.0, 0.0]) == UNSEEN


# ---------------------------------------------------------------------
# Hysteresis behaviour
# ---------------------------------------------------------------------


class TestHysteresis:
    def test_single_hit_below_threshold_stays_unseen(self):
        # Direct counter manipulation to test hysteresis purely.
        g = _make_grid()
        g._n_hit[10, 10, 10] = 1  # one observation only
        assert g.state_at(g.aabb_min + np.array([10.5, 10.5, 10.5]) * g.voxel_size) == UNSEEN

    def test_two_hits_promote_to_occupied(self):
        g = _make_grid()
        g._n_hit[10, 10, 10] = 2
        assert g.state_at(g.aabb_min + np.array([10.5, 10.5, 10.5]) * g.voxel_size) == OCCUPIED

    def test_pass_count_below_threshold_stays_unseen(self):
        g = _make_grid()
        g._n_pass[5, 5, 5] = 2
        assert g.state_at(g.aabb_min + np.array([5.5, 5.5, 5.5]) * g.voxel_size) == UNSEEN

    def test_pass_count_promotes_to_empty(self):
        g = _make_grid()
        g._n_pass[5, 5, 5] = 3
        assert g.state_at(g.aabb_min + np.array([5.5, 5.5, 5.5]) * g.voxel_size) == EMPTY

    def test_occupied_takes_priority_over_empty(self):
        g = _make_grid()
        g._n_hit[5, 5, 5] = 5
        g._n_pass[5, 5, 5] = 100
        assert g.state_at(g.aabb_min + np.array([5.5, 5.5, 5.5]) * g.voxel_size) == OCCUPIED


# ---------------------------------------------------------------------
# integrate_depth: a downward-looking camera over a flat floor
# ---------------------------------------------------------------------


class TestIntegrateDepth:
    def _setup_overhead(self, plane_z=0.4):
        # Camera 1.5m above the plane, looking straight down (camera +z = world -z).
        # T_cw maps camera frame -> world frame.
        T_cw = np.eye(4)
        T_cw[:3, :3] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],   # flip y
            [0.0, 0.0, -1.0],   # camera +z looks down (world -z)
        ])
        T_cw[:3, 3] = np.array([0.0, 0.0, 1.5])
        K = np.array([[64.0, 0.0, 32.0],
                      [0.0, 64.0, 32.0],
                      [0.0, 0.0, 1.0]])
        depth = _flat_floor_depth_frame(K, T_cw, plane_z=plane_z)
        return K, T_cw, depth, plane_z

    def test_voxel_at_plane_becomes_occupied_after_two_frames(self):
        K, T_cw, depth, plane_z = self._setup_overhead(plane_z=0.4)
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
            n_min_hit=2,
            n_min_pass=3,
        )
        g.integrate_depth(depth, K, T_cw, max_range_m=3.0, subsample=1)
        g.integrate_depth(depth, K, T_cw, max_range_m=3.0, subsample=1)
        # voxel at the plane center
        assert g.state_at([0.0, 0.0, plane_z]) == OCCUPIED

    def test_voxel_above_plane_becomes_empty(self):
        K, T_cw, depth, plane_z = self._setup_overhead(plane_z=0.4)
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
            n_min_hit=2,
            n_min_pass=3,
        )
        # 3 frames so n_pass on a free-space voxel reaches threshold
        for _ in range(3):
            g.integrate_depth(depth, K, T_cw, max_range_m=3.0, subsample=1)
        # a voxel halfway between camera (z=1.5) and plane (z=0.4) at column center
        assert g.state_at([0.0, 0.0, 1.0]) == EMPTY

    def test_voxel_below_plane_remains_unseen(self):
        K, T_cw, depth, plane_z = self._setup_overhead(plane_z=0.4)
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
            n_min_hit=2,
            n_min_pass=3,
        )
        for _ in range(5):
            g.integrate_depth(depth, K, T_cw, max_range_m=3.0, subsample=1)
        # below the plane: ray was absorbed at the plane, never observed below
        assert g.state_at([0.0, 0.0, 0.1]) == UNSEEN


# ---------------------------------------------------------------------
# raycast_down: column-state classification
# ---------------------------------------------------------------------


class TestRaycastDown:
    def _grid_with_surface_at(self, surface_z, *, fill_above_empty=True):
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, -0.5), (1.0, 1.0, 2.0)),
            n_min_hit=2,
            n_min_pass=3,
        )
        # Mark a single (0, 0, surface_z) voxel as OCCUPIED.
        ijk = np.array(g.world_to_voxel([0.0, 0.0, surface_z]))
        g._n_hit[ijk[0], ijk[1], ijk[2]] = g.n_min_hit
        if fill_above_empty:
            # mark voxels above as EMPTY
            for s in range(1, 30):
                z = surface_z + s * g.voxel_size
                if z >= g.aabb_max[2]:
                    break
                jjk = np.array(g.world_to_voxel([0.0, 0.0, z]))
                g._n_pass[jjk[0], jjk[1], jjk[2]] = g.n_min_pass
        return g

    def test_hit_occupied_clean_column(self):
        g = self._grid_with_surface_at(0.4, fill_above_empty=True)
        res = g.raycast_down([0.0, 0.0, 1.0], max_distance_m=1.5, floor_z=-0.4)
        assert res.column_state == "hit_occupied"
        assert res.surface_z is not None
        assert abs(res.surface_z - 0.4) < g.voxel_size + 1e-6
        assert res.first_unseen_z is None

    def test_mixed_unseen_when_unseen_above_surface(self):
        g = self._grid_with_surface_at(0.4, fill_above_empty=False)
        # mark only the immediately-adjacent voxel above as EMPTY,
        # leaving a band of UNSEEN voxels higher up.
        ijk = np.array(g.world_to_voxel([0.0, 0.0, 0.4 + g.voxel_size]))
        g._n_pass[ijk[0], ijk[1], ijk[2]] = g.n_min_pass
        res = g.raycast_down([0.0, 0.0, 1.0], max_distance_m=1.5, floor_z=-0.4)
        assert res.column_state == "mixed_unseen"
        assert res.surface_z is not None
        assert res.first_unseen_z is not None
        assert res.first_unseen_z > res.surface_z

    def test_all_unseen_when_no_surface(self):
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, -0.5), (1.0, 1.0, 2.0)),
        )
        res = g.raycast_down([0.0, 0.0, 1.0], max_distance_m=1.5, floor_z=-0.4)
        assert res.column_state == "all_unseen"
        assert res.surface_z is None
        assert res.first_unseen_z is not None

    def test_all_empty_when_void(self):
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, -0.5), (1.0, 1.0, 2.0)),
            n_min_hit=2,
            n_min_pass=3,
        )
        # Mark every voxel along the column EMPTY using integer voxel
        # indexing so floating-point drift on the loop endpoint cannot
        # leave a hole at the bottom.
        i, j, k_start = g.world_to_voxel([0.0, 0.0, 1.0])
        _, _, k_floor = g.world_to_voxel([0.0, 0.0, -0.4])
        for k in range(k_floor, k_start + 1):
            g._n_pass[i, j, k] = g.n_min_pass
        res = g.raycast_down([0.0, 0.0, 1.0], max_distance_m=1.5, floor_z=-0.4)
        assert res.column_state == "all_empty"
        assert res.surface_z is None
        assert res.first_unseen_z is None

    def test_live_object_voxels_treated_as_occupied(self):
        # Pre-fill voxels above the live object with EMPTY so the
        # column above it is observed-empty, then inject a "live object"
        # sphere at (0, 0, 0.5).
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, -0.5), (1.0, 1.0, 2.0)),
        )
        i, j, k_start = g.world_to_voxel([0.0, 0.0, 1.0])
        _, _, k_obj = g.world_to_voxel([0.0, 0.0, 0.5])
        for k in range(k_obj + 1, k_start + 1):
            g._n_pass[i, j, k] = g.n_min_pass
        res = g.raycast_down(
            [0.0, 0.0, 1.0], max_distance_m=1.5, floor_z=-0.4,
            live_object_voxels=[(0.0, 0.0, 0.5, 0.05)],
        )
        assert res.column_state == "hit_occupied"
        assert res.surface_z is not None
        assert abs(res.surface_z - 0.5) < 0.1

    def test_live_object_voxels_with_unseen_above_yields_mixed_unseen(self):
        # When the grid above the live object hasn't been observed,
        # the column-state correctly reports 'mixed_unseen'.
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, -0.5), (1.0, 1.0, 2.0)),
        )
        res = g.raycast_down(
            [0.0, 0.0, 1.0], max_distance_m=1.5, floor_z=-0.4,
            live_object_voxels=[(0.0, 0.0, 0.5, 0.05)],
        )
        assert res.column_state == "mixed_unseen"
        assert res.surface_z is not None
        assert res.first_unseen_z is not None
        assert res.first_unseen_z > res.surface_z


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------


class TestStats:
    def test_stats_counts_known_promotions(self):
        g = _make_grid()
        g._n_hit[1, 1, 1] = 5
        g._n_pass[2, 2, 2] = 5
        s = g.stats()
        assert s["n_occupied"] == 1
        assert s["n_empty"] == 1
        assert s["n_unseen"] == s["n_total"] - 2
        assert s["voxel_size"] == 0.05
        assert s["bytes_used"] > 0

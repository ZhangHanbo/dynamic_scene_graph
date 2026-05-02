"""Unit tests for `pose_update/gravity_predict.py`."""
from __future__ import annotations

import numpy as np
import pytest

from pose_update.manipulation.gravity_predict import (
    EPS_ROUGHNESS_DEFAULT,
    predict_landing_pose,
)
from pose_update.manipulation.object_dynamics import (
    DEFAULT_DYNAMICS,
    ObjectDynamicsProperty,
    lookup_dynamics,
)
from pose_update.perception.voxel_observability import VoxelObservability


def _grid_with_flat_surface_at(surface_z, *, voxel_size=0.05,
                                aabb=((-1.0, -1.0, -0.5), (1.0, 1.0, 2.0))):
    """Build a voxel grid with a 0.5 m × 0.5 m horizontal slab of OCCUPIED
    voxels at world z=surface_z, and EMPTY voxels in the column above
    that slab (so 'hit_occupied' classification is clean)."""
    g = VoxelObservability(
        voxel_size_m=voxel_size, workspace_aabb=aabb,
        n_min_hit=2, n_min_pass=3,
    )
    # Slab voxels (xy ranged ±0.25 m around origin; one voxel thick).
    extent = 0.25
    n = int(np.ceil(extent / voxel_size))
    i_c, j_c, k_s = g.world_to_voxel([0.0, 0.0, surface_z])
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            ii, jj = i_c + di, j_c + dj
            if 0 <= ii < g.shape[0] and 0 <= jj < g.shape[1]:
                g._n_hit[ii, jj, k_s] = g.n_min_hit
    # Column above the slab: EMPTY.
    _, _, k_release = g.world_to_voxel([0.0, 0.0, surface_z + 1.0])
    for k in range(k_s + 1, k_release + 1):
        g._n_pass[i_c, j_c, k] = g.n_min_pass
    return g


def _T_at(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def _P_init(diag=1e-4):
    return np.eye(6) * diag


# ---------------------------------------------------------------------
# Skip path
# ---------------------------------------------------------------------


class TestSkipPath:
    def test_no_voxel_obs_returns_identity(self):
        T = _T_at(0, 0, 1.0)
        P = _P_init()
        T_land, P_land, info = predict_landing_pose(
            T, P, voxel_obs=None, dyn=DEFAULT_DYNAMICS)
        np.testing.assert_array_equal(T_land, T)
        np.testing.assert_array_equal(P_land, P)
        assert info.skipped is True


# ---------------------------------------------------------------------
# Hit-occupied: scaling with e and h
# ---------------------------------------------------------------------


class TestHitOccupied:
    def test_landing_z_at_surface_plus_radius(self):
        g = _grid_with_flat_surface_at(0.4)
        dyn = lookup_dynamics("apple")  # spherical, r=0.04
        T = _T_at(0, 0, 1.0)
        T_land, _, info = predict_landing_pose(T, _P_init(), g, dyn)
        assert info.column_state == "hit_occupied"
        # Predicted landing_z ≈ surface_z + r = 0.4 + 0.04 = 0.44
        # (with quantization to voxel grid the surface_z may be off by half a voxel).
        assert abs(T_land[2, 3] - 0.44) < g.voxel_size + 1e-6

    def test_higher_restitution_means_more_bounce_scatter(self):
        # σ_bounce = ε · sqrt(2gh) / max(0.1, 1-e). As e → 1, the
        # denominator floors at 0.1 and σ_bounce grows. Physically: a
        # rubber ball bounces several times, each accruing scatter; a
        # clay ball lands once and stays.
        g = _grid_with_flat_surface_at(0.4)
        dyn_soft = ObjectDynamicsProperty(
            "soft", e=0.10, mu=0.5, shape="spherical", radius_m=0.04)
        dyn_hard = ObjectDynamicsProperty(
            "hard", e=0.99, mu=0.5, shape="spherical", radius_m=0.04)
        T = _T_at(0, 0, 1.0)
        _, _, info_soft = predict_landing_pose(T, _P_init(), g, dyn_soft)
        _, _, info_hard = predict_landing_pose(T, _P_init(), g, dyn_hard)
        assert info_hard.sigma_bounce > info_soft.sigma_bounce
        # And the same ordering propagates into σ_xy.
        assert info_hard.sigma_xy > info_soft.sigma_xy

    def test_taller_drop_means_larger_xy(self):
        # For the same dynamics, a 60 cm drop produces more lateral
        # spread than a 10 cm drop.
        g_low = _grid_with_flat_surface_at(0.9)   # h = 0.1 m drop
        g_high = _grid_with_flat_surface_at(0.4)  # h = 0.6 m drop
        dyn = lookup_dynamics("apple")
        T = _T_at(0, 0, 1.0)
        _, _, info_low = predict_landing_pose(T, _P_init(), g_low, dyn)
        _, _, info_high = predict_landing_pose(T, _P_init(), g_high, dyn)
        assert info_high.sigma_xy > info_low.sigma_xy
        assert info_high.drop_height_m > info_low.drop_height_m

    def test_already_settled_short_circuit(self):
        g = _grid_with_flat_surface_at(0.4)
        dyn = lookup_dynamics("apple")
        # Object already at the surface (h ≈ 0).
        T = _T_at(0, 0, 0.41)
        T_land, _, info = predict_landing_pose(T, _P_init(), g, dyn)
        # Translation should be unchanged (we're already on the surface).
        # σ_xy collapses to the shape factor only.
        assert info.sigma_xy < 0.05


# ---------------------------------------------------------------------
# Voxel-driven branches
# ---------------------------------------------------------------------


class TestVoxelDrivenBranches:
    def test_all_unseen_inflates_sigma_z(self):
        # Empty grid → column is fully UNSEEN.
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, -1.0), (1.0, 1.0, 2.0)),
        )
        dyn = lookup_dynamics("apple")
        T = _T_at(0, 0, 1.0)
        _, P_land, info = predict_landing_pose(
            T, _P_init(), g, dyn, workspace_floor_z=-1.0)
        assert info.column_state == "all_unseen"
        # σ_z should be huge — at least half the workspace height.
        sigma_z = float(np.sqrt(P_land[2, 2] - 1e-4))
        assert sigma_z > 0.5

    def test_all_empty_lands_at_workspace_floor(self):
        # Build a grid with the entire column EMPTY (no occupied voxel).
        g = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, -1.0), (1.0, 1.0, 2.0)),
        )
        i, j, k_top = g.world_to_voxel([0.0, 0.0, 1.0])
        _, _, k_floor = g.world_to_voxel([0.0, 0.0, -0.9])
        for k in range(k_floor, k_top + 1):
            g._n_pass[i, j, k] = g.n_min_pass
        dyn = lookup_dynamics("apple")
        T = _T_at(0, 0, 1.0)
        T_land, _, info = predict_landing_pose(
            T, _P_init(), g, dyn,
            workspace_floor_z=-0.9, max_drop_m=2.0)
        assert info.column_state == "all_empty"
        # Landing z ≈ floor_z + r = -0.9 + 0.04 = -0.86
        assert T_land[2, 3] < 0.0

    def test_mixed_unseen_inflates_sigma_z_above_clean_case(self):
        # Build two grids: one clean (hit_occupied), one with a band of
        # unseen voxels above the surface.
        g_clean = _grid_with_flat_surface_at(0.4)
        # mixed: same surface, but EMPTY only directly above surface, then UNSEEN.
        g_mixed = VoxelObservability(
            voxel_size_m=0.05,
            workspace_aabb=((-1.0, -1.0, -0.5), (1.0, 1.0, 2.0)),
            n_min_hit=2, n_min_pass=3,
        )
        i, j, k_s = g_mixed.world_to_voxel([0.0, 0.0, 0.4])
        # OCCUPIED slab.
        for di in range(-5, 6):
            for dj in range(-5, 6):
                ii, jj = i + di, j + dj
                if 0 <= ii < g_mixed.shape[0] and 0 <= jj < g_mixed.shape[1]:
                    g_mixed._n_hit[ii, jj, k_s] = g_mixed.n_min_hit
        # EMPTY only at k_s + 1 (one voxel directly above), the rest UNSEEN.
        g_mixed._n_pass[i, j, k_s + 1] = g_mixed.n_min_pass

        dyn = lookup_dynamics("apple")
        T = _T_at(0, 0, 1.0)
        _, P_clean, info_clean = predict_landing_pose(T, _P_init(), g_clean, dyn)
        _, P_mixed, info_mixed = predict_landing_pose(T, _P_init(), g_mixed, dyn)
        assert info_clean.column_state == "hit_occupied"
        assert info_mixed.column_state == "mixed_unseen"
        # Mixed-unseen has a band of UNSEEN voxels between release height
        # and surface, so σ_z is inflated.
        assert P_mixed[2, 2] > P_clean[2, 2]
        # Mean landing z is unchanged (surface is known via the OCCUPIED voxel).
        assert info_mixed.surface_z is not None


# ---------------------------------------------------------------------
# Covariance / mean shape sanity
# ---------------------------------------------------------------------


class TestCovarianceShape:
    def test_returns_psd_covariance(self):
        g = _grid_with_flat_surface_at(0.4)
        dyn = lookup_dynamics("apple")
        T = _T_at(0, 0, 1.0)
        _, P_land, _ = predict_landing_pose(T, _P_init(), g, dyn)
        # Symmetric + PSD.
        np.testing.assert_allclose(P_land, P_land.T)
        eigvals = np.linalg.eigvalsh(P_land)
        assert eigvals.min() >= 0.0

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError):
            predict_landing_pose(np.eye(3), np.eye(6),
                                  voxel_obs=None, dyn=DEFAULT_DYNAMICS)

    def test_invalid_P_raises(self):
        with pytest.raises(ValueError):
            predict_landing_pose(np.eye(4), np.eye(3),
                                  voxel_obs=None, dyn=DEFAULT_DYNAMICS)


# ---------------------------------------------------------------------
# Velocity-at-release contribution
# ---------------------------------------------------------------------


class TestReleaseVelocity:
    def test_horizontal_release_velocity_increases_sigma_xy(self):
        g = _grid_with_flat_surface_at(0.4)
        dyn = lookup_dynamics("apple")
        T = _T_at(0, 0, 1.0)
        _, _, info_still = predict_landing_pose(T, _P_init(), g, dyn,
                                                  v_release_world=np.zeros(3))
        _, _, info_moving = predict_landing_pose(
            T, _P_init(), g, dyn,
            v_release_world=np.array([0.5, 0.0, 0.0]))
        assert info_moving.sigma_release_v > info_still.sigma_release_v
        assert info_moving.sigma_xy >= info_still.sigma_xy

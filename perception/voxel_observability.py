"""3D voxel observability grid: per-voxel state in {UNSEEN, EMPTY, OCCUPIED}
with hysteresis on observation counts.

Used by `pose_update/gravity_predict.py` to determine where a released
object is likely to land. The grid is updated each frame from depth
images via per-pixel ray traversal; voxels traversed by rays before
the depth hit accumulate "pass" count (toward EMPTY); voxels at the
depth hit accumulate "hit" count (toward OCCUPIED); voxels never
touched by a ray remain UNSEEN.

Storage: dense numpy uint8/uint16 arrays. For the default 5m × 5m × 3m
workspace at 2 cm voxel size this is ~28 MB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np


# State constants (returned by state_at and as state-codes inside RaycastDownResult).
UNSEEN: int = 0
EMPTY: int = 1
OCCUPIED: int = 2

_STATE_NAME = {UNSEEN: "unseen", EMPTY: "empty", OCCUPIED: "occupied"}


@dataclass
class RaycastDownResult:
    """Result of `VoxelObservability.raycast_down`.

    Attributes:
        surface_z: world-frame z of the OCCUPIED voxel hit (None if none).
        first_unseen_z: world-frame z of the first UNSEEN voxel encountered
            below the start point (None if none encountered).
        floor_z: workspace floor z used as the column terminator.
        states: list of (z, state-code) pairs along the column, top to bottom.
    """
    surface_z: Optional[float]
    first_unseen_z: Optional[float]
    floor_z: float
    states: List[Tuple[float, int]] = field(default_factory=list)

    @property
    def column_state(self) -> str:
        """Categorize the column for `gravity_predict.predict_landing_pose`.

        Returns:
            * 'hit_occupied': column hit an OCCUPIED voxel cleanly (no UNSEEN
              voxel above the surface).
            * 'mixed_unseen': hit OCCUPIED, but at least one UNSEEN voxel
              sits between start and surface — surface is known but the
              path to it is partly observed.
            * 'all_unseen': no OCCUPIED voxel encountered; at least one
              UNSEEN voxel along the column.
            * 'all_empty': no OCCUPIED voxel encountered, no UNSEEN voxel —
              column is fully observed empty all the way to the floor.
        """
        if self.surface_z is not None:
            if (self.first_unseen_z is not None
                    and self.first_unseen_z > self.surface_z):
                return "mixed_unseen"
            return "hit_occupied"
        if self.first_unseen_z is not None:
            return "all_unseen"
        return "all_empty"


class VoxelObservability:
    """Three-state voxel observability grid with hysteresis.

    A voxel becomes OCCUPIED when n_hit >= n_min_hit; EMPTY when n_pass >=
    n_min_pass and the OCCUPIED gate is not met; otherwise UNSEEN. The
    hysteresis suppresses both stray depth pixels (which would otherwise
    flip a single voxel to OCCUPIED on noise) and stale-state inertia
    (downgrade to EMPTY happens after n_min_pass new rays have passed
    through).

    The grid is updated by `integrate_depth(depth, K, T_cw)` and queried
    by `state_at(world_xyz)` and `raycast_down(start_xyz, ...)`.
    """

    def __init__(
        self,
        voxel_size_m: float = 0.02,
        workspace_aabb: Tuple[Tuple[float, float, float],
                              Tuple[float, float, float]] = (
            (-2.5, -2.5, -1.0), (2.5, 2.5, 2.0)),
        n_min_hit: int = 2,
        n_min_pass: int = 3,
    ):
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        self.voxel_size = float(voxel_size_m)
        self.aabb_min = np.asarray(workspace_aabb[0], dtype=np.float64)
        self.aabb_max = np.asarray(workspace_aabb[1], dtype=np.float64)
        if not np.all(self.aabb_max > self.aabb_min):
            raise ValueError("workspace_aabb max must exceed min on every axis")
        extent = self.aabb_max - self.aabb_min
        self.shape: Tuple[int, int, int] = tuple(
            int(np.ceil(extent[i] / self.voxel_size)) for i in range(3))  # type: ignore[assignment]
        self.n_min_hit = int(n_min_hit)
        self.n_min_pass = int(n_min_pass)
        self._n_hit = np.zeros(self.shape, dtype=np.uint16)
        self._n_pass = np.zeros(self.shape, dtype=np.uint16)

    # -----------------------------------------------------------------
    # Per-frame integration
    # -----------------------------------------------------------------
    def integrate_depth(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        T_cw: np.ndarray,
        *,
        max_range_m: float = 3.0,
        subsample: int = 4,
        min_depth_m: float = 0.05,
    ) -> int:
        """Update the grid from one depth frame.

        Args:
            depth: (H, W) float32 metric depth map.
            K: (3, 3) camera intrinsics.
            T_cw: (4, 4) camera-to-world pose.
            max_range_m: ignore depth pixels beyond this range.
            subsample: take every Nth pixel along each axis (4 → 16× speedup).
            min_depth_m: ignore depth values below this threshold.

        Returns:
            number of rays integrated (post-subsample, post-validity).
        """
        if depth.size == 0:
            return 0
        depth = np.asarray(depth, dtype=np.float32)
        K = np.asarray(K, dtype=np.float64)
        T_cw = np.asarray(T_cw, dtype=np.float64)
        h, w = depth.shape

        ss = max(1, int(subsample))
        us, vs = np.meshgrid(
            np.arange(0, w, ss),
            np.arange(0, h, ss),
            indexing="xy")
        us = us.ravel()
        vs = vs.ravel()
        ds = depth[vs, us]

        valid = (ds > min_depth_m) & (ds < max_range_m) & np.isfinite(ds)
        us = us[valid]
        vs = vs[valid]
        ds = ds[valid].astype(np.float64)
        if ds.size == 0:
            return 0

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        x = (us - cx) * ds / fx
        y = (vs - cy) * ds / fy
        z = ds
        pts_cam = np.stack(
            [x, y, z, np.ones_like(z)], axis=1)  # (N, 4)
        pts_world = (T_cw @ pts_cam.T).T[:, :3]  # (N, 3)
        origin = (T_cw @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]

        N = pts_world.shape[0]
        deltas = pts_world - origin[None, :]
        ray_lengths = np.linalg.norm(deltas, axis=1)
        ray_lengths = np.maximum(ray_lengths, self.voxel_size * 0.5)
        unit_dirs = deltas / ray_lengths[:, None]

        # Sample at voxel-size spacing, starting half a voxel from the
        # origin so the camera origin voxel is not over-counted.
        max_len = float(ray_lengths.max())
        n_samples = int(np.ceil(max_len / self.voxel_size)) + 1
        if n_samples < 1:
            return N
        ts = (np.arange(n_samples) + 0.5) * self.voxel_size  # (n_samples,)

        # samples: (N, n_samples, 3)
        samples = (origin[None, None, :]
                   + ts[None, :, None] * unit_dirs[:, None, :])
        ijk = ((samples - self.aabb_min) / self.voxel_size).astype(np.int32)
        in_bounds = (
            (ijk[..., 0] >= 0) & (ijk[..., 0] < self.shape[0])
            & (ijk[..., 1] >= 0) & (ijk[..., 1] < self.shape[1])
            & (ijk[..., 2] >= 0) & (ijk[..., 2] < self.shape[2])
        )
        ts_grid = np.broadcast_to(ts[None, :], (N, n_samples))
        is_within_ray = ts_grid < ray_lengths[:, None]

        # HIT slot: last sample within the ray (closest to the depth hit).
        n_within = is_within_ray.sum(axis=1)
        last_within_idx = np.where(n_within > 0, n_within - 1, 0)
        is_hit_mask = np.zeros((N, n_samples), dtype=bool)
        is_hit_mask[np.arange(N), last_within_idx] = True
        is_hit_mask[n_within == 0, :] = False

        is_pass_mask = is_within_ray & ~is_hit_mask & in_bounds
        is_hit_mask = is_hit_mask & in_bounds

        ijk_flat = ijk.reshape(-1, 3)
        if is_pass_mask.any():
            sel = is_pass_mask.flatten()
            pass_lin = np.ravel_multi_index(
                (ijk_flat[sel, 0], ijk_flat[sel, 1], ijk_flat[sel, 2]),
                self.shape,
            )
            unique_lin, counts = np.unique(pass_lin, return_counts=True)
            cur = self._n_pass.flat[unique_lin].astype(np.int32) + counts
            self._n_pass.flat[unique_lin] = np.minimum(
                cur, np.iinfo(np.uint16).max).astype(np.uint16)
        if is_hit_mask.any():
            sel = is_hit_mask.flatten()
            hit_lin = np.ravel_multi_index(
                (ijk_flat[sel, 0], ijk_flat[sel, 1], ijk_flat[sel, 2]),
                self.shape,
            )
            unique_lin, counts = np.unique(hit_lin, return_counts=True)
            cur = self._n_hit.flat[unique_lin].astype(np.int32) + counts
            self._n_hit.flat[unique_lin] = np.minimum(
                cur, np.iinfo(np.uint16).max).astype(np.uint16)
        return N

    # -----------------------------------------------------------------
    # State queries
    # -----------------------------------------------------------------
    def _state_from_indices(self, i: int, j: int, k: int) -> int:
        if (i < 0 or j < 0 or k < 0
                or i >= self.shape[0] or j >= self.shape[1] or k >= self.shape[2]):
            return UNSEEN
        if self._n_hit[i, j, k] >= self.n_min_hit:
            return OCCUPIED
        if self._n_pass[i, j, k] >= self.n_min_pass:
            return EMPTY
        return UNSEEN

    def world_to_voxel(self, world_xyz) -> Tuple[int, int, int]:
        """Map a world-frame point to integer voxel indices.

        Adds a tiny epsilon before flooring so that floating-point error
        on the boundary (e.g. 0.9 / 0.05 → 17.999...) lands in the
        intended voxel (18) rather than the one below.
        """
        p = np.asarray(world_xyz, dtype=np.float64)
        rel = (p - self.aabb_min) / self.voxel_size
        return (int(np.floor(rel[0] + 1e-9)),
                int(np.floor(rel[1] + 1e-9)),
                int(np.floor(rel[2] + 1e-9)))

    def state_at(self, world_xyz) -> int:
        """Return UNSEEN(0) / EMPTY(1) / OCCUPIED(2) for the voxel at world_xyz."""
        i, j, k = self.world_to_voxel(world_xyz)
        return self._state_from_indices(i, j, k)

    def raycast_down(
        self,
        start_xyz,
        *,
        max_distance_m: float = 2.0,
        floor_z: Optional[float] = None,
        live_object_voxels: Iterable[Tuple[float, float, float, float]] = (),
    ) -> RaycastDownResult:
        """Walk voxels downward (-z) from start_xyz at voxel-size strides.

        Stops at the first OCCUPIED voxel, the workspace floor, or
        max_distance_m. Records the first UNSEEN voxel encountered for
        column-state classification (does not stop).

        Args:
            start_xyz: starting world-frame point.
            max_distance_m: maximum drop distance to consider.
            floor_z: workspace floor z; defaults to AABB minimum z.
            live_object_voxels: iterable of (x, y, z, radius) — these world
                positions are treated as OCCUPIED for this query only
                (not written to the persistent grid).

        Returns:
            RaycastDownResult.
        """
        start = np.asarray(start_xyz, dtype=np.float64)
        if floor_z is None:
            floor_z = float(self.aabb_min[2])
        bottom_z = max(float(floor_z), float(start[2]) - float(max_distance_m))

        live_list = list(live_object_voxels) or []

        states: List[Tuple[float, int]] = []
        surface_z: Optional[float] = None
        first_unseen_z: Optional[float] = None

        # Use integer voxel indexing for the z-walk so that accumulated
        # floating-point drift cannot bump the walker into a neighbouring
        # voxel after many iterations.
        i, j, k_start = self.world_to_voxel(start)
        _, _, k_floor = self.world_to_voxel(
            np.array([start[0], start[1], bottom_z]))

        # Skip the voxel containing the start point itself (the object's
        # bottom is not part of the support column).
        for k in range(k_start - 1, k_floor - 1, -1):
            # Voxel center in world frame.
            z_center = float(self.aabb_min[2] + (k + 0.5) * self.voxel_size)
            in_bounds = (0 <= i < self.shape[0]
                         and 0 <= j < self.shape[1]
                         and 0 <= k < self.shape[2])
            if not in_bounds:
                state = UNSEEN
            elif _point_in_any_sphere(start[0], start[1], z_center, live_list):
                state = OCCUPIED
            else:
                state = self._state_from_indices(i, j, k)
            states.append((z_center, state))
            if state == OCCUPIED:
                surface_z = z_center
                break
            if state == UNSEEN and first_unseen_z is None:
                first_unseen_z = z_center

        return RaycastDownResult(
            surface_z=surface_z,
            first_unseen_z=first_unseen_z,
            floor_z=bottom_z,
            states=states,
        )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------
    def stats(self) -> dict:
        """Return diagnostic counters."""
        n_total = int(np.prod(self.shape))
        n_occupied = int((self._n_hit >= self.n_min_hit).sum())
        n_empty = int(((self._n_pass >= self.n_min_pass)
                       & (self._n_hit < self.n_min_hit)).sum())
        n_unseen = n_total - n_occupied - n_empty
        return dict(
            n_unseen=n_unseen,
            n_empty=n_empty,
            n_occupied=n_occupied,
            n_total=n_total,
            bytes_used=int(self._n_hit.nbytes + self._n_pass.nbytes),
            shape=self.shape,
            voxel_size=self.voxel_size,
        )


def _point_in_any_sphere(
    x: float, y: float, z: float,
    spheres: List[Tuple[float, float, float, float]],
) -> bool:
    for (sx, sy, sz, sr) in spheres:
        dx = x - sx
        dy = y - sy
        dz = z - sz
        if dx * dx + dy * dy + dz * dz < sr * sr:
            return True
    return False

"""Visibility predicate p_v^(i) via depth-image ray-tracing (z-buffer test).

For each track `i`, we ask: given the predicted object pose and a set of
sampled surface points on the object, what fraction of those points
would actually be visible in the current depth image?

For every sample point `p_obj` (object-local frame) attached to track i:
  1. Transform to camera frame:  p_cam = T_co · p_obj.
  2. Skip if behind camera (`p_cam.z ≤ eps`).
  3. Project through intrinsics K to pixel `(u, v)`.
  4. Skip if outside the image rectangle (frustum gate).
  5. Read observed depth `d_actual = depth[v, u]`.
  6. z-test: point is visible if `d_actual ≥ p_cam.z − τ`, where τ covers
     depth-sensor noise and the object's own depth extent
     (`τ = z_tol_abs + z_tol_rel · z + 2·obj_radius`).

`p_v = n_visible / n_valid` where `n_valid` counts in-frustum samples
with a finite, in-range depth reading. Edge cases:
  * No in-frustum samples  →  p_v = 0.
  * In frustum but all depths invalid (NaN / 0 / out of range)
                            →  p_v = 1 (conservative: cannot claim
                                       occlusion without evidence).

Why depth ray-tracing over bbox overlap
───────────────────────────────────────
The previous `visibility_p_v` used only pairs of tracks with current
detection bounding boxes as occluders, which mis-handled three
common failure modes:
  * Untracked occluders (walls, shelves, objects the detector didn't
    emit). These are PRESENT in the depth image; a proper z-buffer
    test picks them up automatically.
  * Tracks WITHOUT a current detection bbox. The old code treated them
    as fully visible (`p_v=1`), ignoring whatever was in front of them.
  * Bbox overlap is a 2D approximation that systematically over- or
    under-counts relative to the real 3D occlusion.

A single pass over the depth image, vectorised across every sample
point of every track, handles all of these in O(N_samples).

Vectorisation
─────────────
All track samples are concatenated into a single (N_total, 3) array;
the projection, in-frustum mask, and depth lookup are one NumPy
operation each. Per-track reduction uses `np.bincount`; total work is
O(N_total).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Fibonacci-sphere fallback when a track has no ref_points
# ─────────────────────────────────────────────────────────────────────

def _fibonacci_sphere(n: int, radius: float) -> np.ndarray:
    """Return N approximately-uniform points on a sphere of given radius.

    Deterministic (Fibonacci spiral), so results are reproducible across
    runs without needing an RNG.
    """
    n = max(int(n), 1)
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.column_stack([x, y, z]) * float(radius)


# ─────────────────────────────────────────────────────────────────────
# Main entry: vectorised batch visibility over all tracks
# ─────────────────────────────────────────────────────────────────────

def visibility_p_v(
    tracks: List[Dict[str, Any]],
    K: np.ndarray,
    depth: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    max_samples_per_track: int = 256,
    fallback_sphere_samples: int = 64,
    fallback_obj_radius: float = 0.05,
    z_tol_abs: float = 0.02,
    z_tol_rel: float = 0.02,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Dict[int, float]:
    """Compute `p_v[oid]` for every track via depth ray-tracing.

    Args:
        tracks: list of per-track dicts. Each MUST contain:
            `oid: int`
            `T_co: (4, 4) np.ndarray` — camera-frame pose
                (the filter's PREDICTED mean for this frame).
        Each MAY contain:
            `ref_points_obj: (N, 3) np.ndarray` — object-local surface
                cloud (e.g. `PoseEstimator._refs[oid].ref_points`).
                When provided, used as the sample source; otherwise
                falls back to a Fibonacci sphere of `obj_radius`.
            `obj_radius: float` — object extent in metres. Used for
                sphere-sample radius AND for the z-test geometric
                tolerance (`2·obj_radius`). Default
                `fallback_obj_radius`.
        K: (3, 3) camera intrinsics.
        depth: (H, W) float32 depth image in metres. NaN / 0 / out-of-
            range values are treated as "no information" (conservative:
            excluded from the denominator).
        image_shape: (H, W).
        max_samples_per_track: if a track's `ref_points_obj` is longer
            than this, it's downsampled (deterministic stride).
        fallback_sphere_samples: N for the Fibonacci sphere fallback.
        z_tol_abs, z_tol_rel: depth-sensor noise tolerance,
            `τ_sensor = z_tol_abs + z_tol_rel · z`.
        min_depth, max_depth: valid depth range (values outside are
            treated as invalid).

    Returns:
        dict `{oid: p_v in [0, 1]}`. `p_v = 0` means fully out-of-FOV
        or fully occluded; `p_v = 1` means fully visible (no evidence
        of occlusion at any sample).
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    K = np.asarray(K, dtype=np.float64)
    depth = np.asarray(depth)
    if depth.dtype != np.float64:
        depth = depth.astype(np.float64, copy=False)

    if not tracks:
        return {}

    # ── Stage 1: stack all samples across tracks in one (N_total, 3) array.
    all_pts_cam: List[np.ndarray] = []
    all_geom_tol: List[np.ndarray] = []
    all_track_idx: List[np.ndarray] = []
    oids: List[int] = []
    for i, tr in enumerate(tracks):
        oid = int(tr["oid"])
        T_co = np.asarray(tr["T_co"], dtype=np.float64)
        obj_radius = float(tr.get("obj_radius", fallback_obj_radius))
        ref_pts = tr.get("ref_points_obj")
        if ref_pts is None or len(ref_pts) == 0:
            pts_obj = _fibonacci_sphere(fallback_sphere_samples, obj_radius)
        else:
            pts_obj = np.asarray(ref_pts, dtype=np.float64)
            if len(pts_obj) > max_samples_per_track:
                idx = np.linspace(0, len(pts_obj) - 1,
                                   max_samples_per_track).astype(np.int64)
                pts_obj = pts_obj[idx]
        # Transform to camera frame.
        pts_cam = pts_obj @ T_co[:3, :3].T + T_co[:3, 3]
        all_pts_cam.append(pts_cam)
        all_geom_tol.append(np.full(len(pts_cam),
                                      2.0 * obj_radius,
                                      dtype=np.float64))
        all_track_idx.append(np.full(len(pts_cam), i, dtype=np.int64))
        oids.append(oid)

    pts_cam = np.concatenate(all_pts_cam, axis=0)
    geom_tol = np.concatenate(all_geom_tol)
    track_idx_all = np.concatenate(all_track_idx)
    n_tracks = len(oids)

    # ── Stage 2: project everything at once.
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = pts_cam[:, 2]
    in_front = z > min_depth
    safe_z = np.where(in_front, z, 1.0)
    u = fx * pts_cam[:, 0] / safe_z + cx
    v = fy * pts_cam[:, 1] / safe_z + cy
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    in_img = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    # ── Stage 3: vectorised depth lookup (use safe indices for OOB pixels).
    ui_safe = np.where(in_img, ui, 0)
    vi_safe = np.where(in_img, vi, 0)
    d_actual = depth[vi_safe, ui_safe]
    valid_depth = (np.isfinite(d_actual)
                    & (d_actual > min_depth)
                    & (d_actual < max_depth))
    valid = in_img & valid_depth

    # ── Stage 4: z-test. "Visible" = nothing closer than (z − τ).
    tol = z_tol_abs + z_tol_rel * z + geom_tol
    visible = valid & (d_actual >= z - tol)

    # ── Stage 5: per-track reduction via bincount.
    n_in_fov = np.bincount(track_idx_all[in_img], minlength=n_tracks)
    n_valid = np.bincount(track_idx_all[valid], minlength=n_tracks)
    n_vis = np.bincount(track_idx_all[visible], minlength=n_tracks)

    out: Dict[int, float] = {}
    for i, oid in enumerate(oids):
        if n_in_fov[i] == 0:
            out[oid] = 0.0
        elif n_valid[i] == 0:
            out[oid] = 1.0            # no depth info → conservative
        else:
            out[oid] = float(n_vis[i]) / float(n_valid[i])
    return out

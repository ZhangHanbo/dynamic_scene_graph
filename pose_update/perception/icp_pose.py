"""
Per-object pose estimator with three switchable backends.

Methods
═══════

1. **centroid**  — translation-only back-projected centroid.
   Cheapest, no rotation, fixed R_icp. What the original integration
   test used. Useful as a lower-bound reference.

2. **icp_chain** — ICP against a first-frame reference cloud, with the
   ICP initialization taken from the PREVIOUS frame's result (rotation
   warmstart) and the current centroid (translation reset). This is the
   default "continuous-tracking" mode. Can drift because each frame's
   ICP error feeds the next frame's initialization.

3. **icp_anchor** — ICP against the first-frame reference cloud, with
   a fully state-free, fully camera-frame init:
       rotation:    identity.
       translation: current centroid of masked depth.
   No T_wb anywhere. No prev_T_co. Each frame's ICP is independent of
   both localization and past ICP results.

   Semantics: this is the strict interpretation of the uncertainty
   decomposition — localization (Σ_wb) lives in the filter's world-
   frame state; the OBSERVATION is purely a camera-frame quantity
   produced from the CURRENT observation alone. We only care about
   the camera-frame TRANSLATION; any rotation-drift behaviour is the
   filter's problem, not the observation's.


Shared interface
────────────────

    est = PoseEstimator(K, method="icp_chain")   # or "centroid", "icp_anchor"
    for each detection d:
        T_co, R_icp, fitness, rmse = est.estimate(
            oid=d["id"], mask=d["mask"], depth=depth)
        if T_co is None:
            continue                             # dropped (ICP fitness gate)
        d["T_co"], d["R_icp"], d["fitness"], d["rmse"] = ...

None of the three methods reads T_wb / camera-to-world. Layer 1 (world-
frame localization) and Layer 2 (camera-frame object pose from ICP) are
strictly independent — composition happens in the filter, not here.

All three methods share the same fitness gate: if ICP fitness < MIN_FITNESS
or RMSE > MAX_RMSE, return `(None, None, fitness, rmse)` so the caller
drops the detection. (centroid doesn't run ICP and is never rejected.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion


METHODS = (
    "centroid",
    "icp_chain",
    "icp_anchor",
    "icp_chain_strict",   # NEW: prev_T_co full init, no centroid reset
    "icp_anchor_strict",  # NEW: first-frame T_co full init, no centroid reset
)


# ─────────────────────────────────────────────────────────────────────
# Mask-edge cloud filter
#
# Mask edges from SAM2 often straddle the object/background boundary,
# so edge pixels pull background depth and back-project to "long tail"
# outliers far from the real object. Two cheap pre-filters:
#
#   1. Erode the binary mask by N pixels. Drops the worst boundary band
#      where pixel-level assignment is least reliable.
#   2. Reject masked pixels whose local 4-neighbour depth jump exceeds
#      `depth_edge_max_jump`. Surgically kills the "cliff" pixels at
#      foreground-to-background discontinuities (typical object-body
#      curvature gradient is <1 cm; background is 10s of cm).
#
# Both stages self-disable if the filter would take the mask below
# `min_points` — never kill a small mask entirely.
# ─────────────────────────────────────────────────────────────────────

def _clean_mask(mask: np.ndarray,
                depth: np.ndarray,
                erosion_iter: int = 2,
                depth_edge_max_jump: float = 0.02,
                min_depth: float = 0.1,
                max_depth: float = 5.0,
                min_points: int = 30) -> np.ndarray:
    """Clean a binary mask by erosion + per-pixel depth-gradient rejection.

    Args:
        mask:                 (H, W) uint8/bool. Foreground pixels > 0.
        depth:                (H, W) float. Metres. 0/NaN = invalid.
        erosion_iter:         number of 1-pixel binary-erode passes.
        depth_edge_max_jump:  maximum allowed |Δdepth| to any 4-neighbour,
                              in metres. Set 0 / None to skip this stage.
        min_depth, max_depth: depth validity band used to mask out the
                              invalid holes before computing gradients.
        min_points:           if either stage would leave < this many
                              masked pixels, skip it (self-disable).

    Returns:
        (H, W) uint8 mask with outlier edge pixels removed.
    """
    mask_b = np.asarray(mask) > 0
    if mask_b.sum() < min_points:
        return mask_b.astype(np.uint8)

    # Stage 1: binary erosion (drop worst boundary band unconditionally).
    if erosion_iter and erosion_iter > 0:
        eroded = binary_erosion(mask_b, iterations=int(erosion_iter))
        if eroded.sum() >= min_points:
            mask_b = eroded
        # else: keep the un-eroded mask — small masks can't afford to lose
        # their thin rim.

    # Stage 2: depth-gradient rejection. Use the eroded mask as the
    # "believable" core; a masked pixel whose depth jumps >threshold to
    # any 4-neighbour is a foreground/background boundary pixel.
    if depth_edge_max_jump and depth_edge_max_jump > 0:
        d = np.asarray(depth, dtype=np.float64)
        invalid = ~np.isfinite(d) | (d <= min_depth) | (d >= max_depth)
        d = np.where(invalid, np.nan, d)

        ys, xs = np.where(mask_b)
        if ys.size == 0:
            return mask_b.astype(np.uint8)
        d_self = d[ys, xs]

        max_jump = np.zeros_like(d_self)
        H, W = d.shape
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ys_n = ys + dy
            xs_n = xs + dx
            in_bounds = ((ys_n >= 0) & (ys_n < H)
                         & (xs_n >= 0) & (xs_n < W))
            d_n = np.full_like(d_self, np.nan)
            d_n[in_bounds] = d[ys_n[in_bounds], xs_n[in_bounds]]
            jump = np.abs(d_self - d_n)
            # NaN (own invalid OR neighbour invalid) → no information;
            # treat as no-jump so we don't reject based on missing data.
            jump = np.where(np.isnan(jump), 0.0, jump)
            max_jump = np.maximum(max_jump, jump)
        # Also drop pixels whose own depth is NaN — they'd be rejected
        # by _back_project anyway, but we prevent their invalid value
        # from confusing downstream consumers that don't re-check depth.
        keep = (max_jump <= depth_edge_max_jump) & ~np.isnan(d_self)
        filtered = np.zeros_like(mask_b)
        filtered[ys[keep], xs[keep]] = True
        if filtered.sum() >= min_points:
            mask_b = filtered

    return mask_b.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _back_project(mask: np.ndarray,
                  depth: np.ndarray,
                  K: np.ndarray,
                  min_depth: float = 0.1,
                  max_depth: float = 5.0,
                  min_points: int = 30,
                  clean_mask: bool = True) -> Optional[np.ndarray]:
    """Back-project masked, valid-depth pixels to (N, 3) camera-frame points.

    If `clean_mask=True`, `_clean_mask` pre-filters edge leakage pixels
    before back-projection. The filter self-disables when it would drop
    the mask below `min_points`, so small masks still pass through.
    """
    m = _clean_mask(mask, depth,
                    min_depth=min_depth, max_depth=max_depth,
                    min_points=min_points) if clean_mask else mask
    ys, xs = np.where(m > 0)
    if len(xs) < min_points:
        return None
    ds = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(ds) & (ds > min_depth) & (ds < max_depth)
    if valid.sum() < min_points:
        return None
    xs, ys, ds = xs[valid], ys[valid], ds[valid]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = (xs - cx) * ds / fx
    Y = (ys - cy) * ds / fy
    Z = ds
    return np.stack([X, Y, Z], axis=1)


def _voxelize(pts: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc.voxel_down_sample(voxel)


def centroid_cam_from_mask(mask: np.ndarray,
                            depth: np.ndarray,
                            K: np.ndarray,
                            min_points: int = 30,
                            ) -> Optional[np.ndarray]:
    """Return the camera-frame centroid of a masked depth region, or None
    when there are not enough valid-depth pixels.

    This is the "coarse feature" used by `SceneTracker.step()` to build
    the Hungarian cost matrix *before* any ICP runs -- predict ->
    associate (on centroids) -> measure (ICP per matched pair) ->
    update. Centroids are cheap (one `_back_project` + mean) and
    carry the 3-DOF information that dominates association anyway
    (rotation is noisy per-mask and is disambiguated only after
    matching, via fine ICP with the filter prior).

    Returns a length-3 numpy vector (camera frame, `(X, Y, Z)` in
    metres) or `None` if the back-projection dropped below
    `min_points` valid-depth pixels.
    """
    pts = _back_project(mask, depth, K, min_points=min_points)
    if pts is None:
        return None
    return pts.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────
# Per-object state
# ─────────────────────────────────────────────────────────────────────

@dataclass
class _ObjRef:
    """Per-object state — strictly camera-frame. No world-frame info."""
    ref_points: np.ndarray        # (M, 3) in object-local frame (centroid-centered)
    prev_T_co: np.ndarray         # (4, 4) last successful ICP result (chain variants)
    first_T_co: np.ndarray        # (4, 4) first-frame T_co (anchor_strict init)
    obj_radius: float             # metres — for rotation uncertainty scaling
    n_frames_tracked: int


# ─────────────────────────────────────────────────────────────────────
# Unified estimator
# ─────────────────────────────────────────────────────────────────────

class PoseEstimator:
    """Per-object T_co estimator with three switchable backends.

    Args:
        K:        (3, 3) camera intrinsics.
        method:   one of "centroid", "icp_chain", "icp_anchor".
    """

    # Open3D ICP parameters.
    VOXEL_SIZE = 0.005       # 5 mm
    ICP_THRESHOLD = 0.020    # 2 cm correspondence threshold
    ICP_MAX_ITER = 30

    # Fitness gate (strict).  Below this the observation is dropped.
    MIN_FITNESS = 0.90
    MAX_RMSE = 0.015         # 15 mm safety net

    # Covariance floors (variances). The translation floor is the
    # MINIMUM measurement noise per axis; setting it to (2 cm)^2 caps the
    # EKF posterior P from shrinking below realistic per-frame perception
    # jitter (mask-edge depth noise + ICP local-optimum drift). Without
    # this, after many observations P -> below-mm and the chi^2_3
    # outer gate rejects 2-3 cm jitter on the next frame, sending it to
    # birth -- the apple_in_the_tray frame-430 case.
    TRANS_VAR_FLOOR = 4e-4   # (2 cm)^2 per translation axis
    ROT_VAR_FLOOR = 1e-2     # (~0.1 rad)^2 per rotation axis (paper §2.3)

    # Hand-chosen constant covariance for the centroid method (no ICP
    # diagnostics available).
    CENTROID_R_ICP = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])

    # Reference-cloud accumulation (see `_merge_into_ref`):
    #   * Only high-fitness ICP results contribute new points.
    #   * The merged cloud is voxel-downsampled at VOXEL_SIZE (same as the
    #     ICP source/target) so its size is strictly bounded by the object
    #     volume / voxel^3. A hard cap on MAX_REF_POINTS prevents any
    #     pathological growth (it shouldn't happen in practice; the cap is
    #     a safety net).
    REF_UPDATE_MIN_FITNESS = 0.95
    MAX_REF_POINTS = 4000

    def __init__(self, K: np.ndarray, method: str = "icp_chain"):
        if method not in METHODS:
            raise ValueError(
                f"method must be one of {METHODS}, got {method!r}")
        self.K = np.asarray(K, dtype=np.float64)
        self.method = method
        self._refs: Dict[int, _ObjRef] = {}

    # --------------------------------------------------------------- #
    #  Public entry point
    # --------------------------------------------------------------- #

    def estimate(self,
                 oid: int,
                 mask: np.ndarray,
                 depth: np.ndarray,
                 T_cw: Optional[np.ndarray] = None,
                 T_co_init: Optional[np.ndarray] = None,
                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                            float, float]:
        """Estimate (T_co, R_icp) for one object detection at one frame.

        Args:
            oid:   object id (used to key per-object reference state).
            mask:  (H, W) bool detection mask.
            depth: (H, W) float32 depth in metres.
            T_cw:  kept in the signature for API parity; NONE of the
                   three methods currently consume it. The anchor's
                   decomposition premise says ICP is strictly
                   camera-frame and independent of localization.
            T_co_init: optional (4, 4) ICP initialization in the camera
                   frame. When provided AND `oid` already has a saved
                   reference cloud, this overrides the per-method
                   warm-start (`prev_T_co` for chain, identity for
                   anchor). Use this to thread the Kalman filter's
                   predicted pose through ICP, i.e., run ICP with
                   the filter prior as the seed, which is the
                   standard "predict before measure" KF pattern.
                   Ignored on the first observation of `oid` (when
                   there is no reference cloud to refine against;
                   the birth path always anchors at the current
                   centroid).

        Returns:
            T_co:    (4, 4) or None (None → drop this detection).
            R_icp:   (6, 6) or None.
            fitness: ICP inlier fraction or 1.0 (centroid).
            rmse:    ICP inlier RMSE or 0.0 (centroid).
        """
        if self.method == "centroid":
            return self._estimate_centroid(mask, depth)
        return self._estimate_icp(oid, mask, depth,
                                   init_policy=self.method,
                                   T_co_init=T_co_init)

    # --------------------------------------------------------------- #
    #  Method 1: centroid
    # --------------------------------------------------------------- #

    def _estimate_centroid(self,
                           mask: np.ndarray,
                           depth: np.ndarray,
                           ) -> Tuple[Optional[np.ndarray],
                                      Optional[np.ndarray], float, float]:
        pts = _back_project(mask, depth, self.K)
        if pts is None:
            return None, None, 0.0, 0.0
        T_co = np.eye(4, dtype=np.float64)
        T_co[:3, 3] = pts.mean(axis=0)
        return T_co, self.CENTROID_R_ICP.copy(), 1.0, 0.0

    # --------------------------------------------------------------- #
    #  Method 2 + 3: ICP (chain or anchor init)
    # --------------------------------------------------------------- #

    def _estimate_icp(self,
                      oid: int,
                      mask: np.ndarray,
                      depth: np.ndarray,
                      init_policy: str,
                      T_co_init: Optional[np.ndarray] = None,
                      ) -> Tuple[Optional[np.ndarray],
                                 Optional[np.ndarray], float, float]:
        pts_cam = _back_project(mask, depth, self.K)
        if pts_cam is None:
            return None, None, 0.0, 0.0

        centroid_now = pts_cam.mean(axis=0)

        # ── First observation: anchor object-local frame at the centroid ─
        if oid not in self._refs:
            ref_points = pts_cam - centroid_now
            # Voxel-downsample + hard-cap at MAX_REF_POINTS so the
            # birth cloud is bounded the same way subsequent
            # `_merge_into_ref` outputs are. Without this, a large
            # mask (thousands of valid pixels) would seed a ref cloud
            # that exceeds the cap; later merges would still be
            # bounded, but the initial size wouldn't be.
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(ref_points)
            pc = pc.voxel_down_sample(self.VOXEL_SIZE)
            ref_points = np.asarray(pc.points, dtype=np.float64)
            if len(ref_points) > self.MAX_REF_POINTS:
                idx = np.linspace(0, len(ref_points) - 1,
                                    self.MAX_REF_POINTS).astype(np.int64)
                ref_points = ref_points[idx]
            obj_radius = max(float(np.linalg.norm(ref_points, axis=1).max()),
                              0.03)
            T_co_init = np.eye(4, dtype=np.float64)
            T_co_init[:3, 3] = centroid_now
            self._refs[oid] = _ObjRef(
                ref_points=ref_points,
                prev_T_co=T_co_init.copy(),
                first_T_co=T_co_init.copy(),
                obj_radius=obj_radius,
                n_frames_tracked=0,
            )
            # Inflated birth prior (bernoulli_ekf.tex §2.3 / §8): the
            # first-frame centroid can drift 2-5 cm across adjacent frames
            # from mask boundary noise, non-uniform depth sampling, and
            # actual object motion. A tight R here becomes a tight birth
            # covariance P^(i_new) = R under init_cov_from_R, which makes
            # d^2 > G_out=25 on nearly every re-detection and spawns a
            # fresh track every frame. 4cm trans / 0.2rad rot std keeps
            # the chi^2_6 gate permissive for the first few frames until
            # real ICP takes over.
            R_icp = np.diag([0.04**2]*3 + [0.2**2]*3)
            return T_co_init, R_icp, 1.0, 0.0

        ref = self._refs[oid]

        # ── Initial guess ──────────────────────────────────────────────
        # Standard KF order is "predict before measure": when the caller
        # has already run the EKF predict step and supplies the predicted
        # camera-frame pose `T_co_init`, use it directly as the ICP seed.
        # That is the Bayesian-correct way to inject the filter's prior
        # into the measurement function (cf.\ IEKF: a single Kalman gain
        # step at this seed is what \code{update_observation} does
        # downstream). Falls back to the per-method warm-start when no
        # prior is supplied (birth path, or back-compat callers).
        # No T_wb anywhere: localization lives in the filter, the visual
        # observation is strictly camera-frame.
        if T_co_init is not None:
            init_T = np.asarray(T_co_init, dtype=np.float64).copy()
        elif init_policy == "icp_chain":
            # Previous rotation, current-centroid translation.
            init_T = ref.prev_T_co.copy()
            init_T[:3, 3] = centroid_now
        elif init_policy == "icp_anchor":
            # Stateless: identity rotation, current-centroid translation.
            init_T = np.eye(4, dtype=np.float64)
            init_T[:3, 3] = centroid_now
        elif init_policy == "icp_chain_strict":
            # Pure chain: previous T_co (BOTH rotation AND translation).
            # No centroid reset — translation also chains forward.
            init_T = ref.prev_T_co.copy()
        elif init_policy == "icp_anchor_strict":
            # Pure anchor: first-frame T_co (BOTH rotation AND translation).
            # Init never updates from the first observation. Truly stateless
            # across all later frames — the SAME init is used every time.
            init_T = ref.first_T_co.copy()
        else:
            raise ValueError(f"unknown init_policy {init_policy!r}")

        # ── Run ICP ────────────────────────────────────────────────────
        src = _voxelize(ref.ref_points, self.VOXEL_SIZE)
        tgt = _voxelize(pts_cam, self.VOXEL_SIZE)
        result = o3d.pipelines.registration.registration_icp(
            src, tgt,
            self.ICP_THRESHOLD,
            init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.ICP_MAX_ITER),
        )
        T_co = np.asarray(result.transformation, dtype=np.float64)
        fitness = float(result.fitness)
        rmse = float(result.inlier_rmse)

        icp_ok = (fitness >= self.MIN_FITNESS
                  and rmse <= self.MAX_RMSE
                  and np.isfinite(T_co).all())
        if not icp_ok:
            # Refresh chain init so the next frame has a better seed;
            # anchor init is state-free so this is a no-op for it.
            refreshed = ref.prev_T_co.copy()
            refreshed[:3, 3] = centroid_now
            ref.prev_T_co = refreshed
            return None, None, fitness, rmse

        # ── Data-driven R_icp ──────────────────────────────────────────
        fitness_scale = min(5.0, 0.5 / max(fitness, 1e-3))
        trans_var = max(rmse**2, self.TRANS_VAR_FLOOR) * fitness_scale
        rot_var = max((rmse / ref.obj_radius)**2,
                       self.ROT_VAR_FLOOR) * fitness_scale
        R_icp = np.diag([trans_var]*3 + [rot_var]*3)

        ref.prev_T_co = T_co.copy()
        ref.n_frames_tracked += 1

        # ── Object-model update: accumulate new surface points ────────
        # Merges the current frame's camera-frame points (transformed
        # back to the object-local frame via T_co^-1) into the
        # reference cloud, then voxel-downsamples to keep the total
        # bounded (the voxel grid caps density, and MAX_REF_POINTS is
        # a hard safety cap). Only runs on HIGH-FITNESS matches to
        # avoid baking ICP error into the canonical shape.
        if fitness >= self.REF_UPDATE_MIN_FITNESS:
            self._merge_into_ref(ref, pts_cam, T_co)

        return T_co, R_icp, fitness, rmse

    # --------------------------------------------------------------- #
    #  Object-model accumulation
    # --------------------------------------------------------------- #

    def _merge_into_ref(self,
                         ref: _ObjRef,
                         pts_cam: np.ndarray,
                         T_co: np.ndarray) -> None:
        """Fold the current camera-frame observation into `ref.ref_points`.

        Pipeline:
            p_obj_new = T_co^{-1} @ p_cam          (lift to object-local frame)
            merged    = concat(ref.ref_points, p_obj_new)
            merged    = voxel_down_sample(merged, VOXEL_SIZE)
            merged    = subsample(merged, MAX_REF_POINTS) if |merged| too big
            ref.ref_points  = merged
            ref.obj_radius  = max(‖p‖ for p in merged)  (monotone non-decreasing)

        `voxel_down_sample` is deterministic and idempotent under repeated
        merges of noise-free points — so a stationary object converges to
        a stable cloud. ICP drift enters as 2--3 mm noise per frame, which
        voxelisation at VOXEL_SIZE=5 mm absorbs.
        """
        pts_cam = np.asarray(pts_cam, dtype=np.float64)
        T_co_inv = np.linalg.inv(T_co)
        pts_obj_new = pts_cam @ T_co_inv[:3, :3].T + T_co_inv[:3, 3]
        merged = np.vstack([ref.ref_points, pts_obj_new])
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(merged)
        pc = pc.voxel_down_sample(self.VOXEL_SIZE)
        merged = np.asarray(pc.points, dtype=np.float64)
        # Safety cap (deterministic stride).
        if len(merged) > self.MAX_REF_POINTS:
            idx = np.linspace(0, len(merged) - 1,
                               self.MAX_REF_POINTS).astype(np.int64)
            merged = merged[idx]
        ref.ref_points = merged
        ref.obj_radius = max(float(np.linalg.norm(merged, axis=1).max()),
                              0.03)

    # --------------------------------------------------------------- #

    def reset(self, oid: Optional[int] = None) -> None:
        """Drop per-object cache."""
        if oid is None:
            self._refs.clear()
        else:
            self._refs.pop(oid, None)


# ─────────────────────────────────────────────────────────────────────
# Backward-compatibility alias
# ─────────────────────────────────────────────────────────────────────

# Keep the old import path working while we transition visualize_pipeline.
ICPPoseEstimator = PoseEstimator

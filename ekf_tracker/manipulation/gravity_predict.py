"""Gravity-aware one-shot predict for the EKF on object release.

Called once when the gripper releases an object: simulates the
free-fall + bounce + roll dispersion using a parametric model and the
voxel observability grid (`perception.voxel_observability`) to find
the supporting surface below the object.

The output is an updated SE(3) mean and 6×6 covariance (in `[v, ω]`
tangent ordering matching `pose_update/ekf_se3.py`). Subsequent frames
revert to the standard static-predict — this function is invoked at
the release-transition edge only.

Parametric model (full derivation in
`docs/ekf_tracker/latex/bernoulli_ekf.tex`, "Gravity-aware predict at release"):

    σ_xy² = σ_bounce² + σ_release_v² + σ_shape²
        σ_bounce    = ε_roughness · sqrt(2 g h) / max(0.1, 1 - e)
        σ_release_v = 0.5 · ‖v_xy‖ · t_fall
        σ_shape     = r_obj · shape_footprint_factor(shape)
    σ_z       = 0.5 · r_obj · (1 - e/2)              (clean surface)
    σ_yaw     = π · (1 - exp(-h · (1 - e²) / r_obj))
    σ_roll    = σ_pitch = arctan(r_obj / max(h_stable, 0.01))

Voxel-driven adjustment to (h, σ_z):
    hit_occupied: h = surface offset; σ_z as above.
    mixed_unseen: h = surface offset; σ_z += (h_unseen - h_surface) / 2.
    all_unseen:   h = midpoint of [h_unseen, h_floor]; σ_z = half-interval.
    all_empty:    h = h_floor; σ_z = r_obj.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np

from utils.object_dynamics import (
    ObjectDynamicsProperty,
    shape_footprint_factor,
)
from perception.voxel_observability import VoxelObservability


# Default surface-roughness coefficient in σ_bounce. Units: m / (m·s⁻¹) →
# i.e. a 1 m/s impact velocity produces ~5 mm of horizontal scatter per
# bounce. Order-of-magnitude; tune empirically on the apple_drop trajectory.
EPS_ROUGHNESS_DEFAULT = 5.0e-3


@dataclass
class GravityPredictInfo:
    """Diagnostic record returned by `predict_landing_pose`.

    Surfaced via the orchestrator's gravity-predict log so visualizers
    and integration tests can inspect the parametric components.
    """
    column_state: str
    drop_height_m: float
    surface_z: Optional[float]
    first_unseen_z: Optional[float]
    floor_z: float
    sigma_xy: float
    sigma_z: float
    sigma_yaw: float
    sigma_roll: float
    sigma_pitch: float
    sigma_bounce: float
    sigma_release_v: float
    sigma_shape: float
    landing_z: float
    skipped: bool = False
    skip_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "column_state": self.column_state,
            "drop_height_m": float(self.drop_height_m),
            "surface_z": (None if self.surface_z is None
                          else float(self.surface_z)),
            "first_unseen_z": (None if self.first_unseen_z is None
                                else float(self.first_unseen_z)),
            "floor_z": float(self.floor_z),
            "sigma_xy": float(self.sigma_xy),
            "sigma_z": float(self.sigma_z),
            "sigma_yaw": float(self.sigma_yaw),
            "sigma_roll": float(self.sigma_roll),
            "sigma_pitch": float(self.sigma_pitch),
            "sigma_bounce": float(self.sigma_bounce),
            "sigma_release_v": float(self.sigma_release_v),
            "sigma_shape": float(self.sigma_shape),
            "landing_z": float(self.landing_z),
            "skipped": bool(self.skipped),
            "skip_reason": str(self.skip_reason),
        }


def predict_landing_pose(
    T_release: np.ndarray,
    P_release: np.ndarray,
    voxel_obs: Optional[VoxelObservability],
    dyn: ObjectDynamicsProperty,
    *,
    gravity: float = 9.81,
    workspace_floor_z: float = -1.0,
    eps_roughness: float = EPS_ROUGHNESS_DEFAULT,
    max_drop_m: float = 2.0,
    v_release_world: Optional[np.ndarray] = None,
    live_object_voxels: Iterable[Tuple[float, float, float, float]] = (),
) -> Tuple[np.ndarray, np.ndarray, GravityPredictInfo]:
    """One-shot release-time predict: returns settled pose + covariance.

    Args:
        T_release: (4, 4) world-frame pose at release (the object is
            still at the gripper, falling has not started yet).
        P_release: (6, 6) covariance at release in `[v, ω]` tangent
            ordering (translation first then rotation).
        voxel_obs: scene voxel-observability grid. None → skip and
            return identity update (caller falls back to standard
            static predict).
        dyn: dynamics property (resolved via
            `utils.object_dynamics.lookup_dynamics(label)`).
        gravity: gravitational acceleration (m/s²).
        workspace_floor_z: world-frame z below which the column is
            assumed to be void.
        eps_roughness: surface-roughness coefficient in the bounce
            model (m / (m·s⁻¹)).
        max_drop_m: maximum drop distance to consider for the raycast.
        v_release_world: (3,) world-frame velocity at release; defaults
            to zero. Only the horizontal component is used.
        live_object_voxels: iterable of (x, y, z, radius) for other
            tracked objects to overlay as OCCUPIED for this query.

    Returns:
        T_land: (4, 4) predicted post-settling pose.
        P_land: (6, 6) inflated covariance.
        info: `GravityPredictInfo` with diagnostics.
    """
    T_release = np.asarray(T_release, dtype=np.float64)
    P_release = np.asarray(P_release, dtype=np.float64)
    if T_release.shape != (4, 4) or P_release.shape != (6, 6):
        raise ValueError("T_release must be (4, 4) and P_release (6, 6)")

    # Skip safely when no voxel grid is available.
    if voxel_obs is None:
        info = GravityPredictInfo(
            column_state="skipped", drop_height_m=0.0,
            surface_z=None, first_unseen_z=None,
            floor_z=workspace_floor_z,
            sigma_xy=0.0, sigma_z=0.0, sigma_yaw=0.0,
            sigma_roll=0.0, sigma_pitch=0.0,
            sigma_bounce=0.0, sigma_release_v=0.0, sigma_shape=0.0,
            landing_z=float(T_release[2, 3]),
            skipped=True, skip_reason="voxel_obs is None",
        )
        return T_release.copy(), P_release.copy(), info

    # Object bottom in world frame: T_release applied to (0, 0, -r) so
    # we raycast from below the centre of mass.
    bottom_local = np.array([0.0, 0.0, -float(dyn.radius_m), 1.0])
    p_bottom = (T_release @ bottom_local)[:3]

    cast = voxel_obs.raycast_down(
        p_bottom,
        max_distance_m=float(max_drop_m),
        floor_z=float(workspace_floor_z),
        live_object_voxels=tuple(live_object_voxels),
    )

    # Resolve drop height h, landing z, and column-driven σ_z floor.
    h: float
    sigma_z_floor: float
    landing_z: float
    if cast.column_state == "hit_occupied":
        landing_z = float(cast.surface_z) + dyn.radius_m
        h = max(0.0, float(p_bottom[2]) - float(cast.surface_z))
        sigma_z_floor = 0.0
    elif cast.column_state == "mixed_unseen":
        landing_z = float(cast.surface_z) + dyn.radius_m
        h = max(0.0, float(p_bottom[2]) - float(cast.surface_z))
        sigma_z_floor = max(0.0, (float(cast.first_unseen_z)
                                  - float(cast.surface_z)) / 2.0)
    elif cast.column_state == "all_unseen":
        # Surface is unknown. Predict midpoint of [first_unseen, floor].
        first_unseen = (float(cast.first_unseen_z)
                        if cast.first_unseen_z is not None
                        else float(p_bottom[2]))
        midpoint = 0.5 * (first_unseen + float(cast.floor_z))
        landing_z = midpoint + dyn.radius_m
        h = max(0.0, float(p_bottom[2]) - midpoint)
        sigma_z_floor = max(0.0, (first_unseen - float(cast.floor_z)) / 2.0)
    else:  # 'all_empty' — the column is observed empty all the way down.
        landing_z = float(cast.floor_z) + dyn.radius_m
        h = max(0.0, float(p_bottom[2]) - float(cast.floor_z))
        sigma_z_floor = dyn.radius_m

    # Already-settled fast path: object started on/below the surface.
    already_settled = h <= 1e-3

    # Parametric bounce/roll/shape model.
    e = float(dyn.e)
    one_minus_e = max(0.1, 1.0 - e)  # numerical guard
    v_impact = np.sqrt(max(0.0, 2.0 * gravity * h))
    sigma_bounce = eps_roughness * v_impact / one_minus_e
    sigma_shape = dyn.radius_m * shape_footprint_factor(dyn.shape)
    if v_release_world is None:
        v_release_xy_norm = 0.0
    else:
        v = np.asarray(v_release_world, dtype=np.float64).ravel()
        v_release_xy_norm = float(np.linalg.norm(v[:2]))
    t_fall = np.sqrt(max(0.0, 2.0 * h / gravity))
    sigma_release_v = 0.5 * v_release_xy_norm * t_fall

    sigma_xy = float(np.sqrt(
        sigma_bounce * sigma_bounce
        + sigma_release_v * sigma_release_v
        + sigma_shape * sigma_shape
    ))
    if already_settled:
        sigma_xy = sigma_shape  # tiny settling jitter only.
    sigma_z = float(np.sqrt(
        (0.5 * dyn.radius_m * (1.0 - e * 0.5)) ** 2
        + sigma_z_floor ** 2
    ))
    if already_settled:
        sigma_z = 0.5 * dyn.radius_m
    sigma_yaw = float(np.pi * (1.0 - np.exp(
        -h * (1.0 - e * e) / max(dyn.radius_m, 1e-3))))
    if already_settled:
        sigma_yaw = 0.05  # ~3°
    h_stable = max(landing_z - cast.floor_z, 0.01)
    sigma_roll = sigma_pitch = float(np.arctan(dyn.radius_m / h_stable))

    # Build T_land: keep xy and rotation; replace z.
    T_land = T_release.copy()
    if not already_settled:
        T_land[2, 3] = landing_z

    # Inflate covariance additively. Tangent ordering is [v, ω] per
    # pose_update/ekf_se3.py: Σ block-diag(σ_x², σ_y², σ_z², σ_roll²,
    # σ_pitch², σ_yaw²).
    extra = np.diag([
        sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2,
        sigma_roll ** 2, sigma_pitch ** 2, sigma_yaw ** 2,
    ])
    P_land = P_release + extra

    info = GravityPredictInfo(
        column_state=cast.column_state,
        drop_height_m=float(h),
        surface_z=cast.surface_z,
        first_unseen_z=cast.first_unseen_z,
        floor_z=float(cast.floor_z),
        sigma_xy=float(sigma_xy),
        sigma_z=float(sigma_z),
        sigma_yaw=float(sigma_yaw),
        sigma_roll=float(sigma_roll),
        sigma_pitch=float(sigma_pitch),
        sigma_bounce=float(sigma_bounce),
        sigma_release_v=float(sigma_release_v),
        sigma_shape=float(sigma_shape),
        landing_z=float(landing_z),
    )
    return T_land, P_land, info

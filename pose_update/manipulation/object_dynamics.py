"""Per-label dynamics property table for the gravity-aware predict step.

The dynamics property captures the parameters needed by
`pose_update/gravity_predict.py` to estimate the post-release pose
distribution: coefficient of restitution `e` (governs bounce), friction
`mu` (governs slide), shape primitive (governs resting-pose count and
footprint), and characteristic radius (governs the resting-position
spread).

Restitution numbers are ranges from agricultural / engineering
literature; defaults are conservative midpoints. A perception-side
estimator (e.g. learned from video) can supply per-detection overrides
via the optional `override` argument to `lookup_dynamics`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


_VALID_SHAPES = frozenset({"spherical", "cylindrical", "box", "irregular"})


@dataclass(frozen=True)
class ObjectDynamicsProperty:
    """Dynamics-relevant per-object property used by `gravity_predict`.

    Attributes:
        label: object class name (free-form).
        e: coefficient of restitution in [0, 1]. 0 = perfectly inelastic
            (no bounce), 1 = perfect elastic bounce.
        mu: coefficient of friction in [0, ~2].
        shape: one of {'spherical', 'cylindrical', 'box', 'irregular'}.
            Determines the shape factor in the lateral-spread model.
        radius_m: characteristic dimension (m), used for the resting-pose
            footprint and σ_z.
        mass_kg: mass; advisory, not used by the current parametric
            model but reserved for downstream physics.
    """
    label: str
    e: float
    mu: float
    shape: str
    radius_m: float
    mass_kg: float = 0.1

    def __post_init__(self) -> None:
        if not (0.0 <= self.e <= 1.0):
            raise ValueError(f"e must be in [0, 1]; got {self.e}")
        if not (0.0 <= self.mu <= 2.0):
            raise ValueError(f"mu must be in [0, 2]; got {self.mu}")
        if self.shape not in _VALID_SHAPES:
            raise ValueError(
                f"shape must be one of {sorted(_VALID_SHAPES)}; got {self.shape!r}")
        if self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive; got {self.radius_m}")
        if self.mass_kg <= 0.0:
            raise ValueError(f"mass_kg must be positive; got {self.mass_kg}")


# Conservative starting points sourced from:
#   * Apple bruise-prediction literature (e ≈ 0.20–0.30 ripe).
#   * Engineering tables for plastics, glass, wood, rubber.
#   * Friction values for typical tabletop / floor surfaces.
DEFAULT_DYNAMICS = ObjectDynamicsProperty(
    label="default", e=0.40, mu=0.50, shape="irregular", radius_m=0.05,
)

LABEL_DYNAMICS_TABLE: Mapping[str, ObjectDynamicsProperty] = {
    "apple":     ObjectDynamicsProperty("apple",     e=0.30, mu=0.55, shape="spherical",   radius_m=0.04),
    "milkbox":   ObjectDynamicsProperty("milkbox",   e=0.40, mu=0.45, shape="box",         radius_m=0.06),
    "cola":      ObjectDynamicsProperty("cola",      e=0.50, mu=0.40, shape="cylindrical", radius_m=0.04),
    "cup":       ObjectDynamicsProperty("cup",       e=0.40, mu=0.50, shape="cylindrical", radius_m=0.04),
    "pot":       ObjectDynamicsProperty("pot",       e=0.40, mu=0.50, shape="cylindrical", radius_m=0.07),
    "flowerpot": ObjectDynamicsProperty("flowerpot", e=0.30, mu=0.55, shape="cylindrical", radius_m=0.07),
}

# Footprint factor used by gravity_predict to scale the lateral spread
# by an object's resting orientation count. Spherical: settles in place
# (small footprint); irregular: any contact face can be down (full
# footprint).
SHAPE_FOOTPRINT_FACTOR: Mapping[str, float] = {
    "spherical":   0.25,
    "cylindrical": 0.50,
    "box":         0.70,
    "irregular":   1.00,
}


def lookup_dynamics(
    label: Optional[str],
    override: Optional[ObjectDynamicsProperty] = None,
) -> ObjectDynamicsProperty:
    """Resolve the dynamics property for a label, with optional override.

    Resolution order:
      1. `override` if provided (perception-side estimator hook).
      2. `LABEL_DYNAMICS_TABLE[label]` if the label is known.
      3. `DEFAULT_DYNAMICS`.
    """
    if override is not None:
        return override
    if label is None:
        return DEFAULT_DYNAMICS
    return LABEL_DYNAMICS_TABLE.get(label, DEFAULT_DYNAMICS)


def shape_footprint_factor(shape: str) -> float:
    """Return the lateral-spread shape factor (used by `gravity_predict`)."""
    return SHAPE_FOOTPRINT_FACTOR.get(shape, SHAPE_FOOTPRINT_FACTOR["irregular"])

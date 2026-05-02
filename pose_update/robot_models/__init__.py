"""Robot-model registry. Pick a `GripperGeometry` by robot name."""
from __future__ import annotations

from typing import Optional

from pose_update.manipulation.gripper_geometry import GripperGeometry

# Default URDF locations searched (first existing wins) when caller
# doesn't pass an explicit `urdf_path`.
DEFAULT_FETCH_URDF_CANDIDATES = (
    "/Volumes/External/Workspace/nus_deliver/robi_butler/resources/"
    "fetch_ext/fetch ext.urdf",
    "/Volumes/External/Workspace/nus_deliver/Real-world-manipulation/"
    "resources/fetch_ext/fetch ext.urdf",
    "/Volumes/External/Workspace/nus_deliver/robi_butler/motion_planning/"
    "resources/fetch_ext/fetch ext.urdf",
)


def create_gripper_geometry(
        robot_type: str = "fetch",
        urdf_path: Optional[str] = None,
        ) -> GripperGeometry:
    """Construct a `GripperGeometry` for the named robot.

    Args:
        robot_type: ``"fetch"`` for now. Add ``"panda"``, ``"robotiq_2f_85"``
            etc. as new modules land.
        urdf_path: optional explicit URDF file. If None, falls back to
            the per-robot DEFAULT_*_URDF_CANDIDATES list (first existing
            file wins). If no candidate exists, the geometry uses
            documented hardcoded fallback constants.

    Raises:
        ValueError: unknown ``robot_type``.
    """
    rt = robot_type.lower().strip()
    if rt == "fetch":
        from pose_update.robot_models.fetch import FetchGripperGeometry
        return FetchGripperGeometry.from_urdf(
            urdf_path=urdf_path,
            urdf_candidates=DEFAULT_FETCH_URDF_CANDIDATES)
    raise ValueError(f"Unknown robot_type {robot_type!r}. "
                     f"Supported: 'fetch'.")

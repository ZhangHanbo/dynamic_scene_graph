"""Robot-agnostic gripper geometry abstraction.

Defines the interface a robot-specific gripper model must implement so
the grasp-owner detector can reason about what's inside the gripper
without knowing the robot's identity.

The abstraction has two pieces:

* ``AABB`` — axis-aligned bounding box helper. Used both for the
  gripper's internal "where the held object lives" volume and for its
  external pad cuboids (visualisation).
* ``GripperGeometry`` — abstract base class. Concrete subclasses (one
  per robot: Fetch, Panda, Robotiq, …) live under
  ``pose_update.robot_models``.

Frame convention
----------------
All volumes returned by a ``GripperGeometry`` are expressed in the
robot's "gripper-link" frame — the frame to which ``T_bg`` (the
gripper-in-base transform) refers. Each subclass declares which link
that is via ``link_name`` so callers can sanity-check.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AABB:
    """Axis-aligned bounding box. ``mins`` and ``maxs`` are 3-vectors.

    Whatever frame the box is expressed in is the caller's
    responsibility; nothing here references one.
    """
    mins: np.ndarray
    maxs: np.ndarray

    def __post_init__(self) -> None:
        self.mins = np.asarray(self.mins, dtype=np.float64).reshape(3)
        self.maxs = np.asarray(self.maxs, dtype=np.float64).reshape(3)

    @property
    def extents(self) -> np.ndarray:
        return self.maxs - self.mins

    @property
    def volume_m3(self) -> float:
        ex = self.extents
        return float(max(ex[0], 0.0) * max(ex[1], 0.0) * max(ex[2], 0.0))

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.mins + self.maxs)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask of which rows in ``points`` (N, 3) lie
        inside this box. Empty / None inputs return an empty array."""
        if points is None or len(points) == 0:
            return np.zeros((0,), dtype=bool)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return np.all((pts >= self.mins) & (pts <= self.maxs), axis=1)

    def count_inside(self, points: np.ndarray) -> int:
        """Convenience: ``contains(points).sum()`` as a Python int."""
        return int(self.contains(points).sum())

    def corners(self) -> np.ndarray:
        """Return the 8 corner vertices of this box, shape (8, 3)."""
        x0, y0, z0 = self.mins
        x1, y1, z1 = self.maxs
        return np.array([[x0, y0, z0], [x1, y0, z0],
                         [x0, y1, z0], [x1, y1, z0],
                         [x0, y0, z1], [x1, y0, z1],
                         [x0, y1, z1], [x1, y1, z1]],
                         dtype=np.float64)


class GripperGeometry(ABC):
    """Robot-agnostic gripper model.

    A subclass commits to:

      * ``link_name`` — the URDF link to which the orchestrator's
        ``T_bg`` refers. Callers can assert this matches what the
        dataset's `ee_pose.txt` was extracted against.
      * ``state_from_joints(joints)`` — project a raw per-frame
        joints dict (e.g. ``{"l_gripper_finger_joint": 0.012, ...}``)
        into a per-robot state dict that the geometry methods consume.
      * ``inside_volume_g(state)`` — the AABB in gripper-link frame
        where a held object's surface should lie.
      * ``pad_volumes_g(state)`` — solid cuboids representing the
        gripper's own jaws (used only for visual overlays).

    Subclasses must also declare the per-axis convention as instance or
    class attributes so callers can sanity-print:

      ``slide_axis``     — finger separation direction ('x'|'y'|'z')
      ``approach_axis``  — held-object direction
      ``height_axis``    — pad-height direction
    """
    link_name: str = "gripper_link"
    slide_axis: str = "y"
    approach_axis: str = "x"
    height_axis: str = "z"
    robot_name: str = ""

    @abstractmethod
    def inside_volume_g(self, state: Dict[str, Any]) -> AABB:
        """Held-object expectation volume, in gripper-link frame."""

    @abstractmethod
    def pad_volumes_g(self, state: Dict[str, Any]) -> List[AABB]:
        """Solid pad cuboids in gripper-link frame (for overlay)."""

    @abstractmethod
    def state_from_joints(self,
                          joints: Dict[str, Any]
                          ) -> Optional[Dict[str, Any]]:
        """Per-robot state from a per-frame joints dict.

        Return ``None`` if the joints dict lacks the required fields
        (caller should fall back / skip this frame).
        """

    def describe(self) -> str:
        """Human-readable one-liner for logging."""
        return (f"{self.__class__.__name__}(robot={self.robot_name!r}, "
                f"link={self.link_name!r}, slide={self.slide_axis}, "
                f"approach={self.approach_axis}, height={self.height_axis})")

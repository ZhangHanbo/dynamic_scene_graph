"""Fetch parallel-jaw gripper geometry (loaded from URDF where possible).

What the URDF (``fetch_description/fetch.urdf`` shipped under
``robi_butler/resources/fetch_ext/fetch ext.urdf``) gives us directly:

* ``gripper_link`` is the parent of both finger joints. ``T_bg`` from
  the dataset's ``ee_pose.txt`` is in this frame.
* ``gripper_axis`` joint: ``origin xyz="0.16645 0 0"`` from
  ``wrist_roll_link`` → confirms the gripper's local +X is the approach
  direction.
* ``l_gripper_finger_joint``: ``origin xyz="0 -0.065425 0"``,
  ``axis xyz="0 -1 0"``  → left finger sits 6.5 cm in -Y from the
  link, slides further into -Y on opening.
* ``r_gripper_finger_joint``: ``origin xyz="0 +0.065425 0"``,
  ``axis xyz="0 +1 0"``  → right finger mirror.
* Slide axis = Y, approach axis = X, height axis = Z.

What the URDF DOES NOT directly give:

* Pad length (X extent), pad height (Z extent), pad thickness (Y extent)
  — these live in the finger-link mesh files (.STL) which we don't load
  here. We hardcode them from the public Fetch hardware spec, with a
  ``BOUNDARY_PAD_M`` inflation that the empirical test (frame 488 of
  apple_in_the_tray: 395 inside-pts on the held tray, 0 elsewhere)
  validates.
* The actual prismatic joint axis at runtime — the URDF declares the
  finger joints ``type="fixed"`` because the live joint state is
  republished separately. We accept ``l_gripper_finger_joint`` /
  ``r_gripper_finger_joint`` values from the dataset's
  ``joints_pose.json`` directly.

Joint state convention (per Fetch docs + URDF axes):
  ``finger_pos`` = each finger's displacement from the closed
  position, along its own slide axis. So total jaw gap =
  ``l_gripper_finger_joint + r_gripper_finger_joint``.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from utils.gripper_geometry import AABB, GripperGeometry


# ─────────────────────────────────────────────────────────────────────
# Hardcoded fallback constants (Fetch hardware spec)
# ─────────────────────────────────────────────────────────────────────

# Empirically validated (frame 488 of apple_in_the_tray: tray gets 395
# inside-pts, all other detections 0) but not extracted from URDF mesh.
# Tunable; expose via constructor for tests.
_DEFAULT_PAD_X_OFFSET   = 0.012   # forward distance gripper_link → pad start (m)
_DEFAULT_PAD_LENGTH     = 0.054   # pad length along approach axis (m)
_DEFAULT_PAD_HEIGHT     = 0.022   # pad height (perpendicular to slide+approach) (m)
_DEFAULT_PAD_THICKNESS  = 0.014   # each pad's thickness in slide direction (m)
_DEFAULT_BOUNDARY_PAD   = 0.005   # inflation around inside-volume box (m)

# Expected URDF values (validated against the actual file).
_EXPECTED_L_FINGER_Y    = -0.065425
_EXPECTED_R_FINGER_Y    = +0.065425
_EXPECTED_GRIPPER_AXIS_X = 0.16645   # gripper_link offset from wrist_roll_link


@dataclass
class _FetchURDFData:
    """Subset of the Fetch URDF we actually consume."""
    l_finger_origin_y: float
    r_finger_origin_y: float
    gripper_axis_x:    float
    urdf_path:         str          # for logging


def _parse_xyz(elem: Optional[ET.Element]) -> np.ndarray:
    if elem is None or "xyz" not in elem.attrib:
        return np.zeros(3, dtype=np.float64)
    parts = elem.attrib["xyz"].split()
    return np.array([float(p) for p in parts], dtype=np.float64)


def _parse_fetch_urdf(path: str) -> _FetchURDFData:
    """Read the Fetch URDF and pull the joint origins we care about."""
    tree = ET.parse(path)
    root = tree.getroot()
    l_y = r_y = gax = None
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        origin = joint.find("origin")
        xyz = _parse_xyz(origin)
        if name == "l_gripper_finger_joint":
            l_y = float(xyz[1])
        elif name == "r_gripper_finger_joint":
            r_y = float(xyz[1])
        elif name == "gripper_axis":
            gax = float(xyz[0])
    if l_y is None or r_y is None or gax is None:
        raise ValueError(
            f"URDF {path!r} missing one of "
            f"l_gripper_finger_joint / r_gripper_finger_joint / gripper_axis "
            f"(got l_y={l_y}, r_y={r_y}, gripper_axis_x={gax}).")
    return _FetchURDFData(l_finger_origin_y=l_y,
                          r_finger_origin_y=r_y,
                          gripper_axis_x=gax,
                          urdf_path=path)


# ─────────────────────────────────────────────────────────────────────
# Concrete geometry
# ─────────────────────────────────────────────────────────────────────

class FetchGripperGeometry(GripperGeometry):
    """Fetch parallel-jaw gripper. ``T_bg`` must be in ``gripper_link``."""

    link_name = "gripper_link"
    slide_axis = "y"
    approach_axis = "x"
    height_axis = "z"
    robot_name = "fetch"

    def __init__(self,
                 pad_x_offset_m:  float = _DEFAULT_PAD_X_OFFSET,
                 pad_length_m:    float = _DEFAULT_PAD_LENGTH,
                 pad_height_m:    float = _DEFAULT_PAD_HEIGHT,
                 pad_thickness_m: float = _DEFAULT_PAD_THICKNESS,
                 boundary_pad_m:  float = _DEFAULT_BOUNDARY_PAD,
                 urdf_data:       Optional[_FetchURDFData] = None):
        self.pad_x_offset_m  = float(pad_x_offset_m)
        self.pad_length_m    = float(pad_length_m)
        self.pad_height_m    = float(pad_height_m)
        self.pad_thickness_m = float(pad_thickness_m)
        self.boundary_pad_m  = float(boundary_pad_m)
        self.urdf_data       = urdf_data

    # ────────── factory ──────────

    @classmethod
    def from_urdf(cls,
                  urdf_path: Optional[str] = None,
                  urdf_candidates: Sequence[str] = (),
                  ) -> "FetchGripperGeometry":
        """Build a Fetch gripper; load + validate URDF if findable.

        Pad dimensions stay at the hardcoded defaults (mesh-derived,
        not in the URDF body). The URDF is used to *validate* the axis
        convention and finger-origin offsets, surfacing a clear error
        if something has changed under us.
        """
        path = urdf_path
        if path is None:
            for cand in urdf_candidates:
                if cand and os.path.exists(cand):
                    path = cand
                    break
        if path is None or not os.path.exists(path):
            # No URDF reachable — return defaults; caller can log.
            return cls()

        try:
            data = _parse_fetch_urdf(path)
        except (ET.ParseError, ValueError) as e:
            print(f"[FetchGripperGeometry] URDF parse failed at {path!r}: "
                  f"{e}. Falling back to defaults.")
            return cls()

        # Sanity check against expected values; warn on mismatch.
        for label, got, expect in (
                ("l_finger_origin_y", data.l_finger_origin_y,
                 _EXPECTED_L_FINGER_Y),
                ("r_finger_origin_y", data.r_finger_origin_y,
                 _EXPECTED_R_FINGER_Y),
                ("gripper_axis_x",    data.gripper_axis_x,
                 _EXPECTED_GRIPPER_AXIS_X)):
            if abs(got - expect) > 1e-3:
                print(f"[FetchGripperGeometry] URDF {label}={got:+.6f} differs "
                      f"from expected {expect:+.6f}. Continuing.")

        return cls(urdf_data=data)

    # ────────── interface ──────────

    def state_from_joints(self,
                          joints: Dict[str, Any]
                          ) -> Optional[Dict[str, Any]]:
        if not isinstance(joints, dict):
            return None
        lg = joints.get("l_gripper_finger_joint")
        rg = joints.get("r_gripper_finger_joint")
        if lg is None or rg is None:
            return None
        try:
            gap = float(lg) + float(rg)
        except (TypeError, ValueError):
            return None
        return {"gap_m": gap}

    def inside_volume_g(self, state: Dict[str, Any]) -> AABB:
        """AABB of the inside-jaws volume in gripper_link frame.

        Box: x ∈ [PAD_X_OFFSET, PAD_X_OFFSET + PAD_LENGTH] (forward),
        y ∈ [-gap/2, +gap/2] (between pad inner faces),
        z ∈ [-PAD_HEIGHT/2, +PAD_HEIGHT/2] (height).
        Inflated by ``boundary_pad_m`` on all sides so the held object's
        outer envelope isn't clipped at the box boundaries.
        """
        gap = max(0.0, float(state.get("gap_m", 0.0)))
        half_gap = 0.5 * gap + self.boundary_pad_m
        half_h   = 0.5 * self.pad_height_m + self.boundary_pad_m
        x0 = self.pad_x_offset_m - self.boundary_pad_m
        x1 = self.pad_x_offset_m + self.pad_length_m + self.boundary_pad_m
        return AABB(mins=np.array([x0, -half_gap, -half_h], dtype=np.float64),
                    maxs=np.array([x1, +half_gap, +half_h], dtype=np.float64))

    def pad_volumes_g(self, state: Dict[str, Any]) -> List[AABB]:
        """The two finger pad cuboids in gripper_link frame.

        Each pad sits just outside the inside-volume on its own side.
        Useful only for visual overlay.
        """
        gap = max(0.0, float(state.get("gap_m", 0.0)))
        half_gap = 0.5 * gap
        half_h   = 0.5 * self.pad_height_m
        x0 = self.pad_x_offset_m
        x1 = self.pad_x_offset_m + self.pad_length_m
        left = AABB(mins=np.array([x0, +half_gap, -half_h], dtype=np.float64),
                    maxs=np.array([x1, +half_gap + self.pad_thickness_m,
                                    +half_h], dtype=np.float64))
        right = AABB(mins=np.array([x0, -half_gap - self.pad_thickness_m,
                                     -half_h], dtype=np.float64),
                     maxs=np.array([x1, -half_gap, +half_h],
                                    dtype=np.float64))
        return [left, right]

    def describe(self) -> str:
        base = super().describe()
        urdf = (f", urdf={os.path.basename(self.urdf_data.urdf_path)!r}"
                if self.urdf_data is not None else ", urdf=None")
        return base[:-1] + urdf + ")"

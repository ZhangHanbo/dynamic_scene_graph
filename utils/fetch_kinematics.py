"""Fetch-robot kinematic helpers for the base -> head_camera_rgb_optical chain.

The pipeline computes `T_bc(t) = T_{base_link -> head_camera_rgb_optical_frame}`
from per-frame TF data extracted from the rosbag (`pose_txt/T_bc.txt`) --
this is the canonical, calibration-correct source.

This module provides only the SE(3) helpers and the tf-edge composer that
the *extractor* uses to walk the chain from /tf + /tf_static. Run-time
loading happens in `tests/visualize_ekf_tracking.py` via `_load_T_bc`,
which reads pre-extracted poses directly without needing this module.

Why we don't ship a URDF-hardcoded forward kinematics: the published
/tf and /tf_static include factory calibration corrections (small
rotation/translation biases) that the URDF doesn't encode. Reading /tf
at extraction time picks them up automatically. Hardcoded URDF values
were measured to be off by 5--10 mm in translation and ~5 deg in
rotation versus published /tf on the apple_in_the_tray bag.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R_


# ─────────────────────────────────────────────────────────────────────
# SE(3) helpers (thin wrappers around scipy.spatial.transform.Rotation)
# ─────────────────────────────────────────────────────────────────────

def quat_to_R(q: Tuple[float, float, float, float]) -> np.ndarray:
    """ROS quaternion convention: q = (x, y, z, w)."""
    return R_.from_quat([float(q[0]), float(q[1]),
                         float(q[2]), float(q[3])]).as_matrix()


def make_SE3(t: Tuple[float, float, float],
             q: Tuple[float, float, float, float]) -> np.ndarray:
    """Compose a 4x4 SE(3) from a translation and a (x, y, z, w) quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_R(q)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


# ─────────────────────────────────────────────────────────────────────
# tf-chain composer (used by the extractor)
# ─────────────────────────────────────────────────────────────────────

# The full chain `base_link -> head_camera_rgb_optical_frame` for Fetch.
# 3 dynamic + 3 static edges; the dynamic ones depend on torso_lift_joint,
# head_pan_joint, head_tilt_joint and arrive on /tf, the static on
# /tf_static.
FETCH_BASE_TO_OPTICAL_CHAIN = (
    ("base_link", "torso_lift_link"),                          # dynamic (torso)
    ("torso_lift_link", "head_pan_link"),                      # dynamic (pan)
    ("head_pan_link", "head_tilt_link"),                       # dynamic (tilt)
    ("head_tilt_link", "head_camera_link"),                    # static
    ("head_camera_link", "head_camera_rgb_frame"),             # static
    ("head_camera_rgb_frame", "head_camera_rgb_optical_frame"),# static
)


def compose_chain(tf_edges: Mapping[Tuple[str, str], np.ndarray],
                   chain: Tuple[Tuple[str, str], ...]
                          = FETCH_BASE_TO_OPTICAL_CHAIN) -> np.ndarray:
    """Compose a kinematic chain from per-edge SE(3) transforms.

    `tf_edges[(parent, child)]` is the SE(3) that lifts a point in
    `child` to `parent` (i.e. `p_parent = T @ p_child`). This is the
    standard ROS tf convention.

    Raises KeyError if any edge is missing.
    """
    T = np.eye(4, dtype=np.float64)
    for edge in chain:
        if edge not in tf_edges:
            raise KeyError(f"missing tf edge {edge[0]} -> {edge[1]}")
        T = T @ np.asarray(tf_edges[edge], dtype=np.float64)
    return T

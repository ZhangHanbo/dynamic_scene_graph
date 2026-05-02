"""Project the gripper geometry onto an RGB frame for visual sanity.

Visualises three boxes in image space:

  * **inside-jaws** (green) — the AABB ``gripper.inside_volume_g(state)``.
    A held object's surface points should fall inside this in 3D.
  * **left finger pad** (cyan) — `gripper.pad_volumes_g(state)[0]`.
  * **right finger pad** (orange) — `gripper.pad_volumes_g(state)[1]`.

Usage:
    python tools/render_gripper_overlay.py FRAME [--trajectory NAME] [--out PATH]

    python tools/render_gripper_overlay.py 488                     # default trajectory
    python tools/render_gripper_overlay.py 487 488 495 600 998     # multiple frames

Inputs come from
``datasets/<trajectory>/`` (rgb, depth, pose_txt/{amcl_pose,T_bc,ee_pose}.txt,
pose_txt/joints_pose.json) — the same layout the EKF visualizer uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pose_update.manipulation.gripper_geometry import AABB, GripperGeometry  # noqa: E402
from pose_update.robot_models import create_gripper_geometry      # noqa: E402


# 12 box edges as pairs of corner indices (matches AABB.corners() ordering).
_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3),     # bottom face
          (4, 5), (4, 6), (5, 7), (6, 7),     # top face
          (0, 4), (1, 5), (2, 6), (3, 7)]     # verticals


def _load_8col(path: str) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            try:
                idx = int(arr[0])
                t = np.array([float(x) for x in arr[1:4]])
                q = np.array([float(x) for x in arr[4:]])
            except ValueError:
                continue
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(q).as_matrix()
            T[:3, 3] = t
            out[idx] = T
    return out


def _project_box(box: AABB,
                  T_wg: np.ndarray,
                  T_wb: np.ndarray,
                  T_bc: np.ndarray,
                  K: np.ndarray,
                  ) -> np.ndarray:
    """Project 8 corners of `box` (gripper-link frame) to image pixels.

    Returns (8, 2) integer pixel coordinates. Pixels with z<=0 in
    camera frame become NaN so the line draw can skip them.
    """
    corners_g = box.corners()                                # (8, 3)
    homog = np.hstack([corners_g, np.ones((8, 1))])
    # gripper → world → camera
    pts_w = (T_wg @ homog.T).T[:, :3]
    homog_w = np.hstack([pts_w, np.ones((8, 1))])
    T_cb = np.linalg.inv(T_bc)
    T_bw = np.linalg.inv(T_wb)
    pts_cam = (T_cb @ T_bw @ homog_w.T).T[:, :3]
    out = np.full((8, 2), np.nan, dtype=np.float64)
    z = pts_cam[:, 2]
    safe = z > 1e-3
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    out[safe, 0] = fx * pts_cam[safe, 0] / z[safe] + cx
    out[safe, 1] = fy * pts_cam[safe, 1] / z[safe] + cy
    return out


def _draw_box(img: np.ndarray, pts_2d: np.ndarray,
              color: Tuple[int, int, int], thickness: int = 2) -> None:
    """Draw the 12 edges of a box on `img`. Skip edges with NaN endpoints."""
    H, W = img.shape[:2]
    for a, b in _EDGES:
        pa, pb = pts_2d[a], pts_2d[b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        x1, y1 = int(round(pa[0])), int(round(pa[1]))
        x2, y2 = int(round(pb[0])), int(round(pb[1]))
        # Only draw if at least one endpoint is on-screen (clip otherwise).
        if max(x1, x2) < 0 or min(x1, x2) >= W:
            continue
        if max(y1, y2) < 0 or min(y1, y2) >= H:
            continue
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def render_overlay(frame: int,
                    trajectory: str = "apple_in_the_tray",
                    out_path: str = None,
                    gripper: GripperGeometry = None,
                    K: np.ndarray = None) -> str:
    """Render a single frame with the gripper boxes overlaid; return out_path."""
    ds_root = f"/Volumes/External/Workspace/datasets/{trajectory}"
    rgb_path = f"{ds_root}/rgb/rgb_{frame:06d}.png"
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(rgb_path)
    img = cv2.imread(rgb_path)
    if img is None:
        raise RuntimeError(f"cv2.imread failed on {rgb_path}")

    amcl  = _load_8col(f"{ds_root}/pose_txt/amcl_pose.txt")
    ee    = _load_8col(f"{ds_root}/pose_txt/ee_pose.txt")
    tbc   = _load_8col(f"{ds_root}/pose_txt/T_bc.txt")
    with open(f"{ds_root}/pose_txt/joints_pose.json") as f:
        joints = json.load(f)

    if frame not in amcl or frame not in ee or frame not in tbc:
        raise KeyError(f"frame {frame} missing in one of pose files")
    if K is None:
        K = np.array([[554.3827, 0.0, 320.5],
                      [0.0, 554.3827, 240.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    if gripper is None:
        gripper = create_gripper_geometry("fetch")

    T_wb = amcl[frame]
    T_bg = ee[frame]
    T_bc = tbc[frame]
    T_wg = T_wb @ T_bg

    state = gripper.state_from_joints(joints[f"{frame:06d}"])
    if state is None:
        raise RuntimeError(f"gripper.state_from_joints returned None for frame {frame}")

    inside_box = gripper.inside_volume_g(state)
    pad_left, pad_right = gripper.pad_volumes_g(state)

    # Project + draw. (BGR for cv2.)
    pts_inside = _project_box(inside_box, T_wg, T_wb, T_bc, K)
    pts_left   = _project_box(pad_left,   T_wg, T_wb, T_bc, K)
    pts_right  = _project_box(pad_right,  T_wg, T_wb, T_bc, K)
    _draw_box(img, pts_left,   (255, 200,   0), 2)   # cyan-ish
    _draw_box(img, pts_right,  (  0, 165, 255), 2)   # orange
    _draw_box(img, pts_inside, ( 50, 220,  50), 2)   # green

    # Caption
    gap_cm = state.get("gap_m", 0.0) * 100.0
    cap = (f"{gripper.robot_name} | frame {frame} | gap={gap_cm:.1f} cm | "
           f"green=inside-jaws  cyan=L pad  orange=R pad")
    (tw, th), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (4, 4), (8 + tw, 12 + th), (0, 0, 0), -1)
    cv2.putText(img, cap, (8, 4 + th + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if out_path is None:
        out_dir = os.path.join(ROOT, "tests", "out")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gripper_overlay_{frame:06d}.png")
    cv2.imwrite(out_path, img)
    return out_path


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", nargs="+", type=int,
                    help="Frame indices to render (one or more).")
    ap.add_argument("--trajectory", default="apple_in_the_tray")
    ap.add_argument("--out-dir", default=None,
                    help="Directory to write into (default: tests/out/).")
    args = ap.parse_args(argv)

    gripper = create_gripper_geometry("fetch")
    print(f"[overlay] {gripper.describe()}")
    K = np.array([[554.3827, 0.0, 320.5],
                  [0.0, 554.3827, 240.5],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    for fr in args.frames:
        out = None
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out = os.path.join(args.out_dir, f"gripper_overlay_{fr:06d}.png")
        path = render_overlay(fr, trajectory=args.trajectory,
                               out_path=out, gripper=gripper, K=K)
        print(f"[overlay] frame {fr} → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

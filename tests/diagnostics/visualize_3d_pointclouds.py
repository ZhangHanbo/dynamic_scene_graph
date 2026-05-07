#!/usr/bin/env python3
"""
Visualize recovered 3D object point clouds in the world frame.

For each frame and each detected object, back-project the masked depth
pixels using the camera intrinsics K, then transform to world via the
SLAM camera pose. Accumulate across the whole trajectory, color-coded
per object id. Subsample aggressively to keep the output compact.

Produces:
  * Three orthographic views: top-down (XY), front (XZ), side (YZ).
  * Early (pre-grasp) / mid (holding) / late (post-release) small-
    multiples so divergence during transport is visible.
  * Per-object centroid trajectories overlaid on the point clouds.

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python tests/visualize_3d_pointclouds.py [--step 2]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

DATA_ROOT = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects", "apple_bowl_2"
)

K = np.array([
    [554.3827, 0, 320.5],
    [0, 554.3827, 240.5],
    [0, 0, 1],
], dtype=np.float64)


# Per-id colors (RGB for matplotlib, 0-1)
_PALETTE = np.array([
    [0.18, 0.80, 0.44],
    [0.90, 0.30, 0.23],
    [0.20, 0.60, 0.86],
    [0.95, 0.77, 0.06],
    [0.61, 0.35, 0.71],
    [0.90, 0.49, 0.13],
    [0.10, 0.74, 0.61],
    [0.93, 0.44, 0.39],
    [0.36, 0.68, 0.89],
    [0.96, 0.82, 0.25],
])


def _color(oid: int):
    return _PALETTE[oid % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────

def _load_pose_txt(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            a = line.strip().split()
            if len(a) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = map(float, a)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses


def _load_detections(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    out = []
    for det in data.get("detections", []):
        mask_b64 = det.get("mask", "")
        if not mask_b64:
            continue
        mask_bytes = base64.b64decode(mask_b64)
        mask = (np.array(Image.open(BytesIO(mask_bytes)).convert("L")) > 128)
        out.append({
            "id": det.get("object_id"),
            "label": det.get("label", "unknown"),
            "mask": mask.astype(np.uint8),
            "score": float(det.get("score", 0.0)),
        })
    return out


def _backproject_masked(mask: np.ndarray, depth: np.ndarray,
                         K: np.ndarray, sample_step: int = 6
                         ) -> np.ndarray:
    """Back-project masked depth pixels to 3D in camera frame.

    Subsamples by `sample_step` to keep the cloud small.
    """
    # Apply mask + subsample
    H, W = depth.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # Subsample
    xs = xs[::sample_step]
    ys = ys[::sample_step]
    d = depth[ys, xs]
    valid = (d > 0.1) & (d < 5.0) & np.isfinite(d)
    xs, ys, d = xs[valid], ys[valid], d[valid]
    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * d / fx
    Y = (ys - cy) * d / fy
    Z = d
    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
    return pts


def _camera_to_world(pts_cam: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
    """Transform (N,3) points from camera to world frame."""
    if pts_cam.shape[0] == 0:
        return pts_cam
    pts_h = np.concatenate(
        [pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
    pts_w = (T_cw @ pts_h.T).T
    return pts_w[:, :3]


# ─────────────────────────────────────────────────────────────────────
# Main accumulation
# ─────────────────────────────────────────────────────────────────────

def run(step: int = 2,
        max_frames: int = 328,
        sample_step: int = 6
        ) -> Tuple[Dict, Dict, List]:
    """Accumulate world-frame points per (object_id, frame_bucket).

    Returns:
      per_id_points:     dict[oid] → list of (frame_idx, (N,3) points)
      per_id_centroids:  dict[oid] → list of (frame_idx, (3,) centroid in world)
      cam_positions:     list of (frame_idx, (3,) camera position in world)
    """
    cam_poses = _load_pose_txt(os.path.join(DATA_ROOT, "pose_txt",
                                             "camera_pose.txt"))

    per_id_points = defaultdict(list)
    per_id_centroids = defaultdict(list)
    cam_positions = []

    n = min(max_frames, len(cam_poses))
    print(f"Scanning {n} frames, step={step}, sample_step={sample_step}")

    for idx in range(0, n, step):
        rgb_p = os.path.join(DATA_ROOT, "rgb", f"rgb_{idx:06d}.png")
        dep_p = os.path.join(DATA_ROOT, "depth", f"depth_{idx:06d}.npy")
        det_p = os.path.join(DATA_ROOT, "detection_h",
                              f"detection_{idx:06d}_final.json")
        if not (os.path.exists(rgb_p) and os.path.exists(dep_p)
                and os.path.exists(det_p)):
            continue

        depth = np.load(dep_p).astype(np.float32)
        detections = _load_detections(det_p)
        T_cw = cam_poses[idx]
        cam_positions.append((idx, T_cw[:3, 3].copy()))

        for det in detections:
            oid = det["id"]
            if oid is None:
                continue
            pts_cam = _backproject_masked(det["mask"], depth, K, sample_step)
            if pts_cam.shape[0] == 0:
                continue
            pts_world = _camera_to_world(pts_cam, T_cw)
            per_id_points[oid].append((idx, pts_world))
            per_id_centroids[oid].append((idx, pts_world.mean(axis=0)))

    total_pts = sum(p.shape[0] for hist in per_id_points.values()
                    for _, p in hist)
    print(f"Accumulated {total_pts} points across {len(per_id_points)} objects")
    return per_id_points, per_id_centroids, cam_positions


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def _ortho_scatter(ax, per_id_points, per_id_centroids, cam_positions,
                    axes_pair: Tuple[int, int], title: str,
                    frame_filter=None,
                    point_size: float = 0.8,
                    point_alpha: float = 0.25):
    """Scatter plot on one orthographic projection (e.g., XY top-down).

    `axes_pair`: tuple of (x-axis-index, y-axis-index) into (X=0, Y=1, Z=2).
    `frame_filter`: callable frame_idx → bool; only draw points from
                    frames where this returns True.
    """
    a, b = axes_pair
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("xyz"[a] + " (m)")
    ax.set_ylabel("xyz"[b] + " (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal")

    # Camera trajectory (gray line)
    if cam_positions:
        cam_pts = np.stack([p for _, p in cam_positions])
        if frame_filter is not None:
            keep = np.array([frame_filter(idx) for idx, _ in cam_positions])
            if keep.any():
                cam_pts = cam_pts[keep]
        if cam_pts.shape[0] > 0:
            ax.plot(cam_pts[:, a], cam_pts[:, b],
                    color="0.5", linewidth=1.0, alpha=0.6,
                    label="camera", zorder=1)

    # Per-object points + centroid trail
    ordered = sorted(per_id_points.items(), key=lambda kv: kv[0])
    for oid, hist in ordered:
        color = _color(oid)
        all_pts = []
        for frame_idx, pts in hist:
            if frame_filter is not None and not frame_filter(frame_idx):
                continue
            all_pts.append(pts)
        if not all_pts:
            continue
        pts = np.concatenate(all_pts, axis=0)
        ax.scatter(pts[:, a], pts[:, b],
                   c=[color], s=point_size, alpha=point_alpha,
                   linewidths=0, zorder=2, rasterized=True)

        # Centroid trail (brighter, on top)
        cents = per_id_centroids.get(oid, [])
        if frame_filter is not None:
            cents = [(f, c) for f, c in cents if frame_filter(f)]
        if cents:
            cent_arr = np.stack([c for _, c in cents])
            ax.plot(cent_arr[:, a], cent_arr[:, b],
                    color=color, linewidth=1.6, alpha=0.9,
                    label=f"[{oid}]", zorder=3)

    ax.legend(loc="best", fontsize=7, framealpha=0.85, ncol=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=2,
                        help="Process every N-th frame")
    parser.add_argument("--max-frames", type=int, default=328)
    parser.add_argument("--sample-step", type=int, default=6,
                        help="Subsample pixels within each mask")
    parser.add_argument("--output", default=os.path.join(
        SCENEREP_ROOT, "tests", "vis_pointclouds"))
    parser.add_argument("--point-size", type=float, default=0.8)
    parser.add_argument("--point-alpha", type=float, default=0.20)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    per_id_points, per_id_centroids, cam_positions = run(
        step=args.step, max_frames=args.max_frames,
        sample_step=args.sample_step,
    )

    # ── Per-object summary table ────────────────────────────────────
    print("\n=== Per-object point-cloud stats ===")
    print(f"{'id':<4} {'count':>7}  {'#frames':>8}  "
          f"{'x range':>16}  {'y range':>16}  {'z range':>16}")
    for oid, hist in sorted(per_id_points.items()):
        if not hist:
            continue
        allp = np.concatenate([p for _, p in hist], axis=0)
        print(f"{oid:<4} {allp.shape[0]:>7}  {len(hist):>8}  "
              f"[{allp[:,0].min():6.2f},{allp[:,0].max():6.2f}]  "
              f"[{allp[:,1].min():6.2f},{allp[:,1].max():6.2f}]  "
              f"[{allp[:,2].min():6.2f},{allp[:,2].max():6.2f}]")

    # ── Figure 1: three orthographic views of the whole trajectory ───
    fig = plt.figure(figsize=(18, 6))
    ax_xy = fig.add_subplot(1, 3, 1)
    ax_xz = fig.add_subplot(1, 3, 2)
    ax_yz = fig.add_subplot(1, 3, 3)
    _ortho_scatter(ax_xy, per_id_points, per_id_centroids, cam_positions,
                   (0, 1), "Top-down (XY)",
                   point_size=args.point_size, point_alpha=args.point_alpha)
    _ortho_scatter(ax_xz, per_id_points, per_id_centroids, cam_positions,
                   (0, 2), "Front (XZ)",
                   point_size=args.point_size, point_alpha=args.point_alpha)
    _ortho_scatter(ax_yz, per_id_points, per_id_centroids, cam_positions,
                   (1, 2), "Side (YZ)",
                   point_size=args.point_size, point_alpha=args.point_alpha)
    fig.suptitle("Recovered 3D object point clouds in world frame "
                 "(all frames)", fontsize=13)
    fig.tight_layout()
    out1 = os.path.join(args.output, "all_frames_ortho.png")
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out1}")
    plt.close(fig)

    # ── Figure 2: small multiples by manipulation phase ──────────────
    # Frames 0-43: pre-grasp; 44-310: grasping+holding+release; 311+: post
    def pre_filter(idx):  return idx < 44
    def mid_filter(idx):  return 44 <= idx < 314
    def post_filter(idx): return idx >= 314

    fig = plt.figure(figsize=(18, 12))
    titles = [
        ("Pre-grasp (frames 0-43)", pre_filter),
        ("Transport (frames 44-313)", mid_filter),
        ("Post-release (frames 314+)", post_filter),
    ]
    for row, (title, filt) in enumerate(titles):
        for col, (axes_pair, view_name) in enumerate(
                [((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")]):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1)
            _ortho_scatter(ax, per_id_points, per_id_centroids, cam_positions,
                           axes_pair, f"{title} — {view_name}",
                           frame_filter=filt,
                           point_size=args.point_size,
                           point_alpha=args.point_alpha)
    fig.suptitle("3D point clouds by manipulation phase", fontsize=14)
    fig.tight_layout()
    out2 = os.path.join(args.output, "by_phase_ortho.png")
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig)

    # ── Figure 3: zoom on apple (4) vs bowl (3) XY trajectory ────────
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Bowl [3] vs Apple [4] — centroid trails + point clouds (XY)",
                 fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    for oid in (3, 4):
        hist = per_id_points.get(oid, [])
        if not hist:
            continue
        col = _color(oid)
        allp = np.concatenate([p for _, p in hist], axis=0)
        ax.scatter(allp[:, 0], allp[:, 1],
                   c=[col], s=2, alpha=0.12, linewidths=0,
                   rasterized=True)
        cents = per_id_centroids[oid]
        cent_arr = np.stack([c for _, c in cents])
        ax.plot(cent_arr[:, 0], cent_arr[:, 1],
                color=col, linewidth=2.0, alpha=0.95,
                label=f"[{oid}] centroid trail",
                marker=".", markersize=3)

    # Show camera trajectory too
    if cam_positions:
        cam_pts = np.stack([p for _, p in cam_positions])
        ax.plot(cam_pts[:, 0], cam_pts[:, 1], color="0.4",
                linewidth=1.0, alpha=0.7, label="camera")

    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out3 = os.path.join(args.output, "bowl_vs_apple_zoom.png")
    fig.savefig(out3, dpi=140, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Heuristic tracker (TSDF + ID-association) on a synthetic 5-frame trajectory.

Generates a single static "cube" rendered as a constant mask + depth across
frames, and feeds it to ``heuristic_tracker.ObjectTracker``. Prints the
tracked object's pose at each frame.

Run from the repo root:
    python examples/heuristic_offline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from heuristic_tracker import ObjectTracker, FrameDetections


def synth_frame(H=120, W=160, depth_value=0.6):
    """One synthetic RGB-D frame with a single object mask in the centre."""
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    mask[40:80, 60:100] = True
    rgb[mask] = (200, 50, 50)
    depth[mask] = depth_value
    bbox = np.array([60.0, 40.0, 100.0, 80.0])  # x1,y1,x2,y2
    return rgb, depth, mask, bbox


def main():
    fx, fy, cx, cy = 200.0, 200.0, 80.0, 60.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    tracker = ObjectTracker(K=K, voxel_size=0.005)

    # Camera moves 1 cm forward each frame (T_cw is camera→world).
    T_cw = np.eye(4, dtype=np.float64)

    print("frame  oid  T_wo translation (m)")
    for k in range(5):
        rgb, depth, mask, bbox = synth_frame()
        dets = FrameDetections(
            labels=["cube"],
            scores=np.array([0.9], dtype=np.float32),
            masks=[mask],
            bboxes=bbox.reshape(1, 4),
        )
        tracked = tracker.update(dets, rgb, depth, T_cw, integrate=True)
        for tr in tracked:
            t = tr.pose[:3, 3]
            print(f"  {k:3d}  {tr.id:3d}  ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})")
        T_cw[2, 3] += 0.01  # move camera +1 cm in world-z


if __name__ == "__main__":
    main()

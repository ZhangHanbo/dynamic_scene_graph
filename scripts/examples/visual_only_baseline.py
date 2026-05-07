#!/usr/bin/env python3
"""Visual-only ICP baseline on a synthetic 5-frame trajectory.

Drives ``baselines.VisualOnlyTracker`` (no filter, no proprioception) with
a single static "cube" mask + depth and prints the world-frame pose per
frame.

Run from the repo root:
    python examples/visual_only_baseline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from baselines import VisualOnlyTracker


def synth_depth_mask(H=120, W=160, depth_value=0.6):
    depth = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    mask[40:80, 60:100] = True
    depth[mask] = depth_value
    return depth, mask


def main():
    fx, fy, cx, cy = 200.0, 200.0, 80.0, 60.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    tracker = VisualOnlyTracker(K=K, mode="last_frame")

    T_cw = np.eye(4, dtype=np.float64)
    print("frame  oid  T_wo translation (m)")
    for k in range(5):
        depth, mask = synth_depth_mask()
        T_wo, accepted, fitness, rmse = tracker.update(
            oid=0, mask=mask, depth=depth, T_cw=T_cw)
        if T_wo is None:
            print(f"  {k:3d}    0  (no observation accepted)")
        else:
            t = T_wo[:3, 3]
            ok = "acc" if accepted else "rej"
            print(f"  {k:3d}    0  ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) "
                  f"[{ok}, fit={fitness:.2f}, rmse={rmse*1000:.1f}mm]")
        T_cw[2, 3] += 0.01


if __name__ == "__main__":
    main()

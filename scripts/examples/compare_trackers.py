#!/usr/bin/env python3
"""Heuristic + EKF trackers side-by-side on the same synthetic trajectory.

Reuses the inline synthetic data from ``heuristic_offline.py`` and
``ekf_offline.py`` and prints the world-frame translation reported by
each tracker at frames 0, 2, 4 so you can eyeball any drift / disagreement.

Run from the repo root:
    python examples/compare_trackers.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from heuristic_tracker import ObjectTracker, FrameDetections
from ekf_tracker import TwoTierOrchestratorGaussian
from utils.slam_interface import PassThroughSlam


H, W = 120, 160
fx, fy, cx, cy = 200.0, 200.0, 80.0, 60.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def synth_frame():
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    mask[40:80, 60:100] = True
    rgb[mask] = (200, 50, 50)
    depth[mask] = 0.6
    bbox = np.array([60.0, 40.0, 100.0, 80.0])
    return rgb, depth, mask, bbox


def run_heuristic(n_frames: int):
    tracker = ObjectTracker(K=K, voxel_size=0.005)
    T_cw = np.eye(4, dtype=np.float64)
    poses_per_frame = []
    for _ in range(n_frames):
        rgb, depth, mask, bbox = synth_frame()
        dets = FrameDetections(labels=["cube"],
                               scores=np.array([0.9], dtype=np.float32),
                               masks=[mask], bboxes=bbox.reshape(1, 4))
        tracked = tracker.update(dets, rgb, depth, T_cw, integrate=False)
        poses_per_frame.append(tracked[0].pose if tracked else None)
        T_cw[2, 3] += 0.01
    return poses_per_frame


def run_ekf(n_frames: int):
    poses = [np.eye(4) for _ in range(n_frames)]
    slam = PassThroughSlam(poses=poses,
                           default_cov=np.diag([1e-6] * 3 + [1e-6] * 3))
    orch = TwoTierOrchestratorGaussian(slam_backend=slam, T_bc=np.eye(4))
    T_co = np.eye(4)
    T_co[:3, 3] = [0.0, 0.0, 0.6]
    poses_per_frame = []
    for _ in range(n_frames):
        rgb, depth, mask, _ = synth_frame()
        det = {"id": 0, "label": "cube", "mask": mask.astype(np.uint8),
               "score": 0.9, "T_co": T_co.copy(),
               "R_icp": np.diag([1e-4] * 6), "fitness": 0.92, "rmse": 0.004}
        orch.step(rgb=rgb, depth=depth, detections=[det],
                  gripper_state={"phase": "idle", "held_obj_id": None})
        oids = sorted(orch.state.objects.keys())
        if oids:
            pe = orch.state.collapsed_object_world(oids[0])
            poses_per_frame.append(pe.T)
        else:
            poses_per_frame.append(None)
    return poses_per_frame


def fmt(pose):
    if pose is None:
        return "(no track)"
    t = pose[:3, 3]
    return f"({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})"


def main():
    n = 5
    heur = run_heuristic(n)
    ekf = run_ekf(n)
    print(f"{'frame':>5}  {'heuristic T_wo (m)':>30}  {'ekf T_wo (m)':>30}")
    for k in (0, 2, 4):
        print(f"{k:5d}  {fmt(heur[k]):>30}  {fmt(ekf[k]):>30}")


if __name__ == "__main__":
    main()

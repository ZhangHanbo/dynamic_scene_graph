#!/usr/bin/env python3
"""EKF tracker (Gaussian backend) on a synthetic 5-frame trajectory.

Drives ``ekf_tracker.TwoTierOrchestratorGaussian`` with hand-crafted
detection dicts (the orchestrator's expected ``{id, label, mask, score,
T_co, R_icp, fitness, rmse}`` shape) and prints the world-frame pose
per frame.

Run from the repo root:
    python examples/ekf_offline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from ekf_tracker import TwoTierOrchestratorGaussian
from utils.slam_interface import PassThroughSlam


def synth_rgb_depth(H=120, W=160):
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[40:80, 60:100] = 1
    rgb[mask.astype(bool)] = (200, 50, 50)
    depth[mask.astype(bool)] = 0.6
    return rgb, depth, mask


def make_detection(obj_id: int, T_co: np.ndarray, mask: np.ndarray):
    return {
        "id": obj_id,
        "label": "cube",
        "mask": mask,
        "score": 0.9,
        "T_co": T_co,
        "R_icp": np.diag([1e-4] * 6),
        "fitness": 0.92,
        "rmse": 0.004,
    }


def main():
    n_frames = 5
    # Static base over n frames, identity SLAM pose.
    poses = [np.eye(4) for _ in range(n_frames)]
    slam = PassThroughSlam(poses=poses,
                           default_cov=np.diag([1e-6] * 3 + [1e-6] * 3))

    orch = TwoTierOrchestratorGaussian(slam_backend=slam, T_bc=np.eye(4))

    # Object sits 60 cm in front of the camera, identity orientation.
    T_co_base = np.eye(4)
    T_co_base[:3, 3] = [0.0, 0.0, 0.6]

    print("frame  oid  T_wo translation (m)")
    for k in range(n_frames):
        rgb, depth, mask = synth_rgb_depth()
        dets = [make_detection(0, T_co_base.copy(), mask)]
        orch.step(rgb=rgb, depth=depth, detections=dets,
                  gripper_state={"phase": "idle", "held_obj_id": None})
        for oid in sorted(orch.state.objects.keys()):
            pe = orch.state.collapsed_object_world(oid)
            t = pe.T[:3, 3]
            print(f"  {k:3d}  {oid:3d}  ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})")


if __name__ == "__main__":
    main()

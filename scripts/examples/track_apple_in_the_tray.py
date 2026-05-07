"""Minimal end-to-end example using the Gaussian-EKF tracker.

Loads the cached `apple_in_the_tray` trajectory's RGB + depth +
detections + SLAM/EE poses, drives the
``GaussianEkfTracker`` for a few frames, and prints the per-frame
held-set + relation graph. Demonstrates the public API surface
post-refactor:

    ekf_tracker.relations.relation_orchestrator.RelationOrchestrator
    utils.gripper_state.GripperPhaseTracker
    ekf_tracker.manipulation.grasp_owner_detector.GraspOwnerDetector
    ekf_tracker.relations.relation_utils.expand_held_with_relations

No visualization / matplotlib — just the algorithm. For the full
visualization pipeline see ``tests/visualize_ekf_tracking.py``.

Run from the repo root:
    EKF_VIZ_RELATION_BACKEND=llm python examples/track_apple_in_the_tray.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Make repo root importable.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.gripper_state import GripperPhaseTracker
from ekf_tracker.manipulation.grasp_owner_detector import GraspOwnerDetector
from ekf_tracker.relations.relation_orchestrator import RelationOrchestrator
from ekf_tracker.relations.relation_utils import (
    RelationTriggerConfig,
    expand_held_with_relations,
)
from utils.robot_models import create_gripper_geometry

# Reuse the visualization-driver's data loaders. The driver itself is
# heavy because of the matplotlib panels; we import only the loaders.
sys.path.insert(0, str(ROOT / "tests"))
import visualize_ekf_tracking as viz  # noqa: E402


def main():
    traj = "apple_in_the_tray"
    ds_root = ROOT / "datasets" / traj
    viz_root = ROOT / "tests" / "visualization_pipeline" / traj
    det_dir = viz_root / "perception" / "detection_h"
    rgb_dir = ds_root / "rgb"
    depth_dir = ds_root / "depth"

    slam = viz._load_amcl_poses(str(ds_root / "pose_txt" / "amcl_pose.txt"))
    T_bc_map = viz._load_T_bc_poses(str(ds_root / "pose_txt" / "T_bc.txt"))
    T_bg_map = viz._load_ee_poses(str(ds_root / "pose_txt" / "ee_pose.txt"))
    width_map = viz._load_gripper_widths(
        str(ds_root / "pose_txt" / "joints_pose.json"))

    cfg = viz.BernoulliConfig()
    tracker = viz.GaussianEkfTracker(K=viz.K_DEFAULT, bernoulli_cfg=cfg)

    grip = GripperPhaseTracker(
        detector=GraspOwnerDetector(create_gripper_geometry("fetch")))
    relations = RelationOrchestrator(
        backend=os.environ.get("EKF_VIZ_RELATION_BACKEND", "none"),
        cache_dir=str(viz_root / "relation_cache"),
        trigger_cfg=RelationTriggerConfig(relation_every_n_frames=90))

    for idx in range(488, 525):
        rgb = viz._load_rgb(str(rgb_dir / f"rgb_{idx:06d}.png"))
        depth = viz._load_depth(str(depth_dir / f"depth_{idx:06d}.npy"))
        if rgb is None or depth is None:
            continue
        detections = viz._load_detection_json(
            str(det_dir / f"detection_{idx:06d}_final.json"))
        T_wb = slam[idx]
        T_bc = T_bc_map.get(idx) if T_bc_map else None
        T_bg = T_bg_map.get(idx) if T_bg_map else None
        width = width_map.get(idx) if width_map else None

        gs = grip.step(width=width,
                       tracker_state=viz.GaussianEkfTrackerState(tracker),
                       T_wb=T_wb, T_bg=T_bg,
                       detections=detections, depth=depth, K=viz.K_DEFAULT,
                       T_bc=T_bc,
                       live_oids={int(o) for o in tracker.object_labels})
        held_seed = gs.get("held_obj_id")

        # Build det_to_oid for the relation pipeline.
        tau_to_oid = {int(t): int(o)
                      for o, t in tracker.sam2_tau.items() if t is not None}
        det_to_oid = {di: tau_to_oid[int(d.get("id"))]
                      for di, d in enumerate(detections)
                      if d.get("id") is not None
                      and int(d["id"]) in tau_to_oid}
        live_oids = {int(o) for o in tracker.object_labels}
        relations.maybe_update(
            frame=idx, rgb=rgb, detections=detections,
            det_to_oid=det_to_oid,
            current_phase=gs["phase"], current_oids=live_oids)

        held_oids = expand_held_with_relations(held_seed, relations.edges)
        held_oids = {o for o in held_oids if o in tracker.state.objects}

        dbg, _ = tracker.step(
            rgb=rgb, depth=depth, T_wb=T_wb,
            detections=detections, phase=gs["phase"],
            T_bc=T_bc, T_bg=T_bg,
            held_oids=held_oids, held_seed=held_seed,
            relation_edges=relations.edges)
        grip.apply_merges(dbg.get("self_merges", []))
        relations.remap_after_merges(dbg.get("self_merges", []))

        a = dbg.get("assoc", {})
        edges_str = ", ".join(f"{e.parent}↑{e.child}" for e in relations.edges)
        print(f"fr {idx}: phase={gs['phase']} held={held_seed} "
              f"used={sorted(held_oids)} matched={len(a.get('match', {}))} "
              f"rels=[{edges_str}]")


if __name__ == "__main__":
    main()

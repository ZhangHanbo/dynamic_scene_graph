#!/usr/bin/env python3
"""
Full trajectory visualization — 3 separate outputs:

1. detection/  — Per-frame detection results (masks, boxes, labels, scores)
2. scene_graph/  — Scene graph with spatial relations between tracked objects
3. reconstruction/  — Progressive 3D reconstruction of the manipulated object
                      rendered from a FIXED camera angle to show growing completeness

Usage:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python tests/visualize_full.py [--step 2]
"""

import os
import sys
import json
import argparse
import base64
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from api import (
    ObjectReconstructor, ObjectTracker, PoseUpdater, RelationAnalyzer,
    FrameDetections,
)

DATA_ROOT = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects", "apple_bowl_2"
)

K = np.array([[554.3827, 0, 320.5], [0, 554.3827, 240.5], [0, 0, 1]], np.float32)

COLORS = [
    (46, 204, 113), (231, 76, 60), (52, 152, 219),
    (241, 196, 15), (155, 89, 182), (230, 126, 34),
    (26, 188, 156), (236, 112, 99), (93, 173, 226), (244, 208, 63),
]

# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_pose_txt(path):
    poses = []
    with open(path) as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8: continue
            _, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses


def load_detection_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    results = []
    for det in data.get("detections", []):
        mask_b64 = det.get("mask", "")
        if mask_b64:
            mask_bytes = base64.b64decode(mask_b64)
            mask_img = Image.open(BytesIO(mask_bytes)).convert("L")
            mask = (np.array(mask_img) > 128).astype(np.uint8)
        else:
            mask = np.zeros((480, 640), dtype=np.uint8)
        results.append({
            "mask": mask, "label": det.get("label", "unknown"),
            "score": det.get("score", 0.0), "id": det.get("object_id", 0),
            "box": det.get("box", [0, 0, 0, 0]),
        })
    return results


# ─────────────────────────────────────────────────────────────────────
# 1. Detection visualization
# ─────────────────────────────────────────────────────────────────────

def draw_detection_frame(rgb, detections, frame_idx):
    """Draw raw detection results: masks + boxes + labels + scores."""
    vis = rgb.copy()
    h, w = vis.shape[:2]

    for det in detections:
        mask = det["mask"]
        label = det["label"]
        box = det["box"]
        score = det["score"]
        obj_id = det["id"]
        color = COLORS[obj_id % len(COLORS)]

        # Mask overlay
        if mask.shape == (h, w):
            overlay = vis.copy()
            overlay[mask.astype(bool)] = color
            vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)

        # Box
        if len(box) == 4:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Frame info bar
    cv2.rectangle(vis, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(vis, f"Frame {frame_idx:04d} | Detections: {len(detections)}",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


# ─────────────────────────────────────────────────────────────────────
# 2. Scene graph visualization
# ─────────────────────────────────────────────────────────────────────

def draw_scene_graph_frame(rgb, detections, tracked_objects, relations, frame_idx, gripper_state):
    """Draw tracked objects with scene graph relations."""
    vis = rgb.copy()
    h, w = vis.shape[:2]

    # Map tracked objects by label for matching
    tracked_map = {}
    for obj in tracked_objects:
        tracked_map.setdefault(obj.label, []).append(obj)

    obj_centers = {}
    drawn_ids = set()

    for det in detections:
        label = det["label"]
        box = det["box"]

        # Find matching tracked object
        matched = None
        for obj in tracked_map.get(label, []):
            if obj.id not in drawn_ids:
                matched = obj
                drawn_ids.add(obj.id)
                break

        obj_id = matched.id if matched else det["id"]
        color = COLORS[obj_id % len(COLORS)]
        n_pts = len(matched.points) if matched and matched.points is not None else 0

        # Mask
        mask = det["mask"]
        if mask.shape == (h, w):
            overlay = vis.copy()
            overlay[mask.astype(bool)] = color
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # Box + label with tracked ID and point count
        if len(box) == 4:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"[{obj_id}] {label} ({n_pts}pts)"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            obj_centers[obj_id] = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Draw relation arrows
    if relations is not None:
        rel_styles = {
            "on": ((0, 220, 0), "ON"),
            "in": ((0, 200, 255), "IN"),
            "under": ((200, 0, 0), "UNDER"),
            "contain": ((200, 0, 200), "CONTAIN"),
        }
        for obj_id, rels in relations.relations.items():
            if obj_id not in obj_centers:
                continue
            for rel_type, targets in rels.items():
                if not targets:
                    continue
                arrow_color, rel_label = rel_styles.get(rel_type, ((128, 128, 128), rel_type))
                for tid in targets:
                    if tid not in obj_centers:
                        continue
                    p1 = obj_centers[obj_id]
                    p2 = obj_centers[tid]
                    cv2.arrowedLine(vis, p1, p2, arrow_color, 2, tipLength=0.12)
                    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                    cv2.putText(vis, rel_label, (mx + 5, my - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, arrow_color, 1, cv2.LINE_AA)

    # Legend
    y0 = 15
    for rel_type, (color_l, label_l) in rel_styles.items():
        cv2.line(vis, (w - 120, y0), (w - 95, y0), color_l, 2)
        cv2.putText(vis, label_l, (w - 90, y0 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_l, 1)
        y0 += 18

    # Info bar
    n_rels = sum(len(v) for rels in relations.relations.values() for v in rels.values()) if relations else 0
    cv2.rectangle(vis, (0, h - 28), (w, h), (0, 0, 0), -1)
    info = f"Frame {frame_idx:04d} | Objects: {len(tracked_objects)} | Relations: {n_rels} | {gripper_state}"
    cv2.putText(vis, info, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


# ─────────────────────────────────────────────────────────────────────
# 3. Reconstruction visualization (fixed-angle rendering)
# ─────────────────────────────────────────────────────────────────────

def render_pointcloud_fixed_view(points, colors_rgb=None, img_size=(480, 480),
                                  elevation=30, azimuth=45, distance=0.25):
    """Render a point cloud from a fixed viewpoint using simple projection.

    This produces a consistent view angle across all frames so you can
    see the reconstruction progressively filling in.
    """
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    if points is None or len(points) == 0:
        cv2.putText(img, "No points", (img_size[0] // 4, img_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)
        return img

    # Center the point cloud
    center = np.mean(points, axis=0)
    pts = points - center

    # Build rotation matrix from elevation/azimuth
    el_rad = np.radians(elevation)
    az_rad = np.radians(azimuth)
    Rz = np.array([[np.cos(az_rad), -np.sin(az_rad), 0],
                    [np.sin(az_rad), np.cos(az_rad), 0],
                    [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(el_rad), -np.sin(el_rad)],
                    [0, np.sin(el_rad), np.cos(el_rad)]])
    R = Rx @ Rz
    pts_rot = pts @ R.T

    # Orthographic projection
    extent = max(np.abs(pts_rot[:, :2]).max(), 0.01) * 1.3
    scale = min(img_size) / (2 * extent)
    cx, cy = img_size[0] // 2, img_size[1] // 2

    px = (pts_rot[:, 0] * scale + cx).astype(int)
    py = (-pts_rot[:, 1] * scale + cy).astype(int)  # flip Y for image coords

    # Depth-sort for painter's algorithm (back-to-front)
    depth_order = np.argsort(-pts_rot[:, 2])

    # Determine colors
    if colors_rgb is not None and len(colors_rgb) == len(points):
        pt_colors = (colors_rgb * 255).astype(np.uint8) if colors_rgb.max() <= 1.0 else colors_rgb.astype(np.uint8)
    else:
        # Depth-based coloring
        z = pts_rot[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        pt_colors = np.zeros((len(points), 3), dtype=np.uint8)
        pt_colors[:, 0] = (z_norm * 200 + 55).astype(np.uint8)  # R
        pt_colors[:, 1] = ((1 - z_norm) * 150 + 80).astype(np.uint8)  # G
        pt_colors[:, 2] = 180  # B

    for idx in depth_order:
        x, y = px[idx], py[idx]
        if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
            color = tuple(int(c) for c in pt_colors[idx])
            cv2.circle(img, (x, y), 2, color, -1)

    return img


def draw_reconstruction_frame(recon_img, frame_idx, n_points, obj_label, gripper_state):
    """Add info to the reconstruction render."""
    h, w = recon_img.shape[:2]

    # Title
    cv2.putText(recon_img, f"3D Reconstruction: {obj_label}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Point count progress bar
    max_pts = 6000
    ratio = min(n_points / max_pts, 1.0)
    bar_w = w - 40
    cv2.rectangle(recon_img, (20, h - 55), (20 + bar_w, h - 40), (60, 60, 60), -1)
    cv2.rectangle(recon_img, (20, h - 55), (20 + int(bar_w * ratio), h - 40), (46, 204, 113), -1)
    cv2.putText(recon_img, f"{n_points} pts", (20 + int(bar_w * ratio) + 5, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Info bar
    cv2.rectangle(recon_img, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(recon_img, f"Frame {frame_idx:04d} | {gripper_state}",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return recon_img


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--output", default=os.path.join(SCENEREP_ROOT, "tests", "vis_full"))
    parser.add_argument("--target-label", default=None, help="Object to track for reconstruction (auto-detect if None)")
    args = parser.parse_args()

    # Output dirs
    det_dir = os.path.join(args.output, "detection")
    sg_dir = os.path.join(args.output, "scene_graph")
    recon_dir = os.path.join(args.output, "reconstruction")
    combined_dir = os.path.join(args.output, "combined")
    for d in [det_dir, sg_dir, recon_dir, combined_dir]:
        os.makedirs(d, exist_ok=True)

    # Load poses
    cam_poses = load_pose_txt(os.path.join(args.data, "pose_txt", "camera_pose.txt"))
    ee_poses = load_pose_txt(os.path.join(args.data, "pose_txt", "ee_pose.txt"))
    l_finger = load_pose_txt(os.path.join(args.data, "pose_txt", "l_gripper_pose.txt"))
    r_finger = load_pose_txt(os.path.join(args.data, "pose_txt", "r_gripper_pose.txt"))

    n_total = len(cam_poses)

    # Initialize APIs
    tracker = ObjectTracker(K=K, voxel_size=0.003)
    reconstructor = ObjectReconstructor(voxel_size=0.003)
    relations = None

    # Gripper state machine
    last_finger_d = None
    last_state = "idle"
    manipulated_obj_id = None
    obj_id_in_ee = None

    print(f"Processing {n_total} frames (step={args.step}) from {args.data}")
    print(f"Output: {args.output}/")
    print()

    processed = 0
    for idx in range(0, n_total, args.step):
        rgb_path = os.path.join(args.data, "rgb", f"rgb_{idx:06d}.png")
        depth_path = os.path.join(args.data, "depth", f"depth_{idx:06d}.npy")
        det_path = os.path.join(args.data, "detection_h", f"detection_{idx:06d}_final.json")
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)
        T_cw = cam_poses[idx]
        detections = load_detection_json(det_path) if os.path.exists(det_path) else []

        # Gripper state
        finger_d = np.linalg.norm(l_finger[idx][:3, 3] - r_finger[idx][:3, 3])
        if last_finger_d is not None:
            d_diff = finger_d - last_finger_d
            if d_diff < -0.002:
                gripper_state = "GRASPING"
            elif d_diff > 0.002:
                gripper_state = "RELEASING"
            elif last_state == "GRASPING":
                gripper_state = "HOLDING"
            elif last_state == "RELEASING":
                gripper_state = "idle"
            else:
                gripper_state = last_state
        else:
            gripper_state = "idle"
        last_finger_d = finger_d

        T_ec = ee_poses[idx]

        # ── Detect grasped object at grasping onset ──
        if gripper_state == "GRASPING" and last_state != "GRASPING":
            T_ew = T_cw @ T_ec
            obj_id_in_ee = tracker.detect_held_object(T_ew)
            if obj_id_in_ee is not None:
                tracker.set_held_object(obj_id_in_ee)
                print(f"  Grasped object {obj_id_in_ee}")
        elif gripper_state == "idle" and last_state in ("RELEASING", "idle"):
            if obj_id_in_ee is not None:
                tracker.release_object(obj_id_in_ee)
                obj_id_in_ee = None

        # ── EE pose tracking for held object ──
        is_manipulating = gripper_state in ("GRASPING", "HOLDING", "RELEASING")
        if is_manipulating and obj_id_in_ee is not None:
            PoseUpdater.update_from_ee(tracker.internal_objects, obj_id_in_ee, T_cw, T_ec)

        # ── Track (associate_by_id skips pose_uncertain objects internally) ──
        if detections and not (gripper_state in ("GRASPING", "RELEASING")):
            fd = FrameDetections(
                labels=[d["label"] for d in detections],
                scores=np.array([d["score"] for d in detections]),
                masks=[d["mask"] for d in detections],
                bboxes=np.array([d["box"] for d in detections]),
            )
            tracked = tracker.update(fd, rgb, depth.astype(np.float32), T_cw)
        else:
            tracked = tracker._snapshot()

        # ── Reconstruct (skip during manipulation) ──
        if not is_manipulating:
            for det in detections:
                label = det["label"]
                mask = det["mask"]
                for obj in tracker.internal_objects:
                    if obj.label == label:
                        oid = obj.id
                        try:
                            reconstructor.get_points(oid)
                        except KeyError:
                            reconstructor.create(pose=obj.pose_cur, label=label, object_id=oid)
                        reconstructor.fuse(oid, rgb, depth.astype(np.float32), K, T_cw, mask=mask)
                        break

        # ── Relations ──
        if processed % 10 == 0 and len(tracker.internal_objects) > 0:
            relations = RelationAnalyzer.compute(tracker.internal_objects, tolerance=0.02)
        # Also recompute on release
        if gripper_state == "RELEASING" and last_state != "RELEASING":
            relations = RelationAnalyzer.compute(tracker.internal_objects, tolerance=0.02)

        # ── Auto-detect target for reconstruction viz ──
        target_label = args.target_label or "apple"
        if manipulated_obj_id is None:
            for obj in tracker.internal_objects:
                if obj.label == target_label:
                    manipulated_obj_id = obj.id
                    break

        # ── 1. Detection vis ──
        det_img = draw_detection_frame(rgb, detections, idx)
        cv2.imwrite(os.path.join(det_dir, f"det_{idx:04d}.png"),
                    cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR))

        # ── 2. Scene graph vis ──
        sg_img = draw_scene_graph_frame(rgb, detections, tracked, relations, idx, gripper_state)
        cv2.imwrite(os.path.join(sg_dir, f"sg_{idx:04d}.png"),
                    cv2.cvtColor(sg_img, cv2.COLOR_RGB2BGR))

        # ── 3. Reconstruction vis (fixed angle) ──
        if manipulated_obj_id is not None:
            try:
                pts = reconstructor.get_points(manipulated_obj_id)
                obj_internal = None
                for o in tracker.internal_objects:
                    if o.id == manipulated_obj_id:
                        obj_internal = o
                        break
                colors_rgb = obj_internal._colors if obj_internal and hasattr(obj_internal, '_colors') and len(obj_internal._colors) == len(pts) else None
            except KeyError:
                pts = np.empty((0, 3))
                colors_rgb = None

            recon_img = render_pointcloud_fixed_view(
                pts, colors_rgb=colors_rgb,
                img_size=(480, 480), elevation=25, azimuth=45, distance=0.2
            )
            recon_img = draw_reconstruction_frame(
                recon_img, idx, len(pts), target_label, gripper_state
            )
        else:
            recon_img = np.zeros((480, 480, 3), dtype=np.uint8)
            cv2.putText(recon_img, f"Waiting for '{target_label}'...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

        cv2.imwrite(os.path.join(recon_dir, f"recon_{idx:04d}.png"), recon_img)

        # ── 4. Combined: side-by-side ──
        # Resize detection and scene_graph to match reconstruction height
        target_h = recon_img.shape[0]  # 480
        det_bgr = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
        sg_bgr = cv2.cvtColor(sg_img, cv2.COLOR_RGB2BGR)

        # Scale det and sg to same height as recon, preserve aspect ratio
        det_h, det_w = det_bgr.shape[:2]
        scale = target_h / det_h
        det_resized = cv2.resize(det_bgr, (int(det_w * scale), target_h))
        sg_resized = cv2.resize(sg_bgr, (int(det_w * scale), target_h))

        # Add thin white separator lines
        sep = np.full((target_h, 2, 3), 255, dtype=np.uint8)
        combined = np.hstack([det_resized, sep, sg_resized, sep, recon_img])
        cv2.imwrite(os.path.join(combined_dir, f"combined_{idx:04d}.png"), combined)

        last_state = gripper_state

        processed += 1
        n_pts = len(pts) if manipulated_obj_id is not None and 'pts' in dir() else 0
        if processed % 20 == 0 or processed == 1:
            ee_str = f", ee_obj={obj_id_in_ee}" if obj_id_in_ee is not None else ""
            print(f"  [{processed:3d}] Frame {idx:04d}: {len(tracked)} tracked, "
                  f"{gripper_state}, {target_label}={n_pts}pts{ee_str}")

    # ── Summary ──
    print(f"\nDone. {processed} frames processed.")
    print(f"  detection/    -> {det_dir}/")
    print(f"  scene_graph/  -> {sg_dir}/")
    print(f"  reconstruction/ -> {recon_dir}/")

    print(f"\nFinal tracked objects:")
    for obj in tracker._snapshot():
        try:
            n = len(reconstructor.get_points(obj.id))
        except KeyError:
            n = 0
        print(f"  [{obj.id}] {obj.label}: {n} pts")

    if relations:
        print(f"\nFinal relations:")
        for oid, rels in relations.relations.items():
            for rt, tids in rels.items():
                if tids:
                    print(f"  obj {oid} {rt} {tids}")


if __name__ == "__main__":
    main()

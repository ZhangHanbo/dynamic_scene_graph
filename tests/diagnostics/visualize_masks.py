#!/usr/bin/env python3
"""
Visualize all object masks loaded from the apple_bowl_2 trajectory.

For each frame with detections in `detection_h/*_final.json`, overlay all
masks on the RGB image with per-object-id colors, bounding boxes, and
labels. Lets you inspect the mask quality + object persistence across
the trajectory that the integration test and orchestrator consume.

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python tests/visualize_masks.py [--step 2]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

DATA_ROOT = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects", "apple_bowl_2"
)


# Per-id colors (BGR for OpenCV)
_COLORS_BGR = [
    (113, 204, 46),    # green
    (60, 76, 231),     # red
    (219, 152, 52),    # blue
    (15, 196, 241),    # yellow
    (182, 89, 155),    # purple
    (34, 126, 230),    # orange
    (156, 188, 26),    # teal
    (99, 112, 236),    # salmon
    (227, 173, 94),    # light blue
    (63, 209, 245),    # gold
]


def _color_for(obj_id: int):
    return _COLORS_BGR[obj_id % len(_COLORS_BGR)]


def _load_detections(json_path: str) -> List[Dict]:
    """Decode the detection_h/*_final.json masks exactly as the orchestrator
    integration test does — so what you see here is what the system consumes."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    out = []
    for det in data.get("detections", []):
        mask_b64 = det.get("mask", "")
        if mask_b64:
            mask_bytes = base64.b64decode(mask_b64)
            mask = (np.array(Image.open(BytesIO(mask_bytes)).convert("L")) > 128)
            mask = mask.astype(np.uint8)
        else:
            mask = np.zeros((480, 640), dtype=np.uint8)
        out.append({
            "id": det.get("object_id"),
            "label": det.get("label", "unknown"),
            "mask": mask,
            "score": float(det.get("score", 0.0)),
            "box": det.get("box"),  # [x1, y1, x2, y2]
        })
    return out


def draw_mask_overlay(rgb: np.ndarray, detections: List[Dict],
                     frame_idx: int, n_frames: int,
                     alpha: float = 0.45) -> np.ndarray:
    """Overlay all masks on an RGB image with per-object colors + boxes."""
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = vis.shape[:2]

    # Mask overlay (all objects composited together)
    mask_layer = vis.copy()
    for det in detections:
        oid = det["id"]
        if oid is None:
            continue
        color = _color_for(oid)
        mask_bool = det["mask"].astype(bool)
        if mask_bool.shape != (h, w):
            # Resize if needed
            mask_bool = cv2.resize(det["mask"], (w, h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_layer[mask_bool] = color
    vis = cv2.addWeighted(vis, 1 - alpha, mask_layer, alpha, 0)

    # Bounding boxes + labels on top
    for det in detections:
        oid = det["id"]
        if oid is None:
            continue
        color = _color_for(oid)
        box = det.get("box")
        if box is not None and len(box) == 4:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"[{oid}] {det['label']} ({det['score']:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                           0.45, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 8)),
                           (x1 + tw + 6, y1), color, -1)
            cv2.putText(vis, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

    # Info bar
    cv2.rectangle(vis, (0, h - 28), (w, h), (0, 0, 0), -1)
    info = (f"Frame {frame_idx:04d} / {n_frames - 1}  |  "
            f"Detections: {len(detections)}  |  "
            f"source: detection_h/detection_{frame_idx:06d}_final.json")
    cv2.putText(vis, info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def make_montage(per_id_samples: Dict[int, List[np.ndarray]],
                 cell_w: int = 120, cell_h: int = 120,
                 n_per_row: int = 12,
                 margin: int = 4) -> np.ndarray:
    """Small-multiples overview: each row is one object id, columns are
    sampled frames where that id appeared."""
    if not per_id_samples:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    rows = []
    for oid in sorted(per_id_samples.keys()):
        samples = per_id_samples[oid]
        # Pad or truncate to n_per_row
        row_cells = []
        for i in range(n_per_row):
            if i < len(samples):
                cell = cv2.resize(samples[i], (cell_w, cell_h))
            else:
                cell = np.full((cell_h, cell_w, 3), 30, dtype=np.uint8)
            row_cells.append(cell)
        row_img = np.concatenate(row_cells, axis=1)
        # ID label strip on the left
        strip = np.full((cell_h, 60, 3), 40, dtype=np.uint8)
        color = _color_for(oid)
        cv2.rectangle(strip, (2, 2), (58, cell_h - 2), color, -1)
        cv2.putText(strip, f"[{oid}]", (5, cell_h // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        row_img = np.concatenate([strip, row_img], axis=1)
        rows.append(row_img)
    montage = np.concatenate(rows, axis=0)
    return montage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=2,
                        help="Process every N-th frame")
    parser.add_argument("--save-every", type=int, default=4,
                        help="Save visualization every N processed frames")
    parser.add_argument("--output", default=os.path.join(
        SCENEREP_ROOT, "tests", "vis_masks"))
    parser.add_argument("--max-frames", type=int, default=328)
    args = parser.parse_args()

    out_frames_dir = os.path.join(args.output, "frames")
    os.makedirs(out_frames_dir, exist_ok=True)

    rgb_dir = os.path.join(DATA_ROOT, "rgb")
    det_dir = os.path.join(DATA_ROOT, "detection_h")

    rgb_files = sorted(os.listdir(rgb_dir)) if os.path.isdir(rgb_dir) else []
    n_frames = min(args.max_frames, len(rgb_files))
    processed = 0
    saved_viz = 0

    # Track per-id statistics + samples for montage
    per_id_counts: Dict[int, int] = {}
    per_id_label: Dict[int, str] = {}
    per_id_score_sum: Dict[int, float] = {}
    per_id_samples: Dict[int, List[np.ndarray]] = {}
    per_id_frames_seen: Dict[int, List[int]] = {}

    for idx in range(0, n_frames, args.step):
        rgb_path = os.path.join(rgb_dir, f"rgb_{idx:06d}.png")
        det_path = os.path.join(det_dir, f"detection_{idx:06d}_final.json")
        if not os.path.exists(rgb_path) or not os.path.exists(det_path):
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        detections = _load_detections(det_path)

        # Track stats
        for det in detections:
            oid = det["id"]
            if oid is None:
                continue
            per_id_counts[oid] = per_id_counts.get(oid, 0) + 1
            per_id_label[oid] = det["label"]
            per_id_score_sum[oid] = (
                per_id_score_sum.get(oid, 0.0) + det["score"])
            per_id_frames_seen.setdefault(oid, []).append(idx)

        # Per-frame visualization (subsampled to keep output small)
        if processed % args.save_every == 0:
            vis = draw_mask_overlay(rgb, detections, idx, n_frames)
            cv2.imwrite(
                os.path.join(out_frames_dir, f"mask_{idx:04d}.png"), vis)

            # Sample crops for the montage (cropped to detection bbox)
            for det in detections:
                oid = det["id"]
                if oid is None or oid in per_id_samples and \
                        len(per_id_samples[oid]) >= 12:
                    continue
                box = det.get("box")
                if box is not None and len(box) == 4:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    # Pad crop to make it clear
                    pad = 10
                    x1p = max(0, x1 - pad)
                    y1p = max(0, y1 - pad)
                    x2p = min(rgb.shape[1], x2 + pad)
                    y2p = min(rgb.shape[0], y2 + pad)
                    crop = vis[y1p:y2p, x1p:x2p]
                    if crop.size > 0:
                        per_id_samples.setdefault(oid, []).append(crop)

            saved_viz += 1

        processed += 1

    # Montage of sampled crops per id
    if per_id_samples:
        montage = make_montage(per_id_samples,
                                cell_w=120, cell_h=120,
                                n_per_row=12)
        cv2.imwrite(os.path.join(args.output, "montage_per_id.png"), montage)

    # Print summary
    print(f"Processed {processed} frames, saved {saved_viz} annotated frames")
    print(f"Output: {out_frames_dir}/")
    print(f"Montage: {os.path.join(args.output, 'montage_per_id.png')}")
    print(f"\n=== Per-object summary across the trajectory ===")
    print(f"{'id':<4} {'label':<30} {'count':>6}  {'mean_score':>10}  "
          f"{'first':>6}  {'last':>6}")
    for oid in sorted(per_id_counts.keys()):
        cnt = per_id_counts[oid]
        mean_s = per_id_score_sum[oid] / cnt
        first = per_id_frames_seen[oid][0]
        last = per_id_frames_seen[oid][-1]
        print(f"{oid:<4} {per_id_label[oid]:<30} {cnt:>6}  {mean_s:>10.3f}  "
              f"{first:>6}  {last:>6}")


if __name__ == "__main__":
    main()

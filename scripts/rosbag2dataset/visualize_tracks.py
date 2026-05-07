#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize tracked detections under a trajectory folder.

Reads:
    <dataset>/rgb/rgb_NNNNNN.png
    <dataset>/detection_h/detection_NNNNNN_final.json

Writes:
    <dataset>/vis_tracks/overlay/frame_NNNNNN.png
        Full RGB with every track drawn: bbox + translucent mask tint +
        "label id=N score=.XX" label. Color is stable per object_id.

    <dataset>/vis_tracks/per_object/<label>/id<NNN>/frame_NNNNNN.png
        Same RGB, but only this object's bbox/mask highlighted. Nested
        so all tracks of the same class live under one subfolder —
        useful for scanning a single track across its lifetime, or
        comparing every "apple" track side-by-side.

    <dataset>/vis_tracks/track_timeline.png
        One matplotlib row per object showing the frames it was visible
        — quick eyeball for fragmentation.

    <dataset>/vis_tracks/summary.json
        {object_id: {label, first_frame, last_frame, n_frames, lifespan}}

Usage:
    python rosbag2dataset/visualize_tracks.py datasets/apple_drop
    python rosbag2dataset/visualize_tracks.py datasets/apple_drop \\
        --no-per-object --stride 2        # faster, only full overlays
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_mask(b64: Optional[str]) -> Optional[np.ndarray]:
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64, validate=True)
        arr = np.array(Image.open(io.BytesIO(raw)))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr > 127).astype(np.uint8)
    except Exception:
        return None


def _color_for_id(object_id: int) -> Tuple[int, int, int]:
    """Deterministic, distinct BGR color per object_id."""
    # Hash the id into a hue, then HSV→BGR. Saturation/value fixed for
    # strong readable colors.
    hue = (int(object_id) * 47) % 180     # OpenCV hue ∈ [0, 180)
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _safe_label(label: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return safe or "unknown"


def _object_subdir(per_obj_root: str, label: str, object_id: int) -> str:
    """Return ``<per_obj_root>/<label>/id<NNN>/`` (created on demand)."""
    path = os.path.join(per_obj_root, _safe_label(label),
                        f"id{object_id:03d}")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_detection(img_bgr: np.ndarray,
                    box: List[int],
                    mask: Optional[np.ndarray],
                    label: str,
                    object_id: int,
                    score: float,
                    alpha_mask: float = 0.45) -> None:
    """Draw a single detection in-place on img_bgr."""
    color = _color_for_id(object_id)

    if mask is not None and mask.shape[:2] == img_bgr.shape[:2]:
        layer = np.zeros_like(img_bgr)
        layer[mask > 0] = color
        cv2.addWeighted(img_bgr, 1.0, layer, alpha_mask, 0, img_bgr)
        # Outline the mask for crispness
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours, -1, color, 1)

    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    text = f"id={object_id} {label} {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    bx1, by1 = x1, max(0, y1 - th - 6)
    bx2, by2 = x1 + tw + 6, max(0, y1 - 1)
    cv2.rectangle(img_bgr, (bx1, by1), (bx2, by2), color, -1)
    cv2.putText(img_bgr, text, (bx1 + 3, by2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)


def _draw_id_stamp(img_bgr: np.ndarray, text: str) -> None:
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img_bgr, (5, 5), (5 + tw + 10, 5 + th + 10),
                  (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (10, 5 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize(dataset_root: str,
              per_object: bool = True,
              stride: int = 1,
              max_per_object: int = 0) -> None:
    rgb_dir = os.path.join(dataset_root, "rgb")
    det_dir = os.path.join(dataset_root, "detection_h")
    out_dir = os.path.join(dataset_root, "vis_tracks")
    overlay_dir = os.path.join(out_dir, "overlay")
    per_obj_root = os.path.join(out_dir, "per_object")
    os.makedirs(overlay_dir, exist_ok=True)
    if per_object:
        os.makedirs(per_obj_root, exist_ok=True)

    det_files = sorted(glob.glob(
        os.path.join(det_dir, "detection_*_final.json")))
    if not det_files:
        raise FileNotFoundError(
            f"no tracked JSONs in {det_dir} — run the pipeline first.")

    # summary: object_id → {label, frames[]}
    summary: Dict[int, Dict] = defaultdict(
        lambda: {"label": None, "frames": [], "scores": []})

    n_overlay = 0
    n_crops = 0
    frames_per_obj: Dict[int, int] = defaultdict(int)

    for det_path in det_files:
        fname = os.path.basename(det_path)
        stem = fname[len("detection_"):-len("_final.json")]
        try:
            fid = int(stem)
        except ValueError:
            continue
        if stride > 1 and fid % stride != 0:
            continue

        rgb_path = os.path.join(rgb_dir, f"rgb_{fid:06d}.png")
        if not os.path.isfile(rgb_path):
            continue
        img_bgr = cv2.imread(rgb_path)
        if img_bgr is None:
            continue

        with open(det_path, "r") as f:
            dets = json.load(f).get("detections", [])
        if not dets:
            continue

        # Build in-memory list of (box, mask, label, id, score) for reuse.
        parsed = []
        for d in dets:
            oid = int(d.get("object_id", -1))
            if oid < 0:
                continue
            box = d.get("box")
            if not box:
                continue
            score = float(d.get("score", 0.0))
            label = str(d.get("label", "unknown"))
            mask = _decode_mask(d.get("mask"))
            parsed.append((box, mask, label, oid, score))
            summary[oid]["label"] = label
            summary[oid]["frames"].append(fid)
            summary[oid]["scores"].append(score)

        # ---- overlay: all tracks on one frame -------------------------
        overlay = img_bgr.copy()
        for box, mask, label, oid, score in parsed:
            _draw_detection(overlay, box, mask, label, oid, score)
        _draw_id_stamp(overlay, f"frame {fid:06d}  ({len(parsed)} tracks)")
        cv2.imwrite(os.path.join(overlay_dir, f"frame_{fid:06d}.png"),
                    overlay)
        n_overlay += 1

        # ---- per-object: highlight one track at a time ----------------
        if per_object:
            for box, mask, label, oid, score in parsed:
                if max_per_object and frames_per_obj[oid] >= max_per_object:
                    continue
                obj_dir = _object_subdir(per_obj_root, label, oid)
                solo = img_bgr.copy()
                # dim other tracks with a light gray outline for context
                for ob_, ma_, la_, oi_, sc_ in parsed:
                    if oi_ == oid:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in ob_]
                    cv2.rectangle(solo, (x1, y1), (x2, y2),
                                  (150, 150, 150), 1)
                _draw_detection(solo, box, mask, label, oid, score)
                _draw_id_stamp(
                    solo, f"frame {fid:06d}  {label} id={oid}")
                cv2.imwrite(os.path.join(obj_dir, f"frame_{fid:06d}.png"),
                            solo)
                frames_per_obj[oid] += 1
                n_crops += 1

    # ---- JSON summary ------------------------------------------------------
    flat_summary = {}
    for oid, info in summary.items():
        frames = info["frames"]
        flat_summary[str(oid)] = {
            "label":       info["label"],
            "first_frame": int(min(frames)),
            "last_frame":  int(max(frames)),
            "n_frames":    int(len(frames)),
            "lifespan":    int(max(frames) - min(frames) + 1),
            "mean_score":  float(np.mean(info["scores"])) if info["scores"] else 0.0,
        }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(flat_summary, f, indent=2, sort_keys=True)

    # ---- Timeline plot -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sorted_ids = sorted(
            summary.keys(),
            key=lambda o: (summary[o]["label"] or "", min(summary[o]["frames"])))
        fig_h = max(3.0, 0.18 * len(sorted_ids) + 1.5)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        for row, oid in enumerate(sorted_ids):
            frames = sorted(summary[oid]["frames"])
            label = summary[oid]["label"]
            color_bgr = _color_for_id(oid)
            color_rgb = (color_bgr[2]/255., color_bgr[1]/255., color_bgr[0]/255.)
            ax.scatter(frames, [row] * len(frames),
                       s=8, marker="|", color=color_rgb)
            ax.text(-5, row, f"{label}#{oid}",
                    ha="right", va="center", fontsize=6)
        ax.set_yticks([])
        ax.set_xlabel("frame")
        ax.set_title(f"{os.path.basename(dataset_root.rstrip('/'))}  "
                     f"— {len(sorted_ids)} tracks")
        ax.set_xlim(left=0)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "track_timeline.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping track_timeline.png")

    print(f"[vis] dataset      : {dataset_root}")
    print(f"[vis] overlays     : {n_overlay}  ({overlay_dir})")
    if per_object:
        print(f"[vis] per-object crops : {n_crops}  ({per_obj_root})")
    print(f"[vis] tracks       : {len(summary)}  (summary.json)")
    print(f"[vis] timeline plot: {os.path.join(out_dir, 'track_timeline.png')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="trajectory folder (e.g. datasets/apple_drop)")
    ap.add_argument("--no-per-object", action="store_true",
                    help="skip writing per-object frame folders (faster)")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every N-th frame only (default: 1)")
    ap.add_argument("--max-per-object", type=int, default=0,
                    help="cap per-object frames (0 = all, default)")
    args = ap.parse_args()

    visualize(
        dataset_root=args.dataset,
        per_object=not args.no_per_object,
        stride=args.stride,
        max_per_object=args.max_per_object,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

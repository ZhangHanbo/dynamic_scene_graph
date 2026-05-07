#!/usr/bin/env python3
"""
Per-frame visualisation of detection_boxes/*.json (raw OWL output
post-size-filter, no SAM2 tracking). Color per label (hash-stable).

Output: tests/visualization_pipeline/<trajectory>/owl_detections/
    frame_NNNNNN.png
    summary.mp4
    label_histogram.png
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCENEREP_ROOT not in sys.path:
    sys.path.insert(0, SCENEREP_ROOT)
# ``tests/`` has no ``__init__.py``, so ``tests.visualize_sam2_observations``
# isn't importable without extra plumbing. Add the tests/ dir itself.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from visualize_sam2_observations import (
    _palette_color, _pngs_to_mp4, _resolve_data_root,
)  # noqa: E402

DATA_BASE = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects",
)


def _color_for_label(label: str) -> Tuple[int, int, int]:
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    return _palette_color(int(h[:6], 16))


def _load_owl_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return list(data.get("detections", []))


def render_frame(rgb: np.ndarray, dets: List[Dict[str, Any]],
                  frame_idx: int, out_path: str) -> None:
    out = rgb.copy()
    h, w = out.shape[:2]
    for det in dets:
        label = det.get("label", "?")
        score = float(det.get("score", 0.0))
        bb = det.get("box")
        if bb is None or len(bb) != 4:
            continue
        color = _color_for_label(label)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        x0, y0, x1, y1 = map(int, bb)
        cv2.rectangle(out, (x0, y0), (x1, y1), color_bgr[::-1], 2)
        tag = f"{label}  {score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y0 - 4, th + 2)
        cv2.rectangle(out, (x0, ty - th - 2), (x0 + tw + 4, ty + 2),
                      (255, 255, 255), -1)
        cv2.putText(out, tag, (x0 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr[::-1], 1,
                    cv2.LINE_AA)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7), dpi=120)
    ax.imshow(out)
    counts: Dict[str, int] = {}
    for d in dets:
        counts[d.get("label", "?")] = counts.get(d.get("label", "?"), 0) + 1
    ax.set_title(
        f"OWL detections (raw, post size-filter) — frame {frame_idx:04d}   "
        f"{len(dets)} boxes   labels: {dict(sorted(counts.items()))}",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_histogram(label_totals: Dict[str, int],
                      n_frames: int, out_path: str) -> None:
    labels = sorted(label_totals.keys(), key=lambda k: -label_totals[k])
    counts = [label_totals[k] for k in labels]
    per_frame = [c / n_frames for c in counts]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    colors = [tuple(c / 255.0 for c in _color_for_label(l)) for l in labels]
    ax.bar(labels, per_frame, color=colors)
    ax.set_ylabel("detections per frame")
    ax.set_title(f"OWL detections per label (averaged over {n_frames} frames)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(trajectory: str, step: int = 3, make_video: bool = True) -> None:
    data_root = _resolve_data_root(trajectory)
    rgb_dir = os.path.join(data_root, "rgb")
    perception_det = os.path.join(
        SCENEREP_ROOT, "tests", "visualization_pipeline", trajectory,
        "perception", "detection_boxes")
    legacy_det = os.path.join(data_root, "detection_boxes")
    det_dir = perception_det if os.path.isdir(perception_det) else legacy_det
    if not (os.path.isdir(rgb_dir) and os.path.isdir(det_dir)):
        print(f"[viz] missing rgb/ or detection_boxes/ in {data_root}",
              file=sys.stderr)
        return

    out_dir = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline",
                            trajectory, "owl_detections")
    if os.path.isdir(out_dir):
        for f in glob.glob(os.path.join(out_dir, "frame_*.png")):
            os.remove(f)
    os.makedirs(out_dir, exist_ok=True)

    rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    indices_all = [int(f[4:10]) for f in rgb_files]
    indices = indices_all[::step]
    # Skip frames whose detection JSON is older than the most-recent one --
    # handy when OWL is mid-rewrite and the trailing frames still hold
    # stale data from a previous run.
    det_mtimes = []
    for idx in indices_all:
        p = os.path.join(det_dir, f"detection_{idx:06d}.json")
        if os.path.exists(p):
            det_mtimes.append(os.path.getmtime(p))
    if det_mtimes:
        tip = max(det_mtimes)
        fresh_cutoff = tip - 24 * 3600
        indices = [i for i in indices
                   if os.path.exists(os.path.join(det_dir, f"detection_{i:06d}.json"))
                   and os.path.getmtime(os.path.join(det_dir, f"detection_{i:06d}.json")) >= fresh_cutoff]

    print(f"[viz] {len(indices)} frames (step={step}) from {trajectory}")
    label_totals: Dict[str, int] = {}
    rendered = 0
    for idx in indices:
        rgb_path = os.path.join(rgb_dir, f"rgb_{idx:06d}.png")
        det_path = os.path.join(det_dir, f"detection_{idx:06d}.json")
        if not os.path.exists(rgb_path):
            continue
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        dets = _load_owl_json(det_path)
        for d in dets:
            lab = d.get("label", "?")
            label_totals[lab] = label_totals.get(lab, 0) + 1
        out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        render_frame(rgb, dets, idx, out_path)
        rendered += 1
        if rendered % 10 == 0:
            print(f"  [{rendered}/{len(indices)}] frame {idx}: "
                  f"{len(dets)} boxes")

    render_histogram(label_totals, n_frames=rendered,
                      out_path=os.path.join(out_dir, "label_histogram.png"))

    if make_video:
        _pngs_to_mp4(out_dir, os.path.join(out_dir, "summary.mp4"))

    total = sum(label_totals.values())
    print(f"[viz] total labels: {label_totals}")
    print(f"[viz] total detections: {total}  "
          f"(mean {total / max(rendered, 1):.2f} per frame)")
    print(f"[viz] output: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_bowl_2")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    run(trajectory=args.trajectory, step=args.step,
        make_video=not args.no_video)

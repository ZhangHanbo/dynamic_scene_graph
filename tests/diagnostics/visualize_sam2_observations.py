#!/usr/bin/env python3
"""
Observation-layer visualization: dump every SAM2/detector observation,
per frame, with upstream `object_id` as the ONLY identity signal.

For each processed frame we render ONE PNG with:

  * the raw RGB image as backdrop,
  * each detection's binary mask overlaid as a translucent colored region
    (color keyed deterministically by upstream `object_id` modulo the
    palette size, so the same id gets the same color across frames),
  * the detection bounding box,
  * a label of the form
        "id:N  label  s=0.NN"
    pinned near the top-left corner of the bbox.

This is pure observation visualization -- NO tracker runs, no pose is
estimated, no state is propagated. The output lets us debug what
SAM2 / the upstream pipeline hands to the tracker on each frame, which
is the starting point for any downstream tracking investigation.

Output: tests/visualization_pipeline/<trajectory>/sam2_observations/
    frame_NNNNNN.png   -- per-frame overlay
    id_timeline.png    -- which object_ids appear in which frames
    summary.mp4        -- stitched at 5 fps

Run:
    conda run -n ocmp_test python tests/visualize_sam2_observations.py \
        [--trajectory apple_bowl_2] [--frames N] [--step S] [--no-video]
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Primary data root: the sibling Mobile_Manipulation_on_Fetch tree.
DATA_BASE = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects"
)
# Secondary: the repo's own datasets/ dir (pre-extracted rosbag traj).
DATA_BASE_FALLBACK = os.path.join(SCENEREP_ROOT, "datasets")


def _resolve_data_root(trajectory: str) -> str:
    """Return ``<base>/<trajectory>`` for the first base where the
    trajectory actually exists on disk, so viz can target both the
    Mobile_Manipulation_on_Fetch dataset and the SceneRep datasets/
    rosbag trajectories without extra flags."""
    for base in (DATA_BASE, DATA_BASE_FALLBACK):
        cand = os.path.join(base, trajectory)
        if os.path.isdir(os.path.join(cand, "rgb")):
            return cand
    # Fall through to the primary base so the caller's error message
    # continues to point somewhere sane.
    return os.path.join(DATA_BASE, trajectory)


# 20-color palette keyed by `object_id % 20`. Distinguishable on top of
# typical tabletop RGB (warm / cool mix, mid-saturation).
_PALETTE_RGB = [
    (  0, 200,  80), (220,  60,  40), ( 40, 140, 220), (245, 200,  20),
    (160,  80, 200), (240, 130,  30), ( 20, 180, 160), (230, 120, 110),
    (100, 160, 230), (250, 220,  60), ( 80, 200, 120), (200, 100, 160),
    (140, 200,  50), (100, 100, 240), (230, 160,  80), ( 40, 220, 200),
    (220,  80, 200), (120, 120, 120), (200, 220, 120), ( 60, 100, 180),
]


def _palette_color(oid: int) -> Tuple[int, int, int]:
    return _PALETTE_RGB[int(oid) % len(_PALETTE_RGB)]


def _load_detections(json_path: str,
                      min_mean_score: float = 0.0) -> List[Dict[str, Any]]:
    """Load one frame's detections; decode mask PNGs from base64.

    Returns a list of dicts with keys:
        object_id, label, score, mean_score, n_obs, box (xyxy), mask (H, W uint8).
    Detections without a mask / box are skipped. If
    ``min_mean_score > 0``, tracks whose ``mean_score`` (the average OWL
    confidence across all frames the track was re-detected on, written
    by sam2_client) is below the threshold are filtered out here -- so
    the visualisation shows only persistent high-confidence tracks.
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)

    out: List[Dict[str, Any]] = []
    for det in data.get("detections", []):
        mean_sc = float(det.get("mean_score", det.get("score", 0.0)))
        if mean_sc < min_mean_score:
            continue
        mask_b64 = det.get("mask", "")
        if not mask_b64:
            continue
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask = np.array(Image.open(BytesIO(mask_bytes)).convert("L"))
            mask = (mask > 128).astype(np.uint8)
        except Exception:
            continue
        out.append({
            "object_id": det.get("object_id"),
            "label":      det.get("label", "?"),
            "score":      float(det.get("score", 0.0)),
            "mean_score": mean_sc,
            "n_obs":      int(det.get("n_obs", 0)),
            "box":        det.get("box"),
            "mask":       mask,
        })
    return out


def overlay_detections(rgb: np.ndarray,
                        detections: List[Dict[str, Any]],
                        alpha: float = 0.45) -> np.ndarray:
    """Return an RGB overlay of all detections on `rgb`.

    Masks are blended with `alpha`; bbox + text annotations are drawn
    with OpenCV on top. All detections are rendered; none are filtered.
    """
    out = rgb.copy()
    h, w = out.shape[:2]

    for det in detections:
        oid = det.get("object_id")
        if oid is None:
            continue
        color = _palette_color(oid)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))

        mask = det["mask"]
        if mask.shape[:2] != (h, w):
            # Detector and RGB disagree on shape; resize the mask.
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask.astype(bool)

        if mask_bool.any():
            # Blend the mask region with the palette color.
            colored = np.zeros_like(out)
            colored[mask_bool] = color
            out = np.where(mask_bool[..., None],
                            (alpha * colored + (1 - alpha) * out).astype(np.uint8),
                            out)

        bb = det.get("box")
        if bb is not None and len(bb) == 4:
            x0, y0, x1, y1 = map(int, bb)
            cv2.rectangle(out, (x0, y0), (x1, y1), color_bgr[::-1], 2)
            tag = f"id:{oid}  {det['label']}  s={det['score']:.2f}"
            txt_y = max(y0 - 4, 12)
            cv2.rectangle(out, (x0, txt_y - 10), (x0 + 10 + 8 * len(tag),
                                                    txt_y + 3),
                          (255, 255, 255), -1)
            cv2.putText(out, tag, (x0 + 3, txt_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        color_bgr[::-1], 1, cv2.LINE_AA)

    return out


def render_frame(rgb: np.ndarray,
                  detections: List[Dict[str, Any]],
                  frame_idx: int,
                  out_path: str) -> None:
    overlay = overlay_detections(rgb, detections)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=120)
    ax.imshow(overlay)
    ids_present = sorted(int(d["object_id"]) for d in detections
                         if d.get("object_id") is not None)
    ax.set_title(
        f"SAM2 observations — frame {frame_idx:04d}   "
        f"{len(detections)} masks   ids: {ids_present}",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend = unique (id, label) seen this frame, one color each.
    seen: Dict[int, str] = {}
    for det in detections:
        oid = det.get("object_id")
        if oid is None:
            continue
        seen.setdefault(int(oid), det.get("label", "?"))
    handles = [
        mpatches.Patch(color=tuple(c / 255.0 for c in _palette_color(oid)),
                        label=f"id:{oid}  {lab}")
        for oid, lab in sorted(seen.items())
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="lower right",
                  framealpha=0.85)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_id_timeline(id_presence: Dict[int, List[int]],
                        id_labels: Dict[int, str],
                        all_frames: List[int],
                        out_path: str) -> None:
    """Gantt-style plot: each upstream id gets a row, blocks show frames
    where it was observed."""
    fig, ax = plt.subplots(figsize=(12, 1 + 0.25 * max(1, len(id_presence))))
    min_frame = min(all_frames) if all_frames else 0
    max_frame = max(all_frames) if all_frames else 1

    for row, oid in enumerate(sorted(id_presence.keys())):
        frames = id_presence[oid]
        if not frames:
            continue
        color = tuple(c / 255.0 for c in _palette_color(oid))
        # Segments of consecutive frames.
        frames_sorted = sorted(frames)
        seg_start = frames_sorted[0]
        prev = seg_start
        for f in frames_sorted[1:] + [frames_sorted[-1] + 10**6]:
            if f - prev > (all_frames[1] - all_frames[0]) * 2 \
                    if len(all_frames) > 1 else 2:
                ax.plot([seg_start, prev], [row, row],
                        color=color, linewidth=6, solid_capstyle="butt")
                seg_start = f
            prev = f

    ax.set_yticks(range(len(id_presence)))
    ax.set_yticklabels([f"id {oid}  ({id_labels.get(oid, '?')})"
                        for oid in sorted(id_presence.keys())],
                       fontsize=8)
    ax.set_xlabel("frame index")
    ax.set_xlim(min_frame - 5, max_frame + 5)
    ax.set_title("Upstream object_id presence across the trajectory")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pngs_to_mp4(png_dir: str, mp4_path: str, fps: int = 5) -> None:
    frames = sorted(glob.glob(os.path.join(png_dir, "frame_*.png")))
    if not frames:
        print("[viz] no frames to stitch into mp4")
        return
    im0 = cv2.imread(frames[0])
    h, w = im0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
    for f in frames:
        im = cv2.imread(f)
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        writer.write(im)
    writer.release()
    print(f"[viz] summary.mp4 saved ({len(frames)} frames @ {fps} fps)")


def run(trajectory: str = "apple_bowl_2",
        n_frames: Optional[int] = None,
        step: int = 3,
        min_mean_score: float = 0.1,
        make_video: bool = True) -> None:
    data_root = _resolve_data_root(trajectory)
    rgb_dir = os.path.join(data_root, "rgb")
    # Prefer tests/visualization_pipeline/<trajectory>/perception/detection_h/
    # (where run_pipeline.sh now writes); fall back to the legacy
    # <dataset>/detection_h/ for pre-migration data.
    perception_det = os.path.join(
        SCENEREP_ROOT, "tests", "visualization_pipeline", trajectory,
        "perception", "detection_h")
    legacy_det = os.path.join(data_root, "detection_h")
    det_dir = perception_det if os.path.isdir(perception_det) else legacy_det
    if not (os.path.isdir(rgb_dir) and os.path.isdir(det_dir)):
        print(f"[viz] missing rgb/ or detection_h/ in {data_root}",
              file=sys.stderr)
        return

    out_dir = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline",
                            trajectory, "sam2_observations")
    if os.path.isdir(out_dir):
        for f in glob.glob(os.path.join(out_dir, "frame_*.png")):
            os.remove(f)
    os.makedirs(out_dir, exist_ok=True)

    rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    indices_all = [int(f[4:10]) for f in rgb_files]
    total = len(indices_all)
    if n_frames is not None:
        indices_all = indices_all[:n_frames]
    indices = indices_all[::step]

    print(f"[viz] {len(indices)} frames scheduled from {trajectory} "
          f"(total raw frames: {total}, step={step})")

    id_presence: Dict[int, List[int]] = {}
    id_labels: Dict[int, str] = {}

    for local_i, idx in enumerate(indices):
        rgb_path = os.path.join(rgb_dir, f"rgb_{idx:06d}.png")
        det_path = os.path.join(det_dir, f"detection_{idx:06d}_final.json")
        if not os.path.exists(rgb_path):
            continue
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        dets = _load_detections(det_path, min_mean_score=min_mean_score)

        for det in dets:
            oid = det.get("object_id")
            if oid is None:
                continue
            id_presence.setdefault(int(oid), []).append(idx)
            id_labels[int(oid)] = det.get("label", "?")

        out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        render_frame(rgb, dets, idx, out_path)

        if local_i % 10 == 0:
            ids = sorted(int(d["object_id"]) for d in dets
                          if d.get("object_id") is not None)
            print(f"  [{local_i+1}/{len(indices)}] frame {idx}: "
                  f"{len(dets)} masks  ids={ids}")

    render_id_timeline(id_presence, id_labels, indices,
                        os.path.join(out_dir, "id_timeline.png"))
    print(f"[viz] id_timeline.png saved")

    if make_video:
        _pngs_to_mp4(out_dir, os.path.join(out_dir, "summary.mp4"))

    print(f"[viz] done. Output: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_bowl_2")
    ap.add_argument("--frames", type=int, default=None,
                    help="max number of raw frames to cover "
                         "(default: all)")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--min-mean-score", type=float, default=0.1,
                    help="filter out tracks whose averaged OWL score is "
                         "below this threshold (default 0.1)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    run(trajectory=args.trajectory,
        n_frames=args.frames, step=args.step,
        min_mean_score=args.min_mean_score,
        make_video=not args.no_video)

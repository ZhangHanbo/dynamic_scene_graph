#!/usr/bin/env python3
"""Visualise per-frame Hungarian-matching debug data.

Reads the JSON dumps produced by ``sam2_client.py --debug-dir`` and
writes one 2x2 panel PNG per frame plus a stitched mp4:

  TL: OWL detections on RGB (box + label + score)
  TR: tracked objects carried from the PREVIOUS frame
      (dashed outline = dormant track with only last-valid bbox)
  BL: matching internals — cost matrix + Hungarian / fallback
      decisions, rendered as a monospace text panel
  BR: tracks AFTER this frame's matching + new-seed pass

Colours: stable per-oid (same id gets the same colour across frames)
and a separate palette for OWL detections (keyed by det_idx within
the frame). This lets you eyeball "which OWL det mapped to which
track" across the 4 panels.

Run:
    python tests/visualize_matching_debug.py \
        --trajectory apple_in_the_tray \
        --fid-start 500 --fid-end 700
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Stable palette for OIDs (20 entries -- keyed by oid % len).
_PALETTE_OID = [
    (0.00, 0.78, 0.31), (0.86, 0.24, 0.16), (0.16, 0.55, 0.86),
    (0.96, 0.78, 0.08), (0.63, 0.31, 0.78), (0.94, 0.51, 0.12),
    (0.08, 0.71, 0.63), (0.90, 0.47, 0.43), (0.39, 0.63, 0.90),
    (0.98, 0.86, 0.24), (0.31, 0.78, 0.47), (0.78, 0.39, 0.63),
    (0.47, 0.47, 0.78), (0.98, 0.63, 0.16), (0.47, 0.63, 0.31),
    (0.78, 0.55, 0.31), (0.39, 0.78, 0.78), (0.86, 0.31, 0.47),
    (0.24, 0.47, 0.63), (0.63, 0.31, 0.16),
]

# Separate palette for OWL dets in one frame (keyed by det_idx).
_PALETTE_DET = [
    (0.00, 0.40, 1.00), (1.00, 0.20, 0.20), (0.00, 0.70, 0.40),
    (1.00, 0.60, 0.00), (0.60, 0.00, 0.80), (0.00, 0.80, 0.80),
    (1.00, 0.00, 0.60), (0.40, 0.40, 0.00),
]


def _color_oid(oid: int) -> Tuple[float, float, float]:
    return _PALETTE_OID[int(oid) % len(_PALETTE_OID)]


def _color_det(det_idx: int) -> Tuple[float, float, float]:
    return _PALETTE_DET[int(det_idx) % len(_PALETTE_DET)]


def _decode_mask(b64: Optional[str]) -> Optional[np.ndarray]:
    if not b64:
        return None
    raw = base64.b64decode(b64, validate=True)
    arr = np.array(Image.open(io.BytesIO(raw)))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 127).astype(bool)


def _draw_box(ax, x1, y1, x2, y2, color, dashed=False, linewidth=1.5):
    rect = mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor=color, linewidth=linewidth,
        linestyle=("--" if dashed else "-"))
    ax.add_patch(rect)


def _draw_tag(ax, x, y, text, color, fontsize=7):
    ax.text(x, y, text, color="white", fontsize=fontsize,
            va="bottom", ha="left",
            bbox=dict(facecolor=color, alpha=0.85,
                      edgecolor="none", pad=1))


def _overlay_mask(ax, mask: np.ndarray, color: Tuple[float, float, float],
                  alpha: float = 0.35):
    if mask is None or not bool(mask.any()):
        return
    H, W = mask.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[mask] = (color[0], color[1], color[2], alpha)
    ax.imshow(overlay)


def _panel_owl(ax, rgb: np.ndarray, owl_dets: List[Dict[str, Any]]):
    ax.imshow(rgb)
    n_masks = sum(1 for d in owl_dets if d.get("mask_b64"))
    title = f"OWL detections ({len(owl_dets)})"
    if n_masks:
        title += f"   -- {n_masks}/{len(owl_dets)} w/ SAM mask"
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for idx, d in enumerate(owl_dets):
        color = _color_det(idx)
        m = _decode_mask(d.get("mask_b64"))
        if m is not None:
            _overlay_mask(ax, m, color, alpha=0.4)
        x1, y1, x2, y2 = d["box"]
        _draw_box(ax, x1, y1, x2, y2, color, linewidth=1.8)
        _draw_tag(ax, x1, max(0, y1 - 2),
                  f"[{idx}] {d['label']} s={d['score']:.2f}", color)


def _panel_tracks(ax, rgb: np.ndarray, tracks_list: List[Dict[str, Any]],
                   title: str):
    ax.imshow(rgb)
    n_active = sum(1 for t in tracks_list if not t.get("is_dormant"))
    n_dorm = len(tracks_list) - n_active
    subtitle = f"{n_active} active"
    if n_dorm:
        subtitle += f", {n_dorm} dormant"
    ax.set_title(f"{title}  ({subtitle})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for t in tracks_list:
        oid = int(t["oid"])
        color = _color_oid(oid)
        m = _decode_mask(t.get("mask_b64"))
        if m is not None:
            _overlay_mask(ax, m, color)
        x1, y1, x2, y2 = t["box"]
        dashed = bool(t.get("is_dormant"))
        _draw_box(ax, x1, y1, x2, y2, color, dashed=dashed, linewidth=1.5)
        tag = f"#{oid} {t['label']}"
        if dashed:
            tag = "(dorm) " + tag
        _draw_tag(ax, x1, max(0, y1 - 2), tag, color)


def _panel_matching(ax, owl_dets: List[Dict[str, Any]],
                     matching: Dict[str, Any]):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Matching: cost = (1-IoU) + label_penalty",
                 fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.invert_yaxis()
    ax.set_facecolor("#f4f4f4")

    max_cost = float(matching.get("max_cost", 0.7))
    label_pen = float(matching.get("label_penalty", 0.2))
    score_w = float(matching.get("score_weight", 0.0))
    seed_min = float(matching.get("new_seed_min_score", 0.0))
    track_entries = matching.get("track_entries", [])
    cost_matrix = matching.get("cost_matrix")
    hungarian = matching.get("hungarian", [])
    fallback = matching.get("fallback", [])
    new_seeds = matching.get("new_seeds", [])

    hung_accepted = {(h["det_idx"], h["track_idx"])
                     for h in hungarian if h.get("accepted")}
    hung_rejected = {(h["det_idx"], h["track_idx"])
                     for h in hungarian if not h.get("accepted")}
    fb_by_det = {f["det_idx"]: f for f in fallback}
    seed_by_det = {s["det_idx"]: s for s in new_seeds}

    lines: List[str] = []
    lines.append(f"max_cost={max_cost:.2f}   label_pen={label_pen:.2f}"
                 f"   score_w={score_w:.2f}   seed_min={seed_min:.2f}")
    lines.append(f"N_det={len(owl_dets)}   "
                 f"M_track={len(track_entries)}")
    lines.append("")
    lines.append("Tracks (columns):")
    for j, e in enumerate(track_entries):
        k = e.get("kind", "active")[0].upper()
        lines.append(f"  col[{j}] ({k}) oid=#{e['oid']}")
    lines.append("")
    lines.append("OWL dets (rows):")
    for r, d in enumerate(owl_dets):
        lines.append(f"  row[{r}] {d['label']:8s} s={d['score']:.2f}")
    lines.append("")
    # Cost matrix
    if cost_matrix and len(cost_matrix) > 0 and len(cost_matrix[0]) > 0:
        header_cols = "  ".join(f"c{j:<3d}" for j in range(len(cost_matrix[0])))
        lines.append(f"cost matrix    {header_cols}")
        for r, row in enumerate(cost_matrix):
            cells: List[str] = []
            for j, c in enumerate(row):
                if c >= 1e5:
                    s = " inf"
                else:
                    s = f"{c:4.2f}"
                if (r, j) in hung_accepted:
                    s = s + "*"
                elif (r, j) in hung_rejected:
                    s = s + "X"
                else:
                    s = s + " "
                cells.append(s)
            lines.append(f"  r{r:<2d}         " + " ".join(cells))
        lines.append("")
    # Decisions
    lines.append("Hungarian:")
    if not hungarian:
        lines.append("  (none)")
    else:
        for h in hungarian:
            mark = "ACCEPT" if h.get("accepted") else "reject"
            lines.append(f"  det[{h['det_idx']}] -> "
                         f"#{h['oid']}  c={h['cost']:.2f}  {mark}")
    lines.append("Fallback (single-direction):")
    if not fallback:
        lines.append("  (none)")
    else:
        for f in fallback:
            lines.append(f"  det[{f['det_idx']}] -> "
                         f"#{f['oid']}  c={f['cost']:.2f}")
    lines.append("New seeds / drops:")
    if not new_seeds:
        lines.append("  (none)")
    else:
        for s in new_seeds:
            extra = ""
            if s.get("best_reject_oid") is not None:
                extra = (f"  (closest #{s['best_reject_oid']} "
                         f"c={s['best_reject_cost']:.2f})")
            if s.get("dropped_low_score"):
                sc = s.get("score", 0.0)
                lines.append(f"  det[{s['det_idx']}] DROP "
                             f"(score={sc:.2f}<seed_min){extra}")
            else:
                lines.append(f"  det[{s['det_idx']}] -> "
                             f"#{s['oid']} (NEW){extra}")

    self_merges = matching.get("self_merges", []) or []
    self_tau = matching.get("self_match_max_cost")
    if self_tau is not None:
        lines.append(f"Self-match merges (tau={float(self_tau):.2f}):")
    else:
        lines.append("Self-match merges:")
    if not self_merges:
        lines.append("  (none)")
    else:
        for sm in self_merges:
            lines.append(f"  #{sm['keep_oid']} <- #{sm['drop_oid']} "
                         f"c={sm['cost']:.2f}")

    ax.text(0.01, 0.01, "\n".join(lines),
            family="monospace", fontsize=7, va="top", ha="left")


def _load_rgb(rgb_dir: str, fid: int) -> Optional[np.ndarray]:
    for fmt in (f"rgb_{fid:06d}.png", f"{fid:06d}.png",
                f"frame_{fid:06d}.png", f"image_{fid:06d}.png"):
        p = os.path.join(rgb_dir, fmt)
        if os.path.isfile(p):
            return np.array(Image.open(p).convert("RGB"))
    return None


def _render_one(debug_json: Dict[str, Any], rgb: np.ndarray,
                 out_path: str):
    fid = debug_json["fid"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                              gridspec_kw={"hspace": 0.12,
                                           "wspace": 0.08})
    _panel_owl(axes[0, 0], rgb, debug_json.get("owl_dets", []))
    # Panel 2: tracks BEFORE self-merge (post-matching, pre-merge).
    # Falls back to prev_tracks for older JSON dumps that don't
    # have tracks_before_self_merge yet.
    pre_merge = debug_json.get(
        "tracks_before_self_merge",
        debug_json.get("prev_tracks", []))
    _panel_tracks(axes[0, 1], rgb, pre_merge,
                   "Tracks BEFORE self-merge")
    _panel_matching(axes[1, 0], debug_json.get("owl_dets", []),
                     debug_json.get("matching", {}))
    _panel_tracks(axes[1, 1], rgb, debug_json.get("final_tracks", []),
                   "Tracks AFTER self-merge (final)")
    fig.suptitle(f"Hungarian matching debug — fid={fid}",
                 fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--fid-start", type=int, default=0)
    ap.add_argument("--fid-end", type=int, default=10 ** 9)
    ap.add_argument("--debug-dir", default=None,
                    help="Override the default debug dir (default: "
                         "tests/visualization_pipeline/<traj>/"
                         "perception/matching_debug_json).")
    ap.add_argument("--out-dir", default=None,
                    help="Override the default output dir (default: "
                         "tests/visualization_pipeline/<traj>/"
                         "matching_debug).")
    ap.add_argument("--rgb-dir", default=None,
                    help="Override the RGB source dir (default: "
                         "datasets/<traj>/rgb).")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip the summary.mp4 stitching step.")
    args = ap.parse_args()

    viz_root = os.path.join(SCENEREP_ROOT, "tests",
                             "visualization_pipeline", args.trajectory)
    debug_dir = (args.debug_dir or
                  os.path.join(viz_root, "perception",
                               "matching_debug_json"))
    out_dir = (args.out_dir or
                os.path.join(viz_root, "matching_debug"))
    rgb_dir = (args.rgb_dir or
                os.path.join(SCENEREP_ROOT, "datasets",
                             args.trajectory, "rgb"))
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(debug_dir):
        print(f"[viz] missing debug dir: {debug_dir}")
        return 1
    if not os.path.isdir(rgb_dir):
        print(f"[viz] missing rgb dir: {rgb_dir}")
        return 1

    files = sorted(f for f in os.listdir(debug_dir)
                    if f.startswith("debug_") and f.endswith(".json"))
    rendered = 0
    for fn in files:
        stem = fn[len("debug_"):-len(".json")]
        try:
            fid = int(stem)
        except ValueError:
            continue
        if not (args.fid_start <= fid <= args.fid_end):
            continue
        with open(os.path.join(debug_dir, fn)) as f:
            data = json.load(f)
        rgb = _load_rgb(rgb_dir, fid)
        if rgb is None:
            print(f"[viz] missing rgb for fid={fid}")
            continue
        out_path = os.path.join(out_dir, f"debug_{fid:06d}.png")
        _render_one(data, rgb, out_path)
        rendered += 1
        if rendered % 25 == 0:
            print(f"[viz] {rendered} rendered (last fid={fid})")

    print(f"[viz] done. rendered={rendered}  out={out_dir}")
    if not args.no_video and rendered > 0:
        video_path = os.path.join(out_dir, "summary.mp4")
        cmd = ["ffmpeg", "-y", "-framerate", "5",
               "-pattern_type", "glob",
               "-i", os.path.join(out_dir, "debug_*.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path]
        try:
            subprocess.run(cmd, check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
            print(f"[viz] summary.mp4 -> {video_path}")
        except Exception as e:
            print(f"[viz] ffmpeg failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

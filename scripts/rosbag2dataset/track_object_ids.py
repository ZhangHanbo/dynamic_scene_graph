#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assign stable object IDs across frames from OWL+SAM detections.

Pipeline position:
    rosbag2dataset_5hz.py        →  rgb/, depth/, pose_txt/, particles/
    rosbag2dataset/owl/owl_object_scores.py   →  detection_boxes/detection_NNNNNN.json  (boxes)
    rosbag2dataset/sam/sam.py                 →  detection_boxes/detection_NNNNNN.json  (+ base64 masks)
--> THIS FILE                      →  detection_h/detection_NNNNNN_final.json (+ object_id)

The consumer (``detection/mask_extractor._extract_by_json``) reads those
``detection_h/detection_<fid>_final.json`` files and looks for
``label``, ``score``, ``mask`` (base64 PNG) and ``object_id``.

Tracker design
──────────────
Pure 2-D, label-gated Hungarian matcher between consecutive frames:

  cost(prev, curr) = w_label · 𝟙[label_prev ≠ label_curr]
                   + w_iou  · (1 − bbox_IoU(prev, curr))
                   + w_mask · (1 − mask_IoU(prev, curr))

The mask IoU term is computed only when both masks are present (they are,
after SAM). We gate by label first — different classes never match — so
the matrix is block-diagonal by class and fast even with many detections.

Unmatched current-frame detections get a fresh ID; unmatched previous
detections are "parked" and can rematch for ``--miss-tolerance`` frames
(default 15) before being retired. This survives brief occlusions.

The tracker is deliberately standalone: unlike
``detection/hungarian_detection.py``, it does *not* depend on the running
``SceneObject`` list. This is what you want for an offline preprocessing
pass — `data_demo.py`'s own Hungarian pass can still refine IDs at
run-time if needed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Mask / box helpers
# ---------------------------------------------------------------------------

def _decode_mask(b64: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(b64, validate=True)
        arr = np.array(Image.open(io.BytesIO(raw)))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr > 127).astype(np.uint8)
    except Exception:
        return None


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _iou_box(a: Tuple[int, int, int, int],
             b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_mask(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a > 0, b > 0).sum())
    union = int(np.logical_or(a > 0, b > 0).sum())
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Detection representation
# ---------------------------------------------------------------------------

@dataclass
class Det:
    """Parsed detection for a single frame."""
    label: str
    score: float
    box: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    mask: Optional[np.ndarray] = None
    raw_json: dict = field(default_factory=dict)  # unchanged source dict


def _parse_frame(det_json_path: str) -> List[Det]:
    with open(det_json_path, "r") as f:
        data = json.load(f)
    out: List[Det] = []
    for d in data.get("detections", []):
        # OWL-style (pre-filter) has {detection: [{label,score}], box}.
        # SAM-augmented has {label, score, box, mask}. Either works:
        if "detection" in d and isinstance(d["detection"], list) \
                and d["detection"]:
            top = max(d["detection"], key=lambda x: x.get("score", 0.0))
            label = str(top.get("label", "unknown"))
            score = float(top.get("score", 0.0))
        else:
            label = str(d.get("label", "unknown"))
            score = float(d.get("score", 0.0))
        box = d.get("box")
        if box is None:
            continue
        box = tuple(int(v) for v in box)
        m_b64 = d.get("mask")
        mask = _decode_mask(m_b64) if m_b64 else None
        # Prefer mask-tight bbox when we have the mask — it's more precise
        # than OWL's loose box.
        if mask is not None:
            mb = _bbox_from_mask(mask)
            if mb is not None:
                box = mb
        out.append(Det(label=label, score=score, box=box, mask=mask,
                       raw_json=d))
    return out


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    label: str
    box: Tuple[int, int, int, int]
    mask: Optional[np.ndarray]
    score: float
    last_seen_frame: int


class Tracker:
    def __init__(self,
                 miss_tolerance: int = 15,
                 match_threshold: float = 0.5,
                 w_iou: float = 0.6,
                 w_mask: float = 0.4):
        self.miss_tolerance = int(miss_tolerance)
        self.match_threshold = float(match_threshold)
        self.w_iou = float(w_iou)
        self.w_mask = float(w_mask)
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0

    # pairwise cost, +inf if labels disagree
    def _cost(self, tr: Track, det: Det) -> float:
        if tr.label != det.label:
            return float("inf")
        c = self.w_iou * (1.0 - _iou_box(tr.box, det.box))
        if tr.mask is not None and det.mask is not None \
                and tr.mask.shape == det.mask.shape:
            c += self.w_mask * (1.0 - _iou_mask(tr.mask, det.mask))
        else:
            # No mask IoU available; re-weight.
            c = (self.w_iou + self.w_mask) * (1.0 - _iou_box(tr.box, det.box))
        return c

    def step(self, frame_idx: int, dets: List[Det]) -> List[int]:
        """Return an ``object_id`` per detection (parallel to ``dets``)."""
        live_ids = [tid for tid, tr in self._tracks.items()
                    if frame_idx - tr.last_seen_frame <= self.miss_tolerance]
        if not live_ids or not dets:
            return self._spawn_all(frame_idx, dets)

        # Build cost matrix (rows = live tracks, cols = current dets).
        M = np.full((len(live_ids), len(dets)), np.inf, dtype=np.float64)
        for i, tid in enumerate(live_ids):
            tr = self._tracks[tid]
            for j, d in enumerate(dets):
                M[i, j] = self._cost(tr, d)

        # Hungarian can't cope with all-inf rows — replace with a large
        # number, then filter out assignments that exceed the threshold.
        BIG = 1e6
        finite = np.where(np.isfinite(M), M, BIG)
        row_ind, col_ind = linear_sum_assignment(finite)

        assigned_det = [-1] * len(dets)
        used_row = set()
        for r, c in zip(row_ind, col_ind):
            if M[r, c] > self.match_threshold:
                continue
            assigned_det[c] = live_ids[r]
            used_row.add(r)

        # For matched detections, update the track. For unmatched, spawn.
        out_ids: List[int] = []
        for j, d in enumerate(dets):
            if assigned_det[j] >= 0:
                tid = assigned_det[j]
                tr = self._tracks[tid]
                tr.box = d.box
                tr.mask = d.mask
                tr.score = d.score
                tr.last_seen_frame = frame_idx
                out_ids.append(tid)
            else:
                out_ids.append(self._new_track(frame_idx, d))
        return out_ids

    def _spawn_all(self, frame_idx: int, dets: List[Det]) -> List[int]:
        return [self._new_track(frame_idx, d) for d in dets]

    def _new_track(self, frame_idx: int, d: Det) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = Track(
            track_id=tid, label=d.label, box=d.box, mask=d.mask,
            score=d.score, last_seen_frame=frame_idx)
        return tid


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def process_dataset(dataset_root: str,
                    det_dir: str = "detection_boxes",
                    out_dir: str = "detection_h",
                    miss_tolerance: int = 15,
                    match_threshold: float = 0.5,
                    min_score: float = 0.0) -> None:
    """Read OWL+SAM JSONs from ``det_dir`` and write tracked JSONs to ``out_dir``.

    Output file name: ``detection_<frame_id>_final.json`` — matches the
    format ``mask_extractor._extract_by_json`` expects.
    """
    in_folder = os.path.join(dataset_root, det_dir)
    out_folder = os.path.join(dataset_root, out_dir)
    os.makedirs(out_folder, exist_ok=True)

    if not os.path.isdir(in_folder):
        # Most OWL runs parameterize by score threshold and create a
        # directory like ``detection_boxes_0.02``; try that fallback.
        candidates = [f for f in os.listdir(dataset_root)
                      if f.startswith(det_dir + "_")]
        if candidates:
            in_folder = os.path.join(dataset_root, sorted(candidates)[0])
            print(f"[info] using detections at {in_folder}")
        else:
            raise FileNotFoundError(
                f"no detection JSONs found under {dataset_root}")

    files = sorted(f for f in os.listdir(in_folder)
                   if f.startswith("detection_") and f.endswith(".json")
                   and not f.endswith("_final.json"))

    tracker = Tracker(miss_tolerance=miss_tolerance,
                      match_threshold=match_threshold)

    for f in files:
        stem = f[len("detection_"):-len(".json")]
        try:
            fid = int(stem)
        except ValueError:
            # Skip already-tracked files or anything non-numeric.
            continue

        dets = _parse_frame(os.path.join(in_folder, f))
        dets = [d for d in dets if d.score >= min_score]
        ids = tracker.step(fid, dets)

        # Rewrite the source JSON with object_id and normalized top-level
        # label/score/mask fields (in case the OWL stage kept the nested
        # ``detection`` list).
        out = {"detections": []}
        for d, oid in zip(dets, ids):
            rec = dict(d.raw_json)
            rec["label"] = d.label
            rec["score"] = d.score
            rec["box"] = list(d.box)
            rec["object_id"] = int(oid)
            # keep the existing 'mask' field as-is (base64)
            # drop the OWL-style nested 'detection' list to avoid confusion
            rec.pop("detection", None)
            out["detections"].append(rec)

        out_path = os.path.join(out_folder,
                                f"detection_{fid:06d}_final.json")
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2)

    print(f"[done] {dataset_root}: "
          f"{len(files)} frames → {out_folder}, "
          f"{tracker._next_id} total tracks")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="dataset root (containing rgb/, detection_boxes/ …)")
    ap.add_argument("--det-dir", default="detection_boxes",
                    help="subdir with OWL+SAM JSONs (default: detection_boxes)")
    ap.add_argument("--out-dir", default="detection_h",
                    help="subdir for tracked JSONs (default: detection_h)")
    ap.add_argument("--miss-tolerance", type=int, default=15,
                    help="frames an unmatched track survives (default: 15)")
    ap.add_argument("--match-threshold", type=float, default=0.5,
                    help="max Hungarian cost to accept a match (default: 0.5)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop detections below this OWL score (default: 0)")
    args = ap.parse_args()

    process_dataset(
        dataset_root=args.dataset,
        det_dir=args.det_dir,
        out_dir=args.out_dir,
        miss_tolerance=args.miss_tolerance,
        match_threshold=args.match_threshold,
        min_score=args.min_score,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

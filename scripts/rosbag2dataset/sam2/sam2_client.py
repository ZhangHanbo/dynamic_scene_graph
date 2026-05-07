#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM2 video-predictor client — replaces the per-frame SAM + Hungarian tracker.

Pipeline position:

    owl_client.py                   →  detection_boxes/*.json   (boxes)
    THIS FILE (sam2_client.py)      →  detection_h/*_final.json (masks + stable object_ids)

Algorithm
─────────
1. Upload every RGB frame of the dataset to the SAM2 server in one call
   (/sam2_start_session); this materialises the video in GPU memory and
   returns a session_id.
2. Seed the tracker: every OWL detection in frame 0 becomes an initial
   prompt via /sam2_add_box with a fresh ``object_id``.
3. /sam2_propagate — get SAM2's masks for every frame, given the prompts.
4. Walk forward through the video. At each frame, compare OWL's boxes
   against SAM2's current propagated masks. A box whose IoU against every
   same-class mask is below ``new_obj_iou`` is treated as a newly-appeared
   object: it's added as a new prompt (fresh object_id, current frame)
   and we'll re-propagate on the next round.
5. Repeat (3)–(4) until no new prompts fire, or ``max_iters`` is hit.
6. Write ``detection_h/detection_NNNNNN_final.json`` in the existing
   schema so the visualiser and downstream consumers need no changes.
7. /sam2_close_session — free the GPU memory.

Why OWL + SAM2 together?
    OWL tells us *which classes* exist at every frame (and detects new
    instances when they enter the scene). SAM2 supplies *temporally
    consistent masks with stable identity*. This pairing fixes the
    fragmentation we saw with per-frame SAM + 2-D Hungarian.
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from scipy.optimize import linear_sum_assignment

# Allow running as `python scripts/rosbag2dataset/sam2/sam2_client.py`
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.rosbag2dataset.server_configs import (  # noqa: E402
    SAM2_SERVER_URL,
    # batch / offline (kept for back-compat; not used by the streaming driver)
    SAM2_START_PATH, SAM2_ADD_BOX_PATH,
    SAM2_PROPAGATE_PATH, SAM2_CLOSE_PATH,
    # streaming / online
    SAM2_STREAM_INIT_PATH, SAM2_STREAM_FRAME_PATH,
    SAM2_STREAM_ADD_BOX_PATH, SAM2_STREAM_CLOSE_PATH,
    # stateless image-model endpoint used to compute "instant" masks
    # for each OWL bbox (for mask-vs-mask IoU in the matching cost).
    SAM_MASK_BY_BBOX_PATH,
)


# ---------------------------------------------------------------------------
# HTTP helpers (with retry — SAM2 calls can take >30s)
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: float) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _encode_png(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_mask(b64: Optional[str]) -> Optional[np.ndarray]:
    if not b64:
        return None
    raw = base64.b64decode(b64, validate=True)
    arr = np.array(Image.open(io.BytesIO(raw)))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 127).astype(bool)


def _mask_to_png_b64(mask: np.ndarray) -> str:
    arr = (mask.astype(np.uint8) * 255)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _owl_bbox_masks(rgb: np.ndarray,
                     boxes: List[List[int]],
                     server_url: str = SAM2_SERVER_URL,
                     timeout: float = 60.0
                     ) -> List[Optional[np.ndarray]]:
    """Stateless call to ``/sam_mask_by_bbox``: one image + N bboxes
    -> N boolean masks at the input image resolution. Uses the image
    (not video) model, so this does NOT modify the streaming session
    state of our tracker.

    Returns a list of the same length as ``boxes``. Entries are
    numpy bool arrays of shape ``(H, W)`` or ``None`` for slots the
    server didn't return. The caller's failure handling should
    treat ``None`` as "no instant mask available" and fall back to
    the bbox-vs-mask IoU.

    The server wire format is the legacy SAM v1 one:
    ``{"result": [{"segmentation": base64(pickle(ndarray))}, ...]}``.
    """
    if not boxes:
        return []
    # Server expects a 3-D list: (n_prompts, 1, 4).
    wrapped = [[list(map(int, b))] for b in boxes]
    url = server_url.rstrip("/") + SAM_MASK_BY_BBOX_PATH
    payload = {
        "image":       _encode_png(rgb),
        "bboxes":      wrapped,
        "return_best": True,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    out: List[Optional[np.ndarray]] = []
    for item in data.get("result", []):
        try:
            raw = base64.b64decode(item["segmentation"])
            m = pickle.loads(raw)
            m = np.asarray(m).squeeze()
            if m.ndim == 3:
                m = m[0]
            out.append(m.astype(bool))
        except Exception:
            out.append(None)
    while len(out) < len(boxes):
        out.append(None)
    return out


def _bbox_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _bbox_mask_iou(box_xyxy: List[int],
                   mask: np.ndarray) -> float:
    """IoU of a bbox region against a binary mask."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h, w = mask.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    box_area = (x2 - x1) * (y2 - y1)
    mask_area = int(mask.sum())
    if box_area == 0 or mask_area == 0:
        return 0.0
    inter = int(mask[y1:y2, x1:x2].sum())
    union = box_area + mask_area - inter
    return inter / union if union > 0 else 0.0


def _mask_mask_iou(a: Optional[np.ndarray],
                    b: Optional[np.ndarray]) -> float:
    """Binary mask-vs-mask IoU. Returns 0 on shape mismatch or when
    either mask is empty / None -- callers can detect "no info" by
    pairing this with an explicit None check upstream."""
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 or b_sum == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = a_sum + b_sum - inter
    return inter / union if union > 0 else 0.0


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _bbox_intersection_area(a: List[int], b: List[int]) -> int:
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    return iw * ih


def _bbox_area(b: List[int]) -> int:
    x1, y1, x2, y2 = [int(v) for v in b]
    return max(0, x2 - x1) * max(0, y2 - y1)


# ---------------------------------------------------------------------------
# Track-to-track self-match + vote merge (post-matching duplicate
# consolidation). Uses the same cost FORMULA as the OWL->track
# Hungarian matching (mask-mask IoU + label penalty, no score term
# because tracks don't have a single detection score), applied
# pairwise between active tracks on the current frame. Hungarian
# gives the min-cost pairing; pairs below ``max_cost`` are merged.
# ---------------------------------------------------------------------------

def _merge_tracks_vote(tracks: Dict[int, "TrackState"],
                        oid_keep: int, oid_drop: int) -> None:
    """Average-vote merge: absorb ``oid_drop`` into ``oid_keep``.

    * ``label_scores[lbl][fid]``: when both sides have an entry,
      store their average; otherwise keep whichever side has one.
      Averaging makes same-frame duplicate observations contribute
      equally to future ``best_label_mean`` decisions.
    * ``float_mask``: pixel-wise 50/50 blend -- the "pixel voting"
      the user wants. If one side has no mask, adopt the other
      side's mask directly.
    * ``first_frame``: min of the two.

    After this call, ``tracks[oid_drop]`` is gone; the caller is
    responsible for removing ``oid_drop`` from ``prop.object_masks``
    and ``prop.object_bboxes`` as well.
    """
    tr_a = tracks[oid_keep]
    tr_b = tracks[oid_drop]
    for lbl, sbf_b in tr_b.label_scores.items():
        target = tr_a.label_scores.setdefault(lbl, {})
        for fi, sc in sbf_b.items():
            if fi in target:
                target[fi] = 0.5 * (float(target[fi]) + float(sc))
            else:
                target[fi] = float(sc)
    tr_a.first_frame = min(tr_a.first_frame, tr_b.first_frame)
    if tr_b.float_mask is not None:
        if tr_a.float_mask is None:
            tr_a.float_mask = tr_b.float_mask.copy()
        elif tr_a.float_mask.shape == tr_b.float_mask.shape:
            tr_a.float_mask = (0.5 * tr_a.float_mask
                                + 0.5 * tr_b.float_mask)
    tracks.pop(oid_drop, None)


def _self_match_and_merge_tracks(
    tracks: Dict[int, "TrackState"],
    prop: "PropagatedFrame",
    max_cost: float,
    label_penalty: float,
) -> List[Tuple[int, int, float]]:
    """Post-matching Hungarian self-match across active tracks.

    Cost per pair:
        C(t_i, t_j) = (1 - IoU_mask(m_i, m_j)) + label_pen(t_i, t_j)
    where label_pen is 0 when ``best_label`` agrees and
    ``label_penalty`` otherwise. No confidence term -- tracks don't
    have a single score.

    Hungarian on the N x N symmetric cost matrix (diag = +inf) picks
    an optimal 1-to-1 pairing. We dedupe (i, j) vs (j, i) and apply
    merges in ascending cost order, skipping any pair whose keep or
    drop has already been consumed by an earlier merge. This handles
    the case of three+ mutually-duplicate tracks collapsing into one.

    Winner selection (``keep`` vs ``drop``):
        1. more ``total_observations`` wins
        2. ties broken by earlier ``first_frame`` (so identical-
           evidence pairs have a deterministic outcome).

    Returns the list of applied merges as
    ``(keep_oid, drop_oid, cost)``.
    """
    active = [(oid, mask) for oid, mask in prop.object_masks.items()
              if oid in tracks and mask is not None
              and int(mask.sum()) > 0]
    n = len(active)
    if n <= 1:
        return []
    BIG = 1e6
    cost_mx = np.full((n, n), BIG, dtype=np.float64)
    for i in range(n):
        oid_i, m_i = active[i]
        lbl_i = tracks[oid_i].label
        for j in range(i + 1, n):
            oid_j, m_j = active[j]
            iou = _mask_mask_iou(m_i, m_j)
            if iou <= 0.0:
                continue
            lbl_j = tracks[oid_j].label
            pen = 0.0 if lbl_i == lbl_j else label_penalty
            c = (1.0 - iou) + pen
            cost_mx[i, j] = c
            cost_mx[j, i] = c
    row_ind, col_ind = linear_sum_assignment(cost_mx)
    pairs_seen: set = set()
    candidates: List[Tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        if r == c:
            continue
        if cost_mx[r, c] > max_cost:
            continue
        key = frozenset((int(r), int(c)))
        if key in pairs_seen:
            continue
        pairs_seen.add(key)
        candidates.append((int(r), int(c), float(cost_mx[r, c])))
    candidates.sort(key=lambda x: x[2])
    applied: List[Tuple[int, int, float]] = []
    alive = set(oid for oid, _ in active)
    for r, c, cost in candidates:
        oid_r = active[r][0]
        oid_c = active[c][0]
        if oid_r not in alive or oid_c not in alive:
            continue
        n_r = tracks[oid_r].total_observations()
        n_c = tracks[oid_c].total_observations()
        if n_r > n_c:
            keep, drop = oid_r, oid_c
        elif n_c > n_r:
            keep, drop = oid_c, oid_r
        else:
            if tracks[oid_r].first_frame <= tracks[oid_c].first_frame:
                keep, drop = oid_r, oid_c
            else:
                keep, drop = oid_c, oid_r
        _merge_tracks_vote(tracks, keep, drop)
        prop.object_masks.pop(drop, None)
        prop.object_bboxes.pop(drop, None)
        alive.discard(drop)
        applied.append((keep, drop, cost))
    return applied


# ---------------------------------------------------------------------------
# Debug dump helpers (shared between the matcher and the visualiser)
# ---------------------------------------------------------------------------

def _dump_tracks_for_debug(prop: "PropagatedFrame",
                            tracks: Dict[int, "TrackState"],
                            include_dormant: bool = False,
                            track_last_valid: Optional[Dict[int, Tuple[int, List[int]]]] = None,
                            current_i: int = 0,
                            dormant_window: int = 30) -> List[Dict[str, Any]]:
    """Snapshot the current track population for one frame's debug
    JSON. ``prop`` supplies masks for active tracks; ``track_last_valid``
    + ``current_i`` + ``dormant_window`` bring in dormant tracks (recent
    last-valid bbox, no mask). Masks are PNG-base64 for portability."""
    out: List[Dict[str, Any]] = []
    for oid, mask in prop.object_masks.items():
        if oid not in tracks:
            continue
        if mask is None or int(mask.sum()) == 0:
            continue
        tr = tracks[oid]
        out.append({
            "oid": int(oid),
            "label": tr.label,
            "box": list(map(int, _bbox_from_mask(mask))),
            "mask_b64": _mask_to_png_b64(mask),
            "is_dormant": False,
        })
    if include_dormant and track_last_valid is not None:
        for oid in tracks:
            if oid in prop.object_masks:
                mm = prop.object_masks[oid]
                if mm is not None and int(mm.sum()) > 0:
                    continue
            lv = track_last_valid.get(oid)
            if lv is None:
                continue
            last_i, last_bbox = lv
            if current_i - last_i > dormant_window:
                continue
            out.append({
                "oid": int(oid),
                "label": tracks[oid].label,
                "box": list(map(int, last_bbox)),
                "mask_b64": None,
                "is_dormant": True,
                "last_valid_client_idx": int(last_i),
            })
    return out


# ---------------------------------------------------------------------------
# New-prompt clustering
# ---------------------------------------------------------------------------

def _cluster_new_prompts(new_prompts, iou_thresh: float = 0.3):
    """Collapse a list of ``(video_idx, OwlDet)`` into one entry per
    plausible distinct object.

    Two entries belong to the same cluster iff:
      * same label, and
      * bbox IoU with the cluster's rolling "last seen" box ≥ iou_thresh.

    The list is processed in (label, video_idx) order so that a detection
    smoothly moving across consecutive frames ends up in a single cluster.
    Representative entry = highest-score frame in the cluster — gives the
    cleanest prompt to seed SAM2 with.
    """
    if not new_prompts:
        return []
    from collections import defaultdict
    by_label = defaultdict(list)
    for video_idx, owl in new_prompts:
        by_label[owl.label].append((video_idx, owl))

    out = []
    for label, items in by_label.items():
        items.sort(key=lambda p: p[0])
        clusters = []           # list of list of (video_idx, owl)
        for video_idx, owl in items:
            placed = False
            for cl in clusters:
                _, last_owl = cl[-1]
                if _bbox_iou(last_owl.box, owl.box) >= iou_thresh:
                    cl.append((video_idx, owl))
                    placed = True
                    break
            if not placed:
                clusters.append([(video_idx, owl)])
        for cl in clusters:
            video_idx, rep = max(cl, key=lambda p: p[1].score)
            out.append((video_idx, rep))
    return out


# ---------------------------------------------------------------------------
# Dataset IO
# ---------------------------------------------------------------------------

@dataclass
class OwlDet:
    frame_idx: int
    label: str
    score: float
    box: List[int]            # [x1, y1, x2, y2] pixel coords


def _load_owl_detections(det_dir: str,
                         min_score: float = 0.0) -> Dict[int, List[OwlDet]]:
    """Return {frame_idx: [OwlDet, ...]} from the OWL JSONs."""
    files = sorted(f for f in os.listdir(det_dir)
                   if f.startswith("detection_") and f.endswith(".json")
                   and not f.endswith("_final.json"))
    out: Dict[int, List[OwlDet]] = {}
    for fn in files:
        stem = fn[len("detection_"):-len(".json")]
        try:
            fid = int(stem)
        except ValueError:
            continue
        with open(os.path.join(det_dir, fn), "r") as f:
            data = json.load(f)
        dets: List[OwlDet] = []
        for d in data.get("detections", []):
            if "detection" in d and d["detection"]:
                top = max(d["detection"], key=lambda x: x.get("score", 0.0))
                label = str(top.get("label", "unknown"))
                score = float(top.get("score", 0.0))
            else:
                label = str(d.get("label", "unknown"))
                score = float(d.get("score", 0.0))
            box = d.get("box")
            if not box or len(box) != 4:
                continue
            if score < min_score:
                continue
            dets.append(OwlDet(frame_idx=fid, label=label, score=score,
                               box=[int(v) for v in box]))
        out[fid] = dets
    return out


def _load_frames(rgb_dir: str) -> Tuple[List[int], List[np.ndarray]]:
    files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    frame_ids: List[int] = []
    frames: List[np.ndarray] = []
    for fn in files:
        try:
            fid = int(os.path.splitext(fn)[0].split("_")[-1])
        except ValueError:
            continue
        img = Image.open(os.path.join(rgb_dir, fn)).convert("RGB")
        frame_ids.append(fid)
        frames.append(np.asarray(img, dtype=np.uint8))
    return frame_ids, frames


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TrackState:
    """Client-side metadata parallel to SAM2's internal per-object state.

    A single SAM2 track can collect observations from multiple OWL
    labels over time -- e.g. an apple originally seeded as ``apple``
    is later detected as ``cup`` at some frames because OWL is
    confused. We keep per-label score histories and expose ``label``/
    ``score``/``scores_by_frame`` as derived views of the label with
    the highest mean score, so call sites that read those fields get
    the best current classification automatically.

    Each track also maintains a float-valued EMA mask
    (``float_mask``, shape (H, W), float32). Every frame's SAM2
    binary mask gets blended in via
    ``float_mask = alpha*new_binary + (1-alpha)*float_mask``, so the
    exported mask smooths over SAM2's per-frame jitter. Consumers
    binarise the exported mask at ``>= 0.5``.
    """

    def __init__(self, label: str, first_frame: int, score: float) -> None:
        self.first_frame = first_frame
        self.seed_label = label
        self.seed_score = float(score)
        # label -> {frame_idx: owl_score}.
        self.label_scores: Dict[str, Dict[int, float]] = {label: {}}
        # Exponentially-averaged float mask; None until first observed.
        self.float_mask: Optional[np.ndarray] = None

    def observe(self, label: str, frame_idx: int, score: float) -> None:
        """Record a new OWL observation for this track, under ``label``."""
        self.label_scores.setdefault(label, {})[int(frame_idx)] = float(score)

    def update_mask(self, new_binary_mask: np.ndarray,
                     alpha: float = 0.5) -> None:
        """EMA-update the float mask with a new SAM2 observation.

        First valid observation initialises ``float_mask`` as a float
        copy of the binary mask. Subsequent updates blend pixel-wise:
        ``float_mask = alpha*new + (1-alpha)*float_mask``. Shape
        mismatches (shouldn't happen within one session) re-initialise.
        """
        if new_binary_mask is None:
            return
        new_float = new_binary_mask.astype(np.float32)
        if self.float_mask is None or self.float_mask.shape != new_float.shape:
            self.float_mask = new_float
            return
        self.float_mask = (alpha * new_float
                            + (1.0 - alpha) * self.float_mask)

    def binary_mask(self, threshold: float = 0.5) -> Optional[np.ndarray]:
        """Return the EMA mask thresholded at ``threshold`` (default
        0.5), or None if the track has no observations yet."""
        if self.float_mask is None:
            return None
        return self.float_mask >= threshold

    def best_label_mean(self) -> Tuple[str, float]:
        """Return ``(label, mean_score)`` for the label with the highest
        mean OWL score across all frames it was observed on. Falls back
        to the seed label + seed score when no observations exist."""
        best_lbl = self.seed_label
        best_mean = -1.0
        for lbl, sbf in self.label_scores.items():
            if not sbf:
                continue
            mean = sum(sbf.values()) / len(sbf)
            if mean > best_mean:
                best_mean = mean
                best_lbl = lbl
        if best_mean < 0:
            return self.seed_label, self.seed_score
        return best_lbl, best_mean

    def score_at_frame(self, frame_idx: int) -> Optional[float]:
        """Return the OWL score recorded for ``frame_idx`` under any
        label (max if multiple labels fired on the same frame)."""
        fi = int(frame_idx)
        found: List[float] = []
        for sbf in self.label_scores.values():
            if fi in sbf:
                found.append(sbf[fi])
        return max(found) if found else None

    def total_observations(self) -> int:
        """Number of (label, frame) observations across every label."""
        return sum(len(sbf) for sbf in self.label_scores.values())

    # --- back-compat views used by the rest of the driver / writers. ---

    @property
    def label(self) -> str:
        return self.best_label_mean()[0]

    @property
    def score(self) -> float:
        return self.best_label_mean()[1]

    @property
    def scores_by_frame(self) -> Dict[int, float]:
        """Score-by-frame for the currently-best label. Read-only as a
        shim -- write with ``observe(label, frame, score)`` instead."""
        return self.label_scores.get(self.best_label_mean()[0], {})


@dataclass
class PropagatedFrame:
    """Masks that SAM2 returns for one frame."""
    object_masks: Dict[int, np.ndarray]   # object_id -> (H, W) bool
    object_bboxes: Dict[int, List[int]]


class SAM2StreamClient:
    """Thin HTTP client for the SAM2 streaming server.

    Model: one frame at a time. Each /sam2_stream_frame appends the
    frame to the session, propagates every currently-seeded object into
    it via the memory bank, and returns per-object masks immediately.
    Prompts can be dropped at ANY frame_idx via /sam2_stream_add_box;
    those objects start showing up in the next frame's propagated mask
    set.
    """
    def __init__(self, server_url: str = SAM2_SERVER_URL,
                 timeout_init: float = 60.0,
                 timeout_frame: float = 120.0,
                 timeout_prompt: float = 60.0,
                 timeout_close: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.t_init = timeout_init
        self.t_frame = timeout_frame
        self.t_prompt = timeout_prompt
        self.t_close = timeout_close
        self.session_id: Optional[str] = None

    # -- session lifecycle --------------------------------------------------

    def start(self) -> dict:
        r = _post(self.server_url + SAM2_STREAM_INIT_PATH,
                  {}, timeout=self.t_init)
        self.session_id = r["session_id"]
        return r

    def close(self) -> None:
        if self.session_id is None:
            return
        try:
            _post(self.server_url + SAM2_STREAM_CLOSE_PATH,
                  {"session_id": self.session_id},
                  timeout=self.t_close)
        except Exception as e:
            print(f"[sam2-stream] close failed: {e}")
        self.session_id = None

    def __enter__(self) -> "SAM2StreamClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- frame / prompt -----------------------------------------------------

    def frame(self, img_rgb: np.ndarray) -> PropagatedFrame:
        """Feed one frame; return masks for every currently-seeded track
        propagated into this frame."""
        assert self.session_id is not None, "session not started"
        r = _post(
            self.server_url + SAM2_STREAM_FRAME_PATH, {
                "session_id": self.session_id,
                "image":      _encode_png(img_rgb),
            }, timeout=self.t_frame)
        masks: Dict[int, np.ndarray] = {}
        bboxes: Dict[int, List[int]] = {}
        for o in r.get("objects", []):
            oid = int(o["object_id"])
            m = _decode_mask(o.get("mask"))
            if m is None:
                continue
            masks[oid] = m
            bboxes[oid] = o.get("bbox") or _bbox_from_mask(m)
        return PropagatedFrame(object_masks=masks, object_bboxes=bboxes)

    def add_box(self, frame_idx: int, box: List[int],
                 object_id: Optional[int] = None) -> int:
        """Seed a new object at frame_idx with a bbox prompt. If
        object_id is None the server mints a fresh one. Returns the
        (resolved) object id.

        The new object will not appear in `self.frame(frame_idx)`'s
        return -- SAM2 propagates it into subsequent frames, so its
        first SAM2-predicted mask is at frame_idx + 1.
        """
        assert self.session_id is not None, "session not started"
        payload = {
            "session_id": self.session_id,
            "frame_idx":  int(frame_idx),
            "box":        [float(v) for v in box],
        }
        if object_id is not None:
            payload["object_id"] = int(object_id)
        r = _post(self.server_url + SAM2_STREAM_ADD_BOX_PATH, payload,
                   timeout=self.t_prompt)
        return int(r["object_id"])


class SAM2Client:
    def __init__(self, server_url: str = SAM2_SERVER_URL,
                 timeout_start: float = 300.0,
                 timeout_prompt: float = 60.0,
                 timeout_propagate: float = 1800.0):
        self.server_url = server_url.rstrip("/")
        self.t_start = timeout_start
        self.t_prompt = timeout_prompt
        self.t_prop = timeout_propagate
        self.session_id: Optional[str] = None

    # -- session lifecycle --------------------------------------------------

    def start(self, frames: List[np.ndarray]) -> dict:
        frames_b64 = [_encode_png(f) for f in frames]
        t0 = time.time()
        r = _post(self.server_url + SAM2_START_PATH,
                  {"frames": frames_b64}, timeout=self.t_start)
        self.session_id = r["session_id"]
        print(f"[sam2] session {self.session_id}  "
              f"{r['n_frames']} frames  {r['width']}x{r['height']}  "
              f"upload={time.time()-t0:.1f}s")
        return r

    def close(self) -> None:
        if self.session_id is None:
            return
        try:
            _post(self.server_url + SAM2_CLOSE_PATH,
                  {"session_id": self.session_id}, timeout=30.0)
        except Exception as e:
            print(f"[sam2] close failed: {e}")
        self.session_id = None

    def __enter__(self) -> "SAM2Client":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- prompts / propagate ------------------------------------------------

    def add_box(self, frame_idx: int, object_id: int,
                box: List[int]) -> np.ndarray:
        assert self.session_id is not None, "session not started"
        r = _post(self.server_url + SAM2_ADD_BOX_PATH, {
            "session_id": self.session_id,
            "frame_idx":  int(frame_idx),
            "object_id":  int(object_id),
            "box":        [float(v) for v in box],
        }, timeout=self.t_prompt)
        return _decode_mask(r.get("mask"))

    def propagate(self) -> List[PropagatedFrame]:
        assert self.session_id is not None, "session not started"
        r = _post(self.server_url + SAM2_PROPAGATE_PATH,
                  {"session_id": self.session_id}, timeout=self.t_prop)
        out: Dict[int, PropagatedFrame] = {}
        for frame in r.get("results", []):
            masks: Dict[int, np.ndarray] = {}
            bboxes: Dict[int, List[int]] = {}
            for o in frame.get("objects", []):
                oid = int(o["object_id"])
                m = _decode_mask(o.get("mask"))
                if m is None:
                    continue
                masks[oid] = m
                bboxes[oid] = o.get("bbox") or _bbox_from_mask(m)
            out[int(frame["frame_idx"])] = PropagatedFrame(
                object_masks=masks, object_bboxes=bboxes)
        # Return as list sorted by frame_idx
        return [out[f] for f in sorted(out.keys())]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def track_dataset_streaming(dataset_root: str,
                             server_url: str = SAM2_SERVER_URL,
                             det_dir_name: str = "detection_boxes",
                             out_dir_name: str = "detection_h",
                             min_score: float = 0.0,
                             hungarian_max_cost: float = 0.7,
                             hungarian_label_penalty: float = 0.2,
                             hungarian_score_weight: float = 0.15,
                             new_seed_min_score: float = 0.25,
                             self_match_max_cost: float = 0.3,
                             use_owl_masks: bool = True,
                             use_fallback: bool = True,
                             owl_mask_timeout: float = 60.0,
                             reset_every: int = 150,
                             reset_buffer_max: int = 100,
                             reset_buffer_interval: int = 3,
                             debug_dir: Optional[str] = None,
                             debug_fid_start: int = 0,
                             debug_fid_end: int = 10 ** 9) -> None:
    """Streaming SAM2 tracker.

    Three server-side SAM2 quirks the driver works around:

    * ``/sam2_stream_frame`` called before any prompt is registered
      returns ``objects=[]`` without advancing the server's
      ``s.frame_count`` — it only records ``original_size`` so the
      caller can add prompts and resend the frame.
    * The HF Transformers Sam2VideoModel rejects successive
      ``add_inputs_to_inference_session`` calls that share the same
      ``frame_idx`` without a ``video_model(frame=...)`` call in
      between. The second-and-later calls put the session into a
      corrupted state ("maskmem_features in conditioning outputs
      cannot be empty when not is_initial_conditioning_frame"). We
      therefore treat each prompt as a separate SAM2 frame: each
      unmatched OWL detection is added at ``frame_idx =
      sam2_frame_count`` and immediately committed by a
      ``/sam2_stream_frame`` call on the SAME RGB image, which
      advances ``sam2_frame_count`` and registers that prompt in
      SAM2's memory bank without showing the model a new image.
    * The session's GPU memory grows unboundedly with
      ``sam2_frame_count × n_objects`` -- every processed frame
      caches features and every ``add_box`` creates a conditioning
      frame that never gets evicted. On long or track-dense
      trajectories the server 500s with a CUDA OOM. We mitigate by
      resetting the session every ``reset_every`` SAM2 frames:
      close, reopen, then REPLAY a rolling buffer of recent frames
      (``reset_buffer_max`` entries sampled every
      ``reset_buffer_interval`` client frames) so SAM2 rebuilds a
      multi-frame memory bank per track. For each live track we
      condition it at the EARLIEST and LATEST buffered frames where
      it had a valid mask (two anchors per track), then let SAM2
      propagate through the remaining buffered frames. Tracks that
      have no current valid mask at reset time or no bbox_history
      are dropped -- losing them here beats polluting the fresh
      session with stale bboxes that cascade into track-explosion.

    Per dataset frame (clean 5-step flow):
      1. Load OWL detections at this frame.
      2. SAM2 propagate current tracks into this frame.
      3. Hungarian-match OWL dets to tracks. Cost per pair is
         ``(1 - IoU) + label_penalty + score_weight * (1 - score)``
         where IoU is bbox-vs-mask for active tracks (current prop
         mask) and bbox-vs-bbox for dormant tracks (last-valid
         bbox, within DORMANT_MATCH_WINDOW). ``label_penalty`` is
         0 when the OWL label already appears in the track's
         label_scores history, ``hungarian_label_penalty`` otherwise
         -- a soft mismatch cost, not a veto. The score term
         penalises low-confidence detections so they have lower
         matching priority. Pairs with cost > ``hungarian_max_cost``
         are rejected. Matched pairs observe the OWL label on the
         track (label_scores bucket).
      4. Merge unmatched dets: each det Hungarian didn't accept
         re-checks every track with the same cost formula (single
         direction, greedy). If the best cost is <= max, absorb
         into that track (multiple dets can merge into one track).
         Otherwise, if the det's OWL score >= new_seed_min_score,
         seed a new track via seed_one; else drop it.
      5. Track-to-track self-match: reusing the same cost formula
         (mask-mask IoU + label penalty), Hungarian-pair the active
         tracks against themselves. Pairs below
         ``self_match_max_cost`` get merged by vote -- float_masks
         are 50/50 averaged, label_scores are averaged at shared
         (label, frame) cells and inherited at the rest. The
         surviving oid is the one with more observations (tie-break
         by earlier first_frame). This consolidates duplicates that
         OWL->track cannot touch because its cost matrix has no
         track-vs-track cells.

    Side concern: if sam2_frame_count >= reset_every we close+reopen
    the SAM2 session and reseed live tracks -- purely a GPU-memory
    bound, orthogonal to the matching pipeline.
    """
    rgb_dir = os.path.join(dataset_root, "rgb")
    det_dir = (det_dir_name if os.path.isabs(det_dir_name)
               else os.path.join(dataset_root, det_dir_name))
    out_dir = (out_dir_name if os.path.isabs(out_dir_name)
               else os.path.join(dataset_root, out_dir_name))
    os.makedirs(out_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[sam2-stream] debug dumps -> {debug_dir}  "
              f"fid range=[{debug_fid_start}, {debug_fid_end}]")

    print(f"[sam2-stream] dataset={dataset_root}  server={server_url}")
    frame_ids, frames = _load_frames(rgb_dir)
    if not frames:
        raise FileNotFoundError(f"no RGB in {rgb_dir}")
    owl_dets = _load_owl_detections(det_dir, min_score=min_score)
    mean_ow = np.mean([len(v) for v in owl_dets.values()] or [0])
    print(f"[sam2-stream] frames={len(frames)}  "
          f"owl-dets-per-frame (mean)={mean_ow:.1f}")

    tracks: Dict[int, TrackState] = {}
    # Mirror of the server's ``s.frame_count``. Every call to
    # ``/sam2_stream_frame`` that actually runs the model increments
    # both sides by 1; the peek-when-empty call does not. ``add_box``
    # must target this index, not the client loop index ``i`` — they
    # diverge for short-circuited frames (no OWL dets and no tracks
    # yet) and for the per-prompt commit forwards below.
    sam2_frame_count = 0
    empty_prop = PropagatedFrame(object_masks={}, object_bboxes={})
    t0 = time.time()

    def seed_one(owl: OwlDet, rgb_arr: np.ndarray,
                 client_idx: int) -> PropagatedFrame:
        """Add a single OWL detection as a new track and commit it by
        running one SAM2 forward on the same RGB image.

        SAM2 rejects multiple ``add_inputs_to_inference_session`` calls
        at the same ``frame_idx`` without an intervening forward, so we
        give each prompt its own ``frame_idx = sam2_frame_count`` and
        immediately call ``/sam2_stream_frame`` to advance the counter.
        Passing the same RGB on every commit keeps the image content
        frozen while SAM2 reconciles its memory bank — the returned
        masks are always for this frame and always include every
        already-seeded track plus the newly-added one.

        The new track's ``float_mask`` is initialised from the
        committed mask so the exported (EMA-binarised) mask is
        immediately populated for this frame's output.
        """
        nonlocal sam2_frame_count
        obj_id = sam2.add_box(frame_idx=sam2_frame_count, box=owl.box)
        tracks[obj_id] = TrackState(
            label=owl.label, first_frame=client_idx, score=owl.score)
        tracks[obj_id].observe(owl.label, client_idx, owl.score)
        committed = sam2.frame(rgb_arr)
        sam2_frame_count += 1
        new_mask = committed.object_masks.get(obj_id)
        if new_mask is not None:
            tracks[obj_id].update_mask(new_mask, alpha=0.5)
        return committed

    with SAM2StreamClient(server_url=server_url) as sam2:
        sam2.start()

        last_prop = empty_prop
        reset_count = 0

        # Rolling history for warm-restart + dormant-track matching.
        # frame_buffer stores numpy refs into ``frames`` (no copies).
        # bbox_history remembers each track's bbox at every BUFFERED
        # frame (used by reset_session's replay). track_last_valid
        # remembers EACH track's most-recent valid-mask bbox + the
        # client frame it was observed on -- used for the dormant
        # match that prevents cascading new-seed when SAM2
        # temporarily loses a track's mask.
        frame_buffer: List[Tuple[int, np.ndarray]] = []
        bbox_history: Dict[int, Dict[int, List[int]]] = {}
        track_last_valid: Dict[int, Tuple[int, List[int]]] = {}
        # Match against dormant (no-mask-right-now) tracks only if
        # their last-valid observation is within this many client
        # frames -- beyond that they're stale, drop them when reset
        # runs or let OWL seed a fresh track.
        DORMANT_MATCH_WINDOW = 30

        def _note_buffer(i: int, rgb: np.ndarray,
                         prop: PropagatedFrame) -> None:
            """Append this frame to the rolling buffer (at the
            sampled cadence) and snapshot every live track's bbox
            at this moment. Called AFTER per-frame processing so
            ``prop`` reflects matches + new seeds."""
            if reset_buffer_interval <= 0 or reset_buffer_max <= 0:
                return
            if i % reset_buffer_interval != 0:
                return
            frame_buffer.append((i, rgb))
            for oid, mask in prop.object_masks.items():
                if oid not in tracks:
                    continue
                if mask is None or int(mask.sum()) == 0:
                    continue
                bbox_history.setdefault(oid, {})[i] = _bbox_from_mask(mask)
            while len(frame_buffer) > reset_buffer_max:
                evicted_i, _ = frame_buffer.pop(0)
                for d in bbox_history.values():
                    d.pop(evicted_i, None)

        def reset_session(rgb_now: np.ndarray,
                          current_client_i: int) -> PropagatedFrame:
            """Close the current SAM2 session, start a fresh one, and
            re-seed every live track from the bbox of its current mask.

            Tracks with no valid mask in ``last_prop`` at reset time
            are dropped -- SAM2 wasn't tracking them anymore. Client
            side oid/label/score history survives the reset verbatim.
            sam2_frame_count is rebased to 0 on both sides.

            Note: the rolling frame_buffer + bbox_history are kept
            up-to-date elsewhere for potential use by downstream
            logic (e.g. dormant-track matching) -- they're not used
            by the reset itself here.
            """
            nonlocal sam2_frame_count, reset_count
            # Snapshot live tracks whose EMA mask is non-empty at
            # threshold 0.5. Using the smoothed mask (instead of
            # SAM2's raw prop) makes the reseed bbox more stable
            # across brief occlusions. Tracks absent from last_prop
            # AND with empty float_mask are gone.
            live: List[Tuple[int, List[int]]] = []
            for oid in list(tracks.keys()):
                tr = tracks[oid]
                bin_mask = tr.binary_mask(0.5)
                if bin_mask is None or int(bin_mask.sum()) == 0:
                    tracks.pop(oid, None)
                    continue
                live.append((oid, _bbox_from_mask(bin_mask)))

            sam2.close()
            sam2.start()
            sam2_frame_count = 0
            reset_count += 1

            # Peek current frame so the server records original_size
            # (does not advance frame_count).
            sam2.frame(rgb_now)
            new_prop = empty_prop
            for oid, box in live:
                # Preserve the client-side oid. If the server mints a
                # different one, remap the tracks[] key to match.
                committed_oid = sam2.add_box(
                    frame_idx=sam2_frame_count, box=box, object_id=oid)
                if committed_oid != oid and committed_oid not in tracks:
                    tracks[committed_oid] = tracks.pop(oid)
                new_prop = sam2.frame(rgb_now)
                sam2_frame_count += 1
            retained = sum(1 for o in new_prop.object_masks
                           if o in tracks and new_prop.object_masks[o] is not None
                           and int(new_prop.object_masks[o].sum()) > 0)
            print(f"[sam2-stream] RESET #{reset_count}  "
                  f"reseeded={len(live)}  retained={retained}")
            return new_prop

        for i, fid in enumerate(frame_ids):
            rgb = frames[i]
            owls = owl_dets.get(fid, [])
            added_this_frame = 0

            # Debug capture: only active inside the "propagate" branch
            # below (seeding-only frames have no matching to dump).
            debug_enabled = (
                debug_dir is not None
                and debug_fid_start <= fid <= debug_fid_end)
            debug_record: Optional[Dict[str, Any]] = None

            # Short-circuit: nothing seeded yet and no OWL dets to seed
            # with -> no reason to hit the server (it would just record
            # original_size and return empty). Write an empty output so
            # the downstream JSON schema stays complete.
            if not tracks and not owls:
                _write_frame_output(out_dir, fid, i, tracks, empty_prop)
                last_prop = empty_prop
            else:
                if not tracks:
                    # First frame with OWL detections. Peek once to
                    # register original_size on the server, then seed
                    # each OWL det with its own add_box + commit
                    # forward. After the last commit, ``prop`` holds
                    # masks for every seeded track on THIS rgb.
                    sam2.frame(rgb)
                    prop = empty_prop
                    for owl in owls:
                        prop = seed_one(owl, rgb, i)
                        added_this_frame += 1
                else:
                    # Debug: snapshot OWL detections + prior-frame
                    # track state before this frame's propagation.
                    # OWL-bbox-mask PNGs are filled in after the
                    # /sam_mask_by_bbox call below.
                    if debug_enabled:
                        debug_record = {
                            "fid": int(fid),
                            "client_idx": int(i),
                            "frame_shape": [int(rgb.shape[0]),
                                             int(rgb.shape[1])],
                            "owl_dets": [{
                                "box": list(map(int, o.box)),
                                "label": o.label,
                                "score": float(o.score),
                                "mask_b64": None,
                            } for o in owls],
                            "prev_tracks": _dump_tracks_for_debug(
                                last_prop, tracks, include_dormant=True,
                                track_last_valid=track_last_valid,
                                current_i=i,
                                dormant_window=DORMANT_MATCH_WINDOW),
                        }

                    # (2) SAM2 propagation: get this frame's masks
                    # for every currently-seeded track.
                    prop = sam2.frame(rgb)
                    sam2_frame_count += 1

                    # EMA-update each track's float mask from
                    # SAM2's new binary mask. Pure per-track state
                    # maintenance -- no cross-track decisions here;
                    # the exported mask is the smoothed float mask
                    # thresholded at 0.5.
                    for _oid, _mask in prop.object_masks.items():
                        if _oid in tracks and _mask is not None:
                            tracks[_oid].update_mask(_mask, alpha=0.5)

                    # "Instant" masks for each OWL bbox via the
                    # stateless /sam_mask_by_bbox endpoint. Used
                    # below so the Hungarian cost is mask-vs-mask
                    # (for active tracks) instead of bbox-vs-mask.
                    # Returns one bool ndarray per OWL det or None
                    # if the server couldn't segment that box; the
                    # cost function falls back to bbox-vs-mask IoU
                    # for any ``None`` slot.
                    owl_masks: List[Optional[np.ndarray]] = []
                    if owls and use_owl_masks:
                        try:
                            owl_masks = _owl_bbox_masks(
                                rgb, [o.box for o in owls],
                                server_url=server_url,
                                timeout=owl_mask_timeout)
                        except Exception as e:
                            print(f"[sam2-stream] /sam_mask_by_bbox "
                                  f"failed at fid={fid}: {e}")
                            owl_masks = [None] * len(owls)
                    elif owls:
                        # Baseline mode: no OWL SAM masks -- pair cost
                        # falls back to bbox-vs-mask IoU.
                        owl_masks = [None] * len(owls)

                    if debug_record is not None:
                        for _k, _m in enumerate(owl_masks):
                            if _m is not None and bool(_m.any()):
                                debug_record["owl_dets"][_k]["mask_b64"] = (
                                    _mask_to_png_b64(_m))

                    # (3) Hungarian-match OWL dets to tracks.
                    #
                    # Cost model (per pair):
                    #   cost = (1 - IoU) + label_pen + score_pen
                    #   label_pen  = 0 if owl.label in
                    #                  track.label_scores, else
                    #                  hungarian_label_penalty
                    #   score_pen  = hungarian_score_weight *
                    #                  (1 - owl.score)
                    # IoU is bbox-vs-mask for active tracks and
                    # bbox-vs-bbox for dormant tracks (last-valid
                    # bbox within DORMANT_MATCH_WINDOW).
                    # hungarian_label_penalty is a soft cost, not a
                    # veto. hungarian_score_weight deprioritises
                    # low-confidence detections during matching.
                    # Pairs with cost > hungarian_max_cost are
                    # rejected.
                    #
                    # Two passes:
                    #   3a) Hungarian (one-to-one optimal).
                    #   3b) Single-direction greedy fallback for
                    #       dets Hungarian didn't accept: each
                    #       such det re-checks every track and is
                    #       absorbed into the closest one if cost
                    #       <= max. Lets one physical object with
                    #       duplicate OWL detections merge into one
                    #       track rather than spawning spurious new
                    #       tracks.
                    #
                    # Absorption = observe(label, fid, score). The
                    # track's float_mask keeps getting EMA-updated
                    # from SAM2's prop.object_masks every frame.
                    track_entries: List[Tuple[int, np.ndarray, List[int]]] = []
                    # (oid, ref_mask_or_None, ref_bbox_or_None)
                    for _oid, _mask in prop.object_masks.items():
                        if _oid not in tracks:
                            continue
                        if _mask is None or int(_mask.sum()) == 0:
                            continue
                        track_entries.append((_oid, _mask, None))
                    for _oid in tracks:
                        _mm = prop.object_masks.get(_oid)
                        if _mm is not None and int(_mm.sum()) > 0:
                            continue  # already covered as active
                        lv = track_last_valid.get(_oid)
                        if lv is None:
                            continue
                        last_i, last_bbox = lv
                        if i - last_i > DORMANT_MATCH_WINDOW:
                            continue
                        track_entries.append((_oid, None, last_bbox))

                    def _pair_cost(owl_idx: int,
                                    owl: OwlDet,
                                    entry: Tuple[int, Optional[np.ndarray],
                                                 Optional[List[int]]]) -> float:
                        oid, mref, bref = entry
                        det_mask = (owl_masks[owl_idx]
                                     if owl_idx < len(owl_masks)
                                     else None)
                        if mref is not None:
                            # Active track: prefer mask-vs-mask IoU
                            # between the instant OWL mask and the
                            # tracked SAM2 mask; fall back to
                            # bbox-vs-mask if the OWL mask wasn't
                            # available.
                            if (det_mask is not None
                                    and det_mask.shape == mref.shape
                                    and int(det_mask.sum()) > 0):
                                iou = _mask_mask_iou(det_mask, mref)
                            else:
                                iou = _bbox_mask_iou(owl.box, mref)
                        elif bref is not None:
                            # Dormant track: no current mask, only
                            # last-valid bbox -- use bbox-vs-bbox IoU.
                            iou = _bbox_iou(owl.box, bref)
                        else:
                            return float("inf")
                        if iou <= 0.0:
                            return float("inf")
                        label_pen = (
                            0.0 if owl.label in tracks[oid].label_scores
                            else hungarian_label_penalty)
                        score_pen = (hungarian_score_weight
                                      * (1.0 - float(owl.score)))
                        return (1.0 - iou) + label_pen + score_pen

                    matched_det_idx: set = set()
                    N_d = len(owls)
                    M_t = len(track_entries)
                    debug_hungarian: List[Dict[str, Any]] = []
                    debug_fallback: List[Dict[str, Any]] = []
                    debug_new_seeds: List[Dict[str, Any]] = []
                    cost_mx: Optional[np.ndarray] = None
                    if N_d > 0 and M_t > 0:
                        BIG = 1e6
                        cost_mx = np.full((N_d, M_t), BIG, dtype=np.float64)
                        for c_j, entry in enumerate(track_entries):
                            for r_i, owl in enumerate(owls):
                                c = _pair_cost(r_i, owl, entry)
                                if np.isfinite(c):
                                    cost_mx[r_i, c_j] = c
                        row_ind, col_ind = linear_sum_assignment(cost_mx)
                        for r_i, c_j in zip(row_ind, col_ind):
                            c_val = float(cost_mx[r_i, c_j])
                            accepted = c_val <= hungarian_max_cost
                            if debug_record is not None:
                                debug_hungarian.append({
                                    "det_idx": int(r_i),
                                    "track_idx": int(c_j),
                                    "oid": int(track_entries[c_j][0]),
                                    "cost": c_val,
                                    "accepted": bool(accepted),
                                })
                            if not accepted:
                                continue
                            oid = track_entries[c_j][0]
                            tracks[oid].observe(
                                owls[r_i].label, i, owls[r_i].score)
                            matched_det_idx.add(int(r_i))

                    # (4) single-direction greedy fallback + seed new
                    # tracks for dets with no plausible existing
                    # match. The "merge unmatched" step: multiple
                    # unmatched dets can merge into the SAME track
                    # when they're all close to it. New tracks are
                    # ONLY seeded for dets whose OWL score is above
                    # ``new_seed_min_score`` -- low-confidence
                    # detections that don't find a close track are
                    # dropped rather than polluting the tracker
                    # with a fresh spurious track.
                    for j, owl in enumerate(owls):
                        if j in matched_det_idx:
                            continue
                        best_c = hungarian_max_cost + 1.0
                        best_oid: Optional[int] = None
                        if use_fallback:
                            for entry in track_entries:
                                c = _pair_cost(j, owl, entry)
                                if c < best_c:
                                    best_c = c
                                    best_oid = entry[0]
                            if (best_oid is not None
                                    and best_c <= hungarian_max_cost):
                                tracks[best_oid].observe(
                                    owl.label, i, owl.score)
                                if debug_record is not None:
                                    debug_fallback.append({
                                        "det_idx": int(j),
                                        "oid": int(best_oid),
                                        "cost": float(best_c),
                                    })
                                continue
                        if owl.score < new_seed_min_score:
                            if debug_record is not None:
                                debug_new_seeds.append({
                                    "det_idx": int(j),
                                    "oid": -1,
                                    "dropped_low_score": True,
                                    "score": float(owl.score),
                                    "best_reject_cost":
                                        float(best_c) if best_oid is not None
                                        else None,
                                    "best_reject_oid":
                                        int(best_oid) if best_oid is not None
                                        else None,
                                })
                            continue
                        # New seed: diff tracks[] keys to identify the
                        # oid the server minted for this detection.
                        before_oids = set(tracks.keys())
                        prop = seed_one(owl, rgb, i)
                        new_oids = sorted(set(tracks.keys()) - before_oids)
                        added_this_frame += 1
                        # Extend track_entries so LATER dets in this
                        # frame's fallback loop can match against
                        # the freshly-seeded track (prevents same-
                        # frame duplicate seeding when OWL gives
                        # multiple close boxes for one object).
                        for _new_oid in new_oids:
                            _new_mask = prop.object_masks.get(_new_oid)
                            if (_new_mask is not None
                                    and int(_new_mask.sum()) > 0):
                                track_entries.append(
                                    (_new_oid, _new_mask, None))
                        if debug_record is not None:
                            debug_new_seeds.append({
                                "det_idx": int(j),
                                "oid": int(new_oids[0])
                                        if new_oids else -1,
                                "dropped_low_score": False,
                                "score": float(owl.score),
                                "best_reject_cost":
                                    float(best_c) if best_oid is not None
                                    else None,
                                "best_reject_oid":
                                    int(best_oid) if best_oid is not None
                                    else None,
                            })

                    # Debug: snapshot track state AFTER OWL->track
                    # matching (step 4) but BEFORE the track<->track
                    # self-match merge (step 5), so the debug viz
                    # can show what the self-match step actually
                    # changed this frame.
                    if debug_record is not None:
                        debug_record["tracks_before_self_merge"] = (
                            _dump_tracks_for_debug(prop, tracks))

                    # (5) Track-to-track self-match + vote merge.
                    # Reuses the Hungarian cost (mask IoU + label
                    # pen), applied pairwise between active tracks
                    # on this frame. Pairs below self_match_max_cost
                    # are merged by voting/averaging. Consolidates
                    # duplicate SAM2 tracks that the OWL->track
                    # layer can't delete by construction.
                    self_merges: List[Tuple[int, int, float]] = []
                    if self_match_max_cost > 0.0 and len(tracks) >= 2:
                        self_merges = _self_match_and_merge_tracks(
                            tracks, prop,
                            max_cost=self_match_max_cost,
                            label_penalty=hungarian_label_penalty)

                    # Debug: finalise this frame's debug record with
                    # matching internals + the final track state.
                    if debug_record is not None:
                        matching_dbg: Dict[str, Any] = {
                            "max_cost": float(hungarian_max_cost),
                            "label_penalty": float(hungarian_label_penalty),
                            "score_weight": float(hungarian_score_weight),
                            "new_seed_min_score":
                                float(new_seed_min_score),
                            "track_entries": [{
                                "oid": int(e[0]),
                                "kind": ("active" if e[1] is not None
                                         else "dormant"),
                            } for e in track_entries],
                        }
                        if cost_mx is not None:
                            matching_dbg["cost_matrix"] = cost_mx.tolist()
                        matching_dbg["hungarian"] = debug_hungarian
                        matching_dbg["fallback"] = debug_fallback
                        matching_dbg["new_seeds"] = debug_new_seeds
                        matching_dbg["self_merges"] = [
                            {"keep_oid": int(k),
                             "drop_oid": int(d),
                             "cost": float(c)}
                            for k, d, c in self_merges
                        ]
                        matching_dbg["self_match_max_cost"] = (
                            float(self_match_max_cost))
                        debug_record["matching"] = matching_dbg
                        debug_record["final_tracks"] = (
                            _dump_tracks_for_debug(prop, tracks))

                _write_frame_output(out_dir, fid, i, tracks, prop)
                last_prop = prop

                # Debug: dump record for the propagate branch (only).
                if debug_record is not None:
                    dp = os.path.join(debug_dir,
                                       f"debug_{fid:06d}.json")
                    with open(dp, "w") as f:
                        json.dump(debug_record, f)

            # Update the rolling history buffer + per-track bbox
            # snapshots. track_last_valid is updated every frame so
            # the dormant-track matcher has fresh data; bbox_history
            # is updated only at sampled buffer frames.
            for oid, mask in last_prop.object_masks.items():
                if oid not in tracks:
                    continue
                if mask is None or int(mask.sum()) == 0:
                    continue
                track_last_valid[oid] = (i, _bbox_from_mask(mask))
            _note_buffer(i, rgb, last_prop)

            # (4) periodic session reset. Keep the server's per-session
            # memory bounded: sam2_frame_count x n_tracks drives GPU
            # allocation, and SAM2 never evicts conditioning frames
            # once they're added via add_box. Close + reopen +
            # buffer-replay keeps the tracker going with a clean
            # memory bank while preserving ~100 frames of context.
            if reset_every > 0 and sam2_frame_count >= reset_every \
                    and tracks:
                last_prop = reset_session(rgb, i)

            if (i + 1) % 20 == 0 or i == len(frame_ids) - 1:
                dt = time.time() - t0
                print(f"[sam2-stream] [{i+1}/{len(frame_ids)}] fid={fid} "
                      f"tracks={len(tracks)} (+{added_this_frame} new)  "
                      f"sam2_count={sam2_frame_count}  "
                      f"resets={reset_count}  "
                      f"{dt / (i + 1):.2f}s/frame")

        # end-of-video: nothing to do; sessions close in __exit__

    # Post-process: compute per-track mean_score and rewrite JSONs with it.
    # (We could have written it in step 4 already, but collecting the full
    # scores_by_frame first means the final mean is stable.)
    _rewrite_with_mean_scores(out_dir, frame_ids, tracks)
    print(f"[sam2-stream] done. tracks={len(tracks)}  "
          f"wall time={time.time() - t0:.1f}s")


def _write_frame_output(out_dir: str, fid: int, video_idx: int,
                         tracks: Dict[int, TrackState],
                         prop: PropagatedFrame) -> None:
    """Write one frame's ``detection_{fid}_final.json``.

    The exported mask is the track's EMA-binarised mask
    (``TrackState.binary_mask(0.5)``), NOT SAM2's raw per-frame
    binary output. The raw prop determines WHICH tracks are written
    this frame (only tracks SAM2 is currently propagating), but the
    actual mask + bbox come from the smoothed float mask. An empty
    float-mask binary skips the track for this frame.
    """
    dets_out: List[Dict[str, Any]] = []
    for oid, mask in prop.object_masks.items():
        if oid not in tracks:
            continue
        tr = tracks[oid]
        bin_mask = tr.binary_mask(0.5)
        # Fall back to SAM2's raw mask if the EMA hasn't built up
        # yet (shouldn't happen since seed_one / main-loop update
        # populate float_mask, but keeps the writer robust).
        if bin_mask is None or not bool(bin_mask.any()):
            bin_mask = mask
        # Skip tracks that are effectively invisible this frame:
        # both the EMA mask and the raw SAM2 mask are empty. These
        # are kept alive in tracks[] so dormant matching can
        # re-associate them if OWL re-detects later, but they have
        # no mask to export.
        if bin_mask is None or not bool(bin_mask.any()):
            continue
        bbox = _bbox_from_mask(bin_mask)
        best_label, best_mean = tr.best_label_mean()
        # score at this frame: the OWL score seen on THIS frame under
        # any label (fall back to seed score if no observation landed).
        observed = tr.score_at_frame(video_idx)
        sc = observed if observed is not None else tr.seed_score
        # Per-label summary (all labels the track has accumulated so
        # far). Updated again at the end of the video by
        # ``_rewrite_with_mean_scores``.
        label_stats = {
            lbl: {
                "mean_score": float(
                    sum(sbf.values()) / len(sbf)) if sbf else float(tr.seed_score),
                "n_obs": int(len(sbf)),
            }
            for lbl, sbf in tr.label_scores.items()
        }
        dets_out.append({
            "object_id":  int(oid),
            "label":      best_label,
            "score":      float(sc),
            "mean_score": float(best_mean),
            "n_obs":      int(tr.total_observations()),
            "labels":     label_stats,
            "box":        list(map(int, bbox)),
            "mask":       _mask_to_png_b64(bin_mask),
        })
    out_path = os.path.join(out_dir, f"detection_{fid:06d}_final.json")
    with open(out_path, "w") as f:
        json.dump({"detections": dets_out}, f, indent=2)


def _rewrite_with_mean_scores(out_dir: str, frame_ids: List[int],
                                tracks: Dict[int, TrackState]) -> None:
    """Second pass: fill in the final ``label`` / ``mean_score`` /
    ``n_obs`` / ``labels`` per track now that we've walked every
    frame. The winning label can shift as later frames accumulate
    observations, so we rewrite it here too. Cheap JSON rewrite (no
    mask re-encoding -- read/write the detections verbatim)."""
    final: Dict[int, Dict[str, Any]] = {}
    for oid, tr in tracks.items():
        best_label, best_mean = tr.best_label_mean()
        final[oid] = {
            "label":      best_label,
            "mean_score": float(best_mean),
            "n_obs":      int(tr.total_observations()),
            "labels":     {
                lbl: {
                    "mean_score": float(
                        sum(sbf.values()) / len(sbf)) if sbf else float(tr.seed_score),
                    "n_obs": int(len(sbf)),
                }
                for lbl, sbf in tr.label_scores.items()
            },
        }
    for fid in frame_ids:
        p = os.path.join(out_dir, f"detection_{fid:06d}_final.json")
        if not os.path.exists(p):
            continue
        with open(p, "r") as f:
            data = json.load(f)
        changed = False
        for d in data.get("detections", []):
            oid = int(d.get("object_id", -1))
            if oid not in final:
                continue
            ref = final[oid]
            if d.get("label") != ref["label"]:
                d["label"] = ref["label"]
                changed = True
            if abs(d.get("mean_score", -1) - ref["mean_score"]) > 1e-6:
                d["mean_score"] = ref["mean_score"]
                changed = True
            if d.get("n_obs", -1) != ref["n_obs"]:
                d["n_obs"] = ref["n_obs"]
                changed = True
            if d.get("labels") != ref["labels"]:
                d["labels"] = ref["labels"]
                changed = True
        if changed:
            with open(p, "w") as f:
                json.dump(data, f, indent=2)


def track_dataset(dataset_root: str,
                  server_url: str = SAM2_SERVER_URL,
                  det_dir_name: str = "detection_boxes",
                  out_dir_name: str = "detection_h",
                  new_obj_iou: float = 0.3,
                  max_iters: int = 4,
                  min_score: float = 0.0) -> None:

    rgb_dir = os.path.join(dataset_root, "rgb")
    # Absolute det_dir_name / out_dir_name go there directly; relative
    # paths are joined with dataset_root (backwards-compatible).
    det_dir = (det_dir_name if os.path.isabs(det_dir_name)
               else os.path.join(dataset_root, det_dir_name))
    out_dir = (out_dir_name if os.path.isabs(out_dir_name)
               else os.path.join(dataset_root, out_dir_name))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[sam2-client] dataset={dataset_root}  server={server_url}")

    frame_ids, frames = _load_frames(rgb_dir)
    if not frames:
        raise FileNotFoundError(f"no RGB in {rgb_dir}")
    owl_dets = _load_owl_detections(det_dir, min_score=min_score)
    print(f"[sam2-client] frames={len(frames)}  "
          f"owl-dets-per-frame (mean)={np.mean([len(v) for v in owl_dets.values()] or [0]):.1f}")

    # Map dataset frame_id → video index (SAM2 uses contiguous 0..N-1).
    fid_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
    idx_to_fid = {i: fid for fid, i in fid_to_idx.items()}

    tracks: Dict[int, TrackState] = {}
    next_id = 0

    def _seed_prompt(sam2, video_idx: int, owl: OwlDet) -> int:
        nonlocal next_id
        obj_id = next_id; next_id += 1
        tracks[obj_id] = TrackState(
            label=owl.label, first_frame=video_idx, score=owl.score)
        tracks[obj_id].observe(owl.label, video_idx, owl.score)
        sam2.add_box(video_idx, obj_id, owl.box)
        return obj_id

    with SAM2Client(server_url=server_url) as sam2:
        sam2.start(frames)

        # No explicit "seed from frame 0" step. The reconciliation loop
        # below handles initialization itself: in iter 0 with no prompts
        # yet, ``prop_by_idx`` is empty, every OWL detection across the
        # whole video is "unmatched", and the cluster + add-prompt pass
        # picks one high-score representative per (label, IoU-cluster)
        # to seed SAM2. This covers the case where frame 0 is empty and
        # there's no special meaning to "first frame".
        prop_by_idx: List[PropagatedFrame] = []

        for it in range(max_iters):
            # Skip propagate when state is empty (no prompts yet) — SAM2
            # legitimately returns no masks, and the server's empty-state
            # path also returns []. Treat as the same "no propagated
            # masks anywhere" state for the reconciliation pass.
            if tracks:
                t0 = time.time()
                prop_by_idx = sam2.propagate()
                print(f"[sam2-client] iter={it}  propagate took "
                      f"{time.time()-t0:.1f}s  "
                      f"frames_with_masks={len(prop_by_idx)}")
            else:
                prop_by_idx = []
                print(f"[sam2-client] iter={it}  no prompts yet — skipping "
                      f"propagate; treating all OWL detections as unmatched")

            # Walk every frame (not just frames SAM2 returned) so the loop
            # works equally well with an empty propagated state.
            new_prompts: List[Tuple[int, OwlDet]] = []
            for i, fid in enumerate(frame_ids):
                prop = (prop_by_idx[i] if i < len(prop_by_idx)
                        else PropagatedFrame(object_masks={},
                                              object_bboxes={}))
                for owl in owl_dets.get(fid, []):
                    best_iou = 0.0
                    best_oid = None
                    for oid, mask in prop.object_masks.items():
                        if tracks[oid].label != owl.label:
                            continue
                        iou = _bbox_mask_iou(owl.box, mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_oid = oid
                    if best_iou < new_obj_iou:
                        new_prompts.append((i, owl))
                    elif best_oid is not None:
                        # Record observed score on the matched track.
                        tracks[best_oid].observe(owl.label, i, owl.score)

            if not new_prompts:
                print(f"[sam2-client] converged after {it+1} iteration(s)")
                break

            # Dedup: an object visible across consecutive frames shouldn't
            # produce N new prompts just because SAM2 hasn't yet propagated
            # the first one. Cluster unmatched boxes by (label, rolling
            # bbox-IoU) — one prompt per cluster, at its highest-score
            # frame. This collapses a smoothly-moving detection from a
            # dozen entries into one.
            new_prompts = _cluster_new_prompts(new_prompts,
                                               iou_thresh=0.3)

            # Cap how many new prompts we add per iteration so propagate
            # doesn't explode; prefer high-score ones first.
            new_prompts.sort(key=lambda p: -p[1].score)
            MAX_NEW_PER_ITER = 20
            added = 0
            for video_idx, owl in new_prompts[:MAX_NEW_PER_ITER]:
                _seed_prompt(sam2, video_idx, owl)
                added += 1
            print(f"[sam2-client] iter={it}  added {added} new tracks  "
                  f"(pending {len(new_prompts)-added})")

        # Final propagate for consistency after any last additions.
        # Only meaningful if we ever seeded anything.
        if tracks and (prop_by_idx is None or not prop_by_idx):
            prop_by_idx = sam2.propagate()

    # -----------------------------------------------------------------
    #  Write detection_h/*_final.json
    #
    #  Per-track `mean_score` is the average OWL confidence across every
    #  frame this track was re-detected on. Tracks that got seeded once
    #  and never re-observed keep their seed score; tracks that were
    #  frequently re-detected on high-confidence OWL boxes converge to a
    #  high mean. Downstream consumers can filter on this belief to drop
    #  hallucinations (e.g. viz panels keep mean_score > 0.1).
    # -----------------------------------------------------------------
    final_summary: Dict[int, Tuple[str, float, int, Dict[str, Dict[str, Any]]]] = {}
    for oid, tr in tracks.items():
        best_label, best_mean = tr.best_label_mean()
        labels_out = {
            lbl: {
                "mean_score": float(
                    sum(sbf.values()) / len(sbf)) if sbf else float(tr.seed_score),
                "n_obs": int(len(sbf)),
            }
            for lbl, sbf in tr.label_scores.items()
        }
        final_summary[oid] = (best_label, float(best_mean),
                              int(tr.total_observations()), labels_out)

    print(f"[sam2-client] writing {len(frame_ids)} JSONs → {out_dir}")
    for i, prop in enumerate(prop_by_idx):
        fid = idx_to_fid.get(i, i)
        dets_out = []
        for oid, mask in prop.object_masks.items():
            if oid not in tracks:
                continue
            tr = tracks[oid]
            bbox = prop.object_bboxes.get(oid) or _bbox_from_mask(mask)
            observed = tr.score_at_frame(i)
            sc = observed if observed is not None else tr.seed_score
            best_label, best_mean, n_obs, labels_out = final_summary[oid]
            dets_out.append({
                "object_id":  int(oid),
                "label":      best_label,
                "score":      float(sc),
                "mean_score": float(best_mean),
                "n_obs":      int(n_obs),
                "labels":     labels_out,
                "box":        list(map(int, bbox)),
                "mask":       _mask_to_png_b64(mask),
            })
        out_path = os.path.join(out_dir, f"detection_{fid:06d}_final.json")
        with open(out_path, "w") as f:
            json.dump({"detections": dets_out}, f, indent=2)

    # Fill in any frames SAM2 didn't produce results for (shouldn't happen
    # after a full propagate, but be defensive).
    written_idxs = {i for i in range(len(prop_by_idx))}
    for i, fid in enumerate(frame_ids):
        if i in written_idxs:
            continue
        out_path = os.path.join(out_dir, f"detection_{fid:06d}_final.json")
        with open(out_path, "w") as f:
            json.dump({"detections": []}, f, indent=2)

    print(f"[sam2-client] done.  final track count = {len(tracks)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="trajectory folder "
                                     "(e.g. datasets/apple_drop)")
    ap.add_argument("--server", default=SAM2_SERVER_URL,
                    help="SAM2 server URL "
                         "(override with $SAM2_SERVER_URL)")
    ap.add_argument("--det-dir", default="detection_boxes",
                    help="OWL JSONs subdir (default: detection_boxes)")
    ap.add_argument("--out-dir", default="detection_h",
                    help="output subdir (default: detection_h)")
    ap.add_argument("--mode", choices=("streaming", "batch"),
                    default="streaming",
                    help="streaming = one frame at a time against the "
                         "new /sam2_stream_* API (default); batch = the "
                         "legacy upload-all-frames /sam2_start_session "
                         "flow, kept for comparison.")
    ap.add_argument("--new-obj-iou", type=float, default=0.3,
                    help="batch mode only: IoU (bbox vs propagated "
                         "mask) below which an OWL detection spawns "
                         "a NEW track (default: 0.3). Streaming mode "
                         "uses Hungarian matching instead -- see "
                         "--hungarian-max-cost.")
    ap.add_argument("--max-iters", type=int, default=4,
                    help="batch mode only: max propagate+add-prompt "
                         "cycles (default: 4; streaming has no loop).")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop OWL dets below this score "
                         "before seeding (default: 0)")
    ap.add_argument("--hungarian-max-cost", type=float, default=0.7,
                    help="streaming only: Hungarian matching rejects "
                         "OWL<->track pairs whose cost "
                         "(1 - IoU) + label_penalty exceeds this "
                         "value. Soft equivalent of an IoU floor: "
                         "0.7 allows same-label matches down to IoU "
                         "0.3, and cross-label matches (with a 0.2 "
                         "label penalty) down to IoU 0.5. The same "
                         "threshold gates the single-direction "
                         "fallback that absorbs duplicate OWL dets "
                         "into an existing track (default: 0.7).")
    ap.add_argument("--hungarian-label-penalty", type=float,
                    default=0.2,
                    help="streaming only: additive cost when the OWL "
                         "detection's label is NOT yet in the "
                         "track's label_scores history. 0 disables "
                         "the label term entirely (pure IoU "
                         "matching); higher values make cross-label "
                         "matching harder (default: 0.2, i.e. the "
                         "mismatched label needs ~0.2 more IoU "
                         "headroom than a matching label to win).")
    ap.add_argument("--hungarian-score-weight", type=float,
                    default=0.15,
                    help="streaming only: weight of the confidence "
                         "penalty in the Hungarian cost. Each pair's "
                         "cost gets an extra term "
                         "weight * (1 - owl.score), so lower-"
                         "confidence OWL detections are matched to "
                         "existing tracks with lower priority. "
                         "0 disables the score term (default: 0.15; "
                         "smaller than the label penalty so it "
                         "affects priority ordering without vetoing "
                         "legitimate drifted matches).")
    ap.add_argument("--new-seed-min-score", type=float, default=0.25,
                    help="streaming only: hard minimum on OWL score "
                         "for seeding a NEW track. Dets below this "
                         "score may still match existing tracks "
                         "(via Hungarian or the fallback absorb "
                         "pass) but are dropped if unmatched -- "
                         "prevents low-confidence hallucinations "
                         "from creating fresh tracks (default: 0.25).")
    ap.add_argument("--self-match-max-cost", type=float, default=0.3,
                    help="streaming only: post-matching "
                         "track-to-track self-match threshold. "
                         "After the OWL->track matching, every pair "
                         "of active tracks is scored with the same "
                         "cost formula (mask-mask IoU + label "
                         "penalty, no score term). Hungarian picks "
                         "the min-cost pairing; pairs with cost "
                         "below this threshold are MERGED by "
                         "average-voting labels and masks. 0 "
                         "disables the step (default: 0.3 -- roughly "
                         "IoU >= 0.7 for same-label pairs; tighter "
                         "than the OWL->track threshold so only "
                         "clear duplicates merge).")
    ap.add_argument("--owl-mask-timeout", type=float, default=60.0,
                    help="streaming only: HTTP timeout (seconds) "
                         "for the per-frame /sam_mask_by_bbox call "
                         "used to produce an 'instant' mask per "
                         "OWL bbox, which is then matched "
                         "mask-vs-mask against each active track "
                         "(default: 60).")
    ap.add_argument("--no-owl-masks", dest="use_owl_masks",
                    action="store_false", default=True,
                    help="streaming only: skip the "
                         "/sam_mask_by_bbox call; the Hungarian "
                         "cost reverts to bbox-vs-mask IoU. Useful "
                         "for the minimal baseline (OWL + SAM + "
                         "IoU-class Hungarian only).")
    ap.add_argument("--no-fallback", dest="use_fallback",
                    action="store_false", default=True,
                    help="streaming only: skip the single-"
                         "direction greedy absorption of "
                         "Hungarian-rejected detections. Dets that "
                         "don't match via Hungarian go directly "
                         "to the seed path (gated by "
                         "--new-seed-min-score). Reproduces the "
                         "baseline OWL+SAM+Hungarian behaviour.")
    ap.add_argument("--reset-every", type=int, default=150,
                    help="streaming only: close+reopen the SAM2 "
                         "session every N sam2_frame_count steps and "
                         "re-seed live tracks from the bbox of their "
                         "current mask. Caps server GPU memory; at "
                         "0 disables. Reset cost ~1 close + 1 init + "
                         "n_tracks * (add_box + commit) roundtrips "
                         "(default: 150).")
    ap.add_argument("--reset-buffer-max", type=int, default=100,
                    help="streaming only: max number of recent "
                         "frames kept in the rolling buffer for "
                         "warm-restart during reset (default: 100).")
    ap.add_argument("--reset-buffer-interval", type=int, default=3,
                    help="streaming only: sample a frame into the "
                         "rolling buffer every N client frames "
                         "(default: 3; so 100 buffered frames cover "
                         "~300 client frames of history).")
    ap.add_argument("--debug-dir", default=None,
                    help="streaming only: if set, write one JSON per "
                         "propagate-branch frame with OWL dets, "
                         "prior-frame track state, cost matrix, "
                         "Hungarian/fallback decisions, new seeds, "
                         "and final tracks. Consumed by "
                         "tests/visualize_matching_debug.py.")
    ap.add_argument("--debug-fid-start", type=int, default=0,
                    help="streaming only: lower inclusive bound on "
                         "frame_id for debug dump (default: 0).")
    ap.add_argument("--debug-fid-end", type=int, default=10 ** 9,
                    help="streaming only: upper inclusive bound on "
                         "frame_id for debug dump (default: effectively "
                         "unlimited).")
    args = ap.parse_args()

    if args.mode == "streaming":
        track_dataset_streaming(
            dataset_root=args.dataset,
            server_url=args.server,
            det_dir_name=args.det_dir,
            out_dir_name=args.out_dir,
            min_score=args.min_score,
            hungarian_max_cost=args.hungarian_max_cost,
            hungarian_label_penalty=args.hungarian_label_penalty,
            hungarian_score_weight=args.hungarian_score_weight,
            new_seed_min_score=args.new_seed_min_score,
            self_match_max_cost=args.self_match_max_cost,
            use_owl_masks=args.use_owl_masks,
            use_fallback=args.use_fallback,
            owl_mask_timeout=args.owl_mask_timeout,
            reset_every=args.reset_every,
            reset_buffer_max=args.reset_buffer_max,
            reset_buffer_interval=args.reset_buffer_interval,
            debug_dir=args.debug_dir,
            debug_fid_start=args.debug_fid_start,
            debug_fid_end=args.debug_fid_end,
        )
    else:
        track_dataset(
            dataset_root=args.dataset,
            server_url=args.server,
            det_dir_name=args.det_dir,
            out_dir_name=args.out_dir,
            new_obj_iou=args.new_obj_iou,
            max_iters=args.max_iters,
            min_score=args.min_score,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

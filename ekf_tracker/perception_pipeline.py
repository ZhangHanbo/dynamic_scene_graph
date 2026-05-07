"""Live OWL + SAM2-streaming detection pipeline.

Used by ``EkfTracker.detect()`` to produce per-frame detections with
**stable cross-frame ``object_id``** values that the EKF consumes via
``sam2_tau``. Mirrors the algorithm of
``scripts/rosbag2dataset/sam2/sam2_client.py:track_dataset_streaming``
without the offline-only features (debug capture, dormant-track
window, periodic GPU-memory reset, rolling frame buffer).

Public surface:

    pipe = LiveDetectionPipeline(owl_url=..., sam2_url=..., cfg=...)
    pipe.start()
    for rgb in frames:
        dets = pipe.step(rgb, vocabulary)
    pipe.close()

Each detection dict matches the schema produced by
``scripts/visualize_ekf_tracking.py:_load_detection_json``:
    {id, object_id, label, score, mean_score, n_obs, labels, box, mask}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Reuse primitives from the existing offline pipeline.
from scripts.rosbag2dataset.sam2.sam2_client import (
    SAM2StreamClient,
    OwlDet,
    PropagatedFrame,
    TrackState,
    _bbox_from_mask,
    _bbox_mask_iou,
    _mask_mask_iou,
    _bbox_iou,
    _owl_bbox_masks,
    _self_match_and_merge_tracks,
)
from scripts.rosbag2dataset.owl.owl_client import call_owl
from scripts.rosbag2dataset.server_configs import (
    OWL_SERVER_URL,
    SAM2_SERVER_URL,
    OWL_BBOX_CONF,
    OWL_NMS_CROSS,
    OWL_NMS_CAT,
)


@dataclass
class LiveDetectionConfig:
    """Tuning knobs for the live OWL + SAM2 streaming pipeline.

    Defaults match ``track_dataset_streaming`` so live-vs-cached
    output is contract-equivalent on the same trajectory.
    """
    # Hungarian / fallback matching.
    hungarian_max_cost: float = 0.7
    hungarian_label_penalty: float = 0.2
    hungarian_score_weight: float = 0.15
    new_seed_min_score: float = 0.25
    self_match_max_cost: float = 0.3
    use_owl_masks: bool = True
    owl_mask_timeout: float = 60.0
    # OWL client knobs.
    owl_bbox_conf: float = OWL_BBOX_CONF
    owl_nms_cross: float = OWL_NMS_CROSS
    owl_nms_cat: float = OWL_NMS_CAT
    owl_min_score: float = 0.05
    owl_timeout: float = 60.0


@dataclass
class LiveDetectionPipeline:
    """Stateful per-trajectory pipeline.

    Owns one ``SAM2StreamClient`` session, the per-track ``TrackState``
    dict, and the frame counter. Pass an instance in/out of
    ``EkfTracker.detect`` as the ``history`` argument to thread the
    session across frames.
    """
    owl_url: str = OWL_SERVER_URL
    sam2_url: str = SAM2_SERVER_URL
    cfg: LiveDetectionConfig = field(default_factory=LiveDetectionConfig)

    # Mutable state populated by start() / step().
    sam2: Optional[SAM2StreamClient] = None
    tracks: Dict[int, TrackState] = field(default_factory=dict)
    frame_count: int = 0          # SAM2-side frame counter (mirrors server)
    client_idx: int = 0           # caller-side frame counter
    last_prop: Optional[PropagatedFrame] = None

    def start(self) -> "LiveDetectionPipeline":
        """Open the SAM2 streaming session. Hard-errors if the server
        is unreachable (no cached fallback).
        """
        if self.sam2 is None:
            self.sam2 = SAM2StreamClient(server_url=self.sam2_url)
            self.sam2.start()
        return self

    def close(self) -> None:
        """Close the SAM2 session. Safe to call multiple times."""
        if self.sam2 is not None:
            self.sam2.close()
            self.sam2 = None

    def __enter__(self) -> "LiveDetectionPipeline":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ─────────────────────────────────────────────────────────────────
    #  Core per-frame entry point
    # ─────────────────────────────────────────────────────────────────

    def step(self,
             rgb: np.ndarray,
             vocabulary: List[str]) -> List[Dict[str, Any]]:
        """Run one frame: SAM2 propagate → OWL → match/seed/merge → emit.

        Returns a list of detection dicts (schema below). Each dict's
        ``id`` is the SAM2 tracklet ID — stable across frames for
        the same physical object.
        """
        if self.sam2 is None:
            raise RuntimeError(
                "LiveDetectionPipeline.start() must be called before step()")

        i = self.client_idx

        # 1. Live OWL on this frame: list of OwlDet with pixel-frame boxes.
        owls = self._owl(rgb, vocabulary)

        # 2. SAM2 propagate.  Two cases:
        #    (a) no tracks yet AND no OWL dets → server short-circuits, no
        #        useful prop; emit nothing.
        #    (b) no tracks yet, OWL dets present → peek to register
        #        original_size, then seed each OWL det (one add_box +
        #        one frame each).
        #    (c) tracks already seeded → propagate, then OWL match.
        empty_prop = PropagatedFrame(object_masks={}, object_bboxes={})
        if not self.tracks and not owls:
            self.last_prop = empty_prop
            self.client_idx += 1
            return []

        if not self.tracks:
            # First frame with OWL dets: peek + seed each.
            self.sam2.frame(rgb)
            prop = empty_prop
            for owl in owls:
                prop = self._seed_one(owl, rgb, i)
        else:
            # 2c. Propagate → EMA-update each track's mask.
            prop = self.sam2.frame(rgb)
            self.frame_count += 1
            for oid, mask in prop.object_masks.items():
                if oid in self.tracks and mask is not None:
                    self.tracks[oid].update_mask(mask, alpha=0.5)

            # 3. Hungarian-match propagated tracks ↔ OWL detections.
            #    OWL "instant" masks (via stateless /sam_mask_by_bbox)
            #    enable mask-vs-mask cost; fall back to bbox-vs-mask.
            owl_masks = self._owl_instant_masks(rgb, owls)
            track_entries = self._track_entries_for_match(prop)
            matched_det_idx = self._hungarian_pass(
                owls, owl_masks, track_entries, i)

            # 4. Single-direction greedy fallback + new-seed for
            #    detections Hungarian didn't accept.
            for j, owl in enumerate(owls):
                if j in matched_det_idx:
                    continue
                best_c = self.cfg.hungarian_max_cost + 1.0
                best_oid: Optional[int] = None
                for entry in track_entries:
                    c = self._pair_cost(j, owl, entry, owl_masks)
                    if c < best_c:
                        best_c = c
                        best_oid = entry[0]
                if (best_oid is not None
                        and best_c <= self.cfg.hungarian_max_cost):
                    self.tracks[best_oid].observe(
                        owl.label, i, owl.score)
                    continue
                if owl.score < self.cfg.new_seed_min_score:
                    continue
                # New seed.
                before_oids = set(self.tracks.keys())
                prop = self._seed_one(owl, rgb, i)
                new_oids = sorted(set(self.tracks.keys()) - before_oids)
                # Allow later dets in this frame's loop to match the
                # freshly-seeded track.
                for new_oid in new_oids:
                    new_mask = prop.object_masks.get(new_oid)
                    if new_mask is not None and int(new_mask.sum()) > 0:
                        track_entries.append((new_oid, new_mask, None))

            # 5. Track-to-track self-merge.
            if (self.cfg.self_match_max_cost > 0.0
                    and len(self.tracks) >= 2):
                _self_match_and_merge_tracks(
                    self.tracks, prop,
                    max_cost=self.cfg.self_match_max_cost,
                    label_penalty=self.cfg.hungarian_label_penalty,
                )

        # 6. Emit detections (schema = step()'s expected input).
        out = self._emit_detections(prop)
        self.last_prop = prop
        self.client_idx += 1
        return out

    # ─────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────

    def _owl(self,
             rgb: np.ndarray,
             vocabulary: List[str]) -> List[OwlDet]:
        """Live OWL call, returns ``OwlDet`` with pixel-frame boxes."""
        names, bboxes_norm, scores = call_owl(
            rgb, list(vocabulary),
            bbox_conf_threshold=self.cfg.owl_bbox_conf,
            server_url=self.owl_url,
            with_nms=True,
            nms_threshold=self.cfg.owl_nms_cross,
            nms_cat_threshold=self.cfg.owl_nms_cat,
            timeout=self.cfg.owl_timeout,
        )
        if len(names) == 0:
            return []

        keep = scores > self.cfg.owl_min_score
        if not keep.all():
            idx = np.where(keep)[0]
            bboxes_norm = bboxes_norm[keep]
            scores = scores[keep]
            names = [names[int(k)] for k in idx]

        h, w = rgb.shape[:2]
        out: List[OwlDet] = []
        for name, bx, sc in zip(names, bboxes_norm, scores):
            x1 = int(round(bx[0] * w)); y1 = int(round(bx[1] * h))
            x2 = int(round(bx[2] * w)); y2 = int(round(bx[3] * h))
            box_pix = [max(0, x1), max(0, y1),
                       min(w - 1, x2), min(h - 1, y2)]
            out.append(OwlDet(
                frame_idx=self.client_idx,
                label=str(name),
                score=float(sc),
                box=box_pix,
            ))
        return out

    def _owl_instant_masks(self,
                           rgb: np.ndarray,
                           owls: List[OwlDet]
                           ) -> List[Optional[np.ndarray]]:
        if not owls or not self.cfg.use_owl_masks:
            return [None] * len(owls)
        try:
            return _owl_bbox_masks(
                rgb, [o.box for o in owls],
                server_url=self.sam2_url,
                timeout=self.cfg.owl_mask_timeout,
            )
        except Exception:
            return [None] * len(owls)

    def _track_entries_for_match(
            self, prop: PropagatedFrame
            ) -> List[Tuple[int, Optional[np.ndarray], Optional[List[int]]]]:
        """Active-track entries for the Hungarian cost matrix.

        Live-mode skips dormant-track matching (which the offline driver
        uses to bridge brief mask drop-outs across up to 30 frames).
        """
        entries: List[Tuple[int, Optional[np.ndarray], Optional[List[int]]]] = []
        for oid, mask in prop.object_masks.items():
            if oid not in self.tracks:
                continue
            if mask is None or int(mask.sum()) == 0:
                continue
            entries.append((oid, mask, None))
        return entries

    def _pair_cost(self,
                   owl_idx: int,
                   owl: OwlDet,
                   entry: Tuple[int, Optional[np.ndarray],
                                Optional[List[int]]],
                   owl_masks: List[Optional[np.ndarray]]) -> float:
        oid, mref, bref = entry
        det_mask = owl_masks[owl_idx] if owl_idx < len(owl_masks) else None
        if mref is not None:
            if (det_mask is not None
                    and det_mask.shape == mref.shape
                    and int(det_mask.sum()) > 0):
                iou = _mask_mask_iou(det_mask, mref)
            else:
                iou = _bbox_mask_iou(owl.box, mref)
        elif bref is not None:
            iou = _bbox_iou(owl.box, bref)
        else:
            return float("inf")
        if iou <= 0.0:
            return float("inf")
        label_pen = (0.0 if owl.label in self.tracks[oid].label_scores
                     else self.cfg.hungarian_label_penalty)
        score_pen = (self.cfg.hungarian_score_weight
                     * (1.0 - float(owl.score)))
        return (1.0 - iou) + label_pen + score_pen

    def _hungarian_pass(self,
                        owls: List[OwlDet],
                        owl_masks: List[Optional[np.ndarray]],
                        track_entries: List[Tuple[int, Optional[np.ndarray],
                                                   Optional[List[int]]]],
                        client_i: int) -> set:
        matched_det_idx: set = set()
        N_d = len(owls)
        M_t = len(track_entries)
        if N_d == 0 or M_t == 0:
            return matched_det_idx
        BIG = 1e6
        cost_mx = np.full((N_d, M_t), BIG, dtype=np.float64)
        for c_j, entry in enumerate(track_entries):
            for r_i, owl in enumerate(owls):
                c = self._pair_cost(r_i, owl, entry, owl_masks)
                if np.isfinite(c):
                    cost_mx[r_i, c_j] = c
        row_ind, col_ind = linear_sum_assignment(cost_mx)
        for r_i, c_j in zip(row_ind, col_ind):
            c_val = float(cost_mx[r_i, c_j])
            if c_val > self.cfg.hungarian_max_cost:
                continue
            oid = track_entries[c_j][0]
            self.tracks[oid].observe(
                owls[r_i].label, client_i, owls[r_i].score)
            matched_det_idx.add(int(r_i))
        return matched_det_idx

    def _seed_one(self,
                  owl: OwlDet,
                  rgb: np.ndarray,
                  client_idx: int) -> PropagatedFrame:
        """Add one OWL detection as a new track and commit it.

        Mirrors ``track_dataset_streaming.seed_one``: each prompt gets
        its own ``frame_idx = self.frame_count`` and is immediately
        committed by a ``frame()`` call so SAM2 advances + registers.
        """
        obj_id = self.sam2.add_box(frame_idx=self.frame_count, box=owl.box)
        self.tracks[obj_id] = TrackState(
            label=owl.label, first_frame=client_idx, score=owl.score)
        self.tracks[obj_id].observe(owl.label, client_idx, owl.score)
        committed = self.sam2.frame(rgb)
        self.frame_count += 1
        new_mask = committed.object_masks.get(obj_id)
        if new_mask is not None:
            self.tracks[obj_id].update_mask(new_mask, alpha=0.5)
        return committed

    def _emit_detections(self,
                         prop: PropagatedFrame) -> List[Dict[str, Any]]:
        """Build the final per-frame detection list in the EKF-consumed
        schema. Uses each track's EMA-smoothed mask thresholded at 0.5,
        per-track best-label, and rich label statistics."""
        out: List[Dict[str, Any]] = []
        for oid, _mask in prop.object_masks.items():
            if oid not in self.tracks:
                continue
            tr = self.tracks[oid]
            bin_mask = tr.binary_mask(0.5)
            if bin_mask is None:
                continue
            bin_mask_u8 = bin_mask.astype(np.uint8)
            if int(bin_mask_u8.sum()) == 0:
                continue
            best_label, best_mean = tr.best_label_mean()
            label_stats: Dict[str, Dict[str, Any]] = {}
            for lbl, sbf in tr.label_scores.items():
                if not sbf:
                    continue
                label_stats[lbl] = {
                    "n_obs": int(len(sbf)),
                    "mean_score": float(sum(sbf.values()) / max(1, len(sbf))),
                }
            last_score = tr.score_at_frame(self.client_idx)
            if last_score is None:
                last_score = best_mean
            out.append({
                "id": int(oid),                 # what step() consumes
                "object_id": int(oid),          # cached-file alias
                "label": best_label,
                "score": float(last_score),
                "mean_score": float(best_mean),
                "n_obs": int(tr.total_observations()),
                "labels": label_stats,
                "box": _bbox_from_mask(bin_mask_u8),
                "mask": bin_mask_u8,
            })
        return out

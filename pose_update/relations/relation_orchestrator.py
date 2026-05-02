"""Online relation-graph orchestrator.

Wraps a relation backend (LLM or REST), throttles re-detection via the
trigger gate in :mod:`pose_update.relations.relation_utils`, smooths per-call
edge scores via :class:`pose_update.orchestrator.RelationFilter`'s EMA,
and exposes the current filtered edges so callers can run
:func:`pose_update.relations.relation_utils.expand_held_with_relations`.

Conventions
-----------
- Edge convention follows ``expand_held_with_relations``:
  ``parent = supported``, ``child = supporter``. The LLM/REST backend
  reports ``i is parent of j`` to mean ``j rests on i`` (i is the
  supporter); we swap to match the expand convention.
- The backend is constructed lazily on first ``maybe_update`` call so
  runs that never enter a triggering frame don't pay the LLM-init
  cost.
- After a self-merge, callers MUST invoke
  :meth:`RelationOrchestrator.remap_after_merges` so EMA keys + emitted
  edges referring to dropped oids get rewritten to the keepers; without
  this, expansion silently fails on the next frame.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from pose_update.factor_graph import RelationEdge
from pose_update.orchestrator import RelationFilter
from pose_update.relations.relation_utils import (
    RelationTriggerConfig,
    RelationTriggerState,
    should_recompute_relations,
)


class RelationOrchestrator:
    """Triggered relation-graph builder with EMA smoothing.

    Public API:
        - ``maybe_update(frame, rgb, detections, det_to_oid,
          current_phase, current_oids)``: decide whether to fire the
          backend; if so, query it, update the EMA, and refresh
          ``self.edges``. Returns a summary dict for diagnostics.
        - ``edges``: the currently-emitted filtered edges
          (``List[RelationEdge]``).
        - ``remap_after_merges(merges)``: rewrite EMA keys + edges so
          dropped oids are replaced by their keep counterparts after
          a self-merge pass.

    Backends:
        - ``"llm"``: :class:`pose_update.relations.relation_client.LLMRelationClient`
          (default).
        - ``"rest"``: :class:`pose_update.relations.relation_client.RESTRelationClient`.
        - ``"none"``: disable; ``maybe_update`` becomes an EMA decay only.
    """

    def __init__(
        self,
        backend: str = "llm",
        llm_model: str = "gpt-5.1",
        ema_alpha: float = 0.3,
        ema_threshold: float = 0.5,
        score_threshold: float = 0.5,
        trigger_cfg: Optional[RelationTriggerConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        self.backend = backend
        self.llm_model = llm_model
        self.score_threshold = float(score_threshold)
        self._cache_dir = cache_dir
        self._client = None  # lazily initialised
        self._filter = RelationFilter(alpha=ema_alpha,
                                       threshold=ema_threshold)
        self._state = RelationTriggerState()
        self._cfg = trigger_cfg or RelationTriggerConfig()
        self._edges: List[RelationEdge] = []
        self._last_call_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Backend construction
    # ------------------------------------------------------------------

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if self.backend == "none":
            return None
        try:
            if self.backend == "llm":
                from pose_update.relations.relation_client import LLMRelationClient
                inner = LLMRelationClient(model_name=self.llm_model)
            elif self.backend == "rest":
                from pose_update.relations.relation_client import RESTRelationClient
                inner = RESTRelationClient()
            else:
                print(f"[relation] unknown backend {self.backend!r} — disabled")
                return None
            if self._cache_dir:
                from pose_update.relations.relation_client import CachedRelationClient
                os.makedirs(self._cache_dir, exist_ok=True)
                inner = CachedRelationClient(inner, cache_dir=self._cache_dir)
                print(f"[relation] cache dir: {self._cache_dir}")
            self._client = inner
        except Exception as e:
            print(f"[relation] backend init failed ({e}) — disabled")
            self.backend = "none"
            return None
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_update(
        self,
        frame: int,
        rgb: np.ndarray,
        detections: List[Dict[str, Any]],
        det_to_oid: Dict[int, int],
        current_phase: str,
        current_oids: Set[int],
        held_oid: Optional[int] = None,  # accepted for API compat; unused
        live_tracks: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Decide whether to re-call the backend; if so, do it; update
        state. Returns a summary dict (fired flag, edge counts, etc.).
        """
        del held_oid, live_tracks  # geometric fallback removed

        fired = should_recompute_relations(
            self._state, current_phase, current_oids, frame, self._cfg)
        if not fired:
            self._last_call_summary = {
                "fired": False, "frame": frame,
                "n_filtered_edges": len(self._edges),
            }
            return self._last_call_summary

        client = self._ensure_client()
        if client is None:
            self._edges = self._filter.update([])
            self._advance_state(frame, current_phase, current_oids)
            self._last_call_summary = {
                "fired": True, "backend": "none",
                "frame": frame,
                "n_filtered_edges": len(self._edges),
            }
            return self._last_call_summary

        # Build per-oid (bbox, mask) lists from current detections.
        H, W = rgb.shape[:2]
        oid_list: List[int] = []
        bboxes: List[np.ndarray] = []
        masks: List[Optional[np.ndarray]] = []
        seen: Set[int] = set()
        for det_idx, oid in det_to_oid.items():
            if oid in seen:
                continue
            if not (0 <= det_idx < len(detections)):
                continue
            det = detections[det_idx]
            box = det.get("box")
            if box is None:
                continue
            try:
                x0, y0, x1, y1 = (float(b) for b in box)
            except (TypeError, ValueError):
                continue
            bbox_n = np.array([x0 / W, y0 / H, x1 / W, y1 / H],
                               dtype=np.float32)
            mask = det.get("mask")
            if mask is not None:
                mask = np.asarray(mask) > 0
            oid_list.append(int(oid))
            bboxes.append(bbox_n)
            masks.append(mask)
            seen.add(int(oid))

        if len(oid_list) < 2:
            self._edges = self._filter.update([])
            self._advance_state(frame, current_phase, current_oids)
            self._last_call_summary = {
                "fired": True, "n_oids": len(oid_list),
                "n_filtered_edges": len(self._edges),
                "frame": frame, "skipped": "too_few_oids_for_llm",
            }
            return self._last_call_summary

        rgb_pil = (rgb if isinstance(rgb, Image.Image)
                   else Image.fromarray(np.asarray(rgb, dtype=np.uint8)))
        usable_masks = masks if all(m is not None for m in masks) else None
        from pose_update.relations.relation_client import set_relation_context
        try:
            set_relation_context(frame)
            p_parent = client.detect(rgb_pil, np.stack(bboxes, axis=0),
                                      masks=usable_masks)
        except Exception as e:
            print(f"[relation] detect() raised: {e}")
            p_parent = None
        finally:
            set_relation_context(None)

        if p_parent is None:
            self._edges = self._filter.update([])
            self._advance_state(frame, current_phase, current_oids)
            self._last_call_summary = {
                "fired": True, "frame": frame,
                "n_filtered_edges": len(self._edges),
                "skipped": "detect_returned_none",
            }
            return self._last_call_summary

        n = len(oid_list)
        raw: List[RelationEdge] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s = float(p_parent[i, j])
                if s < self.score_threshold:
                    continue
                # LLM convention: "i is parent of j" = "j rests on i" = i
                # supports j. expand_held_with_relations convention:
                # parent = supported, child = supporter. We swap so the
                # closure walks edges with `child=supporter` from the
                # held seed up to its riders.
                raw.append(RelationEdge(parent=oid_list[j],
                                         child=oid_list[i],
                                         relation_type="on", score=s))

        self._edges = self._filter.update(raw)
        self._advance_state(frame, current_phase, current_oids)
        self._last_call_summary = {
            "fired": True, "frame": frame,
            "backend": self.backend,
            "n_oids": n,
            "n_raw_edges": len(raw),
            "n_filtered_edges": len(self._edges),
            "edges": [(e.parent, e.child, e.relation_type, e.score)
                      for e in self._edges],
        }
        return self._last_call_summary

    @property
    def edges(self) -> List[RelationEdge]:
        return list(self._edges)

    def remap_after_merges(self,
                            merges: List[Dict[str, Any]]) -> None:
        """Rewrite EMA keys + filtered edges so dropped oids are
        replaced by their keep counterparts. Self-loops collapse and
        duplicate keys merge by max.
        """
        if not merges:
            return
        drop_to_keep: Dict[int, int] = {}
        for m in merges:
            keep = m.get("keep_oid")
            drop = m.get("drop_oid")
            if keep is None or drop is None:
                continue
            try:
                drop_to_keep[int(drop)] = int(keep)
            except (TypeError, ValueError):
                continue
        if not drop_to_keep:
            return

        def _resolve(o: int) -> int:
            seen: Set[int] = set()
            while o in drop_to_keep and o not in seen:
                seen.add(o)
                o = drop_to_keep[o]
            return o

        new_ema: Dict[Tuple, float] = {}
        for (p, c, t), val in self._filter._ema.items():
            try:
                p2, c2 = _resolve(int(p)), _resolve(int(c))
            except (TypeError, ValueError):
                continue
            if p2 == c2:
                continue
            key = (p2, c2, t)
            new_ema[key] = max(val, new_ema.get(key, 0.0))
        self._filter._ema = new_ema

        new_edges: List[RelationEdge] = []
        seen_keys: Set[Tuple] = set()
        for e in self._edges:
            try:
                p2, c2 = _resolve(int(e.parent)), _resolve(int(e.child))
            except (TypeError, ValueError):
                continue
            if p2 == c2:
                continue
            key = (p2, c2, e.relation_type)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            new_edges.append(RelationEdge(
                parent=p2, child=c2,
                relation_type=e.relation_type,
                score=e.score))
        self._edges = new_edges

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _advance_state(self, frame: int, current_phase: str,
                        current_oids: Set[int]) -> None:
        self._state.last_relation_frame = frame
        self._state.last_phase = current_phase
        self._state.known_oids_before_step = set(current_oids)

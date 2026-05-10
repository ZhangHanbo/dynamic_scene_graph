""":class:`RelationOrchestrator`: triggered relation builder with EMA smoothing; remaps EMA keys after self-merges."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from ekf_tracker.factor_graph import RelationEdge
from ekf_tracker.relations.relation_filter import RelationFilter
from ekf_tracker.relations.relation_utils import (
    RelationTriggerConfig,
    RelationTriggerState,
    should_recompute_relations,
)


class RelationOrchestrator:
    """Triggered relation builder with EMA smoothing; remaps EMA keys after self-merges."""

    def __init__(
        self,
        *,
        backend: str,
        llm_model: str,
        llm_temperature: float,
        ema_alpha: float,
        ema_threshold: float,
        ema_prune_threshold: float,
        score_threshold: float,
        rest_server_url: Optional[str],
        trigger_cfg: RelationTriggerConfig,
        cache_dir: Optional[str] = None,
    ):
        self.backend = backend
        self.llm_model = llm_model
        self.llm_temperature = float(llm_temperature)
        self.score_threshold = float(score_threshold)
        self.rest_server_url = rest_server_url
        self._cache_dir = cache_dir
        self._client = None  # lazily initialised
        self._filter = RelationFilter(
            alpha=ema_alpha,
            threshold=ema_threshold,
            prune_threshold=ema_prune_threshold,
        )
        self._state = RelationTriggerState()
        if trigger_cfg is None:
            raise TypeError(
                "RelationOrchestrator: `trigger_cfg` is required "
                "(no dataclass defaults; build via "
                "ekf_tracker.configs.build_relation_trigger_config).")
        self._cfg = trigger_cfg
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
                from ekf_tracker.relations.relation_client import LLMRelationClient
                inner = LLMRelationClient(model_name=self.llm_model)
            elif self.backend == "rest":
                from ekf_tracker.relations.relation_client import RESTRelationClient
                inner = (RESTRelationClient(server_url=self.rest_server_url)
                         if self.rest_server_url is not None
                         else RESTRelationClient())
            else:
                print(f"[relation] unknown backend {self.backend!r} — disabled")
                return None
            if self._cache_dir:
                from ekf_tracker.relations.relation_client import CachedRelationClient
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
        """Re-run the relation backend if the trigger fires, then EMA-smooth the resulting edges."""
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
        from ekf_tracker.relations.relation_client import set_relation_context
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
        """Re-key the EMA state after fast-tier track self-merges."""
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

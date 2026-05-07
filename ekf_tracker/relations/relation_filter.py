"""Exponential moving-average filter over scene-graph edges.

Smooths the binary present/absent signal of relation detections.
"""
from __future__ import annotations

from typing import Dict, List

from ekf_tracker.factor_graph import RelationEdge


class RelationFilter:
    """Exponential moving-average filter over scene-graph edges.

    Raw edge scores flicker frame-to-frame because the geometric
    relation test is noisy (random mock point clouds, bbox jitter).
    This filter smooths the binary present/absent signal into a
    stable 0-or-1 output.

    Per-edge EMA:
        ema(t) = α · raw(t) + (1 − α) · ema(t − 1)

    Output: emit the edge (score=1) when ema ≥ threshold, suppress
    otherwise. An edge not detected this frame gets raw=0.
    """

    def __init__(self, alpha: float = 0.3, threshold: float = 0.5):
        self.alpha = alpha
        self.threshold = threshold
        self._ema: Dict[tuple, float] = {}

    def update(self, raw_edges: List[RelationEdge]) -> List[RelationEdge]:
        """Accept raw edges from one frame; return the filtered set."""
        detected: Dict[tuple, float] = {}
        raw_meta: Dict[tuple, RelationEdge] = {}
        for edge in raw_edges:
            key = (edge.parent, edge.child, edge.relation_type)
            detected[key] = edge.score
            raw_meta[key] = edge

        all_keys = set(self._ema.keys()) | set(detected.keys())
        filtered: List[RelationEdge] = []
        for key in all_keys:
            raw = detected.get(key, 0.0)
            prev = self._ema.get(key, raw)
            ema = self.alpha * raw + (1.0 - self.alpha) * prev
            self._ema[key] = ema
            if ema >= self.threshold:
                parent, child, rel_type = key
                ref = raw_meta.get(key)
                filtered.append(RelationEdge(
                    parent=parent, child=child,
                    relation_type=rel_type,
                    score=1.0,
                    parent_size=ref.parent_size if ref else None,
                    child_size=ref.child_size if ref else None,
                ))
        # Prune dead edges (EMA decayed to near zero).
        self._ema = {k: v for k, v in self._ema.items() if v > 0.01}
        return filtered

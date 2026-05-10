""":class:`RelationFilter` — exponential moving average over edge scores, threshold-based emit, per-edge prune on staleness."""
from __future__ import annotations

from typing import Dict, List

from ekf_tracker.factor_graph import RelationEdge


class RelationFilter:
    r"""EMA over per-edge scores: :math:`\bar s \leftarrow \alpha s + (1-\alpha)\bar s`, emit when :math:`\bar s > \tau`, prune on staleness."""

    def __init__(self, *, alpha: float, threshold: float,
                 prune_threshold: float):
        self.alpha = alpha
        self.threshold = threshold
        self.prune_threshold = prune_threshold
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
        self._ema = {k: v for k, v in self._ema.items()
                     if v > self.prune_threshold}
        return filtered

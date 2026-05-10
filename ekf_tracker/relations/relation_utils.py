""":func:`expand_held_with_relations` (transitive closure under ``in``/``on``) and :func:`should_recompute_relations` (pure trigger gate)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Set


# ─────────────────────────────────────────────────────────────────────
# Manipulation-set expansion
# ─────────────────────────────────────────────────────────────────────

def expand_held_with_relations(
        held_id: Optional[int],
        edges: Iterable,
        *,
        max_iters: int,
        ) -> Set[int]:
    """Transitive closure of the held set under ``in`` and ``on`` edges (capped at ``max_depth``)."""
    if held_id is None:
        return set()
    manipulated: Set[int] = {int(held_id)}
    edges_list = list(edges)
    for _ in range(max_iters):
        changed = False
        for edge in edges_list:
            rel_type = getattr(edge, "relation_type", None)
            if rel_type not in ("in", "on"):
                continue
            try:
                p = int(edge.parent)
                c = int(edge.child)
            except (TypeError, ValueError, AttributeError):
                continue
            if c in manipulated and p not in manipulated:
                manipulated.add(p)
                changed = True
        if not changed:
            break
    return manipulated


# ─────────────────────────────────────────────────────────────────────
# Trigger gate (when to call the relation backend)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RelationTriggerState:
    """Mutable state for :func:`should_recompute_relations` (last fire frame, last phase, last oid set)."""
    last_relation_frame: int = -10**9   # sentinel forces first-call fire
    last_phase: str = "idle"
    known_oids_before_step: Set[int] = field(default_factory=set)


@dataclass
class RelationTriggerConfig:
    """Config for :func:`should_recompute_relations`: periodic frames, on-grasp, on-release, on-new-object."""
    relation_every_n_frames: int
    relation_on_grasp: bool
    relation_on_release: bool
    relation_on_new_object: bool


def should_recompute_relations(state: RelationTriggerState,
                                 current_phase: str,
                                 current_oids: Set[int],
                                 current_frame: int,
                                 cfg: RelationTriggerConfig,
                                 ) -> bool:
    """Pure trigger gate: True if any of the configured events apply this frame."""
    if state.last_relation_frame < 0:
        return True
    if (cfg.relation_on_release
            and state.last_phase == "releasing"
            and current_phase != "releasing"):
        return True
    if cfg.relation_on_new_object:
        if current_oids - state.known_oids_before_step:
            return True
    if cfg.relation_every_n_frames > 0:
        if (current_frame - state.last_relation_frame
                >= cfg.relation_every_n_frames):
            return True
    return False

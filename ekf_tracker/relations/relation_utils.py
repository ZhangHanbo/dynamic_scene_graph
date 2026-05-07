"""Shared relation-graph helpers.

Two utilities reused by the production orchestrator and the test
visualization driver:

* ``expand_held_with_relations(held_id, edges)`` — transitive closure
  of the held set under "in"/"on" relations. If the gripper grasps a
  bowl and the scene graph says "apple in bowl", the apple rides with
  the bowl under the rigid-attachment predict.

* ``should_recompute_relations(state, current_phase, current_oids,
  current_frame, cfg)`` — pure function form of the orchestrator's
  ``_should_recompute_relations`` trigger gate. State is held in a
  small mutable struct so callers can drive it from anywhere.

Standalone helpers used by the EKF orchestrators and the visualization
driver — kept independent so callers can drive them without
instantiating a full tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Set


# ─────────────────────────────────────────────────────────────────────
# Manipulation-set expansion
# ─────────────────────────────────────────────────────────────────────

def expand_held_with_relations(
        held_id: Optional[int],
        edges: Iterable,
        max_iters: int = 8,
        ) -> Set[int]:
    """Return the transitive closure of {held_id} under "in"/"on" edges.

    Each edge has fields ``parent`` (the supported / contained object),
    ``child`` (the supporting / containing object), and
    ``relation_type`` (one of "in", "on", "under", "contain"). Only
    "in" and "on" propagate the manipulation: if the child rides with
    the gripper, so does the parent.

    Robust to edge objects from any source as long as they expose the
    three attributes above.
    """
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
    """Mutable state the trigger gate consults + updates.

    Kept standalone so callers don't need to instantiate a full tracker.
    """
    last_relation_frame: int = -10**9   # sentinel forces first-call fire
    last_phase: str = "idle"
    known_oids_before_step: Set[int] = field(default_factory=set)


@dataclass
class RelationTriggerConfig:
    """Same knobs as `BernoulliConfig.relation_*` but standalone."""
    relation_every_n_frames: int = 90
    relation_on_grasp: bool = True
    relation_on_release: bool = True
    relation_on_new_object: bool = True


def should_recompute_relations(state: RelationTriggerState,
                                 current_phase: str,
                                 current_oids: Set[int],
                                 current_frame: int,
                                 cfg: RelationTriggerConfig,
                                 ) -> bool:
    """Pure trigger gate for the relation backend.

    Fires on:
      * first call (sentinel `last_relation_frame < 0`),
      * grasp transition (last_phase != "grasping" and current == "grasping"),
      * release transition (last_phase == "releasing" and current != "releasing"),
      * a new confirmed oid since last step,
      * periodic tick every `relation_every_n_frames` frames.
    """
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

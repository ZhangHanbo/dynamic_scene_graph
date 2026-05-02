"""Unit tests for ``pose_update.relations.relation_orchestrator.RelationOrchestrator``."""
from __future__ import annotations

import numpy as np
import pytest

from pose_update.factor_graph import RelationEdge
from pose_update.relations.relation_orchestrator import RelationOrchestrator
from pose_update.relations.relation_utils import RelationTriggerConfig


@pytest.fixture
def orch():
    """Default orchestrator with backend disabled (we exercise EMA + remap directly)."""
    return RelationOrchestrator(backend="none",
                                 ema_alpha=0.3, ema_threshold=0.5,
                                 trigger_cfg=RelationTriggerConfig(
                                     relation_every_n_frames=10,
                                     relation_on_grasp=False,
                                     relation_on_release=False,
                                     relation_on_new_object=False))


def test_first_call_fires(orch):
    """On the very first call, the trigger gate fires regardless of phase/frame."""
    summary = orch.maybe_update(
        frame=0, rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        detections=[], det_to_oid={}, current_phase="idle",
        current_oids=set())
    assert summary["fired"] is True


def test_trigger_skips_between_periodic_ticks(orch):
    """After firing, the gate doesn't fire again until period elapses."""
    orch.maybe_update(
        frame=0, rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        detections=[], det_to_oid={}, current_phase="idle",
        current_oids=set())
    s1 = orch.maybe_update(
        frame=5, rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        detections=[], det_to_oid={}, current_phase="idle",
        current_oids=set())
    assert s1["fired"] is False  # period=10, we're at 5
    s2 = orch.maybe_update(
        frame=10, rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        detections=[], det_to_oid={}, current_phase="idle",
        current_oids=set())
    assert s2["fired"] is True


def test_remap_after_merges_redirects_keys(orch):
    """remap_after_merges substitutes drop_oid → keep_oid in EMA + edges."""
    # Seed an edge so the EMA has a key (parent=1, child=4, on).
    orch._filter._ema[(1, 4, "on")] = 0.9
    orch._edges = [RelationEdge(parent=1, child=4, relation_type="on", score=1.0)]

    orch.remap_after_merges([{"keep_oid": 6, "drop_oid": 4}])

    # The (1, 4) edge becomes (1, 6) after remap.
    assert (1, 6, "on") in orch._filter._ema
    assert (1, 4, "on") not in orch._filter._ema
    assert any(e.parent == 1 and e.child == 6 and e.relation_type == "on"
               for e in orch._edges)


def test_remap_drops_self_loops(orch):
    """A merge that turns an edge into a self-loop drops it."""
    orch._filter._ema[(1, 4, "on")] = 1.0
    orch._edges = [RelationEdge(parent=1, child=4, relation_type="on", score=1.0)]

    # Merge oid 4 into oid 1 → edge becomes (1, 1) which is a self-loop.
    orch.remap_after_merges([{"keep_oid": 1, "drop_oid": 4}])

    assert orch._filter._ema == {}
    assert orch._edges == []


def test_remap_merges_duplicate_keys_with_max(orch):
    """When two pre-merge keys collapse to one post-merge key, take the max EMA."""
    orch._filter._ema[(1, 4, "on")] = 0.7
    orch._filter._ema[(1, 5, "on")] = 0.9

    # Both 4 and 5 merge into 6 → both edges become (1, 6, on).
    orch.remap_after_merges([
        {"keep_oid": 6, "drop_oid": 4},
        {"keep_oid": 6, "drop_oid": 5},
    ])

    assert orch._filter._ema == {(1, 6, "on"): 0.9}


def test_parent_child_convention_in_p_parent_swap():
    """The LLM 'i is parent of j' (= j rests on i) must be emitted as
    RelationEdge(parent=j, child=i): the held-set expansion convention.

    Regression check: a fake p_parent[0,1]=1.0 should produce an edge
    (parent=oid_list[1], child=oid_list[0]).
    """
    # Mock client returns p_parent with one strong edge.
    class _FakeClient:
        available = True
        def detect(self, rgb, bboxes, masks=None):
            return np.array([[0.0, 0.95], [0.0, 0.0]])

    o = RelationOrchestrator(backend="none",
                              trigger_cfg=RelationTriggerConfig(
                                  relation_every_n_frames=1,
                                  relation_on_grasp=False,
                                  relation_on_release=False,
                                  relation_on_new_object=False))
    o._client = _FakeClient()
    o.backend = "fake"

    detections = [
        {"id": 100, "box": [0, 0, 100, 100], "mask": np.ones((50, 50), dtype=bool)},
        {"id": 200, "box": [50, 50, 150, 150], "mask": np.ones((50, 50), dtype=bool)},
    ]
    summary = o.maybe_update(
        frame=0,
        rgb=np.zeros((50, 50, 3), dtype=np.uint8),
        detections=detections,
        det_to_oid={0: 11, 1: 22},   # det 0 → oid 11, det 1 → oid 22
        current_phase="idle",
        current_oids={11, 22},
    )
    assert summary["fired"] is True
    # Convention: p_parent[0,1]=0.95 means "oid 22 (oid_list[1]) rests on oid 11 (oid_list[0])"
    # → RelationEdge(parent=22, child=11). Bump EMA above threshold takes a few frames;
    # we check raw_edges via the filter's internal state.
    assert (22, 11, "on") in o._filter._ema

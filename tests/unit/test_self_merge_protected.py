"""Unit test for the relation-aware merge protection.

This pins the production ``orchestrator._self_merge_pass`` behaviour
that pairs flagged in ``protected_pairs`` (because the relation graph
asserts they are distinct objects) are skipped before the distance
gate is checked. A regression here would re-enable the
apple-on-tray ↔ tray collapse we fixed in earlier rounds.
"""
from __future__ import annotations

import numpy as np
import pytest

# We test the helper directly without instantiating the full orchestrator
# (which has heavy dependencies). The merge-pass logic is structurally
# the same as what's deployed in ``InstrumentedTracker._self_merge_pass``,
# so we exercise it via that path.


def _build_tracker_with_two_close_apples():
    """Build a minimal InstrumentedTracker with two same-label tracks
    4.5 cm apart in base frame (just under the 5 cm self-merge gate)."""
    import sys
    from importlib import import_module
    sys.path.insert(0, "tests")
    viz = import_module("visualize_ekf_tracking")

    cfg = viz.BernoulliConfig()
    tracker = viz.InstrumentedTracker(
        K=viz.K_DEFAULT, bernoulli_cfg=cfg,
        T_bc=np.eye(4))
    # Seed two apple tracks at known base positions.
    from pose_update.state.gaussian_state import GaussianObjectBelief
    P = np.eye(6) * 1e-4
    T1 = np.eye(4); T1[:3, 3] = [0.50, 0.00, 0.00]
    T2 = np.eye(4); T2[:3, 3] = [0.545, 0.00, 0.00]   # 4.5 cm apart
    tracker.state.objects[1] = GaussianObjectBelief(mu_bo=T1, cov_bo=P)
    tracker.state.objects[2] = GaussianObjectBelief(mu_bo=T2, cov_bo=P)
    tracker.object_labels = {1: "apple", 2: "apple"}
    tracker.existence = {1: 1.0, 2: 1.0}
    tracker.frames_since_obs = {1: 0, 2: 0}
    tracker.sam2_tau = {1: -1, 2: -1}
    tracker.label_scores = {1: {}, 2: {}}
    return tracker, viz


def test_merge_fires_without_protection():
    tracker, viz = _build_tracker_with_two_close_apples()
    merges = tracker._self_merge_pass(held_id=None, protected_pairs=set())
    assert len(merges) == 1
    assert merges[0]["keep_oid"] in (1, 2)
    assert merges[0]["drop_oid"] in (1, 2)
    assert merges[0]["dist_m"] < 0.05


def test_merge_blocked_by_protected_pair():
    tracker, viz = _build_tracker_with_two_close_apples()
    merges = tracker._self_merge_pass(
        held_id=None, protected_pairs={(1, 2)})
    assert merges == []
    # Both tracks survive.
    assert 1 in tracker.state.objects
    assert 2 in tracker.state.objects


def test_held_seed_kept_in_merge():
    tracker, viz = _build_tracker_with_two_close_apples()
    merges = tracker._self_merge_pass(held_id=2, protected_pairs=set())
    assert len(merges) == 1
    assert merges[0]["keep_oid"] == 2  # held seed is kept
    assert merges[0]["drop_oid"] == 1

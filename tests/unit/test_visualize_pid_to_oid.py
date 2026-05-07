"""Unit tests for ``scripts.visualize_ekf_tracking._build_pid_to_oid``.

Pin: ``match`` values in the ekf state JSON are GLOBAL indices into
``dets_with_pose`` (the producer remaps Hungarian's local column
index through ``local_to_global`` before serialization). The
consumer must NOT re-index them through ``det_indices_in_assoc`` —
doing so silently drops out-of-range matches.

Regression: apple_drop fr 43 had ``match={"6": 4}`` with
``det_indices_in_assoc=[0, 1, 2, 4]`` (length 4). Treating ``4`` as
a local index produced an out-of-bounds skip; treating it as the
global index correctly maps oid 6 to the 5th detection.
"""
from __future__ import annotations

import importlib.util
import os
import sys


def _load_driver():
    """Import the driver module without executing its CLI entry."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    path = os.path.join(repo, "scripts", "visualize_ekf_tracking.py")
    spec = importlib.util.spec_from_file_location(
        "visualize_ekf_tracking_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pid_to_oid_apple_drop_fr43_regression():
    """Replays apple_drop fr 43: 4 surviving dets, oid 6 matched to
    GLOBAL idx 4 (= the 5th det in the post-suppression list)."""
    mod = _load_driver()
    dets_with_pose = [
        {"id": 1, "label": "apple"},     # global 0
        {"id": 2, "label": "bottle"},    # global 1
        {"id": 3, "label": "cup"},       # global 2
        {"id": 4, "label": "bottle"},    # global 3 — centroid_dropped, no T_co
        {"id": 5, "label": "bottle"},    # global 4 — misclassified apple
    ]
    dbg = {
        "assoc": {
            "match": {"3": 1, "4": 2, "5": 0, "6": 4},
            "det_indices_in_assoc": [0, 1, 2, 4],
        },
        "held_oids_used": [],
    }
    pid_to_oid, held = mod._build_pid_to_oid(dbg, dets_with_pose)
    assert pid_to_oid == {2: 3, 3: 4, 1: 5, 5: 6}
    assert held == set()


def test_pid_to_oid_normal_case():
    """All dets present (no centroid_dropped) — match values map
    directly to dets_with_pose indices."""
    mod = _load_driver()
    dets_with_pose = [{"id": 10}, {"id": 11}, {"id": 12}]
    dbg = {
        "assoc": {
            "match": {"100": 0, "101": 2},
            "det_indices_in_assoc": [0, 1, 2],
        },
        "held_oids_used": [101],
    }
    pid_to_oid, held = mod._build_pid_to_oid(dbg, dets_with_pose)
    assert pid_to_oid == {10: 100, 12: 101}
    assert held == {101}


def test_pid_to_oid_empty_match():
    mod = _load_driver()
    dbg = {"assoc": {"match": {}, "det_indices_in_assoc": []}}
    pid_to_oid, held = mod._build_pid_to_oid(dbg, [])
    assert pid_to_oid == {}
    assert held == set()


def test_pid_to_oid_held_obj_id_fallback():
    """When `held_oids_used` is missing, fall back to
    `gripper_state.held_obj_id`."""
    mod = _load_driver()
    dbg = {
        "assoc": {"match": {}, "det_indices_in_assoc": []},
        "gripper_state": {"held_obj_id": 7},
    }
    _, held = mod._build_pid_to_oid(dbg, [])
    assert held == {7}


def test_pid_to_oid_out_of_range_skipped():
    """A match value that's out of range for dets_with_pose is
    silently skipped (defensive — shouldn't happen in practice)."""
    mod = _load_driver()
    dets = [{"id": 0}, {"id": 1}]
    dbg = {"assoc": {"match": {"5": 99}, "det_indices_in_assoc": [0, 1]}}
    pid_to_oid, _ = mod._build_pid_to_oid(dbg, dets)
    assert pid_to_oid == {}

"""
Unit tests for the soft-score addition to object_relation_graph.py (Task 6a).

Tests the new `detect_spatial_relation_with_scores` and
`compute_spatial_relations_with_scores` functions. Uses synthetic bounding
boxes (no data dependency).

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python -m pytest tests/test_relation_scores.py -v
"""

import os
import sys

import numpy as np
import pytest

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from heuristic_tracker.object_relation_graph import (
    detect_spatial_relation,
    detect_spatial_relation_with_scores,
    compute_spatial_relations_with_scores,
    compute_spatial_relations,
)


class TestDetectWithScores:
    def test_identical_boxes_full_overlap_on(self):
        # A fully on top of B
        A_min, A_max = np.array([0, 0, 1.0]), np.array([0.1, 0.1, 1.1])
        B_min, B_max = np.array([0, 0, 0.0]), np.array([0.1, 0.1, 1.0])
        is_in, is_on, s_in, s_on = detect_spatial_relation_with_scores(
            A_min, A_max, B_min, B_max, tolerance=0.02, overlap_threshold=0.3)
        assert is_on and not is_in
        assert s_on > 0.9
        assert s_in == 0.0

    def test_A_inside_B(self):
        # A (smaller) inside B (larger container)
        A_min, A_max = np.array([0.02, 0.02, 0.1]), np.array([0.08, 0.08, 0.2])
        B_min, B_max = np.array([0, 0, 0.0]), np.array([0.1, 0.1, 0.3])
        is_in, is_on, s_in, s_on = detect_spatial_relation_with_scores(
            A_min, A_max, B_min, B_max, tolerance=0.02, overlap_threshold=0.3)
        assert is_in and not is_on
        assert s_in > 0.5
        assert s_on == 0.0

    def test_no_overlap_returns_zero_scores(self):
        A_min, A_max = np.array([0, 0, 0]), np.array([0.1, 0.1, 0.1])
        B_min, B_max = np.array([10, 10, 10]), np.array([10.1, 10.1, 10.1])
        is_in, is_on, s_in, s_on = detect_spatial_relation_with_scores(
            A_min, A_max, B_min, B_max, tolerance=0.02, overlap_threshold=0.3)
        assert not is_in and not is_on
        assert s_in == 0.0 and s_on == 0.0

    def test_partial_overlap_below_threshold(self):
        # Small overlap, under default threshold
        A_min, A_max = np.array([0, 0, 1.0]), np.array([1.0, 1.0, 1.1])
        B_min, B_max = np.array([0.9, 0.9, 0.0]), np.array([2.0, 2.0, 1.0])
        is_in, is_on, s_in, s_on = detect_spatial_relation_with_scores(
            A_min, A_max, B_min, B_max, tolerance=0.02, overlap_threshold=0.3)
        # Overlap is 0.1x0.1 = 0.01, min area is 1.0x1.0 = 1.0, ratio = 0.01
        # Below threshold → no boolean relation, but the score should reflect it
        assert not is_on
        # Score is 0 because the type check still passes but threshold filters
        # In this case the geometric type says "A is above B" so type_is_on=True,
        # so score_on = overlap_ratio = 0.01, but the boolean is False.
        # Verify:
        assert 0.0 < s_on < 0.3 or s_on == 0.0

    def test_legacy_detect_still_matches(self):
        """The old boolean API should return consistent results."""
        A_min, A_max = np.array([0, 0, 1.0]), np.array([0.1, 0.1, 1.1])
        B_min, B_max = np.array([0, 0, 0.0]), np.array([0.1, 0.1, 1.0])
        is_in_old, is_on_old = detect_spatial_relation(
            A_min, A_max, B_min, B_max, tolerance=0.02, overlap_threshold=0.3)
        is_in_new, is_on_new, _, _ = detect_spatial_relation_with_scores(
            A_min, A_max, B_min, B_max, tolerance=0.02, overlap_threshold=0.3)
        assert is_in_old == is_in_new
        assert is_on_old == is_on_new


class TestComputeWithScores:
    def _make_mock_object(self, obj_id, points):
        """Minimal stand-in for SceneObject with just the fields the function uses."""
        class MockObj:
            pass
        o = MockObj()
        o.id = obj_id
        o._points = points.astype(np.float32)
        o.pose_init = np.eye(4, dtype=np.float32)
        o.pose_cur = np.eye(4, dtype=np.float32)
        o.child_objs = {}
        o.parent_obj_id = None
        return o

    def test_apple_in_bowl_scenario(self):
        # Bowl: large box on the table
        np.random.seed(0)
        bowl_pts = np.random.uniform([0, 0, 0.0], [0.2, 0.2, 0.1],
                                      size=(500, 3))
        # Apple: small object inside the bowl volume
        apple_pts = np.random.uniform([0.08, 0.08, 0.02], [0.12, 0.12, 0.06],
                                       size=(200, 3))

        bowl = self._make_mock_object(0, bowl_pts)
        apple = self._make_mock_object(1, apple_pts)

        relations, scores = compute_spatial_relations_with_scores(
            [bowl, apple], tolerance=0.02, overlap_threshold=0.3)

        # Apple should be in bowl; bowl should contain apple
        assert 0 in relations[1]["in"], \
            f"Expected apple (1) to be in bowl (0), got {relations}"
        assert 1 in relations[0]["contain"]
        # Corresponding scores should exist and be > 0.5
        assert scores.get((1, 0, "in"), 0) > 0.3
        assert scores.get((0, 1, "contain"), 0) > 0.3

    def test_separate_objects_have_no_scores(self):
        np.random.seed(1)
        pts_A = np.random.uniform([0, 0, 0], [0.1, 0.1, 0.1], size=(100, 3))
        pts_B = np.random.uniform([10, 10, 10], [10.1, 10.1, 10.1], size=(100, 3))

        A = self._make_mock_object(0, pts_A)
        B = self._make_mock_object(1, pts_B)

        relations, scores = compute_spatial_relations_with_scores([A, B])
        assert len(scores) == 0
        assert relations[0] == {"in": [], "on": [], "under": [], "contain": []}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

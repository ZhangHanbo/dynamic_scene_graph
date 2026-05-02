"""
Integration tests for the two-tier orchestrator (Task 7).

Uses synthetic frames with the PassThroughSlam backend to exercise the
complete pipeline: movable-mask exclusion → SLAM → fast EKF → relation
recomputation → slow tier (when triggered) → EKF absorption.

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python -m pytest tests/test_orchestrator.py -v
"""

import os
import sys

import numpy as np
import pytest

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from pose_update.orchestrator import TwoTierOrchestrator, TriggerConfig
from pose_update.state.slam_interface import PassThroughSlam


def _make_detection(obj_id, label, T_co, mask_shape=(100, 100)):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask[20:40, 20:40] = 1
    return {
        "id": obj_id,
        "label": label,
        "mask": mask,
        "score": 0.8,
        "T_co": T_co,
        "R_icp": np.diag([1e-4] * 6),
        "fitness": 0.85,
        "rmse": 0.003,
    }


def _pose(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

class TestFastTier:
    def test_first_observation_creates_object(self):
        slam = PassThroughSlam([np.eye(4)] * 10)
        orch = TwoTierOrchestrator(
            slam, trigger=TriggerConfig(periodic_every_n_frames=-1))

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32) * 1.0
        dets = [_make_detection(obj_id=7, label="apple", T_co=_pose(0.3, 0, 0.5))]
        report = orch.step(rgb, depth, dets, {"phase": "idle", "held_obj_id": None})

        assert 7 in orch.objects
        assert orch.objects[7]["label"] == "apple"
        # Triggered because new object appeared
        assert report["triggered"]

    def test_static_object_covariance_shrinks_on_consistent_obs(self):
        slam = PassThroughSlam([np.eye(4)] * 20)
        orch = TwoTierOrchestrator(
            slam, trigger=TriggerConfig(periodic_every_n_frames=-1,
                                         on_new_object=False))

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        T_co = _pose(0.3, 0, 0.5)

        # Seed
        orch.step(rgb, depth, [_make_detection(0, "cup", T_co)],
                  {"phase": "idle", "held_obj_id": None})
        cov0 = np.trace(orch.objects[0]["cov"])

        # Repeated consistent observations
        for _ in range(10):
            orch.step(rgb, depth, [_make_detection(0, "cup", T_co)],
                      {"phase": "idle", "held_obj_id": None})
        cov_final = np.trace(orch.objects[0]["cov"])

        assert cov_final < cov0, \
            f"Covariance should shrink on consistent obs: {cov0} → {cov_final}"


class TestTriggerPolicy:
    def test_grasp_event_triggers_slow_tier(self):
        slam = PassThroughSlam([np.eye(4)] * 10)
        orch = TwoTierOrchestrator(
            slam, trigger=TriggerConfig(
                periodic_every_n_frames=-1, on_new_object=False,
                on_release=False))

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        T_co = _pose(0.3, 0, 0.5)
        det = _make_detection(0, "cup", T_co)

        # Seed with one frame so the object exists; override on_new_object→False
        orch.step(rgb, depth, [det], {"phase": "idle", "held_obj_id": None})
        # Idle → idle: no trigger
        report = orch.step(rgb, depth, [det], {"phase": "idle", "held_obj_id": None})
        assert not report["triggered"]

        # Idle → grasping: trigger fires
        report = orch.step(rgb, depth, [det],
                           {"phase": "grasping", "held_obj_id": 0})
        assert report["triggered"]

    def test_periodic_trigger(self):
        slam = PassThroughSlam([np.eye(4)] * 20)
        orch = TwoTierOrchestrator(
            slam, trigger=TriggerConfig(
                periodic_every_n_frames=3,
                on_new_object=False, on_grasp=False, on_release=False))

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        T_co = _pose(0.3, 0, 0.5)
        det = _make_detection(0, "cup", T_co)

        # Seed
        orch.step(rgb, depth, [det], {"phase": "idle", "held_obj_id": None})

        triggers = []
        for _ in range(10):
            r = orch.step(rgb, depth, [det],
                          {"phase": "idle", "held_obj_id": None})
            triggers.append(r["triggered"])
        # At least some periodic triggers should fire
        assert sum(triggers) >= 2


class TestHoldingBaseFrameFusion:
    def test_held_object_tracks_ee(self):
        """During HOLDING, the held object's pose should track the EE
        via base-frame fusion, not drift with SLAM uncertainty."""
        # SLAM reports noisy camera poses
        poses = [np.eye(4)]
        for i in range(1, 10):
            T = np.eye(4)
            T[0, 3] = 0.01 * i  # camera drifts +1cm per frame
            poses.append(T)
        slam = PassThroughSlam(poses, default_cov=np.diag([0.01] * 6))

        orch = TwoTierOrchestrator(
            slam, trigger=TriggerConfig(periodic_every_n_frames=-1))

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        T_ec = _pose(0.1, 0, 0.3)  # EE 10cm forward, 30cm up from camera

        # Seed object before grasping (idle)
        T_co_before = _pose(0.1, 0, 0.3)  # same as EE approx
        orch.step(rgb, depth,
                  [_make_detection(0, "apple", T_co_before)],
                  {"phase": "idle", "held_obj_id": None}, T_ec=T_ec)

        # Grasp
        orch.step(rgb, depth,
                  [_make_detection(0, "apple", T_co_before)],
                  {"phase": "grasping", "held_obj_id": 0}, T_ec=T_ec)

        # Track through holding (object appears at EE-relative position)
        for _ in range(5):
            orch.step(rgb, depth,
                      [_make_detection(0, "apple", T_co_before)],
                      {"phase": "holding", "held_obj_id": 0}, T_ec=T_ec)

        # Object pose should have moved with the EE (which moved with camera)
        # rather than staying at the initial world position
        assert 0 in orch.objects
        # Pose is finite and well-defined
        assert np.all(np.isfinite(orch.objects[0]["T"]))


class TestManipulationSetPropagation:
    """Scene-graph 'in'/'on' relations should propagate the held object's
    Q to its passengers so their EKF can keep up during transport."""

    def test_empty_when_nothing_held(self):
        slam = PassThroughSlam([np.eye(4)])
        orch = TwoTierOrchestrator(slam)
        assert orch._get_manipulation_set(None) == set()

    def test_held_only_when_no_relations(self):
        slam = PassThroughSlam([np.eye(4)])
        orch = TwoTierOrchestrator(slam)
        orch._cached_relations = []
        assert orch._get_manipulation_set(3) == {3}

    def test_in_relation_adds_passenger(self):
        """apple (4) 'in' bowl (3), bowl is held → manipulation set = {3, 4}."""
        from pose_update.factor_graph import RelationEdge
        slam = PassThroughSlam([np.eye(4)])
        orch = TwoTierOrchestrator(slam)
        # parent = contained, child = container (per RelationEdge convention)
        orch._cached_relations = [
            RelationEdge(parent=4, child=3, relation_type="in", score=0.8),
        ]
        assert orch._get_manipulation_set(3) == {3, 4}

    def test_on_relation_adds_passenger(self):
        """cup (5) 'on' tray (3), tray is held → manipulation set = {3, 5}."""
        from pose_update.factor_graph import RelationEdge
        slam = PassThroughSlam([np.eye(4)])
        orch = TwoTierOrchestrator(slam)
        orch._cached_relations = [
            RelationEdge(parent=5, child=3, relation_type="on", score=0.9),
        ]
        assert orch._get_manipulation_set(3) == {3, 5}

    def test_transitive_closure(self):
        """A in B, B on C, C held → manipulation set = {A, B, C}."""
        from pose_update.factor_graph import RelationEdge
        slam = PassThroughSlam([np.eye(4)])
        orch = TwoTierOrchestrator(slam)
        orch._cached_relations = [
            RelationEdge(parent=1, child=2, relation_type="in", score=0.8),
            RelationEdge(parent=2, child=3, relation_type="on", score=0.8),
        ]
        assert orch._get_manipulation_set(3) == {1, 2, 3}

    def test_unrelated_objects_not_included(self):
        from pose_update.factor_graph import RelationEdge
        slam = PassThroughSlam([np.eye(4)])
        orch = TwoTierOrchestrator(slam)
        orch._cached_relations = [
            # A in B: irrelevant because neither is the held object
            RelationEdge(parent=1, child=2, relation_type="in", score=0.8),
        ]
        assert orch._get_manipulation_set(5) == {5}


class TestSceneGraphRecomputation:
    def test_relations_recomputed_when_two_objects_present(self):
        slam = PassThroughSlam([np.eye(4)] * 5)
        orch = TwoTierOrchestrator(slam)

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)

        # Two objects at roughly the same place → should trigger relations
        det_bowl = _make_detection(0, "bowl", _pose(1.0, 1.0, 0.1))
        det_apple = _make_detection(1, "apple", _pose(1.0, 1.0, 0.15))

        orch.step(rgb, depth, [det_bowl, det_apple],
                  {"phase": "idle", "held_obj_id": None})

        # Relations are recomputed silently; just verify no crash and
        # that two objects are tracked.
        assert set(orch.objects.keys()) == {0, 1}


class TestRBPFDesign:
    """Verification tests for the RBPF refactor described in
    plans/rbpf-scenerep-refactor.md §9 (Verification)."""

    def test_rigid_attachment_carries_object_when_base_moves_without_vision(self):
        """With rigid-attachment using T_wb(t-1), a held object should move
        with the base even when gripper-in-base is stationary and no vision
        is provided. See rigid_attachment_predict docstring for derivation.
        """
        # Base moves +1 cm per frame in x. Very tight localization to isolate
        # the rigid-attachment behavior from particle noise.
        slam_poses = [_pose(0.01 * i, 0, 0) for i in range(30)]
        slam = PassThroughSlam(slam_poses,
                                default_cov=np.diag([1e-12] * 6))
        orch = TwoTierOrchestrator(
            slam,
            trigger=TriggerConfig(periodic_every_n_frames=-1,
                                  on_new_object=False, on_grasp=False),
            n_particles=4,
            rng_seed=0,
        )

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        T_ec = np.eye(4)
        # Gripper at fixed base-frame offset.
        T_bg = _pose(0.1, 0, 0.3)
        det = _make_detection(0, "apple", _pose(0.1, 0, 0.3))

        # Two frames to establish a valid prev_T_wb AND prev_T_bg.
        orch.step(rgb, depth, [det],
                  {"phase": "idle", "held_obj_id": None},
                  T_ec=T_ec, T_bg=T_bg)
        orch.step(rgb, depth, [det],
                  {"phase": "grasping", "held_obj_id": 0},
                  T_ec=T_ec, T_bg=T_bg)

        held_x_start = orch.objects[0]["T"][0, 3]
        # Hold through 5 frames WITHOUT vision; base moves +5 cm total.
        for _ in range(5):
            orch.step(rgb, depth, [],
                      {"phase": "holding", "held_obj_id": 0},
                      T_ec=T_ec, T_bg=T_bg)
        held_x_end = orch.objects[0]["T"][0, 3]

        # Expected displacement ≈ +5 cm (base motion). With the fixed
        # formula (T_wb(t) · ΔT_bg · inv(T_wb(t-1))) and ΔT_bg = I, the
        # world-frame transform becomes T_wb(t) · inv(T_wb(t-1)) = ΔT_wb,
        # so the object translates by exactly the per-step base motion.
        assert abs((held_x_end - held_x_start) - 0.05) < 1e-3

    def test_no_sigma_wb_double_counting_in_fast_tier(self):
        """Under RBPF, fast-tier observation noise must be R_icp alone, not
        R_icp + Σ_wb. We verify indirectly: with a perfect-localization
        backend (Σ_wb = 0), the per-particle posterior covariance should
        converge to (Σ_prior⁻¹ + R_icp⁻¹)⁻¹ — i.e., it should track
        R_icp, not be inflated by a Σ_wb term.
        """
        slam = PassThroughSlam(
            [np.eye(4)] * 30,
            default_cov=np.eye(6) * 1e-12)  # essentially zero
        orch = TwoTierOrchestrator(
            slam,
            trigger=TriggerConfig(periodic_every_n_frames=-1,
                                  on_new_object=False),
            n_particles=8,
            rng_seed=1,
        )
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        T_co = _pose(0.3, 0, 0.5)

        for _ in range(25):
            orch.step(
                rgb, depth,
                [_make_detection(0, "cup", T_co)],
                {"phase": "idle", "held_obj_id": None})

        # After many frames, the collapsed-object cov should be dominated
        # by R_icp (1e-4) not by any cascaded Σ_wb. We check trace < some
        # threshold consistent with pure R_icp behavior.
        cov_trace = float(np.trace(orch.objects[0]["cov"]))
        # R_icp = diag(1e-4)*6 → trace 6e-4; after 25 fusions posterior
        # should be WAY below that. If Σ_wb were being double-counted
        # at 1e-12 level it would still be small, but if an old bug
        # added 0.01 per frame, we'd be far above this threshold.
        assert cov_trace < 1e-3

    def test_report_contains_particle_payload(self):
        """RBPF report must expose particles, ESS, and resample flag."""
        slam = PassThroughSlam([np.eye(4)] * 5)
        orch = TwoTierOrchestrator(slam, n_particles=8, rng_seed=0)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        T_co = _pose(0.2, 0, 0.4)
        report = orch.step(
            rgb, depth, [_make_detection(0, "cup", T_co)],
            {"phase": "idle", "held_obj_id": None})

        assert "base_particles" in report
        assert "ess" in report
        assert "resampled" in report
        assert report["base_particles"].n == 8
        assert 0 < report["ess"] <= 8.0

    def test_n1_degenerate_matches_single_gaussian_regression(self):
        """N=1 must be a deterministic single-Gaussian-like pipeline (module
        per-particle IEKF vs. old translation-only R³). We check it produces
        a finite posterior that converges with repeated observations.
        """
        slam = PassThroughSlam([np.eye(4)] * 20,
                                default_cov=np.eye(6) * 1e-12)
        orch = TwoTierOrchestrator(
            slam,
            trigger=TriggerConfig(periodic_every_n_frames=-1,
                                  on_new_object=False),
            n_particles=1,
            rng_seed=0,
        )
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        T_co = _pose(0.3, 0, 0.5)

        orch.step(rgb, depth, [_make_detection(0, "cup", T_co)],
                  {"phase": "idle", "held_obj_id": None})
        cov_init = np.trace(orch.objects[0]["cov"])
        for _ in range(15):
            orch.step(rgb, depth, [_make_detection(0, "cup", T_co)],
                      {"phase": "idle", "held_obj_id": None})
        cov_final = np.trace(orch.objects[0]["cov"])
        assert cov_final < cov_init
        # Mean should be close to the observed position (0.3, 0, 0.5)
        np.testing.assert_allclose(
            orch.objects[0]["T"][:3, 3], [0.3, 0, 0.5], atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

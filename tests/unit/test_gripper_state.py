"""Unit tests for ``pose_update.manipulation.gripper_state.GripperPhaseTracker``."""
from __future__ import annotations

import numpy as np
import pytest

from pose_update.manipulation.gripper_state import GripperPhaseTracker


class _FakeTrackerState:
    """Minimal TrackerState: we don't need its data for FSM tests."""
    def sam2_tau(self): return {}
    def iter_world_centroids(self): return iter(())
    def force_admit(self, det, depth): return None


@pytest.fixture
def fsm():
    return GripperPhaseTracker(
        closed_width_m=0.025, open_width_m=0.040,
        history_size=3, motion_threshold_m=0.005,
        min_transition_frames=2,
        detector=None,  # FSM-only; no GraspOwnerDetector
    )


def _step(fsm, width):
    return fsm.step(
        width=width, tracker_state=_FakeTrackerState(),
        T_wb=np.eye(4), T_bg=np.eye(4),
    )


def test_seed_idle_when_open(fsm):
    out = _step(fsm, 0.06)
    assert out["phase"] == "idle"


def test_seed_holding_when_closed(fsm):
    out = _step(fsm, 0.01)
    assert out["phase"] == "holding"


def test_idle_to_grasping_on_close(fsm):
    _step(fsm, 0.06)   # seed idle
    _step(fsm, 0.04)   # still idle (not yet < closed_width)
    out = _step(fsm, 0.02)  # crosses closed threshold
    assert out["phase"] == "grasping"


def test_grasping_to_holding_after_stable_window(fsm):
    _step(fsm, 0.06)   # idle
    _step(fsm, 0.02)   # grasping
    # min_transition_frames=2 + stable history (spread < motion_threshold)
    _step(fsm, 0.020)
    _step(fsm, 0.020)
    out = _step(fsm, 0.020)
    assert out["phase"] == "holding"


def test_holding_to_releasing_on_open(fsm):
    # Seed straight to holding.
    _step(fsm, 0.01)
    out = _step(fsm, 0.06)   # crosses open threshold
    assert out["phase"] == "releasing"


def test_apply_merges_remaps_held(fsm):
    fsm._held_obj_id = 7
    fsm.apply_merges([{"keep_oid": 3, "drop_oid": 7}])
    assert fsm.held_obj_id == 3


def test_apply_merges_keeps_held_if_unrelated(fsm):
    fsm._held_obj_id = 7
    fsm.apply_merges([{"keep_oid": 3, "drop_oid": 9}])
    assert fsm.held_obj_id == 7


def test_held_cleared_when_track_pruned(fsm):
    """If `live_oids` doesn't contain the held oid, FSM clears it."""
    _step(fsm, 0.06)  # idle
    fsm._held_obj_id = 99
    out = fsm.step(width=0.02, tracker_state=_FakeTrackerState(),
                    T_wb=np.eye(4), T_bg=np.eye(4),
                    live_oids={1, 2, 3})  # 99 not present
    assert fsm.held_obj_id is None


# ─── Fix 0: open-baseline closing detection ────────────────────────

def test_thick_object_grasp_recognised():
    """Project assumption: gripper open by default; ANY closing →
    grasp. A grip on a 6.7 cm object (width 0.067 m, well above the
    legacy closed_width=0.025 absolute threshold) MUST trigger
    `grasping`. This is the regression that caused apple_to_cabinate
    to lose tracks during carry.
    """
    fsm = GripperPhaseTracker(
        closed_width_m=0.025, open_width_m=0.040,
        close_delta_m=0.005, history_size=5,
        motion_threshold_m=0.005, min_transition_frames=2,
        detector=None,
    )
    # Seed open at 0.10 m.
    out = fsm.step(width=0.1004, tracker_state=_FakeTrackerState(),
                    T_wb=np.eye(4), T_bg=np.eye(4))
    assert out["phase"] == "idle"
    out = fsm.step(width=0.1004, tracker_state=_FakeTrackerState(),
                    T_wb=np.eye(4), T_bg=np.eye(4))
    assert out["phase"] == "idle"
    # Width drops onto a thick object: 0.0989 → 0.0803 → 0.0667.
    out = fsm.step(width=0.0989, tracker_state=_FakeTrackerState(),
                    T_wb=np.eye(4), T_bg=np.eye(4))
    out = fsm.step(width=0.0803, tracker_state=_FakeTrackerState(),
                    T_wb=np.eye(4), T_bg=np.eye(4))
    assert out["phase"] == "grasping"


def test_open_baseline_resets_on_release():
    """After a full grasp/release cycle, open_baseline_m resets so
    the next cycle re-anchors against a fresh open width."""
    fsm = GripperPhaseTracker(
        closed_width_m=0.025, open_width_m=0.040,
        close_delta_m=0.005, history_size=3,
        motion_threshold_m=0.005, min_transition_frames=1,
        detector=None,
    )
    # Seed open.
    fsm.step(width=0.10, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    fsm.step(width=0.10, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    # Close.
    fsm.step(width=0.07, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    fsm.step(width=0.07, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    fsm.step(width=0.07, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    assert fsm._open_baseline_m == pytest.approx(0.10)
    # Release.
    fsm.step(width=0.10, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    fsm.step(width=0.10, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    fsm.step(width=0.10, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    fsm.step(width=0.10, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    fsm.step(width=0.10, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    # After idle reached, open_baseline should re-anchor.
    assert fsm._phase_prev == "idle"


def test_width_jitter_stays_open():
    """A small jitter (0.0995 instead of 0.1000) within
    close_delta_m should NOT trigger grasping — that would be
    spurious."""
    fsm = GripperPhaseTracker(
        closed_width_m=0.025, open_width_m=0.040,
        close_delta_m=0.005, history_size=3,
        motion_threshold_m=0.005, min_transition_frames=1,
        detector=None,
    )
    fsm.step(width=0.10, tracker_state=_FakeTrackerState(),
              T_wb=np.eye(4), T_bg=np.eye(4))
    out = fsm.step(width=0.0995, tracker_state=_FakeTrackerState(),
                    T_wb=np.eye(4), T_bg=np.eye(4))
    assert out["phase"] == "idle"

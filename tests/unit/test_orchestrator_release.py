"""Unit tests for the rigid-attachment phase gate.

Pin: a manipulation-set member gets a rigid-attachment predict ONLY
when the gripper is actually closed around the object, i.e. phase
in {grasping, holding}. During the `releasing` transition window
the FSM still carries `held_obj_id` (it's cleared at releasing →
idle) but the object has been let go and must not ride the base
any further.

Regression: apple_drop fr 269-273 — the apple's tracked centroid
moved by Δ ≈ 0.18 m alongside the base during the 5-frame releasing
window before snapping to a freeze when the FSM hit `idle`.
"""
from __future__ import annotations

import numpy as np
import pytest

# `should_apply_rigid` was lifted to GaussianEkfTracker as part of the
# orchestrator removal; alias the symbol locally to keep the test body
# unchanged.
from ekf_tracker.gaussian_ekf_tracker import GaussianEkfTracker as TwoTierOrchestrator


def _eye4():
    return np.eye(4, dtype=np.float64)


def _shifted4(dx=0.1):
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = dx
    return T


@pytest.mark.parametrize("phase", ["grasping", "holding"])
def test_rigid_applied_when_gripper_closed(phase):
    assert TwoTierOrchestrator.should_apply_rigid(
        T_bg=_shifted4(0.1), prev_T_bg=_eye4(),
        manipulation_set={6}, phase=phase) is True


def test_rigid_skipped_during_releasing():
    """Even with held identity still set and motion available,
    `releasing` must NOT trigger rigid attachment."""
    assert TwoTierOrchestrator.should_apply_rigid(
        T_bg=_shifted4(0.1), prev_T_bg=_eye4(),
        manipulation_set={6}, phase="releasing") is False


def test_rigid_skipped_when_idle():
    assert TwoTierOrchestrator.should_apply_rigid(
        T_bg=_shifted4(0.1), prev_T_bg=_eye4(),
        manipulation_set=set(), phase="idle") is False


def test_rigid_skipped_when_no_prev_proprio():
    """First frame (no previous T_bg) — can't form ΔT_bg yet."""
    assert TwoTierOrchestrator.should_apply_rigid(
        T_bg=_shifted4(0.1), prev_T_bg=None,
        manipulation_set={6}, phase="holding") is False


def test_rigid_skipped_when_no_T_bg():
    """No proprioception this frame — fall back to static predict."""
    assert TwoTierOrchestrator.should_apply_rigid(
        T_bg=None, prev_T_bg=_eye4(),
        manipulation_set={6}, phase="holding") is False


def test_rigid_skipped_when_manipulation_set_empty():
    assert TwoTierOrchestrator.should_apply_rigid(
        T_bg=_shifted4(0.1), prev_T_bg=_eye4(),
        manipulation_set=set(), phase="holding") is False


def test_rigid_phases_constant():
    """The set of rigid-attach phases is exactly {grasping, holding}."""
    assert TwoTierOrchestrator._RIGID_PHASES == frozenset(
        ("grasping", "holding"))

"""Driver-side shim wrapping :class:`utils.gripper_state.GripperPhaseTracker`."""
from __future__ import annotations


class _GripperStateInferrer:
    """Driver-side wrapper that holds the :class:`GripperPhaseTracker` instance."""
    def __init__(self, *args, **kwargs):
        # Deferred import: see module docstring for the cycle this avoids.
        from utils.gripper_state import GripperPhaseTracker as _GripperPhaseTracker
        # GraspOwnerDetector is the only kwarg the driver passes that
        # isn't part of GripperPhaseTracker's defaults; pass through
        # everything else by name.
        self._inner = _GripperPhaseTracker(*args, **kwargs)
        # Legacy attribute name used elsewhere in the driver.
        self._joints_now = None

    @property
    def _held_obj_id(self):
        return self._inner.held_obj_id

    def apply_merges(self, merges):
        self._inner.apply_merges(merges)

    def step(self, width, tracker, T_wb, T_bg, **kwargs):
        # Adapt the GaussianEkfTracker → TrackerState protocol expected
        # by the production phase tracker.
        from ekf_tracker.manipulation.grasp_owner_detector import GaussianEkfTrackerState
        ts = GaussianEkfTrackerState(tracker)
        live_oids = set(int(o) for o in tracker.object_labels.keys())
        return self._inner.step(
            width=width, tracker_state=ts,
            T_wb=T_wb, T_bg=T_bg,
            live_oids=live_oids,
            **kwargs)

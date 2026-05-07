"""Integration test for the gravity-aware predict on the apple_drop trajectory.

Asserts that running `tests/visualize_ekf_tracking.py` on `apple_drop`
produces a release-frame JSON containing a sensible gravity_predict
diagnostic, and that the predicted landing pose is written back to the
filter state in the next frame.

This test depends on the per-frame state JSONs under
`tests/visualization_pipeline/apple_drop/ekf_state/`. If those don't
exist, the test is skipped — generate them with:

    conda run -n ocmp_test python tests/visualize_ekf_tracking.py \\
        --trajectory apple_drop --max-frame 285 --no-png --no-mp4
"""
from __future__ import annotations

import json
import os

import pytest


STATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "visualization_pipeline", "apple_drop", "ekf_state",
)


def _load_state(frame: int) -> dict:
    path = os.path.join(STATE_DIR, f"frame_{frame:06d}.json")
    if not os.path.exists(path):
        pytest.skip(
            f"Missing fixture state at {path}. Regenerate with "
            "`tests/visualize_ekf_tracking.py --trajectory apple_drop "
            "--max-frame 285 --no-png --no-mp4`.")
    with open(path) as f:
        return json.load(f)


def _find_release_frame() -> int:
    """Walk frames forward, return the first one whose phase is 'idle'
    after a non-idle phase. apple_drop releases around fr 274 in the
    current dataset; this helper does not hard-code that number."""
    last_phase: str = "idle"
    for frame in range(0, 420):
        path = os.path.join(STATE_DIR, f"frame_{frame:06d}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        cur_phase = d.get("gripper_state", {}).get("phase", "idle")
        if (last_phase in ("holding", "releasing")
                and cur_phase not in ("holding", "releasing")):
            return frame
        last_phase = cur_phase
    pytest.skip("No release transition found in apple_drop fixtures.")


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestGravityPredictAppleDrop:
    def test_release_frame_has_gravity_predict_record(self):
        release_frame = _find_release_frame()
        d = _load_state(release_frame)
        gp = d.get("gravity_predict")
        assert gp is not None, (
            f"No gravity_predict record at release frame {release_frame}. "
            "Voxel observability + gravity hook may not be wired.")
        # Record schema sanity.
        for k in ("column_state", "drop_height_m", "surface_z",
                  "first_unseen_z", "floor_z", "sigma_xy", "sigma_z",
                  "sigma_yaw", "landing_z", "skipped", "oid", "frame"):
            assert k in gp, f"missing key {k!r} in gravity_predict log"
        assert gp["skipped"] is False, "gravity_predict skipped at release"
        assert gp["column_state"] in (
            "hit_occupied", "mixed_unseen", "all_unseen", "all_empty")
        assert gp["frame"] == release_frame

    def test_predicted_landing_is_below_release(self):
        release_frame = _find_release_frame()
        d = _load_state(release_frame)
        gp = d["gravity_predict"]
        oid = str(gp["oid"])
        # Release-frame mu_w (post_update is AFTER the gravity_predict
        # overwrite, so check the prev-frame's xyz_w as the release height).
        d_prev = _load_state(release_frame - 1)
        tr_prev = d_prev["tracks_post_update"].get(oid)
        assert tr_prev is not None, f"oid {oid} missing in prev frame"
        z_release = float(tr_prev["xyz_w"][2])
        assert gp["landing_z"] <= z_release + 1e-3, (
            f"landing_z={gp['landing_z']:.3f} should be ≤ "
            f"release z={z_release:.3f}")
        assert gp["drop_height_m"] >= 0.0

    def test_post_release_pose_matches_predicted_landing(self):
        # The gravity-predict overwrite should cause the next frame's
        # post_update xyz_w to land on the predicted landing_z.
        release_frame = _find_release_frame()
        d = _load_state(release_frame)
        gp = d["gravity_predict"]
        oid = str(gp["oid"])
        d_next = _load_state(release_frame + 1)
        tr_next = d_next["tracks_post_update"].get(oid)
        if tr_next is None:
            pytest.skip(f"oid {oid} pruned in frame {release_frame + 1}")
        z_next = float(tr_next["xyz_w"][2])
        # Allow ≤ 5 mm slop from base ↔ world rounding.
        assert abs(z_next - gp["landing_z"]) < 0.005, (
            f"post-release z={z_next:.3f} doesn't match landing_z="
            f"{gp['landing_z']:.3f}")

    def test_uncertainty_is_positive(self):
        release_frame = _find_release_frame()
        d = _load_state(release_frame)
        gp = d["gravity_predict"]
        assert gp["sigma_xy"] > 0.0
        assert gp["sigma_z"] > 0.0
        assert 0.0 <= gp["sigma_yaw"] <= 3.2  # ≤ π
        # All sigmas should be sub-workspace (≤ 3 m).
        assert gp["sigma_xy"] < 3.0
        assert gp["sigma_z"] < 3.0

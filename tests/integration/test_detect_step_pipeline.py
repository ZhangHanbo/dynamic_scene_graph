"""End-to-end smoke test: live `detect` → `step` on apple_drop.

Drives 30 frames of `apple_drop` through `EkfTracker.detect` and feeds
the resulting detections into `EkfTracker.step`. Confirms:

  * The OWL + SAM2 streaming session opens, propagates frames, and
    mints stable `object_id` values.
  * The detection schema drops cleanly into `step()` without
    transformation.
  * `step()` produces a non-empty `SceneView` whose oids match the
    detected `object_id`s for at least one frame in the slice.

Skipped automatically when either the OWL or SAM2 server is offline.
The render task (`tests/diagnostics/render_api_videos.py`) hard-errors
in that case; we want this test to skip so unit-test runs aren't blocked
by network state.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ekf_tracker.api import EkfTracker  # noqa: E402
from scripts.rosbag2dataset.server_configs import (  # noqa: E402
    OWL_SERVER_URL, SAM2_SERVER_URL, OWL_DETECT_PATH, SAM2_STREAM_INIT_PATH,
)


DATA = ROOT / "datasets" / "apple_drop"
N_FRAMES = 30


def _server_reachable(url: str, path: str, timeout: float = 2.0) -> bool:
    """Quick HEAD/GET probe to test if the endpoint is up.

    A 200 / 405 / 422 are all evidence of "service is responding";
    anything else (timeout, connection refused) is "unreachable".
    """
    try:
        r = requests.get(url.rstrip("/") + path, timeout=timeout)
        return r.status_code in (200, 405, 422)
    except (requests.exceptions.RequestException, Exception):
        return False


@pytest.fixture(scope="module")
def servers_up() -> bool:
    return (_server_reachable(OWL_SERVER_URL, OWL_DETECT_PATH)
            and _server_reachable(SAM2_SERVER_URL, SAM2_STREAM_INIT_PATH))


def _load_amcl(path: Path) -> List[np.ndarray]:
    from scipy.spatial.transform import Rotation
    out: List[np.ndarray] = []
    for line in open(path):
        a = line.strip().split()
        if len(a) != 8:
            continue
        _, tx, ty, tz, qx, qy, qz, qw = map(float, a)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        out.append(T)
    return out


def test_detect_then_step_produces_stable_ids(servers_up):
    if not servers_up:
        pytest.skip("OWL or SAM2 server unreachable; "
                    "live detect→step test requires both.")
    if not (DATA / "rgb").exists():
        pytest.skip(f"apple_drop dataset not available at {DATA}")

    K = np.array([[554.3827, 0.0, 320.5],
                  [0.0, 554.3827, 240.5],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    slam_poses = _load_amcl(DATA / "pose_txt" / "amcl_pose.txt")
    if len(slam_poses) < N_FRAMES:
        pytest.skip(f"need >= {N_FRAMES} amcl poses, have {len(slam_poses)}")

    api = EkfTracker(K=K, T_bc=np.eye(4), relation_backend="none")
    history = None
    seen_ids: set = set()
    matched_frames = 0

    try:
        for idx in range(N_FRAMES):
            rgb = np.array(Image.open(
                DATA / "rgb" / f"rgb_{idx:06d}.png").convert("RGB"))
            depth = np.load(
                DATA / "depth" / f"depth_{idx:06d}.npy").astype(np.float32)

            dets, history = api.detect(
                rgb,
                vocabulary=["apple", "cup", "bottle", "cabinet", "bowl"],
                history=history,
            )

            # Schema check on first non-empty frame.
            if dets:
                d = dets[0]
                # Drops cleanly into step().
                assert "id" in d and "mask" in d, f"missing keys: {d.keys()}"
                assert isinstance(d["id"], int), type(d["id"])
                assert d["id"] == d["object_id"]
                seen_ids.update(int(x["id"]) for x in dets)

            scene = api.step(
                detections=dets, rgb=rgb, depth=depth,
                slam_pose=slam_poses[idx], T_bc=np.eye(4))

            # When a detection's id appears in the scene, the EKF has
            # successfully consumed it.
            if scene.objects:
                ids_in_dets = {int(x["id"]) for x in dets}
                ids_in_scene = set(scene.objects.keys())
                if ids_in_dets & ids_in_scene:
                    matched_frames += 1
    finally:
        if history is not None:
            history.close()

    # Soft assertions — the live pipeline should produce SOME
    # detections with stable ids over 30 frames of apple_drop, and
    # at least one frame should have detections that survive into the
    # EKF scene (i.e. r >= r_conf).
    assert seen_ids, ("no detection ids seen across 30 frames; OWL "
                      "vocabulary may not match this trajectory.")
    assert matched_frames >= 1, (
        f"no frame had a detection-id surviving as an EKF track; "
        f"seen_ids={sorted(seen_ids)}")

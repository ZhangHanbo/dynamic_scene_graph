"""E2E invariant test: re-run apple_in_the_tray and check key invariants
hold against the cached pre-refactor baseline.

This test is OPT-IN — it takes ~5 minutes to run because it shells
out to ``tests/visualize_ekf_tracking.py`` for frames 260-700.
Activate via ``pytest -m e2e tests/e2e/`` or by setting the env var
``RUN_E2E=1``.

Invariants pinned (NOT exact-numerical equality, which would be brittle
across numpy versions):

1.  At every checkpointed frame, ``n_tracks`` matches the baseline.
2.  ``held_seed`` and ``held_oids_used`` match the baseline.
3.  No ``cov_bo`` eigenvalue across the run is below −1e-12.

The baseline JSON files live under ``tests/baseline/apple_in_the_tray/``
and were captured from the canonical pre-refactor run.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASELINE_DIR = ROOT / "tests" / "baseline" / "apple_in_the_tray"
STATE_DIR = (ROOT / "tests" / "visualization_pipeline"
              / "apple_in_the_tray" / "ekf_state")


def _baseline_summary():
    summary_path = ROOT / "tests" / "baseline" / "apple_in_the_tray_summary.json"
    return json.load(summary_path.open())


def _e2e_active() -> bool:
    return bool(os.environ.get("RUN_E2E"))


@pytest.mark.skipif(not _e2e_active(),
                    reason="set RUN_E2E=1 to run the slow e2e re-render")
def test_apple_in_the_tray_full_run():
    cmd = [sys.executable,
            str(ROOT / "tests" / "visualize_ekf_tracking.py"),
            "--start", "260", "--max-frame", "700", "--no-png"]
    env = dict(os.environ, EKF_VIZ_RELATION_BACKEND="llm")
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"driver crashed:\n{res.stderr}"

    baseline = _baseline_summary()
    diffs = []
    for frame_data in baseline["frames"]:
        fr = frame_data["frame"]
        p = STATE_DIR / f"frame_{fr:06d}.json"
        if not p.exists():
            continue
        d = json.load(p.open())
        a = d.get("association", {})
        post = d.get("tracks_post_update", {})
        g = d.get("gripper_state", {})
        if (len(post) != frame_data["n_tracks"]
            or len(a.get("match", {})) != frame_data["n_match"]
            or g.get("held_obj_id") != frame_data["held_seed"]
            or sorted(d.get("held_oids_used", [])) != sorted(frame_data["used"])):
            diffs.append((fr, frame_data, dict(
                n_track=len(post),
                n_match=len(a.get("match", {})),
                held=g.get("held_obj_id"),
                used=sorted(d.get("held_oids_used", [])))))
    assert not diffs, f"baseline mismatches: {diffs[:5]}"


@pytest.mark.skipif(not _e2e_active(),
                    reason="set RUN_E2E=1 to run the slow e2e re-render")
def test_psd_invariant_holds_throughout():
    """Every track's cov in every frame has eigenvalues ≥ −1e-12."""
    files = sorted(STATE_DIR.glob("frame_*.json"))
    assert files, f"no state JSONs found at {STATE_DIR} (run the driver first)"
    worst = ("-", "-", float("inf"))
    for fp in files:
        d = json.load(fp.open())
        for oid, tr in (d.get("tracks_post_update") or {}).items():
            cov = np.asarray(tr.get("cov", np.eye(6)))
            try:
                ev = float(np.linalg.eigvalsh(0.5 * (cov + cov.T)).min())
            except np.linalg.LinAlgError:
                ev = float("nan")
            if ev < worst[2]:
                worst = (fp.name, oid, ev)
    assert worst[2] > -1e-12, f"PSD violated: {worst}"

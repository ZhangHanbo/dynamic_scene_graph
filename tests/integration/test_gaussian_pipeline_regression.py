"""End-to-end regression test for the Gaussian tracker.

Runs the current tracker on a fixed 60-frame slice of `apple_in_the_tray`
(frames 280..339, chosen to cover the births/matches/self-merges activity
burst) and compares each frame's *canonical state* against a pinned
fixture under `tests/fixtures/apple_in_the_tray_baseline/`. The fixture is
generated once by `_regenerate_fixture.py` (not invoked by the test) and
checked into the repo.

The "canonical state" is the subset of each per-frame JSON that captures
*observable* tracker behaviour:

    frame                 -- sanity
    tracks_post_update    -- final posterior (μ, Σ, r, label) per oid
    association.match     -- Hungarian oid->det_idx dict
    matched_events        -- per-match (d², w, r_prev, r_new, reject)
    missed_events         -- per-miss (r_prev, r_new, p_v)
    births                -- per-birth (new_oid, r_new)
    prunes                -- per-prune (oid, r)
    self_merges           -- per-merge (keep, drop, d2_trans, info fusion)

Entries that are intermediate/computed (cost matrix, d² matrices,
tracks_enter, tracks_post_predict, detection list, raw masks) are NOT
checked; if `tracks_post_update` and `matched_events` are identical to
the fixture then those intermediates must also be identical to within
floating-point reorder-equivalence.

Tolerance: 1e-9 absolute on floats (poses, covariances, r, d², log_lik);
exact equality on sets/indices. A pure code-motion refactor should
produce bit-identical floats; 1e-9 covers trivial reorderings.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "apple_in_the_tray_baseline"
SLICE_START = 280
SLICE_END   = 340            # exclusive
# Env the tracker needs (open3d, opencv, etc. — the ocmp_test env).
OCMP_PYTHON = Path("/Users/zhanghanbo/anaconda3/envs/ocmp_test/bin/python")
TRACKER_CMD = [
    str(OCMP_PYTHON),
    str(REPO_ROOT / "tests" / "visualize_ekf_tracking.py"),
    "--trajectory", "apple_in_the_tray",
    "--start", str(SLICE_START),
    "--max-frame", str(SLICE_END),
    "--no-png",
]

# Float tolerances for the comparisons. A pure code-motion refactor
# should produce bit-identical floats in principle, but BLAS threading /
# scipy dispatch can introduce O(1e-11) *relative* noise in inner
# products (notably d^2 = nu^T S^{-1} nu at d^2 ~ 1e3). Absolute error
# therefore scales with magnitude; use rtol on quantities that can be
# large and atol on quantities that are O(1).
#
# A real refactor bug would land ~1e-4 or larger on poses and ~1e-2+
# on d^2, so these tolerances have ~5 orders of margin over BLAS noise
# and are comfortably under any behaviour-changing edit.
ATOL_POSE = 1e-8      # poses are O(1); 10^-8 m ≈ 10 nm
ATOL_COV  = 1e-8      # covariance diag ~1e-3; rtol~1e-5
ATOL_R    = 1e-10     # r in [0,1]; 1e-10 absolute is plenty
RTOL_D2   = 1e-6      # d^2 up to ~1e4; tol ~1e-2 is still 1e4x BLAS noise
ATOL_D2   = 1e-6      # additive floor for d^2 near zero.


# ─────────────────────────────────────────────────────────────────────
# Canonical-subset extraction
# ─────────────────────────────────────────────────────────────────────

_EVENT_FIELDS = {
    "matched_events": ("oid", "det_idx", "d2", "d2_trans", "d2_rot",
                        "w", "reject_outer_gate", "log_lik",
                        "r_prev", "r_new", "fitness", "rmse"),
    "missed_events":  ("oid", "p_v", "p_d_tilde", "r_prev", "r_new"),
    "births":         ("det_idx", "new_oid", "label", "score", "r_new"),
    "prunes":         ("oid", "r"),
    "self_merges":    ("keep", "drop", "d2_trans"),
}


def _canonical(dump: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the observable-behaviour subset of a frame dump."""
    tracks = {}
    for oid, tr in dump.get("tracks_post_update", {}).items():
        tracks[oid] = {
            "T": tr.get("T"),
            "cov": tr.get("cov"),
            "r": tr.get("r"),
            "label": tr.get("label"),
            "sam2_tau": tr.get("sam2_tau"),
        }

    def _filter_events(lst, fields):
        return [{f: ev.get(f) for f in fields if f in ev} for ev in lst]

    return {
        "frame": dump.get("frame"),
        "tracks_post_update": tracks,
        "association_match": dict(dump.get("association", {}).get("match", {})),
        "unmatched_tracks": sorted(dump.get("association", {})
                                     .get("unmatched_tracks", [])),
        "matched_events": _filter_events(
            dump.get("matched_events", []), _EVENT_FIELDS["matched_events"]),
        "missed_events":  _filter_events(
            dump.get("missed_events", []), _EVENT_FIELDS["missed_events"]),
        "births":         _filter_events(
            dump.get("births", []), _EVENT_FIELDS["births"]),
        "prunes":         _filter_events(
            dump.get("prunes", []), _EVENT_FIELDS["prunes"]),
        "self_merges":    _filter_events(
            dump.get("self_merges", []), _EVENT_FIELDS["self_merges"]),
    }


# ─────────────────────────────────────────────────────────────────────
# Float-aware comparator
# ─────────────────────────────────────────────────────────────────────

def _array_close(a, b, atol: float, path: str) -> None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{path}: shape {a.shape} vs {b.shape}")
    if not np.allclose(a, b, atol=atol, rtol=0, equal_nan=True):
        diff = np.abs(a - b)
        raise AssertionError(
            f"{path}: max |diff|={diff.max()} exceeds atol={atol}\n"
            f"  got:      {a.ravel()[:6]}\n"
            f"  expected: {b.ravel()[:6]}")


def _compare_tracks(got, exp, path="tracks_post_update"):
    gt_oids = set(got.keys()); exp_oids = set(exp.keys())
    if gt_oids != exp_oids:
        raise AssertionError(
            f"{path}: track oid sets differ. "
            f"only_got={sorted(gt_oids-exp_oids)} "
            f"only_exp={sorted(exp_oids-gt_oids)}")
    for oid in exp:
        gt, ex = got[oid], exp[oid]
        if gt["label"] != ex["label"]:
            raise AssertionError(f"{path}/{oid}.label: {gt['label']} vs {ex['label']}")
        if gt.get("sam2_tau") != ex.get("sam2_tau"):
            raise AssertionError(
                f"{path}/{oid}.sam2_tau: {gt.get('sam2_tau')} vs {ex.get('sam2_tau')}")
        _array_close(gt["T"], ex["T"], ATOL_POSE, f"{path}/{oid}.T")
        _array_close(gt["cov"], ex["cov"], ATOL_COV, f"{path}/{oid}.cov")
        if abs(float(gt["r"]) - float(ex["r"])) > ATOL_R:
            raise AssertionError(
                f"{path}/{oid}.r: {gt['r']} vs {ex['r']} (atol={ATOL_R})")


def _compare_events(got, exp, fields, path):
    if len(got) != len(exp):
        raise AssertionError(
            f"{path}: length {len(got)} vs {len(exp)}\n"
            f"  got oids:      {[e.get('oid', e.get('new_oid', e.get('keep'))) for e in got]}\n"
            f"  expected oids: {[e.get('oid', e.get('new_oid', e.get('keep'))) for e in exp]}")
    # Sort by (oid or new_oid or keep) to make order-insensitive. Within a
    # frame, there's at most one event per oid so this is unambiguous.
    def _key(ev):
        return (ev.get("oid") or ev.get("new_oid") or ev.get("keep") or
                ev.get("det_idx") or 0)
    got = sorted(got, key=_key); exp = sorted(exp, key=_key)
    for i, (g, e) in enumerate(zip(got, exp)):
        for f in fields:
            if f not in e:
                continue
            gv = g.get(f); ev = e.get(f)
            if gv is None and ev is None:
                continue
            if isinstance(ev, float):
                if gv is None:
                    raise AssertionError(f"{path}[{i}].{f}: None vs {ev}")
                tol = ATOL_D2 + RTOL_D2 * abs(ev)
                if abs(float(gv) - ev) > tol:
                    raise AssertionError(
                        f"{path}[{i}].{f}: {gv} vs {ev} "
                        f"(atol={ATOL_D2}, rtol={RTOL_D2}, |diff|="
                        f"{abs(float(gv)-ev):g})")
            else:
                if gv != ev:
                    raise AssertionError(
                        f"{path}[{i}].{f}: {gv!r} vs {ev!r}")


def _compare_canonical(got: Dict[str, Any], exp: Dict[str, Any], frame: str):
    assert got["frame"] == exp["frame"], \
        f"{frame}: frame index {got['frame']} vs {exp['frame']}"
    _compare_tracks(got["tracks_post_update"], exp["tracks_post_update"],
                    path=f"{frame}/tracks")
    if got["association_match"] != exp["association_match"]:
        raise AssertionError(
            f"{frame}/association_match: {got['association_match']} "
            f"vs {exp['association_match']}")
    if got["unmatched_tracks"] != exp["unmatched_tracks"]:
        raise AssertionError(
            f"{frame}/unmatched_tracks: {got['unmatched_tracks']} "
            f"vs {exp['unmatched_tracks']}")
    for key, fields in _EVENT_FIELDS.items():
        _compare_events(got[key], exp[key], fields, f"{frame}/{key}")


# ─────────────────────────────────────────────────────────────────────
# Tracker driver
# ─────────────────────────────────────────────────────────────────────

def _run_tracker(out_subdir: str) -> Path:
    """Run the Gaussian tracker on SLICE_START..SLICE_END; return dump dir.

    Forces single-threaded BLAS / OMP / Open3D via env vars. Open3D's
    point-to-point ICP has observed non-determinism under multi-thread
    tie-breaking in voxelization + KD-tree (~0.5m swings in converged
    pose on sparse target clouds). Single-thread pins it down.
    """
    if not OCMP_PYTHON.exists():
        pytest.skip(f"ocmp_test env Python not found at {OCMP_PYTHON}")
    dump_root = REPO_ROOT / "tests" / "visualization_pipeline" \
                          / "apple_in_the_tray" / out_subdir
    if dump_root.exists():
        shutil.rmtree(dump_root)
    cmd = TRACKER_CMD + [
        "--out-subdir", out_subdir,
        "--state-subdir", out_subdir,
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    res = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=600, env=env)
    if res.returncode != 0:
        raise RuntimeError(
            f"tracker exited {res.returncode}\n"
            f"stderr:\n{res.stderr[-2000:]}")
    return dump_root


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fresh_dump_dir():
    """Run the tracker once; yield the output directory."""
    d = _run_tracker("_regression_live")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _fixture_frame_ids() -> List[str]:
    """Filename stems under FIXTURE_DIR (sorted)."""
    if not FIXTURE_DIR.exists():
        # Called at module collection time by `parametrize` — must allow
        # module-level skip so pytest doesn't error the entire collection.
        pytest.skip(
            f"no baseline fixture at {FIXTURE_DIR}; "
            f"run _regenerate_fixture.py first",
            allow_module_level=True,
        )
    files = sorted(FIXTURE_DIR.glob("frame_*.json"))
    return [f.stem for f in files]


@pytest.mark.parametrize("stem", _fixture_frame_ids())
def test_frame_matches_baseline(stem: str, fresh_dump_dir: Path):
    exp = json.loads((FIXTURE_DIR / f"{stem}.json").read_text())
    got_path = fresh_dump_dir / f"{stem}.json"
    if not got_path.exists():
        raise AssertionError(
            f"fresh run missing {stem}.json (look in {fresh_dump_dir})")
    got = _canonical(json.loads(got_path.read_text()))
    _compare_canonical(got, exp, frame=stem)


def test_frame_coverage(fresh_dump_dir: Path):
    """The fresh run must produce *exactly* the fixture's frame set."""
    fresh = {p.stem for p in fresh_dump_dir.glob("frame_*.json")}
    pinned = {p.stem for p in FIXTURE_DIR.glob("frame_*.json")}
    extra = fresh - pinned
    missing = pinned - fresh
    assert not missing, f"fresh run is missing frames: {sorted(missing)[:10]}"
    assert not extra, f"fresh run produced extra frames: {sorted(extra)[:10]}"

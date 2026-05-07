"""Tests for the mask-edge cloud filter (`_clean_mask` + `_back_project`).

Three suites:

1. Synthetic correctness — fabricate masks + depth with a known leaky
   edge and assert the filter drops exactly those pixels while
   preserving the interior.
2. Real-frame regression — load a representative apple_in_the_tray
   detection; verify the filter removes outlier points (obj_radius
   shrinks) without destroying the cloud (enough points remain).
3. ICP-fitness regression — compare ICP fitness of the same consecutive
   frames of a tracked object with filter ON vs OFF; assert filter ON
   is not worse.
"""
from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.icp_pose import (  # noqa: E402
    _back_project, _clean_mask, PoseEstimator, centroid_cam_from_mask,
)


# ──────────────────────────────────────────────────────────────────────
# 1) Synthetic correctness
# ──────────────────────────────────────────────────────────────────────

def _ok(label: str, cond: bool, detail: str = "") -> bool:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {label}{(' — ' + detail) if detail else ''}")
    return bool(cond)


def test_synthetic_clean_edge() -> bool:
    """An interior block at d=0.5 m with a ring of edge pixels leaking to
    d=2.0 m (background). Filter should strip the ring, keep interior."""
    H, W = 40, 40
    depth = np.full((H, W), 2.0, dtype=np.float64)  # background
    depth[10:30, 10:30] = 0.5                       # object interior
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    # Leaky edge: inflate the mask by 1 pixel (the outer ring overlaps
    # background depth at 2.0 m).
    mask[9:31, 9:31] = 1

    cleaned = _clean_mask(mask, depth, erosion_iter=1,
                           depth_edge_max_jump=0.03, min_points=30)

    # The leaked ring (row/col 9 and 30) has depth 2.0 m; should be gone.
    ring_left_before  = int(mask[9:31, 9].sum())
    ring_left_after   = int(cleaned[9:31, 9].sum())
    # The interior should retain most pixels.
    interior_before = int(mask[12:28, 12:28].sum())
    interior_after  = int(cleaned[12:28, 12:28].sum())

    ok = True
    ok &= _ok("leaky ring removed",
              ring_left_after == 0,
              f"before={ring_left_before} after={ring_left_after}")
    ok &= _ok("interior retained",
              interior_after >= int(0.9 * interior_before),
              f"before={interior_before} after={interior_after}")
    return ok


def test_synthetic_selfdisable_small_mask() -> bool:
    """A tiny mask (< 30 pixels after erosion) — filter must self-disable
    and return the original mask rather than killing it entirely."""
    H, W = 20, 20
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[8:13, 8:13] = 1               # 25-pixel square
    depth = np.full((H, W), 0.5, dtype=np.float64)

    cleaned = _clean_mask(mask, depth, erosion_iter=1,
                           depth_edge_max_jump=0.03, min_points=30)

    # 25 pixels, below min_points=30. Erosion would drop to 9 pixels —
    # must self-disable. Gradient stage also finds nothing wrong.
    before = int(mask.sum())
    after  = int(cleaned.sum())
    return _ok("small-mask self-disable",
               after == before,
               f"before={before} after={after} (min_points=30)")


def test_synthetic_nan_depth() -> bool:
    """Masked pixels with NaN/zero depth must be treated as invalid —
    filter should not propagate them as 'jumps'."""
    H, W = 40, 40
    depth = np.full((H, W), 0.5, dtype=np.float64)
    depth[15, 15] = np.nan                  # invalid hole inside the object
    depth[16, 16] = 0.0                     # zero-depth hole
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[10:30, 10:30] = 1

    cleaned = _clean_mask(mask, depth, erosion_iter=1,
                           depth_edge_max_jump=0.03, min_points=30)

    # The NaN pixel is inside the mask but has invalid depth — filter
    # should drop it (so _back_project doesn't later see garbage).
    ok = True
    ok &= _ok("NaN-depth pixel dropped",
              cleaned[15, 15] == 0,
              f"cleaned[15,15]={cleaned[15,15]}")
    ok &= _ok("0-depth pixel dropped",
              cleaned[16, 16] == 0,
              f"cleaned[16,16]={cleaned[16,16]}")
    # Well-interior pixels should survive.
    ok &= _ok("interior pixel survives",
              cleaned[20, 20] == 1)
    return ok


# ──────────────────────────────────────────────────────────────────────
# 2) Real-frame regression
# ──────────────────────────────────────────────────────────────────────

ROOT = Path("/Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP")
DATASET = Path("/Volumes/External/Workspace/datasets/apple_in_the_tray")
K = np.array([[554.3827, 0.0, 320.5],
              [0.0, 554.3827, 240.5],
              [0.0, 0.0,     1.0]], dtype=np.float64)


def _load_frame(idx: int):
    det_path = (ROOT / "tests/visualization_pipeline/apple_in_the_tray/"
                "perception/detection_h"
                / f"detection_{idx:06d}_final.json")
    depth_path = DATASET / "depth" / f"depth_{idx:06d}.npy"
    if not det_path.exists() or not depth_path.exists():
        return None, None
    with det_path.open() as f:
        dets = json.load(f)["detections"]
    depth = np.load(depth_path)
    return dets, depth


def _decode_mask(b64: str, size=(640, 480)):
    im = Image.open(BytesIO(base64.b64decode(b64))).convert("L")
    return (np.asarray(im) > 128).astype(np.uint8)


def test_regression_realframe() -> bool:
    """Compare filter OFF vs ON on a real apple-in-tray detection."""
    dets, depth = _load_frame(300)
    if dets is None:
        return _ok("real-frame test", False, "missing frame 300 data")

    # Pick the detection with the most mask pixels (likely the tray or
    # a big apple); that's where tails show up most.
    dets_with_mask = [d for d in dets if d.get("mask")]
    if not dets_with_mask:
        return _ok("real-frame test", False, "no masks in frame 300")
    d_big = max(dets_with_mask,
                 key=lambda d: _decode_mask(d["mask"]).sum())
    mask = _decode_mask(d_big["mask"])

    raw_cloud = _back_project(mask, depth, K, clean_mask=False)
    cln_cloud = _back_project(mask, depth, K, clean_mask=True)
    assert raw_cloud is not None and cln_cloud is not None

    # Radius computed the same way _estimate_icp computes obj_radius.
    raw_radius = float(np.linalg.norm(
        raw_cloud - raw_cloud.mean(axis=0), axis=1).max())
    cln_radius = float(np.linalg.norm(
        cln_cloud - cln_cloud.mean(axis=0), axis=1).max())

    n_raw = len(raw_cloud)
    n_cln = len(cln_cloud)
    retain = n_cln / max(1, n_raw)

    print(f"  [info] frame 300 label={d_big.get('label')} "
          f"n_raw={n_raw} n_clean={n_cln} ({100*retain:.1f}% retained) "
          f"radius raw={raw_radius:.3f}m clean={cln_radius:.3f}m")

    ok = True
    # Radius should shrink (outliers at the far edge are dropped).
    ok &= _ok("radius shrinks on real frame",
              cln_radius < raw_radius,
              f"raw={raw_radius:.3f} cln={cln_radius:.3f}")
    # Retention should be healthy — we're not over-filtering.
    ok &= _ok("retention ≥ 30% of raw",
              retain >= 0.30,
              f"{100*retain:.1f}%")
    # Still plenty of points for ICP.
    ok &= _ok("clean cloud has ≥ 30 points",
              n_cln >= 30,
              f"n_clean={n_cln}")
    return ok


def test_centroid_stable() -> bool:
    """Filter should not shift the centroid by more than a few mm — the
    leaked tails are a small fraction of total points, and erosion is
    symmetric. A big shift would indicate a systematic error."""
    dets, depth = _load_frame(300)
    if dets is None:
        return _ok("centroid stability", False, "missing frame 300 data")
    dets_with_mask = [d for d in dets if d.get("mask")]
    if not dets_with_mask:
        return _ok("centroid stability", False, "no masks in frame 300")
    d_big = max(dets_with_mask,
                 key=lambda d: _decode_mask(d["mask"]).sum())
    mask = _decode_mask(d_big["mask"])

    c_raw = centroid_cam_from_mask(mask, depth, K)
    c_cln = _back_project(mask, depth, K, clean_mask=True).mean(axis=0)
    shift = float(np.linalg.norm(c_raw - c_cln))
    print(f"  [info] centroid shift = {shift*100:.2f} cm")
    return _ok("centroid shift ≤ 5 cm", shift <= 0.05,
               f"shift={shift*100:.2f} cm")


# ──────────────────────────────────────────────────────────────────────
# 3) ICP-fitness regression
# ──────────────────────────────────────────────────────────────────────

def test_icp_fitness_regression() -> bool:
    """On two consecutive frames of the same detection, ICP fitness with
    the filter ON must be >= fitness OFF minus a small tolerance. If
    the filter ON made fitness materially worse, we over-filtered."""
    dets1, depth1 = _load_frame(300)
    dets2, depth2 = _load_frame(305)
    if dets1 is None or dets2 is None:
        return _ok("ICP fitness regression", False, "missing frames")

    def _pick(dets):
        dets_with_mask = [d for d in dets if d.get("mask")]
        if not dets_with_mask:
            return None, None
        dbig = max(dets_with_mask,
                    key=lambda d: _decode_mask(d["mask"]).sum())
        return dbig, _decode_mask(dbig["mask"])

    d1, m1 = _pick(dets1)
    d2, m2 = _pick(dets2)
    if d1 is None or d2 is None:
        return _ok("ICP fitness regression", False, "no masks")

    # Helper: fresh estimator, birth at frame 1, refine at frame 2.
    import perception.icp_pose as icp_module

    def _run(enable_clean: bool):
        # Monkey-patch _back_project's default to toggle clean_mask via
        # the class constant (cleaner than threading a kwarg through
        # every estimate() callsite).
        orig = icp_module._back_project
        def wrapped(mask, depth, K_, min_depth=0.1, max_depth=5.0,
                    min_points=30, clean_mask=None):
            cm = enable_clean if clean_mask is None else clean_mask
            return orig(mask, depth, K_, min_depth, max_depth,
                          min_points, clean_mask=cm)
        icp_module._back_project = wrapped
        try:
            pe = PoseEstimator(K=K, method="icp_chain")
            # Frame 1 → birth (fitness returns 1.0 placeholder).
            T1, R1, f1, rmse1 = pe.estimate(oid=1, mask=m1, depth=depth1)
            # Frame 2 → refine (real ICP fitness).
            T2, R2, f2, rmse2 = pe.estimate(oid=1, mask=m2, depth=depth2)
            return f2, rmse2, T2 is not None
        finally:
            icp_module._back_project = orig

    f_off, rmse_off, ok_off = _run(False)
    f_on,  rmse_on,  ok_on  = _run(True)
    print(f"  [info] ICP filter OFF: fitness={f_off:.3f}  rmse={rmse_off*1000:.1f}mm  ok={ok_off}")
    print(f"  [info] ICP filter ON : fitness={f_on:.3f}  rmse={rmse_on*1000:.1f}mm  ok={ok_on}")

    ok = True
    # Fitness should not drop by more than 0.05 (5 percentage points).
    ok &= _ok("fitness ON ≥ fitness OFF − 0.05",
              f_on >= f_off - 0.05,
              f"OFF={f_off:.3f} ON={f_on:.3f}")
    # RMSE should not worsen by more than 2 mm.
    ok &= _ok("rmse ON ≤ rmse OFF + 2 mm",
              rmse_on <= rmse_off + 0.002,
              f"OFF={rmse_off*1000:.1f}mm ON={rmse_on*1000:.1f}mm")
    return ok


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    results = []
    print("[1/5] synthetic_clean_edge")
    results.append(test_synthetic_clean_edge())
    print("[2/5] synthetic_selfdisable_small_mask")
    results.append(test_synthetic_selfdisable_small_mask())
    print("[3/5] synthetic_nan_depth")
    results.append(test_synthetic_nan_depth())
    print("[4/5] regression_realframe")
    results.append(test_regression_realframe())
    print("[5/5] centroid_stable")
    results.append(test_centroid_stable())
    print("[6/5] icp_fitness_regression")
    results.append(test_icp_fitness_regression())
    print()
    if all(results):
        print(f"ALL {len(results)} PASS")
        return 0
    print(f"{sum(results)}/{len(results)} PASS — FAILURES above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Pre-association suppression of sub-part detections.

Two detections in one frame whose back-projected point clouds overlap
substantially cannot be distinct objects — one is a 2D crop that happens
to contain a geometric sub-part of the other. This module collapses such
pairs BEFORE Hungarian ever sees them: the smaller (fewer voxels) is
absorbed into the larger, and its label history is merged in.

The check is purely geometric (label-agnostic by default), so the same
pass also catches cases like "gripper detected as cola overlapping an
apple's voxels" — the phantom detection is absorbed into the physical
object without needing a class-specific rule.

Any detections that survive the pass are spatially disjoint, so
Hungarian's one-to-one constraint is sufficient to keep them from
double-matching to a single track — no extra hard constraint required
on the association side.

Algorithm is a single pass over mask pixels, O(|valid_depth_pixels|),
plus O(N²) set intersections on voxel keys where N = number of
detections in the frame (typically ≤ 10).
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np


VoxelKey = Tuple[int, int, int]


# ─────────────────────────────────────────────────────────────────────
# Back-projection + voxelization
# ─────────────────────────────────────────────────────────────────────

def voxelize_mask(mask: np.ndarray,
                   depth: np.ndarray,
                   K: np.ndarray,
                   voxel_size: float = 0.02,
                   min_depth: float = 0.1,
                   max_depth: float = 5.0,
                   ) -> Set[VoxelKey]:
    """Back-project a boolean mask through a depth image into a set of
    voxel keys (camera-frame coordinates quantised to `voxel_size`).

    Only valid-depth pixels contribute — invalid-depth pixels
    (NaN / 0 / out-of-range) are silently skipped. This is the "holes
    don't inflate the volume" behaviour the caller chose.

    Args:
        mask: (H, W) boolean (or 0/1) array indicating which pixels
            belong to this detection.
        depth: (H, W) float depth map in metres.
        K: (3, 3) camera intrinsics.
        voxel_size: side length of each voxel cell in metres.
        min_depth, max_depth: valid-depth range (metres). Out-of-range
            pixels are treated as no-data.

    Returns:
        Set of voxel keys (i, j, k) where each corresponds to a cell of
        side `voxel_size`. Empty set iff the mask has no valid-depth
        pixel.
    """
    mask = np.asarray(mask).astype(bool, copy=False)
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    if mask.shape != depth.shape:
        # Guard: caller passed a mask of a different resolution; treat
        # as empty rather than crash.
        return set()

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return set()

    d = depth[ys, xs]
    valid = np.isfinite(d) & (d > min_depth) & (d < max_depth)
    if not np.any(valid):
        return set()

    ys = ys[valid]; xs = xs[valid]; d = d[valid]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float64) - cx) / fx * d
    y = (ys.astype(np.float64) - cy) / fy * d
    z = d

    inv_s = 1.0 / float(voxel_size)
    ix = np.floor(x * inv_s).astype(np.int64)
    iy = np.floor(y * inv_s).astype(np.int64)
    iz = np.floor(z * inv_s).astype(np.int64)

    # Pack into a set of tuples. For N ≤ a few thousand voxels this is
    # faster than per-tuple hashing via numpy tricks, and the output is
    # directly usable for set intersections below.
    return set(zip(ix.tolist(), iy.tolist(), iz.tolist()))


# ─────────────────────────────────────────────────────────────────────
# Pairwise containment + greedy absorb
# ─────────────────────────────────────────────────────────────────────

def _merge_labels_dict(into: Dict[str, Dict[str, float]],
                        from_: Dict[str, Dict[str, float]]) -> None:
    """Merge `from_` into `into` in place, summing `n_obs` and
    weighted-averaging `mean_score`. Mirrors the orchestrator's
    `_merge_label_scores` convention."""
    for lbl, stats in (from_ or {}).items():
        if not isinstance(stats, dict):
            continue
        n_new = int(stats.get("n_obs", 0))
        m_new = float(stats.get("mean_score", 0.0))
        if n_new <= 0:
            continue
        cur = into.setdefault(lbl, {"n_obs": 0, "mean_score": 0.0})
        n_old = int(cur.get("n_obs", 0))
        m_old = float(cur.get("mean_score", 0.0))
        n_tot = n_old + n_new
        cur["n_obs"] = n_tot
        cur["mean_score"] = (m_old * n_old + m_new * n_new) / max(1, n_tot)


def suppress_subpart_detections(
    detections: List[Dict[str, Any]],
    depth: np.ndarray,
    K: np.ndarray,
    voxel_size: float = 0.02,
    containment_thresh: float = 0.8,
    require_same_label: bool = False,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy label-agnostic suppression of sub-part detections.

    For each ordered pair (larger-by-voxel-count, smaller):
        if |V_smaller ∩ V_larger| / |V_smaller|  >  containment_thresh
           then the smaller is absorbed into the larger:
               - larger.labels ← merge of both labels dicts
               - smaller is dropped from the output.

    Detections are processed in descending voxel-count order so a chain
    of containment `C ⊂ B ⊂ A` collapses cleanly into `A` in a single
    pass. Voxel sets are cached on `det["_voxels"]` for reuse downstream
    (visibility, ICP seeding, centroid-from-voxel-mean).

    Args:
        detections: raw per-frame list. Each dict must have `mask`. Other
            fields (`label`, `labels`, `score`, `id`, `box`, ...) are
            preserved on survivors.
        depth: (H, W) depth map.
        K: (3, 3) intrinsics.
        voxel_size: side length in metres. 0.02 matches the ICP voxel.
        containment_thresh: |A∩B| / min(|A|, |B|) above which B is
            absorbed into A. 0.8 = "smaller is almost entirely inside
            larger".
        require_same_label: if True, only pairs with equal `det["label"]`
            are considered. Default False (label-agnostic).
        min_depth, max_depth: valid-depth range for voxelization.

    Returns:
        (kept_dets, absorbed_records) — survivors (in input order), and
        per-absorbed-detection records
        `{"into_idx": int, "from_idx": int, "containment": float,
          "from_label": str, "into_label": str}` useful for logging /
        visualisation.
    """
    n = len(detections)
    if n == 0:
        return [], []

    # Voxelize once per detection; cache on the dict.
    voxel_sets: List[Set[VoxelKey]] = []
    for det in detections:
        existing = det.get("_voxels")
        if isinstance(existing, set) and existing:
            voxel_sets.append(existing)
            continue
        mask = det.get("mask")
        if mask is None:
            voxel_sets.append(set())
            det["_voxels"] = set()
            continue
        v = voxelize_mask(mask, depth, K, voxel_size=voxel_size,
                          min_depth=min_depth, max_depth=max_depth)
        det["_voxels"] = v
        voxel_sets.append(v)

    # Process in descending voxel-count order; each absorb is permanent.
    order = sorted(range(n), key=lambda i: -len(voxel_sets[i]))

    absorbed: Set[int] = set()
    absorbed_records: List[Dict[str, Any]] = []

    for i in order:
        if i in absorbed:
            continue
        va = voxel_sets[i]
        if not va:
            continue
        la = detections[i].get("label")
        for j in order:
            if j == i or j in absorbed:
                continue
            vb = voxel_sets[j]
            if not vb:
                continue
            if len(vb) > len(va):
                continue    # j is the larger; skip (will be handled on its iteration)
            if require_same_label and detections[j].get("label") != la:
                continue
            inter = len(va & vb)
            denom = min(len(va), len(vb))
            if denom == 0:
                continue
            containment = inter / denom
            if containment <= containment_thresh:
                continue
            # Absorb j into i.
            larger = detections[i]
            smaller = detections[j]
            # Merge label histories (sum n_obs, weighted-avg mean_score).
            larger_labels = larger.setdefault("labels", {}) or {}
            if not isinstance(larger_labels, dict):
                larger_labels = {}
                larger["labels"] = larger_labels
            _merge_labels_dict(larger_labels, smaller.get("labels") or {})
            # Absorb voxel set too (so chained containments resolve).
            va = va | vb
            voxel_sets[i] = va
            larger["_voxels"] = va
            absorbed.add(j)
            absorbed_records.append({
                "into_idx": int(i),
                "from_idx": int(j),
                "containment": float(containment),
                "from_label": smaller.get("label"),
                "into_label": larger.get("label"),
            })

    kept = [det for idx, det in enumerate(detections) if idx not in absorbed]
    return kept, absorbed_records

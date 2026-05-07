#!/usr/bin/env python3
"""
Scan the per-frame EKF state JSONs and identify spurious birth events
(those that create a new track when an existing same-label track was
nearby in 3D world frame).

Inputs:  tests/visualization_pipeline/{traj}/ekf_state/frame_NNNNNN.json
Outputs: stdout summary of spurious births sorted by distance to the
         closest pre-existing same-label track, plus per-frame totals.

Run:
    conda run -n ocmp_test python tests/diagnose_ekf_births.py \
        [--trajectory apple_in_the_tray] [--top 30] \
        [--dist-threshold 0.20]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)


def _load_frame(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _track_xyz(tr: Dict[str, Any]) -> np.ndarray:
    """Return the track centroid in WORLD frame.

    The state dump now stores `xyz` in BASE frame and (optionally) a
    derived `xyz_w` in WORLD frame. Detections are dumped with `xyz_w`
    (camera->world via T_wb), so spurious-birth detection has to compare
    the two in the same frame: prefer world if available, else fall back
    to base (which is what the old, world-frame storage path produced
    under the same key).
    """
    if "xyz_w" in tr:
        return np.asarray(tr["xyz_w"], dtype=np.float64)
    return np.asarray(tr["xyz"], dtype=np.float64)


def _det_xyz_w(det: Dict[str, Any]) -> Optional[np.ndarray]:
    if not det.get("_icp_ok"):
        return None
    return np.asarray(det.get("xyz_w", [None]*3), dtype=np.float64)


def _nearest_same_label(tracks: Dict[str, Dict[str, Any]],
                         label: str,
                         xyz: np.ndarray
                         ) -> Optional[Tuple[int, float, Dict[str, Any]]]:
    """Return (oid, distance_m, track_record) for the nearest same-label
    track, or None if none exist."""
    best: Optional[Tuple[int, float, Dict[str, Any]]] = None
    for oid_s, tr in tracks.items():
        if tr.get("label") != label:
            continue
        d = float(np.linalg.norm(_track_xyz(tr) - xyz))
        if best is None or d < best[1]:
            best = (int(oid_s), d, tr)
    return best


def _est_d2(tr: Dict[str, Any], det: Dict[str, Any]) -> Optional[float]:
    """Approximate d^2 = nu^T (P + R)^{-1} nu using only the translation
    block (object-orientation rarely flips between adjacent frames here)."""
    P = np.asarray(tr["cov"], dtype=np.float64)
    R = np.asarray(det["R_icp"], dtype=np.float64)
    if P.shape != (6, 6) or R.shape != (6, 6):
        return None
    P_t = P[:3, :3]; R_t = R[:3, :3]
    S_t = P_t + R_t
    nu_t = (np.asarray(det["xyz_w"], dtype=np.float64)
            - _track_xyz(tr))
    try:
        d2 = float(nu_t @ np.linalg.solve(S_t, nu_t))
    except np.linalg.LinAlgError:
        return None
    return d2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_in_the_tray")
    ap.add_argument("--state-subdir", default="ekf_state")
    ap.add_argument("--top", type=int, default=30,
                    help="N most-suspicious spurious births to print")
    ap.add_argument("--dist-threshold", type=float, default=0.20,
                    help="meters; births within this distance of a "
                         "same-label existing track are flagged spurious")
    ap.add_argument("--frame-min", type=int, default=0)
    ap.add_argument("--frame-max", type=int, default=10**9)
    args = ap.parse_args()

    state_dir = os.path.join(
        SCENEREP_ROOT, "tests", "visualization_pipeline",
        args.trajectory, args.state_subdir)
    paths = sorted(glob.glob(os.path.join(state_dir, "frame_*.json")))
    if not paths:
        raise SystemExit(f"no frame_*.json under {state_dir}")
    print(f"[scan] {len(paths)} frames under {state_dir}")

    spurious: List[Dict[str, Any]] = []
    legit_first_birth: List[Dict[str, Any]] = []
    no_pre_existing: List[Dict[str, Any]] = []
    total_births = 0
    per_frame_counts: List[Tuple[int, int, int]] = []  # (frame, births, tracks_after)
    cum_active = 0
    label_oid_counts: Dict[str, set] = {}

    for p in paths:
        fr = _load_frame(p)
        idx = int(fr["frame"])
        if idx < args.frame_min or idx > args.frame_max:
            continue
        tracks_post_predict = fr.get("tracks_post_predict", {})
        tracks_post_update = fr.get("tracks_post_update", {})
        births = fr.get("births", [])
        det_records = {d["global_idx"]: d for d in fr.get("detections", [])}

        per_frame_counts.append((idx, len(births), len(tracks_post_update)))
        total_births += len(births)

        # Track every (label, oid) seen so we can count unique-id sprawl per
        # label even after pruning.
        for oid_s, tr in tracks_post_update.items():
            label_oid_counts.setdefault(tr.get("label", "?"),
                                          set()).add(int(oid_s))

        for b in births:
            det = det_records.get(int(b["det_idx"]))
            if det is None or not det.get("_icp_ok"):
                continue
            label = det.get("label")
            xyz = np.asarray(det["xyz_w"], dtype=np.float64)
            score = float(det.get("score", 0.0))

            near = _nearest_same_label(tracks_post_predict, label, xyz)
            entry = {
                "frame": idx,
                "new_oid": int(b["new_oid"]),
                "det_idx": int(b["det_idx"]),
                "label": label,
                "score": score,
                "xyz": xyz.tolist(),
                "r_new": float(b["r_new"]),
            }
            if near is None:
                no_pre_existing.append(entry)
                continue
            oid_near, dist, tr_near = near
            entry["nearest_oid"] = oid_near
            entry["nearest_dist_m"] = float(dist)
            entry["nearest_xyz"] = _track_xyz(tr_near).tolist()
            entry["nearest_r"] = float(tr_near["r"])
            entry["nearest_tr_cov"] = float(tr_near["tr_cov"])
            entry["nearest_frames_since_obs"] = int(tr_near["frames_since_obs"])
            d2 = _est_d2(tr_near, det)
            if d2 is not None:
                entry["d2_to_nearest"] = float(d2)
            if dist < args.dist_threshold:
                spurious.append(entry)
            else:
                legit_first_birth.append(entry)

    # ─────────────── Report ────────────────────────────────────────────
    n_frames = len(per_frame_counts)
    print(f"\n=== Birth summary across {n_frames} frames ===")
    print(f"  total births                      : {total_births}")
    print(f"  no pre-existing same-label track  : {len(no_pre_existing)}")
    print(f"  with same-label track > {args.dist_threshold:.2f}m away : "
          f"{len(legit_first_birth)}")
    print(f"  with same-label track <= {args.dist_threshold:.2f}m  (SPURIOUS) : "
          f"{len(spurious)}  "
          f"({100.0*len(spurious)/max(1,total_births):.1f}% of all births)")

    print("\n=== Unique track ids per label (post-update set) ===")
    for label, oids in sorted(label_oid_counts.items(),
                               key=lambda kv: -len(kv[1])):
        print(f"  {label:>10}: {len(oids)} distinct ids "
              f"({sorted(oids)[:12]}{'...' if len(oids)>12 else ''})")

    print(f"\n=== Top {args.top} spurious births "
          f"(birth despite a same-label track within "
          f"{args.dist_threshold:.2f}m) ===")
    print("  frame newoid label  score   d_near  d2est  nearest_oid r_near "
          "trP_near fr_since_obs det_xyz                  near_xyz")
    spurious_sorted = sorted(spurious, key=lambda e: e["nearest_dist_m"])
    for e in spurious_sorted[: args.top]:
        d2_str = f"{e.get('d2_to_nearest', float('nan')):>6.2f}"
        print(
            f"  {e['frame']:>5} {e['new_oid']:>5} {e['label']:<7} "
            f"{e['score']:>5.2f}  {e['nearest_dist_m']:>5.3f}  {d2_str}  "
            f"{e['nearest_oid']:>5}  {e['nearest_r']:>5.3f}  "
            f"{e['nearest_tr_cov']:>7.2e}  {e['nearest_frames_since_obs']:>3d}    "
            f"({e['xyz'][0]:>5.2f},{e['xyz'][1]:>5.2f},{e['xyz'][2]:>5.2f})  "
            f"({e['nearest_xyz'][0]:>5.2f},{e['nearest_xyz'][1]:>5.2f},"
            f"{e['nearest_xyz'][2]:>5.2f})"
        )

    if not spurious:
        print("  (none)")

    # Frames with the most births
    print("\n=== Top 10 frames by birth count ===")
    by_births = sorted(per_frame_counts, key=lambda t: -t[1])[:10]
    for frame, n_b, n_tr in by_births:
        if n_b == 0:
            continue
        print(f"  frame {frame:>4d}  births={n_b}  tracks_after={n_tr}")


if __name__ == "__main__":
    main()

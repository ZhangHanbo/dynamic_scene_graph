"""Parity test: EkfTracker public API vs visualize_ekf_tracking.main().

For each frame of `apple_drop`, drives the new EkfTracker.step() and
compares its per-track world-frame mean / covariance / Bernoulli r
against the JSON state dumps already produced by main()
(tests/visualization_pipeline/apple_drop/ekf_state/).

Both runs use the SAME LLM relation cache so the relation graph
(which gates self-merge and held-set expansion) is identical.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ekf_tracker.api import EkfTracker  # noqa: E402

DATA = os.path.join(ROOT, "datasets", "apple_drop")
VIZ = os.path.join(ROOT, "tests", "visualization_pipeline", "apple_drop")
DET_DIR = os.path.join(VIZ, "perception", "detection_h")
if not os.path.isdir(DET_DIR):
    DET_DIR = os.path.join(DATA, "detection_h")
STATE_DIR = os.path.join(VIZ, "ekf_state")
RELATION_CACHE = os.path.join(VIZ, "relation_cache")

K = np.array([[554.3827, 0.0, 320.5],
              [0.0, 554.3827, 240.5],
              [0.0, 0.0, 1.0]], dtype=np.float64)


def _load_amcl(p):
    out = []
    for line in open(p):
        a = line.strip().split()
        if len(a) != 8:
            continue
        _, tx, ty, tz, qx, qy, qz, qw = map(float, a)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        out.append(T)
    return out


def _load_idx_pose(p):
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        a = line.strip().split()
        if len(a) != 8:
            continue
        idx = int(a[0])
        tx, ty, tz, qx, qy, qz, qw = map(float, a[1:])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        out[idx] = T
    return out


def _load_widths(p):
    out = {}
    if not os.path.exists(p):
        return out
    for k, v in json.load(open(p)).items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        l = v.get("l_gripper_finger_joint")
        r = v.get("r_gripper_finger_joint")
        if l is not None and r is not None:
            out[idx] = float(l) + float(r)
    return out


def _load_joints(p):
    if not os.path.exists(p):
        return {}
    return {int(k): v for k, v in json.load(open(p)).items()}


def _load_dets(path):
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    out = []
    for det in data.get("detections", []):
        mb = det.get("mask", "")
        if not mb:
            continue
        try:
            mb_bytes = base64.b64decode(mb)
            m = np.array(Image.open(BytesIO(mb_bytes)).convert("L"))
            m = (m > 128).astype(np.uint8)
        except Exception:
            continue
        out.append({
            "id": int(det.get("object_id")),
            "label": det.get("label", "unknown"),
            "labels": det.get("labels", {}),
            "mask": m,
            "score": float(det.get("score", 0.0)),
            "mean_score": float(det.get("mean_score", 0.0)),
            "n_obs": int(det.get("n_obs", 0)),
            "box": det.get("box"),
        })
    return out


def main():
    slam = _load_amcl(os.path.join(DATA, "pose_txt", "amcl_pose.txt"))
    T_bc = _load_idx_pose(os.path.join(DATA, "pose_txt", "T_bc.txt"))
    T_bg = _load_idx_pose(os.path.join(DATA, "pose_txt", "ee_pose.txt"))
    widths = _load_widths(os.path.join(DATA, "pose_txt", "joints_pose.json"))
    joints = _load_joints(os.path.join(DATA, "pose_txt", "joints_pose.json"))

    n_frames = len(slam)
    print(f"frames: {n_frames}, T_bc map: {len(T_bc)}, T_bg map: {len(T_bg)}, "
          f"widths: {len(widths)}, joints: {len(joints)}")

    tr = EkfTracker(
        K=K,
        T_bc=None,                         # main() defaults to None too
        relation_backend="llm",
        relation_cache_dir=RELATION_CACHE,
    )

    # Per-frame parity stats.
    n_compared = 0
    n_oid_match = 0
    n_oid_mismatch = 0
    pose_max_err_world = 0.0
    cov_max_err_world = 0.0
    r_max_err = 0.0
    mismatched_frames = []
    per_frame_pose_err = []  # (idx, max |Δ| over oids)

    DET_TPL = os.path.join(DET_DIR, "detection_{:06d}_final.json")
    RGB_TPL = os.path.join(DATA, "rgb", "rgb_{:06d}.png")
    DPT_TPL = os.path.join(DATA, "depth", "depth_{:06d}.npy")
    STATE_TPL = os.path.join(STATE_DIR, "frame_{:06d}.json")

    for idx in range(n_frames):
        rgb_p = RGB_TPL.format(idx)
        dpt_p = DPT_TPL.format(idx)
        if not (os.path.exists(rgb_p) and os.path.exists(dpt_p)):
            continue
        rgb = np.array(Image.open(rgb_p).convert("RGB"))
        depth = np.load(dpt_p).astype(np.float32)
        dets = _load_dets(DET_TPL.format(idx))

        sv = tr.step(
            detections=dets,
            rgb=rgb,
            depth=depth,
            slam_pose=slam[idx],
            T_bc=T_bc.get(idx),
            T_bg=T_bg.get(idx),
            gripper_width=widths.get(idx),
            joints=joints.get(idx),
        )

        # Compare against the script's JSON dump.
        gt_path = STATE_TPL.format(idx)
        if not os.path.exists(gt_path):
            continue
        gt = json.load(open(gt_path))
        gt_tracks = gt.get("tracks_post_update", {}) or {}
        gt_oids = sorted(int(o) for o in gt_tracks.keys())
        api_oids = sorted(sv.objects.keys())

        if gt_oids != api_oids:
            n_oid_mismatch += 1
            mismatched_frames.append((idx, "oid_set",
                                       set(gt_oids) ^ set(api_oids)))
            continue
        n_oid_match += 1

        frame_pose_err = 0.0
        worst_oid = None
        for oid in gt_oids:
            gt_tr = gt_tracks[str(oid)]
            api_obj = sv.objects[oid]

            # World-frame mean (T_world in the dump).
            T_gt_w = np.asarray(gt_tr["T_world"], dtype=np.float64)
            T_api_w = api_obj.pose
            err = float(np.max(np.abs(T_gt_w - T_api_w)))
            pose_max_err_world = max(pose_max_err_world, err)
            if err > frame_pose_err:
                frame_pose_err = err
                worst_oid = (oid, gt_tr["label"])

            # World-frame covariance (cov_world in the dump; can be None).
            cov_gt_w = gt_tr.get("cov_world")
            if cov_gt_w is not None:
                cov_gt_w = np.asarray(cov_gt_w, dtype=np.float64)
                cov_err = float(np.max(np.abs(cov_gt_w - api_obj.cov)))
                cov_max_err_world = max(cov_max_err_world, cov_err)

            # Bernoulli r.
            r_err = abs(float(gt_tr["r"]) - float(api_obj.r))
            r_max_err = max(r_max_err, r_err)

        per_frame_pose_err.append((idx, frame_pose_err, worst_oid))
        n_compared += 1

    print()
    print("=" * 60)
    print(f"frames compared:        {n_compared}")
    print(f"oid-set match frames:   {n_oid_match}")
    print(f"oid-set mismatch frames:{n_oid_mismatch}")
    print(f"pose (world) max |Δ|:   {pose_max_err_world:.3e}")
    print(f"cov  (world) max |Δ|:   {cov_max_err_world:.3e}")
    print(f"r           max |Δ|:   {r_max_err:.3e}")
    if mismatched_frames:
        print()
        print("first oid-set mismatches:")
        for fr, kind, extra in mismatched_frames[:5]:
            print(f"  fr {fr}: {kind} symdiff={sorted(extra)}")

    # Print first frame where pose error exceeds 1e-6, and first frame
    # where it exceeds 0.01 m — signals start of divergence.
    first_micro = next((idx for idx, e, _ in per_frame_pose_err if e > 1e-6), None)
    first_visible = next((idx for idx, e, _ in per_frame_pose_err if e > 0.01), None)
    print()
    print(f"first frame with pose Δ > 1e-6:  {first_micro}")
    print(f"first frame with pose Δ > 0.01:  {first_visible}")
    print()
    # Group post-274 errors by which oid is the worst offender each frame.
    after_274 = [(i, e, w) for i, e, w in per_frame_pose_err if i >= 274]
    if after_274:
        offenders = {}
        for i, e, w in after_274:
            offenders.setdefault(w, []).append((i, e))
        print("post-274 worst-offender oids and frame counts:")
        for w, items in sorted(offenders.items(),
                                  key=lambda kv: -len(kv[1])):
            errs = [e for _, e in items]
            print(f"  oid={w}: {len(items)} frames; max Δ {max(errs):.3e}, "
                  f"min {min(errs):.3e}")


if __name__ == "__main__":
    main()

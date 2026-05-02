#!/usr/bin/env python3
"""
Diagnostic: measure what the observation update does when the predict is
invalid. Runs the degenerate-mode orchestrator on apple_bowl_2 TWICE:

  (1) rigid-attachment DISABLED (no T_bg)  -- predict is invalid for
      the held object: its world-frame mean is frozen; covariance only
      inflates by Q_HELD_WORLD = 1e-4 per frame.
  (2) rigid-attachment ENABLED (T_bg = T_ec)  -- predict is valid: the
      held object's mean is rotated/translated by the gripper's delta
      per step.

For each frame of each run, we log per held track:

    translation  of  mu (world frame)
    translation  of  observation T_co projected to world via T_cw
    Kalman-gain trace tr(K)
    innovation norm ||nu|| (se(3) tangent)
    covariance trace tr(P)

The two traces are written to traces.csv and plotted in traces.png. This
shows quantitatively how much the observation update alone can
compensate for a broken predict.

Output:
  tests/visualization_pipeline/apple_bowl_2/observation_trace/
    traces.csv
    traces.png

Run:
  conda run -n ocmp_test python tests/trace_observation_updates.py \
      [--trajectory apple_bowl_2] [--step S]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCENEREP_ROOT not in sys.path:
    sys.path.insert(0, SCENEREP_ROOT)

from tests.test_orchestrator_integration import (
    _build_T_co_from_mask, _gripper_state_from_distance, _load_pose_txt, K,
)
from tests.visualize_sam2_observations import _load_detections
from tests.visualize_pipeline import resolve_held_by_proximity
from pose_update.orchestrator import (
    TwoTierOrchestrator, TriggerConfig, BernoulliConfig,
)
from pose_update.state.slam_interface import PassThroughSlam
from pose_update.state.ekf_se3 import se3_log

DATA_BASE = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects",
)


def _gripper_phase(l: np.ndarray, r: np.ndarray,
                    last_d: Optional[float],
                    last_phase: str) -> Tuple[str, float]:
    d = float(np.linalg.norm(l[:3, 3] - r[:3, 3]))
    raw = _gripper_state_from_distance(d, last_d)
    if raw == "grasping":
        phase = "grasping"
    elif raw == "releasing":
        phase = "releasing"
    elif last_phase == "grasping":
        phase = "holding"
    elif last_phase in ("holding", "releasing") and raw == "idle":
        phase = "idle" if last_phase == "releasing" else "holding"
    elif last_phase == "holding" and raw == "idle":
        phase = "holding"
    else:
        phase = "idle"
    return phase, d


def _build_detections(raw: List[Dict[str, Any]],
                       depth: np.ndarray) -> List[Dict[str, Any]]:
    out = []
    for d in raw:
        oid = d.get("object_id")
        if oid is None:
            continue
        T_co = _build_T_co_from_mask(d["mask"], depth, K)
        if T_co is None:
            continue
        out.append({
            "id": int(oid),
            "label": d.get("label", "?"),
            "mask": d["mask"],
            "score": float(d.get("score", 0.0)),
            "T_co": T_co,
            "R_icp": np.diag([1e-4] * 3 + [1e-3] * 3),
            "fitness": float(max(0.3, d.get("score", 0.5))),
            "rmse": 0.005,
            "box": d.get("box"),
        })
    return out


def run_once(trajectory: str, indices: List[int], pass_T_bg: bool,
              rng_seed: int = 42) -> List[Dict[str, Any]]:
    """Drive the orchestrator through the frames; return a list of per-frame
    records with observation/innovation/K/cov stats for every tracked
    object that was matched by a detection on that frame.

    The exact same gripper-phase state machine and held_obj heuristic are
    used as in visualize_degenerate_vs_sam2.py, so the two runs only
    differ in whether T_bg is passed.
    """
    data_root = os.path.join(DATA_BASE, trajectory)
    cam_poses = _load_pose_txt(
        os.path.join(data_root, "pose_txt", "camera_pose.txt"))
    ee_poses = _load_pose_txt(
        os.path.join(data_root, "pose_txt", "ee_pose.txt"))
    l_finger = _load_pose_txt(
        os.path.join(data_root, "pose_txt", "l_gripper_pose.txt"))
    r_finger = _load_pose_txt(
        os.path.join(data_root, "pose_txt", "r_gripper_pose.txt"))

    slam_poses = [cam_poses[i] for i in indices]
    slam = PassThroughSlam(slam_poses, default_cov=np.diag([1e-4] * 6))
    np.random.seed(rng_seed)
    orch = TwoTierOrchestrator(
        slam,
        trigger=TriggerConfig(periodic_every_n_frames=30),
        verbose=False, rng_seed=rng_seed,
        bernoulli=BernoulliConfig.degeneracy(),
    )

    last_finger_d: Optional[float] = None
    last_phase = "idle"
    held_obj: Optional[int] = None
    records: List[Dict[str, Any]] = []

    for local_i, idx in enumerate(indices):
        rgb_path = os.path.join(data_root, "rgb", f"rgb_{idx:06d}.png")
        depth_path = os.path.join(data_root, "depth", f"depth_{idx:06d}.npy")
        det_path = os.path.join(data_root, "detection_h",
                                 f"detection_{idx:06d}_final.json")
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)
        raw_dets = _load_detections(det_path)
        dets = _build_detections(raw_dets, depth)

        phase, finger_d = _gripper_phase(
            l_finger[idx], r_finger[idx], last_finger_d, last_phase)
        last_finger_d = finger_d
        last_phase = phase

        if phase == "grasping" and held_obj is None and dets:
            # Cluster-distance resolver: picks the object whose point cloud
            # has the lowest 5th-percentile distance to the EE -- so the
            # bowl's rim wins over the apple's centroid inside it.
            T_ec = ee_poses[idx]
            ee_cam = T_ec[:3, 3]   # EE position in camera frame
            held_obj = resolve_held_by_proximity(
                dets, depth, ee_cam, cam_K=K)
            if held_obj is None and dets:
                # If every candidate is outside max_dist, fall back to the
                # nearest centroid so the held-obj state machine can
                # still fire (this mirrors the integration test).
                T_cw = cam_poses[idx]
                T_ew = T_cw @ T_ec
                best = min(((d["id"], np.linalg.norm(
                    (T_cw @ d["T_co"])[:3, 3] - T_ew[:3, 3]))
                    for d in dets), key=lambda kv: kv[1])
                held_obj = best[0]
        elif phase == "idle":
            held_obj = None

        # ------ Pre-step bookkeeping: snapshot prior collapsed (mu, P)
        # for every tracked oid before the orchestrator mutates it. This
        # is what the observation UPDATE will see as the prior.
        prior_snapshot: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for oid in orch.object_labels:
            pe = orch.state.collapsed_object(oid)
            if pe is not None:
                prior_snapshot[oid] = (pe.T.copy(), pe.cov.copy())

        # Observation world-frame target for every detected oid (what the
        # update attempts to pull mu towards).
        T_cw = cam_poses[idx]
        obs_world: Dict[int, np.ndarray] = {}
        for d in dets:
            obs_world[d["id"]] = (T_cw @ d["T_co"]).copy()

        T_ec = ee_poses[idx]
        T_bg = T_ec if pass_T_bg else None
        orch.step(rgb, depth, dets,
                   {"phase": phase, "held_obj_id": held_obj},
                   T_ec=T_ec, T_bg=T_bg)

        # ------ Post-step: collapsed posteriors.
        for oid in orch.object_labels:
            post = orch.state.collapsed_object(oid)
            if post is None or oid not in prior_snapshot:
                continue
            mu_prior, P_prior = prior_snapshot[oid]
            mu_post = post.T
            P_post = post.cov
            row = {
                "frame": idx,
                "phase": phase,
                "held_obj": held_obj if held_obj is not None else -1,
                "oid": int(oid),
                "label": orch.object_labels.get(int(oid), "?"),
                "matched": (int(oid) in obs_world),
                "mu_prior_t": mu_prior[:3, 3].copy(),
                "mu_post_t": mu_post[:3, 3].copy(),
                "trP_prior": float(np.trace(P_prior)),
                "trP_post": float(np.trace(P_post)),
            }
            if int(oid) in obs_world:
                T_obs = obs_world[int(oid)]
                row["obs_t"] = T_obs[:3, 3].copy()
                nu = se3_log(np.linalg.inv(mu_prior) @ T_obs)
                row["nu_norm"] = float(np.linalg.norm(nu))
                # Diagonal approximation: K ~ P/(P+R) for a 6x6 diagonal
                # proxy. We report tr(K) as a single scalar.
                R = np.diag([1e-4] * 3 + [1e-3] * 3)
                K_approx = P_prior @ np.linalg.inv(P_prior + R)
                row["trK"] = float(np.trace(K_approx)) / 6.0
            records.append(row)

    return records


def summarise(records: List[Dict[str, Any]],
               oid_of_interest: int) -> Dict[str, np.ndarray]:
    frames: List[int] = []
    phases: List[str] = []
    held: List[int] = []
    obs_t: List[np.ndarray] = []
    mu_post: List[np.ndarray] = []
    trK: List[float] = []
    nu_norm: List[float] = []
    trP_prior: List[float] = []
    trP_post: List[float] = []
    for r in records:
        if r["oid"] != oid_of_interest:
            continue
        if not r["matched"]:
            continue
        frames.append(r["frame"])
        phases.append(r["phase"])
        held.append(int(r["held_obj"]))
        obs_t.append(r["obs_t"])
        mu_post.append(r["mu_post_t"])
        trK.append(r["trK"])
        nu_norm.append(r["nu_norm"])
        trP_prior.append(r["trP_prior"])
        trP_post.append(r["trP_post"])
    return {
        "frames": np.array(frames, dtype=int),
        "phases": np.array(phases),
        "held": np.array(held, dtype=int),
        "obs_t": np.stack(obs_t) if obs_t else np.zeros((0, 3)),
        "mu_post": np.stack(mu_post) if mu_post else np.zeros((0, 3)),
        "trK": np.array(trK),
        "nu_norm": np.array(nu_norm),
        "trP_prior": np.array(trP_prior),
        "trP_post": np.array(trP_post),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_bowl_2")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--oid", type=int, default=3,
                    help="which track to dump (3=bowl, 4=apple)")
    args = ap.parse_args()

    rgb_dir = os.path.join(DATA_BASE, args.trajectory, "rgb")
    rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    indices = [int(f[4:10]) for f in rgb_files][::args.step]

    print(f"[trace] running WITHOUT T_bg (rigid-attachment off) ...")
    off = run_once(args.trajectory, indices, pass_T_bg=False)
    print(f"[trace] running WITH    T_bg (rigid-attachment on)  ...")
    on = run_once(args.trajectory, indices, pass_T_bg=True)

    s_off = summarise(off, args.oid)
    s_on = summarise(on, args.oid)

    out_dir = os.path.join(
        SCENEREP_ROOT, "tests", "visualization_pipeline",
        args.trajectory, "observation_trace")
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, f"oid_{args.oid}_traces.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "phase", "held",
            "mu_off_x", "mu_off_y", "mu_off_z",
            "mu_on_x",  "mu_on_y",  "mu_on_z",
            "obs_x",    "obs_y",    "obs_z",
            "|lag_off|_m", "|lag_on|_m",
            "nu_norm_off", "nu_norm_on",
            "trK_off", "trK_on",
            "trP_off", "trP_on",
        ])
        # Intersect frames where both runs had the oid matched.
        common = sorted(set(s_off["frames"].tolist())
                        & set(s_on["frames"].tolist()))
        idx_off = {f: i for i, f in enumerate(s_off["frames"].tolist())}
        idx_on  = {f: i for i, f in enumerate(s_on["frames"].tolist())}
        for f in common:
            i0 = idx_off[f]
            i1 = idx_on[f]
            mu_off = s_off["mu_post"][i0]
            mu_on  = s_on["mu_post"][i1]
            obs    = s_off["obs_t"][i0]  # obs identical across runs
            lag_off = float(np.linalg.norm(mu_off - obs))
            lag_on  = float(np.linalg.norm(mu_on  - obs))
            w.writerow([
                f, s_off["phases"][i0], int(s_off["held"][i0]),
                *mu_off.tolist(), *mu_on.tolist(), *obs.tolist(),
                lag_off, lag_on,
                float(s_off["nu_norm"][i0]), float(s_on["nu_norm"][i1]),
                float(s_off["trK"][i0]),     float(s_on["trK"][i1]),
                float(s_off["trP_prior"][i0]),
                float(s_on["trP_prior"][i1]),
            ])
    print(f"[trace] wrote {csv_path}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    common = sorted(set(s_off["frames"].tolist())
                    & set(s_on["frames"].tolist()))
    common = np.array(common)
    idx_off = {f: i for i, f in enumerate(s_off["frames"].tolist())}
    idx_on  = {f: i for i, f in enumerate(s_on["frames"].tolist())}
    lag_off = np.array([float(np.linalg.norm(
        s_off["mu_post"][idx_off[f]] - s_off["obs_t"][idx_off[f]]))
        for f in common])
    lag_on = np.array([float(np.linalg.norm(
        s_on["mu_post"][idx_on[f]] - s_on["obs_t"][idx_on[f]]))
        for f in common])
    nu_off = np.array([s_off["nu_norm"][idx_off[f]] for f in common])
    nu_on  = np.array([s_on["nu_norm"][idx_on[f]]   for f in common])
    trK_off = np.array([s_off["trK"][idx_off[f]] for f in common])
    trK_on  = np.array([s_on["trK"][idx_on[f]]   for f in common])
    # Shade holding frames.
    held_mask = np.array([
        int(s_off["held"][idx_off[f]]) == args.oid for f in common])
    for ax in axes:
        ax.grid(True, alpha=0.3)
        for i in range(len(common) - 1):
            if held_mask[i]:
                ax.axvspan(common[i] - 0.5, common[i + 1] + 0.5,
                           alpha=0.08, color="orange")

    axes[0].plot(common, lag_off * 100, "o-", color="C3",
                 label="rigid-attach OFF", markersize=3)
    axes[0].plot(common, lag_on * 100, "o-", color="C2",
                 label="rigid-attach ON",  markersize=3)
    axes[0].set_ylabel("||mu - obs||  [cm]")
    axes[0].set_title(
        f"oid={args.oid}: posterior-vs-observation lag per frame "
        "(orange bands = holding)")
    axes[0].legend(loc="upper left")

    axes[1].semilogy(common, nu_off, "o-", color="C3",
                     label="||nu|| off", markersize=3)
    axes[1].semilogy(common, nu_on, "o-", color="C2",
                     label="||nu|| on", markersize=3)
    axes[1].set_ylabel("||nu|| (se(3) tangent)")
    axes[1].legend(loc="upper left")

    axes[2].plot(common, trK_off, "o-", color="C3",
                 label="tr(K)/6 off", markersize=3)
    axes[2].plot(common, trK_on, "o-", color="C2",
                 label="tr(K)/6 on", markersize=3)
    axes[2].set_ylabel("mean Kalman gain")
    axes[2].set_xlabel("frame")
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    png_path = os.path.join(out_dir, f"oid_{args.oid}_traces.png")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[trace] wrote {png_path}")

    # Print a short ASCII summary.
    held_idx = np.where(held_mask)[0]
    if len(held_idx) > 0:
        print("\n[trace] summary over HELD frames (orange bands):")
        print(f"    n = {len(held_idx)} frames")
        print(f"    mean lag, rigid OFF: {np.mean(lag_off[held_idx])*100:.2f} cm")
        print(f"    mean lag, rigid ON : {np.mean(lag_on[held_idx])*100:.2f} cm")
        print(f"    max  lag, rigid OFF: {np.max(lag_off[held_idx])*100:.2f} cm")
        print(f"    max  lag, rigid ON : {np.max(lag_on[held_idx])*100:.2f} cm")
        print(f"    mean tr(K)/6 OFF  : {np.mean(trK_off[held_idx]):.3f}")
        print(f"    mean tr(K)/6 ON   : {np.mean(trK_on[held_idx]):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase-0 regression baseline for the EKF-cov refactor (plan: keen-imagining-cupcake).

Reads existing per-frame state JSONs under
`tests/visualization_pipeline/<traj>/ekf_state/` and snapshots, per
checkpoint frame and per oid:
    sigma_major, sigma_mid, sigma_minor of cov_world[trans] (cm),
    major-axis xy direction,
    |maj_xy . r2o_xy|, |maj_xy . heading_xy|,
    sigma_predicted_post_refactor (analytical lower bound).

Output: `tests/baseline/cov_baseline_<traj>.csv`. The Phase-2 refactor
must reproduce sigma_predicted_post within tolerance and not exceed
sigma_current. Mechanism-2 contamination shows up as
(sigma_current - sigma_predicted_post).

Run:
    conda run -n ocmp_test python tests/diagnose_cov_baseline.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import List, Tuple

import numpy as np


CHECKPOINTS = [60, 100, 200, 274, 340, 400, 500, 600, 700]
TRAJECTORIES = ["apple_drop", "apple_in_the_tray",
                "apple_to_cabinate_look"]

PASSTHROUGH_SLAM_SIGMA_WB = np.diag([1e-4] * 3 + [1e-4] * 3)
TRANS_VAR_FLOOR = 4e-4   # (2 cm)^2  --  pose_update/icp_pose.py
ROT_VAR_FLOOR = 1e-2     # (0.1 rad)^2

Q_IDLE = 1e-6     # fso < 5
Q_UNSTABLE = 1e-5  # 5 <= fso < 50
Q_STABLE = 1e-8    # fso >= 50


def q_accum_var_per_axis(fso: int) -> float:
    """Diagonal contribution from accumulated Q over `fso` predict steps."""
    n_idle = min(fso, 5)
    n_unstable = max(0, min(fso - 5, 45))
    n_stable = max(0, fso - 50)
    return n_idle * Q_IDLE + n_unstable * Q_UNSTABLE + n_stable * Q_STABLE


def predicted_sigma_xy_post_refactor(fso: int, t_bo_xy_norm: float) -> float:
    """Analytical lower-bound for sigma_major[xy] under the world-frame
    refactor (no mechanism-2 cross-coupling).

        sigma_var = TRANS_VAR_FLOOR + Q_accumulated + lever_arm_from_Sigma_wb
        lever_arm = |t_bo|^2 * Sigma_wb[yaw] (perpendicular to t_bo)
    """
    sigma_var_floor = TRANS_VAR_FLOOR
    sigma_var_q = q_accum_var_per_axis(fso)
    # lever arm from Sigma_wb yaw uncertainty
    sigma_yaw_wb_var = PASSTHROUGH_SLAM_SIGMA_WB[5, 5]
    sigma_var_lever = (t_bo_xy_norm ** 2) * sigma_yaw_wb_var
    return float(np.sqrt(sigma_var_floor + sigma_var_q + sigma_var_lever))


def major_xy_decomposition(cov_world: np.ndarray
                           ) -> Tuple[float, float, float, np.ndarray]:
    """Eigendecompose cov_world[xyz, xyz] and return:
        sigma_major (cm), sigma_mid (cm), sigma_minor (cm),
        major_axis_xy (2,) unit vector (top-down projection of major eigenvec).
    """
    cov_t = 0.5 * (cov_world[:3, :3] + cov_world[:3, :3].T)
    eigvals, eigvecs = np.linalg.eigh(cov_t)
    order = np.argsort(eigvals)[::-1]  # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    sigmas = np.sqrt(np.maximum(eigvals, 0.0)) * 100.0  # cm
    major_xy = eigvecs[:2, 0]
    major_xy_norm = np.linalg.norm(major_xy)
    if major_xy_norm > 1e-9:
        major_xy = major_xy / major_xy_norm
    else:
        major_xy = np.array([1.0, 0.0])
    return float(sigmas[0]), float(sigmas[1]), float(sigmas[2]), major_xy


def topdown_xy_sigma(cov_world: np.ndarray) -> float:
    """Major eigenvalue of the 2x2 top-down xy block (cm). This is what
    the visualizer renders as the ground-plane ellipse."""
    xy = 0.5 * (cov_world[:2, :2] + cov_world[:2, :2].T)
    eigvals = np.linalg.eigvalsh(xy)
    return float(np.sqrt(max(eigvals.max(), 0.0))) * 100.0


def heading_xy(T_wb: np.ndarray) -> np.ndarray:
    h = T_wb[:2, 0]
    n = np.linalg.norm(h)
    return h / n if n > 1e-9 else np.array([1.0, 0.0])


def r2o_xy(T_wb: np.ndarray, T_world_obj: np.ndarray) -> np.ndarray:
    v = T_world_obj[:2, 3] - T_wb[:2, 3]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def t_bo_xy_norm_from_world(T_wb: np.ndarray, T_world_obj: np.ndarray) -> float:
    return float(np.linalg.norm(T_world_obj[:2, 3] - T_wb[:2, 3]))


def process_trajectory(traj: str) -> List[dict]:
    state_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "visualization_pipeline", traj, "ekf_state",
    )
    if not os.path.isdir(state_dir):
        print(f"[skip] {traj}: {state_dir} not found", file=sys.stderr)
        return []

    rows: List[dict] = []
    for fr in CHECKPOINTS:
        path = os.path.join(state_dir, f"frame_{fr:06d}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        T_wb = np.array(d["slam_pose"], dtype=np.float64)
        phase = d.get("gripper_state", {}).get("phase", "?")
        for oid_str, tr in d.get("tracks_post_update", {}).items():
            cov_world = np.array(tr["cov_world"], dtype=np.float64)
            T_world_obj = np.array(tr["T_world"], dtype=np.float64)
            fso = int(tr["frames_since_obs"])

            sM, smid, smin, maj_xy = major_xy_decomposition(cov_world)
            sigma_xy_topdown = topdown_xy_sigma(cov_world)
            hd = heading_xy(T_wb)
            r2o = r2o_xy(T_wb, T_world_obj)
            t_bo_xy = t_bo_xy_norm_from_world(T_wb, T_world_obj)

            sigma_predicted = (predicted_sigma_xy_post_refactor(fso, t_bo_xy)
                               * 100.0)  # cm

            held_obj_id = d.get("gripper_state", {}).get("held_obj_id")
            is_held = (held_obj_id is not None
                       and int(held_obj_id) == int(oid_str))
            rows.append({
                "trajectory": traj,
                "frame": fr,
                "phase": phase,
                "oid": int(oid_str),
                "label": tr["label"],
                "fso": fso,
                "is_held": is_held,
                "world_x": round(float(T_world_obj[0, 3]), 3),
                "world_y": round(float(T_world_obj[1, 3]), 3),
                "world_z": round(float(T_world_obj[2, 3]), 3),
                "t_bo_xy_m": round(t_bo_xy, 3),
                "sigma_major_cm": round(sM, 2),
                "sigma_mid_cm": round(smid, 2),
                "sigma_minor_cm": round(smin, 2),
                "sigma_xy_topdown_cm": round(sigma_xy_topdown, 2),
                "sigma_predicted_post_cm": round(sigma_predicted, 2),
                "delta_cm": round(sM - sigma_predicted, 2),
                "abs_dot_r2o": round(float(abs(np.dot(maj_xy, r2o))), 3),
                "abs_dot_heading": round(float(abs(np.dot(maj_xy, hd))), 3),
                "maj_xy_x": round(float(maj_xy[0]), 3),
                "maj_xy_y": round(float(maj_xy[1]), 3),
            })
    return rows


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "baseline")
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[dict] = []
    for traj in TRAJECTORIES:
        rows = process_trajectory(traj)
        if not rows:
            continue
        out_path = os.path.join(out_dir, f"cov_baseline_{traj}.csv")
        cols = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[wrote] {out_path}  ({len(rows)} rows)")
        all_rows.extend(rows)

    # Summary print: focus on the top-down xy ellipse (what the
    # visualizer actually renders) and the released-bottle case.
    if all_rows:
        print()
        print("Summary (sigma_xy_topdown_cm = top-down ellipse major; "
              "sigma_major_cm = full 3D major):")
        print()
        hdr = (f"{'traj':>22s} {'fr':>4s} {'oid':>3s} {'label':>10s} "
               f"{'fso':>4s} {'σ_xy_td':>7s} {'σ_3D':>7s} {'σ_pred':>7s}")
        print(hdr)
        print("-" * len(hdr))
        for r in all_rows:
            print(
                f"{r['trajectory'][:22]:>22s} {r['frame']:>4d} "
                f"{r['oid']:>3d} {r['label'][:10]:>10s} {r['fso']:>4d} "
                f"{r['sigma_xy_topdown_cm']:>7.2f} "
                f"{r['sigma_major_cm']:>7.2f} "
                f"{r['sigma_predicted_post_cm']:>7.2f}"
            )

        # Acceptance check — plan §C.4.
        # Static unobserved (fso > 0, NOT held): σ_xy_td ≤ 5 cm.
        # Released objects (high fso AND elevated cov): σ_xy_td ≤ 30 cm.
        # Held tracks: explicitly OUT OF SCOPE for §C — they accumulate
        # cov via rigid_attachment_predict's Ad transport (the §B.2
        # hidden mechanism-2 issue, deferred per §C.5 scope).
        print()
        violations = []
        for r in all_rows:
            if r["is_held"]:
                continue  # OUT OF SCOPE for §C
            # Released-object heuristic: fso > 50 AND σ_xy noticeably
            # elevated above the static floor.
            is_released_like = (r["fso"] > 50
                                 and r["sigma_xy_topdown_cm"] > 6.0)
            limit = 30.0 if is_released_like else 5.0
            if r["sigma_xy_topdown_cm"] > limit:
                violations.append((r, limit))
        if violations:
            print(f"** {len(violations)} acceptance violations **")
            for r, lim in violations[:20]:
                print(
                    f"  {r['trajectory']}/fr{r['frame']}/oid{r['oid']}"
                    f"({r['label']}): σ_xy_td={r['sigma_xy_topdown_cm']:.2f}"
                    f" cm > limit {lim:.0f}"
                )
        else:
            print("** All tracks pass §C acceptance "
                  "(static ≤5cm, released ≤30cm; held out of scope) **")


if __name__ == "__main__":
    main()

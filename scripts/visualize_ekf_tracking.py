#!/usr/bin/env python3
"""
Per-frame 5-panel visualization of the Bernoulli-EKF scene-graph tracker.

Panel layout (2 rows x 3 cols):
  [1] Perception overlay   [2] Top-down: state entering frame   [3] EKF step intermediates
  [4] Top-down: post-predict   [5] Top-down: post-update       [6] r / cov evolution

Drives a thin instrumented replica of ekf_tracker.orchestrator._fast_tier_bernoulli
(so we can snapshot state between predict / associate / update / birth / prune
and dump per-track intermediates — d2, log_lik, Huber weight, existence r delta, etc.).

Data: expects apple_in_the_tray dataset layout:
  datasets/apple_in_the_tray/
    rgb/rgb_NNNNNN.png
    depth/depth_NNNNNN.npy
    pose_txt/amcl_pose.txt          (world <- base)
  tests/visualization_pipeline/apple_in_the_tray/perception/detection_h/
    detection_NNNNNN_final.json     (SAM2-tracked detections with masks + IDs)

Output: tests/visualization_pipeline/apple_in_the_tray/ekf_debug/
    frame_NNNNNN.png

Run:
    conda run -n ocmp_test python tests/visualize_ekf_tracking.py \
        --max-frame 700 --step 1
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from perception.association import hungarian_associate
from ekf_tracker.state.bernoulli import (
    r_predict, r_assoc_update_loglik, r_miss_update, r_birth,
)
from utils.ekf_se3 import (
    huber_weight, process_noise_for_phase, saturate_covariance,
)
from ekf_tracker.state.gaussian_state import GaussianState
from perception.det_dedup import suppress_subpart_detections
from perception.icp_pose import (
    PoseEstimator, centroid_cam_from_mask, _back_project,
)
from ekf_tracker.state.obs_chain import ChainStore
from ekf_tracker.manipulation.gravity_predict import predict_landing_pose
from utils.object_dynamics import lookup_dynamics
from perception.voxel_observability import VoxelObservability
from ekf_tracker.config import BernoulliConfig
from ekf_tracker.birth_gate import _PendingBirth, birth_admissible
from ekf_tracker.relations.relation_filter import RelationFilter
from utils.robot_models import create_gripper_geometry
from ekf_tracker.manipulation.grasp_owner_detector import (
    GraspOwnerDetector, GaussianEkfTrackerState,
)
from ekf_tracker.relations.relation_utils import (
    expand_held_with_relations, should_recompute_relations,
    RelationTriggerState, RelationTriggerConfig,
)
from ekf_tracker.factor_graph import RelationEdge
from utils.slam_interface import PoseEstimate
from perception.visibility import visibility_p_v


# ─── data paths ──────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(SCENEREP_ROOT, "datasets")
VIZ_BASE = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline")

# ─── Fetch head camera intrinsics (from configs/*.yaml) ──────────────────
K_DEFAULT = np.array([
    [554.3827, 0.0, 320.5],
    [0.0, 554.3827, 240.5],
    [0.0, 0.0,     1.0],
], dtype=np.float64)

# Centroid measurement noise (camera-frame). 2 cm std absorbs the
# perception-side mask boundary noise that free tracks (cup, bottle,
# free apples) produce; held tracks are anchored by rigid-attach
# predict so don't depend on Kalman gain magnitude. Used at three
# call sites (Hungarian d², matched-update innovation diagnostics,
# the actual EKF update); same value everywhere keeps gate / cost /
# update consistent.
_CENTROID_R_CAM_STD_M = 0.02
_R_CENT_CAM_3D = np.diag([_CENTROID_R_CAM_STD_M ** 2] * 3)
# Rotation-decoupling: centroid measurement carries no rotation info,
# so the 6D-shaped innovation pads the rotation block with ∞ so it
# falls out of every solve.
_ROTATION_DECOUPLE_VAR = 1e6

# Per-frame slack covariance for the rigid-attachment predict (grip
# slip / deformation). Tiny on top of Ad·P·Adᵀ.
_Q_MANIP_SLACK = np.diag([1e-6] * 3 + [1e-6] * 3)

# palette matches visualize_sam2_observations for inter-viz consistency
_PALETTE_RGB = [
    (  0, 200,  80), (220,  60,  40), ( 40, 140, 220), (245, 200,  20),
    (160,  80, 200), (240, 130,  30), ( 20, 180, 160), (230, 120, 110),
    (100, 160, 230), (250, 220,  60), ( 80, 200, 120), (200, 100, 160),
    (140, 200,  50), (100, 100, 240), (230, 160,  80), ( 40, 220, 200),
    (220,  80, 200), (120, 120, 120), (200, 220, 120), ( 60, 100, 180),
]


def _palette_color(oid: int) -> Tuple[int, int, int]:
    return _PALETTE_RGB[int(oid) % len(_PALETTE_RGB)]


def _palette_color_f(oid: int) -> Tuple[float, float, float]:
    r, g, b = _palette_color(oid)
    return (r / 255.0, g / 255.0, b / 255.0)


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def _load_amcl_poses(path: str) -> List[np.ndarray]:
    """Parse `amcl_pose.txt` lines: `idx x y z qx qy qz qw`."""
    out: List[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out.append(T)
    return out


def _load_T_bc_poses(path: str) -> Optional[Dict[int, np.ndarray]]:
    """Parse `T_bc.txt` lines: `idx x y z qx qy qz qw` -> {idx: T_bc}.

    T_bc is the per-frame base_link -> head_camera_rgb_optical_frame
    extrinsic, extracted from /tf at dataset-extraction time. Returns
    None if the file is missing (caller falls back to identity, i.e.
    camera == base, which is wrong on Fetch but back-compat with bags
    extracted before T_bc support was added).
    """
    if not os.path.exists(path):
        return None
    out: Dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            idx, tx, ty, tz, qx, qy, qz, qw = arr
            try:
                idx_i = int(idx)
                tx, ty, tz, qx, qy, qz, qw = (
                    float(tx), float(ty), float(tz),
                    float(qx), float(qy), float(qz), float(qw))
            except ValueError:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out[idx_i] = T
    return out if out else None


def _load_ee_poses(path: str) -> Optional[Dict[int, np.ndarray]]:
    """Parse `ee_pose.txt` lines: `idx x y z qx qy qz qw` -> {idx: T_bg}.

    Each line gives the end-effector-in-base pose. Used as T_bg for the
    rigid-attachment predict on held tracks.
    """
    if not os.path.exists(path):
        return None
    out: Dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            try:
                idx_i = int(arr[0])
                tx, ty, tz, qx, qy, qz, qw = (float(v) for v in arr[1:])
            except ValueError:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out[idx_i] = T
    return out if out else None


def _load_gripper_widths(path: str) -> Optional[Dict[int, float]]:
    """Parse `joints_pose.json` -> {idx: l+r finger joint width}.

    The Fetch gripper's two finger joints each report half the opening;
    their sum is the total jaw width in metres. Open ≈ 0.10 m, fully
    closed (around an object) ≈ 0.003 m.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    out: Dict[int, float] = {}
    for fr_key, joints in data.items():
        try:
            idx_i = int(fr_key)
        except (TypeError, ValueError):
            continue
        lg = joints.get("l_gripper_finger_joint")
        rg = joints.get("r_gripper_finger_joint")
        if lg is None or rg is None:
            continue
        out[idx_i] = float(lg) + float(rg)
    return out if out else None


from ekf_tracker.manipulation.gripper_state_inferrer import _GripperStateInferrer  # noqa: E402,F401


from ekf_tracker.relations.relation_orchestrator import RelationOrchestrator as _RelationPipeline  # noqa: E402,F401


def _load_detection_json(path: str) -> List[Dict[str, Any]]:
    """Decode one detection_h JSON into a list of dicts (mask decoded)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    out: List[Dict[str, Any]] = []
    for det in data.get("detections", []):
        mask_b64 = det.get("mask", "")
        if not mask_b64:
            continue
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask = np.array(Image.open(BytesIO(mask_bytes)).convert("L"))
            mask = (mask > 128).astype(np.uint8)
        except Exception:
            continue
        out.append({
            "id": int(det.get("object_id")),
            "label": det.get("label", "unknown"),
            # Carry the soft per-detection label distribution from
            # perception (used by the EKF cost as a label history seed
            # and by the JSON state dump for inspection).
            "labels": det.get("labels", {}),
            "mask": mask,
            "score": float(det.get("score", 0.0)),
            "mean_score": float(det.get("mean_score", 0.0)),
            "n_obs": int(det.get("n_obs", 0)),
            "box": det.get("box"),
        })
    return out


def _load_rgb(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_depth(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Gaussian-EKF tracker (lifted to ekf_tracker.gaussian_ekf_tracker)
# ─────────────────────────────────────────────────────────────────────────

from ekf_tracker.gaussian_ekf_tracker import GaussianEkfTracker  # noqa: E402,F401



# ─────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────

def _overlay_detections(rgb: np.ndarray,
                         detections: List[Dict[str, Any]],
                         alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    for det in detections:
        oid = det.get("id")
        if oid is None:
            continue
        color = _palette_color(oid)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        mask = det.get("mask")
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mb = mask.astype(bool)
            if mb.any():
                colored = np.zeros_like(out)
                colored[mb] = color
                out = np.where(mb[..., None],
                                (alpha * colored + (1 - alpha) * out).astype(np.uint8),
                                out)
        bb = det.get("box")
        if bb is not None and len(bb) == 4:
            x0, y0, x1, y1 = map(int, bb)
            cv2.rectangle(out, (x0, y0), (x1, y1), color_bgr[::-1], 2)
            tag = f"id:{oid} {det.get('label','?')} s={det.get('score',0):.2f}"
            ty = max(y0 - 4, 12)
            cv2.rectangle(out, (x0, ty - 10), (x0 + 10 + 8 * len(tag), ty + 3),
                          (255, 255, 255), -1)
            cv2.putText(out, tag, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color_bgr[::-1], 1, cv2.LINE_AA)
    return out


def _plot_topdown(ax,
                  tracks: Dict[int, Dict[str, Any]],
                  dets_with_pose: List[Dict[str, Any]],
                  T_wb: np.ndarray,
                  xlim: Tuple[float, float],
                  ylim: Tuple[float, float],
                  title: str,
                  show_obs: bool = False) -> None:
    """Top-down scatter in the world frame.

    Each tracked object: filled circle at (x, y) + dashed uncertainty ellipse
    from the xy block of cov. Radius / ellipse scaled for visibility.

    If `show_obs` True, also draws detection centroids (from T_wb · T_co)
    as unfilled black squares, so you can see measurement vs. state.
    """
    # Camera frustum proxy: robot base location as a small triangle.
    bx, by = float(T_wb[0, 3]), float(T_wb[1, 3])
    # Heading = x-axis of the base in world.
    hx, hy = float(T_wb[0, 0]), float(T_wb[1, 0])
    theta = np.arctan2(hy, hx)

    ax.plot([bx], [by], marker="^", markersize=10,
            color="black", zorder=3)
    # Draw heading arrow.
    ax.annotate("", xy=(bx + 0.15 * np.cos(theta), by + 0.15 * np.sin(theta)),
                xytext=(bx, by),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Track circles + uncertainty. Use the EKF-composed world-frame
    # mean (`T_world`) for visual continuity. If an observation chain
    # is present, ALSO mark the chain-smoothed world-frame mean as a
    # cross -- when the two diverge, the filter has drifted relative
    # to the loop-closure-aware chain.
    for oid, tr in tracks.items():
        T = tr.get("T_world", tr["T"])
        # Prefer world-frame covariance when present so the ellipse axes
        # are in world-x/world-y instead of body-x/body-y. Falls back to
        # the base-frame cov when SLAM has not yet been ingested (first
        # frame) or `collapsed_object_world` returned None for any reason.
        cov = tr.get("cov_world")
        if cov is None:
            cov = tr["cov"]
        r_ex = tr["r"]
        x, y = float(T[0, 3]), float(T[1, 3])
        col = _palette_color_f(oid)
        # Scale circle by existence (more opaque when confident).
        alpha = 0.25 + 0.65 * max(0.0, min(1.0, r_ex))
        ax.scatter([x], [y], s=120, c=[col], alpha=alpha,
                   edgecolors="black", linewidths=0.8, zorder=4)
        # Uncertainty ellipse (3sigma in xy).
        cov_xy = cov[:2, :2]
        try:
            # Standard 3σ ellipse: matplotlib's `width` is the diameter
            # along the LOCAL x-axis BEFORE the `angle` rotation. To put
            # the visible major axis along the larger eigenvector, we
            # need width=major-diameter AND angle=direction-of-major.
            # `np.linalg.eigh` returns ascending eigenvalues, so the
            # major eigenvector is `v_eig[:, 1]`; we sort descending here
            # so width corresponds to it unambiguously.
            w_eig, v_eig = np.linalg.eigh(cov_xy)
            w_eig = np.clip(w_eig, 1e-10, None)
            order = np.argsort(w_eig)[::-1]
            w_eig = w_eig[order]
            v_eig = v_eig[:, order]
            width = 2 * 3 * float(np.sqrt(w_eig[0]))    # major (along angle)
            height = 2 * 3 * float(np.sqrt(w_eig[1]))   # minor (perpendicular)
            angle = np.degrees(np.arctan2(v_eig[1, 0], v_eig[0, 0]))
            ell = mpatches.Ellipse(
                (x, y), width=width, height=height,
                angle=float(angle),
                fill=False, edgecolor=col, lw=1.0, linestyle="--",
                alpha=0.7, zorder=2,
            )
            ax.add_patch(ell)
        except Exception:
            pass
        ax.text(x + 0.015, y + 0.015,
                f"id:{oid}\nr={r_ex:.2f}",
                fontsize=6.5, color="black",
                bbox=dict(facecolor="white", alpha=0.7, pad=0.8,
                          edgecolor="none"), zorder=5)

        # Chain-smoothed world-frame mean (loop-closure-aware).
        # When divergent from the EKF mean it signals filter drift.
        T_chain = tr.get("T_world_chain")
        if T_chain is not None:
            cx, cy = float(T_chain[0, 3]), float(T_chain[1, 3])
            ax.scatter([cx], [cy], s=70, marker="x",
                        color=col, linewidths=1.6, alpha=0.9, zorder=6)
            # Connect filter mean and chain mean if they're far enough
            # apart to be visible.
            if (cx - x) ** 2 + (cy - y) ** 2 > 1e-6:
                ax.plot([x, cx], [y, cy], color=col, linewidth=0.6,
                        alpha=0.5, zorder=2)

    if show_obs:
        for det in dets_with_pose:
            if not det.get("_icp_ok"):
                continue
            T_co = det["T_co"]
            T_wo = T_wb @ T_co
            dx, dy = float(T_wo[0, 3]), float(T_wo[1, 3])
            oid = det.get("id")
            col = _palette_color_f(oid) if oid is not None else (0, 0, 0)
            ax.scatter([dx], [dy], s=60, facecolors="none",
                       edgecolors=col, marker="s", linewidths=1.4, zorder=6)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)


def _format_assoc_matrix(assoc: Dict[str, Any],
                          max_rows: int = 10,
                          max_cols: int = 8) -> List[str]:
    """Render the Hungarian cost matrix, with row/col annotations and a
    star next to the matched cells.

    Cost convention (`association.hungarian_associate`, perception-style):
       cost[i, l] =   d^2[i, l]
                    - alpha * 1[sam2_tau matches]
                    + label_penalty * 1[d_label NOT in track i's label history]
                    + score_weight  * (1 - score_l)
                  = +INF (printed as INF) otherwise (d^2 > G_out, etc.)
    Negative cost = SAM2 tracklet-id bonus fired (alpha overshot d^2);
    INF = either no innovation, no ICP, or d^2 > G_out (=25).
    """
    track_oids = assoc.get("track_oids", [])
    track_labels = assoc.get("track_labels", []) or ["?"] * len(track_oids)
    track_label_hists = assoc.get("track_label_hists",
                                    [""] * len(track_oids))
    track_taus = assoc.get("track_taus", []) or [-1] * len(track_oids)
    cm = assoc.get("cost_matrix", [])
    det_meta = assoc.get("det_meta_in_assoc", [])
    if not track_oids or not cm:
        return ["  (no tracks or no detections this frame)"]
    match_local = {int(o): int(l)
                   for o, l in assoc.get("match_local", {}).items()}

    n_rows = len(track_oids)
    n_cols = len(cm[0]) if cm else 0
    rows_used = min(n_rows, max_rows)
    cols_used = min(n_cols, max_cols)

    out: List[str] = []
    alpha = assoc.get("alpha", 0.0)
    G_out = assoc.get("G_out", 25.0)
    lbl_pen = assoc.get("label_penalty", 0.0)
    sc_w = assoc.get("score_weight", 0.0)
    gate_mode = assoc.get("gate_mode", "full")
    G_out_trans = assoc.get("G_out_trans", 21.108)
    G_out_rot = assoc.get("G_out_rot", 21.108)
    cost_d2_mode = assoc.get("cost_d2_mode", "full")
    out.append(f"COST = d^2[{cost_d2_mode}] - alpha*1[tau match]"
               f" + lbl_pen*1[label miss] + sc_w*(1-score)")
    if gate_mode == "trans":
        gate_str = f"gate=trans  G_out_trans={G_out_trans:.2f}"
    elif gate_mode == "trans_and_rot":
        gate_str = (f"gate=trans+rot  G_out_trans={G_out_trans:.2f}  "
                    f"G_out_rot={G_out_rot:.2f}")
    else:
        gate_str = f"gate=full  G_out={G_out:.1f}"
    out.append(f"  alpha={alpha:.1f}  lbl_pen={lbl_pen:.1f}  "
               f"sc_w={sc_w:.1f}  {gate_str}  "
               f"(INF=infeasible; * = matched)")

    # Column header: per-det metadata
    col_w = 11
    header = "  oid lbl tau hist        |"
    for li in range(cols_used):
        m = det_meta[li] if li < len(det_meta) else {
            "sam2_id": "?", "label": "?", "global_idx": li, "score": 0.0}
        header += (f" d{m['global_idx']:>3}({m['sam2_id']}/{m['label'][:4]:<4}"
                   f"|{m.get('score', 0.0):.2f})")
    if n_cols > max_cols:
        header += f" +{n_cols - max_cols}"
    out.append(header)

    for ri in range(rows_used):
        oid = int(track_oids[ri])
        lbl = str(track_labels[ri])[:4]
        tau = track_taus[ri]
        hist = (track_label_hists[ri] if ri < len(track_label_hists)
                else "")[:11]
        row_str = f"  {oid:>3} {lbl:<4} {tau:>3} {hist:<11} |"
        for li in range(cols_used):
            c = cm[ri][li]
            star = "*" if match_local.get(oid) == li else " "
            if c >= 1e10:
                row_str += f"          INF{star}"
            else:
                row_str += f"      {c:>+7.2f}{star}"
        out.append(row_str)
    if n_rows > max_rows:
        out.append(f"  ... +{n_rows - max_rows} more tracks ...")
    return out


def _format_intermediates_text(dbg: Dict[str, Any]) -> str:
    """Compose the multi-line text block for panel 3."""
    lines: List[str] = []

    enter = dbg["enter_tracks"]
    post_p = dbg["post_predict_tracks"]
    post_u = dbg["post_update_tracks"]
    lines.append(f"tracks before -> after predict -> after update: "
                 f"{len(enter)} / {len(post_p)} / {len(post_u)}")

    # Predict deltas per track: tr(P) grows, T unchanged.
    lines.append("")
    lines.append("PREDICT  (tr P before -> after; T unchanged):")
    pred_rows = []
    for oid in sorted(enter.keys()):
        if oid not in post_p:
            continue
        trP0 = float(np.trace(enter[oid]["cov"]))
        trP1 = float(np.trace(post_p[oid]["cov"]))
        chain_n = int(post_p[oid].get("chain_len", 0))
        pred_rows.append(f"  id:{oid:<3d} tr(P): {trP0:.2e} -> {trP1:.2e}  chain_len={chain_n}")
    lines.extend(pred_rows[:8] if pred_rows
                 else ["  (no tracks)"])
    if len(pred_rows) > 8:
        lines.append(f"  ... +{len(pred_rows) - 8} more ...")

    # Association results.
    assoc = dbg.get("assoc", {})
    lines.append("")
    lines.append(f"ASSOC  n_tracks={len(assoc.get('track_oids', []))}  "
                 f"n_dets_icp={assoc.get('n_dets_for_assoc', 0)}  "
                 f"(total {assoc.get('n_dets_total', 0)})")
    m = assoc.get("match", {})
    lines.append(f"  matched: {len(m)}  "
                 f"unmatched_tr: {len(assoc.get('unmatched_tracks', []))}  "
                 f"unmatched_det: {len(assoc.get('unmatched_dets_local', []))}")
    lines.append("")
    lines.extend(_format_assoc_matrix(assoc))

    # Matched pairs table.
    lines.append("")
    lines.append("MATCHED (d^2 gate = 25):")
    matched = dbg.get("matched", [])
    if matched:
        lines.append("  id   det   d^2  d2_t  d2_r   w    logL     r_prev -> r_new  fit/rmse")
        for m_row in matched[:12]:
            flag = "[REJ]" if m_row.get("reject_outer_gate") else "     "
            d2t = m_row.get("d2_trans", float("nan"))
            d2r = m_row.get("d2_rot", float("nan"))
            lines.append(
                f"  {m_row['oid']:<3d} {m_row['det_idx']:<3d}  "
                f"{m_row['d2']:5.2f} {d2t:5.2f} {d2r:6.2f} "
                f"{m_row['w']:4.2f} {m_row['log_lik']:7.1f}   "
                f"{m_row['r_prev']:.3f} -> {m_row['r_new']:.3f}  "
                f"{m_row['fitness']:.2f}/{m_row['rmse']*1e3:4.1f}mm {flag}"
            )
    else:
        lines.append("  (none)")

    # Missed branch.
    lines.append("")
    lines.append("MISSED (eq:r_miss):")
    missed = dbg.get("missed", [])
    if missed:
        lines.append("  id   p_v   p~_d   r_prev -> r_new")
        for m_row in missed[:10]:
            lines.append(
                f"  {m_row['oid']:<3d}  {m_row['p_v']:.2f}  "
                f"{m_row['p_d_tilde']:.2f}   "
                f"{m_row['r_prev']:.3f} -> {m_row['r_new']:.3f}"
            )
    else:
        lines.append("  (none)")

    # Births + prunes.
    births = dbg.get("births", [])
    prunes = dbg.get("pruned", [])
    lines.append("")
    lines.append(f"BIRTHS: {len(births)}    PRUNES: {len(prunes)}")
    for b in births[:8]:
        lines.append(f"  +id:{b['new_oid']} {b['label']} s={b['score']:.2f} "
                     f"r_new={b['r_new']:.3f}")
    for p in prunes[:8]:
        lines.append(f"  -id:{p['oid']} r={p['r']:.2e}")

    return "\n".join(lines)


def _plot_intermediates(ax, dbg: Dict[str, Any]) -> None:
    ax.axis("off")
    txt = _format_intermediates_text(dbg)
    ax.text(0.0, 1.0, txt, transform=ax.transAxes,
            fontsize=6.8, family="monospace",
            verticalalignment="top", horizontalalignment="left")
    ax.set_title("[3] EKF step intermediates", fontsize=10)


def _plot_r_evolution(ax,
                       r_history: Dict[int, List[Tuple[int, float]]],
                       xlim: Tuple[int, int]) -> None:
    ax.set_title("[6] existence r(t) per track", fontsize=10)
    any_plotted = False
    for oid, hist in r_history.items():
        if not hist:
            continue
        xs = [h[0] for h in hist]
        ys = [h[1] for h in hist]
        ax.plot(xs, ys, marker=".", markersize=3, linewidth=1.0,
                color=_palette_color_f(oid), label=f"id:{oid}")
        any_plotted = True
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(xlim)
    ax.set_xlabel("frame", fontsize=8)
    ax.set_ylabel("r", fontsize=8)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if any_plotted:
        # Unique handles (matplotlib legend picks duplicates up otherwise).
        handles, labels = ax.get_legend_handles_labels()
        seen_l = []
        seen_h = []
        for h, l in zip(handles, labels):
            if l not in seen_l:
                seen_l.append(l)
                seen_h.append(h)
        if seen_l:
            ax.legend(seen_h[:12], seen_l[:12], fontsize=6, ncol=2,
                      loc="lower left", framealpha=0.85)


def _compute_topdown_extent(tracks_snapshots: List[Dict[int, Dict[str, Any]]],
                             dets_with_pose: List[Dict[str, Any]],
                             T_wb: np.ndarray,
                             pad: float = 0.4) -> Tuple[Tuple[float, float],
                                                         Tuple[float, float]]:
    xs: List[float] = [float(T_wb[0, 3])]
    ys: List[float] = [float(T_wb[1, 3])]
    for snap in tracks_snapshots:
        for oid, tr in snap.items():
            T_use = tr.get("T_world", tr["T"])
            xs.append(float(T_use[0, 3]))
            ys.append(float(T_use[1, 3]))
    for det in dets_with_pose:
        if det.get("_icp_ok"):
            T_wo = T_wb @ det["T_co"]
            xs.append(float(T_wo[0, 3]))
            ys.append(float(T_wo[1, 3]))
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    # Enforce a minimum span so an almost-stationary camera doesn't look
    # infinitely zoomed.
    min_span = 0.6
    if x1 - x0 < min_span:
        c = 0.5 * (x0 + x1)
        x0, x1 = c - min_span / 2, c + min_span / 2
    if y1 - y0 < min_span:
        c = 0.5 * (y0 + y1)
        y0, y1 = c - min_span / 2, c + min_span / 2
    return (x0, x1), (y0, y1)


def _track_to_jsonable(tr: Dict[str, Any]) -> Dict[str, Any]:
    """One track snapshot → JSON-serialisable dict (matrices to lists).

    `T`, `cov`, `xyz` are BASE frame (the recursion's storage frame).
    `T_world`, `xyz_w` are derived for visualization only via composition
    with the current SLAM `T_wb` (no Σ_wb propagation -- not the EKF's
    cov anymore).
    """
    out = {
        "T": tr["T"].tolist(),
        "cov": tr["cov"].tolist(),
        "label": tr["label"],
        "label_scores": tr.get("label_scores", {}),
        "r": float(tr["r"]),
        "frames_since_obs": int(tr["frames_since_obs"]),
        "sam2_tau": int(tr["sam2_tau"]),
        "xyz": [float(tr["T"][0, 3]),
                float(tr["T"][1, 3]),
                float(tr["T"][2, 3])],
        "tr_cov": float(np.trace(tr["cov"])),
    }
    cov_world = tr.get("cov_world")
    if cov_world is not None:
        out["cov_world"] = cov_world.tolist() if hasattr(
            cov_world, "tolist") else cov_world
    T_world = tr.get("T_world")
    if T_world is not None:
        out["T_world"] = T_world.tolist() if hasattr(T_world, "tolist") \
            else T_world
        out["xyz_w"] = [float(T_world[0, 3]),
                        float(T_world[1, 3]),
                        float(T_world[2, 3])]
    T_world_chain = tr.get("T_world_chain")
    if T_world_chain is not None:
        out["T_world_chain"] = T_world_chain.tolist() if hasattr(
            T_world_chain, "tolist") else T_world_chain
        out["xyz_w_chain"] = [float(T_world_chain[0, 3]),
                                float(T_world_chain[1, 3]),
                                float(T_world_chain[2, 3])]
    cov_world_chain = tr.get("cov_world_chain")
    if cov_world_chain is not None:
        out["cov_world_chain"] = cov_world_chain.tolist() if hasattr(
            cov_world_chain, "tolist") else cov_world_chain
    out["chain_len"] = int(tr.get("chain_len", 0))
    out["chain_n_used"] = int(tr.get("chain_n_used", 0))
    return out


def _dump_frame_json(out_path: str,
                      dbg: Dict[str, Any],
                      detections_raw: List[Dict[str, Any]],
                      dets_with_pose: List[Dict[str, Any]]) -> None:
    """Save the full per-frame EKF state for offline diagnosis.

    Schema:
      frame, slam_pose, detections (raw + ICP outputs), tracks at three
      snapshots (enter / post_predict / post_update), association
      (cost matrix, match dict, unmatched lists), visibility (per-oid p_v),
      matched_events, missed_events, births, prunes.

    Mask payloads from detections are NOT included (would balloon the file).
    """
    T_wb = dbg["slam_pose"]
    T_bc = np.asarray(dbg.get("T_bc", np.eye(4)), dtype=np.float64)

    # Detections: keep id/label/score/box + ICP outputs + world-frame T_wo
    # for easy diagnosis. Drop the binary mask. Iterate over
    # `dets_with_pose` (post-suppression) — each entry already carries
    # the raw fields (id/label/score/box/labels) plus the pose outputs,
    # so id and pose are guaranteed to come from the same record.
    # `detections_raw` is no longer joined here; it is still consumed
    # by the events panel for subpart-absorbed pid lookups.
    det_records: List[Dict[str, Any]] = []
    for i, dwp in enumerate(dets_with_pose):
        rec: Dict[str, Any] = {
            "global_idx": i,
            "id": int(dwp.get("id")) if dwp.get("id") is not None else None,
            "label": dwp.get("label"),
            "labels": dwp.get("labels", {}),
            "score": float(dwp.get("score", 0.0)),
            "mean_score": float(dwp.get("mean_score", 0.0)),
            "n_obs": int(dwp.get("n_obs", 0)),
            "box": dwp.get("box"),
            "_icp_ok": bool(dwp.get("_icp_ok", False)),
            "fitness": float(dwp.get("fitness", 0.0)),
            "rmse": float(dwp.get("rmse", 0.0)),
        }
        if dwp.get("_icp_ok"):
            T_co = np.asarray(dwp["T_co"], dtype=np.float64)
            # T_co is camera-frame; lift through T_bc to base, then T_wb
            # to world. Previously was T_wb @ T_co (missing T_bc) which
            # produced an xyz_w shifted by ~|t_bc| from the actual world
            # position the panels render via _centroid_cam.
            T_wo = T_wb @ T_bc @ T_co
            rec["T_co"] = T_co.tolist()
            rec["R_icp"] = np.asarray(dwp["R_icp"], dtype=np.float64).tolist()
            rec["T_wo"] = T_wo.tolist()
            rec["xyz_w"] = [float(T_wo[0, 3]),
                            float(T_wo[1, 3]),
                            float(T_wo[2, 3])]
        det_records.append(rec)

    payload = {
        "frame": int(dbg["frame"]),
        "slam_pose": T_wb.tolist(),
        "T_bc": np.asarray(dbg["T_bc"]).tolist() if "T_bc" in dbg else None,
        "gripper_state": dbg.get("gripper_state", {}),
        "held_oids_used": list(dbg.get("held_oids_used", [])),
        "relations": list(dbg.get("relations", [])),
        "relation_call": dict(dbg.get("relation_call", {})),
        "detections": det_records,
        "tracks_enter": {str(oid): _track_to_jsonable(tr)
                          for oid, tr in dbg["enter_tracks"].items()},
        "tracks_post_predict": {str(oid): _track_to_jsonable(tr)
                                 for oid, tr in dbg["post_predict_tracks"].items()},
        "tracks_post_update": {str(oid): _track_to_jsonable(tr)
                                for oid, tr in dbg["post_update_tracks"].items()},
        "association": dbg.get("assoc", {}),
        "matched_events": dbg.get("matched", []),
        "missed_events": dbg.get("missed", []),
        "births": dbg.get("births", []),
        "birth_rejects": dbg.get("birth_rejects", []),
        "subpart_absorbed": dbg.get("subpart_absorbed", []),
        "centroid_dropped": dbg.get("centroid_dropped", []),
        "prunes": dbg.get("pruned", []),
        "self_merges": dbg.get("self_merges", []),
        "self_merge_protected_pairs": dbg.get("self_merge_protected_pairs", []),
        "visibility": {int(o): float(v)
                        for o, v in dbg.get("visibility", {}).items()},
        "gravity_predict": dbg.get("gravity_predict"),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────
# Nine per-step clean panels
# ─────────────────────────────────────────────────────────────────────────

# Fixed world extents for ALL top-down panels. 90° CW rotated: the plot's
# horizontal axis shows world_y (natural +y on the right, range [-2, +2])
# and the vertical axis shows world_x with the direction reversed so that
# world_x = 0 is at the BOTTOM and world_x = -4 is at the TOP. Objects
# ahead of the robot (more negative world_x) therefore sit higher on the
# page -- a driver's-view top-down.
_WORLD_X_RANGE = (-4.0, 0.0)
_WORLD_Y_RANGE = (-2.0, 3.0)


def _set_topdown_axes(ax, title):
    """Shared rotated top-down axis formatting. World y goes left-to-right,
    world x goes bottom-to-top (with 0 at the bottom = robot-side)."""
    ax.set_xlim(_WORLD_Y_RANGE)
    ax.set_ylim(0.0, _WORLD_X_RANGE[0])    # (0, -4): world_x=0 at bottom
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("world y [m]", fontsize=8)
    ax.set_ylabel("world x [m]", fontsize=8)
    ax.tick_params(labelsize=7)


def _draw_base_marker(ax, T_wb, color="k", size=90):
    """Robot base + heading arrow on the rotated top-down view."""
    x = float(T_wb[0, 3]); y = float(T_wb[1, 3])
    hx, hy = float(T_wb[0, 0]), float(T_wb[1, 0])   # base +x in world
    # Plot coords: (horizontal, vertical) = (world_y, world_x).
    ax.scatter([y], [x], marker="s", c=color, s=size,
               edgecolors="white", linewidths=1.2, zorder=10, label="base")
    ax.annotate("", xy=(y + 0.22 * hy, x + 0.22 * hx), xytext=(y, x),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
                zorder=10)


def _cov_ellipse_xy(ax, mean_xy, cov_xy, color, n_std=2.0, lw=1.0,
                      alpha=0.25):
    """Draw a 2-σ covariance ellipse on the rotated top-down.

    ``mean_xy = (world_x, world_y)`` and ``cov_xy`` is the 2x2 xy marginal
    in world frame. The plot uses (world_y, world_x) coordinates so we
    permute rows/columns of ``cov_xy`` before computing the ellipse.
    """
    cov = np.asarray(cov_xy, dtype=np.float64)
    # Permute to (y, x) basis: P = [[0,1],[1,0]]; cov_p = P cov P^T.
    cov_p = np.array([[cov[1, 1], cov[1, 0]],
                      [cov[0, 1], cov[0, 0]]], dtype=np.float64)
    # Sort eigenvalues DESCENDING so width=major and angle=direction-of-major
    # remain consistent with `mpatches.Ellipse(width, height, angle)` semantics
    # ("width is the diameter along the LOCAL x-axis BEFORE rotation"). The
    # ascending default from `np.linalg.eigh` would put the smaller sqrt in
    # `width` while `angle` aligned with the larger eigvec → ellipse rendered
    # 90° rotated.
    vals, vecs = np.linalg.eigh(cov_p)
    vals = np.clip(vals, 0.0, None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width = 2.0 * n_std * float(np.sqrt(vals[0]))
    height = 2.0 * n_std * float(np.sqrt(vals[1]))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # Centre in plot coords = (world_y, world_x).
    ell = mpatches.Ellipse(xy=(float(mean_xy[1]), float(mean_xy[0])),
                           width=max(width, 1e-4),
                           height=max(height, 1e-4), angle=float(angle),
                           facecolor=color, edgecolor=color,
                           alpha=alpha, lw=lw, zorder=3)
    ax.add_patch(ell)


# ─── Hungarian cost-matrix helpers (≥5×5 padding, shared drawing) ──────

_HUNG_MIN = 5


def _pad_matrix_min(mat: np.ndarray,
                    track_oids: List[int],
                    det_meta: List[Dict[str, Any]],
                    min_size: int = _HUNG_MIN):
    """Pad a (n_tr, n_det) matrix to at least (min_size, min_size) with
    NaNs. Returns (padded_matrix, padded_track_labels, padded_det_labels)
    where labels for padding slots are ``None``.
    """
    mat = np.asarray(mat, dtype=np.float64) if mat is not None \
        else np.zeros((0, 0), dtype=np.float64)
    if mat.ndim != 2:
        mat = np.zeros((0, 0), dtype=np.float64)
    n_tr, n_det = mat.shape if mat.size else (0, 0)
    n_tr = max(n_tr, len(track_oids))
    n_det = max(n_det, len(det_meta))
    tgt_r = max(n_tr, min_size)
    tgt_c = max(n_det, min_size)
    padded = np.full((tgt_r, tgt_c), np.nan, dtype=np.float64)
    if mat.size:
        r, c = mat.shape
        padded[:r, :c] = mat
    track_labels = list(track_oids) + [None] * (tgt_r - len(track_oids))
    det_labels = list(det_meta) + [None] * (tgt_c - len(det_meta))
    return padded, track_labels, det_labels


def _draw_hungarian_matrix(ax, dbg, mat, title, *,
                           highlight_matches: bool = False,
                           value_fmt: str = ".2f"):
    """Render one Hungarian cost-component matrix, padded to ≥5×5.

    NaN cells render empty; ``>= 1e11`` cells (infeasible) render as "∞".
    """
    assoc = dbg.get("assoc", {})
    track_oids = [int(o) for o in assoc.get("track_oids", [])]
    det_meta = list(assoc.get("det_meta_in_assoc", []))
    match_local = assoc.get("match_local", {})

    mat_p, track_labels, det_labels = _pad_matrix_min(
        mat, track_oids, det_meta, min_size=_HUNG_MIN)

    # Clip infeasible cells for the colormap.
    mat_view = np.where(mat_p >= 1e11, np.nan, mat_p)
    ax.imshow(mat_view, cmap="viridis_r", aspect="auto")

    ax.set_xticks(range(len(det_labels)))
    ax.set_yticks(range(len(track_labels)))
    # Columns = detections in association order.  Label as ``pid:X``
    # using the per-column ``sam2_id`` snapshot (= the detection's
    # perception id), so the matrix's columns line up with the
    # ``pid:X`` labels in the [1A] Detected panel.
    ax.set_xticklabels([
        (f"pid:{m.get('sam2_id')}"
         if isinstance(m, dict) and m.get("sam2_id") is not None
         else "—")
        for m in det_labels
    ], fontsize=7)
    # Rows = live tracks. Label as ``oid:Y`` for parity with [1B] /
    # [1C].
    ax.set_yticklabels([
        (f"oid:{o}" if o is not None else "—") for o in track_labels
    ], fontsize=7)

    for i in range(mat_p.shape[0]):
        for j in range(mat_p.shape[1]):
            v = mat_p[i, j]
            if np.isnan(v):
                continue
            if v >= 1e11 or not np.isfinite(v):
                ax.text(j, i, "∞", ha="center", va="center",
                        fontsize=6, color="lightgray")
            else:
                ax.text(j, i, format(v, value_fmt), ha="center",
                        va="center", fontsize=6, color="white")

    if highlight_matches:
        for oid_str, l_local in match_local.items():
            oid = int(oid_str)
            if oid in track_oids:
                i = track_oids.index(oid)
                rect = mpatches.Rectangle(
                    (int(l_local) - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="red", lw=1.8, zorder=5)
                ax.add_patch(rect)
    ax.set_title(title, fontsize=10)


def _backproject_mask(mask: np.ndarray,
                      depth: np.ndarray,
                      K: np.ndarray,
                      *, max_samples: int = 800) -> np.ndarray:
    """(N, 3) camera-frame points from a binary mask + depth image."""
    if mask is None or depth is None:
        return np.zeros((0, 3), dtype=np.float64)
    H, W = depth.shape[:2]
    ys, xs = np.where((mask > 0) & np.isfinite(depth)
                       & (depth > 0.1) & (depth < 10.0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(xs) > max_samples:
        idx = np.linspace(0, len(xs) - 1, max_samples).astype(np.int64)
        xs = xs[idx]; ys = ys[idx]
    zs = depth[ys, xs].astype(np.float64)
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    Xs = (xs.astype(np.float64) - cx) * zs / fx
    Ys = (ys.astype(np.float64) - cy) * zs / fy
    return np.stack([Xs, Ys, zs], axis=1)


def _plot_step0_slam(ax, dbg, T_wb):
    """Step 0: SLAM ingest + per-frame T_bc. Base + camera on the rotated
    world top-down (fixed extent)."""
    _set_topdown_axes(ax, "[0] SLAM ingest  (base + camera)")
    T_bc = np.asarray(dbg.get("T_bc", np.eye(4)))
    T_wc = T_wb @ T_bc
    bx, by = float(T_wb[0, 3]), float(T_wb[1, 3])
    cx, cy, cz = float(T_wc[0, 3]), float(T_wc[1, 3]), float(T_wc[2, 3])

    _draw_base_marker(ax, T_wb, color="black", size=160)

    # Camera: dotted line to base + optical-axis arrow.
    ax.plot([by, cy], [bx, cx], linestyle=":", color="gray", lw=1.2,
            zorder=4)
    ax.scatter([cy], [cx], marker="^", c="tab:blue", s=140,
               edgecolors="white", linewidths=1.2, zorder=11)
    ox, oy = float(T_wc[0, 2]), float(T_wc[1, 2])   # optical-axis xy
    ax.annotate("", xy=(cy + 0.50 * oy, cx + 0.50 * ox),
                xytext=(cy, cx),
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.6),
                zorder=11)

    oz = float(T_wc[2, 2])
    tilt_deg = np.degrees(np.arccos(np.clip(-oz, -1.0, 1.0)))
    info = (f"base=({bx:+.2f},{by:+.2f})m  "
            f"cam_h={cz:.2f}m  tilt={tilt_deg:.0f}°")
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            fontsize=7, va="top", color="black",
            bbox=dict(facecolor="white", alpha=0.85, pad=2,
                       edgecolor="none"))


def _iter_tracks_xy(tracks_snap, T_wb):
    for oid, tr in tracks_snap.items():
        T_world = tr.get("T_world")
        if T_world is None:
            T_world = T_wb @ np.asarray(tr["T"])
        yield int(oid), (float(T_world[0, 3]), float(T_world[1, 3]))


def _plot_step1_predict(ax, dbg, T_wb):
    """Step 1: EKF predict. Predicted means + 2σ covariance ellipses."""
    _set_topdown_axes(ax, "[1] EKF predict  (μ + 2σ cov)")
    _draw_base_marker(ax, T_wb, color="lightgray")

    for oid, (x, y) in _iter_tracks_xy(
            dbg.get("post_predict_tracks", {}), T_wb):
        col = _palette_color_f(oid)
        # Plot coords: horizontal = world_y, vertical = world_x.
        ax.scatter([y], [x], s=50, color=col,
                   edgecolors="white", linewidths=1.0, zorder=5)
        tr = dbg["post_predict_tracks"][oid]
        # Plot is world-frame; prefer the world-tangent cov when present.
        cov_full = tr.get("cov_world")
        if cov_full is None:
            cov_full = tr["cov"]
        cov = np.asarray(cov_full)[:2, :2]
        _cov_ellipse_xy(ax, (x, y), cov, color=col, n_std=2.0, alpha=0.18)
        ax.text(y, x, f"  {oid}", color=col, fontsize=7,
                va="center", ha="left", zorder=6)


def _build_pid_to_oid(dbg, dets_with_pose) -> Tuple[Dict[Any, int], set]:
    """Return (pid → oid map, held_set) from this frame's dbg + the
    POST-suppression detections (``dets_with_pose``).

    The map is built from `dbg["assoc"]["match"]`, whose values are
    GLOBAL indices into `dets_with_pose` (the producer at orchestrator
    line 928 already remaps Hungarian's local column index through
    `local_to_global`; the LOCAL form is preserved separately under
    `match_local`). Held set comes from `dbg["held_oids_used"]`
    (relation-expanded) or `dbg["gripper_state"]["held_obj_id"]`.
    """
    pid_to_oid: Dict[Any, int] = {}
    held_set: set = set()
    if dbg is None:
        return pid_to_oid, held_set
    assoc = dbg.get("assoc") or dbg.get("association") or {}
    match = assoc.get("match") or {}
    n_total = len(dets_with_pose)
    for oid_key, l_global in match.items():
        try:
            l_g = int(l_global)
            oid_int = int(oid_key)
        except (TypeError, ValueError):
            continue
        if 0 <= l_g < n_total:
            pid = dets_with_pose[l_g].get("id")
            if pid is not None:
                pid_to_oid[pid] = oid_int
    gs = dbg.get("gripper_state") or {}
    h = gs.get("held_obj_id")
    if h is not None:
        try:
            held_set = {int(h)}
        except (TypeError, ValueError):
            pass
    held_used = dbg.get("held_oids_used")
    if held_used:
        held_set = {int(o) for o in held_used}
    return pid_to_oid, held_set


def _plot_detected_pid(ax, rgb, detections, dbg=None):
    """[1A] Perception-side view: each mask labelled ``pid:X (label)``.

    Faithfully renders EVERY detection received from perception, even
    those rejected before Hungarian (subpart-absorbed,
    centroid-dropped). Color is keyed on the perception id so the
    same SAM2 mask keeps its hue across frames (until SAM2 reseeds
    it). Rejects keep their pid color but get a desaturated/hatched
    style and a label suffix so the user sees *what* perception
    emitted vs. what reached the matcher.
    """
    ax.imshow(rgb)
    n_total = len(detections)
    n_acc = n_abs = n_cd = n_empty = 0
    ax.set_xticks([]); ax.set_yticks([])

    # Build pid → reject-reason maps from dbg.
    absorbed_pid_to_into: Dict[Any, Any] = {}
    if dbg is not None:
        for sa in dbg.get("subpart_absorbed", []) or []:
            try:
                fi = int(sa.get("from_idx"))
                ii = int(sa.get("into_idx"))
                pid_from = detections[fi].get("id")
                pid_into = detections[ii].get("id")
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if pid_from is not None:
                absorbed_pid_to_into[pid_from] = pid_into
    cent_dropped_pids = set()
    if dbg is not None:
        for cd in dbg.get("centroid_dropped", []) or []:
            pid = cd.get("pid")
            if pid is not None:
                cent_dropped_pids.add(pid)

    empty_pids: List[Any] = []
    for det in detections:
        pid = det.get("id")
        mask = det.get("mask")
        mask_has_pixels = (mask is not None
                           and hasattr(mask, "any")
                           and bool(mask.any()))
        box = det.get("box") or det.get("bbox")
        box_has_area = (box is not None and len(box) >= 4
                         and (float(box[2]) > float(box[0]))
                         and (float(box[3]) > float(box[1])))

        # Empty placeholder = perception emitted a SAM2 ID slot with no
        # mask pixels AND no bbox area. These show up at fr 178 of
        # apple_to_cabinate_look as pids 0/2/10/13: tracks the upstream
        # tracker is still keeping ID slots for but has no current
        # support for. We list them in the title so the user sees that
        # perception sent them but they have nothing to render.
        if not mask_has_pixels and not box_has_area:
            n_empty += 1
            if pid is not None:
                empty_pids.append(pid)
            continue

        # Pick a centroid: prefer mask-centroid, else bbox center.
        if mask_has_pixels:
            ys, xs = np.where(mask > 0)
            cu, cv = float(xs.mean()), float(ys.mean())
        else:
            cu = 0.5 * (float(box[0]) + float(box[2]))
            cv = 0.5 * (float(box[1]) + float(box[3]))

        is_absorbed = pid in absorbed_pid_to_into
        is_cdrop = pid in cent_dropped_pids
        pid_col = (_palette_color_f(pid) if pid is not None
                    else (0.4, 0.4, 0.4))

        if is_absorbed:
            n_abs += 1
            into_pid = absorbed_pid_to_into[pid]
            # Desaturate the pid color so the user can still tell which
            # mask was absorbed by hue, but it visually recedes.
            col = tuple(0.55 + 0.30 * c for c in pid_col)
            if mask is not None and np.any(mask):
                ax.scatter(xs, ys, s=1, c=[col], alpha=0.30,
                            marker="x", linewidths=0)
            ax.scatter([cu], [cv], marker="x", c=[col], s=80,
                        linewidths=1.6, zorder=5)
            label = (f"pid:{pid} ({det.get('label','?')})  "
                     f"absorbed→pid:{into_pid}")
            ax.text(cu + 6, cv - 6, label, color=col, fontsize=7,
                    weight="bold", zorder=5)
            continue

        if is_cdrop:
            n_cd += 1
            # Use the pid color but desaturated, with a dashed mask
            # outline + crossed centroid marker. The label spells out
            # "no-depth" so the user understands why it didn't match.
            col = tuple(0.50 + 0.30 * c for c in pid_col)
            if mask is not None and np.any(mask):
                ax.scatter(xs, ys, s=1, c=[col], alpha=0.30,
                            marker="x", linewidths=0)
            ax.scatter([cu], [cv], marker="x", c=[col], s=80,
                        linewidths=1.6, zorder=5)
            label = f"pid:{pid} ({det.get('label','?')})  no-depth"
            ax.text(cu + 6, cv - 6, label, color=col, fontsize=7,
                    weight="bold", zorder=5)
            continue

        # Accepted detection — full saturation, filled circle marker.
        n_acc += 1
        col = pid_col
        if mask is not None and np.any(mask):
            ax.scatter(xs, ys, s=1, c=[col], alpha=0.35, marker=".",
                        linewidths=0)
        ax.scatter([cu], [cv], marker="o", c=[col], s=80,
                    edgecolors="white", linewidths=1.4, zorder=5)
        label = f"pid:{pid}  ({det.get('label', '?')})"
        ax.text(cu + 6, cv - 6, label, color=col, fontsize=7,
                weight="bold", zorder=5)

    parts = [f"[1A] Detected (pid)  ·  {n_total} from perception",
             f"→  acc:{n_acc}  no-depth:{n_cd}  absorbed:{n_abs}"]
    if n_empty:
        empty_str = ",".join(str(p) for p in empty_pids[:6])
        if len(empty_pids) > 6:
            empty_str += f",…(+{len(empty_pids)-6})"
        parts.append(f"empty(pids:{empty_str}):{n_empty}")
    ax.set_title("  ".join(parts), fontsize=9)


def _plot_tracked_oid(ax, rgb, dets_with_pose, dbg, K=None):
    """[1B] EKF-tracker view: visualise the d²_trans geometry directly.

    Each detection's perception centroid (back-projected from depth,
    re-projected to image) is drawn as a coloured circle. Each
    tracker oid's predicted μ_b (transformed to camera, projected)
    is drawn as a square. Matched (oid, pid) pairs share a colour
    and are joined by a dashed line annotated ``d²t=N.NN`` — the
    line length is the Euclidean piece of what Hungarian minimises.

    Unmatched detections: gray circle, ``[no-track]``.
    Unmatched tracker oids: open square in the oid's palette colour,
    ``[no-match]``. Held-set oids get a yellow square edge.

    Note: in trajectories where perception never produces the label
    "tray" (e.g. apple_in_the_tray), the tray is born as an
    ``apple`` track — so seeing oid 1 (apple) match apple detections
    is the correct behaviour given the labels coming in. Fixing it
    requires perception-side label correction, not a tracker change.
    """
    ax.imshow(rgb)
    ax.set_title(
        "[1B] Tracked  ·  ○ perception · □ EKF projected · — — d²_trans",
        fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    pid_to_oid, held_set = _build_pid_to_oid(dbg, dets_with_pose)
    H_img, W_img = rgb.shape[:2]

    if K is None:
        return

    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx_K = float(K[0, 2]); cy_K = float(K[1, 2])

    def _project_cam(p_c):
        """Camera-frame XYZ → pixel (u, v) or None if behind camera."""
        if p_c is None:
            return None
        Z = float(p_c[2])
        if Z <= 0.05:
            return None
        return (fx * float(p_c[0]) / Z + cx_K,
                fy * float(p_c[1]) / Z + cy_K)

    margin_px = 50.0
    u_min, u_max = margin_px, W_img - margin_px
    v_min, v_max = margin_px, H_img - margin_px
    cx_img, cy_img = W_img * 0.5, H_img * 0.5

    def _clamp_to_inset(p):
        """Off-image points get pushed onto the inset rectangle along
        the ray from image centre — preserves direction so multiple
        off-screen tracks spread along the boundary."""
        u, v = float(p[0]), float(p[1])
        if u_min <= u <= u_max and v_min <= v <= v_max:
            return (u, v)
        du = u - cx_img; dv = v - cy_img
        t_candidates = [1.0]
        if du > 1e-6:
            t_candidates.append((u_max - cx_img) / du)
        elif du < -1e-6:
            t_candidates.append((u_min - cx_img) / du)
        if dv > 1e-6:
            t_candidates.append((v_max - cy_img) / dv)
        elif dv < -1e-6:
            t_candidates.append((v_min - cy_img) / dv)
        t = min(t for t in t_candidates if t > 0.0)
        return (cx_img + t * du, cy_img + t * dv)

    assoc = dbg.get("assoc") or dbg.get("association") or {}
    track_oids = assoc.get("track_oids") or []
    track_labels = assoc.get("track_labels") or []
    label_by_oid = {int(o): (str(track_labels[i])
                              if i < len(track_labels) else "?")
                     for i, o in enumerate(track_oids)}
    label_by_pid = {d.get("id"): d.get("label", "?")
                     for d in dets_with_pose
                     if d.get("id") is not None}

    # ── d²t per oid for the dashed-line annotation ────────────────
    d2_by_oid: Dict[int, float] = {}
    for ev in dbg.get("matched", []) or []:
        if (ev.get("reject_outer_gate")
                or ev.get("reject_held_prefilter")
                or ev.get("reject_innov_clamp")):
            continue
        try:
            d2_by_oid[int(ev.get("oid", -1))] = float(
                ev.get("d2_trans", float("nan")))
        except (TypeError, ValueError):
            continue

    # ── Project all detection centroids (Hungarian's measurement) ─
    det_pix: Dict[Any, Optional[Tuple[float, float]]] = {}
    for det in dets_with_pose:
        pid = det.get("id")
        if pid is None or not det.get("_centroid_ok"):
            continue
        det_pix[pid] = _project_cam(det.get("_centroid_cam"))

    # ── Project all tracker μ_b (Hungarian's prior) ───────────────
    T_bc_arr = np.asarray(dbg.get("T_bc", np.eye(4)), dtype=np.float64)
    try:
        T_cb = np.linalg.inv(T_bc_arr)
    except np.linalg.LinAlgError:
        return
    post_predict = dbg.get("post_predict_tracks") or {}
    trk_pix: Dict[int, Optional[Tuple[float, float]]] = {}
    for oid_raw in track_oids:
        oid = int(oid_raw)
        tr = post_predict.get(oid) or post_predict.get(str(oid))
        if tr is None:
            continue
        T = np.asarray(tr.get("T", np.eye(4)), dtype=np.float64)
        if T.shape != (4, 4):
            continue
        mu_b = T[:3, 3]
        mu_c = T_cb @ np.array([mu_b[0], mu_b[1], mu_b[2], 1.0],
                                  dtype=np.float64)
        trk_pix[oid] = _project_cam(mu_c[:3])

    # ── Faint mask scatter for matched dets (background context) ──
    for det in dets_with_pose:
        pid = det.get("id")
        oid = pid_to_oid.get(pid)
        if oid is None:
            continue
        mask = det.get("mask")
        if mask is None or not np.any(mask):
            continue
        ys, xs = np.where(mask > 0)
        col = _palette_color_f(int(oid))
        ax.scatter(xs, ys, s=1, c=[col], alpha=0.10, marker=".",
                    linewidths=0)

    # ── Matched pairs: circle + square + dashed line + d²t ────────
    matched_oids: set = set()
    for pid, oid in pid_to_oid.items():
        oid_i = int(oid)
        col = _palette_color_f(oid_i)
        edge_col = "yellow" if oid_i in held_set else "white"
        edge_lw = 2.4 if edge_col == "yellow" else 1.2
        cir = det_pix.get(pid)
        sq = trk_pix.get(oid_i)
        if cir is not None:
            ax.scatter([cir[0]], [cir[1]], marker="o", c=[col], s=70,
                        edgecolors="white", linewidths=1.2, zorder=6)
            ax.text(cir[0] + 5, cir[1] - 5,
                    f"pid:{pid} ({label_by_pid.get(pid, '?')})",
                    color=col, fontsize=7, weight="bold", zorder=6)
        if sq is not None:
            sq_c = _clamp_to_inset(sq)
            ax.scatter([sq_c[0]], [sq_c[1]], marker="s", c=[col],
                        s=80, edgecolors=edge_col, linewidths=edge_lw,
                        zorder=6)
            ax.text(sq_c[0] + 5, sq_c[1] + 11,
                    f"oid:{oid_i} ({label_by_oid.get(oid_i, '?')})",
                    color=col, fontsize=7, weight="bold", zorder=6)
        if cir is not None and sq is not None:
            sq_c = _clamp_to_inset(sq)
            ax.plot([cir[0], sq_c[0]], [cir[1], sq_c[1]], "--",
                    color=col, lw=1.2, alpha=0.85, zorder=5)
            d2 = d2_by_oid.get(oid_i)
            if d2 is not None and d2 == d2:
                mx = (cir[0] + sq_c[0]) * 0.5
                my = (cir[1] + sq_c[1]) * 0.5
                ax.text(mx, my, f"d²t={d2:.1f}",
                        color=col, fontsize=7, weight="bold",
                        ha="center", va="center",
                        bbox=dict(facecolor="white",
                                   edgecolor="none", alpha=0.55,
                                   pad=1.0),
                        zorder=7)
        matched_oids.add(oid_i)

    # ── Unmatched detections — gray circle ────────────────────────
    for pid, p in det_pix.items():
        if pid in pid_to_oid:
            continue
        if p is None:
            continue
        ax.scatter([p[0]], [p[1]], marker="o",
                    c=[(0.55, 0.55, 0.55)], s=50,
                    edgecolors="white", linewidths=0.8,
                    alpha=0.75, zorder=5)
        ax.text(p[0] + 5, p[1] - 5,
                f"pid:{pid} ({label_by_pid.get(pid, '?')}) [no-track]",
                color=(0.55, 0.55, 0.55), fontsize=7, weight="bold",
                zorder=5)

    # ── Unmatched tracker oids — open square + label ──────────────
    for oid, p in trk_pix.items():
        if oid in matched_oids:
            continue
        if p is None:
            continue
        p_c = _clamp_to_inset(p)
        col = _palette_color_f(oid)
        edge_col = "yellow" if oid in held_set else col
        edge_lw = 2.4 if edge_col == "yellow" else 2.0
        ax.scatter([p_c[0]], [p_c[1]], marker="s",
                    facecolors="none", edgecolors=edge_col,
                    s=110, linewidths=edge_lw, zorder=6)
        ax.text(p_c[0] + 5, p_c[1] + 11,
                f"oid:{oid} ({label_by_oid.get(oid, '?')}) [no-match]",
                color=col, fontsize=7, weight="bold", zorder=6)


def _plot_step3_hungarian_cost(ax, dbg):
    """Step 3a: total Hungarian cost (cost_matrix). Matched cell in red."""
    assoc = dbg.get("assoc", {})
    C = np.asarray(assoc.get("cost_matrix", []), dtype=np.float64)
    _draw_hungarian_matrix(
        ax, dbg, C,
        "[3] Hungarian total cost  (red = matched)",
        highlight_matches=True, value_fmt=".1f",
    )


def _plot_step3_hungarian_d2_trans(ax, dbg):
    """Step 3b: translation-only Mahalanobis d²_trans (the gate metric)."""
    assoc = dbg.get("assoc", {})
    Dt = np.asarray(assoc.get("d2_trans_matrix", []), dtype=np.float64)
    _draw_hungarian_matrix(
        ax, dbg, Dt,
        "[3] d²_trans  (centroid Mahalanobis; gate)",
        highlight_matches=False, value_fmt=".2f",
    )


def _plot_step3_hungarian_adjust(ax, dbg):
    """Step 3c: non-Mahalanobis cost contributions = cost − d²_trans.
    Captures SAM2 continuity bonus (negative), label penalty (positive),
    and score penalty (positive)."""
    assoc = dbg.get("assoc", {})
    C = np.asarray(assoc.get("cost_matrix", []), dtype=np.float64)
    Dt = np.asarray(assoc.get("d2_trans_matrix", []), dtype=np.float64)
    if C.ndim == 2 and Dt.ndim == 2 and C.shape == Dt.shape and C.size:
        # Adjustment only defined on feasible cells.
        feasible = (C < 1e11) & np.isfinite(Dt)
        adj = np.where(feasible, C - Dt, np.nan)
    else:
        adj = np.zeros((0, 0), dtype=np.float64)
    _draw_hungarian_matrix(
        ax, dbg, adj,
        "[3] adjustments  (cost − d²_trans)",
        highlight_matches=False, value_fmt=".2f",
    )


def _plot_step3b_icp_points(ax, dbg, detections, dets_with_pose,
                             depth, T_wb, K):
    """Step 3b: ICP 3D point clouds per matched pair, colored by match.

    For each matched (track, detection) pair we show two clouds in the
    same track-palette color (no other symbols):

      * detection cloud -- back-projected from (mask, depth) into the
        camera frame, then into world via T_wc.
      * reference cloud -- the track's accumulated object-local surface,
        transformed by the ICP-aligned T_co and then into world via T_wc.

    When the two overlap (good alignment), you see a dense, tightly
    co-located blob of the same color; disagreement shows up as two
    same-color clusters that do not overlap.
    """
    _set_topdown_axes(ax, "[3b] ICP 3D clouds  (same color = matched)")
    _draw_base_marker(ax, T_wb, color="lightgray")

    T_bc = np.asarray(dbg.get("T_bc", np.eye(4)))
    T_wc = T_wb @ T_bc
    match_global = dbg.get("assoc", {}).get("match", {})
    track_refs = dbg.get("track_refs", {}) or {}

    def _to_world(pts_cam):
        if pts_cam.size == 0:
            return pts_cam
        return pts_cam @ T_wc[:3, :3].T + T_wc[:3, 3]

    for oid_str, det_gi in match_global.items():
        oid = int(oid_str)
        det_gi = int(det_gi)
        if det_gi >= len(dets_with_pose):
            continue
        dwp = dets_with_pose[det_gi]
        det_raw = detections[det_gi] if det_gi < len(detections) else None
        col = _palette_color_f(oid)

        # Detection cloud.
        if det_raw is not None and depth is not None:
            pts_cam = _backproject_mask(det_raw.get("mask"), depth, K)
            pts_w = _to_world(pts_cam)
            if len(pts_w) > 0:
                ax.scatter(pts_w[:, 1], pts_w[:, 0], s=1.5, c=[col],
                           alpha=0.35, marker=".", linewidths=0, zorder=4)

        # Reference cloud via T_co.
        ref = np.asarray(track_refs.get(oid, []), dtype=np.float64)
        if ref.ndim == 2 and ref.shape[0] > 0 and dwp.get("_icp_ok"):
            T_co = np.asarray(dwp["T_co"], dtype=np.float64)
            pts_cam2 = ref @ T_co[:3, :3].T + T_co[:3, 3]
            if len(pts_cam2) > 800:
                idx = np.linspace(0, len(pts_cam2) - 1, 800).astype(np.int64)
                pts_cam2 = pts_cam2[idx]
            pts_w2 = _to_world(pts_cam2)
            if len(pts_w2) > 0:
                ax.scatter(pts_w2[:, 1], pts_w2[:, 0], s=2.5, c=[col],
                           alpha=0.9, marker=".", linewidths=0, zorder=5)


def _plot_step4_visibility(ax, dbg, depth):
    """Step 4: per-track p_v (depth-raytrace) as a horizontal bar chart.

    Fixed x range [0, 1], padded to at least 5 rows so frame-to-frame
    scale never jumps.
    """
    p_v = dbg.get("visibility", {}) or {}
    oids_sorted = sorted(int(o) for o in p_v.keys())
    display_oids: List[Any] = list(oids_sorted)
    display_vals: List[float] = [float(p_v[o]) for o in oids_sorted]
    display_colors: List[Any] = [_palette_color_f(o) for o in oids_sorted]

    PAD_MIN = 5
    while len(display_oids) < PAD_MIN:
        display_oids.append(None)
        display_vals.append(0.0)
        display_colors.append((0.88, 0.88, 0.88))

    y_pos = np.arange(len(display_oids))
    ax.barh(y_pos, display_vals, color=display_colors,
            edgecolor="black", linewidth=0.5)
    # Label the y-ticks ``oid:Y (label)`` for parity with the row-1
    # tracking panel + events list. Track labels come from
    # ``post_update_tracks`` (best-known); falls back to "?" if missing.
    track_labels = {
        int(oid): tr.get("label", "?")
        for oid, tr in (dbg.get("post_update_tracks") or {}).items()
    }
    ax.set_yticks(y_pos)
    ax.set_yticklabels([
        (f"oid:{o} ({track_labels.get(int(o), '?')})"
         if o is not None else "—")
        for o in display_oids
    ], fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.axvline(0.5, linestyle=":", color="gray", alpha=0.6)
    ax.set_xlabel("p_v", fontsize=9)
    ax.set_title("[4] Visibility  (ray-traced depth)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    for y, v, o in zip(y_pos, display_vals, display_oids):
        if o is not None:
            ax.text(min(v + 0.02, 0.98), y, f"{v:.2f}",
                    va="center", fontsize=7)
    ax.invert_yaxis()


def _plot_events_list(ax, dbg, detections=None):
    """Steps 6/7/8: per-frame events as a text list ``oid ↔ pid: event``.

    The ``oid`` is the tracker's stable id; ``pid`` is the perception
    id of the detection involved (mirrors the ``p:X o:Y`` labels in
    the centroid panel so the two panels cross-reference directly).

    Each line is colored by the tracker's palette so it lines up with
    the same-track artifact in the spatial panels.
    """
    ax.axis("off")
    ax.set_title("[6/7/8] Events  (line 1: identity · line 2: detail)",
                 fontsize=10)

    detections = detections or []
    # events[i] = (sort_key, line1, line2, color)
    events: List[Tuple[int, str, str, Any]] = []

    assoc = dbg.get("assoc") or dbg.get("association") or {}
    track_oids = assoc.get("track_oids") or []
    track_labels = assoc.get("track_labels") or []
    label_by_oid: Dict[int, str] = {}
    for i, o in enumerate(track_oids):
        try:
            label_by_oid[int(o)] = (str(track_labels[i])
                                     if i < len(track_labels) else "?")
        except (TypeError, ValueError):
            continue

    def _pid_str(p):
        if p is None or (isinstance(p, int) and p < 0):
            return "  —"
        return f"{int(p):>3}"

    def _track_lbl(oid):
        return label_by_oid.get(int(oid), "?")

    # Matched events: ↔ means a real Hungarian match this frame.
    for ev in dbg.get("matched", []) or []:
        oid = int(ev.get("oid", -1))
        pid = ev.get("pid")
        lbl = _track_lbl(oid)
        line1 = f"oid {oid:>3} ↔ pid {_pid_str(pid)}  {lbl}"
        if ev.get("reject_outer_gate"):
            line2 = "miss-gate"
        elif ev.get("reject_held_prefilter"):
            line2 = (f"held-skip  d="
                     f"{ev.get('held_meas_err_m', 0.0):.2f}m "
                     f"(>{ev.get('held_meas_radius_m', 0.0):.2f}m)")
        elif ev.get("reject_innov_clamp"):
            line2 = (f"held-innov  d="
                     f"{ev.get('innov_dist_m', 0.0):.2f}m "
                     f"(>{ev.get('innov_max_m', 0.0):.2f}m)")
        else:
            d2 = ev.get("d2_trans", float("nan"))
            line2 = f"hit  d²t={d2:.2f}"
        events.append((oid, line1, line2, _palette_color_f(oid)))

    # Missed-update events: NO match this frame. The `pid` field is the
    # SAM2 continuity bookmark (`sam2_tau[oid]`), i.e. the SAM2 id this
    # track was last associated with -- NOT a current correspondence.
    # We render it as `tau:N` (or `—` when never bookmarked) and use `·`
    # instead of `↔` so it is visually distinct from a real match.
    for ev in dbg.get("missed", []) or []:
        oid = int(ev.get("oid", -1))
        tau = ev.get("pid")     # historical sam2_tau; misleading key name
        lbl = _track_lbl(oid)
        tau_str = (_pid_str(tau).strip() if tau is not None else "—")
        if tau_str in ("", "-1"):
            tau_str = "—"
        line1 = f"oid {oid:>3} ·  tau:{tau_str:>3}  {lbl}"
        pv = ev.get("p_v", 1.0)
        reason = ev.get("reason", "?")
        best_d2 = ev.get("best_d2_trans")
        gate = ev.get("gate")
        detail = ""
        if best_d2 is not None and gate is not None:
            try:
                bd = float(best_d2); gv = float(gate)
                if bd == bd:  # not NaN
                    bp = ev.get("best_pid")
                    bp_s = (str(int(bp)) if bp is not None else "—")
                    detail = (f" best d²t={bd:.1f} (G={gv:.1f}) "
                              f"pid={bp_s}")
            except (TypeError, ValueError):
                pass
        line2 = f"miss  p_v={pv:.2f}{detail}  [{reason}]"
        events.append((oid, line1, line2, (0.45, 0.45, 0.45)))

    # Births.
    for br in dbg.get("births", []) or []:
        new_oid = br.get("new_oid")
        if new_oid is None:
            continue
        oid = int(new_oid)
        pid = br.get("pid")
        lbl = br.get("label", _track_lbl(oid))
        line1 = f"oid {oid:>3} ↔ pid {_pid_str(pid)}  {lbl}"
        line2 = "birth (admit)"
        events.append((oid, line1, line2, _palette_color_f(oid)))

    # Birth rejects — no oid yet, sort by pid.
    for br in dbg.get("birth_rejects", []) or []:
        pid = br.get("pid")
        sort_key = (10**6 + (int(pid) if pid is not None else 999))
        lbl = br.get("label", "?")
        line1 = f"oid   — ↔ pid {_pid_str(pid)}  {lbl}"
        reason = br.get("reason", "?")
        if reason == "confirm":
            n_obs = br.get("n_obs_tracker", 0)
            k = br.get("confirm_k", "?")
            line2 = f"rej:confirm  n_obs={n_obs} (need={k})"
        elif reason == "score":
            s = br.get("tracker_max_score", br.get("score", 0.0))
            mn = br.get("score_min", 0.0)
            line2 = (f"rej:score  s={float(s):.2f} "
                     f"(need ≥{float(mn):.2f})")
        elif reason == "near_live":
            d = br.get("dist_m", 0.0)
            g = br.get("gate_m", 0.0)
            no = br.get("nearest_oid", "?")
            anc = br.get("anchor", "")
            anc_s = f" via {anc}" if anc else ""
            line2 = (f"rej:near_live  d={float(d):.2f}m "
                     f"(gate={float(g):.2f}, near oid {no}{anc_s})")
        elif reason == "border":
            box = br.get("box")
            mp = br.get("margin_px", "?")
            box_s = ""
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    box_s = (f" box=({int(box[0])},{int(box[1])},"
                             f"{int(box[2])},{int(box[3])})")
                except (TypeError, ValueError):
                    pass
            line2 = f"rej:border  margin={mp}{box_s}"
        else:
            line2 = f"rej:{reason}"
        events.append((sort_key, line1, line2, (0.55, 0.55, 0.55)))

    # Prunes.
    for pr in dbg.get("pruned", []) or []:
        oid = int(pr.get("oid", -1))
        r_val = float(pr.get("r", 0.0))
        lbl = _track_lbl(oid)
        line1 = f"oid {oid:>3}        {lbl}"
        line2 = f"prune (r={r_val:.2f})"
        events.append((oid, line1, line2, (0.85, 0.2, 0.2)))

    # Self-merges (drop → keep).
    for mg in dbg.get("self_merges", []) or []:
        keep, drop = mg.get("keep_oid"), mg.get("drop_oid")
        if keep is None or drop is None:
            continue
        d_oid = int(drop)
        lbl = _track_lbl(d_oid)
        line1 = f"oid {d_oid:>3}        {lbl}"
        line2 = f"merge → oid {int(keep)}"
        events.append((d_oid, line1, line2, (0.80, 0.20, 0.80)))

    # Subpart-absorbed (no oid; pid pair).
    for sa in dbg.get("subpart_absorbed", []) or []:
        try:
            from_idx = int(sa.get("from_idx"))
            into_idx = int(sa.get("into_idx"))
            pid_from = detections[from_idx].get("id")
            pid_into = detections[into_idx].get("id")
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if pid_from is None or pid_into is None:
            continue
        cont = float(sa.get("containment", 0.0))
        lbl_from = sa.get("from_label", "?")
        lbl_into = sa.get("into_label", "?")
        sort_key = 10**6 + int(pid_from)
        line1 = (f"pid {_pid_str(pid_from)} → "
                 f"pid {_pid_str(pid_into)}  {lbl_from}→{lbl_into}")
        line2 = f"absorb  c={cont:.2f}"
        events.append((sort_key, line1, line2, (0.55, 0.45, 0.30)))

    # Centroid drops.
    for cd in dbg.get("centroid_dropped", []) or []:
        pid = cd.get("pid")
        reason = cd.get("reason", "?")
        lbl = cd.get("label", "?")
        sort_key = 10**6 + (int(pid) if pid is not None else 999)
        line1 = f"oid   — ↔ pid {_pid_str(pid)}  {lbl}"
        line2 = f"cent-drop ({reason})"
        events.append((sort_key, line1, line2, (0.55, 0.55, 0.55)))

    events.sort(key=lambda x: x[0] if x[0] is not None else 10**9)

    if not events:
        ax.text(0.5, 0.5, "(no events)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        return

    y_start = 0.97
    y_step_event = 0.062
    line2_indent = 0.06
    line2_drop = 0.026
    max_events = 15
    n_drawn_events = 0
    for i, (_k, l1, l2, col) in enumerate(events[:max_events]):
        y1 = y_start - i * y_step_event
        if y1 < 0.04:
            break
        ax.text(0.03, y1, l1, transform=ax.transAxes,
                fontsize=8, color=col, family="monospace",
                va="top", weight="bold")
        ax.text(0.03 + line2_indent, y1 - line2_drop, l2,
                transform=ax.transAxes,
                fontsize=7, color=col, family="monospace", va="top")
        n_drawn_events += 1

    # ── Relations block (LLM scene graph) ─────────────────────────
    # Build per-oid "supporters" map from dbg["relations"].
    # Convention: parent = supported, child = supporter — so the
    # children of edges with our parent are the things this oid sits on.
    supporters: Dict[int, List[int]] = {}
    for rel in dbg.get("relations", []) or []:
        try:
            p_oid = int(rel["parent"])
            c_oid = int(rel["child"])
            rt = str(rel.get("type", "on"))
        except (KeyError, TypeError, ValueError):
            continue
        if rt not in ("on", "in"):
            continue
        supporters.setdefault(p_oid, []).append(c_oid)

    if supporters:
        rel_color = (0.10, 0.45, 0.50)   # dark teal
        # Start the block one event-row below the last event.
        y_rel = max(0.04 + 0.05,
                    y_start - n_drawn_events * y_step_event - 0.015)
        # Title divider.
        ax.text(0.03, y_rel,
                "── Relations (LLM) ──",
                transform=ax.transAxes,
                fontsize=8, color=rel_color, family="monospace",
                va="top", weight="bold")
        y_rel -= 0.034
        # One line per supported oid (sorted).
        max_rel_lines = 8
        rel_lines_drawn = 0
        for p_oid in sorted(supporters.keys()):
            if rel_lines_drawn >= max_rel_lines or y_rel < 0.02:
                break
            kids = sorted(set(supporters[p_oid]))
            kids_str = ", ".join(str(k) for k in kids)
            lbl = label_by_oid.get(p_oid, "?")
            ax.text(0.03, y_rel,
                    f"oid {p_oid:>3} ({lbl})  on: {kids_str}",
                    transform=ax.transAxes,
                    fontsize=8, color=rel_color, family="monospace",
                    va="top")
            y_rel -= 0.034
            rel_lines_drawn += 1


def render_frame(rgb: np.ndarray,
                 detections: List[Dict[str, Any]],
                 dbg: Dict[str, Any],
                 dets_with_pose: List[Dict[str, Any]],
                 r_history: Dict[int, List[Tuple[int, float]]],
                 frame_idx: int,
                 max_frame: int,
                 out_path: str,
                 traj: str,
                 depth: Optional[np.ndarray] = None,
                 K: Optional[np.ndarray] = None) -> None:
    """Nine-panel per-frame visualisation, one concept per panel.

    Layout (3×3):
        [0 SLAM]           [1 Predict]         [2 Coarse mask/centroid]
        [3 Hungarian cost] [3 d²_trans]        [3 adjustments]
        [3b ICP 3D points] [4 Visibility]      [6/7/8 Events id→event]
    """
    T_wb = dbg["slam_pose"]
    K_use = K if K is not None else K_DEFAULT

    # 4-row layout, row 4 spans all 3 cols and is taller for the
    # accumulated point-cloud panel.
    fig = plt.figure(figsize=(19, 22), dpi=100)
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[1.0, 1.0, 1.0, 1.6],
        hspace=0.32, wspace=0.26,
        left=0.05, right=0.98, top=0.96, bottom=0.04,
    )

    # Row 1 — Detected (pid)  |  Tracked (oid)  |  Events (oid ↔ pid)
    _plot_detected_pid(fig.add_subplot(gs[0, 0]), rgb, detections, dbg)
    _plot_tracked_oid (fig.add_subplot(gs[0, 1]), rgb, dets_with_pose, dbg,
                        K=K_use)
    _plot_events_list (fig.add_subplot(gs[0, 2]), dbg, detections=detections)

    # Row 2 — Hungarian matrices
    _plot_step3_hungarian_cost    (fig.add_subplot(gs[1, 0]), dbg)
    _plot_step3_hungarian_d2_trans(fig.add_subplot(gs[1, 1]), dbg)
    _plot_step3_hungarian_adjust  (fig.add_subplot(gs[1, 2]), dbg)

    # Row 3 — SLAM, EKF predict, visibility
    _plot_step0_slam      (fig.add_subplot(gs[2, 0]), dbg, T_wb)
    _plot_step1_predict   (fig.add_subplot(gs[2, 1]), dbg, T_wb)
    _plot_step4_visibility(fig.add_subplot(gs[2, 2]), dbg, depth)

    # Row 4 — wide accumulated point-cloud panel (spans all 3 cols)
    _plot_step3b_icp_points(fig.add_subplot(gs[3, :]),
                             dbg, detections, dets_with_pose,
                             depth, T_wb, K_use)

    fig.suptitle(
        f"EKF tracking pipeline — traj={traj}   frame={frame_idx:04d}   "
        f"base=({T_wb[0,3]:.2f},{T_wb[1,3]:.2f})",
        fontsize=13, y=0.985,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_in_the_tray")
    ap.add_argument("--max-frame", type=int, default=700,
                    help="exclusive upper bound on frame index")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--pose-method", default="icp_chain",
                    choices=("centroid", "icp_chain", "icp_anchor",
                             "icp_chain_strict", "icp_anchor_strict"))
    ap.add_argument("--out-subdir", default="ekf_debug")
    ap.add_argument("--state-subdir", default="ekf_state",
                    help="dir for per-frame JSON state dumps "
                         "(under tests/visualization_pipeline/{traj}/)")
    ap.add_argument("--no-png", action="store_true",
                    help="skip 5-panel PNG rendering (only dump JSON state)")
    ap.add_argument("--no-mp4", action="store_true",
                    help="skip composing the per-frame PNGs into an MP4")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="frame rate for the composed MP4 (default 10)")
    args = ap.parse_args()

    traj = args.trajectory
    ds_root = os.path.join(DATASET_DIR, traj)
    viz_root = os.path.join(VIZ_BASE, traj)
    rgb_dir = os.path.join(ds_root, "rgb")
    depth_dir = os.path.join(ds_root, "depth")
    det_dir = os.path.join(viz_root, "perception", "detection_h")
    if not os.path.isdir(det_dir):
        # Fall back to the dataset-side perception output. apple_in_the_tray
        # has a hand-curated cache under tests/visualization_pipeline/...;
        # the other trajectories ship detections directly in datasets/.
        ds_det = os.path.join(ds_root, "detection_h")
        if os.path.isdir(ds_det):
            det_dir = ds_det
    pose_path = os.path.join(ds_root, "pose_txt", "amcl_pose.txt")
    out_dir = os.path.join(viz_root, args.out_subdir)
    state_dir = os.path.join(viz_root, args.state_subdir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    # ── load poses ──
    slam_poses = _load_amcl_poses(pose_path)
    if not slam_poses:
        raise SystemExit(f"no AMCL poses loaded from {pose_path}")
    # Per-frame base-to-camera-optical extrinsic (head pan/tilt/torso lift).
    # Falls back to None if the dataset wasn't extracted with T_bc support;
    # the tracker then keeps its construction-time T_bc (default identity).
    T_bc_path = os.path.join(ds_root, "pose_txt", "T_bc.txt")
    T_bc_map = _load_T_bc_poses(T_bc_path)
    if T_bc_map is None:
        print(f"[warn] no T_bc.txt at {T_bc_path}; using identity. "
              f"Re-extract the rosbag to enable head-motion-aware tracking.")
    else:
        print(f"[T_bc] loaded {len(T_bc_map)} per-frame extrinsics from "
              f"{T_bc_path}")

    # Per-frame gripper-in-base and gripper width. Without these, the
    # tracker cannot distinguish a moving held object from a world-static
    # one — the Hungarian gate fails and duplicates birth.
    T_bg_map = _load_ee_poses(os.path.join(ds_root, "pose_txt", "ee_pose.txt"))
    gripper_w_map = _load_gripper_widths(
        os.path.join(ds_root, "pose_txt", "joints_pose.json"))
    if T_bg_map is not None:
        print(f"[T_bg] loaded {len(T_bg_map)} per-frame EE poses")
    else:
        print(f"[warn] no ee_pose.txt; held-object tracking disabled")
    if gripper_w_map is not None:
        print(f"[grip] loaded {len(gripper_w_map)} per-frame finger widths")
    else:
        print(f"[warn] no joints_pose.json; phase will remain 'idle'")

    # Raw per-frame joints dict (needed by GraspOwnerDetector to call
    # the gripper geometry's `state_from_joints` with real l/r values).
    joints_path = os.path.join(ds_root, "pose_txt", "joints_pose.json")
    joints_map: Optional[Dict[int, Dict[str, float]]] = None
    if os.path.exists(joints_path):
        with open(joints_path) as f:
            raw = json.load(f)
        joints_map = {int(k): v for k, v in raw.items()}

    # Construct the robot-agnostic grasp-owner detector.
    gripper_geom = create_gripper_geometry(robot_type="fetch")
    grasp_detector = GraspOwnerDetector(gripper=gripper_geom)
    print(f"[grasp] {gripper_geom.describe()}")
    grip_inferrer = _GripperStateInferrer(detector=grasp_detector)

    # Construct the relation pipeline (LLM by default; opt out with
    # env var EKF_VIZ_RELATION_BACKEND=none for fast / offline runs).
    # The cache lives per-trajectory so re-runs on the same dataset
    # replay LLM responses from disk for free.
    relation_backend = os.environ.get("EKF_VIZ_RELATION_BACKEND", "llm")
    relation_cache_dir = os.environ.get(
        "EKF_VIZ_RELATION_CACHE_DIR",
        os.path.join(viz_root, "relation_cache"))
    relation_pipeline = _RelationPipeline(backend=relation_backend,
                                            cache_dir=relation_cache_dir)
    print(f"[relation] backend={relation_backend}  "
          f"trigger=on_grasp/on_release/on_new_object + every "
          f"{relation_pipeline._cfg.relation_every_n_frames} frames")

    # ── tracker setup ──
    cfg = BernoulliConfig(
        association_mode="hungarian",
        p_s=1.0,
        p_d=0.9,
        alpha=4.4,
        lambda_c=1.0,
        lambda_b=1.0,
        r_conf=0.5,
        r_min=1e-3,
        G_in=12.59,
        G_out=25.0,
        P_max=np.diag([0.25**2] * 3 + [(np.pi / 4) ** 2] * 3),
        enable_visibility=True,
        enable_huber=True,
        init_cov_from_R=False,        # fixed 5 cm / 0.05 rad at birth (vs ICP's R)
        # Perception-style soft cost on labels + score (mirrors
        # rosbag2dataset/sam2/sam2_client._pair_cost). The hard label
        # gate is OFF; a noisy label disagreement adds 6.0 chi^2_6 units
        # to the cost (less than the outer gate, so a geometrically
        # excellent pair can still match) and a per-detection score
        # bias of 2.0*(1-score) mildly disfavours low-confidence dets.
        enforce_label_match=False,
        hungarian_label_penalty=6.0,
        hungarian_score_weight=2.0,
        # Translation-only outer gate: per-oid ICP rotation chains
        # drift independently and can lift d^2_rot above the chi^2_6
        # gate even at sub-mm translation match (frame-485 case:
        # d^2_trans=0.02, d^2_rot=112). Trans-only gate at chi^2_3
        # 0.9997 = 21.108 fixes that. Cost still uses 'sum' so a
        # rotation match gets a tie-break.
        gate_mode="trans",
        G_out_trans=21.108,
        cost_d2_mode="sum",
        # Floor P_bo per axis -- prevents the EKF posterior from
        # shrinking below realistic per-frame perception jitter
        # (5 mm trans / 0.05 rad rot). Without this, a track with
        # 10k+ observations rejects 2-3 cm jitter on the next frame
        # via the chi^2_3 gate (frame-430 case).
        P_min_diag=np.array([0.005**2] * 3 + [0.05**2] * 3),
        # Track-to-track self-merge after each step: catches duplicate
        # tracks of the same physical object that survived a one-to-one
        # Hungarian round (e.g. SAM2 ids splitting). Euclidean gate in
        # metres — ≈ one object radius for apples / cups / cans. Does NOT
        # scale with covariance, so two fresh births can't collapse just
        # because σ starts at 5 cm.
        self_merge_trans_m=0.05,
        K=K_DEFAULT,
        image_shape=(480, 640),
    )
    tracker = GaussianEkfTracker(K_DEFAULT, cfg, pose_method=args.pose_method)

    # ── voxel observability (for gravity-aware predict at release) ──
    # Coarse 5 cm grid covering 5 m × 5 m × 3 m around origin; sufficient
    # for the apple_drop / apple_in_the_tray / apple_to_cabinate scenes.
    voxel_obs = VoxelObservability(
        voxel_size_m=0.05,
        workspace_aabb=((-2.5, -2.5, -1.0), (2.5, 2.5, 2.0)),
        n_min_hit=2, n_min_pass=3,
    )
    # Phase-transition tracking for the gravity-predict hook.
    prev_phase = "idle"
    prev_held: Optional[int] = None
    gravity_predict_log: List[Dict[str, Any]] = []

    # ── per-track r(t) history ──
    r_history: Dict[int, List[Tuple[int, float]]] = {}

    max_frame = min(args.max_frame, len(slam_poses))
    frames_processed = 0
    frames_written = 0
    for idx in range(args.start, max_frame):
        if (idx - args.start) % args.step != 0:
            continue

        rgb = _load_rgb(os.path.join(rgb_dir, f"rgb_{idx:06d}.png"))
        depth = _load_depth(os.path.join(depth_dir, f"depth_{idx:06d}.npy"))
        if rgb is None or depth is None:
            continue

        detections = _load_detection_json(
            os.path.join(det_dir, f"detection_{idx:06d}_final.json"))

        T_wb = slam_poses[idx]
        T_bc_now = T_bc_map.get(idx) if T_bc_map is not None else None

        # ── Voxel observability: integrate this frame's depth ─────
        # T_cw maps camera-frame points to world: T_wc = T_wb @ T_bc.
        T_bc_for_vox = T_bc_now if T_bc_now is not None else np.eye(4)
        try:
            voxel_obs.integrate_depth(
                depth=depth.astype(np.float32),
                K=K_DEFAULT,
                T_cw=np.asarray(T_wb, dtype=np.float64) @ np.asarray(T_bc_for_vox, dtype=np.float64),
                max_range_m=3.0,
                subsample=4,
            )
        except Exception as e:
            print(f"[WARN] voxel_obs.integrate_depth failed at fr {idx}: {e}")
        T_bg_now = T_bg_map.get(idx) if T_bg_map is not None else None
        w_now = gripper_w_map.get(idx) if gripper_w_map is not None else None
        joints_now = joints_map.get(idx) if joints_map is not None else None

        # Infer gripper state (phase + held oid) from proprioception.
        # Runs BEFORE predict so the predict step gets the correct
        # phase-dependent Q and rigid-attachment handling.
        # detections + depth + K + T_bc + joints are passed for the
        # geometric grasp-selector at grasp onset.
        gripper_state = grip_inferrer.step(
            width=w_now, tracker=tracker, T_wb=T_wb, T_bg=T_bg_now,
            detections=detections, depth=depth, K=K_DEFAULT,
            T_bc=T_bc_now, joints=joints_now)
        held_seed = gripper_state.get("held_obj_id")
        # Releasing: the gripper just opened, so the held object has
        # been let go. Skip rigid-attachment predict during this
        # transition window — otherwise the released object continues
        # to ride the base for `min_transition_frames` until the FSM
        # reaches `idle` and clears `held_obj_id`. The FSM keeps the
        # id internally so events / timeline still log "release at
        # frame N for oid X".
        if gripper_state.get("phase") == "releasing":
            held_seed = None

        # Maybe re-run the relation backend (throttled by trigger gate).
        # Build per-detection-idx → tracker-oid map from the live tracks'
        # sam2_tau (each track stores the perception id it was last
        # matched to). This lets the relation client reason about
        # *tracker* identities rather than per-frame perception ids.
        det_to_oid: Dict[int, int] = {}
        tau_to_oid = {int(t): int(o) for o, t in tracker.sam2_tau.items()
                       if t is not None}
        for di, d in enumerate(detections):
            pid = d.get("id")
            if pid is None:
                continue
            oid = tau_to_oid.get(int(pid))
            if oid is not None:
                det_to_oid[di] = oid
        live_oids = {int(o) for o in tracker.object_labels.keys()}
        # Build per-track world-frame xyz + label snapshot for the
        # geometric-support fallback inside the relation pipeline.
        live_tracks: Dict[int, Dict[str, Any]] = {}
        for oid in live_oids:
            pe = tracker.state.collapsed_object_base(int(oid))
            if pe is None:
                continue
            mu_b = np.asarray(pe.T, dtype=np.float64)[:3, 3]
            mu_w = (T_wb @ np.append(mu_b, 1.0))[:3]
            live_tracks[int(oid)] = {
                "xyz_w": mu_w.tolist(),
                "label": tracker.object_labels.get(int(oid), "?"),
            }
        rel_summary = relation_pipeline.maybe_update(
            frame=idx, rgb=rgb, detections=detections,
            det_to_oid=det_to_oid,
            current_phase=gripper_state["phase"],
            current_oids=live_oids,
            held_oid=held_seed,
            live_tracks=live_tracks)

        # Expand held_oids using the latest filtered relation graph.
        held_oids = expand_held_with_relations(
            held_seed, relation_pipeline.edges)
        # Drop any oids the tracker doesn't actually have (relation
        # graph might still reference a pruned/merged track).
        held_oids = {o for o in held_oids if o in tracker.state.objects}

        dbg, dets_with_pose = tracker.step(
            rgb=rgb, depth=depth, T_wb=T_wb,
            detections=detections,
            phase=gripper_state["phase"],
            T_bc=T_bc_now,
            T_bg=T_bg_now,
            held_oids=held_oids,
            held_seed=held_seed,
            relation_edges=relation_pipeline.edges,
        )
        # Re-map held_obj_id if self-merge renamed it.
        grip_inferrer.apply_merges(dbg.get("self_merges", []))
        gripper_state["held_obj_id"] = grip_inferrer._held_obj_id
        # Also remap the relation EMA so next frame's held-set
        # expansion still finds the right edges.
        relation_pipeline.remap_after_merges(dbg.get("self_merges", []))
        dbg["gripper_state"] = dict(gripper_state)

        # ── Gravity-aware predict at the release transition ──────
        # Detect the FSM exiting the {holding, releasing} window. When
        # that happens, replace the just-released oid's mean + cov with
        # the post-fall prediction (one-shot). voxel_obs supplies the
        # supporting surface; the parametric model in
        # ekf_tracker.manipulation.gravity_predict handles bounce/roll dispersion.
        cur_phase = gripper_state.get("phase", "idle")
        if (prev_phase in ("holding", "releasing")
                and cur_phase not in ("holding", "releasing")
                and prev_held is not None
                and prev_held in tracker.state.objects):
            try:
                pe_w = tracker.state.collapsed_object_world(int(prev_held))
                if pe_w is not None:
                    label = tracker.object_labels.get(int(prev_held))
                    dyn = lookup_dynamics(label)
                    # Live-object overlay: every other live oid contributes.
                    other_voxels = []
                    for other_oid, other_pe in (
                            tracker.state.collapsed_objects_world() or {}).items():
                        if other_oid == prev_held:
                            continue
                        T_o = other_pe.T
                        other_dyn = lookup_dynamics(
                            tracker.object_labels.get(int(other_oid)))
                        other_voxels.append((
                            float(T_o[0, 3]), float(T_o[1, 3]),
                            float(T_o[2, 3]), float(other_dyn.radius_m)))
                    T_land, P_land, info = predict_landing_pose(
                        T_release=pe_w.T,
                        P_release=pe_w.cov,
                        voxel_obs=voxel_obs,
                        dyn=dyn,
                        workspace_floor_z=-1.0,
                        live_object_voxels=other_voxels,
                    )
                    # Write back into base-frame state. The EKF stores
                    # objects in base frame; convert T_land back via T_wb.
                    obj = tracker.state.objects.get(int(prev_held))
                    if obj is not None:
                        T_wb_arr = np.asarray(T_wb, dtype=np.float64)
                        obj.mu_bo = np.linalg.inv(T_wb_arr) @ T_land
                        obj.cov_bo = P_land.copy()
                    log_entry = info.as_dict()
                    log_entry["oid"] = int(prev_held)
                    log_entry["frame"] = int(idx)
                    log_entry["label"] = label
                    gravity_predict_log.append(log_entry)
                    dbg["gravity_predict"] = log_entry
            except Exception as e:
                print(f"[WARN] gravity_predict failed at fr {idx}: {e}")
        # Update phase / held tracking for next iteration. prev_held is
        # the most-recently-known held oid: we keep the last non-None
        # value seen so the release-transition handler can identify
        # which oid was just dropped (the FSM clears held_obj_id when
        # it returns to idle).
        prev_phase = cur_phase
        held_now = gripper_state.get("held_obj_id")
        if held_now is not None:
            prev_held = int(held_now)

        # Expose the expanded held set + the relation snapshot for the
        # state JSON / mask panel highlight.
        dbg["held_oids_used"] = sorted(int(o) for o in held_oids)
        dbg["relations"] = [
            {"parent": int(e.parent), "child": int(e.child),
             "type": str(e.relation_type), "score": float(e.score)}
            for e in relation_pipeline.edges
        ]
        if rel_summary.get("fired"):
            dbg["relation_call"] = rel_summary
        frames_processed += 1

        for oid, tr in dbg["post_update_tracks"].items():
            r_history.setdefault(oid, []).append((idx, float(tr["r"])))

        # Always dump the JSON state (cheap; useful for offline diagnosis).
        state_path = os.path.join(state_dir, f"frame_{idx:06d}.json")
        try:
            _dump_frame_json(state_path, dbg, detections, dets_with_pose)
        except Exception as e:
            print(f"[WARN] state dump failed at frame {idx}: {e}")

        if not args.no_png:
            out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
            try:
                render_frame(
                    rgb=rgb, detections=detections, dbg=dbg,
                    dets_with_pose=dets_with_pose,
                    r_history=r_history,
                    frame_idx=idx,
                    max_frame=max_frame,
                    out_path=out_path,
                    traj=traj,
                    depth=depth,
                )
                frames_written += 1
            except Exception as e:
                print(f"[WARN] render failed at frame {idx}: {e}")
        else:
            frames_written += 1

        if frames_processed % 20 == 0:
            print(f"[{traj}] frame {idx}: processed {frames_processed}, "
                  f"written {frames_written}, tracks={len(tracker.object_labels)}")

    print(f"[done] wrote {frames_written} frames under {out_dir}")

    if not args.no_png and not args.no_mp4 and frames_written > 0:
        mp4_path = out_dir.rstrip("/\\") + ".mp4"
        try:
            _compose_frames_to_mp4(out_dir, mp4_path, fps=args.fps)
            print(f"[mp4] {mp4_path}")
        except Exception as e:
            print(f"[mp4] composition failed: {e}")


def _compose_frames_to_mp4(png_dir: str, out_path: str,
                            fps: float = 10.0) -> None:
    """Stitch every ``frame_*.png`` in ``png_dir`` into a single MP4.

    Uses ffmpeg's concat demuxer with an explicit file list so that
    sparse PNG sequences (e.g. when ``--step`` skipped frames) compose
    without holes. Output is H.264 / yuv420p so QuickTime, browsers,
    and VLC all play it.
    """
    import glob
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH; install via `brew install ffmpeg`")

    pngs = sorted(glob.glob(os.path.join(png_dir, "frame_*.png")))
    if not pngs:
        raise RuntimeError(f"no frame_*.png files in {png_dir}")

    dur = 1.0 / float(fps)
    with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                       delete=False) as fh:
        list_path = fh.name
        for p in pngs:
            fh.write(f"file '{os.path.abspath(p)}'\n")
            fh.write(f"duration {dur:.6f}\n")
        # ffmpeg concat demuxer requires the last file repeated.
        fh.write(f"file '{os.path.abspath(pngs[-1])}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-vsync", "vfr",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()

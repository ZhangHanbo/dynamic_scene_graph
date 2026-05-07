#!/usr/bin/env python3
"""
Visualize uncertainty through the two-tier orchestrator.

Panels:
  1. Scene view (top-down XY): object positions as colored dots with
     covariance ellipses drawn around them (semi-transparent, radius
     proportional to translation uncertainty).
  2. Entropy time-series: one line per object showing pose entropy
     (log-det of covariance). Manipulation phases as shaded bands,
     slow-tier triggers as dashed vertical lines.
  3. Alpha trace: adaptive kernel's α value over time (when slow tier fires).
  4. Phase ribbon: color-coded manipulation phase timeline.

Design philosophy — three scales of uncertainty:
  * Spatial: "where is each object right now, and how sure are we?"
  * Temporal: "how did each object's uncertainty evolve?"
  * Systemic: "when did the pose graph detect outliers?"

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python tests/visualize_uncertainty.py [--frames 328] [--step 2]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from utils.ekf_se3 import pose_entropy
from tests.test_orchestrator_integration import TrajectoryRunner, DATA_ROOT


# ─────────────────────────────────────────────────────────────────────
# Per-object consistent color palette (stable across frames)
# ─────────────────────────────────────────────────────────────────────

_COLOR_TABLE = np.array([
    [0.18, 0.80, 0.44],   # green
    [0.90, 0.30, 0.23],   # red
    [0.20, 0.60, 0.86],   # blue
    [0.95, 0.77, 0.06],   # yellow
    [0.61, 0.35, 0.71],   # purple
    [0.90, 0.49, 0.13],   # orange
    [0.10, 0.74, 0.61],   # teal
    [0.93, 0.44, 0.39],   # salmon
    [0.36, 0.68, 0.89],   # light blue
    [0.96, 0.82, 0.25],   # gold
])


def _color_for(obj_id: int) -> np.ndarray:
    return _COLOR_TABLE[obj_id % len(_COLOR_TABLE)]


# Phase colors (translucent)
_PHASE_COLORS = {
    "idle":       (0.85, 0.85, 0.85, 0.35),
    "grasping":   (1.00, 0.55, 0.00, 0.45),
    "holding":    (1.00, 0.20, 0.20, 0.25),
    "releasing":  (0.20, 0.40, 0.90, 0.45),
}


# ─────────────────────────────────────────────────────────────────────
# Spatial panel: covariance ellipses in XY
# ─────────────────────────────────────────────────────────────────────

def _translation_cov_2d(cov6: np.ndarray) -> np.ndarray:
    """Extract the XY block of the translation covariance."""
    # Our tangent-space ordering is [v, ω], so translation indices are 0,1,2
    return cov6[:2, :2]


def _ellipse_params(cov_2d: np.ndarray, n_sigma: float = 2.0
                    ) -> Tuple[float, float, float]:
    """Compute (width, height, angle_deg) for a 2-sigma confidence ellipse."""
    # Regularize near-singular covariance
    cov_2d = cov_2d + np.eye(2) * 1e-10
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    eigvals = np.clip(eigvals, 1e-10, None)
    # 2-sigma ellipse
    width = 2 * n_sigma * np.sqrt(eigvals[1])
    height = 2 * n_sigma * np.sqrt(eigvals[0])
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    return width, height, angle


def draw_scene_panel(ax, report: Dict, object_history: Dict[int, List[Tuple]]):
    """Top-down XY scene with covariance ellipses and object trails."""
    ax.clear()
    ax.set_aspect("equal")
    ax.set_title(f"Scene (top-down) — frame {report['frame_idx']}",
                 fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)

    # Collect all positions for auto-framing
    all_pos = []
    for oid, info in report["objects"].items():
        all_pos.append(info["T"][:2, 3])
        for _, pos, _ in object_history.get(oid, []):
            all_pos.append(pos[:2])
    if not all_pos:
        return
    all_pos = np.array(all_pos)
    pad = 0.3
    ax.set_xlim(all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ax.set_ylim(all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    held_id = report["gripper_state"].get("held_obj_id")

    for oid, info in report["objects"].items():
        pos = info["T"][:3, 3]
        cov2d = _translation_cov_2d(info["cov"])
        w, h, a = _ellipse_params(cov2d, n_sigma=2.0)
        color = _color_for(oid)

        # Trail
        history = object_history.get(oid, [])
        if len(history) >= 2:
            xs = [p[0] for _, p, _ in history]
            ys = [p[1] for _, p, _ in history]
            ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1)

        # Covariance ellipse
        ellipse = mpatches.Ellipse(
            (pos[0], pos[1]), w, h, angle=a,
            facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5)
        ax.add_patch(ellipse)

        # Object marker (bigger/bolder if held)
        marker = "*" if oid == held_id else "o"
        size = 220 if oid == held_id else 90
        ax.scatter([pos[0]], [pos[1]], c=[color], s=size, marker=marker,
                   edgecolors="black", linewidths=1.0, zorder=5)

        # Label with per-axis σ
        sigmas = np.sqrt(np.diag(cov2d))
        ax.annotate(
            f"[{oid}] {info['label']}\nσ=({sigmas[0]*100:.1f},{sigmas[1]*100:.1f})cm",
            (pos[0], pos[1]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=7, color=color,
            bbox=dict(boxstyle="round,pad=0.2",
                       facecolor="white", edgecolor=color, alpha=0.85))


# ─────────────────────────────────────────────────────────────────────
# Temporal panel: entropy time-series per object
# ─────────────────────────────────────────────────────────────────────

def draw_entropy_panel(ax, reports: List[Dict], current_local_idx: int,
                        object_entropy_history: Dict[int, List[Tuple]]):
    """Plot log-det of cov per object over time. Shade manipulation phases."""
    ax.clear()
    ax.set_title("Pose entropy (log-det Σ) per object over time", fontsize=11)
    ax.set_xlabel("frame index")
    ax.set_ylabel("entropy = ½ log det Σ")
    ax.grid(True, alpha=0.2)

    if not reports:
        return

    frame_indices = [r["frame_idx"] for r in reports]

    # Shade manipulation phases as contiguous bands
    prev_phase = None
    band_start = None
    for r in reports:
        phase = r["gripper_state"]["phase"]
        f = r["frame_idx"]
        if phase != prev_phase:
            if prev_phase is not None and prev_phase != "idle":
                ax.axvspan(band_start, f,
                           color=_PHASE_COLORS[prev_phase], alpha=0.3)
            band_start = f
            prev_phase = phase
    # Close the last band
    if prev_phase is not None and prev_phase != "idle":
        ax.axvspan(band_start, frame_indices[-1],
                   color=_PHASE_COLORS[prev_phase], alpha=0.3)

    # Plot entropy per object
    for oid, history in object_entropy_history.items():
        frames_o = [f for f, _ in history]
        ents = [e for _, e in history]
        color = _color_for(oid)
        ax.plot(frames_o, ents, color=color, linewidth=1.8, label=f"[{oid}]")

    # Dashed vertical line at current frame
    ax.axvline(reports[current_local_idx]["frame_idx"],
               color="black", linestyle="--", linewidth=1)

    # Triggers: short dashed green lines
    for r in reports[:current_local_idx + 1]:
        if r["triggered"]:
            ax.axvline(r["frame_idx"], color="green",
                       linestyle=":", linewidth=0.8, alpha=0.5)

    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.8)

    # Explicit clip on y-axis to hide outlier spikes
    all_ents = [e for hist in object_entropy_history.values() for _, e in hist]
    if all_ents:
        finite_ents = [e for e in all_ents if np.isfinite(e)]
        if finite_ents:
            y_lo = np.percentile(finite_ents, 2)
            y_hi = np.percentile(finite_ents, 98)
            pad = 0.1 * (y_hi - y_lo + 1)
            ax.set_ylim(y_lo - pad, y_hi + pad)


# ─────────────────────────────────────────────────────────────────────
# Systemic panel: alpha + residual bars at triggers
# ─────────────────────────────────────────────────────────────────────

def draw_alpha_panel(ax, reports: List[Dict], current_local_idx: int):
    """Adaptive kernel α over time + residual bars at triggers."""
    ax.clear()
    ax.set_title("Adaptive kernel α and residuals (at triggers)",
                 fontsize=11)
    ax.set_xlabel("frame index")
    ax.set_ylabel("α")
    ax.grid(True, alpha=0.2)

    triggered_frames, alphas = [], []
    for r in reports[:current_local_idx + 1]:
        if r["triggered"] and r["alpha"] is not None:
            triggered_frames.append(r["frame_idx"])
            alphas.append(r["alpha"])

    if triggered_frames:
        ax.plot(triggered_frames, alphas, "o-",
                color="purple", linewidth=1.2, markersize=4,
                label="α (fitted)")
        ax.axhline(2.0, color="gray", linestyle="--", alpha=0.4,
                   label="α=2 (L2, no outliers)")
        ax.axhline(0.0, color="gray", linestyle=":", alpha=0.4,
                   label="α=0 (Cauchy)")
        ax.axhline(-2.0, color="gray", linestyle=":", alpha=0.4,
                   label="α=-2 (Geman-McClure)")
        ax.set_ylim(-10.5, 2.5)
        ax.legend(loc="lower right", fontsize=7)

    # Overlay residual magnitudes as bars on twin axis
    ax2 = ax.twinx()
    ax2.set_ylabel("max residual (m)", fontsize=9, color="darkred")
    for r in reports[:current_local_idx + 1]:
        if not r["triggered"]:
            continue
        res = r["residuals"]
        # Collect all residuals from observation + relation + manipulation
        all_res = []
        for _, vals in res.items():
            if len(vals) > 0:
                all_res.extend(vals.tolist())
        if not all_res:
            continue
        max_res = max(all_res)
        ax2.bar(r["frame_idx"], max_res, width=1.5,
                alpha=0.3, color="darkred")
    ax2.set_ylim(0, None)


# ─────────────────────────────────────────────────────────────────────
# Phase ribbon at the top
# ─────────────────────────────────────────────────────────────────────

def draw_phase_ribbon(ax, reports: List[Dict], current_local_idx: int):
    """Horizontal timeline of manipulation phases."""
    ax.clear()
    ax.set_title("Manipulation phase timeline", fontsize=10)
    ax.set_yticks([])

    if not reports:
        return

    frame_idx_min = reports[0]["frame_idx"]
    frame_idx_max = reports[-1]["frame_idx"]
    ax.set_xlim(frame_idx_min, frame_idx_max)
    ax.set_ylim(0, 1)

    # Draw phase bands
    prev_phase = None
    band_start = None
    for r in reports:
        phase = r["gripper_state"]["phase"]
        f = r["frame_idx"]
        if phase != prev_phase:
            if prev_phase is not None:
                ax.axvspan(band_start, f, color=_PHASE_COLORS[prev_phase])
            band_start = f
            prev_phase = phase
    if prev_phase is not None:
        ax.axvspan(band_start, frame_idx_max, color=_PHASE_COLORS[prev_phase])

    # Current frame marker
    ax.axvline(reports[current_local_idx]["frame_idx"],
               color="black", linewidth=2)

    # Triggers
    for r in reports[:current_local_idx + 1]:
        if r["triggered"]:
            ax.axvline(r["frame_idx"], color="green",
                       linestyle=":", linewidth=1, alpha=0.7)

    # Held object indicator
    held = reports[current_local_idx]["gripper_state"].get("held_obj_id")
    cur_phase = reports[current_local_idx]["gripper_state"]["phase"]
    label = f"phase={cur_phase}"
    if held is not None:
        label += f"  |  held=[{held}]"
    ax.text(0.02, 0.5, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", verticalalignment="center")

    # Legend patches
    patches = [mpatches.Patch(color=_PHASE_COLORS[p][:3] +
                               (_PHASE_COLORS[p][3] + 0.3,),
                               label=p) for p in _PHASE_COLORS]
    patches.append(mpatches.Patch(color="green", alpha=0.7, label="trigger"))
    ax.legend(handles=patches, loc="center right", fontsize=7, ncol=5)


# ─────────────────────────────────────────────────────────────────────
# RGB preview panel
# ─────────────────────────────────────────────────────────────────────

def draw_rgb_panel(ax, frame_idx: int):
    """Show the RGB frame with simple labels (no heavy overlay — mask
    info is already captured numerically in the scene panel)."""
    rgb_path = os.path.join(DATA_ROOT, "rgb", f"rgb_{frame_idx:06d}.png")
    ax.clear()
    if os.path.exists(rgb_path):
        img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    ax.set_title(f"RGB — frame {frame_idx}", fontsize=11)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=328,
                        help="Number of frames to process")
    parser.add_argument("--step", type=int, default=2,
                        help="Process every N-th frame")
    parser.add_argument("--output", default=os.path.join(
        SCENEREP_ROOT, "tests", "vis_uncertainty"))
    parser.add_argument("--save-every", type=int, default=4,
                        help="Write a frame every N processed steps")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Run the orchestrator
    runner = TrajectoryRunner(n_frames=args.frames, step=args.step)
    reports = runner.run()
    print(f"Processed {len(reports)} frames")

    # Build entropy histories
    object_entropy_history: Dict[int, List[Tuple[int, float]]] = {}
    object_pos_history: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {}
    for r in reports:
        for oid, info in r["objects"].items():
            ent = pose_entropy(info["cov"])
            object_entropy_history.setdefault(oid, []).append(
                (r["frame_idx"], ent))
            object_pos_history.setdefault(oid, []).append(
                (r["frame_idx"], info["T"][:3, 3], info["cov"]))

    # Render frames
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(nrows=3, ncols=2,
                          height_ratios=[1, 4, 3],
                          width_ratios=[1, 1])
    ax_ribbon = fig.add_subplot(gs[0, :])
    ax_rgb = fig.add_subplot(gs[1, 0])
    ax_scene = fig.add_subplot(gs[1, 1])
    ax_entropy = fig.add_subplot(gs[2, 0])
    ax_alpha = fig.add_subplot(gs[2, 1])

    saved = 0
    for local_i, report in enumerate(reports):
        if local_i % args.save_every != 0 and local_i != len(reports) - 1:
            continue
        draw_phase_ribbon(ax_ribbon, reports, local_i)
        draw_rgb_panel(ax_rgb, report["frame_idx"])
        draw_scene_panel(ax_scene, report, object_pos_history)
        draw_entropy_panel(ax_entropy, reports, local_i,
                           object_entropy_history)
        draw_alpha_panel(ax_alpha, reports, local_i)

        fig.tight_layout()
        out_path = os.path.join(args.output,
                                f"frame_{report['frame_idx']:04d}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        saved += 1
        if saved % 10 == 0:
            print(f"  saved {saved} / {len(reports) // args.save_every + 1}")

    plt.close(fig)
    print(f"\nSaved {saved} visualization frames to: {args.output}/")

    # Summary
    final = reports[-1]
    print(f"\n=== Final state ===")
    print(f"Objects tracked: {len(final['objects'])}")
    for oid, info in final["objects"].items():
        sigmas = np.sqrt(np.diag(info["cov"]))
        print(f"  [{oid}] {info['label']:<20} "
              f"pos=({info['T'][0,3]:.2f},{info['T'][1,3]:.2f},{info['T'][2,3]:.2f})m  "
              f"σ_trans=({sigmas[0]*100:.1f},{sigmas[1]*100:.1f},{sigmas[2]*100:.1f})cm")

    n_triggered = sum(1 for r in reports if r["triggered"])
    print(f"\nSlow-tier fired {n_triggered}/{len(reports)} frames")
    alphas = [r["alpha"] for r in reports if r["alpha"] is not None]
    if alphas:
        print(f"α range: [{min(alphas):.2f}, {max(alphas):.2f}], "
              f"mean {np.mean(alphas):.2f}")


if __name__ == "__main__":
    main()

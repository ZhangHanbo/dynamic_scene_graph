#!/usr/bin/env python3
"""
Static overview of tracked uncertainty-aware object trajectories.

For each object, plots:
  * The world-frame XY path it swept over the whole trajectory.
  * Covariance ellipses at periodic samples along the path — each ellipse
    is the 2σ translation uncertainty ellipse at that moment.
  * A color gradient along the line (purple → yellow) to indicate time
    progression.
  * A marker at phase-transition events (grasp, release).

Side panels:
  * Per-object entropy (log det Σ) vs frame.
  * Per-object translation σ magnitude vs frame.
  * Phase timeline with trigger markers.

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python tests/visualize_trajectories.py [--frames 328] [--step 2]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from utils.ekf_se3 import pose_entropy
from tests.test_orchestrator_integration import TrajectoryRunner


# ─────────────────────────────────────────────────────────────────────
# Color palette — stable per object id
# ─────────────────────────────────────────────────────────────────────

_PALETTE = np.array([
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


def _color(oid: int) -> np.ndarray:
    return _PALETTE[oid % len(_PALETTE)]


_PHASE_COLOR = {
    "idle":      (0.90, 0.90, 0.90, 0.30),
    "grasping":  (1.00, 0.55, 0.00, 0.50),
    "holding":   (1.00, 0.20, 0.20, 0.25),
    "releasing": (0.20, 0.40, 0.90, 0.50),
}


# ─────────────────────────────────────────────────────────────────────
# Data assembly
# ─────────────────────────────────────────────────────────────────────

def _collect_object_histories(reports):
    """Return a dict oid → list of {frame, T, cov, entropy, sigma_trans}."""
    hist = defaultdict(list)
    for r in reports:
        for oid, info in r["objects"].items():
            hist[oid].append({
                "frame": r["frame_idx"],
                "T": info["T"],
                "cov": info["cov"],
                "entropy": pose_entropy(info["cov"]),
                "sigma_trans": np.sqrt(np.diag(info["cov"])[:3]),
                "label": info["label"],
                "phase": r["gripper_state"]["phase"],
            })
    return dict(hist)


def _phase_transitions(reports):
    """List of (frame_idx, from_phase, to_phase) transitions."""
    out = []
    prev = None
    for r in reports:
        p = r["gripper_state"]["phase"]
        if prev is not None and p != prev:
            out.append((r["frame_idx"], prev, p))
        prev = p
    return out


# ─────────────────────────────────────────────────────────────────────
# Spatial panel: XY trajectories with uncertainty ellipses
# ─────────────────────────────────────────────────────────────────────

def _ellipse_params(cov_2d, n_sigma=2.0):
    cov_2d = cov_2d + np.eye(2) * 1e-10
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    eigvals = np.clip(eigvals, 1e-10, None)
    width = 2 * n_sigma * np.sqrt(eigvals[1])
    height = 2 * n_sigma * np.sqrt(eigvals[0])
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    return width, height, angle


def draw_trajectories(ax, histories, transitions,
                      ellipse_every_n_frames: int = 20):
    ax.set_aspect("equal")
    ax.set_title("Tracked object trajectories (XY top-down, 2σ ellipses)",
                 fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)

    # Determine plot bounds
    all_pos = np.concatenate([
        np.stack([h["T"][:2, 3] for h in hist])
        for hist in histories.values() if hist
    ])
    pad = 0.4
    ax.set_xlim(all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ax.set_ylim(all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    # Sort objects by first-seen frame for consistent legend
    ordered = sorted(histories.items(), key=lambda kv: kv[1][0]["frame"])

    for oid, hist in ordered:
        col = _color(oid)
        positions = np.stack([h["T"][:2, 3] for h in hist])
        frames = np.array([h["frame"] for h in hist])
        if len(positions) < 2:
            ax.scatter(*positions[0], c=[col], s=80,
                       edgecolors="k", linewidths=0.8,
                       label=f"[{oid}] {hist[0]['label']}")
            continue

        # Colored line: fade hue saturation with time progression
        segments = np.concatenate(
            [positions[:-1, None, :], positions[1:, None, :]], axis=1)
        # Scale alpha from 0.2 (old) to 0.95 (new)
        t = np.linspace(0.0, 1.0, len(segments))
        alphas = 0.2 + 0.75 * t
        colors = np.tile(col, (len(segments), 1))
        colors = np.concatenate([colors, alphas[:, None]], axis=1)
        lc = LineCollection(segments, colors=colors, linewidths=2.2)
        ax.add_collection(lc)

        # Periodic ellipses
        for i in range(0, len(hist), ellipse_every_n_frames):
            h = hist[i]
            pos = h["T"][:2, 3]
            cov2d = h["cov"][:2, :2]
            w, h_e, ang = _ellipse_params(cov2d, n_sigma=2.0)
            # Fade with time to reduce clutter
            alpha = 0.08 + 0.25 * (i / max(len(hist) - 1, 1))
            el = mpatches.Ellipse(pos, w, h_e, angle=ang,
                                   facecolor=col, alpha=alpha,
                                   edgecolor=col, linewidth=0.8)
            ax.add_patch(el)

        # Start and end markers
        ax.scatter(*positions[0], c=[col], s=50, marker="o",
                   edgecolors="k", linewidths=0.8,
                   alpha=0.35, zorder=4)
        ax.scatter(*positions[-1], c=[col], s=140, marker="*",
                   edgecolors="k", linewidths=1.0, zorder=5,
                   label=f"[{oid}] {hist[-1]['label']}")

    # Phase transition markers — small triangles at the held object's
    # position at each transition frame
    for frame, prev_p, new_p in transitions:
        if prev_p == new_p:
            continue
        # Find any object position at this frame for marker placement
        for oid, hist in histories.items():
            match = [h for h in hist if h["frame"] == frame]
            if match:
                pos = match[0]["T"][:2, 3]
                ax.annotate(f"{prev_p}→{new_p}", pos,
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=6, color="dimgray",
                            bbox=dict(boxstyle="round,pad=0.15",
                                       facecolor="white",
                                       edgecolor="dimgray",
                                       alpha=0.7))
                break

    ax.legend(loc="best", fontsize=8, framealpha=0.85, ncol=2)


# ─────────────────────────────────────────────────────────────────────
# Entropy panel
# ─────────────────────────────────────────────────────────────────────

def draw_entropy_panel(ax, histories, reports):
    ax.set_title("Pose entropy (½ log det Σ) per object", fontsize=11)
    ax.set_xlabel("frame")
    ax.set_ylabel("entropy")
    ax.grid(True, alpha=0.25)

    _shade_phases(ax, reports)

    for oid, hist in sorted(histories.items(), key=lambda kv: kv[0]):
        frames = [h["frame"] for h in hist]
        ents = [h["entropy"] for h in hist]
        ax.plot(frames, ents, color=_color(oid),
                linewidth=1.8, label=f"[{oid}]")

    # Trigger markers
    for r in reports:
        if r["triggered"]:
            ax.axvline(r["frame_idx"], color="green",
                       linestyle=":", linewidth=0.6, alpha=0.4)

    all_ents = [h["entropy"] for hist in histories.values() for h in hist]
    finite = [e for e in all_ents if np.isfinite(e)]
    if finite:
        y_lo, y_hi = np.percentile(finite, 2), np.percentile(finite, 98)
        ax.set_ylim(y_lo - 0.1 * (y_hi - y_lo + 1),
                    y_hi + 0.1 * (y_hi - y_lo + 1))
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.8)


# ─────────────────────────────────────────────────────────────────────
# Sigma-trans magnitude panel
# ─────────────────────────────────────────────────────────────────────

def draw_sigma_panel(ax, histories, reports):
    ax.set_title("Translation σ magnitude (max of x,y,z stdev) per object",
                 fontsize=11)
    ax.set_xlabel("frame")
    ax.set_ylabel("σ_translation (cm)")
    ax.grid(True, alpha=0.25)

    _shade_phases(ax, reports)

    for oid, hist in sorted(histories.items(), key=lambda kv: kv[0]):
        frames = [h["frame"] for h in hist]
        sigs = [float(np.max(h["sigma_trans"]) * 100) for h in hist]
        ax.plot(frames, sigs, color=_color(oid),
                linewidth=1.8, label=f"[{oid}]")

    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.8)


# ─────────────────────────────────────────────────────────────────────
# Phase ribbon
# ─────────────────────────────────────────────────────────────────────

def _shade_phases(ax, reports):
    """Shade the manipulation phases behind a time-series plot."""
    if not reports:
        return
    prev_phase = None
    start = None
    for r in reports:
        p = r["gripper_state"]["phase"]
        f = r["frame_idx"]
        if p != prev_phase:
            if prev_phase is not None and prev_phase != "idle":
                ax.axvspan(start, f,
                            color=_PHASE_COLOR[prev_phase], alpha=0.25)
            start = f
            prev_phase = p
    if prev_phase is not None and prev_phase != "idle":
        ax.axvspan(start, reports[-1]["frame_idx"],
                    color=_PHASE_COLOR[prev_phase], alpha=0.25)


def draw_phase_ribbon(ax, reports):
    ax.set_title("Manipulation phase timeline", fontsize=10)
    ax.set_yticks([])

    prev_phase = None
    start = None
    for r in reports:
        p = r["gripper_state"]["phase"]
        f = r["frame_idx"]
        if p != prev_phase:
            if prev_phase is not None:
                ax.axvspan(start, f, color=_PHASE_COLOR[prev_phase])
            start = f
            prev_phase = p
    if prev_phase is not None:
        ax.axvspan(start, reports[-1]["frame_idx"],
                    color=_PHASE_COLOR[prev_phase])

    for r in reports:
        if r["triggered"]:
            ax.axvline(r["frame_idx"], color="green",
                       linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlim(reports[0]["frame_idx"], reports[-1]["frame_idx"])
    patches = [mpatches.Patch(color=_PHASE_COLOR[p][:3] +
                               (_PHASE_COLOR[p][3] + 0.3,), label=p)
               for p in _PHASE_COLOR]
    patches.append(mpatches.Patch(color="green", alpha=0.7, label="trigger"))
    ax.legend(handles=patches, loc="center right", fontsize=7, ncol=5)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=328)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--ellipse-every", type=int, default=12,
                        help="Draw ellipse every N recorded frames")
    parser.add_argument("--output", default=os.path.join(
        SCENEREP_ROOT, "tests", "vis_trajectories.png"))
    args = parser.parse_args()

    runner = TrajectoryRunner(n_frames=args.frames, step=args.step)
    reports = runner.run()
    print(f"Processed {len(reports)} frames")

    histories = _collect_object_histories(reports)
    transitions = _phase_transitions(reports)
    print(f"Tracked {len(histories)} objects, "
          f"{len(transitions)} phase transitions")

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(nrows=4, ncols=3,
                          height_ratios=[0.6, 3.5, 2, 2],
                          width_ratios=[2, 1, 1])
    ax_ribbon = fig.add_subplot(gs[0, :])
    ax_traj = fig.add_subplot(gs[1:, 0])
    ax_entropy = fig.add_subplot(gs[1, 1:])
    ax_sigma = fig.add_subplot(gs[2, 1:])
    ax_stats = fig.add_subplot(gs[3, 1:])
    ax_stats.axis("off")

    draw_phase_ribbon(ax_ribbon, reports)
    draw_trajectories(ax_traj, histories, transitions,
                      ellipse_every_n_frames=args.ellipse_every)
    draw_entropy_panel(ax_entropy, histories, reports)
    draw_sigma_panel(ax_sigma, histories, reports)

    # Summary stats table
    _render_summary_table(ax_stats, histories, reports)

    fig.tight_layout()
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # Print final pose summary
    print("\n=== Final tracked state ===")
    for oid, hist in sorted(histories.items(), key=lambda kv: kv[0]):
        h = hist[-1]
        sig = h["sigma_trans"] * 100
        displacement = np.linalg.norm(h["T"][:3, 3] - hist[0]["T"][:3, 3])
        print(f"  [{oid}] {h['label']:<30} "
              f"start→end: {displacement:.2f}m  "
              f"final σ=({sig[0]:.1f},{sig[1]:.1f},{sig[2]:.1f})cm")


def _render_summary_table(ax, histories, reports):
    """Render a small stats summary in an axes with axis off."""
    n_obj = len(histories)
    n_trig = sum(1 for r in reports if r["triggered"])
    alphas = [r["alpha"] for r in reports if r["alpha"] is not None]
    trans_info = []
    for oid, hist in sorted(histories.items(), key=lambda kv: kv[0]):
        displacement = np.linalg.norm(
            hist[-1]["T"][:3, 3] - hist[0]["T"][:3, 3])
        label = hist[-1]["label"][:18]
        final_sig = float(np.max(hist[-1]["sigma_trans"]) * 100)
        trans_info.append((oid, label, displacement, final_sig))

    lines = [
        f"Frames processed: {len(reports)}",
        f"Objects tracked: {n_obj}",
        f"Slow-tier triggers: {n_trig}",
    ]
    if alphas:
        lines.append(f"α range: [{min(alphas):.1f}, {max(alphas):.1f}],  "
                     f"mean {np.mean(alphas):.1f}")
    lines.append("")
    lines.append("Object         Δ pos (m)   final σ (cm)")
    for oid, label, d, s in trans_info:
        lines.append(f"[{oid}] {label:<18s}  {d:5.2f}       {s:5.1f}")

    txt = "\n".join(lines)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke",
                       edgecolor="lightgray"))


if __name__ == "__main__":
    main()

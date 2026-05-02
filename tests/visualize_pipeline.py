#!/usr/bin/env python3
"""
Full-pipeline visualization with modular stages.

Modules (each independently runnable):
  Stage 1 — TopDownMap:       background from depth → world-frame → bird's-eye
  Stage 2 — RobotPath:        camera/base trajectory on the map
  Stage 3 — ObjectTracks:     tracked objects with colored trails + cov ellipses
  Stage 4 — FrameCompositor:  per-frame composite (map + tracks + RGB inset)

Run:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    conda run -n ocmp_test python tests/visualize_pipeline.py [--stage 1|2|3|4] [--frames N]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

DATA_BASE = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects"
)
# Default trajectory — overridden at runtime by TrajectoryData(trajectory=...)
# and by the --trajectory CLI argument. Kept as a module-level fallback so
# existing callers that read DATA_ROOT continue to work.
DATA_ROOT = os.path.join(DATA_BASE, "apple_bowl_2")

# Reuse data-loading helpers from the integration test.
from tests.test_orchestrator_integration import (
    _load_pose_txt, _load_detections, _build_T_co_from_mask,
    _gripper_state_from_distance,
    K,  # camera intrinsics
)
from pose_update.perception.icp_pose import PoseEstimator, METHODS as POSE_METHODS

OUT_DIR = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline")


# ═════════════════════════════════════════════════════════════════════
#  Held-object resolution via point-cloud proximity
# ═════════════════════════════════════════════════════════════════════

def resolve_held_by_proximity(detections: List[Dict],
                               depth: np.ndarray,
                               ee_cam: np.ndarray,
                               cam_K: np.ndarray = K,
                               percentile: float = 5.0,
                               max_dist: float = 0.20,
                               subsample: int = 3,
                               ) -> Optional[int]:
    """Pick the object whose 3D point cloud is closest to the EE.

    Everything is computed in CAMERA FRAME — the back-projected depth
    points are already there, so no world transform is needed.

    Instead of comparing centroids (which picks the apple inside a bowl
    over the bowl itself), we take the low percentile of point-to-EE
    distances. The bowl's rim points are physically closer to the
    gripper than the apple's surface, so the bowl wins.

    Args:
        detections: list of dicts with 'id', 'mask'.
        depth:      (H, W) float32 depth in m.
        ee_cam:     (3,) EE position in camera frame.
        percentile: percentile of per-point distances (low = closest
                    points dominate).
        max_dist:   reject candidates further than this.
        subsample:  pixel stride for speed.

    Returns:
        Object id of the nearest candidate, or None.
    """
    fx, fy = float(cam_K[0, 0]), float(cam_K[1, 1])
    cx, cy = float(cam_K[0, 2]), float(cam_K[1, 2])

    best_id: Optional[int] = None
    best_dist = float("inf")

    for det in detections:
        oid = det.get("id")
        if oid is None:
            continue
        mask = det.get("mask")
        if mask is None:
            continue

        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            continue
        sel = np.arange(0, len(xs), subsample)
        xs, ys = xs[sel], ys[sel]
        ds = depth[ys, xs].astype(np.float64)
        valid = np.isfinite(ds) & (ds > 0.1) & (ds < 5.0)
        if valid.sum() < 10:
            continue
        xs, ys, ds = xs[valid], ys[valid], ds[valid]

        # Back-project to camera frame (already there — no T_cw needed).
        Xc = (xs - cx) / fx * ds
        Yc = (ys - cy) / fy * ds
        Zc = ds
        pts_cam = np.stack([Xc, Yc, Zc], axis=1)  # (N, 3)

        dists = np.linalg.norm(pts_cam - ee_cam[None, :], axis=1)
        d_pct = float(np.percentile(dists, percentile))

        if d_pct < best_dist and d_pct < max_dist:
            best_dist = d_pct
            best_id = oid

    return best_id

# Consistent object palette — 10 distinct colors.
OBJECT_COLORS = [
    (0.18, 0.80, 0.44),   # green
    (0.90, 0.30, 0.23),   # red
    (0.20, 0.60, 0.86),   # blue
    (0.95, 0.77, 0.06),   # yellow
    (0.61, 0.35, 0.71),   # purple
    (0.90, 0.49, 0.13),   # orange
    (0.10, 0.74, 0.61),   # teal
    (0.93, 0.44, 0.39),   # salmon
    (0.36, 0.68, 0.89),   # light blue
    (0.96, 0.82, 0.25),   # gold
]


def _obj_color(oid: int) -> Tuple[float, float, float]:
    return OBJECT_COLORS[oid % len(OBJECT_COLORS)]


# ═════════════════════════════════════════════════════════════════════
#  Stage 1 — Top-down background map
# ═════════════════════════════════════════════════════════════════════

class TopDownMap:
    """Density-normalized RGB bird's-eye-view map.

    Each frame contributes AT MOST ONCE per cell (per-frame dedup) and
    splats to a 3×3 neighborhood to fill coverage gaps from pixel
    subsampling — together these produce uniform density regardless of
    camera distance or viewing angle.

    Three parallel accumulators per cell:
      * `_color_sum`: running sum of per-frame AVERAGED RGB colors.
      * `_bg_count`:  number of frames this cell was observed as background.
      * `_obj_count`: number of frames this cell was observed as an object.

    Render: cell color = color_sum / total_count. Objects can be
    optionally tinted red to pop out against a similarly-colored background.
    Robot-body pixels (hand mask) are excluded from everything.
    """

    EMPTY_COLOR = np.array([0.96, 0.96, 0.96])
    OBJ_TINT = np.array([0.90, 0.20, 0.20])  # red tint for object cells
    # Plain (non-color) mode colors.
    BG_GREY = np.array([0.70, 0.70, 0.70])
    OBJ_RED = np.array([0.85, 0.20, 0.20])

    def __init__(self,
                 resolution: float = 0.005,   # m per pixel
                 x_range: Tuple[float, float] = (-4.0, 5.0),
                 y_range: Tuple[float, float] = (-2.0, 2.5),
                 z_range: Tuple[float, float] = (0.3, 1.8),
                 K: np.ndarray = K):
        self.res = resolution
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.K = K.astype(np.float64)
        self.W = int((self.x_max - self.x_min) / resolution)
        self.H = int((self.y_max - self.y_min) / resolution)
        self._color_sum = np.zeros((self.H, self.W, 3), dtype=np.float64)
        self._bg_count = np.zeros((self.H, self.W), dtype=np.float64)
        self._obj_count = np.zeros((self.H, self.W), dtype=np.float64)

    def add_frame(self, rgb: np.ndarray, depth: np.ndarray,
                  T_cw: np.ndarray,
                  object_mask: Optional[np.ndarray] = None,
                  hand_mask: Optional[np.ndarray] = None,
                  subsample: int = 4) -> None:
        """Project one frame's depth pixels into the grid with RGB colors.

        Steps per frame:
          1. Valid (non-robot-body, finite-depth) pixel mask.
          2. Back-project + height filter → cell index per pixel.
          3. Build per-frame buffers via splat to 3×3 + color averaging.
          4. Accumulate: each touched cell contributes ONE averaged color
             vote, +1 bg-count OR +1 obj-count (obj takes priority if
             both fired in the splat).
        """
        h, w = depth.shape[:2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        vs, us = np.mgrid[0:h:subsample, 0:w:subsample]
        us = us.ravel().astype(np.int64)
        vs = vs.ravel().astype(np.int64)
        ds = depth[vs, us].astype(np.float64)

        valid = np.isfinite(ds) & (ds > 0.1) & (ds < 5.0)
        if hand_mask is not None:
            valid &= ~hand_mask[vs, us].astype(bool)
        if not valid.any():
            return
        us, vs, ds = us[valid], vs[valid], ds[valid]

        is_obj = (object_mask[vs, us].astype(bool)
                  if object_mask is not None
                  else np.zeros_like(us, dtype=bool))
        colors = rgb[vs, us].astype(np.float64) / 255.0  # (N, 3)

        # Back-project to camera then world.
        Xc = (us - cx) / fx * ds
        Yc = (vs - cy) / fy * ds
        pts_c = np.stack([Xc, Yc, ds, np.ones_like(Xc)], axis=0)
        pts_w = (T_cw @ pts_c)[:3, :]
        Xw, Yw, Zw = pts_w

        hfilt = (Zw >= self.z_min) & (Zw <= self.z_max)
        Xw, Yw = Xw[hfilt], Yw[hfilt]
        colors = colors[hfilt]
        is_obj = is_obj[hfilt]

        gi = ((Xw - self.x_min) / self.res).astype(int)
        gj = ((Yw - self.y_min) / self.res).astype(int)
        in_bounds = (gi >= 0) & (gi < self.W) & (gj >= 0) & (gj < self.H)
        gi, gj = gi[in_bounds], gj[in_bounds]
        colors = colors[in_bounds]
        is_obj = is_obj[in_bounds]

        if len(gi) == 0:
            return

        # Splat each point to a 3×3 neighborhood.
        deltas = np.array(
            [(dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1)],
            dtype=np.int64,
        )  # (9, 2)
        splat_jj = np.clip(gj[:, None] + deltas[None, :, 0],
                           0, self.H - 1).ravel()
        splat_ii = np.clip(gi[:, None] + deltas[None, :, 1],
                           0, self.W - 1).ravel()
        splat_colors = np.repeat(colors, 9, axis=0)
        splat_is_obj = np.repeat(is_obj, 9)

        # Per-frame averaging + dedup via temporary buffers.
        frame_color = np.zeros_like(self._color_sum)
        frame_count = np.zeros_like(self._bg_count)
        frame_obj_hit = np.zeros_like(self._bg_count, dtype=bool)
        np.add.at(frame_color, (splat_jj, splat_ii), splat_colors)
        np.add.at(frame_count, (splat_jj, splat_ii), 1.0)
        # "Any obj pixel touched this cell" — set via scatter-OR.
        obj_jj = splat_jj[splat_is_obj]
        obj_ii = splat_ii[splat_is_obj]
        frame_obj_hit[obj_jj, obj_ii] = True

        touched = frame_count > 0
        # Per-cell averaged color for THIS frame.
        avg = np.zeros_like(frame_color)
        avg[touched] = frame_color[touched] / frame_count[touched, None]

        # Accumulate into global sums with per-frame unit weight.
        self._color_sum[touched] += avg[touched]
        bg_hit = touched & ~frame_obj_hit
        obj_hit = touched & frame_obj_hit
        self._bg_count[bg_hit] += 1.0
        self._obj_count[obj_hit] += 1.0

    def render(self,
               color: bool = False,
               tint_objects: bool = True,
               smooth_sigma: float = 2.0) -> np.ndarray:
        """Return the (H, W, 3) RGB map.

        Args:
            color: if True, render with averaged RGB colors from the
                camera (plus optional red tint on object cells). If
                False (default), render a simple grey/red categorical
                map — grey for observed background, red for object cells.
            tint_objects: only used when `color=True`. Blends object
                cells toward red to keep categorical separation.
            smooth_sigma: mass-preserving Gaussian smoothing σ in cells.
                Applied only in color mode (the categorical mode has no
                view seams to smooth). Set to 0 to disable.
        """
        total = self._bg_count + self._obj_count
        seen = total > 0
        obj_dom = self._obj_count > self._bg_count
        bg_dom = seen & ~obj_dom

        img = np.tile(self.EMPTY_COLOR, (self.H, self.W, 1)).astype(np.float64)

        if not color:
            # Categorical grey/red render — no accumulated RGB used.
            img[bg_dom] = self.BG_GREY
            img[obj_dom] = self.OBJ_RED
            return img

        # Color mode: use accumulated averaged RGB.
        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            csum = gaussian_filter(
                self._color_sum,
                sigma=(smooth_sigma, smooth_sigma, 0.0),
                mode="constant", cval=0.0,
            )
            tsum = gaussian_filter(
                total, sigma=smooth_sigma, mode="constant", cval=0.0,
            )
        else:
            csum = self._color_sum
            tsum = total

        avg = np.zeros_like(self._color_sum)
        valid = tsum > 1e-3
        avg[valid] = csum[valid] / tsum[valid, None]

        if tint_objects:
            # Use the ORIGINAL (unblurred) obj-dominance mask to keep
            # categorical edges sharp.
            img[bg_dom] = avg[bg_dom]
            img[obj_dom] = 0.7 * avg[obj_dom] + 0.3 * self.OBJ_TINT
        else:
            img[seen] = avg[seen]
        return np.clip(img, 0.0, 1.0)

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """World XY → pixel (col, row) in the rendered image."""
        col = int((x - self.x_min) / self.res)
        row = int((y - self.y_min) / self.res)
        return col, row


# ═════════════════════════════════════════════════════════════════════
#  Stage 2 — Robot trajectory overlay
# ═════════════════════════════════════════════════════════════════════

class RobotPathOverlay:
    """Draws the camera/robot path on a top-down map axes."""

    def __init__(self):
        self.positions: List[np.ndarray] = []  # world XYZ

    def add_pose(self, T_cw: np.ndarray) -> None:
        self.positions.append(T_cw[:3, 3].copy())

    def draw(self, ax: plt.Axes, tmap: TopDownMap,
             color: str = "#1a1a2e", lw: float = 1.8,
             label: str = "robot path") -> None:
        """Draw the path on an axes whose image is the top-down map."""
        if len(self.positions) < 2:
            return
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        # Convert world → pixel.
        pxs = [(x - tmap.x_min) / tmap.res for x in xs]
        pys = [(y - tmap.y_min) / tmap.res for y in ys]
        ax.plot(pxs, pys, color=color, lw=lw, alpha=0.8, label=label)
        # Start marker.
        ax.plot(pxs[0], pys[0], "o", color="lime", ms=6, zorder=5)
        # End marker.
        ax.plot(pxs[-1], pys[-1], "s", color="red", ms=6, zorder=5)


# ═════════════════════════════════════════════════════════════════════
#  Stage 3 — Object track overlay
# ═════════════════════════════════════════════════════════════════════

class ObjectTrackOverlay:
    """Draws tracked-object trails and covariance ellipses per frame."""

    def __init__(self):
        # Per-object lists of (frame_idx, world_x, world_y, cov_6x6, label).
        self.tracks: Dict[int, List[Tuple[int, float, float,
                                          np.ndarray, str]]] = {}

    def add_frame(self, frame_idx: int,
                  objects: Dict[int, Dict]) -> None:
        """Record one frame's object posteriors.

        objects: {oid: {"T": 4×4, "cov": 6×6, "label": str, ...}}
        """
        for oid, obj in objects.items():
            if oid not in self.tracks:
                self.tracks[oid] = []
            pos = obj["T"][:3, 3]
            self.tracks[oid].append((
                frame_idx,
                float(pos[0]), float(pos[1]),
                obj["cov"].copy(),
                obj.get("label", "?"),
            ))

    def draw(self, ax: plt.Axes, tmap: TopDownMap,
             current_frame: Optional[int] = None,
             trail_alpha: float = 0.4,
             ellipse_nsigma: float = 3.0) -> List[mpatches.Patch]:
        """Draw all trails and the current-frame covariance ellipses.

        Returns legend handles for the caller.
        """
        handles = []
        for oid, track in sorted(self.tracks.items()):
            color = _obj_color(oid)

            # Trail up to current_frame.
            if current_frame is not None:
                pts = [(x, y) for (fi, x, y, _, _) in track
                       if fi <= current_frame]
            else:
                pts = [(x, y) for (_, x, y, _, _) in track]
            if not pts:
                continue

            pxs = [(x - tmap.x_min) / tmap.res for x, _ in pts]
            pys = [(y - tmap.y_min) / tmap.res for _, y in pts]
            ax.plot(pxs, pys, color=color, lw=1.0, alpha=trail_alpha)

            # Latest point (or at current_frame).
            ax.plot(pxs[-1], pys[-1], "o", color=color, ms=7,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=6)

            # Covariance ellipse at the latest point.
            if current_frame is not None:
                entries = [(fi, x, y, c, l) for (fi, x, y, c, l) in track
                           if fi <= current_frame]
            else:
                entries = track
            if entries:
                _, lx, ly, lcov, label = entries[-1]
                self._draw_ellipse(ax, tmap, lx, ly, lcov,
                                   color=color, nsigma=ellipse_nsigma)
                short = label[:18] + "..." if len(label) > 20 else label
                handles.append(mpatches.Patch(
                    color=color, label=f"[{oid}] {short}"))

        return handles

    @staticmethod
    def _draw_ellipse(ax: plt.Axes, tmap: TopDownMap,
                      wx: float, wy: float, cov: np.ndarray,
                      color, nsigma: float = 3.0) -> None:
        """Draw an XY covariance ellipse at world coords (wx, wy)."""
        cov_xy = cov[:2, :2]
        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        eigvals = np.maximum(eigvals, 1e-12)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        # Ellipse axes in meters → convert to pixels.
        # Enforce a minimum visible radius so sub-cm cov is still apparent.
        MIN_PX = 12.0
        width = max(2 * nsigma * np.sqrt(eigvals[0]) / tmap.res, MIN_PX)
        height = max(2 * nsigma * np.sqrt(eigvals[1]) / tmap.res, MIN_PX)
        px = (wx - tmap.x_min) / tmap.res
        py = (wy - tmap.y_min) / tmap.res
        ell = Ellipse((px, py), width, height, angle=angle,
                      facecolor=(*color, 0.25), edgecolor=color,
                      lw=1.0, zorder=4)
        ax.add_patch(ell)


# ═════════════════════════════════════════════════════════════════════
#  Stage 4 — Per-frame composite renderer
# ═════════════════════════════════════════════════════════════════════

class FrameCompositor:
    """Renders a multi-panel figure per frame.

    Layout:
        ┌───────────────────────────┬──────────┐
        │  Top-down map with        │  RGB     │
        │  robot path + objects     │  inset   │
        ├───────────────────────────┴──────────┤
        │  Phase ribbon + legend               │
        └──────────────────────────────────────┘
    """

    PHASE_COLORS = {
        "idle":       (0.85, 0.85, 0.85, 0.6),
        "grasping":   (1.00, 0.55, 0.00, 0.6),
        "holding":    (1.00, 0.20, 0.20, 0.4),
        "releasing":  (0.20, 0.40, 0.90, 0.6),
    }

    def __init__(self, tmap: TopDownMap, robot_path: RobotPathOverlay,
                 obj_tracks: ObjectTrackOverlay):
        self.tmap = tmap
        self.robot_path = robot_path
        self.obj_tracks = obj_tracks
        self.phases: List[Tuple[int, str]] = []  # (frame_idx, phase)

    def add_phase(self, frame_idx: int, phase: str) -> None:
        self.phases.append((frame_idx, phase))

    def render_frame(self, frame_idx: int,
                     rgb: Optional[np.ndarray] = None,
                     save_path: Optional[str] = None,
                     dpi: int = 150) -> None:
        bg = self.tmap.render()

        fig = plt.figure(figsize=(16, 9))
        # Main map panel.
        ax_map = fig.add_axes([0.02, 0.15, 0.65, 0.82])
        ax_map.imshow(bg, origin="lower", aspect="equal")
        self.robot_path.draw(ax_map, self.tmap)
        handles = self.obj_tracks.draw(ax_map, self.tmap,
                                       current_frame=frame_idx)
        ax_map.set_title(f"Top-down scene — frame {frame_idx}",
                         fontsize=11, fontweight="bold")
        ax_map.set_xlabel("x (world)")
        ax_map.set_ylabel("y (world)")
        # Tick labels in world meters.
        xt = np.arange(self.tmap.x_min, self.tmap.x_max + 0.5, 1.0)
        ax_map.set_xticks([(x - self.tmap.x_min) / self.tmap.res for x in xt])
        ax_map.set_xticklabels([f"{x:.0f}" for x in xt], fontsize=7)
        yt = np.arange(self.tmap.y_min, self.tmap.y_max + 0.5, 1.0)
        ax_map.set_yticks([(y - self.tmap.y_min) / self.tmap.res for y in yt])
        ax_map.set_yticklabels([f"{y:.0f}" for y in yt], fontsize=7)

        # RGB inset.
        if rgb is not None:
            ax_rgb = fig.add_axes([0.69, 0.42, 0.30, 0.55])
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f"RGB — frame {frame_idx}", fontsize=9)
            ax_rgb.axis("off")

        # Phase ribbon at bottom.
        ax_phase = fig.add_axes([0.02, 0.02, 0.96, 0.08])
        self._draw_phase_ribbon(ax_phase, frame_idx)

        # Legend.
        if handles:
            ax_legend = fig.add_axes([0.69, 0.15, 0.30, 0.25])
            ax_legend.axis("off")
            ax_legend.legend(handles=handles, loc="upper left",
                             fontsize=7, frameon=True,
                             title="Tracked objects", title_fontsize=8)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)

    def _draw_phase_ribbon(self, ax: plt.Axes, current_frame: int) -> None:
        if not self.phases:
            ax.axis("off")
            return
        max_frame = max(f for f, _ in self.phases)
        for i, (fi, phase) in enumerate(self.phases):
            next_fi = self.phases[i + 1][0] if i + 1 < len(self.phases) \
                else max_frame + 1
            color = self.PHASE_COLORS.get(phase, (0.5, 0.5, 0.5, 0.3))
            ax.axvspan(fi, next_fi, color=color)
        ax.axvline(current_frame, color="black", lw=1.5, ls="--")
        ax.set_xlim(0, max_frame + 1)
        ax.set_yticks([])
        ax.set_xlabel("frame", fontsize=8)
        ax.set_title("Manipulation phase", fontsize=8)
        # Legend for phases.
        phase_handles = [mpatches.Patch(color=c, label=p)
                         for p, c in self.PHASE_COLORS.items()]
        ax.legend(handles=phase_handles, loc="upper right",
                  fontsize=6, ncol=4, frameon=False)


# ═════════════════════════════════════════════════════════════════════
#  Side-by-side compositor (Ours vs. Baseline)
# ═════════════════════════════════════════════════════════════════════

class SideBySideCompositor:
    """Renders two pipelines' object tracks on twin top-down maps.

    Both panels share the same background, same robot path, same phase
    ribbon; they differ only in `ObjectTrackOverlay`. Lays out as:

        ┌────────────────────────┬────────────────────────┐
        │ Ours (RBPF + proprio + │ Baseline (vision-only, │
        │ rigid-attachment)      │ TSDF++ style)          │
        ├────────────────────────┴────────────────────────┤
        │ Phase ribbon                                    │
        └─────────────────────────────────────────────────┘
    """

    PHASE_COLORS = FrameCompositor.PHASE_COLORS

    def __init__(self,
                 tmap: TopDownMap,
                 robot_path: RobotPathOverlay,
                 tracks_ours: ObjectTrackOverlay,
                 tracks_base: ObjectTrackOverlay,
                 color: bool = False):
        self.tmap = tmap
        self.robot_path = robot_path
        self.tracks_ours = tracks_ours
        self.tracks_base = tracks_base
        self.color = color
        self.phases: List[Tuple[int, str]] = []

    def add_phase(self, frame_idx: int, phase: str) -> None:
        self.phases.append((frame_idx, phase))

    def _draw_panel(self, ax: plt.Axes, bg: np.ndarray,
                    tracks: ObjectTrackOverlay,
                    title: str, frame_idx: int
                    ) -> List[mpatches.Patch]:
        ax.imshow(bg, origin="lower", aspect="equal")
        self.robot_path.draw(ax, self.tmap)
        handles = tracks.draw(ax, self.tmap, current_frame=frame_idx)
        ax.set_title(title, fontsize=10, fontweight="bold")
        xt = np.arange(self.tmap.x_min, self.tmap.x_max + 0.5, 1.0)
        ax.set_xticks([(x - self.tmap.x_min) / self.tmap.res for x in xt])
        ax.set_xticklabels([f"{x:.0f}" for x in xt], fontsize=6)
        yt = np.arange(self.tmap.y_min, self.tmap.y_max + 0.5, 1.0)
        ax.set_yticks([(y - self.tmap.y_min) / self.tmap.res for y in yt])
        ax.set_yticklabels([f"{y:.0f}" for y in yt], fontsize=6)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        return handles

    def render_frame(self, frame_idx: int,
                     rgb: Optional[np.ndarray] = None,
                     save_path: Optional[str] = None,
                     dpi: int = 140) -> None:
        """Layout:
            ┌────────────────┬──────────────────┐
            │                │ Ours map         │
            │  Observation   ├──────────────────┤
            │  (RGB)         │ Baseline map     │
            ├────────────────┴──────────────────┤
            │         Phase ribbon              │
            └───────────────────────────────────┘
        """
        bg = self.tmap.render(color=self.color)
        fig = plt.figure(figsize=(16, 11))

        # Observation (current RGB) — large, on the left.
        if rgb is not None:
            ax_rgb = fig.add_axes([0.02, 0.22, 0.36, 0.75])
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f"Observation — frame {frame_idx}",
                             fontsize=11, fontweight="bold")
            ax_rgb.axis("off")

        # Two map panels stacked vertically on the right.
        # Top: ours.  Bottom: baseline.
        ax_ours = fig.add_axes([0.41, 0.60, 0.57, 0.37])
        ax_base = fig.add_axes([0.41, 0.22, 0.57, 0.37])
        h_ours = self._draw_panel(
            ax_ours, bg, self.tracks_ours,
            f"Ours (RBPF + proprio rigid-attachment) — frame {frame_idx}",
            frame_idx)
        h_base = self._draw_panel(
            ax_base, bg, self.tracks_base,
            f"Baseline (TSDF++-style, vision only) — frame {frame_idx}",
            frame_idx)

        # Legend — bottom-left corner, below the RGB panel.
        if h_ours:
            ax_legend = fig.add_axes([0.02, 0.04, 0.36, 0.14])
            ax_legend.axis("off")
            ax_legend.legend(
                handles=h_ours, loc="upper left",
                fontsize=7, frameon=True, ncol=2,
                title="Tracked objects", title_fontsize=8)

        # Phase/progress ribbon spans the bottom, below both panels.
        ax_phase = fig.add_axes([0.41, 0.04, 0.57, 0.14])
        self._draw_phase_ribbon(ax_phase, frame_idx)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)

    def _draw_phase_ribbon(self, ax: plt.Axes, current_frame: int) -> None:
        if not self.phases:
            ax.axis("off")
            return
        max_frame = max(f for f, _ in self.phases)
        for i, (fi, phase) in enumerate(self.phases):
            next_fi = self.phases[i + 1][0] if i + 1 < len(self.phases) \
                else max_frame + 1
            color = self.PHASE_COLORS.get(phase, (0.5, 0.5, 0.5, 0.3))
            ax.axvspan(fi, next_fi, color=color)
        ax.axvline(current_frame, color="black", lw=1.5, ls="--")
        ax.set_xlim(0, max_frame + 1)
        ax.set_yticks([])
        ax.set_xlabel("frame", fontsize=7)
        ax.set_title("Manipulation phase (ours uses this; baseline ignores)",
                     fontsize=8)
        phase_handles = [mpatches.Patch(color=c, label=p)
                         for p, c in self.PHASE_COLORS.items()]
        ax.legend(handles=phase_handles, loc="upper right",
                  fontsize=6, ncol=4, frameon=False)


# ═════════════════════════════════════════════════════════════════════
#  All-methods compositor (Ours + Baseline × 3 methods)
# ═════════════════════════════════════════════════════════════════════

class AllMethodsCompositor:
    """2×3 grid of object-track panels comparing three visual pose
    estimation methods in both ours-filter and baseline-filter variants.

    Layout:
        ┌────────┬──────────────┬──────────────┬──────────────┐
        │        │ centroid     │ icp_chain    │ icp_anchor   │
        │        ├──────────────┼──────────────┼──────────────┤
        │  Obs   │ Ours         │ Ours         │ Ours         │
        │ (RGB)  ├──────────────┼──────────────┼──────────────┤
        │        │ Baseline     │ Baseline     │ Baseline     │
        ├────────┴──────────────┴──────────────┴──────────────┤
        │                Phase ribbon                         │
        └─────────────────────────────────────────────────────┘

    All six panels share the same background and robot path; they differ
    only in which filter + which visual tracker produced the object
    tracks overlaid on top.
    """

    PHASE_COLORS = FrameCompositor.PHASE_COLORS
    METHOD_LABELS = {
        "centroid":          "Centroid",
        "icp_chain":         "ICP chain (R_prev, t_centroid)",
        "icp_anchor":        "ICP anchor (R=I, t_centroid)",
        "icp_chain_strict":  "ICP chain-strict (R_prev, t_prev)",
        "icp_anchor_strict": "ICP anchor-strict (R=I, t_first)",
    }

    def __init__(self,
                 tmap: TopDownMap,
                 robot_path: RobotPathOverlay,
                 tracks_ours: Dict[str, ObjectTrackOverlay],
                 tracks_base: Dict[str, ObjectTrackOverlay],
                 color: bool = False):
        """Args:
            tracks_ours: method → ObjectTrackOverlay for the ours filter.
            tracks_base: method → ObjectTrackOverlay for the baseline filter.
        """
        self.tmap = tmap
        self.robot_path = robot_path
        self.tracks_ours = tracks_ours
        self.tracks_base = tracks_base
        self.color = color
        self.phases: List[Tuple[int, str]] = []
        self.methods = list(tracks_ours.keys())

    def add_phase(self, frame_idx: int, phase: str) -> None:
        self.phases.append((frame_idx, phase))

    def _draw_panel(self, ax: plt.Axes, bg: np.ndarray,
                    tracks: ObjectTrackOverlay,
                    title: str, frame_idx: int
                    ) -> List[mpatches.Patch]:
        ax.imshow(bg, origin="lower", aspect="equal")
        self.robot_path.draw(ax, self.tmap)
        handles = tracks.draw(ax, self.tmap, current_frame=frame_idx)
        ax.set_title(title, fontsize=9, fontweight="bold")
        xt = np.arange(self.tmap.x_min, self.tmap.x_max + 0.5, 2.0)
        ax.set_xticks([(x - self.tmap.x_min) / self.tmap.res for x in xt])
        ax.set_xticklabels([f"{x:.0f}" for x in xt], fontsize=5)
        yt = np.arange(self.tmap.y_min, self.tmap.y_max + 0.5, 2.0)
        ax.set_yticks([(y - self.tmap.y_min) / self.tmap.res for y in yt])
        ax.set_yticklabels([f"{y:.0f}" for y in yt], fontsize=5)
        return handles

    def render_frame(self,
                     frame_idx: int,
                     rgb: Optional[np.ndarray] = None,
                     save_path: Optional[str] = None,
                     dpi: int = 140) -> None:
        bg = self.tmap.render(color=self.color)
        fig = plt.figure(figsize=(22, 11))

        # Observation (RGB) on the left, spanning both map rows.
        if rgb is not None:
            ax_rgb = fig.add_axes([0.02, 0.22, 0.23, 0.68])
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f"Observation — frame {frame_idx}",
                             fontsize=10, fontweight="bold")
            ax_rgb.axis("off")

        # 2 rows × 3 cols of map panels.
        n_cols = len(self.methods)
        col_w = (1.0 - 0.27 - 0.02) / n_cols       # right panel starts at 0.27
        col_w_usable = col_w - 0.01
        row_top_y = 0.54
        row_bot_y = 0.22
        row_h = 0.32

        handles_collect: List[mpatches.Patch] = []
        for i, method in enumerate(self.methods):
            x0 = 0.27 + i * col_w
            ax_top = fig.add_axes([x0, row_top_y, col_w_usable, row_h])
            ax_bot = fig.add_axes([x0, row_bot_y, col_w_usable, row_h])
            label = self.METHOD_LABELS.get(method, method)
            h_ours = self._draw_panel(
                ax_top, bg, self.tracks_ours[method],
                f"Ours × {label}", frame_idx)
            h_base = self._draw_panel(
                ax_bot, bg, self.tracks_base[method],
                f"Baseline × {label}", frame_idx)
            if i == 0:
                handles_collect = h_ours

        # Row headers on the left — just above each map row.
        fig.text(0.26, row_top_y + row_h + 0.005, "ours",
                 fontsize=9, fontweight="bold", ha="right")
        fig.text(0.26, row_bot_y + row_h + 0.005, "baseline",
                 fontsize=9, fontweight="bold", ha="right")

        # Legend under RGB panel.
        if handles_collect:
            ax_legend = fig.add_axes([0.02, 0.04, 0.23, 0.14])
            ax_legend.axis("off")
            ax_legend.legend(
                handles=handles_collect, loc="upper left",
                fontsize=6, frameon=True, ncol=2,
                title="Tracked objects", title_fontsize=7)

        # Phase ribbon spans the rest.
        ax_phase = fig.add_axes([0.27, 0.04, 0.71, 0.13])
        self._draw_phase_ribbon(ax_phase, frame_idx)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)

    def _draw_phase_ribbon(self, ax: plt.Axes, current_frame: int) -> None:
        if not self.phases:
            ax.axis("off")
            return
        max_frame = max(f for f, _ in self.phases)
        for i, (fi, phase) in enumerate(self.phases):
            next_fi = (self.phases[i + 1][0] if i + 1 < len(self.phases)
                       else max_frame + 1)
            color = self.PHASE_COLORS.get(phase, (0.5, 0.5, 0.5, 0.3))
            ax.axvspan(fi, next_fi, color=color)
        ax.axvline(current_frame, color="black", lw=1.5, ls="--")
        ax.set_xlim(0, max_frame + 1)
        ax.set_yticks([])
        ax.set_xlabel("frame", fontsize=7)
        ax.set_title("Manipulation phase (ours uses this; baseline ignores)",
                     fontsize=8)
        phase_handles = [mpatches.Patch(color=c, label=p)
                         for p, c in self.PHASE_COLORS.items()]
        ax.legend(handles=phase_handles, loc="upper right",
                  fontsize=6, ncol=4, frameon=False)


# ═════════════════════════════════════════════════════════════════════
#  Data loader (shared across stages)
# ═════════════════════════════════════════════════════════════════════

class TrajectoryData:
    """Lazy data loader for a named trajectory under
    `Mobile_Manipulation_on_Fetch/multi_objects/<trajectory>/`.
    """

    def __init__(self,
                 n_frames: int = 10_000,
                 step: int = 1,
                 trajectory: str = "apple_bowl_2"):
        self.trajectory = trajectory
        self.data_root = os.path.join(DATA_BASE, trajectory)
        self.cam_poses = _load_pose_txt(os.path.join(
            self.data_root, "pose_txt", "camera_pose.txt"))
        self.ee_poses = _load_pose_txt(os.path.join(
            self.data_root, "pose_txt", "ee_pose.txt"))
        self.l_finger = _load_pose_txt(os.path.join(
            self.data_root, "pose_txt", "l_gripper_pose.txt"))
        self.r_finger = _load_pose_txt(os.path.join(
            self.data_root, "pose_txt", "r_gripper_pose.txt"))
        self.n_frames = min(n_frames, len(self.cam_poses))
        self.step = step
        self.indices = list(range(0, self.n_frames, step))
        # Joint link poses for hand masking.
        self._joints_data = self._load_joints()

    def _load_joints(self) -> Optional[Dict]:
        import json
        from scipy.spatial.transform import Rotation
        path = os.path.join(self.data_root, "pose_txt", "joints_pose.json")
        if not os.path.exists(path):
            return None
        raw = json.load(open(path))
        parsed: Dict[str, Dict[str, np.ndarray]] = {}
        for frame_key, links in raw.items():
            idx = int(frame_key)
            parsed[idx] = {}
            for link_name, vals in links.items():
                vals = np.asarray(vals, dtype=np.float64)
                T = np.eye(4, dtype=np.float64)
                T[:3, 3] = vals[:3]
                T[:3, :3] = Rotation.from_quat(vals[3:]).as_matrix()
                parsed[idx][link_name] = T
        return parsed

    def load_frame(self, idx: int):
        rgb_path = os.path.join(self.data_root, "rgb", f"rgb_{idx:06d}.png")
        depth_path = os.path.join(self.data_root, "depth",
                                   f"depth_{idx:06d}.npy")
        det_path = os.path.join(self.data_root, "detection_h",
                                f"detection_{idx:06d}_final.json")
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            return None
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)
        detections = _load_detections(det_path)
        return rgb, depth, detections

    def gripper_phase(self, idx: int, last_d: Optional[float]):
        d = np.linalg.norm(
            self.l_finger[idx][:3, 3] - self.r_finger[idx][:3, 3])
        raw = _gripper_state_from_distance(d, last_d)
        return raw, d

    def T_bg(self, idx: int) -> np.ndarray:
        """Gripper (EE) pose in a base-like frame.

        `ee_poses` is stored as T_ec (EE-in-camera) already — evidence:
        ee_poses[44] = (0.014, 0.046, 0.694) has z=0.69m in front of
        the camera, which only makes sense if it's already in camera
        frame. A world-frame EE at (0.01, 0.05, 0.69) would place it
        2m away from the bowl it's grasping.

        For the RBPF orchestrator, SLAM provides T_wb = camera-to-world
        (via PassThroughSlam). So the 'base' it sees is the camera.
        Then T_bg = EE-in-camera = T_ec directly; no inverse needed.
        Applying inv(T_cw) here would double-transform a camera-frame
        quantity and send held objects flying.
        """
        return self.ee_poses[idx]

    def hand_mask(self, idx: int, img_shape: Tuple[int, int]) -> np.ndarray:
        """Generate a mask of the robot's gripper + arm links for frame idx.

        The rosbag pose files (ee_pose.txt, l/r_gripper_pose.txt,
        joints_pose.json) store link poses ALREADY IN CAMERA FRAME
        (evidence: ee origin has z≈0.6–0.9m and x/y near 0, which only
        makes sense as camera-frame coords). So we hand them to
        generate_hand_mask without any T_wc transform. The double-
        transform `T_wc @ pose` that earlier iterations used produced
        empty masks (behind-camera projections), which is why the
        robot body was leaking into the top-down reconstruction.
        """
        try:
            from utils.hand_mask_utils import generate_hand_mask
        except ImportError:
            return np.zeros(img_shape[:2], dtype=np.uint8)

        T_ec = self.ee_poses[idx]
        T_lfc = self.l_finger[idx]
        T_rfc = self.r_finger[idx]

        T_joints_cam = None
        if self._joints_data and idx in self._joints_data:
            T_joints_cam = dict(self._joints_data[idx])

        return generate_hand_mask(
            T_ec, K, img_shape, T_lfc, T_rfc,
            T_joints=T_joints_cam,
        )


# ═════════════════════════════════════════════════════════════════════
#  Main driver
# ═════════════════════════════════════════════════════════════════════

def run_all_methods(data: TrajectoryData,
                    save_every: int = 10,
                    out_dir: Optional[str] = None,
                    color: bool = False,
                    methods: Optional[List[str]] = None) -> None:
    """Single-pass pipeline that runs all three visual pose-estimation
    methods in parallel and renders a 2×3 side-by-side comparison.

    Per frame the same mask/depth/detections are fed to each of:
        * centroid    (translation-only back-projection)
        * icp_chain   (ICP w/ prev-frame warmstart)
        * icp_anchor  (ICP w/ world-anchor warmstart)
    and the resulting detection list (with method-specific T_co, R_icp)
    is fed to BOTH an `ours` orchestrator (full RBPF + proprio) and a
    `baseline` orchestrator (baseline_mode=True). Net: 6 orchestrators
    stepped per frame, 6 ObjectTrackOverlays rendered per composite.
    """
    from pose_update.orchestrator import TwoTierOrchestrator, TriggerConfig
    from pose_update.state.slam_interface import PassThroughSlam

    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    tmap = TopDownMap()
    robot_path = RobotPathOverlay()

    methods = list(methods) if methods is not None else list(POSE_METHODS)
    icps: Dict[str, PoseEstimator] = {
        m: PoseEstimator(K, method=m) for m in methods}

    slam_poses = [data.cam_poses[i] for i in data.indices]
    orch_ours: Dict[str, "TwoTierOrchestrator"] = {}
    orch_base: Dict[str, "TwoTierOrchestrator"] = {}
    tracks_ours: Dict[str, ObjectTrackOverlay] = {}
    tracks_base: Dict[str, ObjectTrackOverlay] = {}
    for m in methods:
        orch_ours[m] = TwoTierOrchestrator(
            PassThroughSlam(slam_poses, default_cov=np.diag([1e-4]*6)),
            trigger=TriggerConfig(periodic_every_n_frames=30),
            n_particles=32, rng_seed=0, baseline_mode=False,
        )
        orch_base[m] = TwoTierOrchestrator(
            PassThroughSlam(slam_poses, default_cov=np.diag([1e-4]*6)),
            trigger=TriggerConfig(periodic_every_n_frames=30),
            n_particles=32, rng_seed=0, baseline_mode=True,
        )
        tracks_ours[m] = ObjectTrackOverlay()
        tracks_base[m] = ObjectTrackOverlay()

    compositor = AllMethodsCompositor(
        tmap, robot_path, tracks_ours, tracks_base, color=color)

    last_d: Optional[float] = None
    last_phase = "idle"
    held_id: Optional[int] = None
    saved = 0
    n_accept = {m: 0 for m in methods}
    n_reject = {m: 0 for m in methods}

    print(f"Running ALL-methods pipeline over {len(data.indices)} frames ...")
    print(f"  methods: {methods}")

    for local_i, idx in enumerate(data.indices):
        frame = data.load_frame(idx)
        if frame is None:
            continue
        rgb, depth, raw_dets = frame

        # Build the top-down map and robot-path overlay (shared).
        object_mask = np.zeros(depth.shape[:2], dtype=bool)
        for d in raw_dets:
            m = d.get("mask")
            if m is not None and m.shape == depth.shape[:2]:
                object_mask |= m.astype(bool)
        robot_mask = data.hand_mask(idx, depth.shape[:2]).astype(bool)
        tmap.add_frame(rgb, depth, data.cam_poses[idx],
                       object_mask=object_mask,
                       hand_mask=robot_mask, subsample=4)
        robot_path.add_pose(data.cam_poses[idx])

        # Build per-method detection lists.
        detections_by_method: Dict[str, List[Dict]] = {m: [] for m in methods}
        for d in raw_dets:
            if d["id"] is None:
                continue
            for m in methods:
                T_co, R_icp, fitness, rmse = icps[m].estimate(
                    oid=int(d["id"]), mask=d["mask"], depth=depth,
                    T_cw=data.cam_poses[idx])
                if T_co is None:
                    n_reject[m] += 1
                    continue
                n_accept[m] += 1
                detections_by_method[m].append({
                    "id": int(d["id"]), "label": d["label"],
                    "mask": d["mask"], "score": d["score"],
                    "T_co": T_co, "R_icp": R_icp,
                    "fitness": fitness, "rmse": rmse,
                })

        # Gripper state (depends only on trajectory, not method).
        raw_phase, last_d = data.gripper_phase(idx, last_d)
        if raw_phase == "grasping" and last_phase != "grasping":
            phase = "grasping"
            ee_cam = data.ee_poses[idx][:3, 3]
            # Use the chain-method detections (translation-only ok here
            # since proximity only needs mask+depth).
            held_id = resolve_held_by_proximity(
                detections_by_method["icp_chain"], depth, ee_cam)
        elif raw_phase == "releasing":
            phase = "releasing"
        elif last_phase in ("grasping", "holding"):
            phase = "holding"
        elif last_phase == "releasing" and raw_phase == "idle":
            phase = "idle"
            held_id = None
        else:
            phase = "idle"
        last_phase = phase
        gripper_state = {"phase": phase, "held_obj_id": held_id}

        T_ec = data.ee_poses[idx]
        T_bg = data.T_bg(idx)

        # Step all six orchestrators with their own detection lists.
        for m in methods:
            dets = detections_by_method[m]
            orch_ours[m].step(rgb, depth, dets, gripper_state,
                              T_ec=T_ec, T_bg=T_bg)
            orch_base[m].step(rgb, depth, dets, gripper_state,
                              T_ec=T_ec, T_bg=T_bg)
            tracks_ours[m].add_frame(idx, orch_ours[m].objects)
            tracks_base[m].add_frame(idx, orch_base[m].objects)

        compositor.add_phase(idx, phase)

        should_save = (local_i % save_every == 0
                       or idx == data.indices[-1]
                       or idx == data.indices[0])
        if should_save:
            path = os.path.join(out_dir, f"frame_{idx:04d}.png")
            compositor.render_frame(idx, rgb=rgb, save_path=path)
            saved += 1

        if (local_i + 1) % 50 == 0 or local_i == len(data.indices) - 1:
            reject_summary = "  ".join(
                f"{m}:{n_reject[m]}/{n_accept[m] + n_reject[m]}"
                for m in methods)
            print(f"  {local_i + 1}/{len(data.indices)} frames | "
                  f"rejects {reject_summary} | {saved} saved")

    print(f"  → {saved} all-methods frames saved to {out_dir}/")


def run_incremental(data: TrajectoryData,
                    save_every: int = 10,
                    out_dir: Optional[str] = None,
                    color: bool = False,
                    pose_method: str = "icp_chain") -> None:
    """Single-pass incremental pipeline.

    Everything — map, robot path, object tracks, composites — is built
    frame by frame. At frame t the visualization shows only what the
    robot has observed up to t. The scene is gradually constructed.
    """
    from pose_update.orchestrator import TwoTierOrchestrator, TriggerConfig
    from pose_update.state.slam_interface import PassThroughSlam

    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ── Modules ──────────────────────────────────────────────────────
    tmap = TopDownMap()
    robot_path = RobotPathOverlay()
    tracks_ours = ObjectTrackOverlay()
    tracks_base = ObjectTrackOverlay()
    compositor = SideBySideCompositor(
        tmap, robot_path, tracks_ours, tracks_base, color=color)

    # ── Orchestrators (both see the SAME vision inputs) ──────────────
    # Each needs its own SLAM backend because PassThroughSlam advances
    # an internal frame counter; they'll step in lockstep.
    slam_poses = [data.cam_poses[i] for i in data.indices]
    slam_ours = PassThroughSlam(slam_poses, default_cov=np.diag([1e-4] * 6))
    slam_base = PassThroughSlam(slam_poses, default_cov=np.diag([1e-4] * 6))
    orch_ours = TwoTierOrchestrator(
        slam_ours,
        trigger=TriggerConfig(periodic_every_n_frames=30),
        n_particles=32,
        rng_seed=0,
        baseline_mode=False,
    )
    orch_base = TwoTierOrchestrator(
        slam_base,
        trigger=TriggerConfig(periodic_every_n_frames=30),
        n_particles=32,
        rng_seed=0,
        baseline_mode=True,
    )

    # Shared visual tracker — both orchestrators see the same T_co.
    # Method selects the observation-extraction backend:
    #   "centroid"   — translation-only centroid back-projection
    #   "icp_chain"  — ICP with chain warmstart (default)
    #   "icp_anchor" — ICP with world-anchor warmstart (drift-free for static)
    print(f"Pose estimation method: {pose_method}")
    icp = PoseEstimator(K, method=pose_method)

    last_d: Optional[float] = None
    last_phase = "idle"
    held_id: Optional[int] = None
    saved = 0
    n_accept = 0
    n_reject = 0
    # Log fitness + rmse for every ICP call (accepted OR rejected)
    # so we can report the typical ranges at the end.
    fitness_log: List[float] = []
    rmse_log: List[float] = []
    rejected_fitness: List[float] = []
    rejected_rmse: List[float] = []

    print(f"Running incremental pipeline over {len(data.indices)} frames ...")
    for local_i, idx in enumerate(data.indices):
        frame = data.load_frame(idx)
        if frame is None:
            continue
        rgb, depth, raw_dets = frame

        # ── 1. Incrementally grow the top-down map ─────────────────
        # Density-normalized RGB accumulation (splat + per-frame dedup).
        # Object pixels take priority when both bg and obj hit the same
        # cell within a single frame.
        object_mask = np.zeros(depth.shape[:2], dtype=bool)
        for d in raw_dets:
            m = d.get("mask")
            if m is not None and m.shape == depth.shape[:2]:
                object_mask |= m.astype(bool)
        robot_mask = data.hand_mask(idx, depth.shape[:2]).astype(bool)
        tmap.add_frame(rgb, depth, data.cam_poses[idx],
                       object_mask=object_mask,
                       hand_mask=robot_mask, subsample=4)

        # ── 2. Extend robot path ──────────────────────────────────
        robot_path.add_pose(data.cam_poses[idx])

        # ── 3. Build detections → orchestrator step ────────────────
        # Full 6-DoF observation per object via ICP against a first-
        # frame reference cloud. Provides T_co, a data-driven R_icp,
        # fitness, and rmse — not the identity-rotation centroid we
        # used before.
        detections = []
        for d in raw_dets:
            if d["id"] is None:
                continue
            T_co, R_icp, fitness, rmse = icp.estimate(
                oid=int(d["id"]), mask=d["mask"], depth=depth,
                T_cw=data.cam_poses[idx])
            fitness_log.append(fitness)
            rmse_log.append(rmse)
            if T_co is None:
                # ICP failed (low fitness / high rmse / degenerate
                # transform). Drop this detection so the filter
                # predicts forward instead of receiving a bad update.
                n_reject += 1
                rejected_fitness.append(fitness)
                rejected_rmse.append(rmse)
                continue
            n_accept += 1
            detections.append({
                "id": int(d["id"]), "label": d["label"],
                "mask": d["mask"], "score": d["score"],
                "T_co": T_co,
                "R_icp": R_icp,
                "fitness": fitness,
                "rmse": rmse,
            })

        # Gripper state machine.
        raw_phase, last_d = data.gripper_phase(idx, last_d)
        if raw_phase == "grasping" and last_phase != "grasping":
            phase = "grasping"
            # EE in camera frame — compare directly with back-projected
            # depth points (no world transform needed).
            T_ec = data.ee_poses[idx]  # EE-to-camera
            ee_cam = T_ec[:3, 3]
            held_id = resolve_held_by_proximity(
                detections, depth, ee_cam)
        elif raw_phase == "releasing":
            phase = "releasing"
        elif last_phase in ("grasping", "holding"):
            phase = "holding"
        elif last_phase == "releasing" and raw_phase == "idle":
            phase = "idle"
            held_id = None
        else:
            phase = "idle"
        last_phase = phase
        gripper_state = {"phase": phase, "held_obj_id": held_id}

        T_ec = data.ee_poses[idx]
        T_bg = data.T_bg(idx)
        orch_ours.step(rgb, depth, detections, gripper_state,
                       T_ec=T_ec, T_bg=T_bg)
        # Baseline: same detections, same SLAM, but baseline_mode=True
        # causes the orchestrator to ignore gripper_state/T_ec/T_bg and
        # use uniform Q. The call is identical in signature — the
        # proprio inputs are stripped internally.
        orch_base.step(rgb, depth, detections, gripper_state,
                       T_ec=T_ec, T_bg=T_bg)

        # ── 4. Record tracks + phase ──────────────────────────────
        tracks_ours.add_frame(idx, orch_ours.objects)
        tracks_base.add_frame(idx, orch_base.objects)
        compositor.add_phase(idx, phase)

        # ── 5. Render composite (if on the save cadence) ──────────
        should_save = (local_i % save_every == 0
                       or idx == data.indices[-1]
                       or idx == data.indices[0])
        if should_save:
            path = os.path.join(out_dir, f"frame_{idx:04d}.png")
            compositor.render_frame(idx, rgb=rgb, save_path=path)
            saved += 1

        if (local_i + 1) % 50 == 0 or local_i == len(data.indices) - 1:
            total = n_accept + n_reject
            rej_pct = 100.0 * n_reject / max(total, 1)
            print(f"  {local_i + 1}/{len(data.indices)} frames | "
                  f"ours: {len(orch_ours.objects)} objs | "
                  f"base: {len(orch_base.objects)} objs | "
                  f"ICP rej {n_reject}/{total} ({rej_pct:.1f}%) | "
                  f"{saved} saved")

    print(f"  → {saved} side-by-side frames saved to {out_dir}/")

    # ── ICP diagnostics ────────────────────────────────────────────
    if fitness_log:
        from pose_update.orchestrator import TwoTierOrchestrator  # noqa
        fit = np.asarray(fitness_log)
        rms = np.asarray(rmse_log)
        # First-observation calls have fitness=1.0, rmse=0.0 (no ICP run).
        # Strip those so percentiles reflect actual ICP outcomes.
        real = (rms > 0) | (fit < 1.0)
        fit_r = fit[real]; rms_r = rms[real]
        print()
        print(f"ICP threshold settings: "
              f"MIN_FITNESS={ICPPoseEstimator.MIN_FITNESS:.2f}, "
              f"MAX_RMSE={ICPPoseEstimator.MAX_RMSE*1000:.1f}mm")
        print(f"Accepted observations: {n_accept}   "
              f"Rejected: {n_reject}   "
              f"(first-frame / no-ICP: {(~real).sum()})")
        if len(fit_r):
            pct = np.percentile(fit_r, [1, 5, 25, 50, 75, 95, 99])
            print(f"Fitness  (real ICP runs, n={len(fit_r)}):  "
                  f"min={fit_r.min():.3f}  "
                  f"p1={pct[0]:.3f}  p5={pct[1]:.3f}  "
                  f"p25={pct[2]:.3f}  median={pct[3]:.3f}  "
                  f"p75={pct[4]:.3f}  p95={pct[5]:.3f}  "
                  f"max={fit_r.max():.3f}")
            pct = np.percentile(rms_r * 1000, [1, 5, 25, 50, 75, 95, 99])
            print(f"RMSE(mm) (real ICP runs, n={len(rms_r)}):  "
                  f"min={rms_r.min()*1000:.2f}  "
                  f"p1={pct[0]:.2f}  p5={pct[1]:.2f}  "
                  f"p25={pct[2]:.2f}  median={pct[3]:.2f}  "
                  f"p75={pct[4]:.2f}  p95={pct[5]:.2f}  "
                  f"max={rms_r.max()*1000:.2f}")
        if rejected_fitness:
            rf = np.asarray(rejected_fitness); rr = np.asarray(rejected_rmse)
            print(f"Rejected (n={len(rf)}):  "
                  f"fitness median={np.median(rf):.3f}  "
                  f"(range {rf.min():.3f}-{rf.max():.3f})  "
                  f"rmse(mm) median={np.median(rr)*1000:.1f}  "
                  f"(range {rr.min()*1000:.1f}-{rr.max()*1000:.1f})")


# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=str, default="apple_bowl_2",
                        help="Sub-dir under multi_objects/ to visualize")
    parser.add_argument("--frames", type=int, default=10_000,
                        help="Max frames to process (capped by trajectory length)")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save a composite every N processed frames")
    parser.add_argument("--out-dir", type=str, default=None,
                        help=("Output directory. Default: "
                              "tests/visualization_pipeline/<trajectory>/"))
    parser.add_argument("--color", action="store_true",
                        help=("Render background with averaged camera "
                              "RGB (smoothed). Default: grey/red "
                              "categorical map."))
    parser.add_argument("--method", type=str, default="icp_chain",
                        choices=list(POSE_METHODS),
                        help=("Per-frame visual pose estimator: "
                              "'centroid' (translation-only back-proj), "
                              "'icp_chain' (ICP with prev-frame init, "
                              "default), or 'icp_anchor' (ICP with "
                              "world-anchor init — drift-free for "
                              "static objects, rejects on manipulation)."))
    parser.add_argument("--all-methods", action="store_true",
                        help=("Run all configured methods in parallel and "
                              "render a 2xN side-by-side grid "
                              "(ours + baseline for each method). "
                              "Overrides --method."))
    parser.add_argument("--methods-set", type=str, default="four_baselines",
                        choices=("default", "four_baselines", "all"),
                        help=("Which methods to include in --all-methods. "
                              "'default' = chain + anchor (3 cols incl. centroid). "
                              "'four_baselines' = chain + chain_strict + "
                              "anchor + anchor_strict (4 cols, what the user asked). "
                              "'all' = every method including centroid."))
    args = parser.parse_args()

    data = TrajectoryData(n_frames=args.frames, step=args.step,
                          trajectory=args.trajectory)
    if args.all_methods:
        method_sets = {
            "default":       ["centroid", "icp_chain", "icp_anchor"],
            "four_baselines": ["icp_chain", "icp_chain_strict",
                                "icp_anchor", "icp_anchor_strict"],
            "all":           list(POSE_METHODS),
        }
        methods = method_sets[args.methods_set]
        out_dir = (args.out_dir or
                   os.path.join(OUT_DIR, args.trajectory,
                                f"all_methods_{args.methods_set}"))
        run_all_methods(data, save_every=args.save_every, out_dir=out_dir,
                        color=args.color, methods=methods)
    else:
        out_dir = (args.out_dir or
                   os.path.join(OUT_DIR, args.trajectory, args.method))
        run_incremental(data, save_every=args.save_every, out_dir=out_dir,
                        color=args.color, pose_method=args.method)


if __name__ == "__main__":
    main()

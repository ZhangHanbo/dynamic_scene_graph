"""Render five per-API mp4s for the apple_drop trajectory.

Drives a single :class:`EkfTracker` instance through all 416 frames
using only the public API surface (`detect / step / get_scene /
get_points / smooth`) and produces 5 separate videos under
``tests/visualization_pipeline/apple_drop/api_videos/``:

    detect.mp4       — RGB + bbox/mask overlay + label table
    step.mp4         — RGB + world-frame top-down (SceneView ellipses)
    get_scene.mp4    — top-down + relations + objects table
    get_points.mp4   — accumulated point-cloud xy projection
    smooth.mp4       — pre-smooth ↔ post-smooth side-by-side

`detect.mp4` requires the OWL+SAM2 servers to be reachable; the other
four mp4s read cached `detection_h` JSONs (step()'s normal input) and
don't need the network. When the SAM2 server is offline, `detect.mp4`
is skipped and a clear log line explains why.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ekf_tracker.api import EkfTracker  # noqa: E402
from scripts.rosbag2dataset.server_configs import (  # noqa: E402
    OWL_SERVER_URL, SAM2_SERVER_URL, OWL_DETECT_PATH, SAM2_STREAM_INIT_PATH,
)


# ─────────────────────────────────────────────────────────────────────
#  Paths & camera
# ─────────────────────────────────────────────────────────────────────

DATA = ROOT / "datasets" / "apple_drop"
VIZ = ROOT / "tests" / "visualization_pipeline" / "apple_drop"
DET_DIR = (VIZ / "perception" / "detection_h"
           if (VIZ / "perception" / "detection_h").is_dir()
           else DATA / "detection_h")
RELATION_CACHE = VIZ / "relation_cache"
OUT_DIR = VIZ / "api_videos"

K_DEFAULT = np.array([[554.3827, 0.0, 320.5],
                      [0.0, 554.3827, 240.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

PALETTE = [
    (0.00, 0.78, 0.31), (0.86, 0.24, 0.16), (0.16, 0.55, 0.86),
    (0.96, 0.78, 0.08), (0.63, 0.31, 0.78), (0.94, 0.51, 0.12),
    (0.08, 0.71, 0.63), (0.90, 0.47, 0.43), (0.50, 0.71, 0.85),
    (0.40, 0.40, 0.40),
]


def _palette(i: int) -> Tuple[float, float, float]:
    return PALETTE[int(i) % len(PALETTE)]


# ─────────────────────────────────────────────────────────────────────
#  Loaders (mirror scripts/visualize_ekf_tracking.py:_load_*)
# ─────────────────────────────────────────────────────────────────────

def _load_amcl(path: Path) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for line in open(path):
        a = line.strip().split()
        if len(a) != 8:
            continue
        _, tx, ty, tz, qx, qy, qz, qw = map(float, a)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        out.append(T)
    return out


def _load_idx_pose(path: Path) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if not path.exists():
        return out
    for line in open(path):
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


def _load_widths(path: Path) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not path.exists():
        return out
    for k, v in json.load(open(path)).items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        l = v.get("l_gripper_finger_joint")
        r = v.get("r_gripper_finger_joint")
        if l is not None and r is not None:
            out[idx] = float(l) + float(r)
    return out


def _load_joints(path: Path) -> Dict[int, Dict[str, float]]:
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.load(open(path)).items()}


def _load_dets(path: Path) -> List[Dict[str, Any]]:
    """Decode cached detection_h JSON, mirroring scripts._load_detection_json."""
    if not path.exists():
        return []
    data = json.load(open(path))
    out: List[Dict[str, Any]] = []
    for det in data.get("detections", []):
        mb = det.get("mask", "")
        if not mb:
            continue
        try:
            mb_bytes = base64.b64decode(mb)
            mask = np.array(Image.open(BytesIO(mb_bytes)).convert("L"))
            mask = (mask > 128).astype(np.uint8)
        except Exception:
            continue
        out.append({
            "id": int(det.get("object_id")),
            "object_id": int(det.get("object_id")),
            "label": det.get("label", "unknown"),
            "labels": det.get("labels", {}),
            "mask": mask,
            "score": float(det.get("score", 0.0)),
            "mean_score": float(det.get("mean_score", 0.0)),
            "n_obs": int(det.get("n_obs", 0)),
            "box": det.get("box"),
        })
    return out


def _load_rgb(idx: int) -> np.ndarray:
    return np.array(Image.open(DATA / "rgb" / f"rgb_{idx:06d}.png")
                     .convert("RGB"))


def _load_depth(idx: int) -> np.ndarray:
    return np.load(DATA / "depth" / f"depth_{idx:06d}.npy").astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
#  Server probe (graceful skip for detect.mp4)
# ─────────────────────────────────────────────────────────────────────

def _server_reachable(url: str, path: str, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(url.rstrip("/") + path, timeout=timeout)
        return r.status_code in (200, 405, 422)
    except (requests.exceptions.RequestException, Exception):
        return False


# ─────────────────────────────────────────────────────────────────────
#  Per-frame renderers (one per API)
# ─────────────────────────────────────────────────────────────────────

def _draw_topdown(ax, scene, T_wb: np.ndarray, title: str,
                   held_oid: Optional[int] = None,
                   lim: Optional[Tuple[Tuple[float, float],
                                        Tuple[float, float]]] = None) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.scatter(T_wb[0, 3], T_wb[1, 3], c="k", s=80, marker="^",
               label="base")
    for oid, obj in scene.objects.items():
        col = _palette(oid)
        x = float(obj.pose[0, 3]); y = float(obj.pose[1, 3])
        c2 = obj.cov[:2, :2]
        try:
            w, V = np.linalg.eigh(c2)
            w = np.maximum(w, 1e-6)
            ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
            ax.add_patch(mpatches.Ellipse(
                (x, y), 4 * np.sqrt(w[0]), 4 * np.sqrt(w[1]),
                angle=ang, fill=False, lw=1.2, color=col, alpha=0.8))
        except np.linalg.LinAlgError:
            pass
        marker = "*" if held_oid is not None and oid == held_oid else "o"
        ax.scatter(x, y, color=col, s=80, marker=marker, zorder=3,
                   edgecolors="k", linewidth=0.5)
        ax.text(x + 0.02, y + 0.02,
                f"{oid}:{obj.label}\nr={obj.r:.2f}",
                fontsize=7, color="k",
                bbox=dict(boxstyle="round,pad=0.15", fc="w",
                          ec=col, alpha=0.7))
    if lim is not None:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])


def _overlay_dets(rgb: np.ndarray, dets: List[Dict[str, Any]],
                   alpha: float = 0.45) -> np.ndarray:
    canvas = rgb.copy().astype(np.float32)
    for i, d in enumerate(dets):
        col = (np.array(_palette(int(d.get("id", i)))) * 255).astype(np.float32)
        m = d["mask"].astype(bool)
        if m.shape != canvas.shape[:2]:
            continue
        canvas[m] = canvas[m] * (1 - alpha) + col[None, :] * alpha
    return canvas.clip(0, 255).astype(np.uint8)


def render_detect_frame(rgb: np.ndarray, dets: List[Dict[str, Any]],
                         idx: int, source: str) -> plt.Figure:
    fig = plt.figure(figsize=(12.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
    ax_im = fig.add_subplot(gs[0, 0])
    ax_im.imshow(_overlay_dets(rgb, dets))
    ax_im.set_title(f"detect(rgb, vocabulary, history) — frame {idx}\n"
                    f"source: {source}",
                    fontsize=10)
    ax_im.axis("off")
    for d in dets:
        oid = int(d.get("id", 0))
        col = _palette(oid)
        if d.get("box") is None:
            continue
        x0, y0, x1, y1 = (float(b) for b in d["box"])
        ax_im.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, ec=col, lw=2))
        ax_im.text(x0, max(0, y0 - 4),
                   f"id={oid} {d['label']} {d.get('score', 0):.2f}",
                   color="white", fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="none"))

    ax_tx = fig.add_subplot(gs[0, 1])
    ax_tx.axis("off")
    rows = [f"# instances: {len(dets)}", ""]
    for d in dets[:8]:
        rows.append(
            f"  id={int(d.get('id', -1)):>2} "
            f"label={d['label']:<8} "
            f"score={d.get('score', 0):.3f}\n"
            f"        n_obs={int(d.get('n_obs', 0))} "
            f"mean_score={d.get('mean_score', 0):.3f}"
        )
    ax_tx.text(0.0, 1.0, "\n".join(rows), va="top", ha="left",
               fontsize=9, family="monospace")
    fig.tight_layout()
    return fig


def render_step_frame(rgb: np.ndarray, scene, T_wb: np.ndarray,
                       idx: int, held_oid: Optional[int],
                       lim) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].imshow(rgb)
    axes[0].set_title(f"step() — frame {idx}", fontsize=10)
    axes[0].axis("off")
    _draw_topdown(axes[1], scene, T_wb,
                  f"#objs={len(scene.objects)} held={held_oid}",
                  held_oid=held_oid, lim=lim)
    fig.tight_layout()
    return fig


def render_get_scene_frame(scene, T_wb: np.ndarray, idx: int,
                            lim) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8),
                              gridspec_kw={"width_ratios": [1.4, 1]})
    _draw_topdown(axes[0], scene, T_wb,
                  f"get_scene() — frame {idx}  "
                  f"({len(scene.objects)} objs)",
                  held_oid=None, lim=lim)
    axes[1].axis("off")
    rows = [f"objects ({len(scene.objects)}):"]
    for oid, obj in sorted(scene.objects.items()):
        rows.append(
            f"  oid={oid:>2} {obj.label:<8} r={obj.r:.2f}  "
            f"({obj.pose[0, 3]:+.2f},{obj.pose[1, 3]:+.2f},"
            f"{obj.pose[2, 3]:+.2f})"
        )
    rows.append("")
    rows.append(f"relations ({len(scene.relations)}):")
    for r in scene.relations[:6]:
        rows.append(f"  {r['parent']} --{r['type']}({r['score']:.2f})--> "
                     f"{r['child']}")
    axes[1].text(0.0, 1.0, "\n".join(rows), va="top", ha="left",
                 fontsize=8, family="monospace")
    fig.tight_layout()
    return fig


def render_get_points_frame(api: EkfTracker, scene, idx: int,
                              T_wb: np.ndarray, lim) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8),
                              gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"get_points(oid) — frame {idx}  "
                 f"(top-down xy projection)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.scatter(T_wb[0, 3], T_wb[1, 3], c="k", s=80, marker="^")

    rows = ["per-oid n_points:"]
    for oid in sorted(scene.objects.keys()):
        pts = api.get_points(oid)
        col = _palette(oid)
        if pts.shape[0] > 0:
            stride = max(1, pts.shape[0] // 600)
            sub = pts[::stride]
            ax.scatter(sub[:, 0], sub[:, 1], c=[col], s=2, alpha=0.5)
        ax.scatter([scene.objects[oid].pose[0, 3]],
                   [scene.objects[oid].pose[1, 3]],
                   c=[col], s=60, edgecolors="k", linewidths=0.5,
                   marker="o", zorder=4)
        rows.append(
            f"  oid={oid:>2} {scene.objects[oid].label:<8} "
            f"n={pts.shape[0]:>5}")
    if lim is not None:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
    axes[1].axis("off")
    axes[1].text(0.0, 1.0, "\n".join(rows), va="top", ha="left",
                 fontsize=9, family="monospace")
    fig.tight_layout()
    return fig


def render_smooth_frame(pre, post, T_wb: np.ndarray,
                         idx: int, lim, did_smooth: bool) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    _draw_topdown(axes[0], pre, T_wb,
                  f"pre-smooth — frame {idx}",
                  held_oid=None, lim=lim)
    _draw_topdown(axes[1], post, T_wb,
                  f"post-smooth — frame {idx}"
                  + (" (smooth() called)" if did_smooth else ""),
                  held_oid=None, lim=lim)
    # Δ-arrows on the right panel.
    if did_smooth:
        for oid, obj in post.objects.items():
            if oid not in pre.objects:
                continue
            x0 = float(pre.objects[oid].pose[0, 3])
            y0 = float(pre.objects[oid].pose[1, 3])
            x1 = float(obj.pose[0, 3]); y1 = float(obj.pose[1, 3])
            if abs(x1 - x0) + abs(y1 - y0) < 1e-6:
                continue
            axes[1].annotate("", xy=(x1, y1), xytext=(x0, y0),
                              arrowprops=dict(arrowstyle="->",
                                              color="red", lw=1.0,
                                              alpha=0.8))
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
#  Video composer
# ─────────────────────────────────────────────────────────────────────

def _compose_mp4(png_dir: Path, out_path: Path, fps: float) -> None:
    pngs = sorted(png_dir.glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(f"no PNGs under {png_dir}")
    list_path = png_dir / "_filelist.txt"
    with open(list_path, "w") as f:
        for p in pngs:
            f.write(f"file '{p.absolute()}'\n")
            f.write(f"duration {1.0 / fps:.6f}\n")
        f.write(f"file '{pngs[-1].absolute()}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ─────────────────────────────────────────────────────────────────────
#  Bounded-range computation across the trajectory (for steady axes)
# ─────────────────────────────────────────────────────────────────────

def _world_lim_from_states(samples: List[Tuple[np.ndarray,
                                                "ekf_tracker.api.SceneView"]]
                           ) -> Tuple[Tuple[float, float],
                                       Tuple[float, float]]:
    xs: List[float] = []; ys: List[float] = []
    for T_wb, sv in samples:
        xs.append(float(T_wb[0, 3])); ys.append(float(T_wb[1, 3]))
        for o in sv.objects.values():
            xs.append(float(o.pose[0, 3])); ys.append(float(o.pose[1, 3]))
    if not xs:
        return ((-1.0, 1.0), (-1.0, 1.0))
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    rng = max(0.5, max(max(xs) - min(xs), max(ys) - min(ys)) * 0.6 + 0.4)
    return ((cx - rng, cx + rng), (cy - rng, cy + rng))


# ─────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frame", type=int, default=416)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--smooth-every", type=int, default=30,
                    help="Call smooth() every N frames (smooth.mp4 only)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--vocabulary", nargs="*",
                    default=["apple", "cup", "bottle", "cabinet", "bowl"])
    ap.add_argument("--no-detect", action="store_true",
                    help="Skip detect.mp4 even if servers are up.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Server reachability probe (only matters for detect.mp4).
    if args.no_detect:
        do_detect = False
        skip_reason = "--no-detect"
    else:
        servers_up = (
            _server_reachable(OWL_SERVER_URL, OWL_DETECT_PATH)
            and _server_reachable(SAM2_SERVER_URL, SAM2_STREAM_INIT_PATH))
        do_detect = servers_up
        skip_reason = ("OWL or SAM2 server unreachable" if not servers_up
                       else "")

    # Loaders.
    slam = _load_amcl(DATA / "pose_txt" / "amcl_pose.txt")
    T_bc_map = _load_idx_pose(DATA / "pose_txt" / "T_bc.txt")
    T_bg_map = _load_idx_pose(DATA / "pose_txt" / "ee_pose.txt")
    widths = _load_widths(DATA / "pose_txt" / "joints_pose.json")
    joints = _load_joints(DATA / "pose_txt" / "joints_pose.json")
    n_frames = min(args.max_frame, len(slam))
    print(f"[render] frames {args.start}..{n_frames}, "
          f"out -> {OUT_DIR}")
    if not do_detect:
        print(f"[render] detect.mp4 SKIPPED ({skip_reason})")

    # EkfTracker + per-API temp dirs.
    api = EkfTracker(K=K_DEFAULT, T_bc=np.eye(4),
                     relation_backend="llm",
                     relation_cache_dir=str(RELATION_CACHE))
    tmp_root = Path(tempfile.mkdtemp(prefix="api_videos_"))
    print(f"[render] frame buffers -> {tmp_root}")
    tmp_dirs: Dict[str, Path] = {}
    for name in ("detect", "step", "get_scene", "get_points", "smooth"):
        d = tmp_root / name
        d.mkdir()
        tmp_dirs[name] = d

    # First pass: drive the tracker and collect (T_wb, sv) for axis limits.
    samples_for_lim: List[Tuple[np.ndarray, Any]] = []

    detect_history = None
    detect_failures = 0

    try:
        for k, idx in enumerate(range(args.start, n_frames)):
            rgb = _load_rgb(idx); depth = _load_depth(idx)

            # detect — live SAM2 streaming if reachable.
            detect_dets: List[Dict[str, Any]] = []
            detect_source = "skipped"
            if do_detect:
                try:
                    detect_dets, detect_history = api.detect(
                        rgb, vocabulary=args.vocabulary,
                        history=detect_history)
                    detect_source = "live OWL+SAM2"
                except Exception as e:
                    detect_failures += 1
                    detect_source = f"failed: {type(e).__name__}"
                    if detect_failures > 3:
                        # Bail out of detect path; fall through to cached.
                        do_detect = False
                        skip_reason = "live pipeline kept failing"
                        print(f"[render] disabling detect.mp4: {e}")

            # step — uses cached detection_h JSONs (unaffected by server).
            step_dets = _load_dets(
                DET_DIR / f"detection_{idx:06d}_final.json")
            scene = api.step(
                detections=step_dets, rgb=rgb, depth=depth,
                slam_pose=slam[idx],
                T_bc=T_bc_map.get(idx),
                T_bg=T_bg_map.get(idx),
                gripper_width=widths.get(idx),
                joints=joints.get(idx))
            samples_for_lim.append((slam[idx], scene))

            held_oids = (api._last_dbg or {}).get("held_oids_used") or []
            held_oid = held_oids[0] if held_oids else None

            # detect.mp4 frame
            if do_detect:
                fig = render_detect_frame(rgb, detect_dets, idx, detect_source)
                fig.savefig(tmp_dirs["detect"] / f"frame_{k:06d}.png",
                            dpi=110, bbox_inches="tight")
                plt.close(fig)

            # get_scene snapshot BEFORE smooth — cache; render later.
            pre_scene = scene  # already returned by step()

            # smooth.mp4 frame: every smooth-every frames trigger smooth.
            did_smooth = (args.smooth_every > 0
                          and k > 0 and (k % args.smooth_every == 0))
            if did_smooth:
                post_scene = api.smooth()
            else:
                post_scene = pre_scene

            # We need a stable axis limit; use last 60 frames so far
            # (running window) — refresh every frame to track drift.
            window = samples_for_lim[-min(60, len(samples_for_lim)):]
            lim = _world_lim_from_states(window)

            # step.mp4 frame
            fig = render_step_frame(rgb, scene, slam[idx], idx,
                                     held_oid, lim)
            fig.savefig(tmp_dirs["step"] / f"frame_{k:06d}.png",
                        dpi=110, bbox_inches="tight")
            plt.close(fig)

            # get_scene.mp4 frame (post-step, pre-smooth)
            fig = render_get_scene_frame(api.get_scene(), slam[idx], idx, lim)
            fig.savefig(tmp_dirs["get_scene"] / f"frame_{k:06d}.png",
                        dpi=110, bbox_inches="tight")
            plt.close(fig)

            # get_points.mp4 frame
            fig = render_get_points_frame(api, api.get_scene(), idx,
                                            slam[idx], lim)
            fig.savefig(tmp_dirs["get_points"] / f"frame_{k:06d}.png",
                        dpi=110, bbox_inches="tight")
            plt.close(fig)

            # smooth.mp4 frame
            fig = render_smooth_frame(pre_scene, post_scene, slam[idx],
                                       idx, lim, did_smooth)
            fig.savefig(tmp_dirs["smooth"] / f"frame_{k:06d}.png",
                        dpi=110, bbox_inches="tight")
            plt.close(fig)

            if (k + 1) % 20 == 0:
                print(f"[render] frame {idx}: rendered {k+1}/{n_frames - args.start}, "
                      f"objs={len(scene.objects)}, held={held_oid}")
    finally:
        if detect_history is not None:
            try:
                detect_history.close()
            except Exception:
                pass

    # Compose mp4s.
    print()
    for name, d in tmp_dirs.items():
        if name == "detect" and not do_detect:
            print(f"[render] detect.mp4 SKIPPED ({skip_reason})")
            continue
        out_path = OUT_DIR / f"{name}.mp4"
        try:
            _compose_mp4(d, out_path, args.fps)
            size_mb = out_path.stat().st_size / 1e6
            print(f"[render] {name}.mp4 -> {out_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[render] {name}.mp4 FAILED: {e}")

    # Cleanup.
    shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Per-API visualizations of the EkfTracker public surface.

For each public method (detect / step / get_scene / get_points / smooth),
exercise the API on the apple_drop trajectory and dump a PNG that
illustrates what the call returns.

Outputs go to tests/visualization_pipeline/apple_drop/api_demo/.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import textwrap
import traceback
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
RELATION_CACHE = os.path.join(VIZ, "relation_cache")
OUT_DIR = os.path.join(VIZ, "api_demo")
os.makedirs(OUT_DIR, exist_ok=True)

K = np.array([[554.3827, 0.0, 320.5],
              [0.0, 554.3827, 240.5],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# Palette for oids / instance index.
PALETTE = [
    (0.00, 0.78, 0.31), (0.86, 0.24, 0.16), (0.16, 0.55, 0.86),
    (0.96, 0.78, 0.08), (0.63, 0.31, 0.78), (0.94, 0.51, 0.12),
    (0.08, 0.71, 0.63), (0.90, 0.47, 0.43),
]


def _palette(i):
    return PALETTE[int(i) % len(PALETTE)]


# ─────────── data loaders (matches main() in visualize_ekf_tracking) ───────────

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


def _load_rgb(idx):
    p = os.path.join(DATA, "rgb", f"rgb_{idx:06d}.png")
    return np.array(Image.open(p).convert("RGB"))


def _load_depth(idx):
    p = os.path.join(DATA, "depth", f"depth_{idx:06d}.npy")
    return np.load(p).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# Helper renderers
# ────────────────────────────────────────────────────────────────────────

def _overlay_dets(rgb, dets, alpha=0.45):
    """RGB image with mask + box + label per instance."""
    canvas = rgb.copy().astype(np.float32)
    for i, d in enumerate(dets):
        col = (np.array(_palette(i)) * 255).astype(np.float32)
        m = d["mask"].astype(bool)
        canvas[m] = canvas[m] * (1 - alpha) + col[None, :] * alpha
    canvas = canvas.clip(0, 255).astype(np.uint8)
    return canvas


def _draw_topdown(ax, scene, T_wb, title, held_oid=None, lim=None):
    """Top-down (x, y) plot of a SceneView: per-track ellipse + label."""
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    # Robot base position.
    ax.scatter(T_wb[0, 3], T_wb[1, 3], c="k", s=80, marker="^",
               label="base")
    # Tracks.
    for oid, obj in scene.objects.items():
        col = _palette(oid)
        x, y = float(obj.pose[0, 3]), float(obj.pose[1, 3])
        # 2-σ translation ellipse from cov[:2,:2]
        c2 = obj.cov[:2, :2]
        try:
            w, V = np.linalg.eigh(c2)
            w = np.maximum(w, 1e-6)
            ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
            ell = mpatches.Ellipse(
                (x, y), 2 * 2 * np.sqrt(w[0]), 2 * 2 * np.sqrt(w[1]),
                angle=ang, fill=False, lw=1.2, color=col, alpha=0.8)
            ax.add_patch(ell)
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
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])


# ────────────────────────────────────────────────────────────────────────
# 1) detect()
# ────────────────────────────────────────────────────────────────────────

def demo_detect(api: EkfTracker, demo_frame=281):
    """Show what `detect()` returns. If the perception server is
    unreachable, fall back to the cached detection_h JSON to demo the
    schema (clearly labelled).
    """
    rgb = _load_rgb(demo_frame)
    note = ""
    detections = None
    try:
        detections, _hist = api.detect(
            rgb, vocabulary=["apple", "bottle", "cup", "cabinet"])
        source = "live RPC call to det_server"
    except Exception as e:
        # Fall back to cached perception output.
        cached = _load_dets(os.path.join(
            DET_DIR, f"detection_{demo_frame:06d}_final.json"))
        # Convert to detect()'s schema.
        detections = [{
            "label": d["label"],
            "score": d["score"],
            "box": np.asarray(d["box"], dtype=np.float32),
            "mask": d["mask"],
        } for d in cached]
        source = ("server unreachable — schema demo from cached "
                  "detection_h JSON")
        note = f"(fallback: {type(e).__name__})"

    # Plot.
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
    ax_im = fig.add_subplot(gs[0, 0])
    overlay = _overlay_dets(rgb, detections)
    ax_im.imshow(overlay)
    ax_im.set_title(f"detect(rgb, vocabulary, history) — frame {demo_frame}\n"
                    f"source: {source} {note}", fontsize=10)
    ax_im.axis("off")
    for i, d in enumerate(detections):
        x0, y0, x1, y1 = (float(b) for b in d["box"])
        col = _palette(i)
        ax_im.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, ec=col, lw=2))
        ax_im.text(x0, max(0, y0 - 4),
                   f"{i}:{d['label']} {d['score']:.2f}",
                   color="white", fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="none"))

    ax_tx = fig.add_subplot(gs[0, 1])
    ax_tx.axis("off")
    rows = ["Returned schema:",
            "  List[dict]: keys = label, score, box, mask",
            "  + history (forwarded SAM2-style state)",
            "",
            f"# of instances: {len(detections)}",
            ""]
    for i, d in enumerate(detections):
        m = d["mask"]
        nz = int(m.sum()) if hasattr(m, "sum") else 0
        rows.append(
            f"  [{i}] label='{d['label']}' score={d['score']:.3f}\n"
            f"        box={[round(float(b),1) for b in d['box']]}, "
            f"mask shape={tuple(m.shape)}, fg_px={nz}"
        )
    ax_tx.text(0.0, 1.0, "\n".join(rows), va="top", ha="left",
               fontsize=9, family="monospace")

    out = os.path.join(OUT_DIR, "01_detect.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[detect]   saved -> {out}  (n={len(detections)}, source: {source})")
    return out


# ────────────────────────────────────────────────────────────────────────
# 2) step()  — drive trajectory and capture per-frame snapshots
# ────────────────────────────────────────────────────────────────────────

def demo_step(api: EkfTracker, snap_frames=(60, 110, 200, 290)):
    slam = _load_amcl(os.path.join(DATA, "pose_txt", "amcl_pose.txt"))
    T_bc = _load_idx_pose(os.path.join(DATA, "pose_txt", "T_bc.txt"))
    T_bg = _load_idx_pose(os.path.join(DATA, "pose_txt", "ee_pose.txt"))
    widths = _load_widths(os.path.join(DATA, "pose_txt", "joints_pose.json"))
    joints = _load_joints(os.path.join(DATA, "pose_txt", "joints_pose.json"))

    last_idx = max(snap_frames) + 1
    snaps = {}
    held_at_snap = {}
    rgb_at_snap = {}

    for idx in range(last_idx):
        rgb = _load_rgb(idx)
        depth = _load_depth(idx)
        dets = _load_dets(os.path.join(
            DET_DIR, f"detection_{idx:06d}_final.json"))
        sv = api.step(
            detections=dets,
            rgb=rgb,
            depth=depth,
            slam_pose=slam[idx],
            T_bc=T_bc.get(idx),
            T_bg=T_bg.get(idx),
            gripper_width=widths.get(idx),
            joints=joints.get(idx),
        )
        if idx in snap_frames:
            snaps[idx] = sv
            held = (api._last_dbg or {}).get("held_oids_used") or []
            held_at_snap[idx] = held[0] if held else None
            rgb_at_snap[idx] = rgb

    # Common world-frame extent for the panel grid.
    xs, ys = [], []
    for sv in snaps.values():
        for o in sv.objects.values():
            xs.append(float(o.pose[0, 3]))
            ys.append(float(o.pose[1, 3]))
    if xs:
        cx, cy = np.mean(xs), np.mean(ys)
        rng = max(0.3, max(np.ptp(xs), np.ptp(ys)) * 0.7 + 0.4)
        lim = ((cx - rng, cx + rng), (cy - rng, cy + rng))
    else:
        lim = ((-1, 1), (-1, 1))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("step(detections, rgb, depth, slam_pose, T_bc, T_bg) — "
                 "per-frame SceneView snapshots", fontsize=12)
    for col, idx in enumerate(snap_frames):
        ax_im = axes[0, col]
        ax_td = axes[1, col]
        ax_im.imshow(rgb_at_snap[idx])
        ax_im.set_title(f"frame {idx}", fontsize=10)
        ax_im.axis("off")
        sv = snaps[idx]
        _draw_topdown(ax_td, sv, slam[idx],
                      f"#objs={len(sv.objects)} held={held_at_snap[idx]}",
                      held_oid=held_at_snap[idx], lim=lim)

    out = os.path.join(OUT_DIR, "02_step.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[step]     saved -> {out}  (snapped frames {list(snap_frames)})")
    return out


# ────────────────────────────────────────────────────────────────────────
# 3) get_scene()
# ────────────────────────────────────────────────────────────────────────

def demo_get_scene(api: EkfTracker):
    sv = api.get_scene()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [1.4, 1]})
    # Top-down.
    T_wb = api._tracker.state.T_wb
    if T_wb is None:
        T_wb = np.eye(4)
    _draw_topdown(axes[0], sv, T_wb,
                  f"get_scene() — final state ({len(sv.objects)} objs)",
                  held_oid=None, lim=None)

    # Object list + relation table.
    axes[1].axis("off")
    rows = ["SceneView.objects:"]
    for oid, obj in sorted(sv.objects.items()):
        cov_diag = np.diag(obj.cov)
        rows.append(
            f"  oid={oid:>2}  label={obj.label:<8}  r={obj.r:.3f}\n"
            f"        T_world[:3,3]="
            f"({obj.pose[0,3]:+.3f}, {obj.pose[1,3]:+.3f}, "
            f"{obj.pose[2,3]:+.3f})\n"
            f"        cov diag (xyz)="
            f"({cov_diag[0]:.2e}, {cov_diag[1]:.2e}, {cov_diag[2]:.2e})"
        )
    rows.append("")
    rows.append(f"SceneView.relations  (n={len(sv.relations)}):")
    if not sv.relations:
        rows.append("  (none — relation backend was 'none' or no edges "
                    "passed the EMA threshold)")
    for r in sv.relations:
        rows.append(f"  {r['parent']} --{r['type']}({r['score']:.2f})--> "
                     f"{r['child']}")
    axes[1].text(0.0, 1.0, "\n".join(rows), va="top", ha="left",
                 fontsize=8, family="monospace")

    out = os.path.join(OUT_DIR, "03_get_scene.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[get_scene] saved -> {out}  (n_objects={len(sv.objects)}, "
          f"n_relations={len(sv.relations)})")
    return out


# ────────────────────────────────────────────────────────────────────────
# 4) get_points()
# ────────────────────────────────────────────────────────────────────────

def demo_get_points(api: EkfTracker, max_n=4):
    sv = api.get_scene()
    # Pick up to `max_n` objects with the largest reference clouds.
    sized = []
    for oid in sv.objects:
        pts = api.get_points(oid)
        sized.append((pts.shape[0], oid, pts))
    sized.sort(reverse=True)
    sized = sized[:max_n]
    n = len(sized)
    if n == 0:
        return None

    fig = plt.figure(figsize=(4.5 * n, 5))
    fig.suptitle(
        "get_points(object_id) — accumulated ICP reference cloud "
        "(world frame)", fontsize=12)
    for i, (cnt, oid, pts) in enumerate(sized):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        col = _palette(oid)
        if cnt > 0:
            sub = pts[::max(1, cnt // 800)]   # cap to ~800 pts for legibility
            ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
                       c=[col], s=4, alpha=0.7, depthshade=False)
            obj = sv.objects[oid]
            ax.scatter([obj.pose[0, 3]], [obj.pose[1, 3]],
                       [obj.pose[2, 3]],
                       c="k", s=60, marker="^", label="μ")
            # Equal aspect by extending limits.
            mn = sub.min(axis=0); mx = sub.max(axis=0)
            ext = max((mx - mn).max(), 0.05)
            ctr = 0.5 * (mn + mx)
            for axis, c in zip(("x", "y", "z"), ctr):
                getattr(ax, f"set_{axis}lim")(c - ext / 2, c + ext / 2)
        ax.set_title(f"oid={oid}  label={sv.objects[oid].label}\n"
                     f"n_points={cnt}", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        try:
            ax.set_zlabel("z [m]")
        except Exception:
            pass

    out = os.path.join(OUT_DIR, "04_get_points.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[get_points] saved -> {out}  "
          f"(top oids by point count: {[(o,c) for c,o,_ in sized]})")
    return out


# ────────────────────────────────────────────────────────────────────────
# 5) smooth()
# ────────────────────────────────────────────────────────────────────────

def demo_smooth(api: EkfTracker):
    """Snapshot pre-smooth poses, run smooth(), report Δ per oid."""
    pre = {oid: o.pose.copy() for oid, o in api.get_scene().objects.items()}
    sv = api.smooth()
    diffs = []
    for oid, o in sv.objects.items():
        if oid in pre:
            d = float(np.max(np.abs(pre[oid] - o.pose)))
            diffs.append((oid, o.label, d))
    diffs.sort(key=lambda t: -t[2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [1.2, 1]})
    # Top-down: post-smooth scene with Δ-arrows from pre-smooth pose.
    T_wb = api._tracker.state.T_wb
    if T_wb is None:
        T_wb = np.eye(4)
    _draw_topdown(axes[0], sv, T_wb,
                  f"smooth() — post-smooth scene ({len(sv.objects)} objs)",
                  held_oid=None, lim=None)
    for oid in pre:
        if oid not in sv.objects:
            continue
        x0, y0 = float(pre[oid][0, 3]), float(pre[oid][1, 3])
        x1, y1 = float(sv.objects[oid].pose[0, 3]), float(sv.objects[oid].pose[1, 3])
        axes[0].annotate("", xy=(x1, y1), xytext=(x0, y0),
                          arrowprops=dict(arrowstyle="->",
                                          color="red", lw=1.0, alpha=0.7))

    # Side text panel.
    axes[1].axis("off")
    rows = ["smooth() — slow-tier pose-graph optimisation",
            "─" * 56,
            "delegates to TwoTierOrchestratorGaussian.smooth(),",
            "which calls ekf_tracker.factor_graph.PoseGraphOptimizer.run",
            "over the current world-frame priors + cached relation graph",
            "and writes posteriors back via state.inject_posterior_world.",
            "",
            f"call:    api.smooth()",
            f"outcome: returned SceneView with {len(sv.objects)} objects",
            "",
            "Δpose per oid (max |Δ T_world|, m):",
            ]
    if not diffs:
        rows.append("  (no objects pre-smooth — smooth() is a no-op here)")
    else:
        for oid, label, d in diffs[:8]:
            rows.append(f"  oid={oid:>2}  label={label:<8}  |Δ|={d:.3e}")
    axes[1].text(0.0, 1.0, "\n".join(rows), va="top", ha="left",
                  fontsize=9, family="monospace")

    out = os.path.join(OUT_DIR, "05_smooth.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    n_moved = sum(1 for _, _, d in diffs if d > 1e-9)
    print(f"[smooth]   saved -> {out}  ({len(sv.objects)} objs, "
          f"{n_moved} moved by smoothing)")
    return out


# ────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────

def main():
    api = EkfTracker(
        K=K, T_bc=None,
        relation_backend="llm",
        relation_cache_dir=RELATION_CACHE,
    )

    # detect() doesn't depend on prior state -> render first.
    # Frame 29 has 5 detections (cabinet, apple, bottle, cup) in this trajectory.
    demo_detect(api, demo_frame=29)

    # step() drives the tracker forward; afterwards api state is non-empty.
    demo_step(api, snap_frames=(60, 110, 200, 290))

    # The remaining APIs read api state.
    demo_get_scene(api)
    demo_get_points(api, max_n=4)
    demo_smooth(api)


if __name__ == "__main__":
    main()

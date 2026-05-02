"""Per-frame run of the throttled relation detector, frames [0, 700].

Mirrors the production path:
  * iterate every frame;
  * on each frame, consult the *real*
    ``TwoTierOrchestrator._should_recompute_relations``;
  * fire the LLM client only when the trigger says so (first-call /
    grasp / release / new-oid / every-N-frames);
  * between fires, keep the last cached edges — the EMA never decays
    during quiet stretches.

Outputs:
  * stdout trace: per-frame fired/skipped + reason, + LLM edges on fire;
  * ``tests/out/llm_throttled_timeline.txt`` — one-line-per-frame summary;
  * ``tests/out/llm_throttled_final_frame<idx>.png`` — tracked-EMA overlay
    on the last successful frame ≤ 700.
"""
from __future__ import annotations

import json
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Stub gtsam so we can import the orchestrator's trigger method. ──
sys.modules.setdefault("gtsam", types.ModuleType("gtsam"))
_g = sys.modules["gtsam"]
_g.noiseModel = types.SimpleNamespace(
    Base=type("Base", (), {}),
    Gaussian=types.SimpleNamespace(Covariance=lambda *a, **k: None),
    Diagonal=types.SimpleNamespace(Sigmas=lambda *a, **k: None),
)
for name in ("Pose3", "NonlinearFactorGraph", "Values",
             "LevenbergMarquardtParams", "LevenbergMarquardtOptimizer",
             "Marginals", "BetweenFactorPose3"):
    setattr(_g, name, lambda *a, **k: None)

from pose_update.orchestrator import (  # noqa: E402
    TwoTierOrchestrator, BernoulliConfig,
)
from pose_update.relations.relation_client import (  # noqa: E402
    LLMRelationClient, decode_mask_b64,
)
from visualize_llm_relation_frame import render_relation_overlay  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────

FRAME_START = 0
FRAME_END   = 700           # inclusive
ALPHA   = 0.3
THR_EMA = 0.5
THR_RAW = 0.5

ROOT = Path("/Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP")
RGB_DIR = Path("/Volumes/External/Workspace/datasets/apple_in_the_tray/rgb")
DET_DIR = (ROOT / "tests/visualization_pipeline/apple_in_the_tray/"
           "perception/detection_h")
OUT_DIR = ROOT / "tests/out"


# Production BernoulliConfig (only the fields _should_recompute_relations
# touches matter).
CFG = BernoulliConfig(
    relation_backend="llm",
    relation_every_n_frames=90,   # 3 s at 30 Hz
    relation_on_grasp=True,
    relation_on_release=True,
    relation_on_new_object=True,
    r_conf=0.5,
)


# Lightweight EMA (same semantics as orchestrator.RelationFilter).
@dataclass
class Edge:
    parent: int
    child: int
    relation_type: str
    score: float


class EMAFilter:
    def __init__(self, alpha=0.3, threshold=0.5):
        self.a, self.thr = alpha, threshold
        self.ema: dict = {}

    def update(self, raw):
        detected = {(e.parent, e.child, e.relation_type): e.score for e in raw}
        keys = set(self.ema) | set(detected)
        out = []
        for k in keys:
            raw_s = detected.get(k, 0.0)
            prev = self.ema.get(k, raw_s)
            v = self.a * raw_s + (1 - self.a) * prev
            self.ema[k] = v
            if v >= self.thr:
                p, c, rt = k
                out.append(Edge(p, c, rt, 1.0))
        self.ema = {k: v for k, v in self.ema.items() if v > 0.01}
        return out


# Stub shape consumed by the real trigger method.
class OrchStub:
    def __init__(self):
        self.bernoulli = CFG
        self.relation_client = object()
        self._last_relation_frame = -10**9
        self.last_state = {"phase": "idle"}
        self.existence: dict = {}
        self._known_oids_before_step: set = set()
        self.frame_count = 0
        self.state = types.SimpleNamespace(collapsed_objects=lambda: {})


should = TwoTierOrchestrator._should_recompute_relations


# ──────────────────────────────────────────────────────────────────────
#  Per-frame loader
# ──────────────────────────────────────────────────────────────────────

def _load_dets(idx: int):
    det_path = DET_DIR / f"detection_{idx:06d}_final.json"
    if not det_path.exists():
        return None
    with det_path.open() as f:
        return json.load(f).get("detections", [])


def _load_rgb_and_masks(idx: int, dets):
    rgb = Image.open(RGB_DIR / f"rgb_{idx:06d}.png").convert("RGB")
    W, H = rgb.size
    bboxes_pix, bboxes_n, labels, oids, masks = [], [], [], [], []
    for d in dets:
        box = d.get("box")
        m64 = d.get("mask")
        if box is None or not m64:
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        bboxes_pix.append([x0, y0, x1, y1])
        bboxes_n.append([x0 / W, y0 / H, x1 / W, y1 / H])
        labels.append(d.get("label", "?"))
        oids.append(int(d.get("object_id", -1)))
        masks.append(decode_mask_b64(m64, size=(W, H)))
    return (rgb,
            np.asarray(bboxes_pix, dtype=np.float32),
            np.asarray(bboxes_n, dtype=np.float32),
            labels, oids, masks)


def _why(orch: OrchStub, gripper) -> str:
    """Short string describing which branch would fire this frame."""
    if orch._last_relation_frame < 0:
        return "first"
    last_phase = orch.last_state.get("phase", "idle")
    cur_phase = gripper.get("phase", "idle")
    if CFG.relation_on_grasp and last_phase != "grasping" and cur_phase == "grasping":
        return "grasp"
    if CFG.relation_on_release and last_phase == "releasing" and cur_phase != "releasing":
        return "release"
    if CFG.relation_on_new_object:
        cur = {oid for oid, r in orch.existence.items() if r >= CFG.r_conf}
        new = cur - orch._known_oids_before_step
        if new:
            return f"new-oid({sorted(new)})"
    if CFG.relation_every_n_frames > 0:
        if orch.frame_count - orch._last_relation_frame >= CFG.relation_every_n_frames:
            return "periodic"
    return ""


# ──────────────────────────────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = LLMRelationClient(model_name="gpt-5.1")
    ema = EMAFilter(alpha=ALPHA, threshold=THR_EMA)
    orch = OrchStub()

    timeline_lines = []
    last_rendered = None  # last state we can overlay at the end

    # Apple-in-tray dataset has no gripper/manipulation info → always idle.
    gripper = {"phase": "idle"}

    print(f"[seq] frames [{FRAME_START}, {FRAME_END}]  "
          f"trigger = every {CFG.relation_every_n_frames} + key events")

    n_fired = 0
    t_all = time.time()
    for idx in range(FRAME_START, FRAME_END + 1):
        dets = _load_dets(idx)
        if dets is None:
            continue
        oids = [int(d.get("object_id", -1)) for d in dets if d.get("object_id") is not None]

        # Maintain the stub's "existence" view from the perception JSON
        # (perception already confirmed these — treat r=1 for all).
        orch.existence = {oid: 1.0 for oid in oids}
        orch.frame_count = idx
        reason = _why(orch, gripper)
        fire = should(orch, gripper)

        if fire:
            rgb, bboxes_pix, bboxes_n, labels, oid_list, masks = _load_rgb_and_masks(idx, dets)
            if len(bboxes_n) < 2:
                # Nothing to relate → still mark as "fired but empty".
                orch._last_relation_frame = idx
                ema.update([])
                timeline_lines.append(
                    f"{idx:04d}  FIRED ({reason:<18s})  n={len(bboxes_n)}  no-call"
                )
                n_fired += 1
            else:
                usable = masks if all(m is not None for m in masks) else None
                t0 = time.time()
                p = client.detect(rgb, bboxes_n, masks=usable)
                dt = time.time() - t0
                orch._last_relation_frame = idx
                raw = []
                if p is not None:
                    for i in range(len(oid_list)):
                        for j in range(len(oid_list)):
                            if i == j:
                                continue
                            s = float(p[i, j])
                            if s >= THR_RAW:
                                raw.append(Edge(oid_list[i], oid_list[j], "on", s))
                ema.update(raw)
                n_fired += 1
                edge_str = "none"
                if raw:
                    edge_str = ", ".join(f"{e.parent}->{e.child}:{e.score:.2f}"
                                         for e in sorted(raw, key=lambda e: -e.score))
                timeline_lines.append(
                    f"{idx:04d}  FIRED ({reason:<18s})  n={len(bboxes_n)}  "
                    f"{dt:.1f}s  edges=[{edge_str}]"
                )
                last_rendered = dict(rgb=rgb, idx=idx, bboxes_pix=bboxes_pix,
                                     labels=labels, oid_list=oid_list,
                                     p_parent=p if p is not None else np.zeros((len(oid_list),) * 2))
        else:
            timeline_lines.append(f"{idx:04d}  skip")

        orch._known_oids_before_step = set(orch.existence.keys())
        orch.last_state = dict(gripper)

    total_dt = time.time() - t_all
    print(f"[done] fired {n_fired} / {FRAME_END - FRAME_START + 1} frames  "
          f"elapsed {total_dt:.1f}s")

    # ── 1) Timeline trace (print key lines + save full trace)
    print()
    for line in timeline_lines:
        if "FIRED" in line or "fired" in line.lower():
            print("  " + line)
    (OUT_DIR / "llm_throttled_timeline.txt").write_text("\n".join(timeline_lines) + "\n")
    print(f"\n[save] full timeline  → {OUT_DIR / 'llm_throttled_timeline.txt'}")

    # ── 2) Final EMA table
    print(f"\nFinal EMA state ({len(ema.ema)} keys, threshold = {THR_EMA}):")
    for key, v in sorted(ema.ema.items(), key=lambda kv: -kv[1]):
        p, c, rt = key
        mark = "*" if v >= THR_EMA else " "
        print(f"  {mark} {p} —({rt})→ {c}  ema={v:.3f}")

    # ── 3) Overlay on last fired frame
    if last_rendered is None:
        print("[warn] no frames with >=2 detections were ever fired; nothing to draw")
        return 1

    idx_last = last_rendered["idx"]
    ema_out = OUT_DIR / f"llm_throttled_final_frame{idx_last:06d}.png"
    oid_list = last_rendered["oid_list"]
    n = len(oid_list)
    ema_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ema_mat[i, j] = ema.ema.get((oid_list[i], oid_list[j], "on"), 0.0)
    img = render_relation_overlay(
        last_rendered["rgb"], last_rendered["bboxes_pix"],
        last_rendered["labels"], ema_mat, thr=THR_EMA,
        caption_prefix=(f"frame {idx_last} — THROTTLED EMA "
                        f"(fired {n_fired} times in [{FRAME_START},{FRAME_END}])"),
    )
    img.save(ema_out)
    print(f"[save] final EMA overlay → {ema_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

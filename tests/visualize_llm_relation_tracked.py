"""Step 2 — tracked support graph across frames.

Runs the LLM relation detector on a strided sequence of
apple_in_the_tray frames, feeds every frame's raw edges
(keyed on the perception ``object_id``) into the same
``RelationFilter`` EMA instance the orchestrator uses,
and reports:

  * per-frame dump: raw edges this frame, filtered edges after EMA update;
  * final EMA table over all edges seen in the window;
  * overlay PNGs on the last frame showing:
      - this-frame raw LLM edges, and
      - the EMA-stable tracked edges (final filter output).

Run:
    python tests/visualize_llm_relation_tracked.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_update.relations.relation_client import LLMRelationClient, decode_mask_b64
from visualize_llm_relation_frame import render_relation_overlay


# Minimal copies of RelationEdge / RelationFilter (pose_update.factor_graph
# pulls in gtsam which isn't needed for this visualization). Kept bit-for-bit
# equivalent to the orchestrator's RelationFilter (alpha=0.3, threshold=0.5).

@dataclass
class RelationEdge:
    parent: int
    child: int
    relation_type: str
    score: float


class RelationFilter:
    def __init__(self, alpha: float = 0.3, threshold: float = 0.5):
        self.alpha = alpha
        self.threshold = threshold
        self._ema: dict = {}

    def update(self, raw_edges):
        detected = {}
        for e in raw_edges:
            detected[(e.parent, e.child, e.relation_type)] = e.score
        all_keys = set(self._ema.keys()) | set(detected.keys())
        out = []
        for key in all_keys:
            raw = detected.get(key, 0.0)
            prev = self._ema.get(key, raw)
            ema = self.alpha * raw + (1.0 - self.alpha) * prev
            self._ema[key] = ema
            if ema >= self.threshold:
                p, c, rt = key
                out.append(RelationEdge(parent=p, child=c,
                                        relation_type=rt, score=1.0))
        self._ema = {k: v for k, v in self._ema.items() if v > 0.01}
        return out


# --------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------- #

FRAMES = list(range(300, 601, 30))   # 300, 330, ..., 600 → 11 frames
THR_RAW = 0.5                        # matches orchestrator pre-filter (BernoulliConfig.relation_score_threshold)
THR_EMA = 0.5                        # matches RelationFilter(threshold=0.5)
ALPHA = 0.3                          # matches RelationFilter(alpha=0.3)

ROOT = Path("/Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP")
RGB_DIR = Path("/Volumes/External/Workspace/datasets/apple_in_the_tray/rgb")
DET_DIR = (ROOT / "tests/visualization_pipeline/apple_in_the_tray/"
           "perception/detection_h")
OUT_DIR = ROOT / "tests/out"


# --------------------------------------------------------------------- #
#  Per-frame loader
# --------------------------------------------------------------------- #

def _load_frame(idx: int):
    rgb_path = RGB_DIR / f"rgb_{idx:06d}.png"
    det_path = DET_DIR / f"detection_{idx:06d}_final.json"
    rgb = Image.open(rgb_path).convert("RGB")
    W, H = rgb.size
    with det_path.open() as f:
        dets = json.load(f)["detections"]
    bboxes_pix, bboxes_n, labels, oids, masks = [], [], [], [], []
    for d in dets:
        box = d.get("box")
        if box is None:
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        bboxes_pix.append([x0, y0, x1, y1])
        bboxes_n.append([x0 / W, y0 / H, x1 / W, y1 / H])
        labels.append(d.get("label", "?"))
        oids.append(int(d.get("object_id", -1)))
        m64 = d.get("mask")
        masks.append(decode_mask_b64(m64, size=(W, H)) if m64 else None)
    return (rgb,
            np.asarray(bboxes_pix, dtype=np.float32),
            np.asarray(bboxes_n, dtype=np.float32),
            labels, oids, masks)


def _raw_edges_from_p(p_parent: np.ndarray,
                      oids: List[int],
                      thr: float) -> List[RelationEdge]:
    """Replicates orchestrator._try_learned_relations edge construction."""
    edges: List[RelationEdge] = []
    n = len(oids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = float(p_parent[i, j])
            if s >= thr:
                edges.append(RelationEdge(
                    parent=oids[i], child=oids[j],
                    relation_type="on", score=s,
                ))
    return edges


# --------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------- #

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = LLMRelationClient(model_name="gpt-5.1")
    rel_filter = RelationFilter(alpha=ALPHA, threshold=THR_EMA)

    # Bookkeeping for the end-of-run table.
    ema_history: Dict[Tuple[int, int, str], List[float]] = {}
    oid_label: Dict[int, str] = {}

    last = {"idx": None, "rgb": None, "bboxes_pix": None,
            "labels": None, "oids": None, "p_parent": None}

    print(f"[seq]   frames = {FRAMES}")
    print(f"[seq]   THR_RAW={THR_RAW}  THR_EMA={THR_EMA}  ALPHA={ALPHA}\n")

    for t, idx in enumerate(FRAMES):
        rgb, bboxes_pix, bboxes_n, labels, oids, masks = _load_frame(idx)
        for o, lab in zip(oids, labels):
            oid_label[o] = lab

        n = len(oids)
        print(f"───────────── frame {idx}  ({t+1}/{len(FRAMES)}) "
              f"─  {n} dets ─────────────")
        for k, (o, lab, b) in enumerate(zip(oids, labels, bboxes_pix)):
            print(f"  [{k}] oid={o:2d}  {lab:8s}  "
                  f"box=({int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])})")

        if n < 2:
            print("  (skip: <2 dets)\n")
            rel_filter.update([])
            continue

        t0 = time.time()
        usable_masks = masks if all(m is not None for m in masks) else None
        p_parent = client.detect(rgb, bboxes_n, masks=usable_masks)
        dt = time.time() - t0
        if p_parent is None:
            print("  (LLM call failed)\n")
            rel_filter.update([])
            continue

        raw_edges = _raw_edges_from_p(p_parent, oids, THR_RAW)
        print(f"  raw edges ≥ {THR_RAW}  ({dt:.1f}s):")
        if not raw_edges:
            print("    (none)")
        for e in sorted(raw_edges, key=lambda x: -x.score):
            print(f"    {oid_label[e.parent]}[{e.parent}] "
                  f"—({e.score:.2f})→ {oid_label[e.child]}[{e.child}]")

        filtered = rel_filter.update(raw_edges)
        print(f"  EMA state after update  ({len(rel_filter._ema)} keys, "
              f"{len(filtered)} passing ≥ {THR_EMA}):")
        for key, ema in sorted(rel_filter._ema.items(), key=lambda kv: -kv[1]):
            p, c, rt = key
            ema_history.setdefault(key, []).append(ema)
            marker = "*" if ema >= THR_EMA else " "
            print(f"   {marker} {oid_label.get(p, '?')}[{p}] "
                  f"—({rt})→ {oid_label.get(c, '?')}[{c}]  ema={ema:.3f}")
        print()

        last = {"idx": idx, "rgb": rgb,
                "bboxes_pix": bboxes_pix, "labels": labels,
                "oids": oids, "p_parent": p_parent}

    # ───────── summary table
    print("─" * 70)
    print("EMA trajectory  (columns = frame index):")
    header = "edge".ljust(28) + "  " + "  ".join(
        f"{FRAMES[i]:>6d}" for i in range(len(FRAMES)))
    print(header)
    for key, hist in sorted(ema_history.items(),
                            key=lambda kv: -max(kv[1])):
        p, c, rt = key
        name = (f"{oid_label.get(p, '?')}[{p}]->"
                f"{oid_label.get(c, '?')}[{c}]")
        # history shorter than frame list if edge was first seen later;
        # pad left with zeros (EMA was 0 before the edge existed).
        padded = [0.0] * (len(FRAMES) - len(hist)) + hist
        vals = "  ".join(f"{v:6.2f}" for v in padded)
        print(f"{name[:28]:28s}  {vals}")

    # ───────── overlay on last frame: this-frame raw, final EMA-tracked
    if last["idx"] is None:
        print("\n[skip]  no frames succeeded; nothing to render")
        return 1

    idx = last["idx"]
    raw_png = OUT_DIR / f"llm_relation_tracked_raw_frame{idx:06d}.png"
    ema_png = OUT_DIR / f"llm_relation_tracked_ema_frame{idx:06d}.png"

    # Raw overlay (this-frame p_parent matrix directly)
    raw_overlay = render_relation_overlay(
        last["rgb"], last["bboxes_pix"], last["labels"],
        last["p_parent"], thr=THR_RAW,
        caption_prefix=f"frame {idx} — RAW LLM p_parent (this frame)",
    )
    raw_overlay.save(raw_png)

    # EMA overlay: build an NxN "score" matrix where entry (i,j) is the
    # EMA value for edge (oids[i] -> oids[j], 'on'), or 0 if not in EMA
    # state. Only edges whose both endpoints exist in this frame show.
    oids = last["oids"]
    n = len(oids)
    ema_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            key = (oids[i], oids[j], "on")
            ema_mat[i, j] = rel_filter._ema.get(key, 0.0)
    ema_overlay = render_relation_overlay(
        last["rgb"], last["bboxes_pix"], last["labels"],
        ema_mat, thr=THR_EMA,
        caption_prefix=(f"frame {idx} — TRACKED EMA "
                        f"(α={ALPHA}, thr={THR_EMA}, over {len(FRAMES)} frames)"),
    )
    ema_overlay.save(ema_png)

    print(f"\n[save]  raw overlay   → {raw_png}")
    print(f"[save]  EMA overlay   → {ema_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
